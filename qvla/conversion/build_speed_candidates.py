"""Build isolated lower-resolution and narrower-FFN speed probes, no quality guarantee."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def digest(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_vision(a):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import numpy as np
    from types import SimpleNamespace
    if a.onnx_site_packages:
        sys.path.append(str(a.onnx_site_packages))
    from qvla.evaluation.haq_offline_eval import load_policy
    from qvla.conversion.export_smolvla_vision_onnx import VisionAndConnector
    root = Path.cwd()
    torch.set_num_threads(4)
    args = SimpleNamespace(model_dir=a.model_dir.resolve(),
                           vlm_assets_dir=root/'artifacts/transfer/smolvlm2_assets', device='cpu')
    policy, _, _ = load_policy(args)
    policy.model.vlm_with_expert.vlm.set_attn_implementation('eager')

    class Probe(VisionAndConnector):
        def forward(self, pixel):
            patch = self.vision.patch_size
            x = self.vision.embeddings.patch_embedding(pixel)
            h, w = pixel.shape[2] // patch, pixel.shape[3] // patch
            side = self.vision.embeddings.num_patches_per_side
            # Scale positions over the full pretrained positional grid.
            rows = torch.arange(h, device=pixel.device) * side // h
            cols = torch.arange(w, device=pixel.device) * side // w
            positions = (rows[:, None] * side + cols[None, :]).reshape(1, -1)
            x = x.flatten(2).transpose(1, 2) + self.vision.embeddings.position_embedding(positions)
            x = self.vision.encoder(inputs_embeds=x, attention_mask=None).last_hidden_state
            features = self.connector(self.vision.post_layernorm(x))
            # Preserve existing 64-token board ABI. This repeats coarse spatial
            # features; it is a speed proxy, not a validated lower-res policy.
            grid = h // 4
            x = features.reshape(1, grid, grid, 960).permute(0, 3, 1, 2)
            x = torch.nn.functional.interpolate(x, size=(8, 8), mode='nearest')
            return x.permute(0, 2, 3, 1).reshape(1, 64, 960)

    wrapper = Probe(policy).eval().requires_grad_(False).float()
    for size in a.sizes:
        folder = a.output / f'vision_{size}'
        folder.mkdir(parents=True, exist_ok=True)
        pixel = np.load(a.pixel_source)
        pixel = torch.nn.functional.interpolate(torch.from_numpy(pixel), size=(size, size), mode='bilinear', align_corners=False)
        with torch.inference_mode():
            y = wrapper(pixel)
            torch.onnx.export(wrapper, pixel, str(folder/'model.onnx'), dynamo=False,
                              input_names=['pixel'], output_names=['features'], opset_version=17)
        np.save(folder/'pixel.npy', pixel.numpy())
        (folder/'dataset.txt').write_text(str((folder/'pixel.npy').resolve())+'\n')
        (folder/'export.json').write_text(json.dumps({'source_checkpoint_sha256':digest(args.model_dir/'model.safetensors'),
            'input_shape':list(pixel.shape), 'output_shape':list(y.shape), 'native_feature_grid':size//64,
            'feature_resampling':'nearest to 8x8 for legacy ABI', 'quality_validated':False}, indent=2))
        print('EXPORTED vision', size, flush=True)


def prune_expert(a):
    import onnx
    import numpy as np
    source = a.source_onnx
    for width in a.widths:
        m = onnx.load(str(source))
        tensors = {x.name:x for x in m.graph.initializer}
        changed = []
        for node in m.graph.node:
            if node.op_type != 'MatMul' or '/mlp/' not in node.name:
                continue
            if not any(k in node.name for k in ['gate_proj', 'up_proj', 'down_proj']):
                continue
            name = node.input[1]
            value = onnx.numpy_helper.to_array(tensors[name])
            is_down = 'down_proj' in node.name
            expected = (2048, 720) if is_down else (720, 2048)
            if value.shape != expected:
                raise ValueError((node.name, value.shape))
            value = value[:width, :] if is_down else value[:, :width]
            tensors[name].CopyFrom(onnx.numpy_helper.from_array(np.ascontiguousarray(value), name))
            changed.append(node.name)
        if len(changed) != 48:
            raise ValueError('Expected 3 FFN matrices in each of 16 layers')
        del m.graph.value_info[:]
        onnx.checker.check_model(m)
        folder = a.output / f'expert_ffn{width}'
        folder.mkdir(parents=True, exist_ok=True)
        onnx.save(m, str(folder/'model.onnx'))
        names=[x.name for x in m.graph.input]
        with np.load(a.boundary) as z:
            for name in names:np.save(folder/(name+'.npy'), z[name])
        (folder/'dataset.txt').write_text(' '.join(str((folder/(n+'.npy')).resolve()) for n in names)+'\n')
        (folder/'export.json').write_text(json.dumps({'source_onnx_sha256':digest(source),
            'width_before':2048, 'width_after':width, 'changed_matrices':changed,
            'layers':16, 'euler_steps_unchanged':True, 'selection':'leading channels, speed-only untrained structural pruning',
            'quality_validated':False}, indent=2))
        print('EXPORTED expert', width, flush=True)


def prune_depth(a):
    import onnx
    from onnx.utils import Extractor
    import numpy as np
    source = a.source_onnx
    for depth in a.depths:
        m = onnx.load(str(source))
        down = next(n for n in m.graph.node if n.name == f'/mlp/down_proj_{depth-1}/MatMul')
        residual = [n for n in m.graph.node if n.op_type == 'Add' and down.output[0] in n.input]
        if len(residual) != 1:
            raise ValueError('Ambiguous residual')
        final_norm = next(n for n in m.graph.node if n.name == '/norm/Cast')
        final_norm.input[0] = residual[0].output[0]
        names = [x.name for x in m.graph.input][:3+2*depth]
        m = Extractor(m).extract_model(names, ['velocity'])
        remaining = [n for n in m.graph.node if n.op_type == 'MatMul' and '/mlp/down_proj' in n.name]
        if len(remaining) != depth:
            raise ValueError('Unexpected retained depth')
        onnx.checker.check_model(m)
        folder = a.output / f'expert_depth{depth}'
        folder.mkdir(parents=True, exist_ok=True)
        onnx.save(m, str(folder/'model.onnx'))
        with np.load(a.boundary) as z:
            for name in names:np.save(folder/(name+'.npy'), z[name])
        (folder/'dataset.txt').write_text(' '.join(str((folder/(n+'.npy')).resolve()) for n in names)+'\n')
        (folder/'export.json').write_text(json.dumps({'source_onnx_sha256':digest(source),
            'layers_before':16,'layers_after':depth,'ffn_width':2048,'input_names':names,
            'selection':'retain leading expert layers, reuse final norm/action projection; no recovery training',
            'quality_validated':False},indent=2))
        print('EXPORTED depth', depth, flush=True)


def compile_graph(a):
    from rknn.api import RKNN
    folder = a.folder.resolve()
    report={'target':'rk3588', 'precision':'w8a8 default eligible kernels', 'source_sha256':digest(folder/'model.onnx'),
            'quality_validated':False, 'status':'building'}
    start=time.monotonic()
    m=RKNN(verbose=True, verbose_file=str(folder/'build.log'))
    def check(name, rc):
        if rc: raise RuntimeError(f'{name}: {rc}')
    previous=Path.cwd()
    try:
        os.chdir(folder)
        check('config',m.config(target_platform='rk3588',quantized_dtype='w8a8',quantized_method='channel',float_dtype='float16',optimization_level=3))
        check('load',m.load_onnx(model=str(folder/'model.onnx')))
        check('build',m.build(do_quantization=True,dataset=str(folder/'dataset.txt')))
        check('export',m.export_rknn(str(folder/'model.rknn')))
        report.update(status='compiled',sha256=digest(folder/'model.rknn'),bytes=(folder/'model.rknn').stat().st_size)
    except BaseException as e:
        report.update(status='failed',error=repr(e));raise
    finally:
        os.chdir(previous);m.release()
        report['elapsed_seconds']=time.monotonic()-start
        (folder/'compile.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    v=sub.add_parser('vision');v.add_argument('--sizes',type=int,nargs='+',default=[256,128]);v.add_argument('--onnx-site-packages',type=Path)
    v.add_argument('--output',type=Path,required=True)
    v.add_argument('--model-dir',type=Path,default=Path('artifacts/transfer/model'))
    v.add_argument('--pixel-source',type=Path,default=Path('runs/smolvla_vision_split_v1/vision_input_0.npy'))
    e=sub.add_parser('expert');e.add_argument('--widths',type=int,nargs='+',default=[1024,512]);e.add_argument('--output',type=Path,required=True)
    d=sub.add_parser('depth');d.add_argument('--depths',type=int,nargs='+',default=[8,4]);d.add_argument('--output',type=Path,required=True)
    for parser in (e,d):
        parser.add_argument('--source-onnx',type=Path,default=Path('runs/smolvla_expert_step_split_v2/expert_step.onnx'))
        parser.add_argument('--boundary',type=Path,default=Path('runs/smolvla_expert_step_split_v2/expert_boundary.npz'))
    c=sub.add_parser('compile');c.add_argument('--folder',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='vision':export_vision(a)
    elif a.mode=='expert':prune_expert(a)
    elif a.mode=='depth':prune_depth(a)
    else:compile_graph(a)

if __name__=='__main__':main()
