#!/usr/bin/env python3
"""Export this checkpoint's vision encoder plus connector with a fixed input."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import json
import os
import sys
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'

import numpy as np
import torch

from qvla.evaluation.haq_offline_eval import load_policy
from qvla.haq.offline_actions import file_sha256


ROOT=Path(__file__).resolve().parents[2]


class VisionAndConnector(torch.nn.Module):
    def __init__(self,policy):
        super().__init__()
        vlm=policy.model.vlm_with_expert.get_vlm_model()
        self.vision=vlm.vision_model
        self.connector=vlm.connector

    def forward(self,pixel):
        # The pinned input has no padded patches. Calling encoder directly
        # avoids Transformers' tracing-incompatible all-valid mask builder.
        patch=self.vision.patch_size
        embeddings=self.vision.embeddings
        patch_values=embeddings.patch_embedding(pixel)
        hidden=patch_values.flatten(2).transpose(1,2)
        h,w=pixel.shape[2]//patch,pixel.shape[3]//patch
        side=embeddings.num_patches_per_side
        positions=(torch.arange(h,device=pixel.device)[:,None]*side+
                   torch.arange(w,device=pixel.device)[None,:]).reshape(1,-1)
        hidden=hidden+embeddings.position_embedding(positions)
        encoded=self.vision.encoder(inputs_embeds=hidden,attention_mask=None)
        states=self.vision.post_layernorm(encoded.last_hidden_state)
        return self.connector(states)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/transfer/model')
    p.add_argument('--vlm-assets-dir',type=Path,default=ROOT/'artifacts/transfer/smolvlm2_assets')
    p.add_argument('--boundary-npz',type=Path,default=ROOT/'runs/smolvla_split_contract_v1.npz')
    p.add_argument('--output',type=Path,default=ROOT/'runs/smolvla_vision_split_v1/vision_connector.onnx')
    p.add_argument('--onnx-site-packages',type=Path,
                   help='Path to an existing compatible ONNX installation if absent from this interpreter')
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.onnx_site_packages is not None:
        sys.path.append(str(args.onnx_site_packages))
    import onnx
    torch.set_num_threads(4)
    policy,_,_=load_policy(args)
    # Rockchip's own SmolVLM export path uses eager attention for ONNX tracing.
    policy.model.vlm_with_expert.vlm.set_attn_implementation('eager')
    wrapper=VisionAndConnector(policy).eval().to('cpu',dtype=torch.float32)
    with np.load(args.boundary_npz,allow_pickle=False) as archive:
        pixels=torch.from_numpy(archive['vision_input_0'].copy()).float()
        expected=archive['connector_output_0'].copy()
    with torch.inference_mode():
        actual=wrapper(pixels).detach().cpu().numpy()
    boundary_max_abs=float(np.max(np.abs(actual-expected)))
    boundary_mae=float(np.mean(np.abs(actual-expected)))
    if not np.allclose(actual,expected,rtol=1e-4,atol=1e-4):
        raise ValueError(f'Vision/connector differs from pinned FP boundary: max abs {boundary_max_abs}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(wrapper,pixels,str(args.output),dynamo=False,
                      input_names=['pixel'],output_names=['features'],opset_version=17,
                      do_constant_folding=True)
    model=onnx.load(str(args.output),load_external_data=True)
    onnx.checker.check_model(model)
    report={'scope':'FP vision+connector ONNX export; no RKNN conversion or board execution',
            'source_checkpoint_sha256':file_sha256(args.model_dir/'model.safetensors'),
            'boundary_npz_sha256':file_sha256(args.boundary_npz),
            'onnx_sha256':file_sha256(args.output),'onnx_bytes':args.output.stat().st_size,
            'input_shape':list(pixels.shape),'output_shape':list(actual.shape),
            'fp_boundary_allclose_rtol_1e_4_atol_1e_4':True,
            'fp_boundary_max_abs':boundary_max_abs,'fp_boundary_mae':boundary_mae,'opset':17,
            'onnx_version':onnx.__version__,'torch_version':torch.__version__,
            'script_sha256':file_sha256(Path(__file__))}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
