#!/usr/bin/env python3
"""Compile three selected-master RKNN graphs, retaining explicit precision mapping."""
import argparse
import json
import os
from pathlib import Path
import time
import yaml
from rknn.api import RKNN
import hashlib


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ok(stage,rc):
    if rc!=0:raise RuntimeError(stage+' returned '+str(rc))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--graph',choices=('vision','prefix','expert'),required=True)
    p.add_argument('--bf16-as-fp16',action='store_true',help='Explicit backend-adaptation experiment, never exact V1 equivalence')
    p.add_argument('--reuse-profile',action='store_true',help='Reuse isolated calibration profile after a mapping-only failure')
    p.add_argument('--output-name',help='Distinct artifact for a corrected backend mapping')
    p.add_argument('--mode',choices=('mixed','fp16'),default='mixed',help='FP16 is a diagnostic conversion, not the selected HAQ assignment')
    p.add_argument('--algorithm',choices=('normal','mmse','kl_divergence'),default='normal')
    p.add_argument('--float-nonparam',action='store_true',help='Backend repair: FP16 nonparameter operations, retain INT8 projection inputs')
    p.add_argument('--vision-connector-int8',action='store_true',help='Diagnostic repair: protect visual backbone, keep connector projections INT8')
    p.add_argument('--preserve-norm',action='store_true',help='Keep normalization output FP16; request weighted kernel outputs INT8 instead')
    p.add_argument('--profile-dir',type=Path,help='Reuse a matching hybrid profile in a separate repair experiment')
    a=p.parse_args();root=a.root.resolve();kind=a.graph;meta=json.loads((root/(kind+'_export.json')).read_text())
    suffix='_fp16_diagnostic' if a.mode=='fp16' else ('' if a.algorithm=='normal' else '_'+a.algorithm)
    if a.float_nonparam:suffix+='_float_nonparam'
    if a.preserve_norm:
        if not a.float_nonparam:raise ValueError('preserve-norm requires float-nonparam')
        suffix+='_preserve_norm'
    if a.vision_connector_int8:
        if kind!='vision' or a.mode!='mixed' or a.float_nonparam:raise ValueError('Connector repair requires vision mixed mode')
        suffix+='_connector_int8'
    out=root/(kind+'_rknn'+suffix);out.mkdir(exist_ok=True);r=dict(graph=kind,source_onnx_sha256=sha(root/(kind+'.onnx')),
        target='rk3588',float_dtype='float16',quantized_dtype='w8a8',quantized_method='channel',
        algorithm=a.algorithm,mode=a.mode,optimization_level=3,source_QAT_master=True,source_local_pack_equivalence=False,
        bf16_as_fp16=a.bf16_as_fp16,float_nonparam=a.float_nonparam,preserve_norm=a.preserve_norm,vision_connector_int8=a.vision_connector_int8,precision_mapping=meta['special'],status='running')
    start=time.time();m=RKNN(verbose=True,verbose_file=str(out/'build.log'));previous=Path.cwd()
    try:
        os.chdir(out)
        ok('config',m.config(target_platform='rk3588',quantized_dtype='w8a8',quantized_method='channel',quantized_algorithm=a.algorithm,float_dtype='float16',optimization_level=3))
        ok('load',m.load_onnx(model=str(root/(kind+'.onnx'))))
        dataset=str(root/(kind+'_dataset.txt'))
        if a.mode=='fp16':
            r.update(stage='build',quantized_dtype=None,precision_mapping={},assignment_preserved=False)
            write(out/'report.json',r)
            ok('build',m.build(do_quantization=False))
        elif not meta['special'] and not a.float_nonparam and not a.vision_connector_int8:
            r['stage']='build';write(out/'report.json',r)
            ok('build',m.build(do_quantization=True,dataset=dataset))
        else:
            r['stage']='hybrid_step1';write(out/'report.json',r)
            profile_dir=a.profile_dir.resolve() if a.profile_dir else out
            if a.profile_dir:
                previous_report=json.loads((profile_dir/'report.json').read_text())
                if previous_report['source_onnx_sha256']!=r['source_onnx_sha256'] or previous_report['algorithm']!=a.algorithm:
                    raise ValueError('Reusable profile source/algorithm differs')
            if not a.reuse_profile and not a.profile_dir:
                ok('hybrid_step1',m.hybrid_quantization_step1(dataset=dataset,proposal=False))
            profiles=[x for x in profile_dir.glob('*.quantization.cfg') if x.name!='selected.quantization.cfg']
            if len(profiles)!=1:raise ValueError('Expected one hybrid profile')
            cfg=yaml.safe_load(profiles[0].read_text());parameters=cfg['quantize_parameters'];custom={};mapping=[]
            if meta.get('quantized_input_bounds'):
                for name,(lo,hi) in meta['quantized_input_bounds'].items():
                    if name not in parameters or not lo<hi:raise ValueError('Invalid pinned input range '+name)
                    parameters[name].update(min=[lo],max=[hi],scale=[],zero_point=[])
                r['pinned_quantized_input_bounds']=meta['quantized_input_bounds']
            if a.float_nonparam or a.vision_connector_int8:
                import onnx
                model_paths=list(profile_dir.glob('*.model'))
                if len(model_paths)!=1:raise ValueError('Missing intermediate graph')
                intermediate=onnx.load(str(model_paths[0]))
                initializers={x.name for x in intermediate.graph.initializer}
                projections=[n for n in intermediate.graph.node if n.op_type in ('Conv','Gemm','MatMul') and any(k in initializers for k in n.input[1:])]
                projection_names={n.name for n in projections}
                for n in intermediate.graph.node:
                    if (a.vision_connector_int8 or n.name not in projection_names) and n.op_type!='Constant':
                        for k in n.output:
                            if k in parameters:custom[k]='float16'
                # Keep weighted kernels in INT8 even where a protected
                # normalization output is also the projection input.
                graph_inputs={x.name for x in intermediate.graph.input}
                kept_int8=[]
                for n in projections:
                    low=not a.vision_connector_int8 or n.name.startswith('/connector/modality_projection/proj/')
                    if low:kept_int8.append(n.name)
                    if not a.preserve_norm and n.input and n.input[0] in parameters and n.input[0] not in graph_inputs:
                        custom[n.input[0]]='int8' if low else 'float16'
                    if a.preserve_norm or (a.vision_connector_int8 and low):
                        for k in [n.name,*n.output]:
                            if k in parameters:custom[k]='int8'
                r['nonparam_mapping']={'fp16_tensors':sum(v=='float16' for v in custom.values()),
                    'int8_tensors':sum(v=='int8' for v in custom.values()),
                    'int8_request_location':'weighted_outputs' if a.preserve_norm else 'projection_inputs_and_optional_outputs',
                    'weighted_compute_nodes':len(projections)}
                if a.vision_connector_int8:r['requested_int8_weight_nodes']=kept_int8
                del intermediate
            for module,spec in meta['special'].items():
                target=spec['onnx_output'];fmt=spec['format']
                # Match preserved original tensor output, allowing RKNN's documented suffixes.
                import onnx
                model_paths=list(profile_dir.glob('*.model'))
                if len(model_paths)!=1:raise ValueError('Missing intermediate graph')
                intermediate=onnx.load(str(model_paths[0]))
                stem=target.split('_output_')[0]
                nodes=[n for n in intermediate.graph.node if n.name==stem+'#2' and n.op_type in ('Conv','MatMul','Gemm')]
                if len(nodes)!=1:raise ValueError('Ambiguous compute producer '+stem)
                compute=nodes[0];matches=[k for k in compute.output if k in parameters]
                if not matches:raise ValueError('No compiled precision tensor for '+module+' / '+target)
                selected=matches[0]
                if fmt=='float16':dtype='float16'
                elif fmt=='bfloat16':dtype='float16' if a.bf16_as_fp16 else 'bfloat16'
                elif fmt=='w16a16i_dfp':dtype='int16'
                else:raise ValueError('Unknown exception '+fmt)
                custom[selected]=dtype
                # A float output alone may compile as an INT8 Conv followed
                # by a float conversion. Preserve high precision weights and
                # its input as well; audit actual Conv dtype in compiler log.
                additional=[]
                if dtype=='float16':
                    for k in compute.input:
                        if k in parameters:custom[k]=dtype;additional.append(k)
                del intermediate
                mapping.append(dict(module=module,requested=fmt,rknn_dtype=dtype,tensor=selected,additional_precision_tensors=additional,
                    exact_quantizer_equivalence=False))
            cfg['custom_quantize_layers']=custom
            chosen=out/'selected.quantization.cfg';chosen.write_text(yaml.safe_dump(cfg,sort_keys=False))
            r.update(stage='hybrid_step2',compiled_precision_mapping=mapping);write(out/'report.json',r)
            models=list(profile_dir.glob('*.model'));data=list(profile_dir.glob('*.data'))
            if len(models)!=1 or len(data)!=1:raise ValueError('Expected one intermediate model/data')
            m.release();m=RKNN(verbose=True,verbose_file=str(out/'step2.log'))
            ok('hybrid_step2',m.hybrid_quantization_step2(model_input=str(models[0]),data_input=str(data[0]),model_quantization_cfg=str(chosen)))
        name=a.output_name or kind+('_master_fp16_diagnostic.rknn' if a.mode=='fp16' else '_selected'+('' if a.algorithm=='normal' else '_'+a.algorithm)+('_float_nonparam' if a.float_nonparam else '')+('_preserve_norm' if a.preserve_norm else '')+('_connector_int8' if a.vision_connector_int8 else '')+'.rknn')
        if Path(name).name!=name:raise ValueError('Invalid output filename')
        output=root/name;ok('export',m.export_rknn(str(output)))
        r.update(status='compiled',rknn_sha256=sha(output),rknn_bytes=output.stat().st_size,
            board_execution='not_measured',QAT_closed_loop_score_transfer='not_verified')
    except BaseException as e:
        r.update(status='failed',error=repr(e));raise
    finally:
        r['elapsed_seconds']=time.time()-start;write(out/'report.json',r);m.release();os.chdir(previous)
        print(json.dumps(r),flush=True)

if __name__=='__main__':main()
