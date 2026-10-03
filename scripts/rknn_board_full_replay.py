#!/usr/bin/env python3
"""Run vision, prefix and ten expert steps entirely on RK3588 from processed inputs."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite
from smolvla_numpy_glue import assemble_prefix,time_embedding,postprocess


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def error(actual,reference):
    d=actual.astype(np.float64)-reference.astype(np.float64)
    return {'mae':float(np.abs(d).mean()),'rmse':float(np.sqrt(np.mean(d*d))),
            'max_abs_error':float(np.abs(d).max())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('vision','prefix','expert','inputs','weights','reference','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--preprocessor-assets',type=Path,help='If set, inputs contains raw RGB/state/task/noise')
    p.add_argument('--preprocessor-manifest',type=Path,help='Pinned hashes for the raw inputs and preprocessing assets')
    p.add_argument('--prefix-input-scale',type=Path,help='Reversible input scale required by a balanced-prefix graph')
    args=p.parse_args()
    prefix_scale=None
    if args.prefix_input_scale:
        prefix_scale=np.load(args.prefix_input_scale,allow_pickle=False)
        if prefix_scale.shape!=(1,177,960) or not np.isfinite(prefix_scale).all() or np.any(prefix_scale<=0):
            raise ValueError('Invalid prefix input scale')
    config=json.loads(args.config.read_text())
    for name,path in [('cpu_weights.npz',args.weights),('fp_reference.npz',args.reference)]:
        if digest(path)!=config['hashes'][name]:raise ValueError('Replay hash mismatch: '+name)
    with np.load(args.inputs,allow_pickle=False) as z:source_inputs={n:z[n].copy() for n in z.files}
    preprocessor=None
    if args.preprocessor_assets:
        from smolvla_board_preprocess import SmolVLAPreprocessor
        if not args.preprocessor_manifest:raise ValueError('Raw mode requires a pinned preprocessing manifest')
        pm=json.loads(args.preprocessor_manifest.read_text());assets=args.preprocessor_assets
        if pm['checkpoint_sha256']!=config['checkpoint_sha256'] or pm['sample']!=config['sample']:
            raise ValueError('Preprocessing sample/model mismatch')
        if digest(args.inputs)!=pm['hashes']['raw_inputs.npz']:raise ValueError('Raw input hash mismatch')
        for name,h in pm['config_hashes'].items():
            if digest(assets/name)!=h:raise ValueError('Preprocessing asset hash mismatch: '+name)
        if digest(assets/'state_stats.npz')!=pm['hashes']['state_stats.npz']:raise ValueError('State statistics hash mismatch')
        preprocessor=SmolVLAPreprocessor(assets/'tokenizer.json',assets/'tokenizer_config.json',
            assets/'policy_preprocessor.json',assets/'config.json',assets/'state_stats.npz')
    elif digest(args.inputs)!=config['hashes']['replay_inputs.npz']:
        raise ValueError('Processed input hash mismatch')
    with np.load(args.weights,allow_pickle=False) as z:weights={n:z[n].copy() for n in z.files}
    with np.load(args.reference,allow_pickle=False) as z:refs={n:z[n].copy() for n in z.files}
    report={'scope':'complete RK3588 processed-observation replay: 2 vision, 1 prefix, 10 expert calls; host preprocessing excluded',
            'sample':config['sample'],'checkpoint_sha256':config['checkpoint_sha256'],
            'reference_kind':config.get('reference_kind','original_FP'),
            'npu_core_mask':'NPU_CORE_0','status':'running','warmup':args.warmup,'repeats':args.repeats,
            'closed_loop_success_rate':'not_measured','hashes':{n:digest(getattr(args,n)) for n in
            ('vision','prefix','expert','inputs','weights','reference','config')}}
    if preprocessor:
        report.update(scope='complete RK3588 raw-observation replay: image resize/normalization, task tokenization, state normalization and all network computation',
            preprocessing_on_board=True,preprocessor_manifest_sha256=digest(args.preprocessor_manifest),
            preprocessing_asset_hashes=pm['config_hashes'],state_stats_sha256=pm['hashes']['state_stats.npz'])
    if prefix_scale is not None:report['prefix_input_scale_sha256']=digest(args.prefix_input_scale)
    models=[];runs=[]
    try:
        start=time.perf_counter()
        for path in (args.vision,args.prefix,args.expert):
            model=RKNNLite();models.append(model)
            if model.load_rknn(str(path)) or model.init_runtime(core_mask=RKNNLite.NPU_CORE_0):
                raise RuntimeError('RKNN initialization failed: '+str(path))
        report['model_load_seconds']=time.perf_counter()-start
        def infer(model,x):
            # Lite2 mutates the format list; create one for every call.
            y=model.inference(inputs=[np.ascontiguousarray(v) for v in x],data_format=['nchw']*len(x))
            if y is None or any(not np.isfinite(v).all() for v in y):raise RuntimeError('Invalid RKNN output')
            return y
        saved_actions=[]
        for run in range(args.warmup+args.repeats):
            start=time.perf_counter();features=[];vision_ms=[]
            if preprocessor:
                inputs=preprocessor(source_inputs['image1'],source_inputs['image2'],source_inputs['state'],
                    str(source_inputs['task'].item()),source_inputs['noise'])
            else:inputs=source_inputs
            preprocess_ms=(time.perf_counter()-start)*1000
            for x in inputs['images']:
                t=time.perf_counter();features.append(infer(models[0],[x])[0]);vision_ms.append((time.perf_counter()-t)*1000)
            t=time.perf_counter();prefix,pad,mask,pos=assemble_prefix(features,inputs,weights)
            prefix_glue_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            prefix_input=prefix if prefix_scale is None else prefix/prefix_scale
            out=infer(models[1],[prefix_input,mask,pos]);prefix_ms=(time.perf_counter()-t)*1000
            if len(out)!=33:raise ValueError('Expected hidden + 32 KV outputs')
            kv=out[1:];actions=inputs['noise'].copy();expert_ms=[];time_ms=[];euler_ms=[];velocities=[]
            dt=-1./config['num_steps']
            for step in range(config['num_steps']):
                t=time.perf_counter()
                embedding=time_embedding(np.array([1.+step*dt],dtype=np.float32),config['expert_hidden_size'],config['min_period'],config['max_period'])
                time_ms.append((time.perf_counter()-t)*1000)
                t=time.perf_counter();velocity=infer(models[2],[actions,embedding,pad,*kv])[0]
                if velocity.shape!=actions.shape:raise ValueError('Velocity shape mismatch')
                expert_ms.append((time.perf_counter()-t)*1000);velocities.append(velocity.copy())
                t=time.perf_counter();actions=actions+np.float32(dt)*velocity;euler_ms.append((time.perf_counter()-t)*1000)
            t=time.perf_counter();final=postprocess(actions,weights);post_ms=(time.perf_counter()-t)*1000
            row={'total_ms':(time.perf_counter()-start)*1000,'preprocess_ms':preprocess_ms,'vision_ms':vision_ms,'prefix_glue_ms':prefix_glue_ms,
                 'prefix_ms':prefix_ms,'expert_ms':expert_ms,'time_embedding_ms':time_ms,'euler_ms':euler_ms,'postprocess_ms':post_ms}
            if run>=args.warmup:runs.append(row);saved_actions.append(final.copy())
            print(json.dumps({'run':run,**row}),flush=True)
        values_path=args.output.with_suffix('.npz')
        np.savez_compressed(values_path,actions=final,raw_actions=actions,prefix=prefix,
                            features=np.stack(features),velocities=np.stack(velocities),repeated_actions=np.stack(saved_actions))
        report.update(status='success',runs=runs,latency_p50_ms=float(np.percentile([r['total_ms'] for r in runs],50)),
            latency_p95_ms=float(np.percentile([r['total_ms'] for r in runs],95)),output_sha256=digest(values_path),
            action_vs_original=error(final,refs['actions']),raw_action_vs_original=error(actions,refs['raw_actions']),
            per_dimension_action_max_abs=np.abs(final-refs['actions']).max(axis=(0,1)).tolist(),
            gripper_sign_disagreements=int(np.count_nonzero((final[:,:,6]>0)!=(refs['actions'][:,:,6]>0))),
            call_counts_per_replay={'vision':2,'prefix':1,'expert':config['num_steps']},
            loaded_graph_bytes=sum(path.stat().st_size for path in (args.vision,args.prefix,args.expert)),
            cpu_parameter_file_bytes=args.weights.stat().st_size)
        report['action_vs_reference']=report['action_vs_original']
        report['raw_action_vs_reference']=report['raw_action_vs_original']
        if report['reference_kind']!='original_FP':
            del report['action_vs_original'];del report['raw_action_vs_original']
        report['repeated_actions_exact_match']=all(np.array_equal(saved_actions[0],a) for a in saved_actions[1:])
    except Exception as e:
        report.update(status='failed',error=repr(e));raise
    finally:
        report['rusage_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for model in reversed(models):model.release()
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2),flush=True)


if __name__=='__main__':main()
