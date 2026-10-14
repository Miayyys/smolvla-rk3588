"""Locate first tensor divergence with frozen observations and expert core changes."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def sha(array):
    value=np.ascontiguousarray(array)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()

def describe(array):
    value=np.asarray(array)
    return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':sha(value),
            'min':float(value.min()),'max':float(value.max()),'finite':bool(np.isfinite(value).all())}

def replay_frozen(a):
    from rknnlite.api import RKNNLite
    manifest=json.loads((a.root/'deployment_manifest.json').read_text())
    part=next(p for p in manifest['partitioned_graphs']['expert']['parts'] if p['filename']==a.part)
    with np.load(a.frozen,allow_pickle=False) as values:
        inputs=[values['input_'+str(i)].copy() for i in range(len(part['input_names']))]
        reference=[values['output_'+str(i)].copy() for i in range(len(part['output_names']))]
    model=RKNNLite();results=[];outputs=[]
    try:
        if model.load_rknn(str(a.root/a.part)) or model.init_runtime(core_mask=a.core_mask):raise RuntimeError('Frozen replay initialization failed')
        for iteration in range(a.repeats):
            copied=[x.copy() for x in inputs];t=time.perf_counter()
            result=model.inference(inputs=copied,data_format=['nchw']*len(copied))
            if result is None:raise RuntimeError('Inference failed')
            metrics=[]
            for name,x,y in zip(part['output_names'],result,reference):
                d=np.abs(x.astype('f8')-y.astype('f8'))
                metrics.append({'name':name,'exact':bool(np.array_equal(x,y)),'max_abs':float(d.max()),'mae':float(d.mean()),
                    'changed_fraction':float(np.mean(x!=y)),'output':describe(x)})
            outputs.append([x.copy() for x in result])
            results.append({'iteration':iteration,'seconds':time.perf_counter()-t,'outputs':metrics})
        report={'part':a.part,'core_mask':a.core_mask,'repeats':a.repeats,
                'all_outputs_exact':all(m['exact'] for r in results for m in r['outputs']),
                'repeat_max_abs':max(float(np.max(np.abs(x.astype('f8')-y.astype('f8')))) for result in outputs for x,y in zip(result,outputs[0])),
                'inputs':[describe(x) for x in inputs],'results':results}
        (a.output/'frozen_report.json').write_text(json.dumps(report,indent=2))
        np.savez(a.output/'frozen_outputs.npz',**{f'run{j}_output{i}':x for j,result in enumerate(outputs) for i,x in enumerate(result)})
        print(json.dumps({k:v for k,v in report.items() if k not in ['results','inputs']}),flush=True)
    finally:model.release()


def replay_native(a):
    import ctypes as c
    m=json.loads((a.root/'deployment_manifest.json').read_text())
    part=next(p for p in m['partitioned_graphs']['expert']['parts'] if p['filename']==a.part)
    with np.load(a.frozen,allow_pickle=False) as v:
        inputs=[np.ascontiguousarray(v['input_'+str(i)]) for i in range(len(part['input_names']))]
        reference=[v['output_'+str(i)].copy() for i in range(len(part['output_names']))]
    lib=c.CDLL(str(a.native.resolve()));lib.qvla_probe_open.argtypes=[c.c_char_p,c.c_int,c.c_uint32,c.POINTER(c.c_int)];lib.qvla_probe_open.restype=c.c_void_p
    lib.qvla_probe_close.argtypes=[c.c_void_p]
    lib.qvla_probe_run.argtypes=[c.c_void_p,c.POINTER(c.c_void_p),c.POINTER(c.c_uint32),c.POINTER(c.c_int),c.c_uint32,c.POINTER(c.c_void_p),c.POINTER(c.c_uint32),c.c_uint32,c.c_int]
    rc=c.c_int();handle=lib.qvla_probe_open(str(a.root/a.part).encode(),a.core_mask,a.init_flags,c.byref(rc))
    if not handle:raise RuntimeError('Native init failed: '+str(rc.value))
    lib.qvla_probe_input_format.argtypes=[c.c_void_p,c.c_uint32]
    formats=[lib.qvla_probe_input_format(handle,i) for i in range(len(inputs))]
    inputs=[np.ascontiguousarray(x.transpose(0,2,3,1)) if x.ndim==4 and fmt==1 else x for x,fmt in zip(inputs,formats)]
    dtype_map={'float32':0,'float16':1,'int8':2,'uint8':3,'int16':4,'int32':6,'int64':8,'bool':9}
    n=len(inputs);no=len(reference);results=[];values=[]
    try:
        for i in range(a.repeats):
            out=[np.empty_like(x) for x in reference]
            assert all(x.dtype==np.float32 for x in out)
            ptr=(c.c_void_p*n)(*[x.ctypes.data for x in inputs]);sizes=(c.c_uint32*n)(*[x.nbytes for x in inputs]);types=(c.c_int*n)(*[dtype_map[str(x.dtype)] for x in inputs])
            outptr=(c.c_void_p*no)(*[x.ctypes.data for x in out]);outsizes=(c.c_uint32*no)(*[x.nbytes for x in out])
            t=time.perf_counter();rc=lib.qvla_probe_run(handle,ptr,sizes,types,n,outptr,outsizes,no,a.delay_ms)
            if rc:raise RuntimeError('Native run failed: '+str(rc))
            metrics=[{'name':name,'exact':bool(np.array_equal(x,y)),'max_abs':float(np.max(np.abs(x.astype('f8')-y.astype('f8')))),
                      'sha256':sha(x)} for name,x,y in zip(part['output_names'],out,reference)]
            results.append({'iteration':i,'seconds':time.perf_counter()-t,'outputs':metrics});values.append(out)
        report={'backend':'standard_native_c','core_mask':a.core_mask,'init_flags':a.init_flags,'delay_ms':a.delay_ms,'input_formats':formats,
                'all_outputs_exact':all(x['exact'] for r in results for x in r['outputs']),
                'repeat_max_abs':max(float(np.max(np.abs(x.astype('f8')-y.astype('f8')))) for row in values for x,y in zip(row,values[0])),
                'results':results}
        if a.init_flags & 8:
            lib.qvla_probe_perf.argtypes=[c.c_void_p,c.c_char_p]
            report['perf_status']=lib.qvla_probe_perf(handle,str(a.output/'operator_perf.txt').encode())
        (a.output/'native_report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
    finally:lib.qvla_probe_close(handle)


def replay_simple(a):
    from rknnlite.api import RKNNLite
    if a.simple_inputs:
        with np.load(a.simple_inputs,allow_pickle=False) as v:input_values=[v['input_'+str(i)].copy() for i in range(len(v.files))]
    else:
        with np.load(a.frozen,allow_pickle=False) as v:input_values=[v['output_5'].copy()]
        assert input_values[0].shape==(1,50,720)
    cases=[];reference=None
    for mask in (1,3,7):
        model=RKNNLite();values=[]
        try:
            if model.load_rknn(str(a.simple_model)) or model.init_runtime(core_mask=mask):raise RuntimeError('Simple model initialization failed')
            for i in range(a.repeats):
                result=model.inference(inputs=[x.copy() for x in input_values],data_format=['nchw']*len(input_values))
                values.append([v.copy() for v in result])
            if reference is None:reference=values[0]
            cases.append({'core_mask':mask,'all_exact':all(np.array_equal(v,r) for row in values for v,r in zip(row,reference)),
                'repeat_max_abs':max(float(np.abs(v-r).max()) for row in values for v,r in zip(row,values[0])),
                'max_abs_vs_core0':max(float(np.abs(v-r).max()) for row in values for v,r in zip(row,reference))})
        finally:model.release()
    report={'model':str(a.simple_model),'samples':[describe(x) for x in input_values],'repeats':a.repeats,'cases':cases}
    (a.output/'simple_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--inputs',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expert-core',default='0')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--frozen',type=Path)
    p.add_argument('--part',default='expert_before_projection.rknn')
    p.add_argument('--core-mask',type=int,default=1)
    p.add_argument('--native',type=Path)
    p.add_argument('--init-flags',type=int,default=0)
    p.add_argument('--delay-ms',type=int,default=0)
    p.add_argument('--simple-model',type=Path)
    p.add_argument('--simple-inputs',type=Path)
    a=p.parse_args()
    if a.repeats<2:p.error('At least two repeats are required to test stability')
    if a.simple_model and not (a.simple_inputs or a.frozen):p.error('Simple controls need --simple-inputs or --frozen')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.simple_model:return replay_simple(a)
    if a.native:
        if not a.frozen:p.error('--native needs --frozen')
        return replay_native(a)
    if a.frozen:return replay_frozen(a)
    if a.inputs is None:p.error('--inputs is required for full-policy capture')
    from qvla.runtime.smolvla_board_runtime import BoardSmolVLA
    with np.load(a.inputs,allow_pickle=False) as values:raw={k:values[k][0].copy() for k in values.files}
    model=BoardSmolVLA(a.root,options={'vision_cores':['0'],'expert_core':a.expert_core})
    traces=[];iteration=0
    try:
        language=model.language;original_language=language.infer
        def language_tap(prefix,pad):
            result=original_language(prefix,pad)
            trace={'iteration':iteration,'stage':'language','inputs':{'prefix':describe(prefix),'pad':describe(pad)},
                   'outputs':{f'kv_{i}':describe(x) for i,x in enumerate(result)}}
            traces.append(trace)
            if iteration==0:
                np.savez(a.output/'language_first.npz',prefix=prefix,pad=pad,**{f'kv_{i}':x for i,x in enumerate(result)})
            return result
        language.infer=language_tap
        partmodel=model.partition_models['expert']
        class Tap:
            def __init__(self,inner,part):self.inner=inner;self.part=part;self.call=0
            def inference(self,inputs,**kwargs):
                call=self.call;self.call+=1;step=call%10
                before={k:describe(x) for k,x in zip(self.part['input_names'],inputs)}
                saved_inputs=[v.copy() for v in inputs] if iteration==0 and step==0 else None
                t=time.perf_counter();result=self.inner.inference(inputs=inputs,**kwargs)
                traces.append({'iteration':iteration,'stage':self.part['filename'],'step':step,
                    'inputs':before,'outputs':{k:describe(x) for k,x in zip(self.part['output_names'],result)},
                    'seconds':time.perf_counter()-t})
                if iteration==0 and step==0:
                    np.savez(a.output/(self.part['filename']+'.first.npz'),
                        **{'input_'+str(i):v for i,v in enumerate(saved_inputs)},
                        **{'output_'+str(i):v.copy() for i,v in enumerate(result)})
                return result
            def release(self):return self.inner.release()
        partmodel.models=[(Tap(inner,part),part) for inner,part in partmodel.models]
        results=[]
        for iteration in range(a.repeats):
            actions,timing=model.predict(raw);results.append(actions.copy())
            print(json.dumps({'iteration':iteration,'expert_core':a.expert_core,'action':describe(actions),'inference_ms':timing['inference_ms']}),flush=True)
        np.savez(a.output/'actions.npz',actions=np.stack(results))
        (a.output/'trace.json').write_text(json.dumps(traces,indent=2))
        report={'expert_core':a.expert_core,'repeats':a.repeats,
                'repeat_max_abs':max(float(np.abs(x-results[0]).max()) for x in results),
                'fixed_input':{k:describe(v) for k,v in raw.items() if v.dtype.kind not in ('U','S')},
                'loaded_graphs':model.hashes}
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
    finally:model.close()

if __name__=='__main__':main()
