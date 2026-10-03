"""Compile prepared V1 partitions, preserving BF16/INT16 DFP exceptions."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--kind',choices=['expert','prefix'],required=True)
p.add_argument('--work-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.work_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
project=Path(__file__).resolve().parents[2];started=time.time()
state={'kind':a.kind,'scope':'V1 native format conversion, not GPU quantizer equivalence or task quality','parts':{}}
for part in ['before_projection','projection','after_projection']:
 name=a.kind+'_'+part;work=root/part;target=out/(name+'.rknn')
 if part=='projection':
  command=[sys.executable,str(project/'qvla/conversion/compile_v1_isolated_projection.py'),'--root',str(work),'--kind',a.kind,'--format','bfloat16' if a.kind=='expert' else 'w16a16i_dfp']
  source=work/'projection.rknn';report_path=work/'compile_report.json'
 else:
  command=[sys.executable,str(project/'qvla/conversion/compile_selected_qat_rknn.py'),'--root',str(work),'--graph',a.kind,'--output-name',name+'.rknn']
  source=work/(name+'.rknn');report_path=work/(a.kind+'_rknn/report.json')
 log=out/(name+'.log');state['stage']=name;state['status']='running';(out/(a.kind+'_conversion.json')).write_text(json.dumps(state,indent=2)+'\n')
 print(json.dumps({'stage':name,'state':'running','elapsed_seconds':time.time()-started}),flush=True)
 env=dict(os.environ,TMPDIR='/dev/shm',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
 with log.open('w') as f:rc=subprocess.run(command,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 if rc or not source.exists():
  state.update(status='failed',returncode=rc);(out/(a.kind+'_conversion.json')).write_text(json.dumps(state,indent=2)+'\n');raise RuntimeError(name+' conversion failed; see '+str(log))
 shutil.copyfile(source,target);shutil.copyfile(report_path,out/(name+'_compile_report.json'))
 state['parts'][part]={'filename':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
 # Only duplicate ONNX compiler checks generated inside this new RAM workdir.
 # Original ONNX, profiles, logs, final models and boundary inputs are retained.
 for base in [work,work/(a.kind+'_rknn')]:
  for check in base.glob('check*.onnx'):check.unlink()
 print(json.dumps({'stage':name,'state':'compiled','bytes':target.stat().st_size,'elapsed_seconds':time.time()-started}),flush=True)
for f in ['split_report.json','calibration_report.json']:shutil.copyfile(root/f,out/(a.kind+'_'+f))
state.update(status='compiled',elapsed_seconds=time.time()-started,bytes=sum(v['bytes'] for v in state['parts'].values()))
(out/(a.kind+'_conversion.json')).write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
