"""Map existing isolated V1 calibration through FP partition boundaries."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--split-root',type=Path,required=True)
p.add_argument('--kind',choices=['expert','prefix'],required=True)
p.add_argument('--threads',type=int,default=4)
a=p.parse_args();source=a.source_root.resolve();root=a.split_root.resolve();kind=a.kind
meta=json.loads((source/(kind+'_export.json')).read_text())
split=json.loads((root/'split_report.json').read_text())
if split['source_onnx_sha256']!=meta['onnx_sha256']:raise ValueError('Split source hash mismatch')
if split.get('fp_partition_parity',{}).get('status')!='passed':raise ValueError('Real source FP parity required')
parts=split['parts'];names=['before_projection','projection','after_projection']
options=ort.SessionOptions();options.intra_op_num_threads=a.threads;options.inter_op_num_threads=1
sessions={n:ort.InferenceSession(str(root/(n+'.onnx')),sess_options=options,providers=['CPUExecutionProvider']) for n in names[:2]}
for n in names:
 out=root/n;out.mkdir(exist_ok=True);model=onnx.load(str(root/(n+'.onnx')))
 outputs={v for node in model.graph.node for v in node.output}
 special={k:v for k,v in meta['special'].items() if v['onnx_output'] in outputs}
 link=out/(kind+'.onnx')
 if not link.exists():link.symlink_to(root/(n+'.onnx'))
 elif link.resolve()!=(root/(n+'.onnx')).resolve():raise ValueError('Unexpected existing partition source')
 m={'onnx_sha256':parts[n]['sha256'],'input_names':parts[n]['inputs'],'output_names':parts[n]['outputs'],'special':special}
 (out/(kind+'_export.json')).write_text(json.dumps(m,indent=2)+'\n')
rows=(source/(kind+'_dataset.txt')).read_text().splitlines();datasets={n:[] for n in names}
checks=[]
for i,row in enumerate(rows):
 paths=row.split()
 if len(paths)!=len(meta['input_names']):raise ValueError('Calibration column mismatch')
 files=dict(zip(meta['input_names'],map(Path,paths)))
 values={n:np.load(path,allow_pickle=False) for n,path in files.items()}
 for n in names:
  datasets[n].append(' '.join(str(files[k]) for k in parts[n]['inputs']))
  if n=='after_projection':continue
  session=sessions[n];outputs=session.run(None,{k:values[k] for k in parts[n]['inputs']})
  for j,(key,value) in enumerate(zip(parts[n]['outputs'],outputs)):
   path=root/n/'calibration'/f'{i:04d}_{j:03d}.npy';path.parent.mkdir(exist_ok=True)
   np.save(path,value,allow_pickle=False);files[key]=path;values[key]=value
  # Do not serialize the original K/V arrays again: later parts reuse their paths.
 checks.append({'row':i,'source_row_sha256':hashlib.sha256(row.encode()).hexdigest()})
 if (i+1)%10==0:print(json.dumps({'stage':'boundary_calibration','kind':kind,'rows':i+1,'total':len(rows)}),flush=True)
for n in names:(root/n/(kind+'_dataset.txt')).write_text('\n'.join(datasets[n])+'\n')
report={'scope':'same isolated calibration rows, propagated through FP partitions; not board quality',
 'kind':kind,'source_dataset_sha256':hashlib.sha256((source/(kind+'_dataset.txt')).read_bytes()).hexdigest(),
 'source_episode_ids':json.loads((source/'export_report.json').read_text())['calibration_episode_ids'],
 'rows':checks,'parts':{n:{'dataset_sha256':hashlib.sha256((root/n/(kind+'_dataset.txt')).read_bytes()).hexdigest(),'cases':len(datasets[n])} for n in names}}
(root/'calibration_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'kind':kind,'status':'prepared','cases':len(rows)}),flush=True)
