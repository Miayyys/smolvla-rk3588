"""Combine paired board screenings and same-input first action chunks."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--inputs',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();pairs=[];timings=[];sources=[];identities=[];seen=set()
for root in a.inputs:
 file=root/'summary.json';s=json.loads(file.read_text())
 if not s['paired_initial_observations_and_noise_verified']:raise ValueError('Unpaired screening')
 sources.append({'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
 identities.append(s['board']['graphs'])
 if identities[-1]!=identities[0]:raise ValueError('Models changed between screenings')
 for row in s['pairs']:
  suite=row['suite'];task=row['task_id'];key=(suite,task)
  if key in seen:raise ValueError('Repeated task would be counted twice')
  seen.add(key);f=root/f'fp_{suite}_{task}';b=root/f'board_{suite}_{task}'
  with np.load(f/'input_000.npz',allow_pickle=False) as x,np.load(b/'input_000.npz',allow_pickle=False) as y:
   if set(x.files)!=set(y.files) or any(not np.array_equal(x[n],y[n]) for n in x.files):raise ValueError('First observations/noise differ')
  fp=np.load(f/'actions_000.npy');board=np.load(b/'actions_000.npy')
  if fp.shape!=(1,50,7) or board.shape!=fp.shape:raise ValueError('Action shape changed')
  error=np.abs(board.astype('f8')-fp.astype('f8'))
  row=dict(row,first_chunk_same_input=True,translation_mae=float(error[...,:3].mean()),rotation_mae=float(error[...,3:6].mean()),
   continuous6_mae=float(error[...,:6].mean()),gripper_sign_disagreements=int(((board[...,6]>0)!=(fp[...,6]>0)).sum()))
  pairs.append(row)
 for row in s['records']:
  if row['mode']=='board':timings.extend(x['inference_ms'] for x in row['trace'])
report={'scope':'limited single-seed paired closed-loop board screening, no formal noninferiority or real-time claim',
 'sources':sources,'loaded_graphs':identities[0],'pairs':pairs,'episodes_per_mode':len(pairs),
 'successes':{'fp':sum(x['fp_success'] for x in pairs),'board':sum(x['board_success'] for x in pairs)},
 'board_rollout_query_timing':{'n':len(timings),'p50_ms':float(np.median(timings)),'p95_ms':float(np.percentile(timings,95)),
 'scope':'mixed task queries including first queries; excludes simulation/RPC; not fixed-input warmup benchmark'}}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('sources','loaded_graphs')},indent=2))
