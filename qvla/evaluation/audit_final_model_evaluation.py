"""Audit saved paired evaluation and reproduce three preselected regressions."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/final_model_comparison_v1';OUT=RUN/'evaluation_audit';OUT.mkdir(exist_ok=True)
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
s=json.loads((RUN/'quality40/summary.json').read_text());checks={};assert len(s['pairs'])==40 and len(s['records'])==80
assert len({(x['suite'],x['task_id'],x['mode']) for x in s['records']})==80
checks['unique_complete_40_pairs']=True
for key,sha in json.loads((RUN/'source_hashes.json').read_text()).items():
 if key in ('scripts/benchmark_smolvla_board_resources.py','qvla/evaluation/benchmark_smolvla_board_resources.py'):continue # resource-only sampler subsequently repaired
 path=ROOT/key
 if key.startswith('scripts/') and not path.exists():
  content=subprocess.check_output(['git','show','22fac9d:'+key],cwd=ROOT)
 else:content=path.read_bytes()
 assert hashlib.sha256(content).hexdigest()==sha,key
checks['frozen_quality_script_checkpoint_manifest_unchanged']=True
checks['historical_source_revision']='22fac9d; historical code, not a claim that reorganized code is byte-identical'
for row in s['records']:
 assert row['init_state_index']==0 and row['env_seed']==0
 assert row['noise_seed']==100000*(SUITES.index(row['suite'])+1)+row['task_id']
 assert row['action_steps_per_chunk']==50
 folder=RUN/'quality40'/f"{row['mode']}_{row['suite']}_{row['task_id']}"
 expected=np.concatenate([np.load(folder/f'actions_{i:03d}.npy')[0] for i in range(len(row['trace']))])[:row['simulation_steps']]
 np.testing.assert_array_equal(expected,np.load(folder/'executed_actions.npy'))
 assert row['simulation_steps']<=row['max_steps']
 for trace in row['trace']:
  with np.load(folder/f"input_{trace['query']:03d}.npz") as raw:
   assert hashlib.sha256(raw['noise'].tobytes()).hexdigest()==trace['noise_sha256']
 for trace in row['trace']:
  if row['mode']=='board':assert trace['language_backend']=='rkllm' and trace['calls']==dict(vision=2,prefix=1,expert=30)
checks['all_noise_hashes_seeds_steps_backend_and_executed_actions_valid']=True
reused=[]
for previous in ['short_tasks','expanded_tasks']:
 r=ROOT/'runs/v1_rkllm_original_fp16_vision_v1'/previous
 for old in json.loads((r/'summary.json').read_text())['records']:
  current=next(c for c in s['records'] if all(c[k]==old[k] for k in ['suite','task_id','mode']))
  name=f"{old['mode']}_{old['suite']}_{old['task_id']}"
  assert old['success']==current['success'] and old['simulation_steps']==current['simulation_steps']
  np.testing.assert_array_equal(np.load(r/name/'executed_actions.npy'),np.load(RUN/'quality40'/name/'executed_actions.npy'))
  reused.append(name)
checks['previous_six_tasks_both_models_trajectories_bitwise_equal']=reused
errors=[]
for file in [RUN/'quality40.log',RUN/'quality40/board_runtime.log']:
 for line in file.open():
  if any(x in line for x in ['E RKNN','E rkllm','CALLBACK_ERROR','Traceback','"error":']):errors.append(line[:200])
assert not errors;checks['no_sdk_or_rpc_errors']=True
(OUT/'static_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({'stage':'static','passed':True}),flush=True)
results=[];started=time.monotonic()
for suite,task in [('libero_spatial',2),('libero_goal',4),('libero_10',2)]:
 name=f'{suite}_{task}';dest=OUT/name
 print(json.dumps({'stage':name,'state':'running'}),flush=True)
 with (OUT/(name+'.log')).open('w') as log:
  subprocess.run([sys.executable,'-u',str(ROOT/'qvla/evaluation/run_smolvla_board_libero.py'),'--board-root','/dev/shm/qvla_v1_rkllm_original_fp16_vision_v1',
   '--suites',suite,'--task-id',str(task),'--seed','0','--replay-inputs',str(ROOT/'runs/smolvla_raw_board_v1/raw_inputs.npz'),
   '--replay-reference',str(ROOT/'runs/v1_rkllm_original_fp16_vision_v1/raw_replay_reference.npz'),'--output',str(dest)],
   stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
 current=json.loads((dest/'summary.json').read_text())
 for row in current['records']:
  old=next(x for x in s['records'] if x['mode']==row['mode'] and x['suite']==suite and x['task_id']==task)
  folder=f"{row['mode']}_{suite}_{task}";a=np.load(RUN/'quality40'/folder/'executed_actions.npy');b=np.load(dest/folder/'executed_actions.npy')
  results.append({'suite':suite,'task_id':task,'mode':row['mode'],'old_success':old['success'],'repeat_success':row['success'],
   'old_steps':old['simulation_steps'],'repeat_steps':row['simulation_steps'],'trajectory_bitwise_equal':bool(a.shape==b.shape and np.array_equal(a,b))})
 (OUT/'repeat_results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps({'stage':name,'state':'completed','results':results[-2:]}),flush=True)
(OUT/'summary.json').write_text(json.dumps({'static_passed':True,'repeat_results':results,'elapsed_seconds':time.monotonic()-started,
 'scope':'audits test correctness and selected failure reproducibility, does not localize model accuracy loss'},indent=2)+'\n')
print(json.dumps({'stage':'audit','state':'completed'}),flush=True)
