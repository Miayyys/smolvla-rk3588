#!/usr/bin/env python3
"""Paired GPU-FP / real RK3588 closed-loop LIBERO screening with videos."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import base64
import hashlib
import io
from itertools import product
import json
import os
import random
import shlex
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from qvla.evaluation.haq_offline_eval import load_policy
from qvla.haq.offline_actions import file_sha256
ROOT=Path(__file__).resolve().parents[2]
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')


def batch_observation(value):
    if isinstance(value,dict):return {k:batch_observation(v) for k,v in value.items()}
    if isinstance(value,np.ndarray):return value[None]
    return value


class BoardClient:
    def __init__(self,root,remote_root='/root/qvla/models/final',board_host='root@10.42.0.252'):
        self.log=(root/'board_runtime.log').open('w');self.counter=0
        command=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=5',board_host,
                 'cd '+shlex.quote(remote_root)+' && export OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/qvla_board_test/python_site && '
                 'if [ -f ../../scripts/deploy.py ]; then python3 -u ../../scripts/deploy.py serve --root .; '
                 'else python3 -u serve_smolvla_board_stdio.py --root .; fi']
        self.proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        self.ready=self.read()
        if not self.ready.get('ready'):raise RuntimeError('Board did not initialize')

    def read(self):
        for line in self.proc.stdout:
            self.log.write(line);self.log.flush()
            if 'E RKNN' in line or 'E rkllm' in line:raise RuntimeError('Board runtime error: '+line)
            if line.startswith('QVLA_REPLY '):return json.loads(line[len('QVLA_REPLY '):])
        raise RuntimeError('Board connection closed')

    def predict(self,raw):
        self.counter+=1;buffer=io.BytesIO();np.savez_compressed(buffer,**raw);payload=buffer.getvalue()
        start=time.perf_counter()
        self.proc.stdin.write(json.dumps({'id':self.counter,'npz':base64.b64encode(payload).decode()})+'\n');self.proc.stdin.flush()
        result=self.read()
        if result.get('id')!=self.counter or 'error' in result:raise RuntimeError(str(result))
        actions=np.asarray(result.pop('actions'),dtype=np.float32)
        if actions.shape!=(1,50,7) or not np.isfinite(actions).all():raise ValueError('Invalid action chunk')
        return actions,{**result,'rpc_ms':(time.perf_counter()-start)*1000,'raw_npz_sha256':hashlib.sha256(payload).hexdigest()}

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write(json.dumps({'quit':True})+'\n');self.proc.stdin.flush();self.proc.stdin.close()
            try:self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:self.proc.terminate();self.proc.wait(timeout=5)
        self.log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'runs/smolvla_board_libero_v1')
    p.add_argument('--suites',nargs='+',choices=SUITES,default=list(SUITES));p.add_argument('--task-id',type=int,default=0)
    p.add_argument('--task-ids',type=int,nargs='+',help='Multiple task IDs, 0..9, with one persistent runtime')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda')
    p.add_argument('--board-root',default='/root/qvla/models/final')
    p.add_argument('--board',default='root@10.42.0.252')
    p.add_argument('--replay-inputs',type=Path,default=ROOT/'runs/smolvla_raw_board_v1/raw_inputs.npz')
    p.add_argument('--replay-reference',type=Path,default=ROOT/'runs/smolvla_raw_board_v1/raw_full_board_report.npz')
    p.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/transfer/model')
    p.add_argument('--vlm-assets-dir',type=Path,default=ROOT/'artifacts/transfer/smolvlm2_assets')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    task_ids=args.task_ids if args.task_ids is not None else [args.task_id]
    if len(set(task_ids))!=len(task_ids) or any(i<0 or i>9 for i in task_ids):raise ValueError('Task IDs must be unique and 0..9')
    os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'runs/libero_local/config'))
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runs/libero_local/mpl'))
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    torch.set_num_threads(2)
    from lerobot.envs.libero import LiberoEnv,_get_suite
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    from lerobot.utils.io_utils import write_video
    policy,pre,post=load_policy(args);env_adapter=LiberoProcessorStep();board=BoardClient(args.output,args.board_root,args.board)
    if board.ready['checkpoint_sha256']!=file_sha256(args.model_dir/'model.safetensors'):raise ValueError('Checkpoint mismatch')
    records=[];started=time.time()
    try:
        # Verify the persistent runtime reproduces the already-tested exact raw replay before rollout.
        with np.load(args.replay_inputs) as z:fixed={n:z[n].copy() for n in z.files}
        values,timing=board.predict(fixed)
        with np.load(args.replay_reference) as z:np.testing.assert_array_equal(values,z['actions'])
        (args.output/'handshake.json').write_text(json.dumps({'board':board.ready,'raw_replay_exact':True,'timing':timing},indent=2)+'\n')
        for suite,task_id in product(args.suites,task_ids):
            task_seed=args.seed+100000*(SUITES.index(suite)+1)+task_id
            for mode in ('fp','board'):
                random.seed(args.seed);np.random.seed(args.seed)
                env=LiberoEnv(_get_suite(suite),task_id,suite,obs_type='pixels_agent_pos',episode_index=0,observation_height=256,observation_width=256)
                out=args.output/f'{mode}_{suite}_{task_id}';out.mkdir(exist_ok=True)
                trace=[];frames=[];all_actions=[];success=False;steps=0;run_started=time.time()
                generator=torch.Generator(device=args.device).manual_seed(task_seed)
                try:
                    observation,_=env.reset(seed=args.seed);policy.reset()
                    # Same environment reset and exact initial images/state for the paired runs.
                    initial=env_adapter.observation(preprocess_observation(batch_observation(observation)))
                    initial_hash=hashlib.sha256(b''.join(initial[k].numpy().tobytes() for k in ('observation.images.image','observation.images.image2','observation.state'))).hexdigest()
                    if mode=='board':
                        previous=next(r for r in records if r['suite']==suite and r['task_id']==task_id and r['mode']=='fp')
                        if initial_hash!=previous['initial_observation_sha256']:raise ValueError('Paired initial observations differ')
                    print(f'START {mode}/{suite}/{task_id}: {env.task_description}',flush=True)
                    while steps<env._max_episode_steps and not success:
                        mapped=env_adapter.observation(preprocess_observation(batch_observation(observation)));mapped['task']=env.task_description
                        noise=torch.randn((1,50,32),generator=generator,device=args.device,dtype=torch.float32)
                        raw={'image1':np.rint(mapped['observation.images.image'][0].numpy()*255).astype(np.uint8),
                             'image2':np.rint(mapped['observation.images.image2'][0].numpy()*255).astype(np.uint8),
                             'state':mapped['observation.state'][0].numpy().copy(),'task':np.array(env.task_description),
                             'noise':noise.cpu().numpy()}
                        query=len(trace);np.savez_compressed(out/f'input_{query:03d}.npz',**raw)
                        if mode=='board':actions,feedback=board.predict(raw)
                        else:
                            torch.cuda.synchronize();t=time.perf_counter()
                            with torch.inference_mode():actions=post(policy.predict_action_chunk(pre(mapped),noise=noise)).cpu().numpy()
                            torch.cuda.synchronize();feedback={'inference_ms':(time.perf_counter()-t)*1000}
                        np.save(out/f'actions_{query:03d}.npy',actions,allow_pickle=False)
                        trace.append({'query':query,'simulation_step':steps,'noise_sha256':hashlib.sha256(raw['noise'].tobytes()).hexdigest(),**feedback})
                        print(f'CHUNK {mode}/{suite} step={steps} inference={feedback["inference_ms"]:.0f}ms',flush=True)
                        for action in actions[0,:policy.config.n_action_steps]:
                            observation,_,terminated,truncated,info=env.step(action)
                            all_actions.append(action.copy());steps+=1;frames.append(env.render().copy())
                            success=bool(info['is_success'])
                            if success or terminated or truncated or steps>=env._max_episode_steps:break
                        if terminated or truncated:break
                    write_video(out/'rollout.mp4',frames,20)
                    row={'suite':suite,'task_id':task_id,'task_description':env.task_description,'mode':mode,
                         'success':success,'simulation_steps':steps,'max_steps':env._max_episode_steps,'initial_observation_sha256':initial_hash,
                         'init_state_index':0,'env_seed':args.seed,'noise_seed':task_seed,'action_steps_per_chunk':policy.config.n_action_steps,
                         'seconds':time.time()-run_started,'trace':trace,'video':str(out/'rollout.mp4')}
                    np.save(out/'executed_actions.npy',np.stack(all_actions),allow_pickle=False)
                    (out/'result.json').write_text(json.dumps(row,indent=2)+'\n');records.append(row)
                    (args.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
                    print(f'FINISH {mode}/{suite}: success={success}, steps={steps}',flush=True)
                finally:env.close()
        pairs=[]
        for suite,task_id in product(args.suites,task_ids):
            fp=next(r for r in records if r['suite']==suite and r['task_id']==task_id and r['mode']=='fp');b=next(r for r in records if r['suite']==suite and r['task_id']==task_id and r['mode']=='board')
            if fp['initial_observation_sha256']!=b['initial_observation_sha256']:raise ValueError('Paired initial observations differ')
            for a,c in zip(fp['trace'],b['trace']):
                if a['noise_sha256']!=c['noise_sha256']:raise ValueError('Paired noise differs')
            pairs.append({'suite':suite,'task_id':task_id,'fp_success':fp['success'],'board_success':b['success'],
                          'fp_steps':fp['simulation_steps'],'board_steps':b['simulation_steps']})
        result={'scope':'real RK3588 closed-loop LIBERO screening; simulator pauses during board inference, not real-time control validation',
                'pairs':pairs,'successes':{'fp':sum(p['fp_success'] for p in pairs),'board':sum(p['board_success'] for p in pairs)},
                'episodes_per_mode':len(pairs),'seed':args.seed,'board':board.ready,'paired_initial_observations_and_noise_verified':True,
                'elapsed_seconds':time.time()-started,'records':records}
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
    finally:board.close()


if __name__=='__main__':main()
