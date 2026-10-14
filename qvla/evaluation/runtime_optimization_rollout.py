"""Two real-board LIBERO trajectories, old runtime vs selected scheduling; no GPU policy."""
import argparse
import hashlib
import json
import os
import random
import shlex
import time
from pathlib import Path
import numpy as np
import torch
from qvla.evaluation.run_smolvla_board_libero import BoardClient,batch_observation,SUITES

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--board-project',default='/dev/shm/qvla_runtime_opt_v1')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'runs/libero_local/config'))
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runs/libero_local/mpl'))
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    from lerobot.envs.libero import LiberoEnv,_get_suite
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    adapter=LiberoProcessorStep();torch.set_num_threads(2);rows=[]
    for variant in ('baseline','optimized'):
        folder=args.output/variant;folder.mkdir(exist_ok=True)
        base=args.board_project
        cwd=base+'/baseline' if variant=='baseline' else base
        command='cd '+shlex.quote(cwd)+' && OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/qvla_board_test/python_site python3 -u -m qvla.runtime.serve_smolvla_board_stdio --root '+shlex.quote(base+'/models/final')
        if variant=='optimized':command+=' --runtime-config config/runtime_optimization_v1.json'
        board=BoardClient(folder,server_command=command)
        try:
            for suite,task_id in [('libero_object',0),('libero_goal',0)]:
                random.seed(0);np.random.seed(0)
                env=LiberoEnv(_get_suite(suite),task_id,suite,obs_type='pixels_agent_pos',episode_index=0,observation_height=256,observation_width=256)
                dest=folder/f'{suite}_{task_id}';dest.mkdir(exist_ok=True)
                generator=torch.Generator(device='cuda').manual_seed(100000*(SUITES.index(suite)+1)+task_id)
                executed=[];trace=[];steps=0;success=False;started=time.perf_counter()
                try:
                    obs,_=env.reset(seed=0)
                    while steps<env._max_episode_steps and not success:
                        mapped=adapter.observation(preprocess_observation(batch_observation(obs)))
                        noise=torch.randn((1,50,32),generator=generator,device='cuda',dtype=torch.float32)
                        raw={'image1':np.rint(mapped['observation.images.image'][0].numpy()*255).astype('uint8'),
                             'image2':np.rint(mapped['observation.images.image2'][0].numpy()*255).astype('uint8'),
                             'state':mapped['observation.state'][0].numpy().copy(),
                             'task':np.array(env.task_description),'noise':noise.cpu().numpy()}
                        input_hash=hashlib.sha256(b''.join(raw[k].tobytes() for k in ['image1','image2','state','noise'])).hexdigest()
                        actions,feedback=board.predict(raw)
                        np.savez_compressed(dest/f'input_{len(trace):03d}.npz',**raw)
                        np.save(dest/f'actions_{len(trace):03d}.npy',actions)
                        trace.append({'simulation_step':steps,'input_sha256':input_hash,**feedback})
                        for action in actions[0]:
                            obs,_,terminated,truncated,info=env.step(action)
                            executed.append(action.copy());steps+=1;success=bool(info['is_success'])
                            if success or terminated or truncated or steps>=env._max_episode_steps:break
                        if terminated or truncated:break
                    np.save(dest/'executed_actions.npy',np.stack(executed))
                    row={'variant':variant,'suite':suite,'task_id':task_id,'seed':0,'init_state':0,'success':success,'steps':steps,
                         'seconds':time.perf_counter()-started,'trace':trace,'task':env.task_description,'handshake':board.ready}
                    rows.append(row);(dest/'result.json').write_text(json.dumps(row,indent=2))
                    print(json.dumps({k:v for k,v in row.items() if k not in ['trace','handshake']}),flush=True)
                finally:env.close()
        finally:board.close()
    pairs=[]
    for suite in ('libero_object','libero_goal'):
        before=next(r for r in rows if r['variant']=='baseline' and r['suite']==suite)
        after=next(r for r in rows if r['variant']=='optimized' and r['suite']==suite)
        a=np.load(args.output/'baseline'/f'{suite}_0'/'executed_actions.npy')
        b=np.load(args.output/'optimized'/f'{suite}_0'/'executed_actions.npy')
        pairs.append({'suite':suite,'task_id':0,'baseline_success':before['success'],'optimized_success':after['success'],
                      'baseline_steps':before['steps'],'optimized_steps':after['steps'],'trajectory_bitwise_equal':bool(np.array_equal(a,b)),
                      'all_query_inputs_equal':[t['input_sha256'] for t in before['trace']]==[t['input_sha256'] for t in after['trace']]})
    summary={'scope':'two fixed tasks, original selected board runtime vs optimized selected board runtime; not GPU FP vs quantized; no robot or realtime control',
             'pairs':pairs,'records':rows}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'stage':'completed','pairs':pairs}),flush=True)

if __name__=='__main__':main()
