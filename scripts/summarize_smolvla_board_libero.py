#!/usr/bin/env python3
"""Summarize actual paired board rollouts and extract video snapshots."""
import hashlib
import json
from pathlib import Path
import av
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('runs/smolvla_board_libero_v1');source=root/'summary.json';d=json.loads(source.read_text())
    board=[r for r in d['records'] if r['mode']=='board'];latencies=[q['inference_ms'] for r in board for q in r['trace']]
    first_errors=[];files={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
    for r in board:
        name=f"{r['suite']}_{r['task_id']}"
        a=np.load(root/f'fp_{name}/actions_000.npy');b=np.load(root/f'board_{name}/actions_000.npy')
        delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
        first_errors.append({'suite':r['suite'],'first_chunk_mae':float(delta.mean()),'first_chunk_max_abs':float(delta.max()),
                             'first_chunk_gripper_sign_changes':int(np.count_nonzero((a[:,:,6]>0)!=(b[:,:,6]>0)))})
        for path in (root/f'fp_{name}/result.json',root/f'board_{name}/result.json',root/f'board_{name}/rollout.mp4'):
            files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    report={'scope':d['scope'],'pairs':d['pairs'],'successes':d['successes'],'first_chunk_errors_same_input_noise':first_errors,
            'board_action_chunks':len(latencies),'inference_p50_ms':float(np.percentile(latencies,50)),
            'inference_p95_ms':float(np.percentile(latencies,95)),
            'maxrss_kib':max(q['maxrss_kib'] for r in board for q in r['trace']),
            'source_hashes':files,'warning':'4 tasks, one initial state/seed; descriptive screening only'}
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(len(board),3,figsize=(10,3*len(board)),layout='constrained')
    names={'libero_spatial':'Bowl onto plate','libero_object':'Soup into basket','libero_goal':'Open middle drawer','libero_10':'Soup + sauce into basket'}
    for row,r in enumerate(board):
        with av.open(r['video']) as video:frames=[f.to_ndarray(format='rgb24') for f in video.decode(video=0)]
        for col,index in enumerate((0,len(frames)//2,len(frames)-1)):
            axes[row,col].imshow(frames[index]);axes[row,col].axis('off')
            axes[row,col].set_title(f"{names[r['suite']]} | step {index+1}"+(' | SUCCESS' if r['success'] else ' | FAILED'),fontsize=9)
    fig.suptitle('Actual RK3588 actions driving LIBERO; simulation pauses during inference',fontsize=11)
    output=Path('figures/smolvla_board_libero_v1');output.parent.mkdir(exist_ok=True)
    fig.savefig(output.with_suffix('.png'),dpi=140)
    output.with_suffix('.json').write_text(json.dumps({'summary_sha256':files[str(source)],'pairs':d['pairs'],'video_hashes':{k:v for k,v in files.items() if k.endswith('.mp4')}},indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},indent=2))


if __name__=='__main__':main()
