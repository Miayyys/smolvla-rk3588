#!/usr/bin/env python3
"""Plot actual three-run latency and final per-dimension error of the board replay."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    source=Path('runs/smolvla_full_board_v1/full_board_report.json')
    d=json.loads(source.read_text());runs=d['runs']
    if d['status']!='success' or len(runs)!=3:raise ValueError('Expected three real successful timings')
    stages=['Vision (2 calls)','Prefix','Expert (10 calls)','CPU + other']
    matrix=[]
    for r in runs:
        a=[sum(r['vision_ms']),r['prefix_ms'],sum(r['expert_ms'])]
        matrix.append(a+[r['total_ms']-sum(a)])
    matrix=np.asarray(matrix)/1000
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    bottom=np.zeros(3)
    for i,stage in enumerate(stages):
        axes[0].bar(np.arange(3)+1,matrix[:,i],bottom=bottom,label=stage);bottom+=matrix[:,i]
    axes[0].set(xlabel='Timed replay',ylabel='Seconds / 50-action chunk',xticks=[1,2,3],ylim=(0,11),title='RK3588 core 0, FP16')
    axes[0].legend(fontsize=8,loc='upper center',ncol=2)
    for i,t in enumerate(bottom):axes[0].text(i+1,t+0.12,f'{t:.3f}s',ha='center',fontsize=9)
    axes[1].bar(range(7),d['per_dimension_action_max_abs'])
    axes[1].set(xticks=range(7),xticklabels=['x','y','z','rx','ry','rz','gripper'],ylabel='Max absolute action error vs FP',title='One fixed observation, 50 actions')
    axes[1].text(.03,.96,f"MAE = {d['action_vs_original']['mae']:.6f}\nGripper sign changes: 0 / 50",transform=axes[1].transAxes,va='top',fontsize=9)
    axes[1].set_ylim(0,0.017)
    fig.suptitle('Complete neural-network replay; input preprocessing and closed-loop tasks excluded',fontsize=10)
    out=Path('docs/images/smolvla_full_board_fp16_v1');out.parent.mkdir(exist_ok=True)
    for ext in ('png','svg'):fig.savefig(out.with_suffix('.'+ext),dpi=160)
    out.with_suffix('.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'stage_order':stages,'stage_seconds':matrix.tolist(),'action_max_abs':d['per_dimension_action_max_abs']},indent=2)+'\n')


if __name__=='__main__':main()
