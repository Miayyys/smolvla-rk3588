#!/usr/bin/env python3
"""Compare frozen candidates on a 16-task seed-1 paired development panel."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/haq_extended16_seed1_v1'
DOC=ROOT/'docs/experiments/2026-10-01-haq-extended16-evaluation.md'
ARMS={
    'old200':'haq_rl_trial_200x4_rl',
    'adjusted100':'haq_adjusted_v1_rl',
    'explore200':'haq_exploration_min200_v1_rl',
    'random200':'haq_exploration_min200_v1_random',
}
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')


def main():
    BASE.mkdir(exist_ok=False)
    started=time.time()
    status={'status':'running','pid':os.getpid(),'seed':1,'task_ids':[0,3,6,9],'steps':[]}
    def save():
        (BASE/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    fp_reference=BASE/'old200'
    frozen={}
    for arm,run in ARMS.items():
        summary=json.loads((ROOT/'runs'/run/'summary.json').read_text())
        frozen[arm]={'source_run':run,'best':summary['best_by_proxy'],
                     'identity':summary['identity']}
    (BASE/'frozen_candidates.json').write_text(json.dumps(frozen,indent=2)+'\n')
    for arm,run in ARMS.items():
        status['stage']=arm;save()
        command=[sys.executable,'scripts/run_haq_local_paired_panel.py',
                 '--run',str(ROOT/'runs'/run),'--output',str(BASE/arm),
                 '--task-ids','0','3','6','9','--seed','1']
        if arm!='old200':command+=['--fp-reference-output',str(fp_reference)]
        tick=time.time()
        done=subprocess.run(command,cwd=ROOT,check=False)
        status['steps'].append({'arm':arm,'command':command,'returncode':done.returncode,
                                'seconds':time.time()-tick})
        if done.returncode:
            status.update(status='failed',elapsed_seconds=time.time()-started);save()
            raise SystemExit(done.returncode)
    panels={a:json.loads((BASE/a/'summary.json').read_text()) for a in ARMS}
    first=panels['old200']
    outcomes={'fp':[int(p['fp']) for p in first['pairs']]}
    outcomes.update({a:[int(p['best']) for p in panels[a]['pairs']] for a in ARMS})
    for panel in panels.values():
        assert panel['seed']==1 and panel['task_ids']==[0,3,6,9]
        assert [(p['suite'],p['task_id'],p['fp']) for p in panel['pairs']]==[(p['suite'],p['task_id'],p['fp']) for p in first['pairs']]
    result={'scope':'local_16task_seed1_development_not_final_board_quality',
            'seed':1,'tasks':[{'suite':p['suite'],'task_id':p['task_id']} for p in first['pairs']],
            'outcomes':outcomes,'successes':{a:sum(v) for a,v in outcomes.items()},
            'per_suite':{s:{a:sum(v[i] for i,p in enumerate(first['pairs']) if p['suite']==s)
                              for a,v in outcomes.items()} for s in SUITES},
            'paired_changes':{a:{'improved':panels[a]['improved'],'worsened':panels[a]['worsened']} for a in ARMS},
            'candidate_file_bytes':{a:frozen[a]['best']['actual_model_bytes'] for a in ARMS},
            'candidate_compression':{a:frozen[a]['best']['compression_fraction'] for a in ARMS}}
    (BASE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (BASE/'task_outcomes.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['suite','task_id',*outcomes])
        for i,p in enumerate(first['pairs']):writer.writerow([p['suite'],p['task_id'],*[v[i] for v in outcomes.values()]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,3.8))
    ax.imshow(list(outcomes.values()),vmin=0,vmax=1,cmap='RdYlGn',aspect='auto')
    ax.set_yticks(range(len(outcomes)),[f'{a}: {sum(v)}/16' for a,v in outcomes.items()])
    ax.set_xticks(range(16),[f'{p["suite"].removeprefix("libero_")}:{p["task_id"]}' for p in first['pairs']],rotation=45,ha='right')
    ax.set_title('Paired LIBERO development: seed 1, 16 tasks (green=success)')
    for y,v in enumerate(outcomes.values()):
        for x,n in enumerate(v):ax.text(x,y,str(n),ha='center',va='center')
    fig.tight_layout();fig.savefig(BASE/'outcomes.png',dpi=160);plt.close(fig)
    with DOC.open('a') as f:
        f.write('\n## 实际结果\n\n| Suite（各4任务） | FP | 旧200 | 调整100 | 加强探索200 | 随机200 |\n| --- | ---: | ---: | ---: | ---: | ---: |\n')
        for suite,counts in result['per_suite'].items():
            f.write('| '+suite+' | '+' | '.join(str(counts[a]) for a in outcomes)+' |\n')
        f.write('| 总计/16 | '+' | '.join(str(result['successes'][a]) for a in outcomes)+' |\n')
        f.write('\n原始逐任务结果、固定候选、命令及耗时见`runs/haq_extended16_seed1_v1/`，图见`outcomes.png`。配对新增成功/丢失FP成功：'+json.dumps(result['paired_changes'],ensure_ascii=False)+'。这是新seed1的开发检查，未参与控制器更新；不是冻结测试或板端完整模型结果。\n')
    status.update(status='complete',elapsed_seconds=time.time()-started);save()
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
