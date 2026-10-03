#!/usr/bin/env python3
"""Local adjusted RL with plateau stop and matched-budget random control."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/haq_adjusted_v1'
DOC=ROOT/'docs/project-route.md'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix',type=Path,default=BASE)
    parser.add_argument('--doc',type=Path,default=DOC)
    parser.add_argument('--min-rounds',type=int,default=100)
    parser.add_argument('--max-rounds',type=int,default=500)
    parser.add_argument('--entropy-weight',type=float,default=0.0001)
    parser.add_argument('--entropy-final',type=float,default=0.00001)
    args=parser.parse_args()
    if not 1<=args.min_rounds<=args.max_rounds:
        parser.error('Require 1 <= min-rounds <= max-rounds')
    if args.entropy_weight<0 or args.entropy_final<0:
        parser.error('Entropy weights must be nonnegative')
    base=args.output_prefix.resolve()
    doc=args.doc.resolve()
    base.parent.mkdir(parents=True,exist_ok=True)
    if any(Path(str(base)+suffix).exists() for suffix in ('_status.json','_rl','_random')):
        parser.error('Output prefix already used; select a fresh experiment prefix')
    if not doc.is_file():
        parser.error('Write experiment protocol before starting the trial')
    started=time.time()
    status={'status':'running','pid':os.getpid(),'started_unix':started,'steps':[],
            'config':{'minimum_rounds':args.min_rounds,'maximum_rounds':args.max_rounds,
                      'entropy_weight':args.entropy_weight,'entropy_final':args.entropy_final}}
    status_file=Path(str(base)+'_status.json')
    env=dict(os.environ,MPLCONFIGDIR='/tmp/qvla-mpl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    def run(stage,args):
        status['stage']=stage;status_file.write_text(json.dumps(status,indent=2)+'\n')
        print('Starting '+stage,flush=True)
        tick=time.time()
        result=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,check=False)
        status['steps'].append({'stage':stage,'command':args,'returncode':result.returncode,'seconds':time.time()-tick})
        if result.returncode:
            status.update(status='failed',elapsed_seconds=time.time()-started)
            status_file.write_text(json.dumps(status,indent=2)+'\n');raise SystemExit(result.returncode)
    common=['--batch-size','4','--seed','29','--int8-logit-prior','4',
            '--entropy-weight',str(args.entropy_weight),'--entropy-final',str(args.entropy_final),'--quality-mode','trajectory_gripper',
            '--score-task-indices','0','2','5','7','10','12','15','17','20','22','25','27','30','32','35','37',
            '--validation-task-indices','1','6','11','16','21','26','31','36','--checkpoint-retention','best']
    rl=Path(str(base)+'_rl');rnd=Path(str(base)+'_random')
    run('rl_search',['qvla/haq/run_haq_local_loop.py','--output',str(rl),'--rounds',str(args.max_rounds),
                     '--early-stop','--min-rounds',str(args.min_rounds),'--patience','60','--check-every','20',
                     '--improvement-threshold','0.0001',*common])
    summary=json.loads((rl/'summary.json').read_text())
    run('rl_audit',['qvla/haq/audit_haq_local_loop.py',str(rl)])
    run('random_search',['qvla/haq/run_haq_local_loop.py','--output',str(rnd),
                         '--rounds',str(summary['rounds']),'--random-control',*common])
    run('random_audit',['qvla/haq/audit_haq_local_loop.py',str(rnd)])
    if summary['best_by_proxy'] is not None and json.loads((rnd/'summary.json').read_text())['best_by_proxy'] is not None:
        run('comparison',['qvla/haq/compare_haq_rl_random.py','--rl',str(rl),'--random',str(rnd),
                          '--output',str(base)+'_comparison'])
    panels={}
    for arm,path in [('rl',rl),('random',rnd)]:
        s=json.loads((path/'summary.json').read_text())
        if s['best_by_proxy'] is None:continue
        run(arm+'_paired_panel',['qvla/haq/run_haq_local_paired_panel.py','--run',str(path),
                                 '--output',str(path)+'_panel','--fp-reference-output','runs/haq_libero_panel_v1'])
        panels[arm]=json.loads(Path(str(path)+'_panel/summary.json').read_text())
    result={'scope':'single_seed_local_proxy_RL_vs_equal_budget_random; not_RKNN_deployment',
            'actual_rounds':summary['rounds'],'stop':summary['stop_rule'],'arms':{}}
    for arm,path in [('rl',rl),('random',rnd)]:
        s=json.loads((path/'summary.json').read_text());best=s['best_by_proxy'];rows=s['results']
        held=s['heldout_best']
        result['arms'][arm]={'candidates':len(rows),'size_feasible':sum(x['feasible_40pct'] for x in rows),
                             'best_reward':best['reward'] if best else None,
                             'best_model_bytes':best['actual_model_bytes'] if best else None,
                             'best_compression':best['compression_fraction'] if best else None,
                             'first80_mean_reward':sum(x['reward'] for x in rows[:80])/len(rows[:80]),
                             'last80_mean_reward':sum(x['reward'] for x in rows[-80:])/len(rows[-80:]),
                             'final_holdout_observations':held['observations'] if held else 0,
                             'final_holdout_chunk_mae':held['metrics']['task_macro']['chunk_mae_vs_fp'] if held else None,
                             'controller_parameter_change_l2':s['parameter_change_l2'],
                             'elapsed_seconds':s['elapsed_seconds'],'paired_panel':panels.get(arm)}
    Path(str(base)+'_analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    with doc.open('a') as f:
        f.write('\n## 已完成的实际结果\n\n')
        f.write(f"RL实际运行{result['actual_rounds']}轮，停止条件：`{result['stop']}`；随机组评价相同候选数量。\n\n")
        f.write('| 指标 | RL | 同预算随机 |\n| --- | ---: | ---: |\n')
        for label,key in [('候选数量','candidates'),('满足40%体积条件','size_feasible'),('最优代理奖励','best_reward'),
                          ('最优本地文件B','best_model_bytes'),('最终16条留出MAE','final_holdout_chunk_mae'),
                          ('前80平均奖励','first80_mean_reward'),('后80平均奖励','last80_mean_reward')]:
            f.write(f"| {label} | {result['arms']['rl'][key]} | {result['arms']['random'][key]} |\n")
        for arm,p in panels.items():
            f.write(f"\n{arm}的12任务单种子配对面板：FP {p['successes']['fp']}/12，候选 {p['successes']['best']}/12，新增{p['improved']}成功/{p['worsened']}失败。\n")
        f.write('\n本轮与历史200轮的搜索面板和质量公式不同，代理奖励不能直接比较；任务通过数也仅是开发筛查。所有后端转换状态仍未验证，不能宣称板端收益。结果不满足最终质量门槛时应保留失败结论。\n')
    status.update(status='complete',elapsed_seconds=time.time()-started)
    status_file.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
