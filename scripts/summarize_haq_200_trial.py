#!/usr/bin/env python3
"""Publish actual completed trial evidence, never planned quality numbers."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/haq_rl_trial_200x4_rl'


def main():
    summary = json.loads((RUN/'summary.json').read_text())
    audit = json.loads((RUN/'audit.json').read_text())
    panel = json.loads(Path(str(RUN)+'_panel/summary.json').read_text())
    if summary['rounds'] != 200 or audit['status'] != 'passed':
        raise ValueError('Complete audited trial required')
    rows = summary['results']
    if len(rows) != 800:
        raise ValueError('800 real candidate evaluations required')
    values = np.array([r['reward'] for r in rows])
    best = summary['best_by_proxy']
    held = summary['heldout_best']
    candidate = RUN/f"round{best['round']:03d}_candidate{best['candidate']:02d}"
    report = json.loads((candidate/'report.json').read_text())
    result = {'scope': summary['scope'], 'rounds': 200, 'candidate_evaluations': len(rows),
              'first80_mean_reward': float(values[:80].mean()),
              'last80_mean_reward': float(values[-80:].mean()),
              'feasible_40pct': sum(r['feasible_40pct'] for r in rows),
              'best_reward': best['reward'], 'best_round': best['round'],
              'best_candidate': best['candidate'], 'best_model_bytes': best['actual_model_bytes'],
              'best_compression_fraction': best['compression_fraction'],
              'best_search_mae': report['metrics']['task_macro']['chunk_mae_vs_fp'],
              'best_heldout_mae': held['metrics']['task_macro']['chunk_mae_vs_fp'],
              'best_table_cost_ms': report['cost']['total_ms'],
              'format_counts': best['format_counts'],
              'controller_parameter_change_l2': summary['parameter_change_l2'],
              'search_seconds': summary['elapsed_seconds'],
              'paired_panel': panel, 'audit': audit,
              'conversion_verified': False,
              'limitations': ['single_seed', 'no_equal_budget_random_control',
                              'offline_quality_proxy_not_success_rate',
                              'lookup_cost_not_measured_end_to_end_latency',
                              'RKNN_mixed_graph_conversion_not_attempted']}
    (RUN/'analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,4), constrained_layout=True)
    rounds = np.arange(1,201)
    ax.plot(rounds, values.reshape(200,4).mean(axis=1), alpha=.5, label='Round mean (4 candidates)')
    feasible_rewards = np.array([r['reward'] if r['feasible_40pct'] else -np.inf for r in rows])
    ax.plot(rounds, np.maximum.accumulate(feasible_rewards).reshape(200,4)[:,-1], label='Best size-feasible proxy reward')
    ax.set(xlabel='RL update round', ylabel='Offline proxy reward', title='SmolVLA: 200-round local diagnostic')
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(ROOT/'figures/haq-rl-200x4.png', dpi=160)
    plt.close(fig)
    doc = ROOT/'docs/experiments/2026-10-01-haq-200-round-trial.md'
    with doc.open('a') as stream:
        stream.write('\n## 已完成的实际结果\n\n')
        stream.write(f"200轮、800候选完成，{result['feasible_40pct']}/800满足本地权重文件40%压缩；逐候选评分及200次控制器更新审计通过。搜索耗时{result['search_seconds']:.2f}s。\n\n")
        stream.write('| 指标 | 实测结果 |\n| --- | ---: |\n')
        for label, key in [('前80候选平均代理奖励','first80_mean_reward'),('后80候选平均代理奖励','last80_mean_reward'),
                           ('最优代理奖励','best_reward'),('最优文件B','best_model_bytes'),
                           ('最优搜索动作MAE','best_search_mae'),('最优32条留出动作MAE','best_heldout_mae'),
                           ('最优查表成本代理ms','best_table_cost_ms'),('控制器参数变化L2','controller_parameter_change_l2')]:
            stream.write(f'| {label} | {result[key]} |\n')
        stream.write(f"\n12任务配对开发闭环：FP {panel['successes']['fp']}/12，候选 {panel['successes']['best']}/12；新增成功{panel['improved']}，新增失败{panel['worsened']}。不是最终测试或RK3588任务质量。\n\n")
        stream.write('最优格式分布：`'+json.dumps(result['format_counts'])+'`。\n\n')
        stream.write('![实测奖励曲线](../../figures/haq-rl-200x4.png)\n\n')
        stream.write('单种子且没有同预算随机对照；更多轮数不自动证明RL优于随机搜索或提升任务质量。RKNN混合精度整图未转换，转换处罚未触发；部署文件40%目标仍未验证。完整原始报告、哈希、配置和动作在`runs/haq_rl_trial_200x4_rl/`，闭环记录在相邻`_panel/`目录。\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
