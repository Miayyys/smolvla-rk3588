"""Describe the deployed graph and publish existing board timing evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module_deployment(name, group):
    if group == 'lm_head':
        stage, backend, calls, precision = 'inactive', 'not_executed', 0, 'not_executed'
    elif name == 'model.state_proj':
        stage, backend, calls, precision = 'prefix_glue', 'cpu_numpy', 1, 'float32'
    elif group == 'language_embedding':
        stage, backend, calls, precision = 'prefix_glue', 'cpu_numpy', 1, 'bf16_values_stored_as_float32'
    elif group.startswith('vision') or group == 'connector':
        stage, backend, calls, precision = 'vision_connector', 'rknn_2.3.2', 2, 'float16_build'
    elif group.startswith('language'):
        stage, backend, calls, precision = 'prefix_with_kv', 'rknn_2.3.2', 1, 'float16_build'
    elif group.startswith('expert') or group in ('action_interface', 'action_time_mlp'):
        stage, backend, calls, precision = 'expert_step_v2', 'rknn_2.3.2', 10, 'float16_build'
    else:
        raise ValueError('Unmapped deployment module: ' + name)
    return {'deployment_stage': stage, 'deployment_backend': backend,
            'stage_calls_per_action_chunk': calls, 'baseline_deployment_format': precision,
            'call_count_scope': 'stage invocation count; not an independent per-operator trace',
            'baseline_graph_status': 'inactive' if calls == 0 else 'executed_on_board',
            'independent_precision_control_status': 'inactive' if calls == 0 else 'pending_export_graph_mapping',
            'full_graph_format_status': {'float16_build': 'board_executed; quality_screening_only'}
                 if backend.startswith('rknn') else {},
            'deployment_evidence': 'scripts/smolvla_board_runtime.py; docs/experiments/2026-10-01-board-libero-closed-loop.md'}


def deployment_costs():
    source = 'runs/smolvla_raw_board_v1/raw_full_board_report.json'
    path = ROOT/source
    raw = json.loads(path.read_text())
    if raw['status'] != 'success' or raw['call_counts_per_replay'] != {'vision': 2, 'prefix': 1, 'expert': 10}:
        raise ValueError('Deployment report contract mismatch')
    rows = []
    def row(unit, times, backend, calls, boundary, model_bytes=None, model_sha=None,
            source_paths=None, rss=None, quality=None, scope='single_fixed_development_observation'):
        sources = source_paths or [source]
        rows.append({'case_id': 'deployment_v1_' + unit,
                     'kind': 'deployment_stage', 'unit': unit,
                     'format': 'fp16_graphs_cpu_float32' if 'rknn' in backend else 'cpu_float32',
                     'backend': backend, 'boundary': boundary, 'core': 'core0' if 'rknn' in backend else 'cpu_not_pinned',
                     'modules': [], 'status': 'measured', 'p50_ms': float(np.percentile(times, 50)),
                     'p95_ms': float(np.percentile(times, 95)), 'mean_ms': float(np.mean(times)),
                     'rounds': 1, 'timed_calls_per_round': len(times),
                     'warmup_per_round': raw['warmup'] if sources == [source] else None,
                     'calls_per_action_chunk': calls, 'model_bytes': model_bytes,
                     'model_sha256': model_sha, 'peak_process_rss_kib': rss,
                     'timing_stable_20pct': None, 'board_reports': sources,
                     'board_report_sha256': [hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources],
                     'quality_diagnostic_mae': quality, 'quality_scope': scope,
                     'cost_use': 'deployment_reference_only; overlapping_rows_do_not_sum',
                     'protocol': 'existing board evidence; not the 3-round synthetic signature protocol'})
    stages = [
        ('vision_connector', 'vision_ms', 'rknn_2.3.2', 2, 212621173, raw['hashes']['vision']),
        ('prefix_with_kv', 'prefix_ms', 'rknn_2.3.2', 1, 326154086, raw['hashes']['prefix']),
        ('expert_step_v2', 'expert_ms', 'rknn_2.3.2', 10, 217425832, raw['hashes']['expert']),
        ('raw_preprocess', 'preprocess_ms', 'cpu_numpy_tokenizers', 1, None, None),
        ('prefix_glue', 'prefix_glue_ms', 'cpu_numpy', 1, None, None),
        ('time_embedding', 'time_embedding_ms', 'cpu_numpy', 10, None, None),
        ('euler_integration', 'euler_ms', 'cpu_numpy', 10, None, None),
        ('action_postprocess', 'postprocess_ms', 'cpu_numpy', 1, None, None),
    ]
    for unit, key, backend, calls, size, digest in stages:
        values = [v for r in raw['runs'] for v in (r[key] if isinstance(r[key], list) else [r[key]])]
        row(unit, values, backend, calls, 'per_call_in_full_raw_replay; host_IO_included', size, digest)
    row('full_raw_policy', [r['total_ms'] for r in raw['runs']], 'rknn_2.3.2+cpu', 1,
        'raw_observation_to_actions; excludes_model_load_and_file_IO',
        raw['loaded_graph_bytes']+raw['cpu_parameter_file_bytes'], rss=raw['rusage_maxrss_kib'],
        quality=raw['action_vs_original']['mae'])
    sources = [f'runs/smolvla_board_libero_v1/board_{suite}_0/result.json'
               for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')]
    traces = [t for p in sources for t in json.loads((ROOT/p).read_text())['trace']]
    if len(traces) != 22 or any(t['calls'] != raw['call_counts_per_replay'] for t in traces):
        raise ValueError('Closed-loop graph invocation mismatch')
    row('full_policy_closed_loop', [t['inference_ms'] for t in traces], 'rknn_2.3.2+cpu', 1,
        'board_raw_observation_to_actions; excludes_SSH_and_simulation',
        raw['loaded_graph_bytes']+raw['cpu_parameter_file_bytes'], source_paths=sources,
        rss=max(t['maxrss_kib'] for t in traces), scope='4_tasks_one_seed; FP_and_board_2/4; not_real_time_validation')
    return rows


def deployment_markdown(rows):
    md = ['', '## 当前完整部署：RKNN＋CPU', '',
          '以下从既有真实板端记录生成：单条开发观测完整回放预热1次、计时3次；闭环行来自4任务22次请求。三张RKNN图均为FP16构建，CPU保留浮点处理。不是最终HAQ/QAT/PTQ结果。', '',
          '| 单元 | 后端 | 每动作块调用次数 | p50 ms/调用 | p95 ms/调用 | 测量次数 | 文件 B |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        md.append(f"| {r['unit']} | {r['backend']} | {r['calls_per_action_chunk']} | {r['p50_ms']:.4f} | {r['p95_ms']:.4f} | {r['timed_calls_per_round']} | {r['model_bytes'] if r['model_bytes'] is not None else '—'} |")
    md += ['', '视觉共享同一模型文件、每动作块运行两次；前缀一次；专家十次。表内完整流程行已经包含子阶段，不能再次相加；单阶段p50之和不等于完整流程p50，p95/RSS也不能求和。',
           '三张图共756,201,091 B，CPU参数文件189,363,274 B，合计945,564,365 B，未达到原checkpoint至少40%压缩要求；大小未包含tokenizer/配置等资产。CPU参数不按重复调用复制计费。',
           '阶段调用次数是执行图次数，不保证各参数算子都逐次执行；独立精度控制边界仍待导出图核查。Norm/融合节点不凭缺少独立测量固定精度。',
           'Lite2图调用时间含输入准备、运行及输出交接；纯CPU↔NPU拷贝/格式转换没有独立计时，不从总耗时相减推算。CPU分项为实测合并阶段，未把分词/lookup/状态投影拆出。',
           '混合精度整图配置成本仍未测量；现有100个基础签名和18项补充可作代理查表，不能把FP16流程参考行作为任意配置的已测收益。',
           '[完整部署证据](../experiments/2026-10-01-board-libero-closed-loop.md) · [表格更新记录](../experiments/2026-10-01-deployment-hardware-tables.md)', '']
    return md
