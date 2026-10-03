"""Generate source-complete module inventory and evidence-only hardware cost tables."""
import csv
import hashlib
import json
import re
import struct
from collections import defaultdict
from pathlib import Path
from hardware_deployment_tables import module_deployment, deployment_costs, deployment_markdown

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def read_published_costs(path):
    """Read the unified spreadsheet export and restore JSON/scalar types."""
    rows=[]
    with path.open(newline='') as f:
        for raw in csv.DictReader(f):
            row={}
            for key,value in raw.items():
                if value=='':
                    row[key]=None
                elif value in ('True','False'):
                    row[key]=value=='True'
                elif value.startswith(('[','{')):
                    row[key]=json.loads(value)
                elif re.fullmatch(r'-?\d+',value):
                    row[key]=int(value)
                elif re.fullmatch(r'-?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?',value):
                    row[key]=float(value)
                else:
                    row[key]=value
            rows.append(row)
    return rows
def main():
    out=ROOT/'docs/hardware';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'artifacts/transfer/model/model.safetensors'
    cfg=read('config/quantization_map_v0.json')
    shape_completion=read('config/hardware_shape_completion.json')
    if shape_completion['checkpoint_weight_sha256']!=cfg['checkpoint_weight_sha256']:
        raise ValueError('Shape completion manifest belongs to another checkpoint')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=cfg['checkpoint_weight_sha256']:raise ValueError('Source hash mismatch')
    with source.open('rb') as f:header=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
    cal=read('runs/mixed_int8_v1/calibration.json');pack=read('runs/mixed_int8_v1/report.json')
    manifests={r['name']:r for r in pack['manifest']}
    modules=defaultdict(list)
    for key,t in header.items():
        if key!='__metadata__':modules[key.rsplit('.',1)[0]].append((key,t))
    rows=[];cases={}
    for name,items in sorted(modules.items()):
        groups={g['id'] for key,t in items for g in cfg['groups'] if re.search(g['pattern'],key)}
        if len(groups)!=1:raise ValueError(f'Group coverage {name}: {groups}')
        group=next(iter(groups));weight=next((t for k,t in items if k.endswith('.weight')),None)
        shape=weight['shape'] if weight else []
        is_embedding=('embedding' in name or name.endswith('.embed_tokens') or group=='language_embedding')
        kind=('embedding' if is_embedding and len(shape)==2 else 'linear' if len(shape)==2 else 'conv2d' if len(shape)==4 else 'norm_or_parameter')
        if group=='lm_head':kind='inactive_linear'
        stats=cal['ranges'].get(name,{})
        completed=shape_completion['shapes'].get(name)
        input_shapes=stats.get('shapes',[])
        shape_evidence='runs/mixed_int8_v1/calibration.json' if input_shapes else None
        if not input_shapes and completed:
            input_shapes=[completed['input_shape']]
            shape_evidence=completed['source']+': '+completed['evidence']
        options=(['w8a8','float16','bfloat16','w16a16i','w16a16i_dfp'] if kind in ('linear','conv2d') else ['native_bf16_row_lookup','cpu_int8_row_lookup'] if group=='language_embedding' else ['native_float'])
        case_ids=[]
        for inp in input_shapes:
            for fmt in options:
                signature={'kind':kind,'input_shape':inp,'weight_shape':shape,'has_bias':any(k.endswith('.bias') for k,t in items),'format':fmt,'backend':'rknn_2.3.2','boundary':'standalone_float32_io','core':'core0'}
                cid=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
                if cid not in cases:cases[cid]={'case_id':cid,**signature,'modules':[],'status':'pending_compile_and_board','p50_ms':None,'p95_ms':None,'rounds':None,'model_bytes':None,'timing_stable_20pct':None}
                cases[cid]['modules'].append(name);case_ids.append(cid)
        rows.append({'module':name,'group':group,'kind':kind,'weight_shape':shape,
                     'source_dtypes':sorted({t['dtype'] for k,t in items}),
                     'source_bytes':sum(t['data_offsets'][1]-t['data_offsets'][0] for k,t in items),
                     'tensor_count':len(items),'current_precision':('w8a8' if manifests.get(name,{}).get('kind')=='linear' else 'int8_row_weight_float_output' if name in manifests else 'native'),
                     'runtime_weight_dtype':manifests.get(name,{}).get('compute_dtype','unmeasured'),
                     'input_shapes':input_shapes,'observed_input_dtypes':stats.get('input_dtypes',[]),
                     'shape_evidence':shape_evidence,
                     'calls_in_40_calibration_actions':stats.get('calls'),
                     'candidate_formats':options,'support_status':'cost_status_pending',
                     'format_cost_status':{},
                     'hardware_case_ids':case_ids,
                     **module_deployment(name,group)})
    assert sum(r['tensor_count'] for r in rows)==500
    assert sum(r['source_bytes'] for r in rows)==906639456
    measured=[]
    sources=[(f'runs/precision_support/board/precision_probe/{fmt}_board.json','synthetic_matmul',fmt,'not_matched_to_smolvla') for fmt in ('float16','bfloat16','w8a8','w16a16i','w16a16i_dfp')]
    sources += [(f'runs/board_validation_v1/{mode}_board.json','expert_mlp_layer0',mode+'_w8a8','whole_mlp_not_individual_linear') for mode in ('ptq','qat')]
    for path,unit,fmt,scope in sources:
        r=read(path)
        if r['status']!='success':raise ValueError(path)
        measured.append({'unit':unit,'format':fmt,'input_shape':r['input_shape'],'model_bytes':r['model_bytes'],
                         'p50_ms':r['latency_ms']['p50'],'p95_ms':r['latency_ms']['p95'],
                         'process_peak_rss_kib':r['rusage_maxrss_kib_after'],
                         'warmup':r['warmup_runs'],'repeats':r['timed_runs'],
                         'core_mask':r.get('npu_core_mask','default_not_recorded'),
                         'scope':scope,'source':path,'model_sha256':r['model_sha256'],
                         'report_sha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest()})
    case_rows=list(cases.values())
    supplemental_path=out/'supplemental_costs.json'
    supplemental=json.loads(supplemental_path.read_text()) if supplemental_path.exists() else []
    supplemental_by_id={r['case_id']:r for r in supplemental}
    measured_cases_path=out/'measured_costs.csv'
    measured_complete=read_published_costs(measured_cases_path) if measured_cases_path.exists() else []
    deployment=deployment_costs()
    measured_complete=[r for r in measured_complete if r.get('kind')!='deployment_stage']+deployment
    write_csv(measured_cases_path,measured_complete)
    if measured_complete:
        measured_by_id={r['case_id']:r for r in measured_complete}
        for case in case_rows:
            result=measured_by_id.get(case['case_id'])
            if result:
                case.update(status=result['status'],p50_ms=result['p50_ms'],p95_ms=result['p95_ms'],
                            rounds=result['rounds'],model_bytes=result['model_bytes'],
                            timing_stable_20pct=result['timing_stable_20pct'],
                            peak_process_rss_kib=result.get('peak_process_rss_kib'),
                            mae_synthetic=result.get('mae_synthetic'))
    cost_by_id={case['case_id']:case for case in case_rows}
    for row in rows:
        if row['group']=='lm_head':
            row['support_status']='inactive_lm_head'
            continue
        if row['group']=='language_embedding':
            embed_ids=['embedding_native_bf16_row_lookup','embedding_cpu_int8_row_lookup']
            row['hardware_case_ids']=embed_ids
            statuses={}
            for candidate,cid in zip(row['candidate_formats'],embed_ids):
                result=supplemental_by_id.get(cid)
                statuses[candidate]=('representative_signature_board_measured'
                                     if result and result['status']=='measured'
                                     else result['status'] if result else 'not_measured')
            row['format_cost_status']=statuses
            row['support_status']=('representative_signatures_board_measured'
                                   if statuses and all(v=='representative_signature_board_measured'
                                                       for v in statuses.values())
                                   else 'some_candidate_signatures_pending_or_failed')
            continue
        if row['kind']=='norm_or_parameter' or row['candidate_formats']==['native_float']:
            row['support_status']='no_multiple_precision_candidates_in_current_inventory; graph_feasibility_unresolved'
            row['format_cost_status']={fmt:'not_measured_as_independent_candidate' for fmt in row['candidate_formats']}
            continue
        statuses={}
        for fmt in row['candidate_formats']:
            fmt_cases=[cost_by_id[cid] for cid in row['hardware_case_ids']
                       if cost_by_id[cid]['format']==fmt]
            if not fmt_cases:
                statuses[fmt]='not_measured'
            elif all(case['status']=='measured' for case in fmt_cases):
                statuses[fmt]='representative_signature_board_measured'
            else:
                statuses[fmt]=','.join(sorted({case['status'] for case in fmt_cases}))
        row['format_cost_status']=statuses
        row['support_status']=('representative_signatures_board_measured'
                               if statuses and all(v=='representative_signature_board_measured'
                                                   for v in statuses.values())
                               else 'some_candidate_signatures_pending_or_failed')
    write_csv(out/'module_options.csv',rows)
    write_csv(out/'cost_cases.csv',case_rows)
    payload={'source_sha256':digest,'modules':rows,'cost_cases':case_rows,
             'pending_cases':case_rows,'measured_costs':measured_complete,
             'supplemental_costs':supplemental,'historical_costs':measured,
             'deployment_costs':deployment,
             'deployment_contract':{'backend':'RKNN_2.3.2+CPU; RKLLM_not_used',
                 'calls_per_action_chunk':{'vision_connector':2,'prefix_with_kv':1,'expert_step_v2':10},
                 'candidate_scope':'standalone_signature_support; full_graph_mixed_precision_unverified',
                 'formal_search_ready':False}}
    (out/'tables.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    groups=defaultdict(list)
    for r in rows:groups[r['group']].append(r)
    md=['# SmolVLA 模块候选配置与 RK3588 成本表','',
        '由 `scripts/build_hardware_tables.py` 根据固定 checkpoint、40 条校准调用记录及板端 JSON 自动生成。',
        '**候选格式不等于该模块已验证支持，也不是最终HAQ动作空间。未测量值为空，不能视为零。** `current_precision` 描述既有PTQ候选，不代表本轮FP基线或HAQ已决定的精度。','',
        '`deployment_stage`/`deployment_backend`/`baseline_deployment_format`描述当前已运行的RKNN＋CPU基线；`stage_calls_per_action_chunk`是阶段调用次数。`full_graph_format_status`与`format_cost_status`分别表示整图执行证据与独立签名测量，不能互换。独立精度控制边界仍待导出图核查；没有据人工敏感度固定搜索位点。','',
        f'原始checkpoint覆盖：{len(rows)} 个按参数路径分组的模块、500 个张量、906,639,456 B。既有PTQ候选曾量化291个Linear并使用1个CPU行量化embedding；它不是本轮HAQ固定精度图。',
        f'输入调用记录覆盖 {len(cal["ranges"])} 个Linear；{sum(r["kind"] == "linear" for r in rows)} 个活动Linear中 {sum(r["kind"] == "linear" and bool(r["input_shapes"]) for r in rows)} 个已有实测或来源可追溯的shape，{sum(r["kind"] == "linear" and not r["input_shapes"] for r in rows)} 个仍无shape；另有1个不活跃lm_head。',
        f'按算子、shape、格式和边界去重得到 {len(case_rows)} 个基础成本配置，其中 {sum(c["status"] == "measured" for c in case_rows)} 项完成三轮板测；另有 {len(supplemental)} 项融合、注意力、精度边界和CPU embedding补充测量。',
        '模块数仅指有持久化参数的模块。缓存按项目范围不测；当前清单没有为Norm/位置参数登记多精度候选，这不证明它们只能原精度；需由完整执行图和后端探针核实是否可独立配置。','',
        '[完整模块配置表](module_options.csv) · [去重成本配置/状态](cost_cases.csv) · [基础及补充实测表](measured_costs.md) · [实测CSV](measured_costs.csv) · [融合/转换明细](supplemental_costs.md) · [机器可读数据](tables.json) · [Linear延迟图](linear_costs.png)','',
        '| 分组 | 参数模块数 | 原权重 MB | 有输入记录 |','| --- | ---: | ---: | ---: |']
    for g,items in groups.items():md.append(f'| {g} | {len(items)} | {sum(r["source_bytes"] for r in items)/1e6:.3f} | {sum(bool(r["input_shapes"]) for r in items)} |')
    md+=['','## 基础可调算子成本','',
         '95个Linear配置覆盖19个shape签名×5种格式；5个patch Conv2D配置覆盖1个shape×5种格式。每项20次预热、100次计时、3轮；使用真实checkpoint代表权重和固定种子合成输入/校准数据。独立FP32 I/O子图包含Lite2调用开销，不代表完整模型延迟或量化质量。','',
         '表中的格式按每个Linear/Conv shape的可编译候选列出；稳定性阈值为三轮p50最大/最小≤1.2。板子未锁频，超过阈值项需在HAQ最终候选阶段复测。','',
         '## 其他历史板端实测','', '| 单元 | 格式 | p50 ms | p95 ms | 文件 B |','| --- | --- | ---: | ---: | ---: |']
    for r in measured:md.append(f'| {r["unit"]} | {r["format"]} | {r["p50_ms"]:.4f} | {r["p95_ms"]:.4f} | {r["model_bytes"]} |')
    md+=['','以上历史测试未统一锁频/核心配置，不作直接性能排名；合成小图不代替真实模块测量。RSS是含Python/runtime的进程峰值。',
         '', '## 补充子图与边界实测','',
         f'{len(supplemental)}项额外测试见[融合/转换明细](supplemental_costs.md)：真实专家/语言/视觉MLP各测FP16与W8A8；QKᵀ→Softmax→PV形状代理测FP16/W8A8；真实专家QKV图测全INT8、全FP16及6种混合精度；真实BF16 token embedding测CPU行INT8查表。',
         '融合子图数据用来校正“逐层相加”的估算；QKV记录的是含内部格式转换的整图延迟，不能解释为纯转换算子单独耗时。embedding微基准只含CPU查表/解量化，未包含tokenizer和CPU到NPU传输。','',
         '## 覆盖边界','',
         '- 11个原先缺失输入shape的活动Linear已由捕获shape或可追溯的同结构输入签名补齐；不活跃的lm_head不进入当前动作路径。',
         '- 缓存独立微基准按用户要求排除；实际完整流程仍包含必要的张量传递。当前清单未为Norm、残差和位置参数建立独立精度候选，暂列待执行图核实；完整FP16策略已在RKNN＋CPU执行，任意混合精度整图尚未验证。',
         '- W4A16不属于目前确认可执行的RK3588候选；W4A4未通过，未放入格式候选。质量选择仍需回到固定LIBERO任务评估，层输出误差只是诊断项。',
         '- 延迟估计须结合真实调用次数；p95与峰值RSS不能逐层直接求和。']
    md+=deployment_markdown(deployment)
    (out/'README.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'parameter_modules':len(rows),'observed_linears':len(cal['ranges']),'unique_cost_cases':len(cases),'historical_rows':len(measured)}))
if __name__=='__main__':main()
