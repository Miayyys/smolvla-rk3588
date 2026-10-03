#!/usr/bin/env python3
"""Summarize board evidence for fused graphs, mixed boundaries, and embedding lookup."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/hardware_supplemental_v1"
EMBED_RUN = ROOT / "runs/hardware_embedding_v1"
OUT = ROOT / "docs/hardware"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize_case(case):
    path = RUN / case["case_id"]
    reports = [path / f"board_{i}.json" for i in range(3)]
    existing = [p for p in reports if p.exists()]
    values = [json.loads(p.read_text()) for p in existing]
    good = [r for r in values if r.get("status") == "success" and r.get("process_returncode", 0) == 0]
    durations = [float(x) for report in good for x in report.get("durations_ms", [])]
    round_p50 = [float(r["latency_ms"]["p50"]) for r in good]
    status = "measured" if len(good) == 3 else "board_failed_or_incomplete"
    mae = good[0].get("parity_vs_reference", {}).get("mae") if good else None
    case_id = case["case_id"]
    if case_id.startswith("fused_mlp_expert_"):
        unit = "expert_MLP_layer0"
    elif case_id.startswith("fused_mlp_language3_"):
        unit = "language_MLP_layer3"
    elif case_id.startswith("fused_mlp_vision11_"):
        unit = "vision_MLP_layer11"
    elif case_id.startswith("qkv_boundary_"):
        unit = "expert_layer0_QKV_projection"
    elif case_id.startswith("attention_core_"):
        unit = "attention_QK_softmax_PV_proxy"
    else:
        unit = case["category"]
    return {
        "case_id": case_id, "unit": unit, "kind": case["category"],
        "format": case["format"], "input_shape": case["input_shape"], "weight_shape": "fused graph",
        "boundary": "rknn_lite2_core0_fused_graph_io" if case["category"] != "attention_core_cost_proxy" else "rknn_lite2_core0_parameter_free_attention_proxy",
        "p50_ms": float(np.percentile(durations, 50)) if durations else None,
        "p95_ms": float(np.percentile(durations, 95)) if durations else None,
        "mean_ms": statistics.fmean(durations) if durations else None,
        "round_p50_ms": round_p50,
        "timing_stable_20pct": max(round_p50) / min(round_p50) <= 1.2 if len(round_p50) == 3 else None,
        "model_bytes": case["model_bytes"], "status": status,
        "quality_diagnostic_mae": mae,
        "quality_scope": ("heldout_activation_vs_fp32_subgraph_only" if case["category"] == "fused_mlp"
                          else "synthetic_activation_numerical_sanity_only" if case["category"] == "attention_core_cost_proxy"
                          else "not_measured_cost_only"),
        "precision_notes": case["precision_notes"],
        "source_model": case["source_model"], "model_sha256": case["model_sha256"],
        "source_graph": case.get("source_graph"), "source_graph_sha256": case.get("source_graph_sha256"),
        "input_sha256": case["input_sha256"], "input_scope": case["input_scope"],
        "board_reports": [str(p.relative_to(ROOT)) for p in existing],
        "board_report_sha256": [sha(p) for p in existing],
        "warmup_per_round": 20, "timed_calls_per_round": 100, "rounds": len(good),
        "cpu_affinity": None,
        "npu_core_mask": good[0].get("npu_core_mask") if good else None,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = json.loads((RUN / "cases.json").read_text())
    rows = [summarize_case(case) for case in cases]
    embed_report_path = EMBED_RUN / "board_embedding_lookup.json"
    embed = json.loads(embed_report_path.read_text())
    prep = json.loads((EMBED_RUN / "prepare_report.json").read_text())
    emb_cases = embed["cases"]
    for name, fmt in (("native_bf16", "native_bf16_row_lookup"),
                      ("cpu_int8_row_lookup", "cpu_int8_row_lookup")):
        rec = emb_cases[name]
        rows.append({
            "case_id": f"embedding_{fmt}", "unit": "language_token_embedding_cpu_lookup",
            "kind": "cpu_embedding_lookup", "format": fmt,
            "input_shape": embed["input_shape"], "weight_shape": prep["source_shape"],
            "boundary": "cpu_numpy_lookup_and_decode_excludes_cpu_to_npu_transfer",
            "p50_ms": rec["p50_ms"], "p95_ms": rec["p95_ms"], "mean_ms": rec["mean_ms"],
            "round_p50_ms": rec["round_p50_ms"],
            "timing_stable_20pct": rec["timing_stable_20pct"],
            "model_bytes": rec["table_bytes"], "status": "measured",
            "quality_diagnostic_mae": rec["mae_vs_bf16_reference"],
            "quality_scope": "synthetic_token_ids_embedding_output_vs_native_bf16_only",
            "precision_notes": rec["operation"] + "; storage=" + rec["weight_precision"],
            "source_model": prep["source_checkpoint"], "model_sha256": embed["tensor_sha256"]["embedding_bf16.npy" if name == "native_bf16" else "embedding_int8.npy"],
            "source_graph": prep["tensor"], "source_graph_sha256": prep["source_checkpoint_sha256"],
            "input_sha256": embed["input_ids_sha256"], "input_scope": "synthetic token IDs; length matches captured downstream text MLP",
            "board_reports": [str(embed_report_path.relative_to(ROOT))],
            "board_report_sha256": [sha(embed_report_path)],
            "warmup_per_round": embed["warmup"], "timed_calls_per_round": embed["repeats_per_round"],
            "rounds": embed["rounds"], "cpu_affinity": embed["cpu_affinity"],
            "weight_bytes": rec["table_bytes"], "source_tensor_bytes": prep["source_tensor_bytes"],
            "compressed_fraction": 1.0 - rec["table_bytes"] / prep["source_tensor_bytes"],
        })
    (OUT / "supplemental_costs.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / "supplemental_costs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})
    md = [
        "# 融合、混合精度边界与 CPU embedding 补充实测", "",
        "三轮均为20次预热、100次计时；RKNN项运行于RK3588 NPU core0，CPU embedding项在RK3588 CPU上运行。所有延迟以完整`RKNNLite.inference`或完整NumPy查表调用计，CPU到NPU搬运和完整策略运行未包含。",
        "",
        "| 单元 | 格式 | 输入shape | p50 ms | p95 ms | 模型/权重 B | 轮间稳定 | 误差诊断 MAE | 误差范围 | 状态 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for row in rows:
        stable = "是" if row["timing_stable_20pct"] else "否"
        mae = f"{row['quality_diagnostic_mae']:.6f}" if row["quality_diagnostic_mae"] is not None else "—"
        md.append(f"| {row['unit']} | {row['format']} | {row['input_shape']} | {row['p50_ms']:.4f} | {row['p95_ms']:.4f} | {row['model_bytes']} | {stable} | {mae} | {row['quality_scope']} | {row['status']} |")
    md.extend([
        "", "## 结论边界", "",
        "MLP子图的INT8误差是单层输出对原始FP32 ONNX的差异，不是LIBERO任务通过率。注意力代理使用固定shape和合成Q/K/V，只用于确认QKᵀ、Softmax、PV组合在板上的成本，不是完整注意力层。QKV混合精度的p50包含Q/K/V计算与RKNN插入的格式转换；编译器日志中的`exDataConvert`周期是静态估计，不等于隔离出的实测转换时延。",
        "词嵌入采用真实checkpoint BF16行权重；INT8用逐行max-abs/127对称量化，行scale为FP32。CPU查表输入token ID为固定种子生成，长度177取自下游文本MLP样本；不含tokenizer和CPU到NPU传输。INT8行权重约节省一半表存储，但查表反量化p50高于直接BF16读取，因此只证明存储收益，不代表端到端更快。",
        "原始每次计时数据保留于`runs/hardware_supplemental_v1/<case>/board_<round>.json`和`runs/hardware_embedding_v1/board_embedding_lookup.json`；本表JSON带输入/模型/报告hash。",
    ])
    (OUT / "supplemental_costs.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"supplemental_cases": len(rows),
                      "status_counts": {s: sum(row["status"] == s for row in rows) for s in sorted({r["status"] for r in rows})},
                      "timing_unstable": [r["case_id"] for r in rows if r["status"] == "measured" and not r["timing_stable_20pct"]]}, indent=2))


if __name__ == "__main__":
    main()
