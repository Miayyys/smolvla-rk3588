"""Publish deduplicated RK3588 cost evidence and retain failures/raw reports."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from qvla.hardware.hardware_deployment_tables import deployment_costs, deployment_markdown

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "config/hardware/tables.json"
V1 = ROOT / "runs/hardware_cost_v1"
V2 = ROOT / "runs/hardware_cost_v2"
OUT = ROOT / "config/hardware"


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def load_reports(case_id):
    directory = V2 / case_id
    if not directory.exists():
        directory = V1 / case_id
    reports = []
    for round_id in range(3):
        path = directory / f"board_{round_id}.json"
        if path.exists():
            reports.append((round_id, path, read(path)))
    return directory, reports


def main():
    tables = read(TABLES)
    base_rows = []
    for case in tables["pending_cases"]:
        compile_path = V1 / case["case_id"] / "compile.json"
        compile_report = read(compile_path) if compile_path.exists() else {"status": "missing"}
        report_dir, reports = load_reports(case["case_id"])
        good = [(idx, path, report) for idx, path, report in reports
                if report.get("status") == "success" and report.get("process_returncode", 0) == 0]
        row = {
            **case,
            "status": ("measured" if len(good) == 3 else
                       "compile_failed" if compile_report.get("status") != "compiled" else
                       "board_failed_or_incomplete"),
            "rounds": len(good),
            "representative": compile_report.get("representative"),
            "model_bytes": good[0][2].get("model_bytes") if good else None,
            "p50_ms": None,
            "p95_ms": None,
            "mean_ms": None,
            "round_p50_ms": [],
            "peak_process_rss_kib": None,
            "mae_synthetic": None,
            "model_sha256": good[0][2].get("model_sha256") if good else None,
            "board_reports": [str(path.relative_to(ROOT)) for _, path, _ in reports],
            "board_report_sha256": [hashlib.sha256(path.read_bytes()).hexdigest()
                                    for _, path, _ in reports],
        }
        if good:
            times = [value for _, _, report in good for value in report["durations_ms"]]
            medians = [report["latency_ms"]["p50"] for _, _, report in good]
            row.update(
                p50_ms=float(np.percentile(times, 50)),
                p95_ms=float(np.percentile(times, 95)),
                mean_ms=statistics.mean(times),
                round_p50_ms=medians,
                peak_process_rss_kib=max(report["rusage_maxrss_kib_after"]
                                         for _, _, report in good),
                mae_synthetic=good[0][2].get("parity_vs_reference", {}).get("mae"),
            )
        row["timing_stable_20pct"] = (
            max(row["round_p50_ms"]) / min(row["round_p50_ms"]) <= 1.2
            if len(good) == 3 else None
        )
        base_rows.append(row)

    # The historical spot probes are retained in tables.json and hardware/README.md.
    # Keep the published measurements in one unified table rather than exporting
    # a second Linear-only table (95 rows) and separate historical copies.
    supplemental = read(OUT / "supplemental_costs.json") if (OUT / "supplemental_costs.json").exists() else []
    deployment = deployment_costs()
    rows = base_rows + supplemental + deployment
    write_csv(OUT / "measured_costs.csv", rows)
    linears = [row for row in base_rows if row["kind"] == "linear"]

    status_counts = {}
    for row in base_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    unstable = [row for row in base_rows if row["status"] == "measured"
                and not row["timing_stable_20pct"]]
    md = [
        "# RK3588 去重成本实测",
        "",
        "表中每行是一个去重的算子/输入形状/权重形状/格式/边界签名；不代表每个真实模块实例都单独板测。",
        "真实 SmolVLA checkpoint 权重；固定种子合成输入和两条合成校准输入只用于成本，不用于量化质量判断。Toolkit/Lite/runtime 2.3.2，core0；20 次预热、100 次计时、3 轮交错顺序。板上时延包含 Lite2 调用，不含模型加载和输入预处理。未锁频。",
        "",
        f"基础算子配置 {len(base_rows)} 项：{sum(row['status']=='measured' for row in base_rows)} 项三轮成功；另附 {len(supplemental)} 项融合、转换和CPU embedding补充测量。基础项状态分布 `{status_counts}`；{len(unstable)} 项基础配置三轮 p50 极差比超过 1.2，最终候选需复测。",
        "",
        "| 类型 | 格式 | 输入 shape | 权重 shape | 边界 | p50 ms | p95 ms | 模型 B | 轮间稳定 | 状态 |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in base_rows:
        stable = "是" if row.get("timing_stable_20pct") else "否" if row["status"] == "measured" else "—"
        md.append(f"| {row['kind']} | {row['format']} | {row['input_shape']} | {row['weight_shape']} | {row['boundary']} | {row['p50_ms']} | {row['p95_ms']} | {row['model_bytes']} | {stable} | {row['status']} |")
    md.extend([
        "", "## 融合、转换与 CPU embedding 补充测量", "",
        "融合 MLP 使用固定 held-out 激活与原始 FP32 ONNX 输出作数值检查；误差仅说明子图输出差异，不代表LIBERO任务质量。注意力核心是固定shape的参数化外代理，输入为合成Q/K/V。QKV混合精度表测整张投影图，包含精度转换和计算成本，不能把差值归因于单一转换算子。CPU embedding行不经过NPU，未计分词或CPU到NPU传输。",
        "",
        "| 单元 | 格式 | 输入 shape | 边界 | p50 ms | p95 ms | 文件/权重 B | 稳定 | 状态 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ])
    for row in supplemental:
        stable = "是" if row.get("timing_stable_20pct") else "否" if row["status"] == "measured" else "—"
        md.append(f"| {row.get('unit', row.get('kind'))} | {row['format']} | {row.get('input_shape')} | {row.get('boundary')} | {row.get('p50_ms')} | {row.get('p95_ms')} | {row.get('model_bytes', row.get('weight_bytes'))} | {stable} | {row['status']} |")
    md.extend(deployment_markdown(deployment))
    (OUT / "measured_costs.md").write_text("\n".join(md) + "\n")

    # Plot only Linear cases so unrelated Conv2D timings do not share an axis.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    complete = [row for row in linears if row["status"] == "measured"]
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for fmt in ("w8a8", "float16", "bfloat16", "w16a16i", "w16a16i_dfp"):
        data = [row for row in complete if row["format"] == fmt]
        ax.plot(range(len(data)), [row["p50_ms"] for row in data], marker="o", label=fmt)
    ax.set_yscale("log")
    ax.set_xlabel("Linear shape signature index (per format)")
    ax.set_ylabel("Board inference p50 (ms, log scale)")
    ax.set_title("RK3588 core0: real representative weights / synthetic inputs, 300 timed calls per point")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(OUT / "linear_costs.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"cost_cases": len(base_rows), "supplemental_cases": len(supplemental), "published_rows": len(rows), "linears": len(linears),
                      "status": status_counts,
                      "unstable_case_ids": [row["case_id"] for row in unstable]}, indent=2))


if __name__ == "__main__":
    main()
