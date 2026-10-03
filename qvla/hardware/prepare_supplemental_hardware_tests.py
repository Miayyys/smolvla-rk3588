#!/usr/bin/env python3
"""Stage representative fused SmolVLA subgraphs for repeatable RK3588 board timing.

Three classes are staged: fused MLPs at FP16/W8A8, a parameter-free attention
QK-softmax-PV core at FP16/W8A8, and real expert QKV projections at all-INT8,
FP16, and six intra-graph mixed precision choices. Synthetic input is used only
for timing QKV/attention; held-out captured activations and FP references are
used for MLP parity checks.
"""

from __future__ import annotations

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
import shutil
import struct
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from rknn.api import RKNN


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/hardware_supplemental_v1"
SEED = 20260929


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stage_case(case_id: str, model: Path, sample: np.ndarray,
               reference: np.ndarray | None, category: str, fmt: str,
               source_graph: Path | None = None, source_log: Path | None = None,
               precision_notes: str = "") -> dict:
    dest = OUT / case_id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, dest / "model.rknn")
    np.save(dest / "input.npy", np.ascontiguousarray(sample.astype(np.float32)))
    if reference is not None:
        np.save(dest / "reference.npy", np.asarray(reference, dtype=np.float32))
    return {
        "case_id": case_id,
        "category": category,
        "format": fmt,
        "precision_notes": precision_notes,
        "source_model": str(model.relative_to(ROOT)),
        "model_sha256": sha(dest / "model.rknn"),
        "model_bytes": (dest / "model.rknn").stat().st_size,
        "input_shape": list(sample.shape),
        "input_sha256": sha(dest / "input.npy"),
        "input_scope": "captured held-out activation" if reference is not None else "synthetic cost-only",
        "reference_sha256": sha(dest / "reference.npy") if reference is not None else None,
        "source_graph": str(source_graph.relative_to(ROOT)) if source_graph else None,
        "source_graph_sha256": sha(source_graph) if source_graph else None,
        "source_compiler_log": str(source_log.relative_to(ROOT)) if source_log else None,
        "source_compiler_log_sha256": sha(source_log) if source_log and source_log.exists() else None,
        "status": "staged",
    }


def fp_reference(graph: Path, sample: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: sample})[0]


def compile_attention_core(dest: Path) -> list[dict]:
    """Compile a parameter-free attention core as a shape-specific cost probe."""
    dest.mkdir(parents=True, exist_ok=True)
    graph_path = dest / "attention_core_fp32.onnx"
    q_idx = numpy_helper.from_array(np.asarray(0, dtype=np.int64), "q_idx")
    k_idx = numpy_helper.from_array(np.asarray(1, dtype=np.int64), "k_idx")
    v_idx = numpy_helper.from_array(np.asarray(2, dtype=np.int64), "v_idx")
    scale = numpy_helper.from_array(np.asarray(48 ** -0.5, dtype=np.float32), "scale")
    nodes = [
        helper.make_node("Gather", ["qkv", "q_idx"], ["q"], axis=0),
        helper.make_node("Gather", ["qkv", "k_idx"], ["k"], axis=0),
        helper.make_node("Gather", ["qkv", "v_idx"], ["v"], axis=0),
        helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 1, 3, 2]),
        helper.make_node("MatMul", ["q", "kt"], ["qk"]),
        helper.make_node("Mul", ["qk", "scale"], ["scaled"]),
        helper.make_node("Softmax", ["scaled"], ["prob"], axis=-1),
        helper.make_node("MatMul", ["prob", "v"], ["context"]),
    ]
    graph = helper.make_graph(
        nodes, "attention_qk_softmax_pv_cost",
        [helper.make_tensor_value_info("qkv", TensorProto.FLOAT, [3, 1, 15, 50, 48])],
        [helper.make_tensor_value_info("context", TensorProto.FLOAT, [1, 15, 50, 48])],
        [q_idx, k_idx, v_idx, scale])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, graph_path)

    rng = np.random.default_rng(SEED)
    sample = rng.standard_normal((3, 1, 15, 50, 48), dtype=np.float32)
    np.save(dest / "attention_input.npy", sample)
    calib_paths = []
    for i in range(2):
        p = dest / f"attention_cal_{i}.npy"
        np.save(p, rng.standard_normal(sample.shape, dtype=np.float32))
        calib_paths.append(str(p))
    dataset = dest / "attention_dataset.txt"
    dataset.write_text("\n".join(calib_paths) + "\n")
    reference = fp_reference(graph_path, sample)
    np.save(dest / "attention_reference.npy", reference.astype(np.float32))
    cases = []
    for fmt in ("float16", "w8a8"):
        model_path = dest / f"attention_core_{fmt}.rknn"
        rknn = RKNN(verbose=False)
        rec = {"format": fmt, "status": "compile_failed"}
        try:
            key = "quantized_dtype" if fmt == "w8a8" else "float_dtype"
            config = {key: fmt}
            assert rknn.config(target_platform="rk3588", **config) == 0
            assert rknn.load_onnx(model=str(graph_path)) == 0
            assert rknn.build(do_quantization=(fmt == "w8a8"),
                              dataset=str(dataset) if fmt == "w8a8" else None) == 0
            assert rknn.export_rknn(str(model_path)) == 0
            rec["status"] = "compiled"
            rec["model"] = model_path.name
            rec["model_bytes"] = model_path.stat().st_size
        except Exception as exc:  # retain a real compiler failure as experiment evidence
            rec["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            rknn.release()
        (dest / f"attention_{fmt}_compile.json").write_text(json.dumps(rec, indent=2) + "\n")
        if rec["status"] == "compiled":
            cases.append({
                "case_id": f"attention_core_{fmt}", "category": "attention_core_cost_proxy",
                "format": fmt,
                "precision_notes": "QK^T + scale + softmax + PV; parameter-free graph; not a full Transformer block",
                "source_model": str(model_path.relative_to(ROOT)),
                "model_sha256": sha(model_path), "model_bytes": model_path.stat().st_size,
                "input_shape": list(sample.shape), "input_sha256": sha(dest / "attention_input.npy"),
                "input_scope": "synthetic cost-only", "reference_sha256": sha(dest / "attention_reference.npy"),
                "source_graph": str(graph_path.relative_to(ROOT)), "source_graph_sha256": sha(graph_path),
                "source_compiler_log": None, "source_compiler_log_sha256": None,
                "status": "staged",
            })
            case_dir = OUT / cases[-1]["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_path, case_dir / "model.rknn")
            shutil.copy2(dest / "attention_input.npy", case_dir / "input.npy")
            shutil.copy2(dest / "attention_reference.npy", case_dir / "reference.npy")
            cases[-1]["input_sha256"] = sha(case_dir / "input.npy")
            cases[-1]["reference_sha256"] = sha(case_dir / "reference.npy")
    return cases


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    mlps = [
        ("expert", ROOT / "runs/qat_ptq_expert0_original/expert_layer0_mlp_fp32.onnx",
         ROOT / "runs/rknn_expert_mlp_heldout/task_00.npy",
         ROOT / "runs/qat_ptq_expert0_original/expert_layer0_mlp_int8_mmse_rk3588.rknn",
         None),
        ("language3", ROOT / "runs/rknn_language3_export_probe/mlp_fp32.onnx",
         ROOT / "runs/rknn_language3_dev_probe/task_00.npy",
         ROOT / "runs/rknn_language3_int8_probe/mlp_int8_mmse_rk3588.rknn",
         ROOT / "runs/rknn_language3_fp16_probe/mlp_fp16_rk3588.rknn"),
        ("vision11", ROOT / "runs/rknn_vision11_export_probe/mlp_fp32.onnx",
         ROOT / "runs/rknn_vision11_dev_probe/task_00.npy",
         ROOT / "runs/rknn_vision11_int8_probe/mlp_int8_mmse_rk3588.rknn",
         ROOT / "runs/rknn_vision11_fp16_probe/mlp_fp16_rk3588.rknn"),
    ]
    expert_fp16 = OUT / "expert_fp16_compile/expert_mlp_fp16.rknn"
    expert_fp16.parent.mkdir(parents=True, exist_ok=True)
    if not expert_fp16.exists():
        rknn = RKNN(verbose=False)
        try:
            assert rknn.config(target_platform="rk3588", float_dtype="float16") == 0
            assert rknn.load_onnx(model=str(mlps[0][1])) == 0
            assert rknn.build(do_quantization=False) == 0
            assert rknn.export_rknn(str(expert_fp16)) == 0
        finally:
            rknn.release()

    for name, graph, sample_path, int8_model, fp16_model in mlps:
        sample = np.load(sample_path, allow_pickle=False).astype(np.float32)
        ref = fp_reference(graph, sample)
        options = [("w8a8", int8_model)]
        options.append(("float16", expert_fp16 if name == "expert" else fp16_model))
        for fmt, model in options:
            if model is None or not model.is_file():
                raise FileNotFoundError(f"Missing {name}/{fmt} model: {model}")
            cases.append(stage_case(
                f"fused_mlp_{name}_{fmt}", model, sample, ref,
                "fused_mlp", fmt, source_graph=graph,
                precision_notes="Real module graph and held-out activation; graph-level timing, not sum of linear timings"))

    # QKV projection artifacts already encode six genuine RKNN intra-graph mixed
    # precision choices. The common input is seeded for cost-only comparison.
    qkv = [
        ("all_w8a8", ROOT / "runs/expert_qkv0_ptq_v1/mlp_int8_mmse_rk3588.rknn", "w8a8", None),
        ("all_fp16", ROOT / "runs/expert_qkv0_fp16_v1/mlp_fp16_rk3588.rknn", "float16", None),
        ("q_fp16", ROOT / "runs/expert_qkv0_hybrid_qfp16_v1/expert_layer0_qkv_hybrid_qfp16_rk3588.rknn", "q_fp16+other_w8a8", ROOT / "runs/expert_qkv0_hybrid_qfp16_v1/compile.log"),
        ("k_fp16", ROOT / "runs/expert_qkv0_hybrid_k_v1/expert_layer0_qkv_hybrid_kfp16_rk3588.rknn", "k_fp16+other_w8a8", ROOT / "runs/expert_qkv0_hybrid_k_v1/compile.log"),
        ("v_fp16", ROOT / "runs/expert_qkv0_hybrid_v_v1/expert_layer0_qkv_hybrid_vfp16_rk3588.rknn", "v_fp16+other_w8a8", ROOT / "runs/expert_qkv0_hybrid_v_v1/compile.log"),
        ("qk_fp16", ROOT / "runs/expert_qkv0_hybrid_qk_v1/expert_layer0_qkv_hybrid_qkfp16_rk3588.rknn", "qk_fp16+v_w8a8", ROOT / "runs/expert_qkv0_hybrid_qk_v1/compile.log"),
        ("qv_fp16", ROOT / "runs/expert_qkv0_hybrid_qv_v1/expert_layer0_qkv_hybrid_qvfp16_rk3588.rknn", "qv_fp16+k_w8a8", None),
        ("kv_fp16", ROOT / "runs/expert_qkv0_hybrid_kv_v1/expert_layer0_qkv_hybrid_kvfp16_rk3588.rknn", "kv_fp16+q_w8a8", None),
    ]
    rng = np.random.default_rng(SEED)
    qkv_input = rng.standard_normal((1, 50, 720), dtype=np.float32)
    for label, model, fmt, log in qkv:
        if not model.is_file():
            raise FileNotFoundError(model)
        cases.append(stage_case(
            f"qkv_boundary_{label}", model, qkv_input, None,
            "qkv_mixed_precision_boundary", fmt,
            source_log=log if log and log.exists() else None,
            precision_notes="Real Q/K/V projection graph; deterministic synthetic timing input; graph-level delta includes changed compute and exDataConvert"))

    cases.extend(compile_attention_core(OUT / "attention_core_source"))
    for case in cases:
        if case["source_graph"]:
            case["source_graph"] = str((ROOT / case["source_graph"]).relative_to(ROOT))
    (OUT / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    shutil.copy2(ROOT / "qvla/runtime/rknn_board_subgraph_smoke.py", OUT / "rknn_board_subgraph_smoke.py")
    shutil.copy2(ROOT / "qvla/hardware/benchmark_cost_board.py", OUT / "benchmark_cost_board.py")
    print(json.dumps({"case_count": len(cases), "categories": {
        category: sum(item["category"] == category for item in cases)
        for category in sorted({item["category"] for item in cases})},
        "attention_compile_status": [json.loads(p.read_text()) for p in sorted((OUT / "attention_core_source").glob("attention_*_compile.json"))]}, indent=2))


if __name__ == "__main__":
    main()
