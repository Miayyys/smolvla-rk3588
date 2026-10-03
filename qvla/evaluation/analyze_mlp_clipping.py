#!/usr/bin/env python3
"""Compare independent symmetric INT8 clipping rules on captured MLP inputs.

The KL curve is an entropy-style diagnostic computed here; it does not expose
or reproduce RKNN Toolkit2's internal clipping thresholds.
"""

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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BINS = 2048
POSITIVE_LEVELS = 127
EPS = 1e-12


def read_activations(folder: Path, expected_split: str):
    report = json.loads((folder / "report.json").read_text())
    if report["split"] != expected_split:
        raise ValueError(f"Expected {expected_split}, got {report['split']}")
    arrays = [np.load(folder / row["file"], allow_pickle=False).astype(np.float32)
              for row in report["records"]]
    if not arrays or len({array.shape for array in arrays}) != 1:
        raise ValueError("Missing activations or mismatched shapes")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Non-finite activations")
    return report, np.concatenate([array.ravel() for array in arrays])


def kl_at_bin(hist: np.ndarray, cutoff_bin: int) -> float:
    """Entropy calibration proxy: clip tail, merge into 128 bins, expand back."""
    p = hist[:cutoff_bin].astype(np.float64).copy()
    p[-1] += hist[cutoff_bin:].sum()
    p /= p.sum()
    q = np.zeros_like(p)
    boundaries = np.linspace(0, cutoff_bin, POSITIVE_LEVELS + 1, dtype=int)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        region = p[left:right]
        occupied = region > 0
        if occupied.any():
            expanded = q[left:right]
            expanded[occupied] = region.sum() / occupied.sum()
    p = (p + EPS) / (p + EPS).sum()
    q = (q + EPS) / (q + EPS).sum()
    return float(np.sum(p * np.log(p / q)))


def quantization_metrics(values: np.ndarray, threshold: float) -> dict:
    scale = threshold / POSITIVE_LEVELS
    q = np.clip(np.rint(values / scale), -POSITIVE_LEVELS,
                POSITIVE_LEVELS).astype(np.int8)
    error = q.astype(np.float32) * scale - values
    return {"scale": float(scale), "zero_point": 0,
            "mae": float(np.mean(np.abs(error))),
            "mse": float(np.mean(error.astype(np.float64) ** 2)),
            "saturation_fraction": float(np.mean(np.abs(values) > threshold))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cal_report, calibration = read_activations(args.calibration_dir, "ptq_calibration")
    test_report, heldout = read_activations(args.heldout_dir, "test")
    cal_episodes = {row["episode_index"] for row in cal_report["records"]}
    test_episodes = {row["episode_index"] for row in test_report["records"]}
    if cal_episodes & test_episodes:
        raise ValueError("Calibration and held-out episodes overlap")
    if cal_report["source_weight_sha256"] != test_report["source_weight_sha256"]:
        raise ValueError("Different source models")

    max_abs = float(np.abs(calibration).max())
    hist, edges = np.histogram(np.abs(calibration), bins=BINS, range=(0, max_abs))
    centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float64)
    candidate_bins = sorted(set(range(POSITIVE_LEVELS + 1, BINS + 1, 16)) | {BINS})
    candidates = []
    for cutoff_bin in candidate_bins:
        threshold = float(edges[cutoff_bin])
        scale = threshold / POSITIVE_LEVELS
        reconstructed = np.rint(np.minimum(centers, threshold) / scale) * scale
        mse_estimate = float(np.dot(hist, (centers - reconstructed) ** 2) /
                             hist.sum())
        candidates.append({"cutoff_bin": cutoff_bin, "threshold": threshold,
                           "kl_divergence": kl_at_bin(hist, cutoff_bin),
                           "histogram_mse_estimate": mse_estimate,
                           "histogram_saturation_fraction":
                           float(hist[cutoff_bin:].sum() / hist.sum())})

    selected = {"minmax": candidates[-1],
                "mse": min(candidates, key=lambda row: row["histogram_mse_estimate"]),
                "kl": min(candidates, key=lambda row: row["kl_divergence"])}
    metrics = {name: {"candidate": candidate,
                      "calibration": quantization_metrics(calibration, candidate["threshold"]),
                      "heldout": quantization_metrics(heldout, candidate["threshold"])}
               for name, candidate in selected.items()}
    report = {"scope": "MLP input activation quantizer diagnostic only",
              "rknn_internal_thresholds": "not reproduced here; RKNN step1 emits its own activation parameters",
              "formula": "symmetric int8: scale=alpha/127, q=clip(round(x/scale),-127,127), xhat=q*scale",
              "kl_definition": "tail mass moved to last retained histogram bin; retained bins merged to 127 positive bins, mass uniformly expanded over occupied original bins; epsilon=1e-12",
              "histogram_bins": BINS, "positive_quantized_levels": POSITIVE_LEVELS,
              "candidate_step_bins": 16, "calibration_samples": cal_report["samples"],
              "heldout_samples": test_report["samples"],
              "activation_shape": list(np.load(args.calibration_dir / cal_report["records"][0]["file"]).shape),
              "calibration_max_abs": max_abs,
              "heldout_max_abs": float(np.abs(heldout).max()),
              "source_weight_sha256": cal_report["source_weight_sha256"],
              "split_sha256": cal_report["split_sha256"],
              "candidates": candidates, "selected": metrics}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "clipping_scan.json").write_text(json.dumps(report, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.2), sharex=True)
    x = [row["threshold"] for row in candidates]
    specs = [("kl_divergence", "KL divergence ↓", "#4169a3"),
             ("histogram_mse_estimate", "Estimated activation MSE ↓", "#14866d"),
             ("histogram_saturation_fraction", "Saturated fraction ↓", "#be7150")]
    marker_colors = {"minmax": "#596574", "mse": "#14866d", "kl": "#4169a3"}
    for ax, (key, ylabel, color) in zip(axes, specs):
        ax.plot(x, [row[key] for row in candidates], color=color, linewidth=1.8)
        for name, selected_row in selected.items():
            ax.axvline(selected_row["threshold"], linewidth=1.2, linestyle="--",
                       color=marker_colors[name], alpha=0.8,
                       label=f"{name}: {selected_row['threshold']:.3f}")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e5eaed")
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    axes[-1].set_xlabel("Symmetric clipping threshold α (MLP input units)")
    fig.subplots_adjust(top=0.86, bottom=0.1, left=0.13, right=0.97, hspace=0.25)
    fig.text(0.13, 0.97, "MLP input · INT8 clipping scan", fontsize=18,
             fontweight="bold", va="top")
    fig.text(0.13, 0.925,
             f"{cal_report['samples']} calibration activations · 2048-bin |x| histogram · 127 positive INT8 levels",
             fontsize=10, color="#526170", va="top")
    fig.text(0.13, 0.04,
             "Independent entropy-style proxy; actual RKNN parameters are in its step1 config.",
             fontsize=9, color="#526170")
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"clipping_scan.{suffix}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({name: {"alpha": row["candidate"]["threshold"],
                             "heldout_mse": row["heldout"]["mse"],
                             "heldout_saturation": row["heldout"]["saturation_fraction"]}
                      for name, row in metrics.items()}), flush=True)


if __name__ == "__main__":
    main()
