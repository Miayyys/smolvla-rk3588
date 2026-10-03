"""Build and validate a provisional HAQ action space from hardware evidence."""

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
import math
import re
from pathlib import Path
from typing import Any


SEARCHABLE_KINDS = {"linear", "conv2d", "embedding"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_key(value: str) -> list[tuple[int, Any]]:
    return [(1, int(part)) if part.isdigit() else (0, part.lower())
            for part in re.split(r"(\d+)", value)]


def _shape(value: Any) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    return [int(dim) for dim in value]


def _formats(value: Any) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    return [str(item) for item in value]


def _site_features(row: dict[str, Any]) -> list[float]:
    shape = _shape(row["weight_shape"])
    elements = math.prod(shape) if shape else 0
    out_dim = shape[0] if shape else 0
    in_dim = shape[1] if len(shape) > 1 else 0
    calls = row.get("calls_in_40_calibration_actions")
    calls = int(calls) if calls not in (None, "") else 0
    kind = row["kind"]
    return [
        math.log1p(int(row["source_bytes"])) / 25.0,
        math.log1p(elements) / 30.0,
        math.log1p(out_dim) / 15.0,
        math.log1p(in_dim) / 15.0,
        math.log1p(calls) / 10.0,
        float(kind == "linear"),
        float(kind == "conv2d"),
        float(kind == "embedding"),
    ]


def build_search_space(tables_path: Path) -> dict[str, Any]:
    tables_path = tables_path.resolve()
    tables = json.loads(tables_path.read_text())
    project_root = Path(__file__).resolve().parents[2]
    try:
        tables_label = tables_path.relative_to(project_root).as_posix()
    except ValueError:
        tables_label = str(tables_path)
    modules = tables["modules"]
    base_cases = tables["cost_cases"]
    supplemental_cases = tables["supplemental_costs"]

    formats: list[str] = []
    for row in modules:
        if row["kind"] in SEARCHABLE_KINDS and len(_formats(row["candidate_formats"])) > 1:
            for fmt in _formats(row["candidate_formats"]):
                if fmt not in formats:
                    formats.append(fmt)
    format_ids = {fmt: index for index, fmt in enumerate(formats)}

    base_by_module_format: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in base_cases:
        for module in case.get("modules", []):
            base_by_module_format.setdefault((module, case["format"]), []).append(case)
    supplemental_by_unit_format = {
        (case.get("unit"), case["format"]): case for case in supplemental_cases
    }

    ordered_modules = sorted(modules, key=lambda row: natural_key(row["module"]))
    action_sites: list[dict[str, Any]] = []
    fixed_or_unresolved: list[dict[str, Any]] = []
    options_without_cost_evidence: list[str] = []

    for row in ordered_modules:
        module = row["module"]
        kind = row["kind"]
        candidates = _formats(row["candidate_formats"])
        if kind not in SEARCHABLE_KINDS or len(candidates) < 2:
            fixed_or_unresolved.append({
                "module": module,
                "kind": kind,
                "reason": ("inactive_in_current_action_path" if kind == "inactive_linear"
                           else "one_candidate_listed_or_not_a_supported_weighted_operator"),
                "candidate_formats": candidates,
            })
            continue

        options = []
        for fmt in candidates:
            evidence = list(base_by_module_format.get((module, fmt), []))
            if kind == "embedding":
                special = supplemental_by_unit_format.get(("language_token_embedding_cpu_lookup", fmt))
                if special:
                    evidence.append(special)
            measured = bool(evidence) and all(item.get("status") == "measured" for item in evidence)
            if not measured:
                options_without_cost_evidence.append(f"{module}:{fmt}")
            options.append({
                "format": fmt,
                "format_id": format_ids[fmt],
                "signature_cost_evidence": "measured" if measured else "missing_or_incomplete",
                "cost_case_ids": [item["case_id"] for item in evidence],
                "cost_measurements_are_quality_results": False,
                "full_graph_feasibility_verified": False,
            })

        shape = _shape(row["weight_shape"])
        action_sites.append({
            "site_id": len(action_sites),
            "module": module,
            "kind": kind,
            "weight_shape": shape,
            "source_bytes": int(row["source_bytes"]),
            "input_shapes": row.get("input_shapes", []),
            "features": _site_features(row),
            "options": options,
        })

    candidate_count = sum(len(site["options"]) for site in action_sites)
    return {
        "schema_version": 1,
        "status": "provisional_parameterized_operator_space",
        "search_ready": False,
        "source": {
            "tables_path": tables_label,
            "tables_sha256": sha256(tables_path),
            "checkpoint_sha256": tables["source_sha256"],
        },
        "selection_policy": {
            "legacy_v0_precision_assignments_used": False,
            "manual_sensitivity_used_to_mask_or_prioritize": False,
            "ordering": "natural module path order; not verified against exported graph order",
            "candidate_evidence": "per-signature cost probes; no full-policy quality evidence",
        },
        "format_vocab": formats,
        "feature_schema": [
            "log1p_source_bytes/25", "log1p_weight_elements/30",
            "log1p_output_dimension/15", "log1p_input_dimension/15",
            "log1p_calls_in_calibration/10", "is_linear", "is_conv2d", "is_embedding",
        ],
        "counts": {
            "parameter_module_inventory": len(modules),
            "action_sites_with_multiple_listed_candidates": len(action_sites),
            "candidate_choices": candidate_count,
            "non_action_or_unresolved_inventory_rows": len(fixed_or_unresolved),
            "hardware_options_missing_signature_measurement": len(options_without_cost_evidence),
        },
        "unresolved_for_full_search": [
            "Reconcile all action sites with a traced/exported full-policy execution graph and backend fusion boundaries.",
            "Prove the listed signature measurements apply to each site in its real graph context.",
            "Apply any complete precision assignment to a policy and connect the fixed-observation evaluator; validate its proxy with closed-loop tasks.",
            "Connect measured-table costs to actual graph calls and fusion/conversion boundaries; verify at least 40% actual inference-model file compression.",
            "Determine whether single-option inventory rows truly have no independently controllable precision choice.",
        ],
        "action_sites": action_sites,
        "non_action_or_unresolved_inventory": fixed_or_unresolved,
    }


def validate_assignment(space: dict[str, Any], assignment: dict[str, str]) -> None:
    sites = {site["module"]: site for site in space["action_sites"]}
    if set(assignment) != set(sites):
        missing = sorted(set(sites) - set(assignment))
        extra = sorted(set(assignment) - set(sites))
        raise ValueError(f"Assignment keys differ from action space; missing={missing[:3]}, extra={extra[:3]}")
    for module, fmt in assignment.items():
        choices = {option["format"] for option in sites[module]["options"]}
        if fmt not in choices:
            raise ValueError(f"Unsupported candidate {fmt!r} for {module}; choices={sorted(choices)}")
