#!/usr/bin/env python3
"""Smoke-test HAQ proposal, update, checkpoint, and reward-gate plumbing only."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))


import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qvla.haq.policy import RecurrentPolicyGradient  # noqa: E402
from qvla.haq.reward import score_complete_evaluation  # noqa: E402
from qvla.haq.search_space import build_search_space, validate_assignment  # noqa: E402


def main() -> None:
    space = build_search_space(ROOT / "config/hardware/tables.json")
    assert space["search_ready"] is False
    assert space["counts"]["parameter_module_inventory"] == 397
    assert space["counts"]["action_sites_with_multiple_listed_candidates"] == 304
    assert space["counts"]["candidate_choices"] == 1517
    assert space["counts"]["hardware_options_missing_signature_measurement"] == 0

    policy = RecurrentPolicyGradient(space, hidden_size=8, seed=29)
    samples = [policy.sample() for _ in range(2)]
    for assignment, _ in samples:
        validate_assignment(space, assignment)
        assert len(assignment) == 304
    update = policy.update_batch([item[1] for item in samples], [-1.0, 0.25])
    assert np.isfinite(update["gradient_norm_before_clip"])
    assert policy.optimizer_step == 1

    with tempfile.TemporaryDirectory(prefix="qvla-haq-smoke-") as temp:
        checkpoint = Path(temp) / "policy.npz"
        policy.save(checkpoint)
        restored = RecurrentPolicyGradient(space, hidden_size=8, seed=1)
        restored.load(checkpoint)
        next_a, _ = policy.sample()
        next_b, _ = restored.sample()
        assert next_a == next_b

    contract = {
        "suite_delta": {"spatial": 0.02, "object": 0.02},
        "max_action_anomaly_rate": 0.01,
        "ram_budget_bytes": 200,
        "p95_budget_ms": 30,
        "reference": {"R0_bytes": 200, "T0_p95_ms": 30, "B0_bytes": 100},
        "infeasible_reward": -1.0,
    }
    result = {
        "evaluation_scope": "full_policy_board_and_paired_libero_dev",
        "comparison": {"paired": True, "split": "development", "partition_sha256": "a" * 64},
        "quality": {"suite_success_rate_difference_lcb95": {"spatial": 0, "object": 0},
                    "action_anomaly_rate": 0},
        "deployment": {"full_policy_board_run_verified": True,
                       "peak_system_ram_bytes": 100, "action_chunk_p95_ms": 20,
                       "deployment_package_bytes": 50},
    }
    scored = score_complete_evaluation(result, contract)
    expected_g = (200 / 100) ** 0.5 * (30 / 20) ** 0.3 * (100 / 50) ** 0.2
    assert scored["feasible"] and np.isclose(scored["G"], expected_g)
    result["evaluation_scope"] = "subgraph_proxy"
    rejected = score_complete_evaluation(result, contract)
    assert not rejected["feasible"] and rejected["G"] is None
    print(json.dumps({"status": "passed", "scope": "controller/reward contract plumbing only",
                      "real_model_or_RK3588_evaluation": False,
                      "policy_update": update}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
