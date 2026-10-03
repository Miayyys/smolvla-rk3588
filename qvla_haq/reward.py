"""Historical quality-first reward contract, retained for earlier smoke tests.

The 2026-09-30 active objective is at least 40% model-file compression plus
quality/table-speed optimization. This historical scorer is not its reward.
"""

from __future__ import annotations

import math
import re
from typing import Any


class RewardContractError(ValueError):
    pass


def _positive_number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RewardContractError(f"Missing numeric field: {key}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise RewardContractError(f"{key} must be finite and positive")
    return value


def score_complete_evaluation(result: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Score one candidate only if all quality and full-board gates are present.

    ``contract`` has explicit per-suite non-inferiority margins, an action
    anomaly limit, RAM/latency budgets, FP-board reference values, and a fixed
    infeasible reward. These project thresholds are intentionally not guessed.
    """
    required_contract = {"suite_delta", "max_action_anomaly_rate", "ram_budget_bytes",
                         "p95_budget_ms", "reference", "infeasible_reward"}
    missing = sorted(required_contract - contract.keys())
    if missing:
        raise RewardContractError(f"Reward contract is incomplete: {missing}")
    if not contract["suite_delta"]:
        raise RewardContractError("suite_delta must name at least one LIBERO suite")
    penalty = contract["infeasible_reward"]
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not math.isfinite(penalty):
        raise RewardContractError("infeasible_reward must be a finite configured number")
    if penalty >= 0:
        raise RewardContractError("infeasible_reward must be negative")

    violations = []
    if result.get("evaluation_scope") != "full_policy_board_and_paired_libero_dev":
        violations.append("evaluation_scope_is_not_full_policy_board_and_paired_dev")
    comparison = result.get("comparison")
    if not isinstance(comparison, dict):
        raise RewardContractError("Missing paired comparison metadata")
    if comparison.get("paired") is not True:
        violations.append("comparison_is_not_paired")
    if comparison.get("split") != "development":
        violations.append("comparison_did_not_use_development_split")
    partition_hash = comparison.get("partition_sha256")
    if not isinstance(partition_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", partition_hash):
        raise RewardContractError("comparison.partition_sha256 must be a lowercase SHA-256")

    quality = result.get("quality")
    deployment = result.get("deployment")
    if not isinstance(quality, dict) or not isinstance(deployment, dict):
        raise RewardContractError("result must contain quality and deployment objects")
    lcb = quality.get("suite_success_rate_difference_lcb95")
    if not isinstance(lcb, dict):
        raise RewardContractError("Missing per-suite paired success-rate confidence bounds")
    for suite, delta in contract["suite_delta"].items():
        if suite not in lcb:
            raise RewardContractError(f"Missing quality lower bound for suite {suite}")
        if not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
            raise RewardContractError(f"Invalid quality tolerance for suite {suite}")
        value = lcb[suite]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RewardContractError(f"Invalid quality lower bound for suite {suite}")
        if value < -float(delta):
            violations.append(f"quality_gate_failed:{suite}")

    anomaly_rate = quality.get("action_anomaly_rate")
    anomaly_limit = contract["max_action_anomaly_rate"]
    if not isinstance(anomaly_rate, (int, float)) or not math.isfinite(anomaly_rate):
        raise RewardContractError("Missing action_anomaly_rate")
    if not isinstance(anomaly_limit, (int, float)) or not math.isfinite(anomaly_limit):
        raise RewardContractError("Missing max_action_anomaly_rate in reward contract")
    if not 0 <= anomaly_rate <= 1 or not 0 <= anomaly_limit <= 1:
        raise RewardContractError("Action anomaly rates must lie in [0, 1]")
    if anomaly_rate > anomaly_limit:
        violations.append("action_anomaly_gate_failed")

    if deployment.get("full_policy_board_run_verified") is not True:
        violations.append("full_policy_board_run_not_verified")
    ram = _positive_number(deployment, "peak_system_ram_bytes")
    p95 = _positive_number(deployment, "action_chunk_p95_ms")
    package = _positive_number(deployment, "deployment_package_bytes")
    ram_budget = _positive_number(contract, "ram_budget_bytes")
    p95_budget = _positive_number(contract, "p95_budget_ms")
    if ram > ram_budget:
        violations.append("ram_budget_exceeded")
    if p95 > p95_budget:
        violations.append("p95_budget_exceeded")

    if violations:
        return {"feasible": False, "reward": float(penalty), "G": None,
                "violations": violations}

    reference = contract["reference"]
    r0 = _positive_number(reference, "R0_bytes")
    t0 = _positive_number(reference, "T0_p95_ms")
    b0 = _positive_number(reference, "B0_bytes")
    log_g = 0.5 * math.log(r0 / ram) + 0.3 * math.log(t0 / p95) + 0.2 * math.log(b0 / package)
    return {"feasible": True, "reward": log_g, "log_G": log_g,
            "G": math.exp(log_g), "violations": []}
