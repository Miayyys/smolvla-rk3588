#!/usr/bin/env bash
# Small real-label diagnostic; does not claim converged training or RKNN parity.
set -euo pipefail
cd /root/qvla
qvla_diag_lr="${QVLA_DIAGNOSTIC_LR:-1e-5}"
qvla_diag_tag="${QVLA_DIAGNOSTIC_TAG:-real40_v1}"
if [[ ! "$qvla_diag_tag" =~ ^[A-Za-z0-9_]+$ ]]; then echo "Invalid run tag" >&2; exit 1; fi
.venv/bin/python scripts/audit_openvla_teacher_labels.py \
    --cache runs/teacher_actions_real_smoke_v2 --inputs runs/teacher_inputs_real_smoke_v2 \
    --dataset-root data/libero --splits data/libero_splits.json \
    --partition config/evaluation_partition_v2.json \
    --output runs/teacher_actions_real_smoke_v2/contract_audit.json
common=(--model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets \
    --dataset-root data/libero --splits data/libero_splits.json \
    --partition config/evaluation_partition_v2.json --candidate config/haq_candidate_v2.json \
    --teacher-cache runs/teacher_actions_real_smoke_v2 --teacher-weight 0.2 \
    --steps 40 --learning-rate "$qvla_diag_lr" --seed 29)
.venv/bin/python scripts/qat_train_haq.py --mode fp-distill "${common[@]}" \
    --output-dir "runs/distill_fp_${qvla_diag_tag}"
.venv/bin/python scripts/qat_train_haq.py --mode qat "${common[@]}" \
    --initial-master "runs/distill_fp_${qvla_diag_tag}/distilled_float_master.safetensors" \
    --output-dir "runs/distill_qat_${qvla_diag_tag}" --allow-unverified-backend-diagnostic
