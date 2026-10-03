#!/usr/bin/env bash
# All network downloads are done locally before this offline install.
set -euo pipefail
cd /root/qvla
if [ ! -x .venv-teacher/bin/python ]; then
    python -m venv --system-site-packages .venv-teacher
fi
.venv-teacher/bin/python -m pip install --no-index --no-deps --no-build-isolation -e third_party/transformers-openvla-oft
python qvla/distillation/patch_openvla_inference_import.py --teacher-repo third_party/openvla-oft
# This unused training package was installed during initial import investigation.
# Its strict tensorflow2.15 dependency is unnecessary after the identical-enum import fix.
if .venv-teacher/bin/python -m pip show dlimp >/dev/null 2>&1; then
    .venv-teacher/bin/python -m pip uninstall -y dlimp
fi
.venv-teacher/bin/python -m pip install --no-index --no-deps \
    --find-links artifacts/teacher_wheelhouse -r config/teacher_inference_lock.txt
.venv-teacher/bin/python qvla/distillation/cache_openvla_teacher.py \
    --teacher-repo third_party/openvla-oft \
    --checkpoint artifacts/teacher/openvla-oft-libero-10 \
    --inputs runs/teacher_inputs_real_smoke_v2 --output runs/teacher_actions_real_smoke_v2 \
    --preflight-only
.venv-teacher/bin/python -m pip freeze > runs/teacher_environment_freeze.txt
