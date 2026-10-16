#!/usr/bin/env bash
# Mid-run closed-loop check for a SO101 scripted training run (runs detached on GPUServer).
# Waits until training has passed STEP (so best/last of that evaluation are fully written), snapshots
# the best checkpoint selected so far, then evaluates it on all validation seeds while training continues.
# Usage: so101_midrun_eval.sh <config> <step> <execute_per_chunk>
set -u
cd /root/qvla_so101
source /root/nvidia-egl-550/env.sh
export PYTHONPATH=/root/qvla_so101 SO101_NEXUS_SOURCE=/root/so101-nexus
PY=/root/qvla/.venv/bin/python
CONFIG=$1; STEP=$2; EXEC=$3
RUN=$($PY -c "import json; print(json.load(open('$CONFIG'))['output'])")
SNAP=$RUN/snapshot_step${STEP}
OUT=${RUN}_eval_step${STEP}_exec${EXEC}
until $PY -c "import json,sys; sys.exit(0 if json.load(open('$RUN/progress.json'))['step'] > $STEP + 10 else 1)" 2>/dev/null; do
  if grep -q TRAIN_EXIT ${RUN}.log 2>/dev/null; then echo "training ended before step $STEP" >> ${OUT}.log; break; fi
  sleep 120
done
mkdir -p $SNAP
cp -r $RUN/best $SNAP/best
$PY - "$RUN" "$SNAP" "$STEP" <<'EOF'
import hashlib, json, sys
run, snap, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
ev = [json.loads(l) for l in open(f'{run}/events.jsonl')]
val = [(e['step'], e['normalized_action_mae']) for e in ev if e.get('stage') == 'validation' and e['step'] <= step]
best = None
for s, m in val:  # same rule as the trainer: strictly better replaces
    if best is None or m < best[1]:
        best = (s, m)
digest = hashlib.sha256(open(f'{snap}/best/model.safetensors', 'rb').read()).hexdigest()
json.dump({'snapshot_after_step': step, 'best_step': best[0], 'best_offline_normalized_mae': best[1],
           'validations': val, 'model_sha256': digest}, open(f'{snap}/snapshot.json', 'w'), indent=2)
print(json.dumps({'best_step': best[0], 'best_metric': best[1], 'model_sha256': digest[:16]}))
EOF
$PY -m qvla.evaluation.finetune_so101_scripted --config $CONFIG --model-dir $SNAP/best --output $OUT \
  --split validation --execute $EXEC --stop-on-success > ${OUT}.log 2>&1 < /dev/null
echo "EVAL_EXIT $?" >> ${OUT}.log
