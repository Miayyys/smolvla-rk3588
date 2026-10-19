#!/usr/bin/env bash
# Experiment B on the local RTX 4060 (2026-10-18), unattended:
#   1) train to step 5000 (A's schedule), snapshot best, closed-loop on A's 30 validation seeds (oracle goals);
#   2) stop-loss: continue only if >= 6/30 succeed (A had 3/30 at steps 5000 and 10000);
#   3) resume to step 10000, snapshot best, 100 validation seeds with oracle goals and with the null goal;
#   4) baseline A (step-10000 backup) re-evaluated locally on the same 100 seeds (same renderer as B).
# Evaluation: execute 10 per chunk, 10 Euler steps, max 900 steps, stop on success, gate max 2 attempts.
set -u
cd /home/loser/Study/QVLA/.claude/worktrees/so101-goal-conditioned
export PYTHONPATH=$PWD MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/loser/Study/QVLA/.venv-haq-local/bin/python
CFG=config/so101_goal_B.json
RUN=runs/so101_goal_B
LOG=runs/so101_goal_B_pipeline.log
mark() { echo "$(date '+%F %T') $*" >> $LOG; }

snapshot() {  # copy the selected checkpoint before training continues and overwrites it
  mkdir -p $RUN/snapshot_step$1 && cp -r $RUN/best $RUN/snapshot_step$1/best
  sha256sum $RUN/snapshot_step$1/best/model.safetensors > $RUN/snapshot_step$1/model.sha256
  $PY - "$RUN" "$1" >> $LOG 2>&1 <<'EOF'
import json, sys
run, step = sys.argv[1], int(sys.argv[2])
val = [json.loads(l) for l in open(f'{run}/events.jsonl')]
val = [(e['step'], e['normalized_action_mae']) for e in val if e.get('stage') == 'validation' and e['step'] <= step]
best = None
for s, m in val:  # trainer rule: strictly better replaces
    if best is None or m < best[1]:
        best = (s, m)
print(json.dumps({'snapshot_step': step, 'best_step': best[0], 'best_offline_normalized_mae': best[1]}))
EOF
}

goal_eval() {  # name model-dir extra-args...
  local name=$1 model=$2; shift 2
  $PY -m qvla.evaluation.so101_goal_eval --config $CFG --model-dir $model --output runs/so101_goal_B_eval_$name \
    --split validation --execute 10 --stop-on-success --gif --gate-max-attempts 2 "$@" > runs/so101_goal_B_eval_$name.log 2>&1
  local rc=$?; mark "EVAL $name EXIT $rc $(tail -1 runs/so101_goal_B_eval_$name.log)"
}

mark START
$PY -m qvla.evaluation.finetune_so100_real --config $CFG --stage train --stop-step 5000 > runs/so101_goal_B_train.log 2>&1
rc=$?; mark "TRAIN_5000 EXIT $rc"; [ $rc -eq 0 ] || { mark FAILED; exit 1; }
snapshot 5000
goal_eval step5000_oracle_30 $RUN/snapshot_step5000/best --seeds $(seq 100000 100029)
wins=$($PY -c "import json; print(json.load(open('runs/so101_goal_B_eval_step5000_oracle_30/report.json'))['summary']['success'])" 2>/dev/null || echo -1)
mark "STOP_LOSS_CHECK success=$wins/30 (continue if >= 6)"
if [ "$wins" -ge 6 ]; then
  $PY -m qvla.evaluation.finetune_so100_real --config $CFG --stage train --resume --stop-step 10000 >> runs/so101_goal_B_train.log 2>&1
  rc=$?; mark "TRAIN_10000 EXIT $rc"; [ $rc -eq 0 ] || { mark FAILED; exit 1; }
  snapshot 10000
  goal_eval step10000_oracle_100 $RUN/snapshot_step10000/best
  goal_eval step10000_null_100 $RUN/snapshot_step10000/best --goal-source null
else
  mark "STOPPED_BY_STOP_LOSS"
  goal_eval step5000_null_30 $RUN/snapshot_step5000/best --seeds $(seq 100000 100029) --goal-source null
fi
$PY -m qvla.evaluation.finetune_so101_scripted --config runs/so101_A_local_eval_config.json \
  --model-dir runs/so101_backup/full_v2_step10000/last --output runs/so101_A_step10000_local_eval_100 \
  --split validation --execute 10 --stop-on-success > runs/so101_A_step10000_local_eval_100.log 2>&1
mark "EVAL A_local_100 EXIT $? $(tail -1 runs/so101_A_step10000_local_eval_100.log)"
mark DONE
