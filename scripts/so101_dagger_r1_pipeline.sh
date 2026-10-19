#!/usr/bin/env bash
# DAgger round 1 on the local RTX 4060, unattended (launch with systemd-run --user + systemd-inhibit):
#   1) baseline A step 10000 drives 400 stored training seeds (2 GPU processes); samples every 5 steps;
#   2) expert-takeover labels (CPU workers);  3) 2-step smoke;  4) fine-tune 5000 steps from A (config/so101_dagger_r1.json);
#   5) closed loop on all 100 validation seeds, twice (execute 10, stop on success);
#   6) only if DAgger beats A over the two 100-seed runs: control fine-tune without DAgger data, same evaluation.
set -u
cd /home/loser/Study/QVLA/.claude/worktrees/so101-goal-conditioned
export PYTHONPATH=$PWD MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/loser/Study/QVLA/.venv-haq-local/bin/python
DATA=runs/so101_dagger_r1_data
LOG=runs/so101_dagger_r1_pipeline.log
A=runs/so101_backup/full_v2_step10000
mark() { echo "$(date '+%F %T') $*" >> $LOG; }
fail() { mark "FAILED $*"; exit 1; }

evaluate() {  # config model output extra...
  local cfg=$1 model=$2 out=$3; shift 3
  $PY -m qvla.evaluation.finetune_so101_scripted --config $cfg --model-dir $model --output $out \
    --split validation --execute 10 --stop-on-success "$@" > $out.log 2>&1
  mark "EVAL $out EXIT $? $(tail -1 $out.log)"
}
successes() { $PY -c "import json,sys; print(sum(json.load(open(f + '/report.json'))['summary']['success'] for f in sys.argv[1:]))" "$@"; }

mark START
SEEDS=$(cat runs/so101_dagger_r1_seeds.txt)
for i in 0 1; do
  $PY -m qvla.evaluation.collect_so101_dagger rollout --config runs/so101_A_local_eval_config.json --model-dir $A/last \
    --stats-dir $A --output $DATA/shard$i --seeds $SEEDS --shard $i --num-shards 2 > $DATA.rollout$i.log 2>&1 &
done
wait
for i in 0 1; do [ -f $DATA/shard$i/rollout.json ] || fail "rollout shard$i"; mark "ROLLOUT shard$i $(tail -1 $DATA.rollout$i.log)"; done
for i in 0 1; do
  $PY -m qvla.evaluation.collect_so101_dagger label --output $DATA/shard$i --workers 8 > $DATA.label$i.log 2>&1 || fail "label shard$i"
  mark "LABEL shard$i $(tail -1 $DATA.label$i.log)"
done

train() {  # config output
  $PY -m qvla.evaluation.finetune_so100_real --config $1 --stage prepare > $2.train.log 2>&1 || fail "prepare $1"
  $PY -m qvla.evaluation.finetune_so100_real --config $1 --stage smoke >> $2.train.log 2>&1 || fail "smoke $1"
  $PY -m qvla.evaluation.finetune_so100_real --config $1 --stage train >> $2.train.log 2>&1 || fail "train $1"
  mark "TRAIN $1 done $(grep -c '\"stage\": \"validation\"' $2/events.jsonl) validations"
}
train config/so101_dagger_r1.json runs/so101_dagger_r1
evaluate config/so101_dagger_r1.json runs/so101_dagger_r1/last runs/so101_dagger_r1_eval_100_rep1 --trace --gif
evaluate config/so101_dagger_r1.json runs/so101_dagger_r1/last runs/so101_dagger_r1_eval_100_rep2
D=$(successes runs/so101_dagger_r1_eval_100_rep1 runs/so101_dagger_r1_eval_100_rep2)
B=$(successes runs/so101_A_step10000_local_eval_100 runs/so101_A_step10000_local_eval_100_rep2)
mark "COMPARE dagger=$D/200 baseline_A=$B/200 (control runs only if dagger > A)"
if [ "$D" -gt "$B" ]; then
  train config/so101_dagger_r1_control.json runs/so101_dagger_r1_control
  evaluate config/so101_dagger_r1_control.json runs/so101_dagger_r1_control/last runs/so101_dagger_r1_control_eval_100_rep1 --trace
  evaluate config/so101_dagger_r1_control.json runs/so101_dagger_r1_control/last runs/so101_dagger_r1_control_eval_100_rep2
else
  mark "CONTROL_SKIPPED"
fi
mark DONE
