#!/usr/bin/env bash
# Goal-conditioned structure on the DAgger round 1 data (same init, data, schedule and seed as DAgger round 1):
#   prepare -> 2-step smoke -> 5000 steps -> closed loop on 100 validation seeds with takeover goals (twice) and
#   with the null goal (once). Execute 10 per chunk, stop on success. Launch with systemd-run --user + systemd-inhibit.
set -u
cd /home/loser/Study/QVLA/.claude/worktrees/so101-goal-conditioned
export PYTHONPATH=$PWD MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/loser/Study/QVLA/.venv-haq-local/bin/python
CFG=config/so101_dagger_r1_goal.json
RUN=runs/so101_dagger_r1_goal
LOG=runs/so101_dagger_r1_goal_pipeline.log
mark() { echo "$(date '+%F %T') $*" >> $LOG; }
fail() { mark "FAILED $*"; exit 1; }

mark START
for i in 0 1; do [ -f runs/so101_dagger_r1_data_goal/shard$i/manifest.json ] || fail "goal labels shard$i missing"; done
$PY -m qvla.evaluation.finetune_so100_real --config $CFG --stage prepare > $RUN.train.log 2>&1 || fail prepare
$PY -m qvla.evaluation.finetune_so100_real --config $CFG --stage smoke >> $RUN.train.log 2>&1 || fail smoke
$PY -m qvla.evaluation.finetune_so100_real --config $CFG --stage train >> $RUN.train.log 2>&1 || fail train
mark "TRAIN done"

goal_eval() {  # name extra...
  local name=$1; shift
  $PY -m qvla.evaluation.so101_goal_eval --config $CFG --model-dir $RUN/last --output ${RUN}_eval_$name \
    --split validation --execute 10 --stop-on-success "$@" > ${RUN}_eval_$name.log 2>&1
  mark "EVAL $name EXIT $? $(tail -1 ${RUN}_eval_$name.log)"
}
goal_eval takeover_100_rep1 --goal-source takeover --trace --gif
goal_eval takeover_100_rep2 --goal-source takeover
goal_eval null_100 --goal-source null
mark DONE
