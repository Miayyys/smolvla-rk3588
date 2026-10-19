#!/usr/bin/env bash
# Chained (3 pick-and-place tasks per episode, robot not reset) closed loop on the 100 validation seeds:
#   lane 1: original structure + DAgger round 1 (twice), goal structure with takeover goals (once, upper bound);
#   lane 2: goal structure with its planner head (twice, deployable).
set -u
cd /home/loser/Study/QVLA/.claude/worktrees/so101-goal-conditioned
export PYTHONPATH=$PWD MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/loser/Study/QVLA/.venv-haq-local/bin/python
LOG=runs/so101_chain_eval_pipeline.log
mark() { echo "$(date '+%F %T') $*" >> $LOG; }
chain() {  # name config model goal-source extra...
  local name=$1 cfg=$2 model=$3 source=$4; shift 4
  $PY -m qvla.evaluation.so101_chain_eval --config $cfg --model-dir $model --output runs/so101_chain_$name \
    --split validation --chain 3 --goal-source $source "$@" > runs/so101_chain_$name.log 2>&1
  mark "CHAIN $name EXIT $? $(tail -1 runs/so101_chain_$name.log)"
}
mark START
(
  chain dagger_r1_rep1 config/so101_dagger_r1.json runs/so101_dagger_r1/last none --gif
  chain dagger_r1_rep2 config/so101_dagger_r1.json runs/so101_dagger_r1/last none
  chain goal_takeover_rep1 config/so101_dagger_r1_goal.json runs/so101_dagger_r1_goal/last takeover
) &
(
  chain goal_planner_rep1 config/so101_dagger_r1_goal.json runs/so101_dagger_r1_goal/last planner --gif
  chain goal_planner_rep2 config/so101_dagger_r1_goal.json runs/so101_dagger_r1_goal/last planner
) &
wait
mark DONE
