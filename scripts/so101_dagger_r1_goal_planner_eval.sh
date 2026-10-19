#!/usr/bin/env bash
# Experiment D on the goal-conditioned DAgger model: goals from the model's own planner head (no simulator ground
# truth reaches the policy), 100 validation seeds twice; execute 10 per chunk, stop on success.
set -u
cd /home/loser/Study/QVLA/.claude/worktrees/so101-goal-conditioned
export PYTHONPATH=$PWD MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/loser/Study/QVLA/.venv-haq-local/bin/python
RUN=runs/so101_dagger_r1_goal
LOG=runs/so101_dagger_r1_goal_pipeline.log
mark() { echo "$(date '+%F %T') $*" >> $LOG; }
for rep in 1 2; do
  extra=""; [ $rep -eq 1 ] && extra="--trace --gif"
  $PY -m qvla.evaluation.so101_goal_eval --config config/so101_dagger_r1_goal.json --model-dir $RUN/last \
    --output ${RUN}_eval_planner_100_rep$rep --split validation --execute 10 --stop-on-success --goal-source planner $extra \
    > ${RUN}_eval_planner_100_rep$rep.log 2>&1
  mark "EVAL planner_100_rep$rep EXIT $? $(tail -1 ${RUN}_eval_planner_100_rep$rep.log)"
done
mark PLANNER_DONE
