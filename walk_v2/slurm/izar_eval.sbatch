#!/bin/bash -l
# Greedy evaluation of a walk_v2 run's latest checkpoint on one V100 (debug QOS, fast queue): 16 dashes on
# the nominal plant with the assist at 0, a JSON summary and an mp4 of env 0.
#   sbatch --export=ALL,RUN=v2c_fast_dp4x_s0 walk_v2/slurm/izar_eval.sbatch
#   sbatch --export=ALL,RUN=v2c_fast_dp4x_s0,CKPT=walk_v2/runs/v2c_fast_dp4x_s0/ckpt_30000000.msgpack ...
#SBATCH --job-name=dash-v2-eval
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module purge
export PATH=$HOME/.local/bin:$PATH
source "${VENV:-$HOME/venvs/dash-v2}/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
cd "${REPO:-$HOME/running_robot}"
RUN="${RUN:?set RUN}"
TAG="${TAG:-$(date +%H%M)}${ASSIST:+_assist$ASSIST}${STOPLIGHT:+_light$STOPLIGHT}"
python walk_v2/evaluate.py --run "walk_v2/runs/$RUN" ${CKPT:+--checkpoint "$CKPT"} --episodes "${EPISODES:-16}" \
    --seconds "${SECONDS_CAP:-60}" --assist "${ASSIST:-0}" --stoplight "${STOPLIGHT:-0}" ${FREE_CLOCK:+--free-clock} --video "walk_v2/runs/$RUN/greedy_${TAG}.mp4" --json "walk_v2/runs/$RUN/greedy_${TAG}.json"
echo "[eval] done"
