#!/bin/bash -l
# Cross-entropy search for an open-loop braking schedule on one V100, then (optionally) a run-and-stop demo.
# The schedule is POLICY-SPECIFIC: it rides on top of the policy's own per-tick residual, so re-run it for
# every checkpoint you want to brake.
#   search: sbatch --export=ALL,RUN=v2c_s2_brakeprior_s41,CKPT=walk_v2/runs/.../best_speed.msgpack walk_v2/slurm/izar_brake.sbatch
#   demo:   sbatch --export=ALL,RUN=...,CKPT=...,DEMO=walk_v2/results/brake_x.json,SEED=1 walk_v2/slurm/izar_brake.sbatch
# ALWAYS demo on a SEED the fit did not see. env.reset splits the key per env, so the same seed
# replays the very episodes the CEM selected against and the stop rate comes back flattering.
#SBATCH --job-name=dash-v2-brake
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
TAG="${TAG:-$(date +%H%M)}"
python walk_v2/tools/brake_search.py --run "walk_v2/runs/$RUN" ${CKPT:+--checkpoint "$CKPT"} \
    --pop "${POP:-512}" --iters "${ITERS:-8}" --reps "${REPS:-4}" ${HOLD_RUN:+--hold-run} ${FREE_CLOCK:+--free-clock} --stagger-s "${STAGGER_S:-0}" --cruise-s "${CRUISE_S:-6}" --brake-s "${BRAKE_S:-8}" \
    ${DEMO:+--demo "$DEMO"} ${BRAKE_AT:+--brake-at "$BRAKE_AT"} --seed "${SEED:-0}" \
    --json "walk_v2/results/brake_${RUN}_${TAG}.json" \
    ${DEMO:+--video "walk_v2/results/run_and_stop_${RUN}_${TAG}.mp4"}
echo "[brake] done"
