#!/bin/bash -l
# Post-processing for ONE stage of the m1..m6_CPG chain: training plot, greedy eval, cadence
# diagnostic, and an mp4.
#
#   sbatch --dependency=afterany:<train job> --export=ALL,RUN=m3_CPG training/slurm/cpg_stage_post.sbatch
#
# Why a SEPARATE job instead of appending these commands to the training sbatch: a stage stopped by
# its wall clock is SIGKILLed, so trailing commands in the training script would never run and the
# unattended chain would silently produce no plots or videos. As its own `afterany` dependent, this
# runs whether the stage finished, timed out, or crashed.
#SBATCH --job-name=cpg-post
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out

set -uo pipefail        # NOT -e: a failed video must not cost us the plot and the eval numbers

module purge
module load gcc python
source "${VENV:-$HOME/venvs/dash}/bin/activate"
cd "${REPO:-$HOME/running_robot}"

RUN="${RUN:?set RUN=<run name>}"
DIR="training/runs/$RUN"
[ -d "$DIR" ] || { echo "[post] $DIR does not exist"; exit 1; }

shopt -s nullglob
ckpts=("$DIR"/ppo_*_steps.zip)
[ -f "$DIR/final_model.zip" ] && ckpts+=("$DIR/final_model.zip")
if [ ${#ckpts[@]} -eq 0 ]; then
    echo "[post] $RUN has no checkpoint — stage produced nothing, nothing to plot"
    exit 0
fi
echo "[post] $RUN: ${#ckpts[@]} checkpoint(s)"

echo "===== training plot ====="
python training/plot_training.py --run "$DIR" && echo "[post] wrote $DIR/training_plots.png"

echo "===== greedy eval ====="
python training/evaluate.py --run "$DIR" --episodes 5 2>&1 | tail -12

echo "===== cadence / gait ====="
python training/diagnose_cadence.py --run "$DIR" --episodes 4 2>&1 | tail -16

echo "===== video ====="
# Headless rendering needs an offscreen GL backend and which one works is node-dependent, so try
# them in order rather than assuming. Without this the unattended chain would produce no videos.
wrote=""
for gl in egl osmesa glfw; do
    echo "[post] trying MUJOCO_GL=$gl"
    if MUJOCO_GL=$gl python training/evaluate.py --run "$DIR" --episodes 1 --seconds 30 \
            --video "$DIR/$RUN.mp4" 2>&1 | tail -4; then
        if [ -s "$DIR/$RUN.mp4" ]; then wrote=$gl; break; fi
    fi
done
if [ -n "$wrote" ]; then
    echo "[post] wrote $DIR/$RUN.mp4 via MUJOCO_GL=$wrote ($(du -h "$DIR/$RUN.mp4" | cut -f1))"
else
    echo "[post] WARNING: no GL backend could render; pull the checkpoint and render locally"
fi

echo "[post] done: $RUN"
