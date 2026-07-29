#!/bin/bash -l
# One STAGE of the teleop (joystick + sim2real) ladder on SCITAS Izar.
#
# Differs from izar_train.sbatch in exactly one way that matters: it takes WARMRUN (a run
# DIRECTORY) instead of WARM (a checkpoint file), and resolves it to that run's NEWEST checkpoint.
# The ladder is time-boxed rather than step-boxed — each stage runs until its --time wall and is
# KILLED, so the previous stage will usually have no final_model.zip to warm-start from. Pointing
# at the directory means the chain works whether the parent finished cleanly or was cut off.
# Pair with --dependency=afterany for the same reason.
#
#   sbatch --time=6:00:00 --export=ALL,PRESET=teleop_easy,NAME=teleop_easy izar_teleop.sbatch
#   sbatch --time=12:00:00 --dependency=afterany:$j1 \
#          --export=ALL,PRESET=teleop,NAME=teleop,WARMRUN=training/runs/teleop_easy izar_teleop.sbatch
#
#SBATCH --job-name=dash-teleop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module purge
module load gcc python
source "${VENV:-$HOME/venvs/dash}/bin/activate"

REPO="${REPO:-$HOME/rr_teleop}"
PRESET="${PRESET:-teleop_easy}"
STEPS="${STEPS:-400000000}"   # deliberately unreachable: the --time wall ends the stage, and
                              # checkpoints land every ~1M steps so nothing is lost when it does
NENVS="${NENVS:-18}"          # leave ~2 of the 20 cores for the learner + OS
NAME="${NAME:-$PRESET}"
WARMRUN="${WARMRUN:-}"        # previous stage's run DIRECTORY (not a file)

cd "$REPO"
python training/smoke_test.py                     # fail fast before burning GPU hours

# resolve WARMRUN -> newest checkpoint. final_model.zip if the parent finished cleanly, else the
# highest-numbered ppo_<steps>_steps.zip. Empty if the parent produced nothing (then this stage
# just starts cold, which is the right fallback rather than dying).
WARM=""
if [ -n "$WARMRUN" ]; then
    if [ -f "$WARMRUN/final_model.zip" ]; then
        WARM="$WARMRUN/final_model.zip"
    else
        WARM=$(ls -1 "$WARMRUN"/ppo_*_steps.zip 2>/dev/null \
               | sed 's/.*ppo_\([0-9]*\)_steps.zip/\1 &/' | sort -n | tail -1 | cut -d' ' -f2- || true)
    fi
    if [ -n "$WARM" ]; then
        echo "[stage] warm-starting $NAME from $WARM"
    else
        echo "[stage] WARNING: no checkpoint in $WARMRUN — starting $NAME COLD"
    fi
fi

# --resume auto so a requeue of THIS stage continues its own progress; --warm-start only applies
# on a fresh start, so the two compose safely.
python training/train.py --preset "$PRESET" --steps "$STEPS" --n-envs "$NENVS" \
     --subproc --resume auto --no-progress --name "$NAME" \
     ${WARM:+--warm-start "$WARM"}
