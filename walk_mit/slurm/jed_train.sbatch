#!/bin/bash -l
# walk_mit training on SCITAS JED (CPU, academic partition). Adapted from
# training/slurm/jed_train.sbatch with ONLY the code paths changed: this arm runs the
# self-contained walk_mit/ copy, so cluster-side edits to training/ can never touch it mid-run.
#
# Submit:  sbatch --export=ALL,PRESET=walk_fwd_m2_mit,STEPS=100000000,SEED=0,NAME=mit_m2_s0,WARM=training/runs/ladder3_m2_s0 \
#                 walk_mit/slurm/jed_train.sbatch
#
#SBATCH --job-name=dash-mit
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module purge
module load gcc python
source "${VENV:-$HOME/venvs/dash-jed}/bin/activate"

# One BLAS thread per env worker; the learner gets its own pool (see training/slurm/jed_train.sbatch).
export OMP_NUM_THREADS=1
export LEARNER_THREADS=8

REPO="${REPO:-$HOME/running_robot}"
PRESET="${PRESET:?set PRESET}"
STEPS="${STEPS:-100000000}"
# 64 x 288 = 18432 = the same PPO rollout as every other arm: bit-identical optimiser maths.
NENVS="${NENVS:-64}"
NSTEPS="${NSTEPS:-288}"
NAME="${NAME:-$PRESET}"
WARM="${WARM:-}"
SEED="${SEED:-}"

cd "$REPO"

if [ -n "$WARM" ] && [ -d "$WARM" ]; then
    RESOLVED=$({ ls -1t "$WARM"/final_model.zip "$WARM"/ppo_*_steps.zip 2>/dev/null || true; } \
               | head -1)
    if [ -z "$RESOLVED" ]; then
        echo "[sbatch] FATAL: WARM=$WARM is a directory with no checkpoint in it" >&2
        exit 1
    fi
    echo "[sbatch] WARM directory $WARM -> $RESOLVED"
    WARM="$RESOLVED"
fi

python walk_mit/smoke_test.py                     # fail fast before burning node-days

python walk_mit/train.py --preset "$PRESET" --steps "$STEPS" --n-envs "$NENVS" \
     --subproc --resume auto --no-progress --name "$NAME" \
     ${WARM:+--warm-start "$WARM"} ${SEED:+--seed "$SEED"} ${NSTEPS:+--n-steps "$NSTEPS"}
