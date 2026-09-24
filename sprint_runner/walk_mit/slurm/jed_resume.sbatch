#!/bin/bash -l
# RESUME variant of jed_train.sbatch: identical, plus CONFIG=<run>/resolved_config.json so a run
# whose recorded config differs from the live preset only by JSON typing (tuple vs list on
# Any-typed fields, e.g. motor_vel_limit) continues under --resume auto instead of tripping the
# stale-config guard: --config rebuilds the SAME Config on both sides of the comparison.
#
# Submit:  sbatch --export=ALL,PRESET=v2c_s1,STEPS=300000000,SEED=0,NAME=v2c_s1_s0,CONFIG=walk_mit/runs/v2c_s1_s0/resolved_config.json \
#                 walk_mit/slurm/jed_resume.sbatch
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
CONFIG="${CONFIG:-}"
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
     --subproc --resume auto --no-progress --name "$NAME" ${CONFIG:+--config "$CONFIG"} \
     ${WARM:+--warm-start "$WARM"} ${SEED:+--seed "$SEED"} ${NSTEPS:+--n-steps "$NSTEPS"}
