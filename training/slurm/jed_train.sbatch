#!/bin/bash -l
# DASH-01 training on SCITAS JED (CPU cluster, ACADEMIC partition — the only one students may use).
#
# Why jed (2026-08-10): izar had 0 idle nodes, 23 pending and our fairshare burned to 0.0017,
# while jed academic had 576 idle cores and our RawUsage there is 0 (fairshare is PER-CLUSTER).
# A 72-core Ice Lake node runs ~64 envs against izar's 18, and the measured izar "GPU 1.9x faster"
# was core contention at 20 cores, not the GPU earning its keep (3% utilisation, 256x256 MLP).
#
# Submit:  sbatch --export=ALL,PRESET=walk_fwd_m2,STEPS=100000000,SEED=0,NAME=ladder3_m2_s0 \
#                 training/slurm/jed_train.sbatch
#
#SBATCH --job-name=dash-jed
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
# jed venv (Python 3.11, CPU-only torch) — built per training/slurm/README.md; NOT the izar venv,
# which carries a CUDA torch against izar's Python 3.9 module.
source "${VENV:-$HOME/venvs/dash-jed}/bin/activate"

# One BLAS thread per env worker: 64 workers on 72 cores must not each open a 72-thread pool.
# train.py gives the LEARNER its own torch.set_num_threads(LEARNER_THREADS) — workers never
# import torch, so the two settings do not fight.
export OMP_NUM_THREADS=1
export LEARNER_THREADS=8

REPO="${REPO:-$HOME/running_robot}"
PRESET="${PRESET:?set PRESET}"
STEPS="${STEPS:-100000000}"
# 64 x 288 = 18432 = the same PPO rollout as izar's 36x512 and 18x1024: bit-identical optimiser
# maths, only the worker count changes. 8 of the 72 cores stay free for the learner + OS.
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

python training/smoke_test.py                     # fail fast before burning node-days

python training/train.py --preset "$PRESET" --steps "$STEPS" --n-envs "$NENVS" \
     --subproc --resume auto --no-progress --name "$NAME" \
     ${WARM:+--warm-start "$WARM"} ${SEED:+--seed "$SEED"} ${NSTEPS:+--n-steps "$NSTEPS"}
