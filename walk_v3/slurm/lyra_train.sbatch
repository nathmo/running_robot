#!/bin/bash -l
# DASH-01 Walker v2 training on SCITAS Lyra: ONE GPU, JAX + MJX, thousands of envs in one process.
#
# Submit:   sbatch --export=ALL,PRESET=v2_s1_planar,STEPS=300000000,NAME=v2_s1_planar_s0,SEED=0 \
#                  walk_v3/slurm/lyra_train.sbatch
# B200:     sbatch --partition=b200 --cpus-per-task=12 --mem=180G --export=ALL,... walk_v3/slurm/lyra_train.sbatch
# Chain:    walk_v3/slurm/lyra_chain.sh 3 --export=ALL,PRESET=...   (each link resumes: --resume auto)
#
# Sizing per the Lyra announcement: rtx6000 nodes are 8 GPU / 128 cores / 768 GB -> 16 cores
# and ~90 GB per GPU; b200 nodes are 8 GPU / 96 cores / 1.5 TB -> 12 cores and ~190 GB per GPU.
# The partition is MANDATORY on Lyra. Lyra is in beta: jobs can be killed at short notice, so
# checkpoints land every cfg.checkpoint_every_steps (5 M) and the job is requeue-safe.
#SBATCH --job-name=dash-v3
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=2-23:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module purge
module load gcc/14.3.0 python/3.12.12
source "${VENV:-$HOME/venvs/dash-v2}/bin/activate"

REPO="${REPO:-$HOME/running_robot}"
PRESET="${PRESET:?set PRESET}"
STEPS="${STEPS:-300000000}"
NENVS="${NENVS:-1024}"
NSTEPS="${NSTEPS:-18}"
NAME="${NAME:-$PRESET}"
WARM="${WARM:-}"
SEED="${SEED:-}"
EVAL_ENVS="${EVAL_ENVS:-16}"

# JAX: don't grab 75% of the GPU up front (lets a bench share it); MJX likes 32-bit
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export JAX_ENABLE_X64=0
export MUJOCO_GL=egl

cd "$REPO"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import jax; print('jax', jax.__version__, jax.devices())"

if [ -n "$WARM" ] && [ -d "$WARM" ]; then
    RESOLVED=$({ ls -1t "$WARM"/final.msgpack "$WARM"/ckpt_*.msgpack 2>/dev/null || true; } | head -1)
    if [ -z "$RESOLVED" ]; then
        echo "[sbatch] FATAL: WARM=$WARM has no checkpoint" >&2
        exit 1
    fi
    echo "[sbatch] WARM directory $WARM -> $RESOLVED"
    WARM="$RESOLVED"
fi

python walk_v3/smoke_test.py --quick            # fail fast before burning GPU hours

python walk_v3/train.py --preset "$PRESET" --steps "$STEPS" --n-envs "$NENVS" --n-steps "$NSTEPS" \
    --resume auto --name "$NAME" --eval-envs "$EVAL_ENVS" \
    ${WARM:+--warm-start "$WARM"} ${SEED:+--seed "$SEED"}
