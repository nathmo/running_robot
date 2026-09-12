#!/bin/bash -l
# DASH-01 Walker v2 training on SCITAS Izar: ONE V100 (half a node: 20 cores, 94 GB), JAX + MJX.
#
# Submit:   sbatch --export=ALL,PRESET=v3,NAME=v3_s0,SEED=0,DEVICES=2,NENVS=4096,NSTEPS=9 \
#                  walk_v3/slurm/izar_train.sbatch
# Chain:    walk_v3/slurm/izar_chain.sh 3 --export=ALL,PRESET=...   (each link: --resume auto)
#
# One GPU = half an Izar node; asking for more cores strands the node's other GPU (SCITAS asked
# us not to, 2026-08-10). The V100 (sm_70, 32 GB) runs the pip CUDA-12 jax wheels on the 535
# driver; NENVS 1024 x NSTEPS 18 = the contract's 18 432-sample rollout (64 x 288 on the CPU arm).
#SBATCH --job-name=dash-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:30:00
#SBATCH --output=walk_v3/logs/%x-%j.out

set -euo pipefail
module purge
export PATH=$HOME/.local/bin:$PATH
source "${VENV:-$HOME/venvs/dash-v2}/bin/activate"

REPO="${REPO:-$HOME/running_robot}"
PRESET="${PRESET:?set PRESET}"
# EMPTY by default: the preset owns the budget. A hard default here silently overrode it --
# the 40 M probe presets all launched as 300 M runs.
STEPS="${STEPS:-}"
NENVS="${NENVS:-1024}"
NSTEPS="${NSTEPS:-18}"
NAME="${NAME:-$PRESET}"
WARM="${WARM:-}"
SEED="${SEED:-}"
EVAL_ENVS="${EVAL_ENVS:-16}"
DEVICES="${DEVICES:-1}"        # data-parallel GPUs; submit with --gres=gpu:$DEVICES (--partition gpu-xl for 4)

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

python walk_v3/smoke_test.py --quick

python walk_v3/train.py --preset "$PRESET" ${STEPS:+--steps "$STEPS"} --n-envs "$NENVS" --n-steps "$NSTEPS" \
    --resume auto --name "$NAME" --eval-envs "$EVAL_ENVS" --devices "$DEVICES" \
    ${WARM:+--warm-start "$WARM"} ${SEED:+--seed "$SEED"}
