#!/bin/bash -l
# DASH-01 Walker v2 training on SCITAS Izar: ONE V100 (half a node: 20 cores, 94 GB), JAX + MJX.
#
# Submit:   sbatch --export=ALL,PRESET=v2_s1_planar,STEPS=300000000,NAME=v2_s1_planar_s0,SEED=0 \
#                  walk_v2/slurm/izar_train.sbatch
# Chain:    walk_v2/slurm/izar_chain.sh 3 --export=ALL,PRESET=...   (each link: --resume auto)
#
# One GPU = half an Izar node; asking for more cores strands the node's other GPU (SCITAS asked
# us not to, 2026-08-10). The V100 (sm_70, 32 GB) runs the pip CUDA-12 jax wheels on the 535
# driver; NENVS 2048 x NSTEPS 64 keeps the same 131 072-sample rollout as the Lyra sizing.
#SBATCH --job-name=dash-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module purge
export PATH=$HOME/.local/bin:$PATH
source "${VENV:-$HOME/venvs/dash-v2}/bin/activate"

REPO="${REPO:-$HOME/running_robot}"
PRESET="${PRESET:?set PRESET}"
STEPS="${STEPS:-300000000}"
NENVS="${NENVS:-2048}"
NSTEPS="${NSTEPS:-64}"
NAME="${NAME:-$PRESET}"
WARM="${WARM:-}"
SEED="${SEED:-}"
EVAL_ENVS="${EVAL_ENVS:-16}"

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

python walk_v2/smoke_test.py --quick

python walk_v2/train.py --preset "$PRESET" --steps "$STEPS" --n-envs "$NENVS" --n-steps "$NSTEPS" \
    --resume auto --name "$NAME" --eval-envs "$EVAL_ENVS" \
    ${WARM:+--warm-start "$WARM"} ${SEED:+--seed "$SEED"}
