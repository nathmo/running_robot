#!/bin/bash -l
# BalanceRL on SCITAS Izar: 2 x V100 (one node), JAX + MJX, data-parallel pmap.
# Launch ONLY through izar_balance.sh (it sets REPO, the chain and the seeds).
#SBATCH --job-name=balance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --gres=gpu:2
#SBATCH --time=04:30:00
#SBATCH --output=BalanceRL/logs/%x-%j.out

set -euo pipefail
module purge
export PATH=$HOME/.local/bin:$PATH
source "${VENV:-$HOME/venvs/dash-v2}/bin/activate"

REPO="${REPO:?set REPO}"
NAME="${NAME:?set NAME}"
SEED="${SEED:-0}"
DEVICES="${DEVICES:-2}"
NENVS="${NENVS:-4096}"
NSTEPS="${NSTEPS:-24}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export JAX_ENABLE_X64=0
export MUJOCO_GL=egl

cd "$REPO"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import jax; print('jax', jax.__version__, jax.devices())"
python BalanceRL/train.py --preset "${PRESET:-balance}" --name "$NAME" --seed "$SEED" \
    --devices "$DEVICES" --n-envs "$NENVS" --n-steps "$NSTEPS" --resume auto ${EXTRA:-}
