#!/usr/bin/env bash
# Server training entry-point. Run inside tmux:
#   tmux new -s train
#   bash run_train.sh
# Detach with Ctrl+b d, reattach with `tmux attach -t train`.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Activate the project venv. Adjust the path here if yours lives elsewhere.
VENV_PATH="${VENV_PATH:-$REPO_DIR/.venv}"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "venv not found at $VENV_PATH — set VENV_PATH=... or create one:"
    echo "    python3 -m venv $VENV_PATH && source $VENV_PATH/bin/activate && pip install -r RL/requirements_rl.txt"
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

# Pin to GPU 0 (harmless on CPU-only, keeps behavior consistent).
export CUDA_VISIBLE_DEVICES=0

# Match n_envs to the box's core count to avoid oversubscription.
# Override from the environment if you want something else: N_ENVS=4 bash run_train.sh
N_ENVS="${N_ENVS:-8}"
N_STEPS="${N_STEPS:-4096}"
N_EPOCHS="${N_EPOCHS:-500}"
PRESET="${PRESET:-fast_training}"
VARIANT="${VARIANT:-gpu_mjx}"

python RL/train.py \
    --preset "$PRESET" \
    --variant "$VARIANT" \
    --n-envs "$N_ENVS" \
    --n-steps "$N_STEPS" \
    --n-epochs "$N_EPOCHS" \
    "$@"
