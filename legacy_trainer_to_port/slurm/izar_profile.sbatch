#!/bin/bash -l
# Izar debug-QOS profile of the bare MJX physics under solver-setting variants (tools/profile_step.py).
#   sbatch legacy_trainer_to_port/slurm/izar_profile.sbatch
#SBATCH --job-name=dash-v3-prof
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module purge
export PATH=$HOME/.local/bin:$PATH
source "${VENV:-$HOME/venvs/dash-v2}/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
cd "${REPO:-$HOME/running_robot}"
nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p legacy_trainer_to_port/results
python legacy_trainer_to_port/tools/profile_step.py --n-envs 256 2048 4096 --check-ticks 200 --reps 5 \
    --json "legacy_trainer_to_port/results/profile_step_izar_${SLURM_JOB_ID}.json"
python legacy_trainer_to_port/tools/profile_step.py --n-envs 2048 --variants xml newton8 --batched-timestep --check-ticks 0 \
    --json "legacy_trainer_to_port/results/profile_step_izar_${SLURM_JOB_ID}_bts.json"
echo "[profile] done"
