#!/bin/bash -l
# Izar debug-QOS: solver caps that keep the golden replay intact, timed with the FIXED bench (env under
# random actions + one PPO iteration each) at 2048 envs.
#   sbatch walk_v2/slurm/izar_solver_bench3.sbatch
#SBATCH --job-name=dash-v2-solver3
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
mkdir -p walk_v2/results
for it in "100 50" "100 8" "32 8" "16 8" "8 8" "4 4"; do
    set -- $it
    python walk_v2/bench.py --n-envs 2048 --steps 64 --ppo --iterations $1 --ls-iterations $2 \
        --json "walk_v2/results/bench3_izar_solver_${SLURM_JOB_ID}_${1}_${2}.json"
done
echo "[solver3] done"
