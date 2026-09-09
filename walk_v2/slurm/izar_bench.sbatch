#!/bin/bash -l
# Izar (V100) check + benchmark on the free debug QOS (1 h): full smoke test on the GPU, then
# env steps/s vs batch size and one PPO iteration per size.
#
#   sbatch walk_v2/slurm/izar_bench.sbatch
#SBATCH --job-name=dash-v2-bench
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
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import jax; print('jax', jax.__version__, jax.devices())"
python walk_v2/smoke_test.py
mkdir -p walk_v2/results
python walk_v2/bench.py --ppo --n-envs 256 1024 2048 4096 \
    --json "walk_v2/results/bench_izar_v100_${SLURM_JOB_ID}.json"
