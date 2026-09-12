#!/bin/bash -l
# Lyra beta-test benchmark: MJX env throughput vs batch size + one PPO iteration, on ONE GPU.
#
#   sbatch walk_v3/slurm/lyra_bench.sbatch                       # rtx6000
#   sbatch --partition=b200 --cpus-per-task=12 --mem=120G walk_v3/slurm/lyra_bench.sbatch
#
# Writes walk_v3/results/bench_<partition>_<jobid>.json; compare against the CPU stack with
# walk_v3/tools/compare_cpu_gpu.py (throughput) and the paired-policy check described there.
#SBATCH --job-name=dash-v3-bench
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module purge
module load gcc/14.3.0 python/3.12.12
source "${VENV:-$HOME/venvs/dash-v2}/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
cd "${REPO:-$HOME/running_robot}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import jax; print('jax', jax.__version__, jax.devices())"
mkdir -p walk_v3/results
python walk_v3/bench.py --ppo --n-envs 512 1024 2048 4096 8192 \
    --json "walk_v3/results/bench_${SLURM_JOB_PARTITION}_${SLURM_JOB_ID}.json"
