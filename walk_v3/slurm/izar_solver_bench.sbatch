#!/bin/bash -l
# Izar debug-QOS: MJX solver-cap study. Bare physics timing + drift per variant (tools/profile_step.py),
# then the real env under random actions (bench.py) for each cap — the state-dependent cost.
#   sbatch walk_v3/slurm/izar_solver_bench.sbatch
#SBATCH --job-name=dash-v3-solver
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
mkdir -p walk_v3/results
python walk_v3/tools/profile_step.py --n-envs 2048 --check-ticks 200 --reps 5 \
    --variants xml newton100_ls8 newton32_ls8 newton16_ls8 newton16 newton8 \
    --json "walk_v3/results/profile_step_izar_${SLURM_JOB_ID}.json"
for it in "100 50" "100 8" "32 8" "16 8" "8 8"; do
    set -- $it
    python walk_v3/bench.py --n-envs 2048 --steps 64 --iterations $1 --ls-iterations $2 \
        --json "walk_v3/results/bench_izar_solver_${SLURM_JOB_ID}_${1}_${2}.json"
done
echo "[solver] done"
