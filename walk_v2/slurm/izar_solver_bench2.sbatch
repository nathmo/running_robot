#!/bin/bash -l
# Izar debug-QOS: the MJX batched-solver recipe (1-2 Newton iterations x 4 line-search per 1 kHz substep,
# warm-started): drift on a held stance (planar plant) + the real env AND a PPO iteration per cap.
#   sbatch walk_v2/slurm/izar_solver_bench2.sbatch
#SBATCH --job-name=dash-v2-solver2
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
python walk_v2/tools/profile_step.py --xml walk_v2/model/dash01_v2_planar.xml --n-envs 2048 --check-ticks 300 --reps 5 \
    --variants xml newton1_ls4 newton2_ls4 newton4_ls4 newton8 \
    --json "walk_v2/results/profile_step_izar_planar_${SLURM_JOB_ID}.json"
for it in "1 4" "2 4" "4 4"; do
    set -- $it
    python walk_v2/bench.py --n-envs 2048 --steps 64 --ppo --iterations $1 --ls-iterations $2 \
        --json "walk_v2/results/bench_izar_solver_${SLURM_JOB_ID}_${1}_${2}.json"
done
echo "[solver2] done"
