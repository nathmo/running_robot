#!/bin/bash -l
# v2 gate job: smoke suite + golden-trace parity + throughput benchmark + a 2 M-step training
# sanity run on one JED node. Reads the same env vars as jed_train.sbatch.
#SBATCH --job-name=dash-v2-bench
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
set -euo pipefail
module purge
module load gcc python
source "${VENV:-$HOME/venvs/dash-jed}/bin/activate"
export OMP_NUM_THREADS=1
export LEARNER_THREADS=8
cd "${REPO:-$HOME/running_robot}"
echo "== versions"; python -c "import mujoco,stable_baselines3,torch,numpy;print(mujoco.__version__,stable_baselines3.__version__,torch.__version__,numpy.__version__)"
echo "== smoke"; python walk_mit/smoke_test.py | tail -3
echo "== golden parity (recorded on the laptop, mujoco 3.7.0)"
python walk_mit/golden_v2.py --check walk_mit/golden/v2_s1_clean_seed0.npz || true
python walk_mit/golden_v2.py --preset v2_lib_s1 --check walk_mit/golden/v2_lib_s1_seed0.npz || true
echo "== bench (64 envs x 288, subproc)"
python walk_mit/bench_v2.py --n-envs "${NENVS:-64}" --n-steps "${NSTEPS:-288}" --subproc --rollouts 3
echo "== training sanity: ${STEPS:-2000000} steps"
python walk_mit/train.py --preset "${PRESET:-v2_s1}" --steps "${STEPS:-2000000}" --n-envs "${NENVS:-64}" \
     --n-steps "${NSTEPS:-288}" --subproc --no-progress --name "${NAME:-v2_bench}" ${SEED:+--seed "$SEED"}
tail -2 "walk_mit/runs/${NAME:-v2_bench}/progress.csv" | cut -c1-400
