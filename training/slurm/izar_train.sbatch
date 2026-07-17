#!/bin/bash -l
# DASH-01 sprint training on SCITAS Izar (1 node, 1x V100, 20 cores).
#
# Submit:            sbatch training/slurm/izar_train.sbatch
# With overrides:    sbatch --export=ALL,PRESET=m2_sprint,STEPS=240000000,NENVS=18 \
#                           training/slurm/izar_train.sbatch
# Chain (>3 days):   training/slurm/izar_chain.sh 2      (each job resumes where the last stopped
#                                                         thanks to --resume auto)
#
# QOS: default `gpu` = 3-day limit. For one 7-day job add:  #SBATCH --qos=long  and raise --time.
#SBATCH --job-name=dash-sprint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module purge
module load gcc python
# venv created per training/slurm/README.md (login node, --system-site-packages)
source "${VENV:-$HOME/venvs/dash}/bin/activate"

REPO="${REPO:-$HOME/running_robot}"
PRESET="${PRESET:-m2_sprint}"
STEPS="${STEPS:-240000000}"
NENVS="${NENVS:-18}"          # leave ~2 of the 20 cores for the learner + OS
NAME="${NAME:-$PRESET}"
WARM="${WARM:-}"              # optional: checkpoint to warm-start a NEW milestone stage from,
                              # e.g. WARM=training/runs/m1_sprint/final_model.zip

cd "$REPO"
python training/smoke_test.py                     # fail fast before burning GPU hours

# --resume auto: starts fresh if no checkpoint exists, else continues — the SAME command is
# correct for the first job and every chained/requeued job. --warm-start only applies on a
# fresh start (train.py ignores it when a resume checkpoint is found), so WARM is chain-safe.
# (plain python, no srun: single-task job — srun adds nothing and on newer SLURM a job step
# does not inherit --cpus-per-task, which can silently confine the workers to 1 core)
python training/train.py --preset "$PRESET" --steps "$STEPS" --n-envs "$NENVS" \
     --subproc --resume auto --no-progress --name "$NAME" \
     ${WARM:+--warm-start "$WARM"}
