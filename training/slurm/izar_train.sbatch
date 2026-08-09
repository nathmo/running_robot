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
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
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
# Izar nodes are 40 cores / 192 GB / 2x V100. We used to book 20 cores + 1 GPU and MEASURED
# 3% GPU utilisation with a load average of 8-10 -- i.e. half a node reserved, a quarter of it
# used, and the GPU idle. Book the cores we actually need instead.
# NSTEPS moves with NENVS to hold the PPO rollout (n_steps*n_envs) CONSTANT: 18x1024 and 36x512
# are both 18432, so this is a pure throughput change and the learning dynamics are untouched.
NENVS="${NENVS:-36}"          # leave ~4 of the 40 cores for the learner + OS
NSTEPS="${NSTEPS:-512}"       # 36*512 == the old 18*1024
NAME="${NAME:-$PRESET}"
WARM="${WARM:-}"              # optional: checkpoint to warm-start a NEW milestone stage from,
                              # e.g. WARM=training/runs/m1_sprint/final_model.zip
SEED="${SEED:-}"              # optional: override cfg.seed (seed replicates of an A/B arm)

cd "$REPO"

# WARM may name a run DIRECTORY instead of a .zip. That is what makes a wall-clock-bounded stage
# chain safely: a stage stopped by its time limit never writes final_model.zip, so the next stage
# resolves the directory to whatever checkpoint the previous one actually reached.
if [ -n "$WARM" ] && [ -d "$WARM" ]; then
    # `|| true` is load-bearing: a stage stopped by its wall clock has NO final_model.zip, `ls` then
    # exits 2 for the missing operand, and `set -o pipefail` propagates that through `head` so
    # `set -e` kills the job before it prints anything (this is exactly how ab_cpg_wide_m3warm_s0
    # died in 2 s). Listing both patterns is still right â€” final_model.zip must win when it exists.
    RESOLVED=$({ ls -1t "$WARM"/final_model.zip "$WARM"/ppo_*_steps.zip 2>/dev/null || true; } \
               | head -1)
    if [ -z "$RESOLVED" ]; then
        echo "[sbatch] FATAL: WARM=$WARM is a directory with no checkpoint in it" >&2
        exit 1
    fi
    echo "[sbatch] WARM directory $WARM -> $RESOLVED"
    WARM="$RESOLVED"
fi
python training/smoke_test.py                     # fail fast before burning GPU hours

# --resume auto: starts fresh if no checkpoint exists, else continues â€” the SAME command is
# correct for the first job and every chained/requeued job. --warm-start only applies on a
# fresh start (train.py ignores it when a resume checkpoint is found), so WARM is chain-safe.
# (plain python, no srun: single-task job â€” srun adds nothing and on newer SLURM a job step
# does not inherit --cpus-per-task, which can silently confine the workers to 1 core)
python training/train.py --preset "$PRESET" --steps "$STEPS" --n-envs "$NENVS" \
     --subproc --resume auto --no-progress --name "$NAME" \
     ${WARM:+--warm-start "$WARM"} ${SEED:+--seed "$SEED"} ${NSTEPS:+--n-steps "$NSTEPS"}
