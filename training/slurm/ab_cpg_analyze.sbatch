#!/bin/bash -l
# Post-training analysis for the CPG-vs-Fourier A/B: greedy eval + cadence diagnostics for every
# ab_* run, plus the common-step comparison table and learning-curve plot.
#
#   sbatch --dependency=afterany:<all training job ids> training/slurm/ab_cpg_analyze.sbatch
#
# Deliberately ONE job over all runs rather than one job per run: these are short, and a single
# stdout is what gets read in the morning. Every run is wrapped so one bad checkpoint cannot abort
# the rest — a missing arm in the report is worse than a noisy line.
#SBATCH --job-name=ab-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out

set -uo pipefail        # NOT -e: a single failing run must not kill the report

module purge
module load gcc python
source "${VENV:-$HOME/venvs/dash}/bin/activate"
cd "${REPO:-$HOME/running_robot}"

echo "########## common-step comparison ##########"
python training/compare_ab.py || echo "[analyze] compare_ab FAILED"
for lin in cold m2 m3warm; do
    python training/compare_ab.py --lineage $lin \
        --out training/runs/ab_compare_$lin.png >/dev/null 2>&1 \
        && echo "[analyze] wrote training/runs/ab_compare_$lin.png"
done

for run in training/runs/ab_*; do
    [ -d "$run" ] || continue
    name=$(basename "$run")
    # a run killed by its wall clock has no final_model.zip; evaluate.py falls back to the newest
    # checkpoint, but skip directories that never wrote one at all
    ls "$run"/final_model.zip "$run"/ppo_*_steps.zip >/dev/null 2>&1 || {
        echo "########## $name: NO CHECKPOINT, skipped ##########"; continue; }
    echo
    echo "########## $name ##########"
    python training/evaluate.py --run "$run" --episodes 5 2>&1 | tail -12 \
        || echo "[analyze] evaluate FAILED for $name"
    # the cadence read is the point of the cpg_nr arm: does the oscillator's amplitude filter
    # remove the ~6 Hz footfall limit cycle that no m7 lever could touch?
    python training/diagnose_cadence.py --run "$run" --episodes 4 2>&1 | tail -16 \
        || echo "[analyze] diagnose_cadence FAILED for $name"
done

echo
echo "########## done ##########"
