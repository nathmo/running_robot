#!/bin/bash -l
# CPG (Ijspeert) vs Fourier generator A/B — submit the whole campaign on Izar.
#
#   bash training/slurm/ab_cpg_launch.sh [STAGE1_HOURS] [STAGE2_HOURS]
#
# The design (see the "CPG vs Fourier" block in training/config.py): one plant, one reward, one
# schedule; only the gait generator changes. Three arms —
#   f      : fourier_gait, the m1..m7 generator (the baseline)
#   cpg    : cpg_gait + the 6-dim PMTG residual channel (authority-matched to the baseline)
#   cpg_nr : cpg_gait with NO residual channel (the pure oscillator)
#
# and two lineages per arm, because "is the CPG better" has two honest readings:
#   COLD  : one job straight at m3 (pitch free) for the whole budget — sample efficiency from zero.
#   WARM  : cold at m2 (pitch locked), then m3 warm-started from it — the way the Fourier lineage
#           was actually built (m3_stiff_hi warm-started from a 34 M-step m2_reactive checkpoint).
#
# Every job is wall-clock bounded rather than step bounded, and --steps is set far above what the
# wall allows ON PURPOSE: the arms are then compared at the smallest step count all of them reached,
# which is the only apples-to-apples read when throughput differs between arms. The stage-2 jobs
# depend on stage 1 with `afterany` and warm-start from the stage-1 run DIRECTORY, so a stage that
# is killed by its time limit (and so never writes final_model.zip) still chains correctly.
set -euo pipefail

S1H="${1:-5}"        # stage-1 (cold m2) hours
S2H="${2:-5}"        # stage-2 (warm m3) hours
COLDH="${COLDH:-10}" # cold-m3 hours (runs the whole budget in one job)
STEPS="${STEPS:-400000000}"
NENVS="${NENVS:-18}"
SB="training/slurm/izar_train.sbatch"

sub() {  # sub <name> <preset> <hours> <seed> [warm-dir] [dependency-jobid]
    local name=$1 preset=$2 hours=$3 seed=$4 warm=${5:-} dep=${6:-}
    local args=(--parsable --time="${hours}:00:00" --job-name="$name")
    [ -n "$dep" ] && args+=(--dependency="afterany:$dep")
    local exp="ALL,PRESET=$preset,STEPS=$STEPS,NENVS=$NENVS,NAME=$name,SEED=$seed"
    [ -n "$warm" ] && exp="$exp,WARM=$warm"
    sbatch "${args[@]}" --export="$exp" "$SB"
}

echo "=== cold m3 (whole budget in one job) ==="
for a in "f:ab_f_m3" "cpg:ab_cpg_m3" "cpg_nr:ab_cpg_nr_m3"; do
    arm=${a%%:*}; preset=${a##*:}
    id=$(sub "ab_${arm}_cold_s0" "$preset" "$COLDH" 0)
    echo "  ab_${arm}_cold_s0  job $id"
done

echo "=== warm lineage: stage 1 (cold m2) -> stage 2 (m3, warm) ==="
# two seeds on the two headline arms (f, cpg) to guard against seed luck; one on the ablation
for spec in "f:ab_f_m2:ab_f_m3:0" "f:ab_f_m2:ab_f_m3:1" \
            "cpg:ab_cpg_m2:ab_cpg_m3:0" "cpg:ab_cpg_m2:ab_cpg_m3:1" \
            "cpg_nr:ab_cpg_nr_m2:ab_cpg_nr_m3:0"; do
    IFS=: read -r arm p2 p3 seed <<< "$spec"
    n2="ab_${arm}_m2_s${seed}"
    n3="ab_${arm}_m3warm_s${seed}"
    j2=$(sub "$n2" "$p2" "$S1H" "$seed")
    j3=$(sub "$n3" "$p3" "$S2H" "$seed" "training/runs/$n2" "$j2")
    echo "  $n2  job $j2   ->   $n3  job $j3"
done

echo
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.10l %R"
