#!/bin/bash -l
# Launch the m3..m6_wskill_gait WARM chain from the two-legged m3_sym checkpoint.
#
#   bash training/slurm/wskill_chain_launch.sh
#
# wskill = the sym plant + the HARD workspace-kill (terminate when a foot leaves the measured
# reachable box; see config.py mX_wskill_gait). sym_gait made m3 two-legged but the m4/m5/m6 DOF
# releases COLLAPSED (ep_len ~230) as the policy reverted to a one-legged crutch that parks a leg
# OUTSIDE the real workspace. This chain WARM-STARTS m3_wskill from the good two-legged m3_sym run
# (the kill barely bites there) and re-runs the m4->m6 ladder with the crutch made impossible — the
# test of whether the lateral/roll/yaw releases finally survive. afterany deps + WARM=<dir> hand-off
# + one post job per stage, same as the other chains. Watch m3_wskill's EARLY ep_len: if it holds,
# the kill box is well-calibrated (doesn't over-fire on the good gait); if it pins at ~10 steps, the
# box is too tight and needs loosening.
set -euo pipefail

SB=training/slurm/izar_train.sbatch
POST=training/slurm/cpg_stage_post.sbatch
NENVS="${NENVS:-18}"
SEED="${SEED:-0}"
WARM0="${WARM0:-training/runs/m3_sym_gait}"   # the two-legged base m3_wskill warm-starts from

steps_for() { case "$1" in m6) echo 180000000;; *) echo 150000000;; esac; }
time_for()  { case "$1" in m3) echo 20:00:00;; m6) echo 18:00:00;; *) echo 16:00:00;; esac; }

echo "=== m3..m6_wskill_gait WARM chain (workspace-kill), warm0=$WARM0, seed $SEED, $NENVS envs ==="
prev_train=""
prev_run=""
for m in m3 m4 m5 m6; do
    run="${m}_wskill_gait"
    st=$(steps_for "$m"); tl=$(time_for "$m")
    args=(--parsable --time="$tl" --job-name="$run")
    exp="ALL,PRESET=${m}_wskill_gait,STEPS=$st,NENVS=$NENVS,NAME=$run,SEED=$SEED"
    if [ -n "$prev_train" ]; then
        args+=(--dependency="afterany:$prev_train")
        exp="$exp,WARM=training/runs/$prev_run"
        warmfrom="$prev_run"
    else
        exp="$exp,WARM=$WARM0"
        warmfrom="$WARM0"
    fi
    jid=$(sbatch "${args[@]}" --export="$exp" "$SB")
    pid=$(sbatch --parsable --dependency="afterany:$jid" --job-name="post_$run" \
                 --export="ALL,RUN=$run" "$POST")
    echo "  $run  train $jid (warm <- $warmfrom)   post $pid   $st steps / $tl"
    prev_train=$jid
    prev_run=$run
done

echo
squeue -u "$USER" -o "%.10i %.16j %.9T %.11l %R"
