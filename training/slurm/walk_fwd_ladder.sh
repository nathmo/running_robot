#!/bin/bash -l
# THE walk_fwd BASE-DOF LADDER (2026-08-09).
#
#   bash training/slurm/walk_fwd_ladder.sh                 # m2 -> m3 -> m4 -> m5 -> m6, seed 0
#   DRY=1 bash training/slurm/walk_fwd_ladder.sh           # print, submit nothing
#   SEED=1 STAGES="m5 m6" bash training/slurm/walk_fwd_ladder.sh
#
# WHY. Every run in the teleop -> walk_fwd lineage trains at m6 -- all six base DOFs free -- from
# step 0. Checked against every resolved_config in the archive: teleop, teleop_easy, v2, v3, v5,
# walk_fwd, walk_fwd_easy, walk_fwd2 are ALL base_lock (0,0,0,0,0,0). It is the only lineage here
# that skipped the ladder and the only one that will not converge: three plateaus, ~1.4 G steps,
# dr_scale self-limiting at 0.27 because the retreating ramp cannot find competence to ramp on.
#
# Everything that ever worked on this robot used the ladder. 97 archived runs sit at m3, and the
# only controller to reach 2.55 m/s on the MEASURED plant (ankle2_m3_rigid) was m3 with roll and
# yaw locked -- 15.5 s survival in 73 M steps, against 300 M spent going nowhere at m6.
#
# SHAPE. Each rung differs from the next by base_lock and nothing else, so obs (550) and action
# (26) are identical the whole way up and each stage warm-starts the one below by DIRECTORY
# (izar_train.sbatch resolves a dir to its newest checkpoint, so a stage killed by its wall clock
# still hands off). Stage k+1 depends on `afterany` of stage k's LAST link.
#
#   m2  x,z free, y/roll/PITCH/yaw locked - it cannot fall, so the only thing to learn is how
#       the gait converts into forward speed. That is the measured deficit: the m6 policy tops
#       out at 0.14 m/s against a 0.6 m/s command and survives 0/12 episodes at +0.40. Kept
#       SHORT on purpose (STEPS_m2/LINKS_m2) -- with pitch locked a policy can lunge with no
#       consequence, and m2 -> m3 is the hard transition in this project's history.
#   m3  + pitch. y/roll/yaw locked - sagittal, the configuration that has actually worked
#   m4  + lateral translation
#   m5  + ROLL          <- THE RUNG TO WATCH. The CPG A/B walled here at matched budget.
#   m6  + yaw           == walk_fwd3 exactly, so it is a clean comparison against the control run
#
# One seed: this is a STRUCTURAL probe (does it clear m3? where does it wall?), not an A/B. Add
# seeds once the ladder tells us which rung is the problem.
set -euo pipefail

SB=training/slurm/izar_train.sbatch
STAGES="${STAGES:-m2 m3 m4 m5 m6}"
SEED="${SEED:-0}"
# Full node: Izar is 40 cores / 2 GPUs, and the old 20-core booking measured 3% GPU use at load
# 8-10. NSTEPS moves with NENVS so the PPO rollout stays 18432 and only throughput changes.
NENVS="${NENVS:-36}"
NSTEPS="${NSTEPS:-512}"
STEPS="${STEPS:-80000000}"          # per stage; ankle2_m3_rigid reached 15.5 s in 73 M
LINKS="${LINKS:-3}"                 # 4 h each -> 12 h of container per stage; a stage that hits
                                    # STEPS early just exits and its spare links exit immediately
TIME="${TIME:-4:00:00}"
# Per-stage overrides: STEPS_m2 / LINKS_m2 etc. m2 is COLD and has to learn a gait from nothing,
# so it gets the largest budget of any rung. The counter-pressure is real -- with pitch locked a
# policy can lunge with no consequence, and m2 -> m3 is the hard transition in this project's
# history (the whole m3 anti-topple sweep) -- so this is a budget worth revisiting once we see
# where m2 saturates.
STEPS_m2="${STEPS_m2:-100000000}"
LINKS_m2="${LINKS_m2:-4}"
# Rung 1 trains COLD by default. Warm-starting the m6 walker DOWN into m2 was measured to be
# strictly harmful -- same walk_fwd_easy_s0 policy, only base_lock changed, 8 paired episodes:
#     m6 (as trained) 8.5 s median, 3/8 full   m5 2.5 s, 0/8
#     m4 1.1 s, 0/8   m3 1.1 s, 0/8            m2 1.0 s, 0/8
# Every "easier" rung is HARDER for it, monotonically. The policy balances using the DOFs the
# lower rungs remove, so locking them is a distribution shift, not a simplification: at m2 it does
# not topple (it cannot) -- it sinks, base z 1.010 -> 0.451 against a 0.45 term_height, in ~1 s.
# The ladder only works in the UP direction: cold at the bottom, each rung warm-starting the one
# below it. Set WARM0=<dir> to override.
WARM0="${WARM0:-}"
# DEP=<jobid>: make rung 1 wait on an existing job. Used to graft later rungs onto a stage that
# is ALREADY RUNNING (resource changes only take effect on newly submitted jobs -- SLURM snapshots
# the batch script at submit time, so re-submitting is the only way to apply a new core count).
DEP="${DEP:-}"
DRY="${DRY:-}"

echo "=== walk_fwd ladder, seed $SEED ==="
echo "    stages: $STAGES"
echo "    default ${LINKS}x${TIME} per stage, ${STEPS} steps, ${NENVS} envs x ${NSTEPS} n_steps (m2: ${LINKS_m2}x, ${STEPS_m2})"
echo "    rung 1: ${WARM0:-COLD (from scratch)}"
echo

prev_last=""        # last job id of the previous stage
prev_run=""         # previous stage's run dir, the warm start for this one
for m in $STAGES; do
    preset="walk_fwd_${m}"
    name="ladder_${m}_s${SEED}"
    warm="${prev_run:-$WARM0}"   # empty on rung 1 unless WARM0 is set = COLD
    # per-stage budget, e.g. STEPS_m2 / LINKS_m2, falling back to the global default
    eval "steps=\${STEPS_${m}:-$STEPS}"
    eval "links=\${LINKS_${m}:-$LINKS}"
    export_str="ALL,PRESET=${preset},STEPS=${steps},NENVS=${NENVS},NSTEPS=${NSTEPS},SEED=${SEED},NAME=${name}"
    [ -n "$warm" ] && export_str="${export_str},WARM=${warm}"

    dep=""
    if [ -n "$prev_last" ]; then
        dep="--dependency=afterany:${prev_last}"
    elif [ -n "$DEP" ]; then
        dep="--dependency=afterany:${DEP}"
    fi

    echo "--- $name   (preset $preset, warm <- ${warm:-COLD}, ${links}x${TIME}, ${steps} steps)"
    if [ -n "$DRY" ]; then
        echo "    sbatch --job-name=$name --time=$TIME $dep --export=$export_str $SB   x$links"
        prev_last="<dry:${name}>"
        prev_run="training/runs/${name}"
        continue
    fi

    jid=$(sbatch --parsable --job-name="$name" --time="$TIME" $dep \
                 --export="$export_str" "$SB")
    echo "    link 1: $jid${dep:+  ($dep)}"
    for i in $(seq 2 "$links"); do
        jid=$(sbatch --parsable --job-name="$name" --time="$TIME" \
                     --dependency=afterany:"$jid" --export="$export_str" "$SB")
        echo "    link $i: $jid (afterany)"
    done
    prev_last="$jid"
    prev_run="training/runs/${name}"
done

echo
echo "submitted. watch:  squeue -u \$USER -o '%.10i %.18j %.2t %.11M'"
echo "the rung to watch is m5 (roll)."
