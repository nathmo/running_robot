#!/bin/bash -l
# THE FOOT-SHAPE LADDERS (2026-08-11) — jed.
#
#   bash training/slurm/foot_ladder.sh                       # blade + flat, seeds 0 and 1
#   DRY=1 bash training/slurm/foot_ladder.sh                 # print, submit nothing
#   FEET=blade SEEDS=0 STAGES="m2 m3" bash training/slurm/foot_ladder.sh
#
# WHY. Every controller this project has produced walks on a 25 mm point toe, which carries no
# moment about its contact: it has ZERO centre-of-pressure authority in any direction, so the only
# way it can arrest a lean is to move the foot, and the 4-bar cannot deliver that step in time
# (feet planted ~8 cm BEHIND the CoM while falling forward, 0.0 N on the swung foot through an
# entire fall). The full rebuilt-plant ladder now says where that costs the most: ladder3 clears m2
# (both seeds, one WALKS), drops to ~2 s at m3, and WALLS AT m5 — both seeds flat at ~400 training
# ep_len for the whole 80 M with no slope, while the retreating curriculum backed off and could not
# buy them anything. m5 is the rung that first unlocks ROLL.
#
# So: does a foot with a FOOTPRINT clear it? Two arms, chosen to decompose the question rather than
# to try the same idea twice:
#   blade  25 mm cylinder, 100 mm long, axis ACROSS the robot. Sagittal profile bit-identical to
#          the shipped ball (still rolls fore-aft, still carries no pitch moment); the point
#          contact becomes a 100 mm line. Buys LATERAL CoP and nothing else — aimed squarely at
#          the m5 roll wall. Also the running-blade geometry.
#   flat   30 x 100 x 10 mm plate. The blade's lateral CoP PLUS +-15 mm of sagittal, which is the
#          axis m3 is short of.
# sphere->blade is the roll increment, blade->flat the pitch increment on top. If the blade clears
# m5 and the plate does not add anything at m3, that is a clean and useful answer too.
#
# CONFOUND TO SETTLE BEFORE READING m5 EITHER WAY: the 3 Hz drive retune was measured only on the
# LEFT cam+thigh. Abduction — the axis m5 actually leans on — was left as the in-run control and
# still sits at 0.76 Hz, while the plant model claims 3 Hz uniform. Chirp abduction before calling
# any m5 result. m2..m4 lock roll and are unaffected.
#
# THE CONTROL ALREADY EXISTS and costs nothing: ladder3_m*_s{0,1} is this ladder, same reward, same
# obs (550), same action (26), same 3 Hz drive, same rigid carbon ankle, same curricula, differing
# ONLY in cfg.model_path. Do not re-run it. Its greedy numbers to beat (5 eps, dr=0, unpaired):
#   m2_s0 4622 / m2_s1 5589   m3_s0 385 / m3_s1 220   m4_s0 661 / m4_s1 222
#   m5_s0  126 / m5_s1  221   m6_s0 220 / m6_s1 218
#
# SHAPE. Cold at m2, each rung warm-starting the one below it by DIRECTORY (jed_train.sbatch
# resolves a dir to its newest checkpoint, so a rung killed by its wall clock still hands off).
# Rung k+1 depends on `afterany` of rung k. Identical budget to ladder3 — m2 gets 100 M because it
# is cold and has to find a gait from nothing, every rung above it 80 M.
#
# WARM-STARTING ACROSS FEET IS NOT ALLOWED HERE, even though the dims match. The plant is the
# independent variable; seeding the flat arm from a sphere policy would make "flat is better"
# unfalsifiable (it inherits ladder3's gait) and "flat is worse" uninterpretable (distribution
# shift). Cold at m2, same as the control.
set -euo pipefail

SB=${SB:-training/slurm/jed_train.sbatch}
FEET="${FEET:-blade flat}"
SEEDS="${SEEDS:-0 1}"
# m2..m5, not m2..m6. m5 is where the control walls and m6 measurably just inherits it (ladder3
# m6 372-399 against m5 385-404, both seeds), so a fourth rung of yaw would cost 4 more jobs to
# re-measure something already known. Extend with STAGES="m2 m3 m4 m5 m6" if m5 moves.
STAGES="${STAGES:-m2 m3 m4 m5}"
# 64 x 288 = 18432 = ladder3's PPO rollout exactly: identical optimiser maths, so the only thing
# that differs between these runs and the control is the foot.
NENVS="${NENVS:-64}"
NSTEPS="${NSTEPS:-288}"
STEPS="${STEPS:-80000000}"
STEPS_m2="${STEPS_m2:-100000000}"
TIME="${TIME:-1-00:00:00}"
DEP="${DEP:-}"                      # make each ladder's first rung wait on an existing job
DRY="${DRY:-}"

echo "=== foot-shape ladders (control: ladder3_*, already trained) ==="
echo "    feet:   $FEET"
echo "    seeds:  $SEEDS"
echo "    stages: $STAGES   (m2 ${STEPS_m2} steps, the rest ${STEPS}; ${NENVS} envs x ${NSTEPS})"
echo

for foot in $FEET; do
    for seed in $SEEDS; do
        prev_job=""
        prev_run=""
        for m in $STAGES; do
            preset="walk_fwd_${m}_${foot}"
            name="foot_${foot}_${m}_s${seed}"
            eval "steps=\${STEPS_${m}:-$STEPS}"
            export_str="ALL,PRESET=${preset},STEPS=${steps},NENVS=${NENVS},NSTEPS=${NSTEPS},SEED=${seed},NAME=${name}"
            [ -n "$prev_run" ] && export_str="${export_str},WARM=${prev_run}"

            dep=""
            if [ -n "$prev_job" ]; then
                dep="--dependency=afterany:${prev_job}"
            elif [ -n "$DEP" ]; then
                dep="--dependency=afterany:${DEP}"
            fi

            if [ -n "$DRY" ]; then
                echo "sbatch --job-name=$name --time=$TIME $dep --export=$export_str $SB"
                prev_job="<dry:${name}>"
                prev_run="training/runs/${name}"
                continue
            fi
            jid=$(sbatch --parsable --job-name="$name" --time="$TIME" $dep \
                         --export="$export_str" "$SB")
            echo "    $name  <- ${prev_run:-COLD}   job $jid${dep:+  ($dep)}"
            prev_job="$jid"
            prev_run="training/runs/${name}"
        done
    done
done

echo
echo "submitted. watch:  squeue -u \$USER -o '%.10i %.22j %.2t %.11M %R'"
echo "rungs to watch: m5 (roll — where ladder3 walls, and what the blade is aimed at)"
echo "                m3 (pitch — the ~2 s rung, and what the plate adds over the blade)"
