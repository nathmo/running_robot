#!/bin/bash -l
# Resume the m1..m6_CPG ladder from a stage that already succeeded, re-running the rest.
#
#   bash training/slurm/cpg_chain_resume.sh m3 [HOURS] [FIRST_STAGE_HOURS]
#
# Why this exists: the first run of the chain got a good m2_CPG (100 m dash, ep_len 10987) and then
# lost m3..m6, because the clock-driven stance ramp demanded a flight phase at 19.6 M steps while
# m3's ep_len was 257. curriculum_gate_ep_len now holds that ramp until the policy is competent,
# so m3 onward is worth re-running from the m2 that worked rather than from scratch.
#
# The already-trained runs for the stages being redone are MOVED ASIDE (suffix _clockramp) rather
# than overwritten: they are the evidence for why the gate is needed, and train.py would otherwise
# resume straight into them via --resume auto.
set -euo pipefail

FROM="${1:-m3}"
H="${2:-4.5}"
H1="${3:-$H}"          # the first redone stage is the wall (m3 = pitch unlock); give it more time
SB=training/slurm/izar_train.sbatch
POST=training/slurm/cpg_stage_post.sbatch
STEPS="${STEPS:-60000000}"
NENVS="${NENVS:-18}"
SEED="${SEED:-0}"

ALL=(m1 m2 m3 m4 m5 m6)
start=-1
for i in "${!ALL[@]}"; do [ "${ALL[$i]}" = "$FROM" ] && start=$i; done
[ $start -ge 1 ] || { echo "FROM must be one of m2..m6 (got '$FROM')"; exit 1; }

prev_run="${ALL[$((start-1))]}_CPG"
[ -d "training/runs/$prev_run" ] || { echo "missing warm-start source training/runs/$prev_run"; exit 1; }

hours_to_hms() { python3 -c "import sys;t=int(round(float(sys.argv[1])*3600));print('%d:%02d:%02d'%(t//3600,(t%3600)//60,t%60))" "$1"; }

echo "=== resuming CPG ladder at ${FROM}_CPG, warm <- $prev_run ==="
for ((i=start; i<${#ALL[@]}; i++)); do
    run="${ALL[$i]}_CPG"
    if [ -d "training/runs/$run" ]; then
        mv "training/runs/$run" "training/runs/${run}_clockramp"
        echo "  archived previous training/runs/$run -> ${run}_clockramp"
    fi
done

prev_train=""
for ((i=start; i<${#ALL[@]}; i++)); do
    m="${ALL[$i]}"
    run="${m}_CPG"
    hms=$(hours_to_hms "$([ $i -eq $start ] && echo "$H1" || echo "$H")")
    args=(--parsable --time="$hms" --job-name="$run")
    [ -n "$prev_train" ] && args+=(--dependency="afterany:$prev_train")
    jid=$(sbatch "${args[@]}" \
        --export="ALL,PRESET=${m}_CPG,STEPS=$STEPS,NENVS=$NENVS,NAME=$run,SEED=$SEED,WARM=training/runs/$prev_run" \
        "$SB")
    pid=$(sbatch --parsable --dependency="afterany:$jid" --job-name="post_$run" \
                 --export="ALL,RUN=$run" "$POST")
    echo "  $run  train $jid ($hms, warm <- $prev_run)   post $pid"
    prev_train=$jid
    prev_run=$run
done

echo
squeue -u "$USER" -o "%.10i %.16j %.9T %.11l %R" | head -20
