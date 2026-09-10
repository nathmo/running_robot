#!/bin/bash
# Hourly status of the v2b chains (and the artifact-literal v2 controls) on JED. Run from anywhere:
#     ssh jed 'bash ~/running_robot/walk_mit/monitor/v2_check.sh'
# Prints the queue, the last progress rows per run (every 5 M), the instability/traceback counts
# of the job logs, and -- once per NEW 5 M checkpoint -- greedy_peek on it, on the nominal plant
# and at the run's own DR scale (3 episodes each). The signature that matters (readout-1):
# committed clock off its rail, spec-family rms well below 1, residual share < 0.4, and a greedy
# ep_len that tracks the stochastic one. State (which checkpoints were already peeked) lives in
# walk_mit/monitor/.v2_check_state so repeated calls stay cheap.
source "${VENV:-$HOME/venvs/dash-jed}/bin/activate"
cd "${REPO:-$HOME/running_robot}"
export OMP_NUM_THREADS=2
STATE=walk_mit/monitor/.v2_check_state
touch "$STATE"
RUNS="${RUNS:-v2c_s1_s0 v2c_s1_s1 v2cw_s1_s0 v2c_s2_s0 v2c_s2_s1}"
echo "== $(date '+%F %T') queue"
squeue -u "$USER" -o "%.10i %.14j %.3t %.11M %R" | tail -n +2
for r in $RUNS; do
    d=walk_mit/runs/$r
    [ -f "$d/progress.csv" ] || continue
    echo "== $r"
    out=$(python walk_mit/monitor/progress_curve.py "$d/progress.csv" 5e6 2>/dev/null)
    echo "$out" | sed -n 2p
    echo "$out" | tail -3
    [ -f "$d/curriculum.json" ] && { echo -n "   curriculum: "; cut -c1-300 "$d/curriculum.json"; echo; }
    # newest checkpoint on a 5 M multiple, peeked once
    ck=$(ls "$d"/ppo_*000000_steps.zip 2>/dev/null | sed 's/.*ppo_\([0-9]*\)_steps.zip/\1/' | sort -n | awk '$1 % 5000000 == 0' | tail -1)
    [ -z "$ck" ] && continue
    if ! grep -q "^$r $ck\$" "$STATE"; then
        echo "-- greedy_peek $r @ $ck (nominal plant, then the run's DR scale)"
        for dr in "--dr 0" ""; do
            timeout 900 python walk_mit/monitor/greedy_peek.py "$d" "ppo_${ck}_steps" --episodes 3 $dr 2>&1 \
                | grep -v "^WARNING\|^$\|Deprecat\|warn\|\[eval\]"
        done
        echo "$r $ck" >> "$STATE"
    fi
done
echo "== job logs"
for f in dash-mit-*.out; do
    [ -f "$f" ] || continue
    if [ "$(find "$f" -mmin -180 | wc -l)" = "1" ]; then
        echo "   $f: unstable=$(grep -c -i unstable "$f") traceback=$(grep -c Traceback "$f") gates=$(grep -c "gate '" "$f") lines=$(wc -l < "$f")"
        grep "curriculum gate\|entropy anneal opened\|Traceback\|Error" "$f" | tail -4 | sed 's/^/      /'
    fi
done
