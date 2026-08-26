#!/bin/bash
# One-shot status dump for the overnight monitor.
# Run from the laptop:  ssh ncmorand@jed.hpc.epfl.ch 'bash running_robot/walk_mit/slurm/status_remote.sh'
cd ~/running_robot
echo "== queue"
squeue -u ncmorand -o '%.10i %.14j %.8T %.10M %.9l %R'
for r in mit_m2_s0 mit_m2_s1 mit_m3_s0 mit_m3_s1 imp_m2_s0 imp_m2_s1 imp_m3_s0 imp_m3_s1 \
         mit_m2_easy_s0 mit_m2_easy_s1 mit_m3_easy_s0 mit_m3_easy_s1 \
         imp_m2_easy_s0 imp_m2_easy_s1 imp_m3_easy_s0 imp_m3_easy_s1; do
  f=walk_mit/runs/$r/progress.csv
  if [ -f "$f" ]; then
    echo "== $r  ($(wc -l < "$f") rows)"
    head -1 "$f"
    tail -1 "$f"
  fi
done
echo "== log errors (last 3 lines per file that has any)"
grep -liE 'traceback|fatal|srun: error' dash-mit-*.out 2>/dev/null | while read -r f; do
  echo "-- $f"; grep -iE 'traceback|fatal|srun: error' "$f" | tail -3
done
echo "== newest checkpoints"
for r in mit_m2_s0 mit_m2_s1 mit_m3_s0 mit_m3_s1 imp_m2_s0 imp_m2_s1 imp_m3_s0 imp_m3_s1; do
  d=walk_mit/runs/$r
  [ -d "$d" ] && echo "$r  $(ls -1t "$d"/ppo_*_steps.zip 2>/dev/null | head -1)"
done
