#!/bin/bash
# 2026-08-26 overnight: the MIT-drive ladder, 2 seeds, m2 -> m3 chained per seed.
# afterany (not afterok) so a wall-clock kill of m2 still lets m3 start from the newest
# checkpoint; if m2 left NO checkpoint the m3 job FATALs at the WARM check, which the
# monitor treats as "m2 needs attention", not "resubmit m3".
set -euo pipefail
cd ~/running_robot

J2S0=$(sbatch --parsable --export=ALL,PRESET=walk_fwd_m2_mit,STEPS=100000000,SEED=0,NAME=mit_m2_s0,WARM=training/runs/ladder3_m2_s0 walk_mit/slurm/jed_train.sbatch)
J2S1=$(sbatch --parsable --export=ALL,PRESET=walk_fwd_m2_mit,STEPS=100000000,SEED=1,NAME=mit_m2_s1,WARM=training/runs/ladder3_m2_s0 walk_mit/slurm/jed_train.sbatch)
J3S0=$(sbatch --parsable --dependency=afterany:$J2S0 --export=ALL,PRESET=walk_fwd_m3_mit,STEPS=80000000,SEED=0,NAME=mit_m3_s0,WARM=walk_mit/runs/mit_m2_s0 walk_mit/slurm/jed_train.sbatch)
J3S1=$(sbatch --parsable --dependency=afterany:$J2S1 --export=ALL,PRESET=walk_fwd_m3_mit,STEPS=80000000,SEED=1,NAME=mit_m3_s1,WARM=walk_mit/runs/mit_m2_s1 walk_mit/slurm/jed_train.sbatch)

echo "LAUNCHED mit_m2_s0=$J2S0 mit_m2_s1=$J2S1 mit_m3_s0=$J3S0 mit_m3_s1=$J3S1"
squeue -u ncmorand -o '%.10i %.14j %.8T %.10M %.9l %R'
