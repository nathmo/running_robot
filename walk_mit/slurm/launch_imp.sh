#!/bin/bash
# 2026-08-26 late addition (user's call): COLD from-scratch runs with the per-step impedance
# channel (+4 per-leg kp/kd dims, action 26 -> 30). No WARM on m2 — nothing fits the new widths.
# m3 warm-starts from imp m2 WITHIN the lineage. The plain _mit warm chain stays running as the
# fixed-gain control arm.
set -euo pipefail
cd ~/running_robot

I2S0=$(sbatch --parsable --export=ALL,PRESET=walk_fwd_m2_mit_imp,STEPS=100000000,SEED=0,NAME=imp_m2_s0 walk_mit/slurm/jed_train.sbatch)
I2S1=$(sbatch --parsable --export=ALL,PRESET=walk_fwd_m2_mit_imp,STEPS=100000000,SEED=1,NAME=imp_m2_s1 walk_mit/slurm/jed_train.sbatch)
I3S0=$(sbatch --parsable --dependency=afterany:$I2S0 --export=ALL,PRESET=walk_fwd_m3_mit_imp,STEPS=80000000,SEED=0,NAME=imp_m3_s0,WARM=walk_mit/runs/imp_m2_s0 walk_mit/slurm/jed_train.sbatch)
I3S1=$(sbatch --parsable --dependency=afterany:$I2S1 --export=ALL,PRESET=walk_fwd_m3_mit_imp,STEPS=80000000,SEED=1,NAME=imp_m3_s1,WARM=walk_mit/runs/imp_m2_s1 walk_mit/slurm/jed_train.sbatch)

echo "LAUNCHED imp_m2_s0=$I2S0 imp_m2_s1=$I2S1 imp_m3_s0=$I3S0 imp_m3_s1=$I3S1"
squeue -u ncmorand -o '%.10i %.14j %.8T %.10M %.9l %R'
