# Running walk_v2 on SCITAS **Lyra** (GPU beta, 2026-09)

Lyra: `ssh ncmorand@lyra.hpc.epfl.ch` (VPN). Partitions `rtx6000` (14 × 8 RTX PRO 6000 Blackwell,
96 GB each, 16 cores/GPU) and `b200` (8 × 8 B200, 12 cores/GPU). **The partition is mandatory.**
Free until 2026-09-30; beta — jobs may be killed, so every run is requeue-safe.

## Account status (checked 2026-09-09)

`sacctmgr show user ncmorand withassoc` on Lyra returns account `master`, QOS **`disable`**
(MaxTRES `gres/gpu=0,node=0`), and `srun --partition rtx6000 --gpus 1` is refused with
`QOSMaxCpuPerUserLimit`; `--qos normal|debug` returns `Invalid qos specification`. Nothing can
be scheduled until SCITAS attaches a real QOS/account (the beta-test enrolment) to the user.
The venv, the code and the job scripts are in place so the first job is one `sbatch` away.

Verified on the login node (CPU, `JAX_PLATFORMS=cpu`, 2026-09-10): imports, the MJX plant load,
and every unit check of `smoke_test.py` pass; constructing `DashEnvV2` passes; the first jit of
the vmapped `reset` is SIGKILLed (exit 137) by the login node's resource limiter. That is where
a compute node is needed — do not run `smoke_test.py` without `--quick` or any training on the
login node; `lyra_train.sbatch` runs `smoke_test.py --quick` and then trains on the GPU node.

## One-time setup (login node; pip needs the login node's internet)

```bash
ssh ncmorand@lyra.hpc.epfl.ch
~/venvs/make_dash_v2.sh          # = module load gcc/14.3.0 python/3.12.12; venv ~/venvs/dash-v2;
                                 #   pip install jax[cuda12]==0.11.1 mujoco==3.13.0 mujoco-mjx==3.13.0 flax optax ...
# code: from the laptop (Windows: scp, no rsync)
scp -r walk_v2 ncmorand@lyra.hpc.epfl.ch:running_robot/
```

The system Python on Lyra is 3.9; jax ≥ 0.10 needs 3.12, hence the module. `jax[cuda12]` ships
the CUDA runtime as pip wheels, so only the node's driver matters (Blackwell needs ≥ 570).
Check on a node: `srun --partition rtx6000 --gpus 1 nvidia-smi`.

## Jobs

```bash
# benchmark (1 h): env steps/s vs batch + one PPO iteration per size
sbatch walk_v2/slurm/lyra_bench.sbatch
sbatch --partition=b200 --cpus-per-task=12 --mem=120G walk_v2/slurm/lyra_bench.sbatch

# training, S1 then S2 warm
sbatch --export=ALL,PRESET=v2_s1_planar,NAME=v2_s1_planar_s0,SEED=0 walk_v2/slurm/lyra_train.sbatch
sbatch --export=ALL,PRESET=v2_s2_free,NAME=v2_s2_free_s0,SEED=0,WARM=walk_v2/runs/v2_s1_planar_s0 \
       walk_v2/slurm/lyra_train.sbatch
# chain 3 links of the same run (each resumes the newest checkpoint)
walk_v2/slurm/lyra_chain.sh 3 --export=ALL,PRESET=v2_s2_free,NAME=v2_s2_free_s0
```

Sizing: one GPU per job; `NENVS` 4096 × `NSTEPS` 32 = 131 072 samples per PPO rollout (the
walk_mit stack used 64 × 288 = 18 432 on 72 CPU cores). Memory per env is small (an
18-DOF plant); 8192 envs fit on a 96 GB card with room. CPU cores are only for the Python
loop and jit compilation.

## Pull results

```bash
RUN=v2_s1_planar_s0
for f in final.msgpack resolved_config.json progress.csv eval.csv training_plots.png; do
  scp ncmorand@lyra.hpc.epfl.ch:running_robot/walk_v2/runs/$RUN/$f walk_v2/runs/$RUN/; done
```
