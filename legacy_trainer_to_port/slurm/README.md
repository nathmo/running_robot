# Running walk_v2 on SCITAS GPUs

The working cluster is **Izar** (V100 32 GB, sm_70, driver 535). Lyra (RTX PRO 6000 / B200 beta)
is where this was meant to run; the account there is QOS `disable` (see the bottom) and nothing
can be scheduled until SCITAS enrols the user, so everything below was measured on Izar.

## Izar setup (done 2026-09-10; redo only on a fresh account)

```bash
ssh ncmorand@izar.hpc.epfl.ch                       # VPN
# Python 3.12 via uv (the system python is 3.9 and the module python filters jax's wheels)
curl -LsSf https://astral.sh/uv/install.sh | sh     # -> ~/.local/bin/uv
~/.local/bin/uv python install 3.12
~/.local/bin/uv venv ~/venvs/dash-v2 --python 3.12
source ~/venvs/dash-v2/bin/activate
~/.local/bin/uv pip install -r running_robot/walk_v2/requirements.txt "jax[cuda12]==0.11.1"
# code: from the laptop (Windows: scp, no rsync); the meshes are read from ../../walk_mit/model
scp -r walk_v2 ncmorand@izar.hpc.epfl.ch:running_robot/
scp -r walk_mit/model/meshes ncmorand@izar.hpc.epfl.ch:running_robot/walk_mit/model/
```

`jax[cuda12]` ships the CUDA runtime as pip wheels, so only the node's driver matters (535 is
enough for CUDA 12 wheels on a V100). Non-login ssh shells do not load modules: every job script
is `#!/bin/bash -l` and puts `~/.local/bin` on `PATH` itself. Do not run the vmapped reset or
any training on the login node (it SIGKILLs the first jit); `--quick` smoke checks are fine.

## Jobs

One GPU = half an Izar node (10 cores, 90 GB); asking for more cores strands the node's other
GPU (SCITAS asked us not to, 2026-08-10). QOS: `normal` (3 days), `debug` (1 h, fast queue),
`long`.

```bash
# check + benchmark on the debug QOS (1 h): full smoke test on the GPU, env steps/s vs batch,
# one PPO iteration per batch size -> walk_v2/results/bench_izar_v100_<job>.json
sbatch walk_v2/slurm/izar_bench.sbatch

# training: S1 (planar) from scratch, then S2 warm-started from it; 300 M steps each
sbatch --export=ALL,PRESET=v2_s1_planar,NAME=v2_s1_planar_s0,SEED=0 walk_v2/slurm/izar_train.sbatch
sbatch --export=ALL,PRESET=v2_s2_free,NAME=v2_s2_free_s0,SEED=0,WARM=walk_v2/runs/v2_s1_planar_s0 \
       walk_v2/slurm/izar_train.sbatch
# chain 3 links of one run (each link: --resume auto on the newest checkpoint)
walk_v2/slurm/izar_chain.sh 3 --export=ALL,PRESET=v2_s2_free,NAME=v2_s2_free_s0

# solver / step profiling on the debug QOS (tools/profile_step.py, tools/profile_env.py, bench.py caps)
sbatch walk_v2/slurm/izar_profile.sbatch          # bare MJX physics per solver variant + drift
sbatch walk_v2/slurm/izar_solver_bench3.sbatch    # env + PPO iteration per solver cap (fixed bench)
```

Sizing on the V100: `NENVS` 2048 × `NSTEPS` 64 = 131 072 samples per PPO rollout (the walk_mit
CPU stack uses 64 × 288 = 18 432 on 72 cores). Memory per env is small (an 18-DOF plant).

## Throughput lessons (2026-09-10, see `walk_v2/results/`)

* **The first bench numbers were wrong by ~15×.** `bench_env` warmed up with a 3-step scan and
  timed a 64-step scan with the scan length static, so the timed call recompiled (~40 s on the
  V100) and the compile showed up as a batch-independent ~600 ms/step floor on CPU and GPU alike.
  Fixed: warm up at the timed length. The PPO-iteration numbers and the training rollout times
  never had this problem.
* **Under `jax.vmap` the Newton solver runs to the slowest env.** The XML carries classic
  MuJoCo's `iterations=100 ls_iterations=50`; in healthy states Newton converges in one
  iteration, but one flailing env in the batch makes every env pay the extra iterations.
  `Config.mjx_iterations / mjx_ls_iterations` cap this at model load (0 = XML). The batched-MJX
  recipe of 1-2 iterations breaks this plant: the stiff loop closure (solref 0.002) needs a
  converged solve — the golden replay ends at tick 13 instead of 64. Caps are only acceptable if
  `tools/replay_golden.py --iterations N --ls-iterations M` still matches the CPU fixture.

## Pull results

```bash
RUN=v2_s1_planar_s0
for f in final.msgpack resolved_config.json progress.csv eval.csv training_plots.png; do
  scp ncmorand@izar.hpc.epfl.ch:running_robot/walk_v2/runs/$RUN/$f walk_v2/runs/$RUN/; done
```

## Lyra (blocked)

`ssh ncmorand@lyra.hpc.epfl.ch`; partitions `rtx6000` (14 × 8 RTX PRO 6000 Blackwell, 96 GB,
16 cores/GPU) and `b200` (8 × 8 B200, 12 cores/GPU); the partition is mandatory; free until
2026-09-30, beta. `sacctmgr show user ncmorand withassoc` returns account `master`, QOS
**`disable`** (MaxTRES `gres/gpu=0,node=0`); `srun --partition rtx6000 --gpus 1` is refused with
`QOSMaxCpuPerUserLimit` and `--qos normal|debug` is `Invalid qos specification`. The venv
(`~/venvs/make_dash_v2.sh`: `module load gcc/14.3.0 python/3.12.12`) and the code are in place;
`lyra_bench.sbatch` / `lyra_train.sbatch` / `lyra_chain.sh` are the Izar scripts with the
partition line, `NENVS` 4096 × `NSTEPS` 32, and are the first jobs to submit once enabled.
