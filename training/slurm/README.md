# Training on the EPFL SCITAS **Izar** cluster

Everything below was checked against the SCITAS docs (https://scitas-doc.epfl.ch/supercomputers/izar/)
on 2026-07-17.

**Izar in one paragraph** — 70 GPU nodes, each 2x Xeon Gold 6230 (40 cores), 2x NVIDIA V100 32 GB,
196–384 GB RAM. Partition `gpu` is the default; QOS `gpu` (default) allows jobs up to **3 days**,
QOS `long` up to **7 days** (lower priority), QOS `debug` gives 1 h at high priority for free
(testing only). GPUs are requested with `--gres=gpu:X`. Access is by SCITAS **student/course
account** (your supervisor submits the student-account form), from the EPFL network or VPN.

## 1. One-time setup (login node)

```bash
ssh ncmorand@izar.hpc.epfl.ch
(public key added for passwordless access)

# get the code onto the cluster (either clone, or rsync from the laptop):
git clone <your-repo-url> ~/running_robot
#   rsync -av --exclude .venv --exclude training/runs ./ <gaspar>@izar.hpc.epfl.ch:running_robot/

# venv, exactly per the SCITAS python-venv guide (pip runs on the LOGIN node — internet on
# compute nodes is not guaranteed):
module load gcc python py-virtualenv
virtualenv --system-site-packages ~/venvs/dash
source ~/venvs/dash/bin/activate
pip install --no-cache-dir -r ~/running_robot/training/requirements.txt
# V100 is sm_70 (Volta): recent default torch wheels dropped it, install from the cu126 index.
# (If this fights the requirements install order, run it AFTER; it just replaces torch.)
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# sanity check on the login node (CPU, ~30 s):
python ~/running_robot/training/smoke_test.py
```

Storage rules that matter: `/home` = 100 GB, backed up — fine for the repo and run folders
(checkpoints are small). `/scratch/<user>` = no quota, **files auto-deleted after 30 days**,
cluster-local — use it only for bulk outputs you'll copy back.

## 2. Quick pipeline test (free, high priority — do this Monday first)

```bash
# 1 h interactive GPU shell on the debug QOS:
Sinteract -q debug -g gpu:1 -c 8 -m 16G -t 1:0:0
source ~/venvs/dash/bin/activate
cd ~/running_robot
python training/train.py --preset m2_sprint --steps 300000 --n-envs 6 --subproc --no-progress \
       --name debug_check
python training/evaluate.py --run training/runs/debug_check --episodes 2
exit
```

## 3. The real run

```bash
cd ~/running_robot
# single job (fits in the 3-day QOS if throughput >= ~1000 steps/s; check the .out early):
sbatch training/slurm/izar_train.sbatch
# with overrides:
sbatch --export=ALL,PRESET=m2_sprint,STEPS=240000000,NENVS=18 training/slurm/izar_train.sbatch
# safety chain of 2 jobs (the 2nd starts when the 1st ends FOR ANY REASON and resumes
# automatically from the newest checkpoint — nothing is lost at the 3-day wall):
training/slurm/izar_chain.sh 2 --export=ALL,PRESET=m2_sprint,STEPS=240000000
# alternative: one 7-day job -> edit izar_train.sbatch: add `#SBATCH --qos=long`,
# set `#SBATCH --time=6-23:30:00`
```

Milestone chaining (recommended order; each warm-starts from the previous stage — obs/action
dims are identical across milestones so checkpoints load as-is):

```bash
j1=$(sbatch --parsable --export=ALL,PRESET=m1_sprint,STEPS=60000000,NAME=m1_sprint \
     training/slurm/izar_train.sbatch)
j2=$(sbatch --parsable --dependency=afterok:$j1 \
     --export=ALL,PRESET=m2_sprint,STEPS=90000000,NAME=m2_sprint,WARM=training/runs/m1_sprint/final_model.zip \
     training/slurm/izar_train.sbatch)
sbatch --dependency=afterok:$j2 \
     --export=ALL,PRESET=m3_sprint,STEPS=90000000,NAME=m3_sprint,WARM=training/runs/m2_sprint/final_model.zip \
     training/slurm/izar_train.sbatch
```

(`WARM` is chain-safe: it only applies when the stage has no checkpoint of its own yet; a
requeued/chained job of the same stage resumes its own newest checkpoint instead.)

## 4. Monitor / results

```bash
squeue -u $USER                     # queue state (also: Squeue, Sjob <id>)
tail -f dash-sprint-<jobid>.out     # live training log
# training_plots.png regenerates in the run dir every 500k steps:
scp <gaspar>@izar.hpc.epfl.ch:running_robot/training/runs/m2_sprint/training_plots.png .
# TensorBoard from the laptop:
ssh -L 6006:localhost:6006 <gaspar>@izar.hpc.epfl.ch \
    'source ~/venvs/dash/bin/activate && tensorboard --logdir ~/running_robot/training/runs'
# then open http://localhost:6006
# fetch everything (checkpoints are ~10 MB each):
rsync -av <gaspar>@izar.hpc.epfl.ch:running_robot/training/runs/ ./training/runs/
```

Videos: render locally after rsync (`python training/evaluate.py --run training/runs/m2_sprint
--video dash.mp4`), or on a debug GPU node with headless EGL:
`MUJOCO_GL=egl python training/evaluate.py ... --video dash.mp4` (fallback: `MUJOCO_GL=osmesa`).

Notes:
- The milestone chain above uses `afterok` on purpose (never warm-start from a stage that
  didn't finish cleanly) — but if a stage fails, its dependents sit pending forever as
  `DependencyNeverSatisfied`; `scancel` them, fix, resubmit. The same-stage safety chain
  (`izar_chain.sh`) uses `afterany` instead, so a wall-clock kill still triggers the resume job.
- If `izar_chain.sh` arrives without its executable bit, run it as
  `bash training/slurm/izar_chain.sh 2`.

## 5. Throughput expectations

18 subprocess envs on 20 cores should give roughly 1500–3500 env-steps/s for this 6-actuator
model (1 kHz sim, decimation 20; the measured single-machine rate was ~1055 fps with only 4
envs). 240 M steps is then ~19–45 h — one `gpu`-QOS job usually suffices; the chain is
insurance, it costs nothing extra. To scale to the whole node, raise BOTH the allocation and
the env count on the sbatch command line (CLI flags override the script's `#SBATCH` lines;
`--export` alone cannot change the allocation):

```bash
sbatch --cpus-per-task=40 --mem=128G --export=ALL,NENVS=36 training/slurm/izar_train.sbatch
```
