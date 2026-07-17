# DASH-01 sprint training

Self-contained RL training stack for the DASH-01 biped's **100 m dash**: stand at the line,
sprint 100 m as fast as possible, stop. Clone the repo, make a venv, install
`requirements.txt`, train. Nothing outside this folder is needed (the MuJoCo model + meshes +
ride-height LUT live in `model/`).

## Quickstart

```bash
python -m venv .venv
# activate it — PowerShell: .venv\Scripts\Activate.ps1 | Git Bash: source .venv/Scripts/activate
#                | Linux/macOS: source .venv/bin/activate
pip install -r training/requirements.txt

python training/smoke_test.py         # ~30 s sanity suite — run before every long launch

# the weekend run (240 M steps; ~1-2 days depending on cores — checkpoints every ~1M steps):
python training/train.py --preset m2_sprint --steps 240000000 --n-envs 20 --subproc

# watch it:
python -m tensorboard.main --logdir training/runs      # or open training/runs/<name>/training_plots.png

# after (or during, on any checkpoint):
python training/evaluate.py --run training/runs/m2_sprint --episodes 5
python training/evaluate.py --run training/runs/m2_sprint --video dash.mp4
python training/plot_training.py --run training/runs/m2_sprint
```

Interrupted? The same command + `--resume auto` continues from the newest checkpoint.
Cluster: see [slurm/README.md](slurm/README.md) for the EPFL Izar (SLURM) setup.

## The approach (post-literature-review, 2026-07-17)

The action is a **per-step re-parameterized Fourier gait + residuals** — the hybrid the running
literature converged on (CPG-RL: per-step oscillator re-parameterization; PMTG: trajectory
generator + per-step residual corrections). At every 50 Hz control step the policy emits 24 numbers:

| slice | meaning |
|---|---|
| 14 | cam + thigh Fourier coefficients (N=3 harmonics, 1/sqrt(k) weighting), symmetric: right leg = mirrored, antiphase |
| 1 | gait frequency in [0.5, 3.0] Hz (the phase integrates it) |
| 3 | abduction reflex gains: hip_roll = kp*roll + kd*roll_rate + bias (50 Hz feedback; inert while roll is railed) |
| 6 | per-step residual corrections on the 6 PD targets (±0.08 rad) — the fast-feedback channel |

Rewriting the gait spec mid-cycle is priced by a phase-gated penalty (free exactly at the cycle
boundary); residuals are never billed by it — they exist to be per-step.

**Observation** (55 x 5-frame history = 275): motor pos/vel/torque, IMU gravity + gyro,
**base velocity** (privileged sim state — the quantity the reward maximizes must be observable),
sin/cos gait phase, task channel (**run/stop flag + distance-to-go**, so the policy sees the
line coming and the value function sees the state its return depends on), previous action.

**Reward** (raw units, no reward normalization; two-level suicide-proofing: per-term caps AND a
−1.0/step floor on the pre-terminal total, so neither a single term nor their sum ever makes
dying value-optimal):
- **Run phase:** dense speed income `2.0 * clip(vx, −3.0, +3.0)` — **symmetric**, so backward
  travel pays negative (a one-sided clip makes shuttling before the line beat crossing it) —
  **minus a constant clock cost** 0.5/step: per-step speed income alone integrates to
  w*distance whatever the pace; only the clock prices TIME. Break-even pace 0.25 m/s.
- **Stop phase** (after the line, latched): a 'be stationary' kernel that is deliberately
  **net-negative** (max +0.4 income vs the 0.5 clock) — the hold-timer reset is under policy
  control, so any net-positive stop income is farmable forever by twitching; 5 m free braking
  zone, then a per-meter overrun penalty; standing still 1 s ends the episode **+100**.
- **Anti-skate set** (measured necessity — the pure per-step-PD policy skated): foot-slip,
  per-foot stance-time cap, fresh-swing clearance, one-sided touchdown air-time credit.
- **Phase-gated contact schedule** (Siekmann-style, NEW): each foot pays for ground contact
  during its expected swing window; the windows use the action's own gait clock. The stance
  ratio ramps 0.65 → 0.42 over training — below 0.5 the swing windows overlap, so contact by
  either foot in the overlap pays: **a flight phase is explicitly demanded** (this is the term
  that asks for running rather than fast walking).
- **Efficiency** (Cassie-100 m recipe, ramped in over 60 M steps so it can't smother gait
  emergence): torque above the standing baseline, motor-velocity cost, positive mechanical
  power (CoT proxy). This is what produced long-stride running instead of cadence thrash there.
- Posture/smoothness: upright, height (when Z free), lateral velocity, no-crossing stance
  width, hip-roll neutrality, action-rate, residual magnitude.

**Curricula** (all continue seamlessly across `--resume auto`): sprint line 25 → 100 m over
30 M steps; stance ratio 0.65 → 0.42 over 60 M; efficiency weight 0 → 1 over 60 M;
competence-gated entropy anneal (holds exploration until stepping has emerged).

## Milestones (base-DOF rail curriculum)

`--preset mK_sprint` rails base DOFs so balance is learned incrementally; obs/action dims never
change, so each stage **warm-starts** from the previous:

| preset | free base DOFs | note |
|---|---|---|
| `m1_sprint` | X | Z railed at a per-episode random ride height (LUT-seated) |
| `m2_sprint` | X, Z | carries its own ride height — **the default weekend run** |
| `m3_sprint` | X, Z, pitch | fore/aft attitude live |
| `m4_sprint` | + Y | lateral translation |
| `m5_sprint` | + roll | abduction reflex becomes live |
| `m6_sprint` | all six | the real plant |

```bash
python training/train.py --preset m3_sprint --warm-start training/runs/m2_sprint/final_model.zip
```

`mK_speed` variants are the same plants with an endless max-speed objective (no line/clock) for
gait debugging.

## Files

| file | role |
|---|---|
| `config.py` | every tunable, one dataclass + presets |
| `fourier_gait.py` | pure-numpy gait reconstruction + phase windows (Pi-shareable) |
| `env.py` | the Gymnasium environment (plant, obs, reward, sprint logic) |
| `train.py` | PPO + callbacks (ramps, entropy gate, checkpoints, plots) |
| `evaluate.py` | metrics / mp4 / live viewer + dash telemetry |
| `plot_training.py` | training_plots.png from progress.csv |
| `smoke_test.py` | pre-launch sanity suite |
| `model/` | dash01.xml + meshes + ride-height LUT + model build/validation scripts |
| `slurm/` | Izar cluster job scripts + instructions |

## Known debts (deliberate, documented)

- **Sim2real observations:** base velocity and distance-to-go are privileged sim state; hardware
  needs a velocity estimator (or teacher-student distillation) and odometry before transfer.
- **Domain randomization:** not implemented yet (the old config had unused knobs; removed —
  reintroduce when a policy is worth transferring).
- **GPU-vectorized training (MJX):** the per-step action mode is fixed-shape and portable in
  principle; do it if CPU throughput becomes the bottleneck.
- Rebuilding `model/dash01.xml` from CAD (`model/build_model.py`) reads the CAD export that now
  lives with the robot hardware folder/repo — day-to-day training never needs it.
