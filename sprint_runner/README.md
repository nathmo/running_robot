# sprint_runner — THE RUNNER (3.06 m/s), restored

The stack that produced the project's first real runner, `sprint_m3_mit_s0`: **100 m in 32.5–32.7 s
at 3.06–3.08 m/s, 8/8 greedy episodes, no falls.** It was deleted on 2026-09-17 by commit
`0031885` ("Remove the legacy stacks") together with the rest of `walk_mit/`, because the flat-foot
CAD invalidated the plant it trains on. This folder restores it, self-contained, so the policy can
still be run, measured and filmed.

## Provenance — what was restored, and how

Everything is the **pre-deletion tree at `0031885^` (= `f11eed8`)**, copied out with
`git cat-file blob`, never `git show`/`git checkout`: this repo sets `core.autocrlf=true`, and the
line-ending filter silently corrupts the .zip/.pkl/.mp4 blobs. All 140 restored files were verified
byte-identical against their blob SHAs with `git hash-object`.

```
sprint_runner/
  walk_mit/          the training + eval stack, verbatim (config.py env.py train.py evaluate.py
                     gait_diag.py fourier_gait.py cpg_gait.py domain_rand.py ... + model/ + slurm/)
  walk_mit/runs/     ONLY sprint_m3_mit_s0 and _s1 (final_model.zip, vecnormalize.pkl,
                     resolved_config.json, curriculum.json, the 600 M progress CSV)
  walk_mit/monitor/  the two FINAL_600M films, the archived eval printouts, the gait-diag JSON
  tools/             torque_dash.py (NEW, see below) + the restored eval_envelope / summarize /
                     plot_sprint_{dash,training,margins} / eval_dash_xy
  results/           the restored thesis figures + the new torque figures and data
```

The rest of `walk_mit/` — the other 25 runs (imp_m2/m3 lineage, sprint_m4/m5/m6, v2c), ~170 MB of
monitor videos, the v2 latched-spec lineage's fixtures — was left in history rather than copied;
pull it the same way if you ever need it.

## Running it

The stack is SB3/PyTorch, not the MJX/JAX stack the rest of the repo now uses, so it gets its own
venv (`.venv/`, gitignored):

```
python -m venv .venv
.venv/Scripts/pip install mujoco==3.13.0 "stable-baselines3>=2.3" "gymnasium>=0.29" \
    "torch>=2.2" "numpy>=1.26" matplotlib imageio imageio-ffmpeg
.venv/Scripts/python walk_mit/evaluate.py --run walk_mit/runs/sprint_m3_mit_s0 --episodes 8
.venv/Scripts/python walk_mit/evaluate.py --run walk_mit/runs/sprint_m3_mit_s0 --video dash.mp4
```

**Re-run 2026-09-23 on mujoco 3.13.0 / torch 2.14 / SB3 2.9 reproduces the archived result
exactly**: 8/8 dashes, 100.6 ± 0.2 m in 32.75 ± 0.08 s, mean body speed 3.06 m/s — against the
archived `monitor/eval_sprint_s0_final.txt` (ep_len 6562 ± 22, 3.06 m/s, 32.49–32.69 s). Nothing in
the restored stack needed patching for the newer libraries.

## Torque measurement (`tools/torque_dash.py`, new)

```
.venv/Scripts/python tools/torque_dash.py --episodes 8 --tag s0
```

Runs `evaluate.py`'s protocol (run config + curriculum, greedy, pushes off, one seed per episode)
and logs `actuator_force` **at every 1 ms physics substep** by wrapping `mujoco.mj_step` — reading
it once per 5 ms control step aliases the contact spikes and under-reads the peak. Writes
`results/torque_{dash,stride,bars}_s0.{png,pdf}`, the per-episode numbers as JSON, and the plotted
episode's raw 1 kHz trace as NPZ.

### Result — 8 dashes, 262,020 samples at 1 kHz

Steady run (t = 2 s to the line; launch and the post-line stop excluded). Peak limit is the model's
`forcerange` = the measured delivered peak; continuous is the thermal rating (AKE90-8 measured 55,
AK60-39 **estimated** 23.3 — never benched).

| actuator | RMS | peak | P95 | peak limit | % of peak | RMS / continuous | % of samples ≥95 % peak |
|---|---|---|---|---|---|---|---|
| hip_roll_L | 21.0 | 61.2 | 58.2 | 61.2 | 100 % | 90 % | 5.1 % |
| cam_L | 48.4 | 144.5 | 139.1 | 144.5 | 100 % | 88 % | 5.4 % |
| thigh_L | 25.2 | 127.7 | 60.1 | 144.5 | 88 % | 46 % | 0.0 % |
| hip_roll_R | 19.9 | 61.2 | 59.6 | 61.2 | 100 % | 86 % | 6.3 % |
| cam_R | 50.7 | 144.5 | 144.5 | 144.5 | 100 % | 92 % | 9.1 % |
| thigh_R | **54.1** | 144.5 | 144.5 | 144.5 | 100 % | **98 %** | 7.7 % |

Torque in N·m. Whole-dash figures (including launch and stop) are in the JSON; they differ only for
thigh_L, whose 144.5 peak comes from the stop, not the run.

Three things this says, all of which confirm the 2026-09-01 limits audit against an independent
re-implementation of the measurement:

1. **No thermal margin.** thigh_R runs the whole 32.7 s at 54.1 N·m RMS = 98 % of the AKE90-8's
   55 N·m continuous rating, cam_R at 92 %. One dash is fine; back-to-back dashes, a hot day or any
   voltage sag have nowhere to go. Both hip_rolls sit at 86–90 % of an *estimated* AK60 rating, so
   that pair is the one number worth putting on a dyno before believing it.
2. **No torque reserve.** Five of six actuators touch the peak limit inside the steady run, and
   cam_R spends 9.1 % of the dash at ≥95 % of it. The policy is riding the saturation boundary, not
   cruising below it.
3. **The load is grossly asymmetric** — thigh_R 54.1 RMS vs thigh_L 25.2. That is the one-legged
   hop, and the contact log in the same trace confirms it: right-foot stance duty 0.197,
   left-foot 0.005 (0.000 after t = 2 s), a touchdown every 207 ms (4.8 Hz), airborne 80 % of the
   dash. `torque_stride_s0.png` shows one second of it. The m3 rail (y/roll/yaw locked) is what
   makes the asymmetry affordable; `sprint_m6_lim2` is the run that later fixed it.

Two caveats inherited from the run itself, not from this measurement: it trained with
`motor_r_ohm=[]` and `motor_vel_limit=0`, so there is no back-EMF derate — part of the 3.06 m/s
rides swing-leg speeds (~2× no-load) a real 48 V bus cannot deliver. And the policy's delay margin
is zero: it survives only at exactly the trained 5 ms actuation delay.
