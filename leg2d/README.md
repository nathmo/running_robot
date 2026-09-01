# leg2d — single-leg boom rig + gait-parameter sweep

A planar (sagittal-plane) single-leg dynamic MuJoCo model, built from the real
`training/model/dash01.xml` plant (measured 15.14 kg, real AKE90-8 actuator specs, real passive
ankle spring), used to sweep gait parameters against the real motor torque-speed envelope and the
real single-support mass — extending `memory/hardware-speed-ceiling.md`'s static capability
analysis into an actual dynamic simulation.

## The rig

A boom-mounted single leg, the standard reduced-order rig for characterising leg-mechanism
capability independent of balance control (MIT Leg Lab, ETH SL1M-style benches):

- **Only `base_x` and `base_z` are free.** `base_y` / `base_roll` / `base_pitch` / `base_yaw` are
  railed via the equality locks already present in `dash01.xml` (just flipped `active="true"`).
  The torso can translate in the sagittal plane but never rotate or drift sideways. This sidesteps
  the whole-body pitch-balance problem — the thing that has actually walled every RL run at m3+
  (see `memory/cmd-curriculum-deadlock.md`, `memory/m3-pitch-fix.md`) — so a gait can be commanded
  **open-loop** and simply checked for feasibility, instead of requiring a trained/tuned balance
  controller just to stay upright.
- **The right leg is deleted** (single-leg rig) but its mass (5.44 kg) is preserved as a fixed
  point mass welded to the torso — real single support carries the WHOLE robot's weight, including
  the idle swing leg. Total system mass is unchanged: 15.14 kg.
- **`cam_L` / `thigh_L` are direct-torque `<motor>` actuators**, not `dash01.xml`'s `<position>`
  actuators. `dash01.xml`'s position actuator has a *constant* `forcerange` (144.5 N·m) regardless
  of joint speed — a real motor can't do that; deliverable torque falls off linearly to zero at the
  no-load speed. That falloff is exactly the mechanism behind the "swing is power-limited" finding
  in `memory/hardware-speed-ceiling.md`, so `sweep.py` drives these joints with its own outer PD +
  the real torque-speed clamp (`motor.py`) every control step.

Build it: `.venv/Scripts/python.exe leg2d/build_leg2d.py` → writes `leg2d/model/leg2d.xml`.

## The gait — not learned, not fit

No RL, no gradient descent, no Fourier-coefficient tuning. Locking the boom removes the balance
problem, so the gait isn't discovered — cadence / duty factor / stride length / clearance directly
*are* the trajectory (`gait.py`), reusing the exact stance-ramp + smoothstep-swing shape already
validated in `training/scripted_walk.py`. Mapped to joint targets via the same measured foot-IK
table (`training/model/cpg_foot_lut.npz`) the CPG/scripted controllers use.

**Duty factor is also the walk↔run knob.** This is a single leg on a boom with nothing to catch
the body — whenever the foot is up (fraction `1-duty`), the body is necessarily airborne. `duty
→ 0.85` is a mostly-grounded walk; `duty → 0.3` is a short-stance, long-flight run. The sweep spans
both automatically; nothing about the trajectory generator changes between them.

**This produces a capability lower bound, not the true optimum.** The trajectory *shape* is fixed
and hand-picked (not arbitrary — it's the shape two independent parts of this project already
converged on — but not proven optimal either). The sweep answers "how fast can *this* gait family
go, across its free parameters" — not "what is the fastest possible gait for this mechanism."
Finding the true optimum would mean trajectory optimization or turning RL loose on this same rig.

## Running the sweep

```
.venv/Scripts/python.exe leg2d/sweep.py --quick     # 12-point smoke test, ~few seconds
.venv/Scripts/python.exe leg2d/sweep.py             # full grid, 384 points, ~80 s
.venv/Scripts/python.exe leg2d/plot_sweep.py         # -> leg2d/results/sweep_heatmap.png
.venv/Scripts/python.exe leg2d/render_gait.py --f 8 --duty 0.85 --stride 0.2   # visual sanity check
```

Each grid point runs 6 hop cycles (2 discarded as transient from the keyframe), measures: forward
speed (linear fit of `base_x` over the measured window), per-motor RMS torque vs the 55 N·m
continuous rating, whether the raw PD command ever asked for more torque than the motor can deliver
at that joint speed, and whether `base_z` stayed in a sane band (collapse / fly-off detection).

## Results (`leg2d/results/sweep.json`, `sweep_heatmap.png`)

**165/384 points feasible.** The dominant failure mode is mechanical, not thermal or motor-power:
**196/384 collapsed** (the torso `base_z` crashed toward/through the floor — a real buckling
failure, not a normal hop dip) vs. only 23 thermal-limited. This is a *different* binding
constraint than the earlier static analysis found, because the static analysis explicitly excluded
dynamic loading and momentum effects. Two findings stand out:

1. **Collapse is dominated by cadence, not stride length.** In the heatmap, the *low*-cadence
   column (0.5–2 Hz) collapses almost everywhere regardless of stride, while the *high*-cadence
   column (6–8 Hz) is feasible almost everywhere, including at large strides. This is the opposite
   of the earlier static analysis's framing (which treated high cadence as the hard limit, via
   motor peak power) — here, in a real dynamic single-leg loading regime, *slow* gaits are the
   risky ones, plausibly because a slower step gives the parallel-linkage knee's known
   snap-through-collapse mode (`memory/m2-skating-and-v3-reward.md`) more time/excursion to reach
   an unrecoverable pose before the next stance transition. This is a genuinely new result this
   rig surfaces that the static capability bound structurally couldn't see.
2. **Best measured speed among clearly-intentional gaits (stride ≥ 0.15 m) is ~0.23 m/s** — far
   below both the earlier static ceiling (3.7–6.9 m/s) and below the per-gait commanded target in
   every single case. Tracking is poor across the whole feasible region: the mechanism is not
   failing to hold peak torque (saturation is ~0% almost everywhere), it's failing to *complete*
   the commanded excursion without buckling first. **The standing crouch itself already droops
   from 1.01 m (two-leg) to 0.81 m under single-leg full-body loading** (P-only position loop,
   `memory/spiderbot-hardware.md`) — the single-support leg is operating much closer to its
   structural margin than the double-support case ever exercised.
3. **The near-zero-stride points reproduce a known quirk**: very small commanded strides (≤0.1 m)
   at moderate-to-high cadence sometimes show *more* net speed than a barely-moving trajectory
   should produce (up to 0.4+ m/s). This independently reproduces
   `training/scripted_walk.py`'s own documented finding — "a fixed floor makes v_des=0 march at
   ~0.5 m/s... standing means stepping small" — from a completely different rig and control
   scheme. Treat headline "fastest feasible point" numbers with `stride < 0.15 m` as this artifact,
   not intentional locomotion; the results file's `stride ≥ 0.15 m` filter in the analysis above
   excludes it.

## Caveats

- Stance `dz` is held at 0 the whole stance phase (rigid-pendulum leg, per
  `memory/foot-arc-geometry.md`'s two-leg finding) — under single-leg loading this may itself be
  contributing to the collapse-at-low-cadence result; an active/compliant stance profile is
  untried.
- `motor.py`'s thermal check is a simple RMS-vs-continuous-rating proxy, not the full winding/case
  RC model (`robot/deploy/thermal.py` is explicitly uncalibrated for this motor).
- The 794.75 W peak-power figure implied by (144.5 N·m, 22 rad/s) differs ~15% from
  `memory/hardware-speed-ceiling.md`'s 935 W, which paired the *raw* 170 N·m stall torque with the
  same no-load speed — an inconsistency inherited from that earlier ad-hoc analysis. `motor.py`
  uses the internally-consistent (delivered, measured) pairing throughout.
- Grid resolution is coarse (8×6×8); the collapse boundary is a hard edge in some panels, so a
  finer local grid near duty∈[0.3,0.5], f∈[2,4] would sharpen where it actually sits.

## rail_bound.py — no-load upper bound on running speed

A separate, deliberately optimistic model (2026-09-01): weld the torso to the world entirely (rail,
no load, contact disabled) and measure the fastest sustainable back-and-forth fore-aft toe sweep.
Grounded running at duty 0.5 moves the body exactly as fast as the stance foot sweeps backward
relative to it, so this brackets every real grounded gait from above. Speed is the measured mean
|ẋ_toe| relative to the torso over steady-state cycles, and **only in-workspace travel counts**:
the cycle must keep the toe within 5 cm of the measured arc height z(x) and cover ≥ 90% of it.

The workspace is measured, not assumed (PD scan + fold-branch cut + smooth arc fit + a
quasi-static fold probe run by the sweep controller itself). Key finding on the way: the raw
max-extension arc spans 1.48 m, but the front 0.42 m is **fold territory** — poses there exist as
isolated settled points, yet commanding a sweep into them snaps the parallel knee through
(toe x keeps growing *while the foot leaves the workspace*, which fakes speed). The
**traversable workspace is 1.04 m** (x ∈ [−0.62, +0.43] of the torso).

- **Pass 1, velocity limit only** (unlimited torque, just the 22 rad/s no-load joint speed through
  the measured path Jacobian): one-way traversal 72 ms → **14.5 m/s**.
- **Pass 2, real torque-speed envelope** (144.5 N·m → 0 at 22 rad/s, full peak when braking, real
  inertia, arc-tracking reference time-scaled to the fastest feasible cycle, best over a PD-gain
  sweep): **7.4 m/s** (1.10 m at 3.4 Hz, saturation 33%) — **torque saturation alone costs 49%**.
- **Pass 2 + actuator bandwidth** (measured MIT-mode 7 ms command→torque transport delay):
  **6.55 m/s** (2.9 Hz, saturation 93%, RMS 99 N·m ≫ 55 N·m continuous: burst only) — another
  12%. The same rig with servo-mode latency (200 ms) cannot track the arc at any tested speed.
  Each delay gets its own best gains — delay caps usable stiffness, which is exactly how
  bandwidth costs speed.

Consistent with `memory/hardware-speed-ceiling.md`'s static estimate (3.7 credible / 6.9
ceiling): this dynamic no-load bound sits just above the static ceiling, as it must. An
unconstrained bang-bang probe (travel allowed to leave the workspace between end poses) gives
7.65 m/s — kept in the json as a diagnostic upper probe only.

Outputs: `results/rail_bound.json`, `results/rail_pass1.mp4`, `results/rail_pass2.mp4` (a few
real-time cycles, then one at 10× slow-mo).

## Files

| File | Purpose |
|---|---|
| `build_leg2d.py` | Builds `model/leg2d.xml` from `training/model/dash01.xml` |
| `motor.py` | Real AKE90-8 torque-speed clamp + RMS thermal tracker |
| `gait.py` | Parametric stance/swing foot trajectory |
| `sweep.py` | Grid sweep driver + feasibility metrics |
| `plot_sweep.py` | Renders `results/sweep_heatmap.png` |
| `render_gait.py` | Renders one gait point to mp4 for visual sanity-checking |
| `rail_bound.py` | No-load rail-sweep upper bound on running speed (kinematic + torque-limited) |
| `plot_rail_sweep.py` | Figure: sweep extent with the two extreme poses superposed |
| `plot_rail_cycle.py` | Figure: joint/EE pos/vel/acc/torque over one cycle of the 7 ms-delay bound |
| `rail_sensitivity.py` | Bound sensitivity to leg inertia / rotor inertia / torque limit / bus voltage |
