# BalanceRL — push recovery for DASH-01

A side quest off `RLframework/`: one policy that keeps the robot **standing** while it is shoved,
robust to a ±3 cm CoM shift and to the walker's full sensor-noise model. It writes the MIT
force-control frame directly: a position target, Kp and Kd per joint, every 10 ms. There is **no
torque feed-forward**, because the drives' torque span has not been measured (`controller/deploy/mit.py`,
rule 2).

## Layout

| file | what |
|---|---|
| `config.py` | every tunable + presets (`balance`, `smoke`) |
| `plant.py`, `drive.py`, `model/` | copied from RLframework; the plant draw adds the whole-robot CoM shift |
| `env.py` | `BalanceEnv`: action → MIT frame → drive (delay, back-EMF clamp) → MJX → noise chain → obs; pushes; per-push survival verdicts |
| `networks.py` | estimator 420→128→64→3, actor [420+3]→256→256→18, critic |
| `ppo.py` | pmap PPO, adaptive lr, symmetry loss, the two gated curricula, greedy push-ladder eval |
| `train.py` | entry point; `--resume auto` |
| `policy_io.py` | load a checkpoint without the trainer |
| `export.py` | checkpoint → **v3 bundle** (`kind: balance`) for the Pi |
| `verify_bundle.py` | runs the sim and the numpy runtime on the same noisy measurements, diffs obs/action/target/gains |
| `tools/push_envelope.py` | survival by push size × direction × CoM offset (+ `--baseline` = zero action) |
| `slurm/izar_balance.sh` | Izar launcher: `SEEDS × LINKS` chained 4.5 h jobs on 2 × V100 |

## The task

* **Action (18)** = `[q 6 | kp 6 | kd 6]`, clipped to [-1, 1].
  `q_ref = clip(nominal_cmd + q_scale·a + reflex, joint range)`, then a one-pole filter (τ 80 ms)
  and the no-load slew cap. The **reflex** is a stabilising prior the policy adds to:
  `thigh ∓(0.3·g_x + 0.06·ω_y)`, from the newest measured frame, clipped to ±0.25 rad.
  `kp = kp0·(500/kp0)^a` for a ≥ 0 and `kp0·(kp0/20)^a` below (kd: 0.2–5). **a = 0 is exactly the
  standing command** (kp 120/200, kd 4/5).
* **Observation (436)** = 10 frames, 20 ms apart, of `[q − q_stance, q̇, τ, gravity, gyro, previous action]`,
  plus a 16-wide **slow block**: leaky integrals (τ 2 s, gyro 0.3 s) of measured gravity xy, gyro xy,
  joint error and torque. Without it the DC lean that cancels a CoM offset is not inferable — see below.
  passed through RLframework's measurement chain (encoder / velocity / torque noise and bias, IMU
  noise, bias, random walk, mount rotation, accelerometer leak, dropout). The critic also sees
  base velocity, height, contacts, the CoM shift, the push force and the plant draw.
* **Push**: a 50–150 ms force pulse on the torso. Azimuth is uniform, elevation ±30°, and it lands
  at a random point on the torso (so it also torques it). Its size is the impulse as a whole-robot
  Δv = J/M. A push is **survived** if the robot is still up 2 s after it started.
* **Curricula**, independent of each other:
  1. plant width (CoM shift + DR) 0.2 → 1.0 on a **clock**, over the first 20 M steps;
  2. the push level from 0.15 m/s, ×1.08 while ≥ 80% of *hard* pushes (drawn from [L/2, L]) are
     survived, ×0.95 below 50% — gated on push verdicts, never on episode length. 30% of pushes stay
     in [0, L/2] so small pushes aren't forgotten.

  **Both of those are corrections** (see below): in the first campaign the plant width was gated on
  "≤ 5% of episodes fall with no push", which stalled at 0.50–0.55, and the push level was chained
  behind it, so it never left 0.15 m/s in 171 M steps.
* **Reward**: alive + upright + height, plus (once a push is 1 s over) staying still and returning to
  the stance pose; minus tilt rate, torque², action rate, foot slip, mean Kp (high gains cost
  nothing in the sim but are not free on the robot), and joint-limit proximity. Fall = −20, terminal.

## Measured before training (CPU, 32 envs, 6 s)

| condition | upright |
|---|---|
| nominal plant, zero action | 32/32 |
| **full plant (DR + CoM ±3 cm), zero action** | **1/32** |
| 20% plant, action noise std 0.03 (at 0.8 rad/unit) | 11/32 |
| 20% plant, action noise std 0.10 | 0/32 |
| 20% plant, pushes ≤ 0.3 m/s, zero action | 15 of 45 pushes survived |

So the fixed stance cannot carry the CoM shift, and the stand is knife-edged under position noise.

## What this plant actually is (measured 2026-09-23) — read before changing anything

1. **The stance is PASSIVELY stable.** The zero action (a position-held PD at the stance command)
   stands 32/32 for 6 s on the nominal plant with sensor noise on. No attitude feedback is required
   to stand still.
2. **The standing basin is ±0.02 rad** of commanded thigh. Every constant lean 0.02 rad away from the
   right one falls within 1–2 s.
3. **Cancelling a CoM offset is a DC lean of ~0.04 rad of thigh per 3 cm** (thigh is the fore-aft
   lever: 0.1 rad moves the foot centre ~8 cm; cam is height, it moves the CoM/foot relation 1.2 mm
   over 0.6 rad).
4. **Attitude feedback on measured gravity makes it WORSE.** kp 1.0 / kd 0.05 stood 9/32 at CoM 0
   where doing nothing stood 32/32: the IMU mount error (1° = 0.017 rad), gravity bias (0.01) and
   joint homing (0.009 rad) are each basin-sized, and a stiff kp puts them straight in the command.
5. So the controller wants: **hold the pose, apply a slow inferred DC lean, damp rates gently.** The
   DC lean cannot be read off one frame (the calibration error dominates it), which is why the slow
   block exists, and per-tick exploration noise is nearly pure disturbance, which is why the action
   filter exists and why exploration is 0.01 rad.
6. **−3 cm (CoM toward the heels) is near a kinematic limit**: it needs the feet ~3 cm back and
   flat-sole travel backward is only ~3.3 cm. Expect an asymmetric envelope.

## Campaign 1 (bal_s0/s1/s2, 2026-09-22, ~171 M steps each at 15.5k sps on 2 x V100) — what it taught

Cancelled rather than finished. Three things to keep:

1. **A gated plant width starves everything behind it.** quiet-fall rate sat at ~10% against a 5%
   gate, so plant_scale stopped at 0.50–0.55 and the push level, chained behind it, never left its
   0.15 m/s floor. Both are fixed above: the plant is a clock and the pushes are on their own gate.
2. **std 0.05 on the position dims (0.025 rad) cannot find a recovery step.** 171 M steps of it
   produced a stander that still lost 20% of 0.15 m/s pushes. Now 0.12 (0.06 rad), gains 0.25.
3. **w_gain 0.1 bought a cheap pose**: the policy walked Kp from 200 down to 92–130, giving away the
   authority a recovery needs. Now 0.02.

Evaluation of that campaign, for the record (greedy, full plant): 20% survival at 0.3 m/s, 0% above,
and ~20% of episodes fell with no push at all.

## Campaign 2 (bal2_s0/s1/s2, 2026-09-23, 200 M steps each) — what it taught

The plant clock worked (full width by 20 M). The push level STILL never promoted (`n_promote: 0`):
hard-push survival sat at 0.58–0.68 against the 0.80 gate. Final eval 27% at 0.3 m/s, 5% at 0.6, 0
above, 21% quiet falls. Raising exploration from 0.025 to 0.06 rad made episodes SHORTER (ep_len 517
vs 851): three times the basin width is a disturbance, not exploration. That result is what sent the
investigation into the plant measurements above.

## Campaign 3 (bal5_s0/s1/s2, 2026-09-24, 200 M steps each) — the result

The first campaign that worked. All three seeds agree, and the push curriculum MOVED for the first
time (13–15 promotions, 0.15 → 0.23 m/s). At the end: ep_len 620–720 of 1200, unprompted falls 4–7%
of episodes, Kp settled at 181–237 (the policy chooses a STIFFER stance than the 200 default).

Greedy eval (full plant, CoM drawn in the ±3 cm box, noise on), the three seeds:

| push | 0.1 | 0.2 | 0.3 | 0.5 | 0.8 | ≥1.0 |
|---|---|---|---|---|---|---|
| survival | 76–83% | 32–50% | 26–42% | 5–20% | 0–6% | 0% |

`envelope_best.png` is the corner-case version (CoM pinned at each ±3 cm corner, 8 directions, one
push per trial), beside `envelope_baseline.png` (the zero action) — the policy is 3× the baseline at
0.1 m/s and 8× at 0.3 m/s. Two structural facts show in it:

* **LATERAL pushes are survived far better than fore-aft** (left/right 65–80% at 0.2–0.3 m/s vs
  5–10% front/back). The stance is 38 cm wide and the sole is 8.6 cm long; that ratio is the answer.
* **the ceiling is ~0.2–0.3 m/s, and it is the CoP budget, not the policy.** A 0.3 m/s shove is
  4.3 N·s; with a 3 cm CoM offset eating most of the ±4.3 cm margin, the ~1.3 cm left gives ~2.3 N,
  which needs ~2 s to arrest a topple that completes in ~0.4 s. **Past ~0.2 m/s recovery requires a
  STEP**, and nothing in this action space discovers a coordinated 200 ms swing from per-tick noise
  small enough to keep the ±0.02 rad basin. That is the next piece of work: a capture-step prior,
  the same prior-plus-residual pattern as the pitch reflex one level up.

Deployed artifact: `controller/deploy/bundles/bal5_s0.npz` (v3, parity PASS, mock dry run PASS).

## Workflow

```bash
# train (Izar): three seeds, 4 x 4.5 h each
BalanceRL/slurm/izar_balance.sh
# evaluate
python BalanceRL/tools/push_envelope.py --run BalanceRL/runs/bal_s0            # best.*
python BalanceRL/tools/push_envelope.py --run BalanceRL/runs/bal_s0 --baseline # zero action
# export + parity check (fails the export if the runtime disagrees with the sim)
python BalanceRL/export.py --run BalanceRL/runs/bal_s0 --out controller/deploy/bundles/bal_s0.npz
# dress rehearsal on the mock bus
python controller/deploy/run_policy.py --bundle controller/deploy/bundles/bal_s0.npz --mock --max-seconds 5
```

On the robot the bundle runs through the same `run_policy.py` / web UI daemon path as every other
bundle: the approach crawls to the stance, then `controller_balance.py` takes over under the safety
governor. The panel shows "no command", and End-run / Kill are the only inputs.
