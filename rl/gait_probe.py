"""Gait probe: quantify HOW a policy locomotes, not just how much reward it earns.

The v2 walking policy scored ~1700 reward while translating entirely by sliding its loaded feet
(duty factor 0.74-0.83, median 'swing' = one control step, in-contact slip ~0.2 m/s). Reward
curves can't tell skating from walking — these metrics can, so they are the acceptance gates
for any policy that should go near the hardware.

  # probe the trained policy at 0.5 m/s and at stand
  .venv/Scripts/python.exe -m rl.gait_probe --run rl/runs/m2_walk

Gates (walk @ 0.5 m/s unless noted):
  duty factor <= 0.70 per foot          fraction of steps with the foot grounded
  both-feet-grounded fraction <= 0.50
  median swing air time >= 0.20 s       (swings longer than one control step)
  mean in-contact toe slip <= 0.07 m/s
  |mean vx - cmd| <= 0.15 m/s
  stand drift <= 0.15 m per 10 s        (stand condition)
  cam/thigh RMS torque <= 55 Nm         (AKE90-8 continuous rating)
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np

from .evaluate import build, set_command, infer_preset


def probe_condition(model, venv, raw, vx_cmd, episodes, max_steps, deterministic=True):
    """Metrics come from the env's OWN gait state (raw._grounded_prev / raw._air_time, updated
    with substep-accumulated contact) so the probe measures the same 'grounded' the reward
    machinery gates on — a boundary-sampled re-derivation would classify sub-20 ms hops
    differently than the terms being validated.

    Sampling happens in the env's per-CONTROL-step hook (raw.on_control_step), not per env.step():
    in fourier mode one step() replays a whole gait cycle, so per-step() polling saw the gait
    state only at cycle boundaries (duty/swing/slip garbage) and counted cycles against a
    control-step budget. The hook also fires while raw.data still holds the true state — the
    VecEnv auto-reset only swaps in a fresh keyframe after step() returns."""
    dt = raw.control_dt
    duty = np.zeros(2)
    counts = dict(both=0, neither=0, total=0)
    touchdowns = np.zeros(2)
    air_times = []
    slips = []
    fwd_vels = []
    torques = []
    ep_lens, drift_10s = [], []
    st = {}   # per-episode sampling state: prev grounded/air/xy + control-step count

    def on_ctrl():
        if st["n"] >= max_steps:      # hold the step budget exactly (a fourier macro-step
            return                    # would otherwise overshoot it mid-cycle)
        grounded = raw._grounded_prev.copy()          # this control step's grounded, env-defined
        air_now = raw._air_time.copy()
        xy = raw.data.geom_xpos[raw.foot_gids_arr, 0:2].copy()
        v = np.linalg.norm(xy - st["prev_xy"], axis=1) / dt
        for i in range(2):
            if grounded[i]:
                duty[i] += 1
                if st["prev_grounded"][i]:            # both-ends gating, like the slip term
                    slips.append(float(v[i]))
                if st["prev_air"][i] > 0 and air_now[i] == 0:
                    touchdowns[i] += 1
                    air_times.append(float(st["prev_air"][i]))
        counts["both"] += bool(grounded.all())
        counts["neither"] += bool(~grounded.any())
        counts["total"] += 1
        fwd_vels.append(float((raw._base_rot().T @ raw.data.qvel[0:3])[0]))
        torques.append(np.abs(raw.data.actuator_force[:raw.nu]))
        st["prev_grounded"], st["prev_air"], st["prev_xy"] = grounded, air_now, xy
        st["last_xy"] = raw.data.qpos[0:2].copy()
        st["n"] += 1

    raw.on_control_step = on_ctrl
    for _ in range(episodes):
        obs = venv.reset()
        set_command(raw, vx_cmd, 0.0)
        start = raw.data.qpos[0:2].copy()
        st.update(n=0, prev_grounded=raw._grounded_prev.copy(), prev_air=raw._air_time.copy(),
                  prev_xy=raw.data.geom_xpos[raw.foot_gids_arr, 0:2].copy(), last_xy=start.copy())
        done = [False]
        while not done[0] and st["n"] < max_steps:
            a, _ = model.predict(obs, deterministic=deterministic)
            obs, _, done, _ = venv.step(a)
            set_command(raw, vx_cmd, 0.0)
        ep_lens.append(st["n"])                       # control steps, in pd AND fourier mode
        dist = float(np.linalg.norm(st["last_xy"] - start))
        drift_10s.append(dist / max(st["n"] * dt, 1e-9) * 10.0)
    raw.on_control_step = None
    total = counts["total"]
    swings = [a for a in air_times if a > dt * 1.5]     # real swings, not 1-step chatter
    tq = np.sqrt(np.mean(np.square(torques), axis=0)) if torques else np.zeros(raw.nu)
    act_names = [mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                 for i in range(raw.nu)]
    seconds = total * dt
    return dict(
        cmd_vx_mps=vx_cmd * raw.cfg.vx_max,
        mean_vx_mps=round(float(np.mean(fwd_vels)), 3) if fwd_vels else 0.0,
        duty_factor=[round(float(d / max(total, 1)), 3) for d in duty],
        frac_both=round(counts["both"] / max(total, 1), 3),
        frac_neither=round(counts["neither"] / max(total, 1), 3),
        touchdowns_per_s=[round(float(t / max(seconds, 1e-9)), 2) for t in touchdowns],
        swing_air_s=dict(n=len(swings),
                         median=round(float(np.median(swings)), 3) if swings else 0.0,
                         mean=round(float(np.mean(swings)), 3) if swings else 0.0),
        contact_slip_mps=dict(mean=round(float(np.mean(slips)), 3) if slips else 0.0,
                              p90=round(float(np.percentile(slips, 90)), 3) if slips else 0.0),
        rms_torque_nm={n: round(float(t), 1) for n, t in zip(act_names, tq)},
        ep_len=[int(e) for e in ep_lens],
        max_steps=max_steps,
        drift_m_per_10s=[round(d, 3) for d in drift_10s],
    )


def gates(walk, stand):
    cam_thigh = [v for k, v in walk["rms_torque_nm"].items()
                 if k.startswith(("cam", "thigh"))]
    checks = {
        "duty_factor<=0.70": max(walk["duty_factor"]) <= 0.70,
        "frac_both<=0.50": walk["frac_both"] <= 0.50,
        "median_swing>=0.20s": walk["swing_air_s"]["median"] >= 0.20,
        "slip_mean<=0.07": walk["contact_slip_mps"]["mean"] <= 0.07,
        "vx_err<=0.15": abs(walk["mean_vx_mps"] - walk["cmd_vx_mps"]) <= 0.15,
        "cam_thigh_rms<=55Nm": max(cam_thigh) <= 55.0,
        "stand_drift<=0.15m/10s": max(stand["drift_m_per_10s"]) <= 0.15,
        "no_falls": min(walk["ep_len"] + stand["ep_len"]) >= walk["max_steps"],
    }
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--preset", default=None, help="default: inferred from the run name")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=500, help="max control steps per episode")
    ap.add_argument("--vx", type=float, default=0.5, help="walk command in [-1,1]")
    ap.add_argument("--stochastic", action="store_true",
                    help="ALSO probe sampled actions (training-time behavior)")
    args = ap.parse_args()
    run = Path(args.run)
    preset = args.preset or infer_preset(run)

    model, venv, raw = build(run, preset, args.checkpoint)
    out = {"walk": probe_condition(model, venv, raw, args.vx, args.episodes, args.steps),
           "stand": probe_condition(model, venv, raw, 0.0, args.episodes, args.steps)}
    if args.stochastic:
        out["walk_stochastic"] = probe_condition(model, venv, raw, args.vx, args.episodes,
                                                 args.steps, deterministic=False)
    out["gates"] = gates(out["walk"], out["stand"])
    out["verdict"] = "PASS" if all(out["gates"].values()) else "FAIL"
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
