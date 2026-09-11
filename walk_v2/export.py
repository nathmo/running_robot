"""Export a v2 run to a deployment bundle (numpy .npz, no code on load) — bundle version 2.

    python walk_v2/export.py --run walk_v2/runs/v2_s2_free_s0 --out robot/deploy/bundles/v2_s2_free_s0.npz

The Pi runtime for v2 is `robot/deploy/controller_v2.py` (100 Hz: estimator + actor + the latch +
the free-running clock + gait_v2.py, under the same safety governor as v1). This writes every
number that runtime needs, read off the LIVE env object exactly as robot/deploy/export_policy.py
does for walk_mit -- never re-derived from the config, which is how four separate eval-restore
bugs happened in this project. `robot/deploy/bundle.py` states which meta keys the runtime
requires (REQUIRED_META_V2); adding a field the runtime reads means adding it here. Arrays:

    est_w0/b0, est_w1/b1, est_w2/b2      estimator 377 -> 128 -> 64 -> 3   (weights as [out, in])
    pi_w0/b0, pi_w1/b1, act_w/act_b      actor [377 + 3] -> 256 -> 256 -> 50, tanh on both hiddens
    obs_mean, obs_var                    actor slice of the running stats (clip 10)
    nominal_ctrl, default_motor_pos, q_lo, q_hi, motor_vel_limit, forcerange, drive_kp, drive_kd,
    stand_torque, hist_idx, latched_dims
meta: layout, control_dt, gait params (gait.GaitParams._asdict()), delay, thermal constants,
resync parameters, thresholds.
"""
import argparse
import json
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np

from evaluate import load_run

BUNDLE_VERSION = 2


def export(run, out, checkpoint=None):
    cfg, env, agent = load_run(run, checkpoint, n_envs=1)
    P = agent.params["params"]
    f32 = np.float32
    W = lambda mod, i: np.asarray(P[mod][f"Dense_{i}"]["kernel"], f32).T      # flax: [in, out] -> [out, in]
    B = lambda mod, i: np.asarray(P[mod][f"Dense_{i}"]["bias"], f32)
    p = env.plant
    arrays = {
        "est_w0": W("estimator", 0), "est_b0": B("estimator", 0),
        "est_w1": W("estimator", 1), "est_b1": B("estimator", 1),
        "est_w2": W("estimator", 2), "est_b2": B("estimator", 2),
        "pi_w0": W("policy_net", 0), "pi_b0": B("policy_net", 0),
        "pi_w1": W("policy_net", 1), "pi_b1": B("policy_net", 1),
        "act_w": np.asarray(P["action_net"]["kernel"], f32).T, "act_b": np.asarray(P["action_net"]["bias"], f32),
        "obs_mean": np.asarray(agent.stats.mean, np.float64)[:env.actor_dim],
        "obs_var": np.asarray(agent.stats.var, np.float64)[:env.actor_dim],
        "nominal_ctrl": p.nominal_ctrl.astype(np.float64),
        "default_motor_pos": p.default_motor_pos.astype(np.float64),
        "q_lo": p.q_lo.astype(np.float64), "q_hi": p.q_hi.astype(np.float64),
        "motor_vel_limit": np.asarray(cfg.motor_vel_limit, np.float64),
        "forcerange": p.tau_peak.astype(np.float64),
        "drive_kp": np.asarray(cfg.drive_kp, np.float64), "drive_kd": np.asarray(cfg.drive_kd, np.float64),
        "stand_torque": p.stand_torque.astype(np.float64),
        "hist_idx": env.hist_idx.astype(np.int32),
        "latched_dims": env.latched_dims.astype(np.int32),
        "log_std": np.asarray(P["log_std"], f32),
    }
    # The base DOFs the plant railed, in the runtime's order (X Y Z roll pitch yaw): the planar
    # model has no base y / roll / yaw joints at all, so a policy trained on it has never had to
    # stabilise them and the panel must say so before anyone stands the robot up.
    base_lock = [False, bool(p.planar), False, bool(p.planar), False, bool(p.planar)]
    # THE GAIT BLOCK IS THE CURRICULUM'S, NOT THE CONFIG'S. `freq_lo` is a curriculum value under
    # the frequency-floor presets: it ramps from gait_freq_lo_start down to gait_freq_hz[0] on the
    # step clock, and _step_one maps freq_raw through the CURRENT value. A checkpoint taken
    # mid-ramp was trained against a different frequency map from the one the config describes, so
    # exporting the config's would deploy a policy whose whole gait clock is mis-scaled -- and it
    # would look like a slow gait, not like a bug. The ramp is a pure function of the step count
    # (PPO._clock), so it is reproduced here exactly rather than assumed to be finished.
    gait_block = {k: (list(v) if isinstance(v, tuple) else v) for k, v in env.gp._asdict().items()}
    freq_floor_steps = int(getattr(cfg, "gait_freq_floor_steps", 0) or 0)
    if freq_floor_steps > 0:
        lo0, lo1 = float(cfg.gait_freq_lo_start), float(cfg.gait_freq_hz[0])
        frac = min(1.0, int(agent.step) / freq_floor_steps)
        gait_block["freq_lo"] = lo0 + frac * (lo1 - lo0)
        if frac < 1.0:
            print(f"[export] frequency-floor curriculum still ramping at {agent.step:,} of "
                  f"{freq_floor_steps:,} steps: this checkpoint's gait clock spans "
                  f"[{gait_block['freq_lo']:.2f}, {gait_block['freq_hi']:.2f}] Hz, not "
                  f"[{lo1:.2f}, {gait_block['freq_hi']:.2f}]")
    meta = {
        "bundle_version": BUNDLE_VERSION, "run": Path(run).name, "step": int(agent.step),
        "checkpoint": Path(checkpoint).name if checkpoint else "step_{}".format(agent.step),
        "spec_source": cfg.spec_source, "control_dt": float(env.control_dt),
        "nu": int(p.nominal_ctrl.size),
        "frame_dim": 33, "history_len": int(cfg.history_len), "history_stride": int(cfg.history_stride),
        "once_dim": int(env.once_dim), "actor_dim": int(env.actor_dim), "action_dim": int(env.action_dim),
        "wrap_index": int(env.wrap_index), "obs_scales": {k: float(v) for k, v in cfg.obs_scales.items()},
        "clip_obs": 10.0, "obs_eps": 1e-8, "lp_yaw_tau_s": float(cfg.lp_yaw_tau_s),
        # read off the weights, not the config: a layer-size drift then cannot be a silent one
        "est_hidden": [int(arrays["est_w0"].shape[0]), int(arrays["est_w1"].shape[0])],
        "policy_hidden": [int(arrays["pi_w0"].shape[0]), int(arrays["pi_w1"].shape[0])],
        "task_brake_m": float(cfg.task_brake_m), "sprint_dist_m": float(cfg.sprint_dist_m),
        "objective": str(cfg.objective),
        "gait": gait_block,
        "gait_freq_floor_steps": freq_floor_steps,
        # NOT in GaitParams but read by the control law every tick: the EMA on the pitch reflex's
        # rate term (0 = the raw rate; both settings exist across the lineage and the difference
        # is 0.4 rad of thigh target), the commanded-target slew cap, and the series order.
        "pitch_reflex_rate_lp": float(cfg.pitch_reflex_rate_lp),
        "motor_accel_limit": float(cfg.motor_accel_limit),
        "n_harmonics": int(cfg.n_harmonics),
        "residual_scale": float(cfg.residual_scale), "action_scale": float(cfg.action_scale),
        "drive_delay_ms": float(cfg.drive_delay_ms), "motor_kt_joint": list(cfg.motor_kt_joint),
        "motor_r_ohm": list(cfg.motor_r_ohm), "motor_bus_volts": float(cfg.motor_bus_volts),
        "thermal": {"tau_s": float(cfg.thermal_tau_s), "tau_cont": list(cfg.thermal_tau_cont),
                    "penalty_frac": float(cfg.thermal_penalty_frac)},
        "resync": {"kappa": float(cfg.resync_kappa), "window_cycle": float(cfg.resync_window_cycle),
                   "ema_cycles": float(cfg.resync_ema_cycles), "warmup_cycles": int(cfg.resync_warmup_cycles),
                   "enable": bool(cfg.resync_enable)},
        # did this checkpoint ever see a red light? The run/stop button on the robot drives task[0],
        # which is exactly the stoplight signal -- a checkpoint trained with stoplight_prob 0 has
        # only ever seen task[0] drop once, at the finish line, where every runner so far falls.
        "stop": {"stoplight_prob_final": float(cfg.stoplight_prob_final),
                 "stoplight_curriculum_steps": int(cfg.stoplight_curriculum_steps),
                 "stop_decel_s": float(cfg.stop_decel_s),
                 "stop_speed_eps": float(cfg.stop_speed_eps), "stop_hold_s": float(cfg.stop_hold_s)},
        "term_height": float(cfg.term_height), "term_gravity_z": float(cfg.term_gravity_z),
        "height_stand": float(p.height_stand), "model_path": cfg.model_path,
        "actuator_names": p.actuator_names(), "planar": bool(p.planar), "base_lock": base_lock,
        "policy_std_mean": float(np.exp(arrays["log_std"]).mean()),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    print(f"[export] wrote {out} ({out.stat().st_size / 1e3:.0f} kB): actor {env.actor_dim} -> {env.action_dim}, "
          f"std {meta['policy_std_mean']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    export(args.run, args.out, args.checkpoint)


if __name__ == "__main__":
    main()
