"""Export a v2 run to a deployment bundle (numpy .npz, no code on load) — bundle version 2.

    python walk_v2/export.py --run walk_v2/runs/v2_s2_free_s0 --out robot/deploy/bundles/v2_s2_free_s0.npz

The Pi runtime for v2 (100 Hz: estimator + actor + gait.py + the latch + resync + the winding
observer) is a separate deployment step; this writes every number that runtime needs, read off
the LIVE env object exactly as robot/deploy/export_policy.py does for walk_mit, plus the numpy
reference of the control law is walk_v2/gait.py itself (xp=numpy). Arrays:

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
    meta = {
        "bundle_version": BUNDLE_VERSION, "run": Path(run).name, "step": int(agent.step),
        "spec_source": cfg.spec_source, "control_dt": float(env.control_dt),
        "frame_dim": 33, "history_len": int(cfg.history_len), "history_stride": int(cfg.history_stride),
        "once_dim": int(env.once_dim), "actor_dim": int(env.actor_dim), "action_dim": int(env.action_dim),
        "wrap_index": int(env.wrap_index), "obs_scales": {k: float(v) for k, v in cfg.obs_scales.items()},
        "clip_obs": 10.0, "obs_eps": 1e-8, "lp_yaw_tau_s": float(cfg.lp_yaw_tau_s),
        "task_brake_m": float(cfg.task_brake_m), "sprint_dist_m": float(cfg.sprint_dist_m),
        "gait": {k: (list(v) if isinstance(v, tuple) else v) for k, v in env.gp._asdict().items()},
        "drive_delay_ms": float(cfg.drive_delay_ms), "motor_kt_joint": list(cfg.motor_kt_joint),
        "motor_r_ohm": list(cfg.motor_r_ohm), "motor_bus_volts": float(cfg.motor_bus_volts),
        "thermal": {"tau_s": float(cfg.thermal_tau_s), "tau_cont": list(cfg.thermal_tau_cont),
                    "penalty_frac": float(cfg.thermal_penalty_frac)},
        "resync": {"kappa": float(cfg.resync_kappa), "window_cycle": float(cfg.resync_window_cycle),
                   "ema_cycles": float(cfg.resync_ema_cycles), "warmup_cycles": int(cfg.resync_warmup_cycles),
                   "enable": bool(cfg.resync_enable)},
        "term_height": float(cfg.term_height), "term_gravity_z": float(cfg.term_gravity_z),
        "height_stand": float(p.height_stand), "model_path": cfg.model_path,
        "actuator_names": p.actuator_names(), "planar": bool(p.planar),
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
