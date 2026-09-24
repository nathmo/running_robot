"""Export a balance checkpoint to a deployment bundle -- bundle version 3, kind "balance".

    python BalanceRL/export.py --run BalanceRL/runs/bal_s0 [--ckpt best] \
        --out controller/deploy/bundles/bal_s0.npz

The Pi runtime is controller/deploy/controller_balance.py (estimator + actor + the MIT-frame map + the
slew cap, under the same safety governor as every bundle). Every number it needs is read off the
LIVE env object and the checkpoint's own sidecar, never re-derived from a config default.

Arrays (weights as [out, in], float32):
    est_w0/b0 .. est_w2/b2    estimator 420 -> 128 -> 64 -> 3
    pi_w0/b0, pi_w1/b1        actor [420 + 3] -> 256 -> 256, tanh on both
    act_w/act_b               256 -> 18   (the MIT frame [q 6 | kp 6 | kd 6])
    obs_mean, obs_var         actor slice of the running stats
    nominal_ctrl, default_motor_pos, q_lo, q_hi, q_scale, motor_vel_limit, forcerange,
    drive_kp, drive_kd (the gains at a = 0), hist_idx
After writing, the bundle is loaded back through controller/deploy/bundle.py and the parity check
(verify_bundle.py) is run unless --no-verify.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import numpy as np                                    # noqa: E402

from policy_io import load_policy                     # noqa: E402

BUNDLE_VERSION = 3


def export(run, out, ckpt=None):
    cfg, env, pol = load_policy(run, ckpt, n_envs=1)
    P = pol.params["params"]
    f32 = np.float32
    W = lambda mod, i: np.asarray(P[mod][f"Dense_{i}"]["kernel"], f32).T
    B = lambda mod, i: np.asarray(P[mod][f"Dense_{i}"]["bias"], f32)
    p = env.plant
    s = pol.side["stats"]
    arrays = {
        "est_w0": W("estimator", 0), "est_b0": B("estimator", 0),
        "est_w1": W("estimator", 1), "est_b1": B("estimator", 1),
        "est_w2": W("estimator", 2), "est_b2": B("estimator", 2),
        "pi_w0": W("policy_net", 0), "pi_b0": B("policy_net", 0),
        "pi_w1": W("policy_net", 1), "pi_b1": B("policy_net", 1),
        "act_w": np.asarray(P["action_net"]["kernel"], f32).T, "act_b": np.asarray(P["action_net"]["bias"], f32),
        "obs_mean": np.asarray(s["mean"], np.float64)[:env.actor_dim],
        "obs_var": np.asarray(s["var"], np.float64)[:env.actor_dim],
        "nominal_ctrl": np.asarray(p.nominal_ctrl, np.float64),
        "default_motor_pos": np.asarray(p.default_motor_pos, np.float64),
        "q_lo": np.asarray(p.q_lo, np.float64), "q_hi": np.asarray(p.q_hi, np.float64),
        "q_scale": np.asarray(cfg.q_scale, np.float64),
        "motor_vel_limit": np.asarray(cfg.motor_vel_limit, np.float64),
        "forcerange": np.asarray(p.tau_peak, np.float64),
        "drive_kp": np.asarray(cfg.drive_kp, np.float64), "drive_kd": np.asarray(cfg.drive_kd, np.float64),
        "hist_idx": np.asarray(env.hist_idx, np.int32),
    }
    run = Path(run)
    side = pol.side
    meta = {
        "bundle_version": BUNDLE_VERSION, "kind": "balance",
        "run": run.name, "checkpoint": pol.path.name, "step": int(side["step"]),
        "nu": 6, "control_dt": float(env.control_dt), "frame_dim": int(env.frame_dim),
        "history_len": int(cfg.history_len), "history_stride": int(cfg.history_stride),
        "actor_dim": int(env.actor_dim), "action_dim": int(env.action_dim),
        "once_dim": int(env.once_dim),
        "slow": {"slow_s": float(cfg.ema_slow_s), "mid_s": float(cfg.ema_mid_s),
                 "fast_s": float(cfg.ema_fast_s)},
        "action_filter_tau_s": float(cfg.action_filter_tau_s),
        "obs_scales": dict(cfg.obs_scales), "clip_obs": float(cfg.clip_obs), "obs_eps": float(cfg.obs_eps),
        "gains": {"kp_lo": cfg.kp_lo, "kp_hi": cfg.kp_hi, "kd_lo": cfg.kd_lo, "kd_hi": cfg.kd_hi},
        "motor_accel_limit": float(cfg.motor_accel_limit),
        "term_gravity_z": float(cfg.term_gravity_z),
        "est_hidden": [int(arrays["est_w0"].shape[0]), int(arrays["est_w1"].shape[0])],
        "policy_hidden": [int(arrays["pi_w0"].shape[0]), int(arrays["pi_w1"].shape[0])],
        "command": {"kind": "none", "v_max": 0.0},
        # the stabilising prior is PART OF THE CONTROL LAW: the runtime must reproduce it exactly
        "reflex": {"enable": bool(cfg.reflex_enable), "kp_pitch": float(cfg.reflex_kp_pitch),
                   "kd_pitch": float(cfg.reflex_kd_pitch), "kp_roll": float(cfg.reflex_kp_roll),
                   "kd_roll": float(cfg.reflex_kd_roll), "clip_rad": float(cfg.reflex_clip_rad)},
        "base_lock": [0, 0, 0, 0, 0, 0],
        "objective": "balance",
        "drive_delay_ms": [cfg.drive_delay_range_ms[0], cfg.drive_delay_range_ms[1]],
        "trained": {
            "push_level_mps": float(side["cur"]["push_level"]),
            "plant_scale": float(side["cur"]["plant_scale"]),
            "com_shift_m": list(cfg.com_shift_m),
            "hold_s_range": list(cfg.hold_s_range),
            "eval": side.get("eval"),
        },
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None, help="best (default), final, ckpt_<n>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    out, meta = export(a.run, a.out, a.ckpt)
    sys.path.insert(0, str(REPO / "controller" / "deploy"))
    from bundle import Bundle
    b = Bundle.load(out)
    print(f"wrote {out}: v{b.version} {meta['kind']}, {meta['run']}/{meta['checkpoint']} @ {meta['step']:,} "
          f"steps, push level {meta['trained']['push_level_mps']:.2f} m/s, actor {b.n_actor}, action {b.action_dim}")
    if not a.no_verify:
        from verify_bundle import verify
        ok = verify(a.run, out, a.ckpt)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
