"""Write the CPU stack's trace in the walk_v2 (MJX/JAX port) cross-implementation protocol.

    python walk_mit/trace_mjx.py --out walk_mit/golden/trace_cpu.json [--preset v2_s2_clean] [--ticks 300]
    python walk_v2/tools/compare_traces.py walk_v2/results/trace_mjx.json walk_mit/golden/trace_cpu.json

The protocol (walk_v2/tools/trace.py): nominal plant (DR off, noise off, no pushes/wind/trips,
thermal cold), 12 ms delay, reset at the keyframe with NO joint noise, pitch assist 0; a FIXED
spec (cam a1 0.3, thigh a1 -0.2, freq_raw -0.2 = 2.3 Hz) committed at t0 and re-committed
unchanged at every wrap, residual 0.05 * sin(2 pi 3 t + j); per tick, recorded AFTER the tick:
t, commit (the flag the tick started with), phase, spec, target (this tick's command, pre-delay,
minus the homing offset), kp, kd, the 6 motor qpos/qvel, tau (actuator force), base z / pitch,
grav, gyro, grounded, reward + every term, thermal state, the newest obs frame, done + cause.

`gait_params` is filled from this side's Config with the walk_v2 field names, so their
compare_traces.py can recompute our targets with THEIR law on OUR states (the functional
agreement test) and report the trajectory divergence time.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import gait_v2  # noqa: E402
from config import get_config  # noqa: E402
from env import DashEnv  # noqa: E402

FIXED_SPEC = np.zeros(44)
FIXED_SPEC[1] = 0.3        # cam a1
FIXED_SPEC[8] = -0.2       # thigh a1
FIXED_SPEC[35] = -0.2      # 2.3 Hz


def residual_at(t):
    return 0.05 * np.sin(2 * np.pi * 3.0 * t + np.arange(6))


def gait_params(cfg, env):
    return dict(cam_amp=float(cfg.cam_amp), thigh_amp=float(cfg.thigh_amp), roll_amp=float(cfg.roll_amp),
                delta_max=float(cfg.delta_max_rad), o_max=[float(x) for x in cfg.offset_max_rad],
                imp_kp_up=float(cfg.imp_kp_up), imp_kp_dn=float(cfg.imp_kp_dn),
                imp_kd_up=float(cfg.imp_kd_up), imp_kd_dn=float(cfg.imp_kd_dn),
                reflex_kp_scale=float(cfg.reflex_kp_scale), reflex_kd_scale=float(cfg.reflex_kd_scale),
                reflex_bias_scale=float(cfg.reflex_bias_scale),
                pitch_kp=float(cfg.pitch_kp), pitch_kd=float(cfg.pitch_kd), pitch_bias=float(cfg.pitch_bias),
                pitch_clip=float(cfg.pitch_clip), residual_scale=float(cfg.residual_scale),
                freq_lo=float(cfg.gait_freq_hz[0]), freq_hi=float(cfg.gait_freq_hz[1]),
                drive_kp=[float(x) for x in env.model.actuator_gainprm[:6, 0]],
                drive_kd=[float(-x) for x in env.model.actuator_biasprm[:6, 2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v2_s2_clean")
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = get_config(args.preset)
    cfg.reset_joint_noise = 0.0
    cfg.dr_delay_ms_range = (0.0, 0.0)
    cfg.drive_delay_ms = 12.0
    env = DashEnv(cfg)
    env.reset(seed=0)
    rows = []
    for i in range(args.ticks):
        t = i * env.control_dt
        a = np.concatenate([FIXED_SPEC, residual_at(t)]).astype(np.float32)
        commit = bool(env._commit_flag)
        obs, r, term, trunc, info = env.step(a)
        d = env.data
        cmd = env._ring[(env._ring_head - 1) % env._ring_len]      # this tick's command, pre-delay
        R = env._base_rot()
        z = float(d.qpos[2])
        rows.append(dict(
            t=round(t + env.control_dt, 4), commit=commit, phase=float(env._phase),
            spec=env._spec_live.round(6).tolist(),
            target=(cmd[:6] - env._noise.zero_offset[:6]).round(6).tolist(),
            kp=cmd[6:12].round(4).tolist(), kd=cmd[12:18].round(4).tolist(),
            qpos=d.qpos[env.act_qadr].round(6).tolist(),
            qvel=d.qvel[env.act_dadr].round(5).tolist(),
            tau=d.actuator_force[:6].round(4).tolist(),
            base_z=z, base_pitch=float(d.qpos[4]),
            grav=(R.T @ np.array([0, 0, -1.0])).round(5).tolist(),
            gyro=env._ang_vel_body().round(5).tolist(),
            grounded=env._grounded_prev.astype(int).tolist(),
            reward=float(r), terms={k: float(v) for k, v in info["reward_terms"].items()},
            thermal=env._theta.round(6).tolist(),
            frame=env._history[-1].round(6).tolist(),
            done=bool(term or trunc),
            cause=dict(term_low=bool(z < cfg.term_height),
                       term_tip=bool(env._gravity_body()[2] > cfg.term_gravity_z),
                       term_floor=bool(env._floor_violation()), term_ws=bool(env._workspace_violation()),
                       term_nan=bool(not np.all(np.isfinite(d.qpos)))),
        ))
        if term or trunc:
            break
    out = dict(impl="walk_mit (classic MuJoCo, CPU)", preset=args.preset, model=cfg.model_path,
               control_dt=env.control_dt, fixed_spec=FIXED_SPEC.tolist(),
               residual="0.05*sin(2*pi*3*t + j)", ticks=len(rows), rows=rows,
               nominal_ctrl=np.asarray(env.nominal_ctrl).tolist(),
               default_motor_pos=np.asarray(env.default_motor_pos).tolist(),
               gait_params=gait_params(cfg, env),
               notes="tau = actuator_force (position actuators); target = pre-delay command; "
                     "resync kappa %.2f; armature %s" % (env._kappa, list(cfg.drive_armature)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    cause = [k for k, v in rows[-1]["cause"].items() if v]
    print(f"[trace] wrote {args.out}: {len(rows)} ticks, ended {('by ' + ','.join(cause)) if rows[-1]['done'] else 'at the cap'}, "
          f"final z {rows[-1]['base_z']:.3f}, commits {sum(r['commit'] for r in rows)}")


if __name__ == "__main__":
    main()
