"""Cross-implementation trace: the numbers two implementations of the v2 design must agree on.

    python walk_v2/tools/trace.py --out walk_v2/results/trace_mjx.json [--preset v2_s2_free] [--ticks 300]
    python walk_v2/tools/trace.py --rig --out walk_v2/results/trace_mjx_rig.json      # welded in the air
    python walk_v2/tools/compare_traces.py walk_v2/results/trace_mjx.json <other_impl_trace.json>

Protocol (so the CPU/walk_mit arm can produce the same file):
  * plant: nominal (dr off, noise off, no pushes/wind/trips, thermal cold), delay 12 ms, reset at
    the keyframe with NO joint noise, the sim-only pitch assist at scale 1.0 (kp 100 / kd 10 on
    the base pitch DOF, exactly walk_mit's) so an open-loop gait stays up for the whole trace
  * actions: a FIXED spec committed at t0 (below), residual = 0.05 * sin(2 pi 3 t + j) per joint j
    (deterministic, no policy), for `ticks` ticks at 100 Hz
  * per tick, recorded AFTER the tick: t, phase, commit flag, spec (44), target (6), kp (6), kd (6),
    qpos (6 motor joints), qvel (6), tau (6, last substep), base z, base pitch, grav (3),
    gyro (3), grounded (2), reward, every reward term, thermal x (6), obs frame (33)
  * the FIXED spec: cam a1 = 0.3, thigh a1 = -0.2, all else 0, freq_raw = -0.2 (2.3 Hz)
  * --rig: the in-air test rig (walk_mit set_fixed_base): the base is re-pinned at the keyframe
    pose lifted 0.25 m, zero base velocity, after every tick; no ground contact, so an open-loop
    gait cycles for the whole trace and the physics stays close between implementations; the fixed
    pitch reflex is OFF on the rig (pitch_clip 0), the roll reflex gains are 0 in the fixed spec. On the
    ground (no --rig) the point-toe plant topples in ~0.5 s without a policy, so that trace is
    short by nature; both are worth comparing.

Differences to expect between MJX (float32) and classic MuJoCo (float64): the gait targets,
gains and reward terms evaluated on the SAME state agree to ~1e-6; the physics diverges slowly
(a chaotic biped) so qpos agreement is a divergence-time question. compare_traces.py reports
both: the per-state functional agreement (target/kp/kd/reward terms recomputed on the other
trace's state via walk_v2/gait.py in numpy) and the trajectory divergence time.
"""
import argparse
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np
import jax
import jax.numpy as jnp

import gait
from config import get_config
from env import DashEnvV2, EnvParams
from train import make_eval_env

FIXED_SPEC = np.zeros(44)
FIXED_SPEC[1] = 0.3        # cam a1
FIXED_SPEC[8] = -0.2       # thigh a1
FIXED_SPEC[35] = -0.2      # 2.3 Hz


def residual_at(t):
    return 0.05 * np.sin(2 * np.pi * 3.0 * t + np.arange(6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v2_s2_free")
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig", action="store_true", help="weld the base in the air (re-pinned every tick)")
    ap.add_argument("--assist", type=float, default=1.0, help="pitch assist scale (0 = the CPU arm's protocol)")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    cfg.reset_joint_noise = 0.0
    env = make_eval_env(cfg, n=1)
    cfg.pitch_assist_kp, cfg.pitch_assist_kd = 100.0, 10.0
    if args.rig:
        cfg.pitch_clip = 0.0          # the fixed pitch reflex is off on the rig (a pinned base chatters the sensed rate)
    env = make_eval_env(cfg, n=1, keep_assist=True)
    params = EnvParams.final(cfg)._replace(dr_scale=0.0, pitch_assist=float(args.assist), ctrl_jitter_ms=0.0,
                                           ctrl_drop_prob=0.0)
    state, obs = env.reset(jax.random.PRNGKey(0), params)
    p = env.plant
    base_idx = [p.base_q[n] for n in ("x", "y", "z", "roll", "pitch", "yaw") if p.base_q[n] >= 0]
    base_dofs = [p.base_d[n] for n in ("x", "y", "z", "roll", "pitch", "yaw") if p.base_d[n] >= 0]
    pin_q = np.asarray(p.key_qpos)[base_idx].copy()
    pin_q[base_idx.index(p.base_q["z"])] += 0.25

    def pin(state):
        q = state.data.qpos.at[0, jnp.asarray(base_idx)].set(jnp.asarray(pin_q, jnp.float32))
        v = state.data.qvel.at[0, jnp.asarray(base_dofs)].set(0.0)
        return state.replace(data=state.data.replace(qpos=q, qvel=v))

    if args.rig:
        state = pin(state)
    rows = []
    for i in range(args.ticks):
        t = i * env.control_dt
        a = np.concatenate([FIXED_SPEC, residual_at(t)])[None].astype(np.float32)
        commit = bool(np.asarray(state.commit)[0])
        state, obs, r, d, info = env.step(state, jnp.asarray(a), params)
        if args.rig:
            state = pin(state)
            d = jnp.zeros_like(d)
        dat = state.data
        cmd = np.asarray(state.cmd_buf[0, 0])
        R = np.asarray(dat.xmat[0, p.base_bid]).reshape(3, 3)
        rows.append(dict(
            t=round(t + env.control_dt, 4), commit=commit, phase=float(state.phase[0]),
            spec=np.asarray(state.spec[0]).round(6).tolist(),
            target=(cmd[:6] - np.asarray(state.draw.joint_zero[0])).round(6).tolist(),
            kp=cmd[6:12].round(4).tolist(), kd=cmd[12:18].round(4).tolist(),
            qpos=np.asarray(dat.qpos[0, p.act_qadr]).round(6).tolist(),
            qvel=np.asarray(dat.qvel[0, p.act_dadr]).round(5).tolist(),
            tau=np.asarray(dat.ctrl[0]).round(4).tolist(),
            base_z=float(dat.xpos[0, p.base_bid, 2]), base_pitch=float(dat.qpos[0, p.base_q["pitch"]]),
            grav=(R.T @ np.array([0, 0, -1.0])).round(5).tolist(),
            gyro=np.asarray(dat.sensordata[0, p.gyro_adr:p.gyro_adr + 3]).round(5).tolist(),
            grounded=np.asarray(state.grounded_prev[0]).astype(int).tolist(),
            reward=float(r[0]), terms={k: float(v[0]) for k, v in info["reward_terms"].items()},
            thermal=np.asarray(state.thermal_x[0]).round(6).tolist(),
            frame=np.asarray(state.hist[0, -1]).round(6).tolist(),
            done=bool(d[0]),
            cause={k: bool(info[k][0]) for k in ("term_low", "term_tip", "term_floor", "term_ws", "term_nan")},
        ))
        if bool(d[0]):
            rows[-1]["note"] = "episode ended on this tick; numbers are the auto-reset state"
            break
    out = dict(impl="walk_v2 (MJX/JAX)", rig=bool(args.rig), assist=float(args.assist), preset=args.preset, model=cfg.model_path, control_dt=env.control_dt,
               fixed_spec=FIXED_SPEC.tolist(), residual="0.05*sin(2*pi*3*t + j)", ticks=len(rows), rows=rows,
               nominal_ctrl=p.nominal_ctrl.tolist(), default_motor_pos=p.default_motor_pos.tolist(),
               gait_params=dict(env.gp._asdict(), pitch_reflex_rate_lp=float(cfg.pitch_reflex_rate_lp)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    cause = [k for k, v in rows[-1]["cause"].items() if v]
    print(f"[trace] wrote {args.out}: {len(rows)} ticks, ended {('by ' + ','.join(cause)) if rows[-1]['done'] else 'at the cap'}, "
          f"final z {rows[-1]['base_z']:.3f}")


if __name__ == "__main__":
    main()
