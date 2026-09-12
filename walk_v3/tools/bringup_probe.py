"""Bring-up probe: start the policy with the base held on its stand, then let go.

The deployment question this answers (2026-09-10): on the real robot the operator holds the
base with the feet on the floor and the trunk roughly vertical, starts the policy, and releases
it a few seconds later. Training always starts from the settled keyframe at rest, so the held
seconds -- clock running, legs cycling, feet scuffing, body velocity pinned at zero -- are off
distribution, and so is the release transient (the legs take the full weight in one tick).

    python walk_v3/tools/bringup_probe.py --run walk_v3/runs/v2c_s2_free_dp2x_s0 --sweep basic
    python walk_v3/tools/bringup_probe.py --run ... --hold 3.0 --pitch 10 --video drop.mp4

`cfg.hold_enable` clamps the six base DOFs at every 1 kHz substep for `hold_s` seconds (an
infinitely stiff hand; no fall is scored while held), then releases with zero base velocity.
The hold height is solved so the lowest toe just touches the floor at the requested tilt, plus
`--dz` (positive = released from that far up, i.e. the operator lets go early).
"""
import argparse
import json
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import jax
import jax.numpy as jnp

from config import config_from_dict
from env import DashEnvV2, EnvParams
from ppo import PPO
from train import make_eval_env, latest_checkpoint

DEG = np.pi / 180.0


def load(run, checkpoint=None, n_envs=16, keep_assist=False, dr=False):
    run = Path(run)
    cfg = config_from_dict(json.loads((run / "resolved_config.json").read_text())["config"])
    cfg.hold_enable = True
    ck = Path(checkpoint) if checkpoint else latest_checkpoint(run)
    if ck is None:
        raise FileNotFoundError(f"no checkpoint in {run}")
    env = DashEnvV2(cfg, n_envs=n_envs) if dr else make_eval_env(cfg, n_envs, keep_assist=keep_assist)
    agent = PPO(cfg, env, run, cfg.total_steps, seed=0, eval_env=None)
    agent.load(ck)
    print(f"[bringup] {ck.name} @ {agent.step:,} steps  plant {cfg.model_path}"
          f"  {'randomized (DR, noise, pushes)' if dr else 'nominal'}")
    return cfg, env, agent


def touch_height(cfg, pitch, roll):
    """Base z that puts the lowest toe exactly on the floor at this tilt (keyframe legs)."""
    import mujoco
    from plant import resolve
    m = mujoco.MjModel.from_xml_path(resolve(cfg.model_path))
    d = mujoco.MjData(m)
    O = mujoco.mjtObj
    kid = mujoco.mj_name2id(m, O.mjOBJ_KEY, cfg.keyframe)
    mujoco.mj_resetDataKeyframe(m, d, kid)
    jid = {n: mujoco.mj_name2id(m, O.mjOBJ_JOINT, "base_" + n) for n in ("z", "roll", "pitch")}
    adr = {n: int(m.jnt_qposadr[j]) for n, j in jid.items() if j >= 0}
    z_key = float(d.qpos[adr["z"]])
    d.qpos[adr["pitch"]] = pitch
    if "roll" in adr:
        d.qpos[adr["roll"]] = roll
    mujoco.mj_forward(m, d)
    gids = [mujoco.mj_name2id(m, O.mjOBJ_GEOM, "foot_" + s + "_col") for s in "LR"]
    r = float(m.geom_size[gids[0], 0])
    low = min(float(d.geom_xpos[g, 2]) - r for g in gids)
    return z_key - low, z_key


def make_runner(env, agent, n_max):
    """One compiled greedy rollout; hold_s / hold_z / tilt are traced, so a whole sweep of
    release conditions reuses the same executable (the scan itself costs ~2 min to compile)."""
    dt = env.control_dt
    act = agent._act_greedy
    n = env.n_envs
    foot_gids = jnp.asarray(env.plant.foot_gids)

    def body(carry, _):
        (state, obs, alive, dist, fell, tend, tq_hold, viol_hold, slip_hold, vmax, freq_h,
         sat_h, term, params, policy_at) = carry
        # before policy_at the action is zero = the neutral spec: no gait, the nominal stance
        # held by the drive PD with the pitch/roll reflex on top (the deploy runtime's APPROACH
        # pose). policy_at = 0 is the plain "policy from tick 0" bring-up.
        a = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        a = jnp.where((state.t >= policy_at)[:, None], a, 0.0)
        held = alive & (state.t < params.hold_s - 0.5 * dt)
        toe0 = state.data.geom_xpos[foot_gids][:, :2]
        state2, obs2, r, done, info = env.step(state, a, params)
        vx = (info["sprint_d"] - dist) / dt
        ending = alive & done
        dist = jnp.where(alive, info["sprint_d"], dist)
        fell = fell | (ending & info["fallen"])
        why = jnp.stack([info["term_low"], info["term_tip"], info["term_floor"],
                         info["term_ws"], info["term_nan"]], axis=-1).astype(jnp.float32)
        term = jnp.where(ending[:, None], why, term)
        tend = jnp.where(ending, state.t + dt, tend)
        # held-phase diagnostics
        toe1 = state2.data.geom_xpos[foot_gids][:, :2]
        slip = jnp.sum(jnp.linalg.norm(toe1 - toe0, axis=-1)) / dt
        tq_hold = jnp.where(held, jnp.maximum(tq_hold, info["torque_util"]), tq_hold)
        viol_hold = viol_hold | (held & (info["term_floor"] | info["term_ws"]))
        slip_hold = jnp.where(held, jnp.maximum(slip_hold, slip), slip_hold)
        freq_h = jnp.where(held, info["freq_hz"], freq_h)
        sat_h = jnp.where(held, jnp.maximum(sat_h, info["residual_sat"]), sat_h)
        vmax = jnp.where(alive & ~held, jnp.maximum(vmax, vx), vmax)
        alive2 = alive & ~done
        return (state2, obs2, alive2, dist, fell, tend, tq_hold, viol_hold, slip_hold, vmax,
                freq_h, sat_h, term, params, policy_at), state.data.qpos[0]

    @jax.jit
    def run(key, params, policy_at):
        state, obs = env.reset(key, params)
        init = (state, obs, jnp.ones(n, bool), jnp.zeros(n), jnp.zeros(n, bool),
                jnp.full(n, n_max * dt), jnp.zeros(n), jnp.zeros(n, bool), jnp.zeros(n),
                jnp.full(n, -1e9), jnp.zeros(n), jnp.zeros(n), jnp.zeros((n, 5)),
                params, policy_at)
        carry, qs = jax.lax.scan(body, init, None, length=n_max)
        keys = ("dist", "fell", "t_end", "tq_hold", "viol_hold", "slip_hold", "vmax",
                "freq_hold", "sat_hold", "term")
        return dict(zip(keys, carry[3:13])), carry[2], qs

    return run


def rollout(run, agent, seed, hold_s, hold_z, pitch, roll, dr=False, policy_at=0.0, dr_scale=1.0,
            start_red=0.0, rv=(0.0, 0.0, 0.0)):
    """Greedy rollout: held for hold_s, then free plant to the end of the compiled window."""
    params = EnvParams.final(agent.cfg)._replace(
        dr_scale=dr_scale if (dr and agent.cfg.dr_enable) else 0.0,
        ctrl_jitter_ms=agent.cfg.ctrl_jitter_ms_final * dr_scale if dr else 0.0,
        ctrl_drop_prob=agent.cfg.ctrl_drop_prob_final * dr_scale if dr else 0.0,
        pitch_assist=0.0, stoplight_prob=0.0,
        hold_s=float(hold_s), hold_z=float(hold_z), hold_pitch=float(pitch), hold_roll=float(roll),
        start_red_s=float(start_red), release_vx=float(rv[0]), release_vy=float(rv[1]),
        release_vz=float(rv[2]))
    st, alive, qs = run(jax.random.PRNGKey(seed), params, jnp.float32(policy_at))
    st = {k: np.asarray(v) for k, v in st.items()}
    st["alive"] = np.asarray(alive)
    st["surv"] = st["t_end"] - hold_s              # seconds upright after the release
    return st, np.asarray(qs)


TERMS = ("low", "tipped", "floor", "workspace", "NaN")


def report(tag, st, after_s):
    s = np.minimum(st["surv"], after_s)
    ok = st["surv"] >= after_s - 1e-6
    why = np.asarray(st["term"])[~ok]
    hows = " ".join("{} {}".format(int(why[:, i].sum()), TERMS[i])
                    for i in range(5) if why[:, i].sum() > 0)
    print("  {:<32} survive {:2d}/{:2d}  upright after release {:5.2f} s (min {:4.2f} max {:5.2f})"
          "  dist {:5.1f} m  vmax {:4.2f} m/s".format(
              tag, int(ok.sum()), len(ok), float(s.mean()), float(s.min()), float(s.max()),
              float(st["dist"].mean()), float(np.median(st["vmax"]))) + ("  [" + hows + "]" if hows else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--hold", type=float, default=3.0, help="seconds on the stand before release")
    ap.add_argument("--after", type=float, default=8.0, help="seconds of free plant after release")
    ap.add_argument("--pitch", type=float, default=0.0, help="base pitch while held (deg, + = nose DOWN / leaning forward)")
    ap.add_argument("--roll", type=float, default=0.0, help="base roll while held (deg)")
    ap.add_argument("--dz", type=float, default=0.0, help="mm above the touching height at release")
    ap.add_argument("--sweep", default=None, choices=["hold", "phase", "tilt", "dz", "handover", "early", "drpair", "stopped",
                             "pitchfine", "rollfine", "dzfine", "combo", "all"])
    ap.add_argument("--video", default=None)
    ap.add_argument("--npz", default=None, help="single mode: save env 0's qpos trajectory")
    ap.add_argument("--policy-at", type=float, default=0.0,
                    help="hand over to the policy at this time (s); zero action = stance hold "
                         "before it. 1e9 = never (pure stance-hold bring-up)")
    ap.add_argument("--start-red", type=float, default=0.0,
                    help="bring up with the run flag DOWN (task[0]=0, the runtime's STOPPED "
                         "start) and turn it green this many seconds in")
    ap.add_argument("--dr-scale", type=float, default=1.0,
                    help="with --dr: randomization strength (the curriculum value; 1.0 = final)")
    ap.add_argument("--dr", action="store_true",
                    help="bring up on the randomized training plant (DR, sensor noise, pushes)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg, env, agent = load(args.run, args.checkpoint, n_envs=args.episodes, dr=args.dr)
    z_touch, z_key = touch_height(cfg, 0.0, 0.0)
    print("[bringup] keyframe base z {:.4f} m, feet-touching z {:.4f} m".format(z_key, z_touch))
    holds = dict(hold=[0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0], phase=[], tilt=[args.hold],
                 dz=[args.hold], all=[0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    hold_max = max(holds.get(args.sweep, [args.hold]) + ([args.hold + 0.3] if args.sweep in
                                                         ("phase", "all") else [args.hold]))
    n_max = int(round((hold_max + args.after) / env.control_dt))
    print("[bringup] window {:.1f} s = {} ticks ({} envs), one compile for the sweep"
          .format(hold_max + args.after, n_max, args.episodes))
    run = make_runner(env, agent, n_max)

    def run_one(hold, pitch, roll, dz, qpos=False, rv=(0.0, 0.0, 0.0)):
        zt, _ = touch_height(cfg, pitch * DEG, roll * DEG)
        st, qs = rollout(run, agent, args.seed, hold, zt + dz * 1e-3, pitch * DEG, roll * DEG,
                         dr=args.dr, policy_at=args.policy_at, dr_scale=args.dr_scale,
                         start_red=args.start_red, rv=rv)
        return (st, qs) if qpos else st

    out, t0 = {}, __import__("time").time()

    def note(key, tag, st, held=True):
        out[key] = {k: np.asarray(v).tolist() for k, v in st.items()}
        report(tag, st, args.after)
        if held:
            print("      while held: torque util {:.2f}, toe slip {:.2f} m/s, clock {:.2f} Hz, "
                  "residual sat {:.2f}, limit flags {}  [{:.0f} s]".format(
                      float(st["tq_hold"].mean()), float(st["slip_hold"].mean()),
                      float(st["freq_hold"].mean()), float(st["sat_hold"].mean()),
                      int(st["viol_hold"].sum()), __import__("time").time() - t0))

    if args.sweep in ("hold", "all"):
        print("\n[hold duration]  upright, released at the feet-touching height")
        for h in holds["hold"]:
            note("hold_{}".format(h), "held {:5.1f} s".format(h), run_one(h, 0.0, 0.0, args.dz), h > 0)
    if args.sweep in ("phase", "all"):
        print("\n[release phase]  hold {:.2f} s + one gait cycle in 0.03 s steps".format(args.hold))
        for k in range(11):
            h = args.hold + 0.03 * k
            note("phase_{:.2f}".format(h), "released at {:5.2f} s".format(h),
                 run_one(h, 0.0, 0.0, args.dz), False)
    if args.sweep in ("tilt", "all"):
        print("\n[release attitude]  hold {:.1f} s. + pitch = leaning FORWARD (nose down); "
              "roll lifts one foot (35 mm at 5 deg, 70 mm at 10 deg)".format(args.hold))
        for pitch, roll in [(-10, 0), (-5, 0), (0, 0), (5, 0), (10, 0), (0, 5), (0, 10),
                            (0, -10), (7, 7), (-7, 7)]:
            note("tilt_p{}_r{}".format(pitch, roll), "pitch {:+3d} roll {:+3d} deg".format(pitch, roll),
                 run_one(args.hold, pitch, roll, args.dz), False)
    if args.sweep == "pitchfine":
        print("\n[pitch tolerance]  hold {:.0f} s, both feet flat, + = leaning FORWARD".format(args.hold))
        for pitch in [-10, -8, -6, -4, -2, -1, 0, 1, 2, 3, 4, 5]:
            note("pf_{}".format(pitch), "pitch {:+5.1f} deg ({})".format(
                pitch, "forward" if pitch > 0 else ("back" if pitch < 0 else "level")),
                run_one(args.hold, pitch, 0.0, args.dz), False)
    if args.sweep == "rollfine":
        print("\n[roll tolerance]  hold {:.0f} s; roll lifts one foot off the floor".format(args.hold))
        for roll in [-5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5]:
            note("rf_{}".format(roll), "roll {:+5.1f} deg (one foot {:+.0f} mm)".format(
                roll, 1000 * abs(touch_height(cfg, 0.0, roll * DEG)[0] - z_touch) * 2),
                run_one(args.hold, 0.0, roll, args.dz), False)
    if args.sweep == "dzfine":
        print("\n[height tolerance]  hold {:.0f} s, upright; 0 = feet just touching".format(args.hold))
        for dz in [-10, -7, -5, -2, 0, 10, 20, 25, 30, 35, 40]:
            note("df_{}".format(dz), "height {:+5.0f} mm".format(dz),
                run_one(args.hold, 0.0, 0.0, dz), False)
    if args.sweep == "combo":
        print("\n[combined tilt, then release velocity]  hold {:.0f} s".format(args.hold))
        for pitch, roll in [(0, 0), (-2, 1), (-2, 2), (0, 1), (0, 2), (2, 1), (-4, 2)]:
            note("cb_{}_{}".format(pitch, roll), "pitch {:+3.0f} roll {:+3.0f} deg".format(pitch, roll),
                run_one(args.hold, pitch, roll, args.dz), False)
        for v in [(0, 0, -0.05), (0, 0, -0.1), (0, 0, -0.2), (0, 0, -0.4), (0.1, 0, 0), (-0.1, 0, 0),
                  (0.2, 0, 0), (0, 0.1, 0), (0, 0.2, 0)]:
            note("rv_{}".format(v), "release v ({:+.2f} {:+.2f} {:+.2f}) m/s".format(*v),
                run_one(args.hold, 0.0, 0.0, args.dz, rv=v), False)
    if args.sweep == "stopped":
        print("[STOPPED bring-up]  the deployed runtime comes up with the run flag down; "
              "held {:.0f} s, then released, then RUN".format(args.hold))
        for green in [1e9, args.hold + 5.0, args.hold + 2.0, args.hold + 0.5, args.hold]:
            args.start_red = green
            tag = ("never pressed RUN" if green > 1e8 else
                   "RUN at {:+.1f} s after release".format(green - args.hold))
            note("stopped_{}".format(green), tag, run_one(args.hold, 0.0, 0.0, args.dz), False)
        args.start_red = 0.0
    if args.sweep == "drpair":
        print("\n[randomization pair]  normal start vs held-then-released, same plants")
        for scale in [0.0, 0.25, 0.5, 0.75, 1.0]:
            args.dr_scale = scale
            for h in (0.0, args.hold):
                note("drpair_{}_{}".format(scale, h),
                     "dr_scale {:4.2f}  {}".format(scale, "no hold" if h == 0 else "held {:.0f} s".format(h)),
                     run_one(h, 0.0, 0.0, args.dz), False)
    if args.sweep == "early":
        print("\n[early release]  how long may the policy run before the operator lets go?")
        for h in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]:
            note("early_{}".format(h), "held {:5.2f} s".format(h), run_one(h, 0.0, 0.0, args.dz), h > 0)
    if args.sweep == "handover":
        print("\n[hand-over]  stance hold (zero action) while held and after the release, "
              "policy starts later")
        for extra in [1e9, 3.0, 1.0, 0.5, 0.2, 0.0]:
            at = args.hold + extra if extra < 1e8 else 1e9
            args.policy_at = at
            tag = ("stance hold only" if extra > 1e8 else
                   "policy {:+.1f} s after release".format(extra))
            note("handover_{}".format(extra), tag, run_one(args.hold, 0.0, 0.0, args.dz), True)
        args.policy_at = 0.0
    if args.sweep in ("dz", "all"):
        print("\n[release height]  hold {:.1f} s, upright (- = feet pressed into the floor)"
              .format(args.hold))
        for dz in [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 40.0, 100.0]:
            note("dz_{}".format(dz), "released {:+5.0f} mm".format(dz),
                 run_one(args.hold, 0.0, 0.0, dz), True)
    if args.sweep is None:
        st, qs = run_one(args.hold, args.pitch, args.roll, args.dz, qpos=True)
        note("single", "held {:.1f} s".format(args.hold), st)
        for i in range(args.episodes):
            print("    ep{:02d}: {:7s} at t {:5.2f} s ({:5.2f} s after release)  {:5.1f} m  "
                  "vmax {:5.2f} m/s".format(
                      i, "FELL" if st["fell"][i] else "upright", float(st["t_end"][i]),
                      float(min(st["surv"][i], args.after)), float(st["dist"][i]),
                      float(st["vmax"][i])))
        if args.npz:
            np.savez(args.npz, qpos=qs, dt=env.control_dt, hold_s=args.hold,
                     t_end=st["t_end"], term=st["term"])
            print("    wrote {} ({} ticks)".format(args.npz, len(qs)))
        if args.video:
            from evaluate import render_video
            render_video(cfg, qs, float(st["t_end"][0]), args.video,
                         seconds=min(float(st["t_end"].max()), args.hold + args.after))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
