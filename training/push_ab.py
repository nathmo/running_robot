"""Backward push-recovery A/B: does standing lower actually buy backward capture?

THE CLAIM UNDER TEST. The stance leg is a fixed-length pendulum at ~99.6 % of its reach, so
footholds at floor level lie on one arc about the hip pivot, and the balanced stance sits 142 mm
behind that pivot -- i.e. 2.7 cm from the REAR end of its own arc and 31 cm from the front. Measured
closed form (see model/ws notes): forward capture 1.02 m/s, backward capture 0.09 m/s. Dropping the
ride height 4-5 cm moves the floor line up the arc and predicts backward capture 0.09 -> 0.67 m/s
while forward barely moves.

WHY THIS TEST AND NOT THE LADDER. That prediction is sagittal and one-sided, so it is falsifiable in
minutes on CPU, and the ladder's wall is m5 (roll) which this says nothing about. If the backward
curve below does not move, the analysis is wrong and no training run is worth launching.

WHAT MAKES IT AN EXPERIMENT RATHER THAN A DEMO:
  * FORWARD IS THE INTERNAL CONTROL. Every plant is pushed both ways on the same seeds. "Crouching
    helps balance" predicts both directions improve; the arc geometry predicts backward improves
    several-fold and forward improves slightly. Only the second is evidence for the mechanism.
  * dv = 0 IS A ROW. If the un-pushed robot does not survive the horizon, a push sweep measures
    nothing, so the no-push control is run and reported next to the rest.
  * PAIRED SEEDS + SURVIVOR COUNTS, never a bare mean. Outcomes on this robot are bimodal; an
    unpaired 5-episode sample once invented a 27x effect that does not exist ([[walk-fwd-lineage]]).
  * The push is the env's OWN push model (a velocity jump on base_x, as cfg.push_dv), so a number
    here means the same thing it means in every RL run these plants were graded on.
  * ACHIEVED CoM speed is measured and reported, not assumed equal to the commanded jump -- the
    contact takes some of it back, and the capture-point prediction is stated in CoM speed.

TWO CONTROLLERS, and they answer different questions:
  reflex     zero action + the plant's OWN built-in pitch reflex. No tuned gains, so nothing can be
             confounded by a gain set fitted to one stance. This is the gain-free probe, and it is
             the same test that caught the keyframe fix (0.91 s 0/6 -> 24.7 s 3/5).
  scripted   the hand-tuned controller. NOTE its gains are fitted to a particular stance, so on a
             crouched plant a DEGRADATION here is uninformative until you re-run scripted_walk
             --tune; an IMPROVEMENT at fixed gains is strong evidence. Read it in that direction
             only.

    python training/push_ab.py --plants model/dash01_bal.xml model/dash01_c50.xml
    python training/push_ab.py --controller scripted --gains scripted_gains_m3_bal.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import mujoco

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from config import get_config                                          # noqa: E402
from env import DashEnv                                                # noqa: E402
from scripted_walk import GAINS, ScriptedWalker                        # noqa: E402


def make_env(milestone, model_path, episode_s, controller, reflex=(2.0, 0.10, 0.25),
             resettle=True, key_ctrl=None):
    """See below. `key_ctrl` = (cam, thigh) overrides the stance and lets the env re-solve it."""
    return _make_env(milestone, model_path, episode_s, controller, reflex, resettle, key_ctrl)


def _make_env(milestone, model_path, episode_s, controller, reflex=(2.0, 0.10, 0.25),
              resettle=True, key_ctrl=None):
    """The graded plant with the RANDOM adversity switched off, so the only disturbance is ours.

    `reflex` mode reproduces render_keyframe_ab.py EXACTLY -- same rung, zero action, and the
    built-in pitch reflex forced to (kp, kd, clip) = (2.0, 0.10, 0.25). That is not cosmetic: the
    walk_fwd_m3 preset ships pitch_kd = 0.2, and at the preset value the un-pushed balanced plant
    falls in ~1.9 s instead of standing, which would leave this sweep with no control condition.
    Everything else about the config is left alone (cam_amp/thigh_amp stay at their preset 0.45),
    again to match the A/B that produced the 24.7 s baseline.

    `scripted` zeroes the gait generator and the built-in reflex exactly as scripted_walk.make_env
    does, so the hand-built controller owns all six targets.
    """
    cfg = get_config("walk_fwd_" + milestone)
    cfg.push_interval_s = 0.0
    cfg.trip_prob = 0.0
    if key_ctrl is not None:
        # Resettle EXACTLY ONCE, from the shipped keyframe. The constructor normally runs
        # _resettle_keyframe itself, so setting key_ctrl afterwards and calling it again is a
        # SECOND settle starting from the first one's output -- and it is not idempotent, so the
        # stance drifts (measured: toe-CoM +29.8 mm instead of ~0, and the control that holds 12 s
        # as a shipped plant died at 1.2 s). Suppress the constructor's pass and do it once here.
        cfg.ankle_resettle = False
    if not resettle:
        # A crouched keyframe is ALREADY an equilibrium of this ankle law, and
        # _resettle_keyframe is not idempotent: re-settling from it lands on the
        # four-bar's fold branch (base_z 0.962 -> 0.816, toe 441 mm ahead).
        cfg.ankle_resettle = False
    cfg.episode_s = float(episode_s)
    if model_path:
        cfg.model_path = str(model_path)
    if controller == "reflex":
        cfg.pitch_kp, cfg.pitch_kd, cfg.pitch_clip = (float(v) for v in reflex)
    else:
        cfg.residual_scale = 1.5
        cfg.cam_amp = cfg.thigh_amp = 0.0
        cfg.pitch_clip = 0.0
        cfg.reflex_kp_scale = cfg.reflex_kd_scale = cfg.reflex_bias_scale = 0.0
    env = DashEnv(cfg)
    env.set_dr_scale(0.0)
    if key_ctrl is not None:
        # CROUCH THE RIGHT WAY. Shipping a crouched keyframe as an XML variant does not work: the
        # stance a variant carries is not an equilibrium of the rung's plant (ankle_mode="rigid"
        # both welds the ankle and zeroes its spring), and _resettle_keyframe then re-solves it onto
        # the four-bar's fold branch — base_z 0.962 -> 0.816 with the toe 441 mm ahead, every seed
        # dead at 0.2 s. Solving the variant harder does not help either: a BALANCED stance is not a
        # static fixed point of the settle at all (measured — at toe-CoM ~ 0 the pose always drifts,
        # which is just an inverted pendulum being an inverted pendulum), so there is nothing to
        # converge to and every "solved" crouch was a 2 s snapshot of a drift.
        #
        # So do not hand the pipeline a stance — hand it a TARGET and let it solve its own, exactly
        # as it already does for the shipped one. Setting key_ctrl and re-running the env's own
        # _resettle_keyframe yields balanced crouches (toe-CoM within a few mm) down to base_z
        # 0.978, with height_target, _stand_torque and _ws_ref all re-derived consistently.
        cam, thigh = float(key_ctrl[0]), float(key_ctrl[1])
        env.model.key_ctrl[env.key_id] = np.array([0., cam, thigh, 0., -cam, -thigh])
        env.nominal_ctrl[:] = env.model.key_ctrl[env.key_id]
        env._resettle_keyframe()
    return env


def stance_of(env):
    """(base_z, toe-CoM) of the stance the env actually reset into — always report it, because a
    crouch that silently collapsed looks exactly like a crouch that failed on its merits."""
    d = env.data
    mujoco.mj_resetDataKeyframe(env.model, d, env.key_id)
    mujoco.mj_forward(env.model, d)
    toe = float(np.mean([d.geom_xpos[g][0] for g in env.foot_gids]))
    return float(env.height_target), toe - float(d.subtree_com[0][0])


def _recorder(env, path, fps=50):
    """Follow-cam frame grabber. Decimated to `fps` so the mp4 plays at REAL TIME -- one frame per
    200 Hz control step would be a correct and unplayable 200 fps file."""
    import imageio.v2 as imageio
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, 720)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, 560)
    ren = mujoco.Renderer(env.model, 560, 720)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.distance, cam.elevation, cam.azimuth = 2.8, -10, 90   # side-on: the sagittal step is the story
    every = max(1, int(round(1.0 / (env.control_dt * fps))))
    w = imageio.get_writer(str(path), fps=int(round(1.0 / (env.control_dt * every))))
    state = {"i": 0, "n": 0}

    def grab():
        state["i"] += 1
        if state["i"] % every == 0:
            cam.lookat[:] = env.data.qpos[:3]
            ren.update_scene(env.data, cam)
            w.append_data(ren.render())
            state["n"] += 1

    def close():
        # freeze on the final frame so a 1 s clip is watchable
        for _ in range(fps // 2):
            w.append_data(ren.render())
        w.close()
        return state["n"]

    return grab, close


def push_rollout(env, seed, dv, controller, gains, t_push=1.0, horizon_s=8.0, video=None):
    """One episode: settle, jump base_x by `dv` (negative = backward), then run to the horizon."""
    env.reset(seed=int(seed))
    if env.command_mode:
        env.set_command(0.0, 0.0)
    ctl = None
    if controller == "scripted":
        ctl = ScriptedWalker(env, gains, est="odom", v_des=0.0)
        ctl.reset()
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    dt = env.control_dt
    n_push, n_max = int(round(t_push / dt)), int(round(horizon_s / dt))

    com_x = lambda: float(env.data.subtree_com[0][0])
    dadr = env._base_x_dadr
    z0 = float(env.data.qpos[2])
    x_at_push, v_peak, foot_behind = None, 0.0, 0.0
    grab = close = None
    if video is not None:
        grab, close = _recorder(env, video)
    n, terminated = 0, False
    for k in range(n_max):
        if grab is not None:
            grab()
        if k == n_push:
            env.data.qvel[dadr] += float(dv)
            x_at_push = com_x()
        act = ctl() if ctl is not None else zero
        _, _, terminated, truncated, _ = env.step(act)
        n += 1
        if k >= n_push:
            v = float(env.data.qvel[dadr])
            v_peak = min(v_peak, v) if dv < 0 else max(v_peak, v)
            # How far BEHIND the CoM the rearmost foot got -- the mechanism, measured directly.
            # Only while the robot is still UP: once it is going down the legs splay and the geom
            # spread stops being a foothold, which is how this read 40 cm on a robot that never
            # took a backward step.
            if float(env.data.qpos[2]) > 0.9 * z0:
                cx = com_x()
                rear = min(float(env.data.geom_xpos[g][0]) for g in env.foot_gids)
                foot_behind = max(foot_behind, cx - rear)
        if terminated or truncated:
            break
    if close is not None:
        close()
    t_alive = n * dt
    return dict(seed=int(seed), dv=float(dv), t_alive=float(t_alive),
                t_after_push=float(max(t_alive - t_push, 0.0)),
                survived=bool(not terminated), v_peak=float(v_peak),
                foot_behind=float(foot_behind),
                travel=float(com_x() - x_at_push) if x_at_push is not None else 0.0)


def summarise(rows):
    surv = sum(r["survived"] for r in rows)
    return dict(n=len(rows), survived=surv,
                med_t=statistics.median(r["t_after_push"] for r in rows),
                med_vpeak=statistics.median(r["v_peak"] for r in rows),
                med_behind=statistics.median(r["foot_behind"] for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", nargs="+", default=["model/dash01.xml"],
                    help="model paths to compare; run them on the SAME seeds")
    ap.add_argument("--stances", nargs="+", default=None, metavar="CAM,THIGH",
                    help="stance targets as cam,thigh pairs; the env re-solves each into its own "
                         "balanced keyframe. This is how to sweep CROUCH — do not ship crouched "
                         "XML variants, they collapse on load (see _make_env)")
    ap.add_argument("--milestone", default="m3",
                    help="m3 = x/z/pitch free, roll+yaw locked — the sagittal rung this predicts")
    ap.add_argument("--controller", default="reflex", choices=["reflex", "scripted"])
    ap.add_argument("--gains", default=None, help="gains json for --controller scripted")
    ap.add_argument("--dv", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8],
                    help="push magnitudes (m/s); each is run BACKWARD and FORWARD")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--horizon", type=float, default=8.0)
    ap.add_argument("--t-push", type=float, default=1.0)
    ap.add_argument("--no-resettle", action="store_true",
                    help="load the keyframe as-written (crouched plants need this)")
    ap.add_argument("--reflex", type=float, nargs=3, default=[2.0, 0.10, 0.25],
                    metavar=("KP", "KD", "CLIP"),
                    help="built-in pitch reflex gains for --controller reflex; the default is "
                         "the set render_keyframe_ab.py used, NOT the walk_fwd_m3 preset value")
    ap.add_argument("--video-dir", default=None,
                    help="record seed 0 of every backward condition to this directory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gains = dict(GAINS)
    if args.gains:
        gains.update(json.loads((PKG_DIR / args.gains).read_text()))
    seeds = list(range(args.seeds))
    mags = sorted(set(abs(v) for v in args.dv))
    out = []

    combos = [(p, None) for p in args.plants]
    if args.stances:
        combos = [(args.plants[0], tuple(float(v) for v in st.split(",")))
                  for st in args.stances]

    for plant, kc in combos:
        env = make_env(args.milestone, plant, args.horizon + 1.0, args.controller,
                       reflex=args.reflex, resettle=not args.no_resettle, key_ctrl=kc)
        bz, tc = stance_of(env)
        tag = plant if kc is None else ("stance cam %+.4f thigh %+.4f" % kc)
        print("\n=== %s   [%s, %s, %d paired seeds, horizon %.0f s]"
              % (tag, args.milestone, args.controller, len(seeds), args.horizon))
        print("    settled: base_z %.4f  toe-CoM %+.1f mm%s ==="
              % (bz, tc * 1000,
                 "   *** NOT BALANCED — collapsed, ignore this row ***" if abs(tc) > 0.05 else ""))
        print("   push        BACKWARD                          FORWARD")
        print("   dv (m/s)    surv   med s   CoM v   foot-behind |  surv   med s   CoM v")
        for mag in mags:
            cells = {}
            for sgn in ((-1, 1) if mag > 0 else (-1,)):
                vid = None
                if args.video_dir and sgn < 0:
                    vd = Path(args.video_dir); vd.mkdir(parents=True, exist_ok=True)
                    lbl = "plant" if kc is None else ("cam%+.3f" % kc[0])
                    vid = vd / ("%s_%s_back%03.0f.mp4" % (lbl, args.controller, mag * 100))
                rows = [push_rollout(env, s, sgn * mag, args.controller, gains,
                                     t_push=args.t_push, horizon_s=args.horizon,
                                     video=(vid if s == seeds[0] else None)) for s in seeds]
                cells[sgn] = summarise(rows)
                out.append(dict(plant=plant, stance=kc, base_z=bz, toe_com=tc,
                                milestone=args.milestone,
                                controller=args.controller, dv=sgn * mag,
                                rows=rows, **cells[sgn]))
            b = cells[-1]
            f = cells.get(1)
            line = ("   %6.2f     %d/%d   %5.2f  %+5.2f      %5.1f cm  "
                    % (mag, b["survived"], b["n"], b["med_t"], b["med_vpeak"],
                       b["med_behind"] * 100))
            if f:
                line += "|  %d/%d   %5.2f  %+5.2f" % (f["survived"], f["n"], f["med_t"],
                                                      f["med_vpeak"])
            else:
                line += "|   (no-push control)"
            print(line)
            sys.stdout.flush()

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print("\nwrote " + args.out)
    print("\nRead the BACKWARD column against the FORWARD one. A plant that improves both is just "
          "\nbetter balanced; the arc-geometry claim is specifically that BACKWARD improves several-"
          "\nfold while forward barely moves. Anything smaller than the seed-to-seed spread is not "
          "\na result.")


if __name__ == "__main__":
    main()
