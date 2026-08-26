"""Side-on PNGs of the stance a plant ACTUALLY resets into, plus the numbers under it.

Written after a silent failure that a picture would have caught in one look: crouched keyframes
built against the spring ankle were loaded into the rigid-ankle rung, env._resettle_keyframe moved
them, and the robot reset into a folded leg with the toe 56 cm forward. Every seed died at 0.2 s and
the table just said 0.00 — the pose was never looked at.

So this renders the pose the ENV produces on reset (not the pose the XML claims), optionally after
letting it run for a while, and annotates each frame with base_z, toe-vs-CoM and the stance search
delta the env chose.

    python training/render_pose.py --plants model/dash01_bal.xml model/dash01_r50.xml
    python training/render_pose.py --plants model/dash01_r50.xml --after 2.0    # 2 s in
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from config import get_config                                          # noqa: E402
from env import DashEnv                                                # noqa: E402

W, H = 720, 560


def shot(plant, milestone, seed, after_s, outdir, reflex=(2.0, 0.10, 0.25), resettle=True,
         key_ctrl=None, label=None):
    cfg = get_config("walk_fwd_" + milestone)
    cfg.model_path = str(plant)
    cfg.push_interval_s = 0.0
    cfg.trip_prob = 0.0
    if not resettle:
        # A crouched keyframe is ALREADY an equilibrium of this ankle law, and
        # _resettle_keyframe is not idempotent: re-settling from it lands on the
        # four-bar's fold branch (base_z 0.962 -> 0.816, toe 441 mm ahead).
        cfg.ankle_resettle = False
    cfg.pitch_kp, cfg.pitch_kd, cfg.pitch_clip = (float(v) for v in reflex)
    env = DashEnv(cfg)
    env.set_dr_scale(0.0)
    if key_ctrl is not None:
        # sweep the stance the way push_ab does: hand the env a TARGET and let it re-solve its own
        # balanced keyframe. Shipping a crouched XML variant collapses on load.
        cam, thigh = float(key_ctrl[0]), float(key_ctrl[1])
        env.model.key_ctrl[env.key_id] = np.array([0., cam, thigh, 0., -cam, -thigh])
        env.nominal_ctrl[:] = env.model.key_ctrl[env.key_id]
        env._resettle_keyframe()
    env.reset(seed=int(seed))
    if env.command_mode:
        env.set_command(0.0, 0.0)

    term = False
    t = 0.0
    if after_s > 0:
        a = np.zeros(env.action_space.shape, np.float32)
        for _ in range(int(round(after_s / env.control_dt))):
            _, _, term, trunc, _ = env.step(a)
            t += env.control_dt
            if term or trunc:
                break

    mujoco.mj_forward(env.model, env.data)
    toe = float(np.mean([env.data.geom_xpos[g][0] for g in env.foot_gids]))
    com = env.data.subtree_com[0]
    stats = dict(base_z=float(env.data.qpos[2]), toe_com=toe - float(com[0]),
                 delta=getattr(env, "stance_search_delta", None), t=t, fell=bool(term))

    # the model ships a 640x480 offscreen framebuffer; raise it rather than shrinking the shot
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, W)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, H)
    renderer = mujoco.Renderer(env.model, H, W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.distance, cam.elevation, cam.azimuth = 2.6, -8, 90      # side-on: the sagittal pose is the story
    cam.lookat[:] = (float(env.data.qpos[0]), 0.0, 0.55)
    renderer.update_scene(env.data, cam)
    img = renderer.render()

    name = (label or Path(plant).stem) + ("_t%.1f" % after_s if after_s > 0 else "_reset") + ".png"
    out = Path(outdir) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    imageio.imwrite(str(out), img)
    print("%-26s base_z %.4f | toe-CoM %+6.1f mm | search_delta %s | %s -> %s"
          % (label or Path(plant).name, stats["base_z"], stats["toe_com"] * 1000, stats["delta"],
             ("FELL at %.2f s" % t) if stats["fell"] else ("held %.1f s" % t), out))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", nargs="+", required=True)
    ap.add_argument("--milestone", default="m3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--after", type=float, default=0.0,
                    help="seconds of zero-action sim before the shot (0 = the reset pose)")
    ap.add_argument("--stances", nargs="+", default=None, metavar="CAM,THIGH",
                    help="stance targets; the env re-solves each into its own balanced keyframe")
    ap.add_argument("--no-resettle", action="store_true",
                    help="load the keyframe as-written (needed for crouched plants)")
    ap.add_argument("--outdir", default=str(PKG_DIR / "milestones" / "poses"))
    args = ap.parse_args()
    if args.stances:
        for st in args.stances:
            kc = tuple(float(v) for v in st.split(","))
            shot(args.plants[0], args.milestone, args.seed, args.after, args.outdir,
                 resettle=not args.no_resettle, key_ctrl=kc,
                 label="stance_cam%+.3f" % kc[0])
    else:
        for p in args.plants:
            shot(p, args.milestone, args.seed, args.after, args.outdir,
                 resettle=not args.no_resettle)


if __name__ == "__main__":
    main()
