"""Render the keyframe A/B: same robot, same 3-gain reflex, only key_ctrl differs.

There is no network and no scripted controller in these clips. The action is IDENTICALLY ZERO for
every step; the only thing acting is fourier_gait's built-in fixed pitch reflex (pitch_kp/kd/clip),
which has been in the codebase all along. The single difference between the two videos is
`model_path` -- the shipped `stand` keyframe, which settles with the toes 9.2 cm ahead of the CoM
(a 5.7 deg backward lean at t=0), versus the re-solved one from make_balanced_keyframe.py, which
settles at -0.13 deg.

    python training/render_keyframe_ab.py --milestone m3 --seed 0
"""
import argparse
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import mujoco

from config import get_config
from env import DashEnv

VIDEO_FPS = 50


def render(model_path, milestone, seed, out, seconds, kp, kd, clip, label):
    cfg = get_config(f"walk_fwd_{milestone}")
    cfg.model_path = model_path
    cfg.episode_s = float(seconds)
    cfg.push_interval_s = 0.0
    cfg.trip_prob = 0.0
    cfg.pitch_kp, cfg.pitch_kd, cfg.pitch_clip = kp, kd, clip
    env = DashEnv(cfg)
    env.set_dr_scale(0.0)
    env.reset(seed=int(seed))
    env.set_command(0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)
    com = env.data.subtree_com[0]
    toe = float(np.mean([env.data.geom_xpos[g][0] for g in env.foot_gids]))
    lean = np.degrees(np.arctan2(toe - com[0], com[2]))

    import imageio.v2 as imageio
    renderer = mujoco.Renderer(env.model, 480, 640)
    every = max(1, int(round(1.0 / (env.control_dt * VIDEO_FPS))))
    writer = imageio.get_writer(out, fps=int(round(1.0 / (env.control_dt * every))))
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.distance, cam.elevation, cam.azimuth = 3.0, -12, 90     # side-on: pitch is the story
    a = np.zeros(env.action_dim, np.float32)
    n = nf = 0
    peak = 0.0
    # hold the LAST frame for a beat after a fall so a 1 s clip is actually watchable
    while True:
        if n % every == 0:
            cam.lookat[:] = env.data.qpos[:3]
            renderer.update_scene(env.data, cam)
            writer.append_data(renderer.render())
            nf += 1
        _, _, term, trunc, _ = env.step(a)
        peak = max(peak, abs(np.degrees(np.arcsin(np.clip(env._gravity_body()[0], -1, 1)))))
        n += 1
        if term or trunc:
            break
    if term:
        cam.lookat[:] = env.data.qpos[:3]
        renderer.update_scene(env.data, cam)
        frame = renderer.render()
        for _ in range(VIDEO_FPS):                 # 1 s freeze on the fall
            writer.append_data(frame)
            nf += 1
    writer.close()
    t = n * env.control_dt
    print(f"{label:>22s}: start lean {lean:+5.2f} deg  ->  {t:6.2f} s  peak pitch {peak:5.1f} deg  "
          f"{'SURVIVED' if not term else 'FELL'}   {out}  ({nf} frames)")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", default="m3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--kp", type=float, default=2.0)
    ap.add_argument("--kd", type=float, default=0.10)
    ap.add_argument("--clip", type=float, default=0.25)
    ap.add_argument("--outdir", default=str(PKG_DIR / "milestones"))
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    print(f"zero policy action; only the built-in pitch reflex kp={args.kp} kd={args.kd} "
          f"clip={args.clip} is acting.  rung {args.milestone}, seed {args.seed}")
    for tag, mp in (("shipped", "model/dash01.xml"), ("balanced", "model/dash01_bal.xml")):
        render(mp, args.milestone, args.seed,
               f"{args.outdir}/keyframe_{tag}_{args.milestone}_s{args.seed}.mp4",
               args.seconds, args.kp, args.kd, args.clip, f"{tag} keyframe")


if __name__ == "__main__":
    main()
