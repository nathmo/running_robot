"""100 m dash re-scored by ACTUAL horizontal travel, not world-x progress.

  python tools/eval_dash_xy.py --run walk_mit/runs/sprint_m6_mit_s0 --episodes 16

WHY THIS EXISTS ALONGSIDE walk_mit/evaluate.py
  The sprint metric d is world-x only (env._update_sprint: qpos[0] - x0). That is the right score
  for the dash *task*, but on a yaw-free plant (m6) a policy that runs fast in a curve reads ~0 m
  while mean vx says 3 m/s. This reports what the base actually did in the horizontal plane:
    net    = ||xy_end - xy_start||        crow-flight displacement, any direction
    path   = sum ||delta xy|| at 200 Hz   ground covered, curves and all (bobbing inflates it a
                                          little; net is the conservative number)
  alongside the official x metric, so straight-line and veering runs are comparable.

Protocol matches tools/eval_envelope.py: greedy, paired seeds (episode e = seed0+e in every run),
env built through evaluate.build() so all the curriculum-restore eval traps stay fixed, and
raw.data is never sampled after a done step (SB3 auto-reset). Trajectories are sampled via
env.on_control_step, cleared after reset so keyframe resettling is excluded.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "walk_mit"))

from evaluate import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--json", default=None, help="write per-episode metrics here")
    ap.add_argument("--npz", default=None, help="write 20 Hz xy trajectories here")
    args = ap.parse_args()
    run = Path(args.run)
    model, venv, raw = build(run, None, None)

    xy_buf = []
    raw.on_control_step = lambda: xy_buf.append((float(raw.data.qpos[0]),
                                                 float(raw.data.qpos[1])))

    episodes, trajs = [], {}
    for e in range(args.episodes):
        venv.seed(args.seed0 + e)
        obs = venv.reset()
        xy_buf.clear()                       # drop anything fired during reset/resettle
        done, sprint, fell = False, None, False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, d, info = venv.step(a)
            sprint = info[0].get("sprint", sprint)
            if d[0]:
                # SB3 auto-resets here: raw.data is ALREADY the next episode. xy_buf still holds
                # the final control step because on_control_step fired inside env.step.
                fell = not bool(info[0].get("TimeLimit.truncated", False))
                done = True
        traj = np.asarray(xy_buf)
        steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        t = len(traj) * raw.control_dt
        ep = dict(
            seed=args.seed0 + e,
            t=round(t, 2),
            fell=bool(fell),
            x=None if sprint is None else sprint["d"],
            t_line=None if sprint is None else sprint["t_line"],
            net=round(float(np.linalg.norm(traj[-1] - traj[0])), 2),
            path=round(float(steps.sum()), 2),
        )
        ep["net_speed"] = round(ep["net"] / max(t, 1e-9), 2)
        ep["path_speed"] = round(ep["path"] / max(t, 1e-9), 2)
        episodes.append(ep)
        trajs[f"ep{e:02d}"] = traj[::10].astype(np.float32)   # 20 Hz is plenty for a path plot
        line = "line %6.2f s" % ep["t_line"] if ep["t_line"] is not None else "line    DNF"
        print(f"  ep{e:02d}: x {ep['x']:7.1f} m   net {ep['net']:6.1f} m   "
              f"path {ep['path']:6.1f} m   {line}   {'FELL ' if fell else 'timeo'} at "
              f"{ep['t']:5.1f} s   net {ep['net_speed']:.2f} / path {ep['path_speed']:.2f} m/s",
              flush=True)

    x = np.array([ep["x"] for ep in episodes], dtype=float)
    net = np.array([ep["net"] for ep in episodes], dtype=float)
    path = np.array([ep["path"] for ep in episodes], dtype=float)
    tl = [ep["t_line"] for ep in episodes if ep["t_line"] is not None]
    falls = sum(ep["fell"] for ep in episodes)
    print(f"{run.name}: {args.episodes} greedy eps  |  x {np.mean(x):.1f}+/-{np.std(x):.1f} m  |  "
          f"net {np.mean(net):.1f}+/-{np.std(net):.1f} m  |  path {np.mean(path):.1f}+/-"
          f"{np.std(path):.1f} m  |  finishes {len(tl)}/{args.episodes}"
          + (f" (line {np.mean(tl):.2f}+/-{np.std(tl):.2f} s)" if tl else "")
          + f"  |  falls {falls}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            dict(run=str(run), episodes=episodes, seed0=args.seed0), indent=1))
    if args.npz:
        np.savez_compressed(args.npz, **trajs)


if __name__ == "__main__":
    main()
