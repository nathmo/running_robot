"""Greedy evaluation of a v2 run on the MJX plant (nominal, no noise, no disturbances), and an
mp4 rendered with classic MuJoCo from the recorded qpos trajectory.

    python walk_v2/evaluate.py --run walk_v2/runs/v2_s1_planar_s0 --episodes 16
    python walk_v2/evaluate.py --run walk_v2/runs/v2_s1_planar_s0 --video dash.mp4 --seconds 40
    python walk_v2/evaluate.py --run ... --dr            # with the training randomization on

Greedy = the distribution MEAN, always (the determinism gap: stochastic ep_len lies). Paired
seeds (1000+) so two checkpoints are compared on the same plants and pushes.
"""
import argparse
import json
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import jax
import jax.numpy as jnp

from config import config_from_dict
from env import DashEnvV2, EnvParams
from ppo import PPO
from train import make_eval_env, latest_checkpoint


def load_run(run, checkpoint=None, n_envs=16, dr=False, keep_assist=False):
    run = Path(run)
    cfg = config_from_dict(json.loads((run / "resolved_config.json").read_text())["config"])
    ck = Path(checkpoint) if checkpoint else latest_checkpoint(run)
    if ck is None:
        raise FileNotFoundError(f"no checkpoint in {run}")
    env = DashEnvV2(cfg, n_envs=n_envs) if dr else make_eval_env(cfg, n_envs, keep_assist=keep_assist)
    agent = PPO(cfg, env, run, cfg.total_steps, seed=0, eval_env=None)
    agent.load(ck)
    print(f"[eval] {ck.name} @ {agent.step:,} steps  plant {'DR' if dr else 'nominal'}")
    return cfg, env, agent


def rollout(env, agent, seed, n_max, record=False, assist=0.0):
    """Greedy rollout of all envs; returns per-env dash stats (+ env-0 qpos trajectory)."""
    params = EnvParams.final(agent.cfg)._replace(dr_scale=1.0 if env.cfg.dr_enable else 0.0,
                                                 ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0, pitch_assist=float(assist))
    key = jax.random.PRNGKey(seed)
    state, obs = env.reset(key, params)
    act = agent._act_greedy

    def body(carry, _):
        state, obs, alive, dist, tline, fin, fell, tend, vsum = carry
        a = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        state2, obs2, r, done, info = env.step(state, a, params)
        ending = alive & done
        dist = jnp.where(alive, info["sprint_d"], dist)
        tline = jnp.where(ending & info["finished"], info["t_line"], tline)
        fin = fin | (ending & info["finished"])
        fell = fell | (ending & info["fallen"])
        tend = jnp.where(ending, state.t + env.control_dt, tend)
        alive2 = alive & ~done
        q0 = state.data.qpos[0] if record else jnp.zeros(1)
        return (state2, obs2, alive2, dist, tline, fin, fell, tend, vsum), q0

    n = env.n_envs
    init = (state, obs, jnp.ones(n, bool), jnp.zeros(n), jnp.full(n, -1.0), jnp.zeros(n, bool),
            jnp.zeros(n, bool), jnp.full(n, n_max * env.control_dt), jnp.zeros(n))
    (_, _, alive, dist, tline, fin, fell, tend, _), qs = jax.lax.scan(body, init, None, length=n_max)
    return dict(dist=np.asarray(dist), t_line=np.asarray(tline), finished=np.asarray(fin),
                fell=np.asarray(fell), t_end=np.asarray(tend), alive=np.asarray(alive)), np.asarray(qs)


def render_video(cfg, qs, t_end, path, seconds=None, fps=None):
    import mujoco
    import imageio.v2 as imageio
    from plant import resolve
    m = mujoco.MjModel.from_xml_path(resolve(cfg.model_path))
    d = mujoco.MjData(m)
    dt = m.opt.timestep * cfg.control_decimation
    n = int(min(len(qs), (seconds or t_end) / dt))
    fps = fps or int(round(1.0 / dt))
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance, cam.elevation = 2.5, -15
    with imageio.get_writer(path, fps=fps) as w:
        for i in range(n):
            d.qpos[:] = qs[i]
            mujoco.mj_forward(m, d)
            cam.lookat[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bodyNCS-v1")]
            renderer.update_scene(d, cam)
            w.append_data(renderer.render())
    print(f"[eval] wrote {path} ({n} frames, {n * dt:.1f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=None, help="cap per episode (default: cfg.episode_s)")
    ap.add_argument("--video", default=None)
    ap.add_argument("--dr", action="store_true", help="evaluate on the randomized training plant")
    ap.add_argument("--assist", type=float, default=0.0,
                    help="pitch-assist level (0 = deployable test; the CPU arm reads out at the training level)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cfg, env, agent = load_run(args.run, args.checkpoint, n_envs=args.episodes, dr=args.dr, keep_assist=args.assist > 0)
    n_max = int(round((args.seconds or cfg.episode_s) / env.control_dt))
    stats, qs = rollout(env, agent, args.seed, n_max, record=args.video is not None, assist=args.assist)
    fin = stats["finished"]
    print(f"episodes={args.episodes}  finishes {fin.sum()}/{args.episodes}  falls {stats['fell'].sum()}  "
          f"timeouts {stats['alive'].sum()}")
    for i in range(args.episodes):
        line = f"line {stats['t_line'][i]:6.2f} s" if fin[i] else "line    DNF"
        how = "finished" if fin[i] else ("FELL" if stats["fell"][i] else "timeout")
        print(f"  ep{i:02d}: {stats['dist'][i]:6.1f} m in {stats['t_end'][i]:6.2f} s  {line}  {how}  "
              f"avg {stats['dist'][i] / max(stats['t_end'][i], 1e-6):.2f} m/s")
    if fin.any():
        print(f"  mean t_line {stats['t_line'][fin].mean():.2f} s  mean speed "
              f"{(cfg.sprint_dist_m / stats['t_line'][fin]).mean():.2f} m/s")
    if args.json:
        Path(args.json).write_text(json.dumps({k: np.asarray(v).tolist() for k, v in stats.items()}, indent=1))
    if args.video:
        render_video(cfg, qs, float(stats["t_end"][0]), args.video, seconds=args.seconds)


if __name__ == "__main__":
    main()
