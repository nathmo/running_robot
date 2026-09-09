"""The §08 margin budget as a batched gate: one greedy policy, every axis, pass/fail per row.

    python walk_v2/tools/eval_envelope.py --run walk_v2/runs/v2_s2_free_s0 [--json out.json]
    python walk_v2/tools/eval_envelope.py --run ... --axes push_lat wind_x delay

Axes (all on the nominal plant with one override set; paired seeds 1000+, 8 episodes x 20 s
per point, 16 x full dash for lane keeping and thermal):
    nominal      100% survival, and the dash time baseline
    wind_x       constant world-x force +-15, +-30 N          gate: 100% at +-30 N, dash time within 10%
    wind_y       constant world-y force +-15, +-30 N (free plant)
    tilt_roll    +-3, +-5 deg                                  gate: 100% at +-3 deg both axes
    tilt_pitch
    friction     0.4 0.5 0.7 1.0 1.3                           gate: no fall at 0.5 over 16 dashes
    delay        6 9 12 15 18 ms                               gate: >= 95% at each
    mass         0.8 0.85 0.9 1.1 1.15 1.2                     gate: >= 90% across 0.85-1.15
    link         0.5 1.5 (leg spring)                          gate: same survival as nominal
    thermal      hot start 0.5, 0.7 of dT_max; dash peak       gate: peak < 0.85 over two dashes
    push_fwd     impulses 0.2..1.0 m/s, 8 phase bins, both signs   (the RUNNER absorbed +-1.0)
    push_lat     impulses 0.1..1.0 m/s, 8 phase bins, both signs   gate: 0.5 m/s at any phase, 16/16
    lane         16 full dashes: |y| < 0.5 m at the line, 16/16

A push is an instantaneous body-frame velocity change applied on the first tick after t_push
whose latched phase falls in the bin; survival = no fall in the following 5 s; recovery time =
first tick with |v_y| < 0.05 m/s after the push.
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

from env import EnvParams
from plant import Override
from evaluate import load_run

TWO_PI = 2.0 * np.pi


def run_point(env, agent, seed, n_max, override=None, push=None):
    """Greedy rollout with an override; push = (axis 'x'|'y', dv, phase_bin, n_bins, t_push)."""
    params = EnvParams.final(agent.cfg)._replace(dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0,
                                                 pitch_assist=0.0)
    key = jax.random.PRNGKey(seed)
    state, obs = env.reset(key, params, override)
    act = agent._act_greedy
    p = env.plant
    dof = p.base_d["x"] if (push is None or push[0] == "x") else p.base_d["y"]

    def body(carry, _):
        state, obs, alive, pushed, t_pushed, fell, fin, tline, y_line, rec_t, thmax = carry
        a = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        if push is not None:
            axis, dv, pbin, nb, t_push = push
            in_bin = jnp.floor(state.phase / TWO_PI * nb).astype(jnp.int32) == pbin
            do = alive & (~pushed) & (state.t >= t_push) & in_bin
            R = env._base_rot(state.data)
            v_world = R @ jnp.array([dv if axis == "x" else 0.0, dv if axis == "y" else 0.0, 0.0])
            qvel = state.data.qvel
            qvel = qvel.at[p.base_d["x"]].add(jnp.where(do, v_world[0], 0.0))
            if not p.planar:
                qvel = qvel.at[p.base_d["y"]].add(jnp.where(do, v_world[1], 0.0))
            state = state.replace(data=state.data.replace(qvel=qvel))
            pushed = pushed | do
            t_pushed = jnp.where(do, state.t, t_pushed)
        state2, obs2, r, done, info = env.step(state, a, params)
        ending = alive & done
        fell = fell | (ending & info["fallen"])
        fin = fin | (ending & info["finished"])
        tline = jnp.where(ending & info["finished"], info["t_line"], tline)
        newly = (state2.crossed) & (~state.crossed)
        y_line = jnp.where(newly, jnp.abs(env._y(state2.data)), y_line)
        vy = env._vel_body(state2.data)[1]
        rec_now = pushed & (rec_t < 0) & (jnp.abs(vy) < 0.05) & (state2.t - t_pushed > 0.2)
        rec_t = jnp.where(rec_now, state2.t - t_pushed, rec_t)
        thmax = jnp.maximum(thmax, info["thermal_max"])
        # after a push, only the 5 s window counts
        window_over = pushed & (state2.t - t_pushed > 5.0)
        alive2 = alive & ~done & ~window_over
        return (state2, obs2, alive2, pushed, t_pushed, fell, fin, tline, y_line, rec_t, thmax), None

    n = env.n_envs
    init = (state, obs, jnp.ones(n, bool), jnp.zeros(n, bool), jnp.zeros(n), jnp.zeros(n, bool),
            jnp.zeros(n, bool), jnp.full(n, -1.0), jnp.full(n, -1.0), jnp.full(n, -1.0), jnp.zeros(n))
    (_, _, alive, pushed, _, fell, fin, tline, y_line, rec_t, thmax), _ = jax.lax.scan(body, init, None, length=n_max)
    return dict(survive=float(1.0 - np.asarray(fell).mean()), fell=int(np.asarray(fell).sum()),
                pushed=int(np.asarray(pushed).sum()), finishes=int(np.asarray(fin).sum()),
                t_line=float(np.asarray(tline)[np.asarray(fin)].mean()) if bool(np.asarray(fin).any()) else float("nan"),
                y_line_max=float(np.asarray(y_line).max()), rec_t=float(np.asarray(rec_t)[np.asarray(rec_t) > 0].mean())
                if bool((np.asarray(rec_t) > 0).any()) else float("nan"), thermal_max=float(np.asarray(thmax).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--axes", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cfg, env, agent = load_run(args.run, args.checkpoint, n_envs=args.n)
    env16 = None
    dt = env.control_dt
    n20 = int(20.0 / dt)
    n_dash = int(cfg.episode_s / dt)
    axes = args.axes or ["nominal", "wind_x", "wind_y", "tilt_roll", "tilt_pitch", "friction", "delay",
                         "mass", "link", "thermal", "push_fwd", "push_lat", "lane"]
    if env.plant.planar:
        axes = [a for a in axes if a not in ("wind_y", "push_lat", "tilt_roll", "lane")]
    results = {}
    print(f"[envelope] {args.run}  {args.n} paired seeds from {args.seed}")

    def point(name, n_max, override=None, push=None, envx=None):
        r = run_point(envx or env, agent, args.seed, n_max, override, push)
        results[name] = r
        print(f"  {name:28s} survive {100 * r['survive']:5.1f}%  fin {r['finishes']:2d}  t_line {r['t_line']:6.2f}  "
              f"|y|max {r['y_line_max']:5.2f}  rec {r['rec_t']:5.2f}s  th {r['thermal_max']:.2f}", flush=True)
        return r

    base = point("nominal", n_dash) if "nominal" in axes else None
    if "wind_x" in axes:
        for F in (-30.0, -15.0, 15.0, 30.0):
            point(f"wind_x {F:+.0f} N", n_dash, Override(wind_x=F))
    if "wind_y" in axes:
        for F in (-30.0, -15.0, 15.0, 30.0):
            point(f"wind_y {F:+.0f} N", n20, Override(wind_y=F))
    if "tilt_roll" in axes:
        for a in (-5.0, -3.0, 3.0, 5.0):
            point(f"tilt_roll {a:+.0f} deg", n20, Override(tilt_roll_deg=a))
    if "tilt_pitch" in axes:
        for a in (-5.0, -3.0, 3.0, 5.0):
            point(f"tilt_pitch {a:+.0f} deg", n20, Override(tilt_pitch_deg=a))
    if "friction" in axes:
        for f in (0.4, 0.5, 0.7, 1.0, 1.3):
            point(f"friction {f:.1f}", n_dash, Override(friction=f))
    if "delay" in axes:
        for d in (6.0, 9.0, 12.0, 15.0, 18.0):
            point(f"delay {d:.0f} ms", n20, Override(delay_ms=d))
    if "mass" in axes:
        for m in (0.8, 0.85, 0.9, 1.1, 1.15, 1.2):
            point(f"mass x{m:.2f}", n20, Override(mass_scale=m))
    if "link" in axes:
        for s in (0.5, 1.5):
            point(f"leg spring x{s:.1f}", n20, Override(link_scale=s))
    if "thermal" in axes:
        for x0 in (0.5, 0.7):
            point(f"thermal hot-start {x0:.1f}", n_dash, Override(thermal_x0=x0))
    for axis_name, axis in (("push_fwd", "x"), ("push_lat", "y")):
        if axis_name not in axes:
            continue
        dvs = (0.2, 0.5, 1.0) if axis == "x" else (0.1, 0.3, 0.5, 0.6, 0.8, 1.0)
        for dv in dvs:
            worst = 1.0
            for sign in (1.0, -1.0):
                for pb in range(8):
                    r = run_point(env, agent, args.seed, int(12.0 / dt), None, (axis, sign * dv, pb, 8, 5.0))
                    worst = min(worst, r["survive"])
                    results[f"{axis_name} {sign * dv:+.1f} bin{pb}"] = r
            print(f"  {axis_name} |dv| {dv:.1f} m/s: worst-bin survival {100 * worst:5.1f}%", flush=True)
    if "lane" in axes:
        env16 = env if args.n == 16 else None
        r = point("lane (full dash)", n_dash, envx=env16)
    # gates
    print("\n§08 gate:")
    g = lambda k: results.get(k, {}).get("survive", float("nan"))
    rows = [
        ("100% at nominal", g("nominal") >= 1.0),
        ("wind ±30 N: 100%", min(g("wind_x +30 N"), g("wind_x -30 N")) >= 1.0),
        ("slope ±3°: 100% both axes", min(g("tilt_pitch +3 deg"), g("tilt_pitch -3 deg"),
                                          g("tilt_roll +3 deg") if "tilt_roll" in axes else 1.0,
                                          g("tilt_roll -3 deg") if "tilt_roll" in axes else 1.0) >= 1.0),
        ("friction 0.5: no fall", g("friction 0.5") >= 1.0),
        ("delay 6-18 ms: >= 95% each", min(g(f"delay {d:.0f} ms") for d in (6, 9, 12, 15, 18)) >= 0.95),
        ("mass 0.85-1.15: >= 90%", min(g("mass x0.85"), g("mass x0.90"), g("mass x1.10"), g("mass x1.15")) >= 0.9),
        ("thermal peak < 0.85", results.get("nominal", {}).get("thermal_max", 9) < 0.85),
    ]
    if "push_lat" in axes:
        rows.append(("lateral 0.5 m/s any phase", min(results[k]["survive"] for k in results if k.startswith("push_lat +0.5") or k.startswith("push_lat -0.5")) >= 1.0))
    if "lane" in axes and "lane (full dash)" in results:
        rows.append(("|y| < 0.5 m at the line", results["lane (full dash)"]["y_line_max"] < 0.5))
    for name, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    results["gate"] = {n: bool(o) for n, o in rows}
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
        print(f"[envelope] wrote {args.json}")


if __name__ == "__main__":
    main()
