"""Per-actuator torque budget: what each motor spends torque ON, and which one runs out first.

  python tools/torque_budget.py --run training/runs_restored/m5_stiff_600M --episodes 6
  python tools/torque_budget.py --run training/runs_restored/m5_stiff_600M --mass 0.9,1.0,1.1,1.2

THE DECOMPOSITION IS BY REGIME, NOT BY SUPERPOSITION.
Joint torque in a contact-rich nonlinear system does not decompose additively, and any table that
claims tau = tau_gravity + tau_locomotion + tau_disturbance exactly is lying. What IS measurable,
and what a control engineer will accept, is the same robot under three regimes:

  STATIC   base fully railed, stand keyframe held by the position servos, no policy.
           -> the torque it costs merely to carry the robot's own weight in that pose.
  RUN      the real policy, greedy, nominal plant, no pushes/trips, DR off.
           -> static + whatever locomotion costs.
  PUSH     same, with body shoves enabled.
           -> run + whatever staying upright under disturbance costs.

Differences between regimes are reported as INCREMENTS and labelled as such. The static regime is
measured with the base locked because a free base at m5 falls over in ~1 s and there is no
steady state to read.

Sampling is at control rate (200 Hz), not physics rate (1 kHz), so peaks are a mild
under-estimate; saturation duty is computed on the same samples and is comparable across regimes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent


def stats(tau, limit):
    """tau: (n_samples,) signed torque for one actuator. limit: +forcerange."""
    a = np.abs(tau)
    return dict(rms=float(np.sqrt(np.mean(tau ** 2))), p95=float(np.percentile(a, 95)),
                peak=float(a.max()), pct_limit=float(a.max() / limit * 100.0),
                sat_duty=float(np.mean(a >= 0.95 * limit) * 100.0),
                rms_pct=float(np.sqrt(np.mean(tau ** 2)) / limit * 100.0))


def measure_static(cfg_dict, config_from_dict, DashEnv, mass_scale=1.0, hold_s=2.0, read_s=0.5):
    """Gravity/posture hold torque: base fully railed, stand keyframe, servos holding, no policy."""
    import copy
    d = copy.deepcopy(cfg_dict)
    d["base_lock"] = [1, 1, 1, 1, 1, 1]          # nothing free: pure static hold
    d["push_interval_s"] = 0.0
    d["trip_prob"] = 0.0
    cfg = config_from_dict(d)
    env = DashEnv(cfg)
    env._dr.enabled = False
    env.reset(seed=0)
    m, dat = env.model, env.data
    if mass_scale != 1.0:
        m.body_mass[:] = env._dr.n_mass * mass_scale
        m.body_inertia[:] = env._dr.n_inertia * mass_scale
    # hold the stand keyframe with the position servos and let it settle
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, cfg.keyframe)
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(m, dat, kid)
    dat.ctrl[:] = m.key_ctrl[kid][:m.nu] if kid >= 0 else 0.0
    n_hold = int(hold_s / m.opt.timestep)
    n_read = int(read_s / m.opt.timestep)
    buf = []
    for k in range(n_hold):
        mujoco.mj_step(m, dat)
        if k >= n_hold - n_read:
            buf.append(dat.actuator_force[:m.nu].copy())
    return np.array(buf), m.actuator_forcerange[:m.nu, 1].copy()


def measure_policy(model_, venv, raw, episodes, seconds, seed0, push_interval_s=0.0):
    """Roll out greedy and log per-actuator torque at every control step."""
    raw.cfg.push_interval_s = push_interval_s
    raw.cfg.trip_prob = 0.0
    max_steps = int(seconds / raw.control_dt)
    nu = raw.model.nu
    buf, n_fall, n_ep = [], 0, 0
    for e in range(episodes):
        venv.seed(seed0 + e)
        obs = venv.reset()
        n_ep += 1
        for n in range(max_steps):
            a, _ = model_.predict(obs, deterministic=True)
            obs, r, d, info = venv.step(a)
            if d[0]:
                n_fall += 0 if bool(info[0].get("TimeLimit.truncated", False)) else 1
                break
            buf.append(raw.data.actuator_force[:nu].copy())
    return np.array(buf), n_fall, n_ep


def table(names, arr, limits, title, ref=None):
    print(f"\n  {title}")
    hdr = f"    {'actuator':<12} {'RMS':>8} {'P95':>8} {'peak':>8} {'%lim':>7} {'sat%':>7} {'RMSlim%':>8}"
    if ref is not None:
        hdr += f" {'d(peak)':>9}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    rows = {}
    for i, nm in enumerate(names):
        s = stats(arr[:, i], limits[i])
        rows[nm] = s
        line = (f"    {nm:<12} {s['rms']:8.1f} {s['p95']:8.1f} {s['peak']:8.1f} "
                f"{s['pct_limit']:6.0f}% {s['sat_duty']:6.1f}% {s['rms_pct']:7.0f}%")
        if ref is not None:
            line += f" {s['peak'] - ref[nm]['peak']:+9.1f}"
        print(line)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pkg", default=None, choices=["training", "walk_mit"])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--push-interval", type=float, default=2.0,
                    help="push period for the disturbance regime (s)")
    ap.add_argument("--mass", default="0.9,1.0,1.1,1.2",
                    help="mass scales for the bottleneck sweep")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    run = Path(args.run).resolve()
    pkg = args.pkg or ("walk_mit" if "walk_mit" in run.parts else "training")
    sys.path.insert(0, str(ROOT / pkg))
    from evaluate import build, load_run_config          # noqa: E402
    from config import config_from_dict                  # noqa: E402
    from env import DashEnv                              # noqa: E402

    cfg_dict = json.loads((run / "resolved_config.json").read_text())
    cfg_dict = cfg_dict.get("config", cfg_dict)

    model_, venv, raw = build(run, None, None)
    raw._dr.enabled = False
    names = [mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(raw.model.nu)]
    limits = raw.model.actuator_forcerange[:raw.model.nu, 1].copy()

    print(f"[budget] {run.name}   {args.episodes} eps x {args.seconds:.0f}s, greedy, paired seeds")
    print(f"[budget] mass {raw._dr.n_mass.sum():.2f} kg   limits " +
          " ".join(f"{n}={l:.0f}" for n, l in zip(names, limits)))

    out = {"run": run.name, "actuators": names, "limits": limits.tolist(), "regimes": {}}

    # ---- REGIME 1: static gravity/posture hold, base railed -------------------------------
    st, st_lim = measure_static(cfg_dict, config_from_dict, DashEnv)
    out["regimes"]["static"] = table(names, st, st_lim,
                                     "STATIC  (base railed, stand keyframe held) = carry own weight")

    # ---- REGIME 2: running, undisturbed ---------------------------------------------------
    rn, nf, ne = measure_policy(model_, venv, raw, args.episodes, args.seconds, args.seed, 0.0)
    out["regimes"]["run"] = table(names, rn, limits,
                                  f"RUN     (policy, no disturbance; {ne - nf}/{ne} survived) "
                                  f"= + locomotion", ref=out["regimes"]["static"])

    # ---- REGIME 3: running with body shoves -----------------------------------------------
    pu, nfp, nep = measure_policy(model_, venv, raw, args.episodes, args.seconds, args.seed,
                                  args.push_interval)
    out["regimes"]["push"] = table(names, pu, limits,
                                   f"PUSH    (shoves every {args.push_interval:.0f}s; "
                                   f"{nep - nfp}/{nep} survived) = + disturbance rejection",
                                   ref=out["regimes"]["run"])

    # ---- INCREMENTS ------------------------------------------------------------------------
    print("\n  BUDGET BY REGIME (RMS torque, N*m, and share of each actuator's own limit)")
    print(f"    {'actuator':<12} {'gravity':>9} {'+locomotion':>13} {'+disturb':>10} "
          f"{'total RMS':>10} {'of limit':>9}")
    print("    " + "-" * 68)
    for nm in names:
        g = out["regimes"]["static"][nm]["rms"]
        r = out["regimes"]["run"][nm]["rms"]
        p = out["regimes"]["push"][nm]["rms"]
        lim = limits[names.index(nm)]
        print(f"    {nm:<12} {g:9.1f} {r - g:+13.1f} {p - r:+10.1f} {p:10.1f} {p / lim * 100:8.0f}%")

    # ---- BOTTLENECK: which actuator saturates first as mass rises --------------------------
    scales = [float(s) for s in args.mass.split(",")]
    print(f"\n  BOTTLENECK SWEEP: peak torque as % of each actuator's OWN limit, vs mass")
    print(f"    {'mass x':>7}  " + "".join(f"{n:>12}" for n in names) + f"{'surv':>7}")
    print("    " + "-" * (9 + 12 * len(names) + 7))
    out["bottleneck"] = []
    for s in scales:
        raw._dr.restore(raw.model)
        raw.model.body_mass[:] = raw._dr.n_mass * s
        raw.model.body_inertia[:] = raw._dr.n_inertia * s
        arr, nf, ne = measure_policy(model_, venv, raw, args.episodes, args.seconds, args.seed, 0.0)
        pcts = [stats(arr[:, i], limits[i])["pct_limit"] for i in range(len(names))]
        sats = [stats(arr[:, i], limits[i])["sat_duty"] for i in range(len(names))]
        print(f"    {s:7.2f}  " + "".join(f"{p:11.0f}%" for p in pcts) +
              f"{(ne - nf) / ne * 100:6.0f}%")
        out["bottleneck"].append(dict(mass=s, pct_limit=pcts, sat_duty=sats,
                                      survival=(ne - nf) / ne))
    print(f"\n    (sat duty %, same order)")
    for row in out["bottleneck"]:
        print(f"    {row['mass']:7.2f}  " + "".join(f"{d:11.1f}%" for d in row["sat_duty"]))
    raw._dr.restore(raw.model)

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\n[budget] wrote {args.json}")


if __name__ == "__main__":
    main()
