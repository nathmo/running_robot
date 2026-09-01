"""Performance envelope + stability margins for a trained policy, over plant perturbations.

  python tools/eval_envelope.py --run walk_mit/runs/imp_m2_long --episodes 6
  python tools/eval_envelope.py --run training/runs_restored/m5_stiff_600M --axes mass,torque,delay
  python tools/eval_envelope.py --run walk_mit/runs/imp_m2_long --json envelope.json

WHY THIS EXISTS ALONGSIDE training/eval_robustness.py
  eval_robustness reports SURVIVAL only. A margin table with no performance axis cannot answer
  "what does robustness cost in top speed" -- which is the whole question. This reports speed and
  distance at every operating point as well, so each axis yields both a failure boundary and a
  degradation curve.

THREE THINGS IT DOES THAT THE OLD HARNESS DOES NOT
  1. PAIRED SEEDS. Outcomes here are bimodal (a policy either catches its first stumble or does
     not), so an unpaired sample can invent an effect that is not there. Episode e uses seed
     seed0+e at EVERY operating point, so points differ by the perturbation and nothing else.
  2. IMPEDANCE-AWARE kp AXIS. With imp_enable the policy rewrites actuator_gainprm/biasprm every
     control step from _imp_base, and reset() re-seeds _imp_base from _imp_pristine. Writing
     model.actuator_gainprm (what eval_robustness does) is therefore overwritten within one
     control step and the kp axis silently measures NOTHING. We scale _imp_pristine instead.
  3. Distance is read from the env's own sprint accounting where available, because SB3 auto-resets
     on done -- reading raw.data after a terminal step gives you the NEXT episode's state.

Domain randomization, pushes and trips are all disabled: the sweep sets exact operating points, so
a per-episode random draw would mean the measured point is the point plus noise.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# axis -> (label, values, kind)  kind: "x" multiplier of nominal, "abs" absolute value
AXES = {
    "mass":     ("total mass x",          [0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50], "x"),
    "inertia":  ("link inertia x",        [0.6, 0.8, 1.0, 1.25, 1.6], "x"),
    "friction": ("floor friction mu",     [0.30, 0.50, 0.70, 1.00, 1.30], "abs"),
    "ankle_k":  ("ankle spring k x",      [0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5], "x"),
    "ankle_c":  ("ankle damping x",       [0.3, 0.6, 1.0, 1.5, 2.5], "x"),
    "kp":       ("servo kp x  (GAIN)",    [0.6, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7], "x"),
    "torque":   ("torque limit x",        [0.5, 0.6, 0.75, 0.9, 1.0], "x"),
    "delay":    ("action delay (steps)",  [2, 3, 4, 5, 6, 8, 10], "abs"),
    "slope":    ("ground slope (deg)",    [0.0, 2.0, 4.0, 6.0, 8.0], "abs"),
    "com":      ("CoM offset x (m)",      [0.0, 0.01, 0.02, 0.04, 0.06], "abs"),
    "drive_bw": ("drive bandwidth (Hz)",  [0.8, 1.5, 3.0, 6.0, 12.0], "abs"),
}

CMD_SCHED = [(0.0, 0.0), (0.75, 0.0), (0.5, 0.6), (0.0, 0.0), (-0.5, 0.0), (0.9, 0.0)]


def snapshot_imp(raw):
    """Nominal copy of the impedance channel's base gains (None on non-impedance runs)."""
    if not getattr(raw, "imp_dim", 0):
        return None
    return tuple(p.copy() for p in raw._imp_pristine)


def nominal_value(raw, axis, nominal_delay, nominal_torque):
    """The as-built value of this axis. Multiplier axes are 1.0 by construction; the absolute ones
    must be read off the plant -- taking the first swept value instead reports the LOW END of the
    sweep as nominal (friction would read 0.3, not 1.0) and the margin band comes out nonsense."""
    if axis in ("mass", "inertia", "ankle_k", "ankle_c", "kp"):
        return 1.0
    if axis == "torque":
        return nominal_torque
    if axis == "friction":
        return float(raw._dr.n_friction[:, 0].max())
    if axis in ("slope", "com"):
        return 0.0
    if axis == "delay":
        return float(nominal_delay)
    if axis == "drive_bw":
        return float(getattr(raw.cfg, "drive_bandwidth_hz", 0.0))
    raise ValueError(axis)


def ankle_k_available(raw):
    """The ankle_k axis divides by k_old. walk_mit runs carry jnt_stiffness=0 on the ankle (the
    spring is not modelled as joint stiffness there), so the axis is not just wrong but a
    ZeroDivisionError -- which is a live crash in training/eval_robustness.py as well."""
    dr = raw._dr
    if not dr._ankle_j or dr.stand_qpos is None:
        return False
    return all(abs(float(dr.n_jnt_stiffness[j])) > 1e-12 for j in dr._ankle_j)


def restore_nominal(raw, imp_nom, nominal_delay, nominal_torque):
    raw._dr.restore(raw.model)
    raw.cfg.action_delay_steps = nominal_delay
    raw.set_torque_limit(nominal_torque)
    if imp_nom is not None:
        raw._imp_pristine = tuple(p.copy() for p in imp_nom)


def apply_point(raw, axis, val, imp_nom):
    """Set ONE parameter to an exact value on top of the nominal plant."""
    dr = raw._dr
    dr.restore(raw.model)
    if imp_nom is not None:
        raw._imp_pristine = tuple(p.copy() for p in imp_nom)
    m = raw.model
    if axis == "mass":
        m.body_mass[:] = dr.n_mass * val
        m.body_inertia[:] = dr.n_inertia * val
    elif axis == "inertia":
        m.body_inertia[:] = dr.n_inertia * val
    elif axis == "friction":
        m.geom_friction[:, 0] = val
    elif axis == "ankle_k":
        for j in dr._ankle_j:
            qadr = int(dr._jnt_qposadr[j])
            k_old = float(dr.n_jnt_stiffness[j])
            k_new = k_old * val
            ref_old = float(dr.n_qpos_spring[qadr])
            q_stand = float(dr.stand_qpos[qadr])
            m.qpos_spring[qadr] = q_stand - (k_old / k_new) * (q_stand - ref_old)
            m.jnt_stiffness[j] = k_new
    elif axis == "ankle_c":
        for j in dr._ankle_j:
            d = int(dr._jnt_dofadr[j])
            m.dof_damping[d] = dr.n_dof_damping[d] * val
    elif axis == "kp":
        m.actuator_gainprm[:, 0] = dr.n_gainprm[:, 0] * val
        m.actuator_biasprm[:, 1] = dr.n_biasprm[:, 1] * val
        # impedance runs: the per-step channel re-derives gains from _imp_pristine at every
        # reset, so the model write above is undone before the first step. Scale the source.
        if imp_nom is not None:
            ng = raw.n_gait_act
            gp, b1, b2 = (p.copy() for p in imp_nom)
            gp[:ng] *= val
            b1[:ng] *= val
            raw._imp_pristine = (gp, b1, b2)
    elif axis == "torque":
        # NOT a direct forcerange write: env._apply_torque_limit() re-derives forcerange from
        # _orig_forcerange * _torque_scale * _dr_torque_scale * _sag_scale on every step, so a
        # direct write is overwritten within one step and the axis reads bit-identical at every
        # value (observed on m5_stiff: 17%/7.5s/18.0m at 0.5x through 1.0x). Use the setter.
        raw.set_torque_limit(val)
    elif axis == "delay":
        raw.cfg.action_delay_steps = int(val)
    elif axis == "slope":
        g = float(np.linalg.norm(dr.n_gravity))
        tilt = np.deg2rad(val)
        m.opt.gravity[:] = g * np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    elif axis == "com":
        m.body_ipos[:] = dr.n_ipos + np.array([val, 0.0, 0.0])
    elif axis == "drive_bw":
        raw.set_drive_bandwidth_log10(float(np.log10(val)))
    else:
        raise ValueError(axis)


def run_point(model_, venv, raw, episodes, max_steps, seed0):
    """Roll out `episodes` at the current operating point, greedy, paired seeds."""
    eps = []
    seg = max(1, max_steps // len(CMD_SCHED))
    for e in range(episodes):
        venv.seed(seed0 + e)          # PAIRED: same initial condition at every operating point
        obs = venv.reset()
        x0 = float(raw.data.qpos[0])
        x_run, vxs, errs, stands = x0, [], [], []
        sprint, fell, n = None, False, 0
        for n in range(1, max_steps + 1):
            if raw.command_mode:
                f, w = CMD_SCHED[min((n - 1) // seg, len(CMD_SCHED) - 1)]
                v_fwd, v_back, yaw = raw._cmd_box()
                raw.set_command(f * (v_fwd if f >= 0 else v_back), w * yaw)
            a, _ = model_.predict(obs, deterministic=True)
            obs, r, d, info = venv.step(a)
            sprint = info[0].get("sprint", sprint)
            if d[0]:
                # SB3 auto-resets here: raw.data is ALREADY the next episode. Do not sample it.
                fell = not bool(info[0].get("TimeLimit.truncated", False))
                break
            vx = float(raw._vel_body()[0])
            vxs.append(vx)
            x_run = float(raw.data.qpos[0])
            if raw.command_mode:
                if abs(raw._v_cmd) > 1e-6:
                    errs.append(abs(vx - raw._v_cmd))
                else:
                    stands.append(abs(vx))
        dt = raw.control_dt
        dist = float(sprint["d"]) if sprint and sprint.get("d") is not None else (x_run - x0)
        eps.append(dict(survived=not fell, steps=n, t=n * dt, dist=dist,
                        v_mean=float(np.mean(vxs)) if vxs else 0.0,
                        v_peak=float(np.max(vxs)) if vxs else 0.0,
                        err=float(np.mean(errs)) if errs else float("nan"),
                        stand=float(np.mean(stands)) if stands else float("nan")))

    def agg(k):
        vals = [e[k] for e in eps]
        return float(np.nanmean(vals)) if not all(np.isnan(vals)) else float("nan")

    return dict(survival=float(np.mean([e["survived"] for e in eps])),
                t_mean=agg("t"), dist_mean=agg("dist"),
                dist_max=float(max(e["dist"] for e in eps)),
                v_mean=agg("v_mean"), v_peak=float(max(e["v_peak"] for e in eps)),
                track_mae=agg("err"), stand=agg("stand"), episodes=eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pkg", default=None, choices=["training", "walk_mit"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--axes", default=None, help="comma list (default: all applicable)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    run = Path(args.run).resolve()
    pkg = args.pkg or ("walk_mit" if "walk_mit" in run.parts else "training")
    sys.path.insert(0, str(ROOT / pkg))
    from evaluate import build                                   # noqa: E402

    model_, venv, raw = build(run, args.preset, args.checkpoint)
    raw._dr.enabled = False
    raw.cfg.push_interval_s = 0.0
    raw.cfg.trip_prob = 0.0
    imp_nom = snapshot_imp(raw)
    nominal_delay = int(raw.cfg.action_delay_steps)
    nominal_torque = float(getattr(raw, "_torque_scale", 1.0))
    max_steps = int(args.seconds / raw.control_dt)

    axes = [a.strip() for a in args.axes.split(",")] if args.axes else list(AXES)
    if "drive_bw" in axes and float(getattr(raw.cfg, "drive_bandwidth_hz", 0.0)) <= 0.0:
        axes.remove("drive_bw")
        print("[env] drive_bw axis SKIPPED: this run has no drive model (drive_bandwidth_hz=0)")
    if "ankle_k" in axes and not ankle_k_available(raw):
        axes.remove("ankle_k")
        print("[env] ankle_k axis SKIPPED: ankle jnt_stiffness is 0 on this run")

    print(f"[env] {run.name} pkg={pkg}  {args.episodes} eps x {args.seconds:.0f}s/point  "
          f"command_mode={raw.command_mode}  impedance={'YES' if imp_nom else 'no'}\n")

    out = {"run": run.name, "pkg": pkg, "episodes": args.episodes, "seconds": args.seconds,
           "command_mode": bool(raw.command_mode), "impedance": imp_nom is not None, "axes": {}}
    for ax in axes:
        label, vals, kind = AXES[ax]
        print(f"  {label}")
        rows = []
        for v in vals:
            apply_point(raw, ax, v, imp_nom)
            r = run_point(model_, venv, raw, args.episodes, max_steps, args.seed)
            bar = "#" * int(round(r["survival"] * 16))
            extra = ""
            if raw.command_mode and not np.isnan(r["track_mae"]):
                extra = f"  track {r['track_mae']:.3f}"
            print(f"    {v:>7}  surv {r['survival']:4.0%} |{bar:<16}|  "
                  f"t {r['t_mean']:5.1f}s  dist {r['dist_mean']:6.1f}m  "
                  f"v {r['v_mean']:4.2f} m/s{extra}")
            rows.append(dict(value=v, **{k: r[k] for k in r if k != "episodes"}))
        out["axes"][ax] = dict(label=label, kind=kind, rows=rows,
                               nominal=nominal_value(raw, ax, nominal_delay, nominal_torque))
        restore_nominal(raw, imp_nom, nominal_delay, nominal_torque)
        print()

    # MARGIN, defined RELATIVE TO NOMINAL. Requiring 100% survival is the wrong bar: a policy that
    # only survives 67% of episodes on its own nominal plant (imp_m2_long does) can never show a
    # 100% band, and the axis would read "NONE" everywhere -- which says nothing about robustness.
    # We report the contiguous band CONTAINING the nominal point over which survival holds at
    # >= half the nominal value, i.e. how far the plant can move before the policy loses half its
    # basin. Nominal survival is printed alongside so the band is interpretable.
    print("[env] MARGIN (contiguous band around nominal holding >=50% of nominal survival):")
    for ax in axes:
        rows = out["axes"][ax]["rows"]
        nom_v = out["axes"][ax]["nominal"]
        i_nom = min(range(len(rows)), key=lambda i: abs(rows[i]["value"] - nom_v))
        s_nom = rows[i_nom]["survival"]
        thr = 0.5 * s_nom
        lo = hi = i_nom
        if s_nom <= 0.0:
            band = f"NONE (0% survival at nominal {nom_v})"
        else:
            while lo > 0 and rows[lo - 1]["survival"] >= thr:
                lo -= 1
            while hi < len(rows) - 1 and rows[hi + 1]["survival"] >= thr:
                hi += 1
            band = (f"{rows[lo]['value']} .. {rows[hi]['value']}"
                    f"   (nominal {nom_v} = {s_nom:.0%} survival)")
        print(f"  {AXES[ax][0]:<24} {band}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\n[env] wrote {args.json}")


if __name__ == "__main__":
    main()
