"""Greedy (or stochastic) episodes of a v2 checkpoint: length, distance, fall cause, COMMITTED clock
frequency (commit ticks only), residual saturation, committed-spec content, residual authority.
  python walk_mit/monitor/greedy_peek.py RUN CKPT_STEM [--episodes 6] [--stochastic] [--dr 0] [--assist 0.37]

The fall cause is read from the env's OWN termination checks (recorded through a wrapper), never
re-evaluated from the hook: _workspace_violation() advances the per-foot grace timer on every
call, so an extra call per tick halved the 0.10 s grace and killed healthy gaits at 13-19 ticks
(the 2026-09-10 "ws at reset" artifact on every v2/v2b checkpoint).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from evaluate import build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("ckpt")
ap.add_argument("--episodes", type=int, default=6)
ap.add_argument("--stochastic", action="store_true")
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--dr", type=float, default=None, help="override dr_scale (default: the run's curriculum value)")
ap.add_argument("--assist", type=float, default=None, help="override the pitch-assist scale")
ap.add_argument("--sprint", type=float, default=None, help="override the sprint line distance (m)")
args = ap.parse_args()
run = Path(args.run)
model, venv, raw = build(run, None, str(run / args.ckpt))
cfg = raw.cfg
if args.dr is not None:
    raw.set_dr_scale(args.dr)
if args.assist is not None:
    raw.set_pitch_assist(args.assist)
if args.sprint is not None:
    raw.set_sprint_dist(args.sprint)

cause_buf = []


def snap():
    """After each tick's physics, BEFORE the env's termination checks: read-only state only."""
    d = raw.data
    z = float(d.qpos[2])
    cause_buf.append(dict(low=z < cfg.term_height, tip=raw._gravity_body()[2] > cfg.term_gravity_z,
                          floor=False, ws=False, nan=not np.all(np.isfinite(d.qpos)),
                          z=z, pitch=float(d.qpos[4])))


raw.on_control_step = snap
_ws_orig, _fl_orig = raw._workspace_violation, raw._floor_violation


def _ws_rec():
    r = _ws_orig()
    if cause_buf:
        cause_buf[-1]["ws"] = bool(r)
    return r


def _fl_rec():
    r = _fl_orig()
    if cause_buf:
        cause_buf[-1]["floor"] = bool(r)
    return r


raw._workspace_violation, raw._floor_violation = _ws_rec, _fl_rec

det = not args.stochastic
print(f"[{run.name} {args.ckpt}] {'stochastic' if args.stochastic else 'greedy'}, {args.episodes} eps, "
      f"dr_scale={raw._dr.scale:.3f} pitch_assist={raw._pitch_assist:.3f} sprint={args.sprint}")
rows = []
nom = np.asarray(raw.nominal_ctrl)[:6]
for e in range(args.episodes):
    venv.seed(args.seed0 + e)
    obs = venv.reset()
    cause_buf.clear()
    fhz, res, fam, dev, td = [], [], [], [], []
    ncommit, n, sprint, theta = 0, 0, None, 0.0
    while True:
        a, _ = model.predict(obs, deterministic=det)
        obs, r, d, info = venv.step(a)
        a0 = np.clip(a[0], -1, 1)
        res.append(a0[44:50])
        n += 1
        v2 = info[0].get("v2", {})
        if v2.get("commit"):
            ncommit += 1
            fhz.append(v2["f_hz"])
            sl = np.asarray(raw._spec_live, dtype=float)
            fam.append([np.sqrt(np.mean(sl[0:7] ** 2)), np.sqrt(np.mean(sl[7:14] ** 2)), np.sqrt(np.mean(sl[14:21] ** 2)),
                        np.sqrt(np.mean(sl[21:28] ** 2)), np.sqrt(np.mean(sl[28:35] ** 2)), *sl[39:44]])
        dev.append(raw._ring[(raw._ring_head - 1) % raw._ring_len][:6] - nom)
        td.append(v2.get("td_err", np.nan))
        theta = max(theta, v2.get("theta_max", 0.0))
        sprint = info[0].get("sprint", sprint)
        if d[0]:
            trunc = bool(info[0].get("TimeLimit.truncated", False))
            finished = sprint is not None and sprint.get("t_line") is not None
            break
    res = np.asarray(res)
    dev = np.asarray(dev)
    fhz = np.asarray(fhz) if fhz else np.array([np.nan])
    fam = np.asarray(fam).mean(axis=0) if fam else np.full(10, np.nan)
    dev_rms = float(np.sqrt(np.mean(dev ** 2)))
    auth = float(np.sqrt(np.mean((cfg.residual_scale * res) ** 2)) / max(1e-9, dev_rms))
    c = cause_buf[-1]
    cause = "finish" if finished else ("timeout" if trunc else ",".join(k for k in ("low", "tip", "floor", "ws", "nan") if c[k]) or "?")
    dist = np.nan if sprint is None else sprint["d"]
    t = n * raw.control_dt
    sat = float(np.mean(np.abs(res) >= 0.95))
    tdm = float(np.nanmean(np.abs(td))) if np.isfinite(np.nanmean(td)) else np.nan
    rows.append(dict(n=n, t=t, dist=dist, cause=cause, fmed=np.nanmedian(fhz), sat=sat))
    print(f"  ep{e}: {n:5d} ticks {t:6.2f} s  x {dist:6.2f} m  v {dist / t if t > 0 else 0:5.2f} m/s  end={cause:8s} "
          f"commits {ncommit:3d}  f_hz med {np.nanmedian(fhz):.2f} [{np.nanmin(fhz):.2f},{np.nanmax(fhz):.2f}]  "
          f"res sat {sat:.2f} rms {np.sqrt(np.mean(res ** 2)):.2f}  theta_max {theta:.2f}  |td| {tdm:.3f}  "
          f"final z {c['z']:.2f} pitch {c['pitch']:+.2f}", flush=True)
    print(f"        spec rms  cam {fam[0]:.2f} thigh {fam[1]:.2f} hip {fam[2]:.2f} kp {fam[3]:.2f} kd {fam[4]:.2f} | "
          f"knobs delta {fam[5]:+.2f} s {fam[6]:+.2f} o {fam[7]:+.2f},{fam[8]:+.2f},{fam[9]:+.2f} | "
          f"residual share of target deviation {auth:.2f}  target-dev rms {dev_rms:.3f} rad", flush=True)
T = np.array([r["t"] for r in rows])
D = np.array([r["dist"] for r in rows])
print(f"  MEAN: {T.mean():6.2f} s  {np.nanmean(D):6.2f} m  {np.nanmean(D) / T.mean():.2f} m/s  "
      f"f_hz med {np.nanmedian([r['fmed'] for r in rows]):.2f}  res sat {np.mean([r['sat'] for r in rows]):.2f}  "
      f"causes {dict(zip(*np.unique([r['cause'] for r in rows], return_counts=True)))}")
