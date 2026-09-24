"""Stage 5 of the gait-library pipeline (artifact §09): feedforward absorption.

Phase-average the stabilizer's residual over >= 50 cycles, project r_bar(phi) per joint onto the
weighted Fourier basis (least squares), add it to theta, repeat <= 5 times or until
RMS(r_bar) < 0.01 rad. Stays inside the stabilizer's box, so nothing retrains. On the robot the
same pass runs on logged residuals: residual effort far above its simulated value means the
nominal is wrong for the real leg -- absorb once, redeploy.

    python library/absorb.py --run runs/v2_lib_s2_s0 --entry lib.json:1 --out lib_absorbed.json
    python library/absorb.py --entry lib.json:1 --residual-log resid.npz   # hardware residuals
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import gait_v2  # noqa: E402
from fourier_gait import _weights  # noqa: E402

FAMS = (("cam", gait_v2.CAM_L, gait_v2.CAM_R), ("thigh", gait_v2.THIGH_L, gait_v2.THIGH_R),
        ("hip", gait_v2.HIP_ROLL_L, gait_v2.HIP_ROLL_R))


def phase_average(phi, r, n_bins=36):
    """r_bar(phi) per joint over n_bins of phase; phi in [0, 2pi), r [T, 6] in rad."""
    b = np.minimum((phi / (2 * np.pi) * n_bins).astype(int), n_bins - 1)
    out = np.zeros((n_bins, r.shape[1]))
    cnt = np.zeros(n_bins)
    np.add.at(out, b, r)
    np.add.at(cnt, b, 1.0)
    return out / np.maximum(cnt, 1)[:, None], (np.arange(n_bins) + 0.5) * 2 * np.pi / n_bins


def project(theta, r_bar, phi_bins, cfg, lay):
    """Add the residual's Fourier projection to the shared series: the LEFT leg's residual at
    phi and the (mirrored) RIGHT leg's at phi + pi + Delta are two samples of the same S, so both
    are used. Returns the new theta and the RMS of what was absorbed."""
    N = lay.N
    w = _weights(N)
    th = theta.copy()
    s = float(np.clip(th[lay.stride], -1, 1))
    _, _, delta = gait_v2.leg_phases(0.0, th, cfg, lay)
    amps = dict(cam=cfg.cam_amp, thigh=cfg.thigh_amp, hip=cfg.roll_amp)
    fam_slices = dict(cam=lay.s_cam, thigh=lay.s_thigh, hip=lay.s_hip)
    rms = 0.0
    for name, iL, iR in FAMS:
        A = amps[name]
        # samples of S: left at phi (gain 1+s), right at phi_R (gain -(1-s), mirrored)
        phis = np.concatenate([phi_bins, phi_bins - np.pi - delta])
        vals = np.concatenate([r_bar[:, iL] / (A * (1 + s)), -r_bar[:, iR] / (A * (1 - s) + 1e-9)])
        cols = [w[0] * np.ones_like(phis)]
        for k in range(1, N + 1):
            cols += [w[k] * np.cos(k * phis), w[k] * np.sin(k * phis)]
        M = np.stack(cols, 1)
        coef, *_ = np.linalg.lstsq(M, vals, rcond=None)
        th[fam_slices[name]] = np.clip(th[fam_slices[name]] + coef, -1.0, 1.0)
        rms += float(np.mean(r_bar[:, [iL, iR]] ** 2))
    return th, float(np.sqrt(rms / len(FAMS)))


def collect(model, venv, raw, theta, v_ref, cycles=50, seed=1000, max_s=60.0):
    """Roll the frozen stabilizer at theta and log (phi, residual in rad) over >= `cycles`."""
    raw.set_library_entry(theta, v=v_ref)
    venv.seed(seed)
    obs = venv.reset()
    phis, res, n_cyc = [], [], 0
    for _ in range(int(max_s / raw.control_dt)):
        phis.append(float(raw._phase))
        a, _ = model.predict(obs, deterministic=True)
        obs, _, d, info = venv.step(a)
        res.append(raw.cfg.residual_scale * raw._prev_residual.copy())
        n_cyc = info[0].get("v2", {}).get("cycle", n_cyc)
        if d[0] or n_cyc >= cycles:
            break
    return np.asarray(phis), np.asarray(res), n_cyc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--entry", required=True)
    ap.add_argument("--residual-log", default=None, help="npz with phi [T] and residual [T,6] (rad)")
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--cycles", type=int, default=50)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path, i = args.entry.split(":")
    lib = json.loads(Path(path).read_text())
    entry = lib["entries"][int(i)]
    theta = np.asarray(entry["theta"], dtype=float)
    v_ref = float(entry.get("v", 1.0))
    if args.residual_log:
        from config import get_config
        cfg = get_config("v2_lib_s2")
        lay = gait_v2.Layout(cfg.n_harmonics)
        d = np.load(args.residual_log)
        r_bar, phi_b = phase_average(d["phi"], d["residual"])
        theta, rms = project(theta, r_bar, phi_b, cfg, lay)
        print(f"hardware pass: absorbed residual RMS {rms:.4f} rad")
    else:
        from evaluate import build
        model, venv, raw = build(Path(args.run), None, None)
        for p in range(args.passes):
            phi, res, n = collect(model, venv, raw, theta, v_ref, args.cycles)
            r_bar, phi_b = phase_average(phi, res)
            rms_before = float(np.sqrt(np.mean(r_bar ** 2)))
            theta, rms = project(theta, r_bar, phi_b, raw.cfg, raw._lay)
            print(f"pass {p}: {n} cycles, phase-averaged residual RMS {rms_before:.4f} rad")
            if rms_before < args.tol:
                break
    out = dict(source=args.entry, entries=[dict(v=v_ref, theta=theta.tolist(),
                                                x_star=entry.get("x_star"), source="absorb")]
               + [e for e in lib["entries"] if e.get("v", 0.0) == 0.0])
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
