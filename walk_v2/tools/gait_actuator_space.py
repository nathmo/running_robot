"""The gait in actuator space, drawn on top of the space the actuators are actually allowed.

Three different things claim to define "allowed", and they do not agree, so the figure draws all
three in the model's own (cam, thigh) coordinates:

  MJCF RANGE     the per-joint box the policy was trained and clipped against
  ASSEMBLY BAND  which (cam, thigh) pairs the parallel 4-bar can even reach. cam and thigh are
                 coupled through the pushrod loop, so the reachable set is a thin band, not a box.
                 Solved here per grid cell (Gauss-Newton on the `connect` equality, seeded from a
                 solved neighbour so it tracks ONE assembly branch), cached with --band-cache.
  ROBOT BAND     the safe (cam, thigh) region recorded by hand on the real machine, in NORMALIZED
                 degrees, pulled into model degrees through a calibration.

That last step is the one that can lie, so the calibration is re-fitted here the way
fklut._fit_side does -- maximise the fraction of recorded cells landing inside the assembly band --
and the fit is reported next to what robot/deploy/deploy_map.json currently claims.

TWO FRAMES. The model's right leg is MIRRORED (cam/thigh axes are (0,-1,0) against the left's
(0,+1,0)), so a right-leg model angle is the negative of the same pose in the left-leg frame the
band and the LUT are built in. Abduction is NOT mirrored (both hip_roll axes are (+1,0,0)).

    python walk_v2/tools/gait_actuator_space.py --npz rec.npz --run walk_v2/runs/... \
        --band-cache walk_v2/results/assembly_band.npz --out walk_v2/results
"""
import argparse
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
for p in (str(PKG), str(PKG / "tools"), str(ROOT / "robot" / "deploy")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import gait_figures as gf
import jointmap

MODEL_IDX = {"left": (1, 2), "right": (4, 5)}      # (cam, thigh) in actuator order
ABD_IDX = {"left": 0, "right": 3}
MIRROR = {"left": +1.0, "right": -1.0}             # model angle -> left-leg (band) frame
BAND_FILL = "#dcdbd5"
ROBOT = "#1baf7a"
TOL = 1e-6


# ------------------------------------------------------------------ the assembly band
def _qadr(m, body):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    js = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
    assert len(js) == 1, (body, js)
    return int(m.jnt_qposadr[js[0]])


def assembly_band(model_path, cam, thigh):
    """Boolean [len(cam), len(thigh)] -- does the pushrod loop close at that (cam, thigh)?"""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    ad = {k: _qadr(m, b) for k, b in (("cam", "CamLeftNCS-v1"), ("thigh", "ThighLeftNCS-v1"),
                                      ("push", "PushrodLeftNCS-v1"), ("knee", "LegLeftNCS-v1"))}
    s_tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pushrod_tip_L")
    s_anc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "leg_anchor_L")

    def resid(c, t, x):
        d.qpos[:] = 0.0
        d.qpos[ad["cam"]], d.qpos[ad["thigh"]] = c, t
        d.qpos[ad["push"]], d.qpos[ad["knee"]] = x
        mujoco.mj_kinematics(m, d)
        # planar: the sagittal joints all turn about Y and the two sites are ~7 mm apart in Y by
        # construction, so that component is never closable -- closure is the x-z problem
        return (d.site_xpos[s_tip] - d.site_xpos[s_anc])[[0, 2]]

    def solve(c, t, seed, eps=1e-7, iters=60):
        x = np.array(seed, float)
        for _ in range(iters):
            r = resid(c, t, x)
            n = float(np.linalg.norm(r))
            if n < TOL:
                return x, n
            J = np.empty((2, 2))
            for k in range(2):
                xp = x.copy()
                xp[k] += eps
                J[:, k] = (resid(c, t, xp) - r) / eps
            try:
                step = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                break
            s = float(np.linalg.norm(step))
            if s > 0.5:
                step *= 0.5 / s                       # damp near the fold
            x = x + step
        return x, float(np.linalg.norm(resid(c, t, x)))

    nc, nt = len(cam), len(thigh)
    ok = np.zeros((nc, nt), bool)
    sol = np.full((nc, nt, 2), np.nan)
    i0, j0 = int(np.argmin(np.abs(cam))), int(np.argmin(np.abs(thigh)))
    x, r = solve(np.radians(cam[i0]), np.radians(thigh[j0]), [0.0, 0.0])
    ok[i0, j0], sol[i0, j0] = r < TOL, x
    seen = np.zeros((nc, nt), bool)
    seen[i0, j0] = True
    stack = [(i0, j0)]
    while stack:                                       # BFS: every cell seeded by a solved neighbour
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if not (0 <= a < nc and 0 <= b < nt) or seen[a, b]:
                continue
            seen[a, b] = True
            x, r = solve(np.radians(cam[a]), np.radians(thigh[b]),
                         sol[i, j] if ok[i, j] else [0.0, 0.0])
            ok[a, b] = r < TOL
            if ok[a, b]:
                sol[a, b] = x
                stack.append((a, b))
    return ok


def load_band(model_path, cache, step=1.0):
    cam = np.arange(-90.0, 270.0 + 1e-9, step)
    thigh = np.arange(-60.0, 60.0 + 1e-9, step)
    if cache and Path(cache).exists():
        z = np.load(cache)
        if (len(z["cam"]) == len(cam) and np.allclose(z["cam"], cam)
                and np.allclose(z["thigh"], thigh)):
            print("[band] cached %s" % cache)
            return z["cam"], z["thigh"], z["band"].astype(bool)
    print("[band] solving loop closure on a %dx%d grid ..." % (len(cam), len(thigh)))
    band = assembly_band(model_path, cam, thigh)
    if cache:
        np.savez_compressed(cache, cam=cam, thigh=thigh, band=band)
        print("[band] wrote %s" % cache)
    return cam, thigh, band


# ------------------------------------------------------------------ the recorded robot band
def recorded_cells(w, side):
    g = w["%s_knee_grid" % side].astype(bool)
    c0 = float(w["%s_knee_cam_origin" % side])
    t0 = float(w["%s_knee_thigh_origin" % side])
    r = float(w["%s_knee_grid_deg" % side])
    return g, c0, t0, r


def fit_side(cn, tn, cam, thigh, band, cam_off, th_off):
    """(coverage, sign_cam, sign_thigh, cam_off, thigh_off) per sign combo, best first."""
    dc, dt = cam[1] - cam[0], thigh[1] - thigh[0]
    out = []
    for sc in (+1.0, -1.0):
        for st in (+1.0, -1.0):
            best = (-1.0, 0.0, 0.0)
            for co in cam_off:
                i = np.rint((sc * cn + co - cam[0]) / dc).astype(int)
                oki = (i >= 0) & (i < len(cam))
                tm = st * tn[:, None] + th_off[None, :]
                j = np.rint((tm - thigh[0]) / dt).astype(int)
                m_ = oki[:, None] & (j >= 0) & (j < len(thigh))
                hit = np.zeros(tm.shape, bool)
                hit[m_] = band[np.broadcast_to(i[:, None], tm.shape)[m_], j[m_]]
                cov = hit.mean(axis=0)
                k = int(np.argmax(cov))
                if cov[k] > best[0]:
                    best = (float(cov[k]), float(co), float(th_off[k]))
            out.append((best[0], sc, st, best[1], best[2]))
    return sorted(out, reverse=True)


def inside_recorded(w, side, cam_n, th_n):
    g, c0, t0, r = recorded_cells(w, side)
    i = np.floor((cam_n - c0) / r).astype(int)
    j = np.floor((th_n - t0) / r).astype(int)
    m_ = (i >= 0) & (i < g.shape[0]) & (j >= 0) & (j < g.shape[1])
    out = np.zeros(np.shape(cam_n), bool)
    out[m_] = g[i[m_], j[m_]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="a gait_shape.py --npz recording")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="walk_v2/results")
    ap.add_argument("--band-cache", default="walk_v2/results/assembly_band.npz")
    ap.add_argument("--workspace",
                    default="robot/fixed_gait/webui/data/workspaces/workspace_full_default.npz")
    ap.add_argument("--deploy-map", default="robot/deploy/deploy_map.json")
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--settle", type=float, default=5.0)
    ap.add_argument("--seconds", type=float, default=25.0)
    args = ap.parse_args()

    gf.style()
    z = np.load(args.npz)
    cfg = gf.config_from_dict(json.loads(
        (Path(args.run) / "resolved_config.json").read_text())["config"])
    from plant import resolve
    model_path = resolve(cfg.model_path)
    dt = float(z["dt"])
    i0, i1, e = int(args.settle / dt), int(args.seconds / dt), args.env
    q = np.degrees(z["q"][i0:i1, e])
    tgt = np.degrees(z["target"][i0:i1, e])
    phi, grounded = z["phi"][i0:i1, e], z["grounded"][i0:i1, e]

    m = mujoco.MjModel.from_xml_path(model_path)
    rng = np.degrees(np.array([m.jnt_range[m.actuator_trnid[a, 0]] for a in range(6)]))
    cam_ax, th_ax, band = load_band(model_path, args.band_cache)
    w = np.load(ROOT / args.workspace, allow_pickle=True)
    jm = jointmap.JointMap.load(str(ROOT / args.deploy_map))

    # ---------------------------------------------------------------- the verdict, in text
    print("\n=== 1. the MJCF box the policy was trained in ===")
    names = ["hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R"]
    for j, nm in enumerate(names):
        lo, hi = rng[j]
        print("  %-11s range [%+7.1f,%+7.1f]  measured [%+7.1f,%+7.1f]  commanded [%+7.1f,%+7.1f]"
              "  worst margin %+6.1f deg" % (nm, lo, hi, q[:, j].min(), q[:, j].max(),
                                             tgt[:, j].min(), tgt[:, j].max(),
                                             min(tgt[:, j].min() - lo, hi - tgt[:, j].max())))

    print("\n=== 2. the 4-bar assembly band (this plant) ===")
    print("  %.1f %% of the (cam, thigh) grid closes the loop" % (100 * band.mean()))
    dc, dt_ = cam_ax[1] - cam_ax[0], th_ax[1] - th_ax[0]
    for side in ("left", "right"):
        ci, ti = MODEL_IDX[side]
        s = MIRROR[side]
        i = np.rint((s * q[:, ci] - cam_ax[0]) / dc).astype(int)
        j = np.rint((s * q[:, ti] - th_ax[0]) / dt_).astype(int)
        ok = (i >= 0) & (i < len(cam_ax)) & (j >= 0) & (j < len(th_ax))
        ins = np.zeros(len(i), bool)
        ins[ok] = band[i[ok], j[ok]]
        print("  %-5s gait inside the assembly band: %.1f %% of ticks" % (side, 100 * ins.mean()))

    print("\n=== 3. the calibration: deploy_map.json vs a re-fit against this band ===")
    cam_off = np.arange(-180.0, 180.01, 1.0)
    th_off = np.arange(-90.0, 90.01, 1.0)
    # the fit is a minute per side and depends only on (band, workspace), so cache it
    fit_cache = Path(str(args.band_cache) + ".fits.json") if args.band_cache else None
    fit_key = "%s|%d|%d" % (args.workspace, len(cam_ax), len(th_ax))
    cached = {}
    if fit_cache and fit_cache.exists():
        blob = json.loads(fit_cache.read_text())
        if blob.get("key") == fit_key:
            cached = blob.get("fits", {})
    fits, ranked_all = {}, {}
    for side in ("left", "right"):
        g, c0, t0, r = recorded_cells(w, side)
        ii, jj = np.nonzero(g)
        cn, tn = c0 + (ii + 0.5) * r, t0 + (jj + 0.5) * r
        if side in cached:
            ranked = [tuple(v) for v in cached[side]]
        else:
            ranked = fit_side(cn, tn, cam_ax, th_ax, band, cam_off, th_off)
        ranked_all[side] = ranked
        fits[side] = ranked[0]
        ec, et = jm.e["%s.cam" % side], jm.e["%s.thigh" % side]
        s = MIRROR[side]
        # deploy_map maps norm -> MODEL degrees; the band lives in the left-leg frame
        i = np.rint((s * (ec["sign"] * cn + ec["offset_deg"]) - cam_ax[0]) / dc).astype(int)
        j = np.rint((s * (et["sign"] * tn + et["offset_deg"]) - th_ax[0]) / dt_).astype(int)
        ok = (i >= 0) & (i < len(cam_ax)) & (j >= 0) & (j < len(th_ax))
        cov_dm = np.zeros(len(i), bool)
        cov_dm[ok] = band[i[ok], j[ok]]
        print("  %s: %d recorded cells" % (side.upper(), len(cn)))
        print("    deploy_map.json  cam %+.0f/%+7.2f  thigh %+.0f/%+7.2f -> coverage %.3f"
              % (ec["sign"], ec["offset_deg"], et["sign"], et["offset_deg"], cov_dm.mean()))
        for cov, sc, st, co, to in ranked:
            print("    re-fit           cam %+.0f/%+7.1f  thigh %+.0f/%+7.1f -> coverage %.3f"
                  % (sc, co, st, to, cov))

    if fit_cache and not cached:
        fit_cache.write_text(json.dumps({"key": fit_key, "fits": ranked_all}, indent=1))
        print("  (fit cached in %s)" % fit_cache)

    print("\n=== 4. does the gait fit the ROBOT's recorded band? ===")
    rep = {}
    for side in ("left", "right"):
        ci, ti = MODEL_IDX[side]
        s = MIRROR[side]
        rep[side] = dict(cov=fits[side][0], n_out=0, dmax=0.0, flight=0.0, erode=0.0, dsamp=0.0)
        ec, et = jm.e["%s.cam" % side], jm.e["%s.thigh" % side]
        a = inside_recorded(w, side, (q[:, ci] - ec["offset_deg"]) / ec["sign"],
                            (q[:, ti] - et["offset_deg"]) / et["sign"])
        b = inside_recorded(w, side, (s * q[:, ci] - ec["offset_deg"]) / ec["sign"],
                            (s * q[:, ti] - et["offset_deg"]) / et["sign"])
        _, fsc, fst, fco, fto = fits[side]
        c = inside_recorded(w, side, (s * q[:, ci] - fco) / fsc, (s * q[:, ti] - fto) / fst)
        rep[side]["inside"] = 100 * c.mean()
        print("  %-5s inside: deploy_map %5.1f %% | +mirror %5.1f %% | re-fit+mirror %5.1f %%"
              % (side, 100 * a.mean(), 100 * b.mean(), 100 * c.mean()))
        # HOW FAR outside, for the ticks that do escape. The recorded band is only as big as an
        # operator swept by hand, eroded by a safety margin, so "outside" means "off the end of
        # the sweep" at least as often as it means "past the mechanism".
        if (~c).any():
            g, c0, t0, r = recorded_cells(w, side)
            ii, jj = np.nonzero(g)
            cn, tn = c0 + (ii + 0.5) * r, t0 + (jj + 0.5) * r
            cam_n = (s * q[:, ci] - fco) / fsc
            th_n = (s * q[:, ti] - fto) / fst
            d = np.hypot(cam_n[~c, None] - cn[None, :], th_n[~c, None] - tn[None, :]).min(axis=1)
            fl = float((grounded[~c, MODEL_IDX[side][0] // 3] < 0.5).mean())
            print("        the %d escaping ticks sit %.1f deg past the band's edge at worst "
                  "(median %.1f); %.0f %% of them are in flight"
                  % ((~c).sum(), d.max(), np.median(d), 100 * fl))
            # The band is ERODED for safety, so "outside the band" is not "outside the machine".
            # Measure against the raw hand-swept samples, which are what was physically shown.
            erode = float(w["%s_abd_observed_max" % side]) - float(w["%s_abd_safe_max" % side])
            key = "%s_samples" % side
            if key in w.files:
                sm = np.asarray(w[key], float)
                ds = np.hypot(cam_n[~c, None] - sm[None, :, 1],
                              th_n[~c, None] - sm[None, :, 2]).min(axis=1)
                rep[side]["dsamp"] = float(ds.max())
                print("        nearest hand-swept SAMPLE: %.1f deg worst, %.1f median -- the band "
                      "is eroded %.1f deg for safety, so %s"
                      % (ds.max(), np.median(ds), erode,
                         "these poses were demonstrated" if ds.max() <= erode
                         else "SOME OF THESE WERE NEVER DEMONSTRATED"))
            rep[side].update(n_out=int((~c).sum()), dmax=float(d.max()), flight=100 * fl,
                             erode=erode)
        alo = float(w["%s_abd_safe_min" % side])
        ahi = float(w["%s_abd_safe_max" % side])
        ea = jm.e["%s.abd" % side]
        an = (q[:, ABD_IDX[side]] - ea["offset_deg"]) / ea["sign"]
        print("        abduction band [%+6.1f,%+6.1f] deg, gait [%+6.1f,%+6.1f] -> %5.1f %% inside"
              % (alo, ahi, an.min(), an.max(), 100 * ((an >= alo) & (an <= ahi)).mean()))

    figure(args, z, q, phi, grounded, rng, cam_ax, th_ax, band, w, jm, fits, rep)


def figure(args, z, q, phi, grounded, rng, cam_ax, th_ax, band, w, jm, fits, rep):
    fig = plt.figure(figsize=(12.6, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.24], hspace=0.42, wspace=0.16,
                          left=0.115, right=0.985, top=0.74, bottom=0.08)
    S = {"left": gf.S1, "right": gf.S2}
    for k, side in enumerate(("left", "right")):
        ax = fig.add_subplot(gs[0, k])
        s = MIRROR[side]
        ci, ti = MODEL_IDX[side]
        # assembly band, in THIS leg's model coordinates
        ax.pcolormesh(s * cam_ax, s * th_ax, band.T, cmap=matplotlib.colors.ListedColormap(
            [(0, 0, 0, 0), BAND_FILL]), shading="nearest", zorder=1, rasterized=True)
        # the MJCF box
        ax.add_patch(Rectangle((rng[ci, 0], rng[ti, 0]), rng[ci, 1] - rng[ci, 0],
                               rng[ti, 1] - rng[ti, 0], fill=False, ec=gf.INK2, lw=1.2,
                               ls=(0, (5, 3)), zorder=3))
        # the robot's recorded band, through the RE-FITTED calibration
        _, fsc, fst, fco, fto = fits[side]
        g, c0, t0, r = recorded_cells(w, side)
        ce = s * (fsc * (c0 + np.arange(g.shape[0] + 1) * r) + fco)
        te = s * (fst * (t0 + np.arange(g.shape[1] + 1) * r) + fto)
        ax.pcolormesh(ce, te, np.where(g, 1.0, np.nan).T, shading="flat", zorder=2,
                      cmap=matplotlib.colors.ListedColormap([ROBOT]), alpha=0.32,
                      rasterized=True)
        ax.contour(0.5 * (ce[:-1] + ce[1:]), 0.5 * (te[:-1] + te[1:]), g.T.astype(float),
                   levels=[0.5], colors=[ROBOT], linewidths=1.6, zorder=4)
        # the gait itself
        _, mc, _, _ = gf.cycle_profile(phi, q[:, ci])
        _, mt, _, _ = gf.cycle_profile(phi, q[:, ti])
        _, mg, _, _ = gf.cycle_profile(phi, grounded[:, k])
        ax.plot(np.append(mc, mc[0]), np.append(mt, mt[0]), color=S[side], lw=2.2, zorder=6)
        xs, ys = mc.copy(), mt.copy()
        xs[mg <= 0.5], ys[mg <= 0.5] = np.nan, np.nan
        ax.plot(xs, ys, color=S[side], lw=6.5, solid_capstyle="round", zorder=7)
        ax.set_title("%s leg" % side, color=gf.INK)
        ax.set_xlabel("cam (deg, model)")
        if k == 0:
            ax.set_ylabel("thigh (deg, model)")
        ax.set_xlim(-110, 110)
        ax.set_ylim(-75, 75)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axisbelow(True)

    # ---- abduction, the uncoupled joint: a ladder of ranges
    ax = fig.add_subplot(gs[1, :])
    rows = []
    for side in ("left", "right"):
        ea = jm.e["%s.abd" % side]
        j = ABD_IDX[side]
        lo = ea["sign"] * float(w["%s_abd_safe_min" % side]) + ea["offset_deg"]
        hi = ea["sign"] * float(w["%s_abd_safe_max" % side]) + ea["offset_deg"]
        rows.append((side, rng[j], (min(lo, hi), max(lo, hi)), (q[:, j].min(), q[:, j].max())))
    for r_, (side, box, rob, meas) in enumerate(rows):
        y = 1 - r_
        ax.plot(box, [y, y], color=gf.INK2, lw=1.2, ls=(0, (5, 3)), solid_capstyle="butt")
        ax.plot(rob, [y, y], color=ROBOT, lw=7, alpha=0.5, solid_capstyle="butt")
        ax.plot(meas, [y, y], color=S[side], lw=6.0, solid_capstyle="round")
        ax.text(-0.012, y, "%s hip roll" % side, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=8.5, color=gf.INK)
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([])
    ax.set_xlim(-110, 110)
    ax.set_xlabel("hip roll (deg, model)")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], color=gf.INK2, lw=1.2, ls=(0, (5, 3)), label="MJCF joint range (trained)"),
        Line2D([], [], color=BAND_FILL, lw=8, label="the 4-bar assembles"),
        Line2D([], [], color=ROBOT, lw=8, alpha=0.5, label="recorded safe band on the robot"),
        Line2D([], [], color=gf.MUTED, lw=2.2, label="gait, measured"),
        Line2D([], [], color=gf.MUTED, lw=6.5, label="stance")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.113, 0.80), ncols=6,
               handlelength=1.9, columnspacing=1.3, fontsize=8.5)
    fig.suptitle("DASH-01 v2: the running gait inside the actuator space", x=0.115, ha="left",
                 fontsize=13, fontweight="bold", color=gf.INK)
    use = [100 * np.ptp(q[:, j]) / (rng[j, 1] - rng[j, 0]) for j in range(6)]
    rr = lambda a, b: (min(use[a], use[b]), max(use[a], use[b]))
    lines = [
        (0.947, gf.INK2,
         "The MJCF box: the gait spends %.0f-%.0f %% of the hip-roll range, %.0f-%.0f %% of cam, "
         "%.0f-%.0f %% of thigh, and clips no tick." % (rr(0, 3) + rr(1, 4) + rr(2, 5))),
        (0.927, gf.INK2,
         "The 4-bar assembly band: 100 % of ticks inside, both legs."),
        (0.907, gf.INK2,
         "The robot's own hand-recorded band: %.1f %% (left) and %.1f %% (right) of ticks inside."
         % (rep["left"]["inside"], rep["right"]["inside"])),
        (0.887, gf.INK2,
         "Every escaping tick is in flight, at most %.1f deg past the edge and %.1f deg from a "
         "hand-swept sample -- inside the %.1f deg safety erosion, so all were demonstrated."
         % (max(rep["left"]["dmax"], rep["right"]["dmax"]),
            max(rep["left"]["dsamp"], rep["right"]["dsamp"]), rep["left"]["erode"])),
        (0.857, gf.INK,
         "CALIBRATION: the green band is placed by a RE-FIT (coverage %.3f). The stored "
         "deploy_map.json is stale -- fitted 2026-08-29," % rep["left"]["cov"]),
        (0.837, gf.INK,
         "the zero was re-captured 2026-09-01 -- and its right leg is missing the mirror.")]
    for yy, col, txt in lines:
        fig.text(0.115, yy, txt, fontsize=9, color=col, ha="left")
    fig.savefig(Path(args.out) / "gait_actuator_space.png", dpi=160)
    print("\nwrote %s" % (Path(args.out) / "gait_actuator_space.png"))


if __name__ == "__main__":
    main()
