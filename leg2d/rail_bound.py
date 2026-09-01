"""Optimistic upper bound on running speed from a rail-mounted, UNLOADED leg sweep.

THE MODEL. Grounded (no-flight) running at duty 0.5 moves the body at exactly the speed the
stance foot sweeps backward relative to the body. So an upper bound on running speed is: weld the
torso to the world (rail), remove all contact (the leg carries nothing -- not even itself against
a floor), and measure the fastest PERIODIC back-and-forth fore-aft toe sweep the leg can sustain.
Deliberately optimistic -- no body weight, no ground-reaction requirement, no balance -- it brackets
every real gait from above (flight phases excepted: ballistic flight can stretch the effective
stride beyond the workspace, so treat this as the bound for grounded gaits).

Two passes:
  PASS 1 -- kinematic (velocity-limit only). Unlimited torque; the only physics is the AKE90-8
      no-load JOINT speed (22 rad/s, motor.NO_LOAD_RAD_S) mapped through the measured path
      Jacobian dq/dx. Time-optimal traversal of the workspace arc:
          v_lim(x) = NO_LOAD / max(|dq_cam/dx|, |dq_thigh/dx|),  T = integral dx / v_lim(x).
      Reported both over the FULL arc and over the best sub-segment (the arc ends are
      Jacobian-expensive: a lot of joint travel buys almost no toe travel).
  PASS 2 -- real torque-speed envelope (motor.clamp_torque: 144.5 N*m falling linearly to zero at
      22 rad/s, full peak when braking) in the full MuJoCo dynamics. Bang-bang target switching
      between two poses on the arc (a grid over end-margins finds the best stride/cadence
      trade-off); the metric is the measured mean |xdot_toe| over an integer number of
      steady-state cycles = 2 * extent / period.

THE WORKSPACE PATH is measured, not assumed: a warm-started PD scan over (cam, thigh) targets
(the same scheme as training/build_cpg_lut.py, same fold-branch filter dz <= 0.40 m), then the
lower envelope over fore-aft bins = the maximum-extension arc (memory/foot-arc-geometry.md: the
foot workspace is effectively this one arc). The arc is then verified by a slow tracked sweep and
all analytics use the MEASURED slow-sweep joint/toe trajectories, not the commanded ones.

Outputs: results/rail_bound.json, results/rail_pass1.mp4, results/rail_pass2.mp4
(each video: a few real-time cycles, then one cycle at 10x slow motion).

Run:  .venv/Scripts/python.exe leg2d/rail_bound.py            # everything (scan is cached)
      .venv/Scripts/python.exe leg2d/rail_bound.py --rescan   # force a fresh workspace scan
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
import motor  # noqa: E402

MODEL_PATH = PKG / "model" / "leg2d.xml"
RESULTS = PKG / "results"
SCAN_NPZ = RESULTS / "rail_scan.npz"
PATH_NPZ = RESULTS / "rail_path.npz"
OUT_JSON = RESULTS / "rail_bound.json"

NO_LOAD = motor.NO_LOAD_RAD_S            # 22 rad/s, joint side, both AKE90-8
PEAK = motor.PEAK_NM                     # 144.5 N*m delivered peak

# scan box (target deltas from the keyframe pose). Joint ranges are cam +-1.5 / thigh +-1.047;
# the box stays inside them and the fold filter drops the snapped-through branch afterwards.
CAM_HALF, THIGH_HALF = 1.2, 0.9
N_CAM, N_THIGH = 33, 25
FOLD_DZ = 0.40                           # same folded-branch cut as build_cpg_lut.py
SCAN_KP, SCAN_KV = 200.0, 5.0            # dash01-like PD for the scan/slow sweep (real-ish drive)
SETTLE_COL, SETTLE_PT = 400, 250         # physics steps (1 ms) per column start / per grid point

VIDEO_FPS = 50


# --------------------------------------------------------------------------------------- rig ----
class Rig:
    """leg2d.xml with the torso welded to the world and ALL contact disabled (the no-load rail).

    Optional what-if scales (all default 1.0, used by rail_sensitivity.py):
      leg_mass_scale  -- scales mass AND rotational inertia of every body in the leg subtree
      armature_scale  -- scales cam/thigh reflected rotor inertia (dof armature)
      peak_scale      -- scales the deliverable peak torque (envelope + actuator forcerange)
      w0_scale        -- scales the no-load speed (back-EMF ceiling: proportional to bus voltage)
    """

    def __init__(self, base_h=None, leg_mass_scale=1.0, armature_scale=1.0,
                 peak_scale=1.0, w0_scale=1.0):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        m = self.model
        self.a_cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "cam_L")
        self.a_thigh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "thigh_L")
        self.a_hr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_roll_L")
        # joint ids via the actuator transmission, NOT retyped names (mis-encoded CAD strings --
        # see build_leg2d.py's warning; mj_name2id would silently return -1).
        j_cam = int(m.actuator_trnid[self.a_cam, 0])
        j_thigh = int(m.actuator_trnid[self.a_thigh, 0])
        self.qadr_cam, self.dadr_cam = int(m.jnt_qposadr[j_cam]), int(m.jnt_dofadr[j_cam])
        self.qadr_thigh, self.dadr_thigh = int(m.jnt_qposadr[j_thigh]), int(m.jnt_dofadr[j_thigh])
        self.g_toe = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "foot_L_col")
        assert self.g_toe >= 0
        self.peak = PEAK * peak_scale
        self.w0 = NO_LOAD * w0_scale
        for a in (self.a_cam, self.a_thigh):
            m.actuator_forcerange[a] = (-self.peak, self.peak)
            m.actuator_ctrlrange[a] = (-self.peak, self.peak)
        if armature_scale != 1.0:
            m.dof_armature[self.dadr_cam] *= armature_scale
            m.dof_armature[self.dadr_thigh] *= armature_scale
        if leg_mass_scale != 1.0:
            b0 = int(m.jnt_bodyid[int(m.actuator_trnid[self.a_hr, 0])])
            for b in range(m.nbody):
                bb = b
                while bb not in (0, b0):
                    bb = int(m.body_parentid[bb])
                if bb == b0:
                    m.body_mass[b] *= leg_mass_scale
                    m.body_inertia[b] *= leg_mass_scale
        self.data = mujoco.MjData(self.model)
        self.base_h = float(base_h) if base_h is not None else None
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        if self.base_h is not None:
            self.data.qpos[2] = self.base_h
        self._pin()
        mujoco.mj_forward(self.model, self.data)
        self.q0 = np.array([self.data.qpos[self.qadr_cam], self.data.qpos[self.qadr_thigh]])

    def _pin(self):
        d = self.data
        d.qpos[0], d.qpos[1] = 0.0, 0.0
        if self.base_h is not None:
            d.qpos[2] = self.base_h
        d.qpos[3:6] = 0.0
        d.qvel[0:6] = 0.0

    def step_tau(self, tau_cam, tau_thigh):
        d = self.data
        d.ctrl[self.a_cam] = tau_cam
        d.ctrl[self.a_thigh] = tau_thigh
        d.ctrl[self.a_hr] = 0.0
        mujoco.mj_step(self.model, self.data)
        self._pin()

    def q(self):
        d = self.data
        return np.array([d.qpos[self.qadr_cam], d.qpos[self.qadr_thigh]])

    def dq(self):
        d = self.data
        return np.array([d.qvel[self.dadr_cam], d.qvel[self.dadr_thigh]])

    def toe(self):
        """Toe (x, z) relative to the torso (torso welded at x=0; z measured from the base joint,
        so the same number regardless of which rail height the rig was built at)."""
        p = self.data.geom_xpos[self.g_toe]
        return np.array([p[0], p[2] - self.data.qpos[2]])

    def pd_step(self, q_target, kp, kv, clamp):
        q, dq = self.q(), self.dq()
        tau = kp * (np.asarray(q_target) - q) - kv * dq
        if clamp:
            tau = motor.clamp_torque(tau, dq, peak=self.peak, omega0=self.w0)
        else:
            tau = np.clip(tau, -1e5, 1e5)
        self.step_tau(float(tau[0]), float(tau[1]))
        return tau


def unlock_torque(rig):
    """PASS-1 only: lift the XML's +-144.5 N*m ctrl/force clamps (unlimited-torque thought
    experiment). The velocity limit is enforced by the reference trajectory, not the actuator."""
    for a in (rig.a_cam, rig.a_thigh):
        rig.model.actuator_ctrlrange[a] = (-1e5, 1e5)
        rig.model.actuator_forcerange[a] = (-1e5, 1e5)


# ------------------------------------------------------------------------------ workspace scan ----
def scan_workspace(rescan=False):
    if SCAN_NPZ.exists() and not rescan:
        z = np.load(SCAN_NPZ)
        return z["targets"], z["q_meas"], z["toe"]
    print(f"[scan] PD scan {N_CAM}x{N_THIGH} over cam +-{CAM_HALF} / thigh +-{THIGH_HALF} rad ...")
    rig = Rig(base_h=2.0)                        # high in the air; contact is off anyway
    cam_grid = np.linspace(-CAM_HALF, CAM_HALF, N_CAM)
    thigh_grid = np.linspace(-THIGH_HALF, THIGH_HALF, N_THIGH)
    targets, q_meas, toe = [], [], []
    for dc in cam_grid:
        rig.reset()                              # fresh column, same as build_cpg_lut
        col = rig.q0 + np.array([dc, 0.0])
        for _ in range(SETTLE_COL):
            rig.pd_step(col, SCAN_KP, SCAN_KV, clamp=True)
        for dt_ in thigh_grid:
            tgt = rig.q0 + np.array([dc, dt_])
            for _ in range(SETTLE_PT):
                rig.pd_step(tgt, SCAN_KP, SCAN_KV, clamp=True)
            targets.append(tgt)
            q_meas.append(rig.q())
            toe.append(rig.toe())
    targets, q_meas, toe = map(np.array, (targets, q_meas, toe))
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(SCAN_NPZ, targets=targets, q_meas=q_meas, toe=toe)
    print(f"[scan] wrote {SCAN_NPZ} ({len(toe)} points)")
    return targets, q_meas, toe


def build_arc(targets, q_meas, toe, bin_w=0.02, band=0.015):
    """Lower envelope of the unfolded workspace cloud over fore-aft bins = the max-extension arc.
    Branch-continuous: middle-out greedy pick (closest joints to the neighbour bin) among the
    candidates within `band` of each bin's lowest toe. Returns target poses + measured toe."""
    # Fold cut is ABSOLUTE from max extension: the 4-bar is redundant (same cam/thigh, two leg
    # shapes), so referencing "the point nearest the nominal joints" can land on the folded branch
    # and keep everything. The running-relevant workspace is the band within FOLD_DZ of the
    # longest measured leg; everything above it is the snapped/folded branch or leg-up poses.
    keep = toe[:, 1] <= toe[:, 1].min() + FOLD_DZ
    t_k, q_k, p_k = targets[keep], q_meas[keep], toe[keep]
    print(f"[arc] {keep.sum()}/{len(keep)} points on the unfolded branch; "
          f"x span [{p_k[:, 0].min():+.3f}, {p_k[:, 0].max():+.3f}] m")
    bins = np.arange(p_k[:, 0].min(), p_k[:, 0].max() + bin_w, bin_w)
    idx_of_bin = {}
    for b in range(len(bins) - 1):
        sel = np.where((p_k[:, 0] >= bins[b]) & (p_k[:, 0] < bins[b + 1]))[0]
        if len(sel):
            zlo = p_k[sel, 1].min()
            idx_of_bin[b] = sel[p_k[sel, 1] <= zlo + band]
    order = sorted(idx_of_bin)
    mid = order[len(order) // 2]
    chosen = {}

    def pick(b, anchor):
        cand = idx_of_bin[b]
        k = cand[np.argmin(np.linalg.norm(q_k[cand] - anchor, axis=1))] if anchor is not None \
            else cand[np.argmin(p_k[cand, 1])]
        chosen[b] = k

    pick(mid, None)
    for b in [x for x in order if x > mid]:
        pick(b, q_k[chosen[max(c for c in chosen if c < b)]])
    for b in sorted((x for x in order if x < mid), reverse=True):
        pick(b, q_k[chosen[min(c for c in chosen if c > b)]])
    ks = [chosen[b] for b in sorted(chosen)]
    arc_t, arc_p = t_k[ks], p_k[ks]
    # The raw envelope picks still hop between redundant 4-bar solutions (measured: a slow tracked
    # sweep of them moves the toe monotonically only ~70% of the time). Fit a SMOOTH low-order
    # curve (cam(u), thigh(u), u = normalized toe x) through the picks, with one outlier-rejection
    # refit, and use THAT as the path -- downstream everything is measured off a slow sweep of the
    # fitted path anyway, so the fit only needs to stay on one branch, not be exact.
    u = (arc_p[:, 0] - arc_p[0, 0]) / (arc_p[-1, 0] - arc_p[0, 0])
    sm = arc_t
    for _ in range(2):
        coef = [np.polyfit(u, sm_col, 5) for sm_col in arc_t.T]
        fit = np.stack([np.polyval(c, u) for c in coef], axis=1)
        resid = np.linalg.norm(fit - arc_t, axis=1)
        ok = resid < max(3 * resid.std(), 0.05)
        u, arc_t, arc_p = u[ok], arc_t[ok], arc_p[ok]
    uu = np.linspace(0, 1, 60)
    sm = np.stack([np.polyval(c, uu) for c in coef], axis=1)
    print(f"[arc] {len(sm)} fitted path poses; toe x [{arc_p[0, 0]:+.3f} .. {arc_p[-1, 0]:+.3f}] m, "
          f"toe z range [{arc_p[:, 1].min():+.3f}, {arc_p[:, 1].max():+.3f}] (rel torso)")
    return sm, arc_p


def slow_verify(arc_targets, one_way_s=6.0, save=True, tag="arc"):
    """Track the path slowly (quasi-static), record the MEASURED joint/toe path both ways.
    Everything downstream (pass-1 analytics, pass-2 end poses, video keyposes) uses this."""
    lo_z = None
    rig = Rig(base_h=2.0)
    n = len(arc_targets)
    rec = dict(t=[], q=[], tgt=[], toe=[], qpos=[])
    # settle to the arc start first
    for _ in range(1500):
        rig.pd_step(arc_targets[0], SCAN_KP, SCAN_KV, clamp=True)
    steps = int(one_way_s * 1000)
    for k in range(2 * steps):                    # fwd then back
        u = k / steps
        s = u if u <= 1.0 else 2.0 - u
        f = s * (n - 1)
        i0 = min(int(f), n - 2)
        tgt = arc_targets[i0] * (1 - (f - i0)) + arc_targets[i0 + 1] * (f - i0)
        rig.pd_step(tgt, SCAN_KP, SCAN_KV, clamp=True)
        if k % 5 == 0:
            rec["t"].append(k * 1e-3)
            rec["q"].append(rig.q())
            rec["tgt"].append(tgt.copy())
            rec["toe"].append(rig.toe())
            rec["qpos"].append(rig.data.qpos.copy())
    rec = {k: np.array(v) for k, v in rec.items()}
    half = len(rec["t"]) // 2
    fwd_toe, back_toe = rec["toe"][:half], rec["toe"][half:]
    x_f = fwd_toe[:, 0]
    hyst = abs(back_toe[-1, 0] - x_f[0])          # does the return sweep retrace to the start?
    print(f"[verify:{tag}] measured one-way toe x: {x_f[0]:+.3f} -> {x_f[-1]:+.3f} m "
          f"(L = {abs(x_f[-1] - x_f[0]):.3f} m), monotone frac "
          f"{np.mean(np.sign(np.diff(x_f)) == np.sign(x_f[-1] - x_f[0])):.2f}, "
          f"return hysteresis {hyst * 1000:.1f} mm")
    if save:
        lo_z = rec["toe"][:, 1].min()
        np.savez(PATH_NPZ, **rec, arc_targets=arc_targets, lo_z=lo_z)
        print(f"[verify:{tag}] wrote {PATH_NPZ}")
    return rec


# ------------------------------------------------------------------------------------- pass 1 ----
def pass1_analytics(rec, arc_p=None, tag="arc", min_seg=0.25, band=0.15, x_cap=None):
    """Time-optimal traversal under the joint no-load speed alone (unlimited torque).

    If `arc_p` (the scan's settled arc picks) is given, the measured sweep is first TRIMMED to
    the longest contiguous stretch that stays within `band` of the settled arc height z(x) --
    `band` (0.15 m) is a FOLD detector, wide enough to pass the ~<=0.1 m quasi-static PD droop of
    a moving sweep vs the 250 ms settled scan points, far below the ~1.4 m fold excursion: the
    scan proves poses beyond that exist as isolated settled points, but traversing to them along
    the sweep snaps the 4-bar through its fold (measured: the front ~0.45 m of the raw envelope
    is only "covered" by the toe swinging up-and-forward as the knee folds -- x keeps growing
    while the foot leaves the workspace, which also fakes dq/dx ~ 0 "fast" segments). Only the
    path-TRAVERSABLE workspace counts.

    v_lim is capped at 2 * R * NO_LOAD and sub-segments shorter than `min_seg` are ignored --
    guards against flat-spot artifacts in the measured q(x)."""
    half = len(rec["t"]) // 2
    q, toe = rec["q"][:half], rec["toe"][:half]
    tgt = rec["tgt"][:half]
    R_max = float(np.sqrt((rec["toe"] ** 2).sum(axis=1)).max())
    if arc_p is not None:
        o_arc = np.argsort(arc_p[:, 0])
        z_arc = np.interp(np.clip(toe[:, 0], arc_p[o_arc[0], 0], arc_p[o_arc[-1], 0]),
                          arc_p[o_arc, 0], arc_p[o_arc, 1])
        good = np.abs(toe[:, 1] - z_arc) <= band
        runs, s = [], None
        for i, g in enumerate(good):
            if g and s is None:
                s = i
            if (not g or i == len(good) - 1) and s is not None:
                runs.append((s, i if g else i - 1))
                s = None
        s, e = max(runs, key=lambda r: abs(toe[r[1], 0] - toe[r[0], 0]))
        print(f"[pass1:{tag}] trimmed to the on-arc traversable stretch: "
              f"x [{toe[s, 0]:+.3f} .. {toe[e, 0]:+.3f}] m "
              f"(dropped {len(good) - (e - s + 1)}/{len(good)} fold-contaminated samples)")
        q, toe, tgt = q[s:e + 1], toe[s:e + 1], tgt[s:e + 1]
        # traversable ALSO means monotone under slow tracking: near the fold boundary the toe
        # stalls and see-saws several cm while the linkage fights the snap-through (measured:
        # a 6 cm x-reversal at the front edge) -- cut at the first >4 cm reversal against the
        # running extreme; poses beyond it are approachable but not sweep-traversable.
        xf = toe[:, 0]
        sgn_l = np.sign(xf[-1] - xf[0])
        run_ext = np.maximum.accumulate(sgn_l * xf)
        bad = np.where(run_ext - sgn_l * xf > 0.04)[0]
        if len(bad):
            e2 = int(np.argmax(sgn_l * xf[:bad[0]]))
            print(f"[pass1:{tag}] monotone trim: fold see-saw at x = {xf[bad[0]]:+.3f}; "
                  f"path ends at x = {xf[e2]:+.3f}")
            q, toe, tgt = q[:e2 + 1], toe[:e2 + 1], tgt[:e2 + 1]
    if x_cap is not None:
        # empirical fold boundary from a quasi-static probe with the PASS-2 controller itself
        # (the fold-marginal front zone is hypersensitive to controller details, so the only
        # self-consistent workspace is the one THAT controller can traverse)
        sgn_l = np.sign(toe[-1, 0] - toe[0, 0])
        m_cap = sgn_l * toe[:, 0] <= x_cap
        q, toe, tgt = q[m_cap], toe[m_cap], tgt[m_cap]
        print(f"[pass1:{tag}] fold-probe cap: path ends at x = {x_cap:+.3f}")
    x = toe[:, 0]
    sgn = np.sign(x[-1] - x[0])
    keep = np.where(np.abs(np.diff(x)) > 1e-6)[0]         # drop stalled samples
    x_m, q_m, t_m = x[keep], q[keep], np.asarray(tgt)[keep]
    o = np.argsort(sgn * x_m)
    x_m, q_m, t_m = sgn * x_m[o], q_m[o], t_m[o]           # ascending
    # resample onto a uniform x grid, smooth joints a touch, differentiate
    xs = np.linspace(x_m[0], x_m[-1], 400)
    qc = np.interp(xs, x_m, q_m[:, 0])
    qt = np.interp(xs, x_m, q_m[:, 1])
    k = 7
    ker = np.ones(k) / k
    qc = np.convolve(np.pad(qc, k // 2, mode="edge"), ker, mode="valid")
    qt = np.convolve(np.pad(qt, k // 2, mode="edge"), ker, mode="valid")
    dqc = np.gradient(qc, xs)
    dqt = np.gradient(qt, xs)
    v_lim = NO_LOAD / np.maximum(np.abs(dqc), np.abs(dqt))
    # physical cap: BOTH joints can add up to ~R*NO_LOAD of toe speed each (on this leg both cam
    # and thigh move the toe mostly fore-aft, see build_cpg_lut.py) -- anything above 2*R*w0 in
    # the measured-Jacobian estimate is a flat-spot artifact, not kinematics
    v_lim = np.minimum(v_lim, 2.0 * R_max * NO_LOAD)
    dx = xs[1] - xs[0]
    dt_ = dx / v_lim
    T_full = float(dt_.sum())
    L = float(xs[-1] - xs[0])
    # time at each node, then best sub-segment >= min_seg: max (x_b - x_a) / (t_b - t_a)
    tt = np.concatenate([[0.0], np.cumsum(dt_[:-1])])
    best = (0.0, 0, len(xs) - 1)
    for a in range(0, len(xs) - 1, 2):
        vv = (xs[a + 1:] - xs[a]) / (tt[a + 1:] - tt[a])
        vv[xs[a + 1:] - xs[a] < min_seg] = 0.0
        b = int(np.argmax(vv))
        if vv[b] > best[0]:
            best = (float(vv[b]), a, a + 1 + b)
    v_best, a, b = best
    # measured toe height along the path (for the stay-on-the-arc band check in pass 2) and the
    # COMMANDED targets along it (pass 2 must track the branch-clean commanded path -- feeding
    # back measured joints as targets mixes droop and, near the singular front, branch states)
    z_m = toe[keep, 1][o]  # noqa: same keep/order as x_m
    zs = np.interp(xs, x_m, z_m)
    zs = np.convolve(np.pad(zs, k // 2, mode="edge"), ker, mode="valid")
    cc = np.interp(xs, x_m, t_m[:, 0])
    ct = np.interp(xs, x_m, t_m[:, 1])
    cc = np.convolve(np.pad(cc, k // 2, mode="edge"), ker, mode="valid")
    ct = np.convolve(np.pad(ct, k // 2, mode="edge"), ker, mode="valid")
    out = dict(L_full=L, T_oneway=T_full, v_full=L / T_full,
               v_best=v_best, seg=[float(xs[a]), float(xs[b])],
               x_grid=xs, t_of_x=tt, v_lim=v_lim, q_cam=qc, q_thigh=qt,
               c_cam=cc, c_thigh=ct, z_of_x=zs, sgn=float(sgn))
    print(f"[pass1:{tag}] L = {L:.3f} m, one-way T = {T_full * 1000:.1f} ms  ->  "
          f"v_full = {out['v_full']:.2f} m/s;  best sub-segment (>= {min_seg} m) "
          f"[{xs[a]:+.3f},{xs[b]:+.3f}] ({xs[b] - xs[a]:.3f} m) -> v = {v_best:.2f} m/s")
    return out


def render_frames(model, renderer, cam, qpos_seq, writer):
    d = mujoco.MjData(model)
    for qp in qpos_seq:
        d.qpos[:] = qp
        mujoco.mj_forward(model, d)
        renderer.update_scene(d, cam)
        writer.append_data(renderer.render())


def pass1_video(rec, p1, base_h, out=RESULTS / "rail_pass1.mp4"):
    """Kinematic playback of the recorded slow-sweep poses on the pass-1 time-optimal clock
    (this is a KINEMATIC bound -- the video is its visualization, not a dynamics run)."""
    import imageio.v2 as imageio
    half = len(rec["t"]) // 2
    toe, qpos = rec["toe"][:half], rec["qpos"][:half]
    x = toe[:, 0]
    sgn = np.sign(x[-1] - x[0])
    xs, t_of_x = p1["x_grid"], p1["t_of_x"]
    T = p1["T_oneway"]

    def qpos_at_x(xq):                                     # nearest recorded pose for toe x
        return qpos[np.argmin(np.abs(sgn * x - xq))]

    def x_at_t(tq):                                        # invert the time-optimal clock
        return np.interp(tq, t_of_x, xs)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 960)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 540)
    renderer = mujoco.Renderer(model, 540, 960)
    camv = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camv)
    camv.distance, camv.elevation, camv.azimuth = 1.9, -12, 90
    camv.lookat[:] = [0.05, 0.0, base_h * 0.55]

    seq = []
    n_rt = max(3, int(np.ceil(1.5 / (2 * T))))             # >= ~1.5 s of real-time footage
    for phase, cycles in ((1.0, n_rt), (0.1, 1)):          # real time, then 10x slow-mo
        step = phase / VIDEO_FPS
        t, direction = 0.0, +1
        while cycles > 0:
            xq = x_at_t(t if direction > 0 else T - t)
            qp = qpos_at_x(xq).copy()
            qp[2] = base_h
            seq.append(qp)
            t += step
            if t >= T:
                t -= T
                direction *= -1
                if direction > 0:
                    cycles -= 1
    writer = imageio.get_writer(str(out), fps=VIDEO_FPS)
    render_frames(model, renderer, camv, seq, writer)
    writer.close()
    print(f"[pass1] wrote {out} ({len(seq)} frames: {n_rt} real-time cycles + 1 at 10x slow-mo)")


# -------------------------------------------------------------------- pass 2 (arc-constrained) ----
def pass2_tracked(p1, base_h, delay_ms=7, band=0.05, kp=400.0, kv=6.0, cycles=5, k_max=40.0,
                  record_qpos=False, k_force=None, rig_kwargs=None, n_k=24,
                  refines=(0.94, 0.88)):
    """Torque-limited sweep CONSTRAINED TO THE WORKSPACE ARC, with actuator bandwidth.

    The reference retraces the measured arc in both directions using the pass-1 time-optimal
    profile uniformly slowed by k; the smallest feasible k is found by bisection. A cycle is
    feasible only if, over the steady-state window, the toe (a) never strays more than `band`
    from the measured arc height z(x) and never leaves the x-range, and (b) covers >= 90% of the
    arc extent -- so no folded-knee flailing and no off-workspace travel can count toward speed.

    `delay_ms` is a pure transport delay on the torque command (the measured MIT-mode
    command->response latency, memory/mit-bandwidth-measured.md: ~7 ms; servo SET_POS would be
    ~200 ms). Torque is computed on the state at time t and reaches the joint at t + delay.
    """
    xs, tt, zs = p1["x_grid"], p1["t_of_x"], p1["z_of_x"]
    qc, qt = p1["c_cam"], p1["c_thigh"]                    # commanded targets, not measured joints
    T1 = p1["T_oneway"]
    L = xs[-1] - xs[0]
    sgn = p1["sgn"]

    def ref_at(t, k):
        """(q_ref, x_ref) at time t for a cycle of period 2*k*T1 (fwd then back along the arc)."""
        tc = t % (2.0 * k * T1)
        tau = tc / k if tc < k * T1 else (2.0 * k * T1 - tc) / k
        x = np.interp(tau, tt, xs)
        return np.array([np.interp(x, xs, qc), np.interp(x, xs, qt)]), x

    def run(k, want_qpos=False):
        rig = Rig(base_h=base_h, **(rig_kwargs or {}))
        q_start, _ = ref_at(0.0, k)
        for _ in range(1500):
            rig.pd_step(q_start, SCAN_KP, SCAN_KV, clamp=True)
        n_delay = max(0, int(round(delay_ms)))
        buf = []
        therm = motor.ThermalTracker()
        sat = tot = 0
        T_cyc = 2.0 * k * T1
        t_end = cycles * T_cyc
        t_meas0 = 2.0 * T_cyc                          # discard 2 transient cycles
        log = dict(t=[], x=[], z=[], q=[], dq=[], tau=[], qpos=[])
        viol = 0
        xs_meas = []
        t = 0.0
        while t < t_end:
            q_now, dq_now = rig.q(), rig.dq()
            qr0, _ = ref_at(t, k)
            qr1, _ = ref_at(t + 1e-3, k)
            tau_cmd = kp * (qr0 - q_now) + kv * ((qr1 - qr0) / 1e-3 - dq_now)
            buf.append(tau_cmd)
            tau_apply = buf.pop(0) if len(buf) > n_delay else buf[0]
            tau = motor.clamp_torque(tau_apply, dq_now, peak=rig.peak, omega0=rig.w0)
            rig.step_tau(float(tau[0]), float(tau[1]))
            t += 1e-3
            x, z = rig.toe()
            if t >= t_meas0:
                sat += int(np.any(np.abs(tau) < np.abs(tau_apply) - 1e-6))
                tot += 1
                therm.add(tau)
                xs_meas.append((t, x))
                z_arc = np.interp(np.clip(sgn * x, xs[0], xs[-1]), xs, zs)
                out_of_x = (sgn * x < xs[0] - 0.03) or (sgn * x > xs[-1] + 0.03)
                viol += int(abs(z - z_arc) > band or out_of_x)
            log["t"].append(t)
            log["x"].append(x)
            log["z"].append(z)
            log["q"].append(rig.q())
            log["dq"].append(rig.dq())
            log["tau"].append(tau.copy())
            if want_qpos:
                log["qpos"].append(rig.data.qpos.copy())
        pts = np.array(xs_meas)
        ext = float(pts[:, 1].max() - pts[:, 1].min())
        v = float(np.sum(np.abs(np.diff(pts[:, 1]))) / (pts[-1, 0] - pts[0, 0]))
        feasible = (viol <= 0.02 * tot) and (ext >= 0.90 * L)
        return dict(k=k, feasible=feasible, v_mean=v, extent=ext, viol_frac=viol / max(tot, 1),
                    f_hz=1.0 / T_cyc, rms_tau=therm.rms, sat_frac=sat / max(tot, 1),
                    log={kk: np.array(vv) for kk, vv in log.items()})

    if k_force is not None:
        return run(k_force, want_qpos=record_qpos)
    # Ascending-k scan, first feasible wins (ascending k = descending speed). NOT a bisection:
    # feasibility is not monotone in k -- some fast cycles ride the linkage/ankle-spring dynamics
    # and track BETTER than slower ones, and a bisection converges to an arbitrary boundary
    # instead of the fastest feasible pocket (measured: it made 7 ms delay "beat" 0 ms).
    k_star = None
    for k in np.geomspace(1.1, k_max, n_k):
        if run(k)["feasible"]:
            k_star = k
            break
    if k_star is None:
        probe = run(k_max)
        print(f"[pass2-arc] delay={delay_ms}ms kp={kp:.0f}: NOT trackable even {k_max}x slower "
              f"than pass 1 (viol_frac {probe['viol_frac']:.3f}, "
              f"extent {probe['extent']:.3f}/{L:.3f} m)")
        return None
    for f in refines:                                      # micro-refine below the grid point
        k_try = f * k_star
        if k_try >= 1.0 and run(k_try)["feasible"]:
            k_star = k_try
    r = run(k_star, want_qpos=record_qpos)
    print(f"[pass2-arc] delay={delay_ms}ms kp={kp:.0f} band={band * 100:.0f}cm: k*={k_star:.2f} -> "
          f"v={r['v_mean']:.2f} m/s (extent {r['extent']:.3f} m at {r['f_hz']:.2f} Hz, "
          f"rms={r['rms_tau']:.1f} N*m, sat={r['sat_frac']:.2f}, viol={r['viol_frac']:.3f})")
    r["kp"], r["kv"], r["delay_ms"] = kp, kv, delay_ms
    return r


def pass2_best_gains(p1, base_h, delay_ms, record_qpos=False, k_max=40.0):
    """The bound must not be an artifact of one arbitrary PD stiffness: sweep controller gains
    and keep the fastest feasible result. The transport delay itself caps the usable stiffness
    (too-high kp with delay oscillates and blows the band) -- so each delay gets its own best
    gains, which is exactly how bandwidth really costs speed."""
    best = None
    for kp, kv in ((400.0, 6.0), (1000.0, 12.0), (2000.0, 20.0), (4000.0, 35.0)):
        r = pass2_tracked(p1, base_h, delay_ms=delay_ms, kp=kp, kv=kv, k_max=k_max)
        if r and (best is None or r["v_mean"] > best["v_mean"]):
            best = r
    if best and record_qpos:
        r = pass2_tracked(p1, base_h, delay_ms=delay_ms, kp=best["kp"], kv=best["kv"],
                          k_force=best["k"], record_qpos=True)
        r["kp"], r["kv"], r["delay_ms"] = best["kp"], best["kv"], delay_ms
        best = r
    return best


# --------------------------------------------------------------- pass 2 (unconstrained probe) ----
def pass2_run(end_lo, end_hi, base_h, kp=300.0, kv=3.0, sim_s=8.0, record_qpos=False):
    """Bang-bang between two poses with the real torque-speed envelope. Returns steady-state
    metrics measured over complete switch-to-switch cycles."""
    rig = Rig(base_h=base_h)
    m = rig.model
    j_cam = int(m.actuator_trnid[rig.a_cam, 0])
    j_thigh = int(m.actuator_trnid[rig.a_thigh, 0])
    rng = np.array([m.jnt_range[j_cam], m.jnt_range[j_thigh]])          # (2, 2)
    lim_margin = 0.03                                                    # rad from the hard stop
    for _ in range(1500):
        rig.pd_step(end_lo["q_t"], SCAN_KP, SCAN_KV, clamp=True)
    ends = (end_lo, end_hi)
    which = 1
    t_sw, switches = 0.0, []
    therm = motor.ThermalTracker()
    sat = tot = lim = 0
    peak_w = 0.0
    log = dict(t=[], x=[], z=[], qpos=[])
    t = 0.0
    while t < sim_s:
        tgt = ends[which]
        tau = rig.pd_step(tgt["q_t"], kp, kv, clamp=True)
        x, z = rig.toe()
        t += 1e-3
        t_sw += 1e-3
        qj, dqj = rig.q(), rig.dq()
        raw = kp * (tgt["q_t"] - qj) - kv * dqj
        sat += int(np.any(np.abs(tau) < np.abs(raw) - 1e-6))
        lim += int(np.any((qj < rng[:, 0] + lim_margin) | (qj > rng[:, 1] - lim_margin)))
        peak_w = max(peak_w, float(np.abs(dqj).max()))
        tot += 1
        therm.add(tau)
        log["t"].append(t)
        log["x"].append(x)
        log["z"].append(z)
        if record_qpos:
            log["qpos"].append(rig.data.qpos.copy())
        arrived = (x >= tgt["x"] - 0.02) if which == 1 else (x <= tgt["x"] + 0.02)
        if arrived or t_sw > 1.5:
            switches.append(t)
            which = 1 - which
            t_sw = 0.0
    log = {k: np.array(v) for k, v in log.items()}
    if len(switches) < 6:
        return dict(ok=False, v_mean=0.0, log=log, switches=switches)
    # steady window: from the 3rd switch to the last one with an EVEN number of half-cycles
    n_half = len(switches) - 3
    n_half -= n_half % 2
    if n_half < 2:
        return dict(ok=False, v_mean=0.0, log=log, switches=switches)
    w0, w1 = switches[2], switches[2 + n_half]
    m = (log["t"] >= w0) & (log["t"] <= w1)
    xw, tw = log["x"][m], log["t"][m]
    v_mean = float(np.sum(np.abs(np.diff(xw))) / (tw[-1] - tw[0]))
    ext = float(xw.max() - xw.min())
    period = 2.0 * (w1 - w0) / n_half
    return dict(ok=True, v_mean=v_mean, extent=ext, period=period,
                f_hz=1.0 / period, rms_tau=therm.rms, sat_frac=sat / max(tot, 1),
                limit_frac=lim / max(tot, 1), peak_w=peak_w,
                log=log, switches=switches)


def pass2(rec, base_h):
    half = len(rec["t"]) // 2
    q, toe = rec["q"][:half], rec["toe"][:half]
    x = toe[:, 0]
    sgn = np.sign(x[-1] - x[0])
    xs = sgn * x
    order = np.argsort(xs)
    xs_o, q_o = xs[order], q[order]
    L = xs_o[-1] - xs_o[0]

    def end_at(frac):
        xq = xs_o[0] + frac * L
        i = np.argmin(np.abs(xs_o - xq))
        return dict(q_t=q_o[i], x=float(sgn * xs_o[i]))

    margins = [0.0, 0.05, 0.10, 0.15, 0.22, 0.30]
    rows, best, best_clean = [], None, None
    print(f"[pass2] bang-bang margin grid ({len(margins)}x{len(margins)}), "
          f"envelope {PEAK:.1f} N*m -> 0 @ {NO_LOAD:.0f} rad/s ...")
    for ma in margins:
        for mb in margins:
            lo, hi = end_at(ma), end_at(1.0 - mb)
            if (hi["x"] - lo["x"]) * sgn < 0.05:
                continue
            r = pass2_run(lo, hi, base_h)
            row = dict(m_lo=ma, m_hi=mb, ok=r["ok"], v_mean=r.get("v_mean", 0.0),
                       extent=r.get("extent", 0.0), f_hz=r.get("f_hz", 0.0),
                       rms_tau=r.get("rms_tau", 0.0), sat_frac=r.get("sat_frac", 0.0),
                       limit_frac=r.get("limit_frac", 0.0), peak_w=r.get("peak_w", 0.0))
            rows.append(row)
            if r["ok"] and (best is None or r["v_mean"] > best[0]["v_mean"]):
                best = (row, lo, hi)
            # "clean" = never rides the joint-limit hard stops (elastic limit bounces are free
            # energy in sim; the real gearbox would be eating that impact instead)
            if r["ok"] and row["limit_frac"] < 0.01 and \
                    (best_clean is None or r["v_mean"] > best_clean[0]["v_mean"]):
                best_clean = (row, lo, hi)
    for row in sorted([r for r in rows if r["ok"]], key=lambda r: -r["v_mean"])[:8]:
        print(f"  m=({row['m_lo']:.2f},{row['m_hi']:.2f})  v={row['v_mean']:.2f} m/s  "
              f"extent={row['extent']:.3f} m  f={row['f_hz']:.2f} Hz  "
              f"rms={row['rms_tau']:.1f} N*m  sat={row['sat_frac']:.2f}  "
              f"limit={row['limit_frac']:.2f}  peak_w={row['peak_w']:.1f} rad/s")
    # the full-workspace point (margin 0/0) for the like-for-like comparison with pass 1
    full = next((r for r in rows if r["m_lo"] == 0.0 and r["m_hi"] == 0.0), None)
    return rows, best, best_clean, full


def pass2_tracked_video(r, base_h, out=RESULTS / "rail_pass2.mp4"):
    """Video of the arc-constrained torque-limited sweep at its fastest feasible speed."""
    import imageio.v2 as imageio
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 960)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 540)
    renderer = mujoco.Renderer(model, 540, 960)
    camv = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camv)
    camv.distance, camv.elevation, camv.azimuth = 1.9, -12, 90
    camv.lookat[:] = [0.05, 0.0, base_h * 0.55]
    qpos = r["log"]["qpos"]
    period_steps = max(1, int(round(1000.0 / r["f_hz"])))
    i0 = min(2 * period_steps, max(0, len(qpos) - period_steps))
    rt = qpos[i0:i0 + max(2000, 2 * period_steps):1000 // VIDEO_FPS]
    slow = qpos[i0:i0 + period_steps:max(1, 1000 // (VIDEO_FPS * 10))]
    writer = imageio.get_writer(str(out), fps=VIDEO_FPS)
    render_frames(model, renderer, camv, list(rt) + list(slow), writer)
    writer.close()
    print(f"[pass2-arc] wrote {out} (v={r['v_mean']:.2f} m/s at {r['f_hz']:.2f} Hz, "
          f"{len(rt)} real-time + {len(slow)} slow-mo frames)")


def pass2_video(best, base_h, out=RESULTS / "rail_pass2_unconstrained.mp4"):
    import imageio.v2 as imageio
    row, lo, hi = best
    r = pass2_run(lo, hi, base_h, sim_s=8.0, record_qpos=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 960)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 540)
    renderer = mujoco.Renderer(model, 540, 960)
    camv = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camv)
    camv.distance, camv.elevation, camv.azimuth = 1.9, -12, 90
    camv.lookat[:] = [0.05, 0.0, base_h * 0.55]
    qpos = r["log"]["qpos"]
    # skip the settle-in, show ~2 s real time from the 3rd switch, then one period at 10x slow-mo
    sw = r["switches"]
    i0 = int(sw[2] * 1000)
    rt = qpos[i0:i0 + 2000:1000 // VIDEO_FPS]
    period_steps = max(1, int(r["period"] * 1000))
    slow = qpos[i0:i0 + period_steps:max(1, 1000 // (VIDEO_FPS * 10))]
    writer = imageio.get_writer(str(out), fps=VIDEO_FPS)
    render_frames(model, renderer, camv, list(rt) + list(slow), writer)
    writer.close()
    print(f"[pass2] wrote {out} (best point m=({row['m_lo']:.2f},{row['m_hi']:.2f}), "
          f"v={r['v_mean']:.2f} m/s, f={r['f_hz']:.2f} Hz)")
    return r


# --------------------------------------------------------------------------------------- main ----
def prepare(rescan=False):
    """The shared front half of the pipeline: workspace scan (cached) -> arc -> slow verify ->
    pass-1 analytics with the empirical fold-boundary probe. Returns (rec, arc_p, p1, base_h).
    Used by main() and by the plotting/sensitivity scripts."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    targets, q_meas, toe = scan_workspace(rescan=rescan)
    arc_t, arc_p = build_arc(targets, q_meas, toe)
    rec = slow_verify(arc_t)
    base_h = float(-rec["toe"][:, 1].min() + 0.02)         # arc skims 2 cm above the floor
    print(f"[rig] rail height: {base_h:.3f} m")

    p1 = pass1_analytics(rec, arc_p=arc_p, tag="arc")
    # empirical fold boundary: quasi-static probe with the pass-2 controller; trim the path just
    # below the first out-of-band sample and re-probe until clean (<= 3 rounds)
    for _ in range(3):
        probe = pass2_tracked(p1, base_h, delay_ms=0, k_force=40.0)
        if probe["feasible"] or probe["viol_frac"] <= 0.02:
            break
        lg = probe["log"]
        xs_, zs_, sgn_ = p1["x_grid"], p1["z_of_x"], p1["sgn"]
        t_, x_, z_ = lg["t"], lg["x"], lg["z"]
        m_ = t_ >= 2.0 / probe["f_hz"]
        za_ = np.interp(np.clip(sgn_ * x_[m_], xs_[0], xs_[-1]), xs_, zs_)
        viol_x = sgn_ * x_[m_][np.abs(z_[m_] - za_) > 0.05]
        x_cap = float(viol_x.min() - 0.05)
        print(f"[fold-probe] quasi-static fold onset at x = {viol_x.min():+.3f}; "
              f"capping path at {x_cap:+.3f}")
        p1 = pass1_analytics(rec, arc_p=arc_p, tag="arc", x_cap=x_cap)
    return rec, arc_p, p1, base_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    rec, arc_p, p1, base_h = prepare(rescan=args.rescan)
    # THE pass-1 bound is the trimmed full-workspace traversal (the model as posed: back and
    # forth ACROSS the traversable workspace). Sub-segment numbers are kept in the json but are
    # dominated by measurement noise where PD droop makes dq/dx ~ 0 -- not a bound.
    p1_best = p1["v_full"]
    if not args.no_video:
        pass1_video(rec, p1, base_h)

    # THE pass-2 bound: arc-constrained tracked sweep (toe must stay within 5 cm of the measured
    # arc and cover >= 90% of it -- no off-workspace travel can count), with the real envelope and
    # the measured MIT-mode 7 ms command->response delay. delay=0 isolates the bandwidth cost;
    # delay=200 is what servo-mode SET_POS latency would leave of it.
    p2_ideal = pass2_best_gains(p1, base_h, delay_ms=0)
    p2_mit = pass2_best_gains(p1, base_h, delay_ms=7, record_qpos=not args.no_video)
    p2_servo = pass2_best_gains(p1, base_h, delay_ms=200, k_max=60.0)
    if p2_mit and not args.no_video:
        pass2_tracked_video(p2_mit, base_h)

    # unconstrained bang-bang kept as a diagnostic upper probe (its travel is allowed to leave
    # the workspace between the end poses -- see README; NOT the bound)
    rows, best, best_clean, full = pass2(rec, base_h)

    out = dict(
        L_workspace=p1["L_full"],
        pass1=dict(arc=dict(v_full=p1["v_full"], T_oneway=p1["T_oneway"],
                            v_best_segment=p1["v_best"], segment=p1["seg"]),
                   v_bound=p1_best,
                   caveat="v_best_segment values are measurement-noise artifacts (PD-droop flat "
                          "spots in q(x)); the bound is v_bound = full-workspace traversal"),
        pass2=dict(
            arc_constrained=dict(
                ideal={k: v for k, v in p2_ideal.items() if k != "log"} if p2_ideal else None,
                mit_7ms={k: v for k, v in p2_mit.items() if k != "log"} if p2_mit else None,
                servo_200ms={k: v for k, v in p2_servo.items() if k != "log"} if p2_servo else None),
            unconstrained_probe=dict(best=dict(best[0]) if best else None,
                                     best_clean=dict(best_clean[0]) if best_clean else None,
                                     full_workspace=full, grid=rows)),
        base_h=base_h,
        notes="v = max sustainable mean fore-aft toe speed rel. body; grounded-running bound "
              "(duty 0.5, no flight). Pass1: no-load speed only, unlimited torque. "
              "Pass2 arc_constrained (THE bound): AKE90-8 torque-speed envelope + real inertia, "
              "toe within 5 cm of the measured arc covering >=90% of it, transport delay on the "
              "torque command (0 / 7 ms MIT / 200 ms servo). unconstrained_probe's travel may "
              "leave the workspace between end poses -- diagnostic only.",
    )
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n=== SUMMARY ===")
    print(f"traversable workspace fore-aft extent L = {p1['L_full']:.3f} m "
          f"[{p1['x_grid'][0]:+.3f} .. {p1['x_grid'][-1]:+.3f}]")
    print(f"PASS 1 (velocity limit only, on-arc): bound {p1_best:.2f} m/s")
    if p2_ideal:
        print(f"PASS 2 (envelope, ON-ARC, no delay):   v = {p2_ideal['v_mean']:.2f} m/s "
              f"(extent {p2_ideal['extent']:.3f} m at {p2_ideal['f_hz']:.2f} Hz)")
    if p2_mit:
        print(f"PASS 2 (envelope, ON-ARC, 7 ms MIT):   v = {p2_mit['v_mean']:.2f} m/s "
              f"(extent {p2_mit['extent']:.3f} m at {p2_mit['f_hz']:.2f} Hz)  <-- THE BOUND")
        print(f"torque saturation lowers the bound: {p1_best:.2f} -> {p2_ideal['v_mean']:.2f} m/s "
              f"({(1 - p2_ideal['v_mean'] / p1_best) * 100:.0f}% down); "
              f"7 ms bandwidth: -> {p2_mit['v_mean']:.2f} m/s "
              f"({(1 - p2_mit['v_mean'] / p2_ideal['v_mean']) * 100:.0f}% more)")
    print(f"PASS 2 (envelope, ON-ARC, 200 ms servo): "
          + (f"v = {p2_servo['v_mean']:.2f} m/s at {p2_servo['f_hz']:.2f} Hz"
             if p2_servo else "not trackable at any tested speed"))
    if best:
        b = (best_clean or best)[0]
        print(f"[diagnostic] unconstrained bang-bang probe: {b['v_mean']:.2f} m/s "
              f"(travel may exit the workspace; not a bound)")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
