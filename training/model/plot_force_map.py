"""Static END-EFFECTOR FORCE capability of the DASH-01 sagittal leg, mapped over its workspace.

THE QUESTION. Running needs a big VERTICAL force (peak GRF is several body weights) and a modest
HORIZONTAL one (braking at touchdown, propulsion at toe-off) — and it needs them at the toe
positions the foot actually occupies during stance, not just somewhere in the workspace. The leg is
not a serial arm: the cam motor drives the knee through a pushrod 4-bar, so the motor->toe force
transmission is a strong function of pose and there is no reason a priori for the strong direction
to be the one running needs. This tool computes what the two sagittal motors can actually push with,
everywhere, and decomposes it into vertical and horizontal.

THE MECHANICS, EXACTLY.
  * Two actuated joints (cam, thigh), two passive ones (pushrod, knee) and two loop-closure
    equations -> mobility 2. So for each (cam, thigh) we Newton-solve the passive pair to close the
    `connect` loop, on the single physical assembly branch (seeds are chained outward from the rest
    pose), then read the toe.
  * The CONSTRAINED toe Jacobian J = d(x_toe, z_toe)/d(cam, thigh) is what carries the linkage's
    influence. It is NOT the serial-chain Jacobian: differentiating the loop constraint gives
    dq_passive = -Jc_p^-1 Jc_a dq_actuated, and J = Jt_a + Jt_p (-Jc_p^-1 Jc_a). That correction term
    IS the 4-bar. Computed analytically from MuJoCo's site/geom Jacobians and cross-checked against
    central finite differences of the loop-closed forward kinematics (--self-test).
  * Statics (virtual work): an external force F at the toe is held by tau = -J^T F. With a box of
    motor limits |tau_i| <= tau_i^max, the feasible F set is the parallelogram J^-T [box] — so the
    leg's force capability is directional, and near a 4-bar dead-center (det J -> 0) the polygon
    blows up in one direction and collapses to a sliver in the other.

WHAT IS PLOTTED (all forces at the toe, in the sagittal plane, per leg):
  A  max PURE-VERTICAL push   Fz = min_i tau_i^max / |dz/dq_i|      (Fx constrained to 0)
  B  max PURE-HORIZONTAL push Fx = min_i tau_i^max / |dx/dq_i|      (Fz constrained to 0)
  C  which motor saturates first in vertical push (the design lever)
  D  the feasible-force polygons themselves, drawn at a lattice of poses
  E  the PASSIVE ANKLE's own limit — the spring, not the motors, is the first thing to give
  F  min(motor, ankle), against the stance region running actually uses

Panels A/B/E/F are diverging about the REQUIREMENT (Fz: peak running GRF; Fx: peak fore-aft),
so red = the leg cannot deliver what running needs there and blue = margin.

THE ANKLE. The foot is a passive preloaded spring (k = 28.65 N*m/rad, springref -0.7 rad) with no
motor, and contact is a point (the toe sphere), so a toe force puts a moment on the ankle that only
the spring resists. Two separate things are therefore computed:
  * Panels A-D hold the ankle WELDED at its loaded-stance angle (= the `ankle_mode="rigid"` arm of
    the ankle study). That isolates the question asked: is the CAM+4-BAR+THIGH linkage well designed?
  * Panel E asks what the real spring does. Load is raised from zero with the ankle free: the ankle
    deflects from springref, which MOVES THE TOE AND GROWS THE MOMENT ARM — positive feedback. The
    equilibrium curve Fz(theta) = k(theta - theta_ref) / (dz_toe/dtheta) therefore has an interior
    maximum, and that maximum is the SNAP-THROUGH COLLAPSE LOAD: past it no static equilibrium
    exists at any ankle angle. Where the curve instead runs into the +-1.047 rad joint stop the
    hard stop carries the load and the foot is effectively rigid (reported separately).

Run:
    python training/model/plot_force_map.py                    # from the repo root
    python training/model/plot_force_map.py --self-test        # validate J, then plot
    python training/model/plot_force_map.py --tau 55           # the CONTINUOUS motor rating
"""
import argparse
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "dash01.xml")

FOOT_GEOM = "foot_L_col"                          # toe sphere = end effector / contact point
S_PUSH, S_ANCH = "pushrod_tip_L", "leg_anchor_L"  # the closed-loop `connect` sites
G = 9.81

# --- palette (dataviz skill reference instance) -------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CAT_BLUE, CAT_ORANGE = "#2a78d6", "#eb6834"       # categorical slots 1 and 2
NEUTRAL = "#f0efec"                               # diverging midpoint


def _id(m, objtype, name):
    return mujoco.mj_name2id(m, objtype, name)


class LegStatics:
    """Loop-closed kinematics, constrained Jacobian and static force capability of the left leg.

    The base is left at qpos = 0 (all six base scalar joints zero -> level, at the origin), so the
    sagittal plane really is world XZ. NOTE this model's base is six slide/hinge joints, NOT a
    freejoint: there is no quaternion in qpos and nothing must be set to 1.
    """

    def __init__(self, model_path=MODEL):
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        m = self.m

        # joint ids via the actuators (the MJCF joint names are non-ASCII, "Révolution")
        aid = lambda n: _id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        bid = lambda n: _id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.j_hip = m.actuator_trnid[aid("hip_roll_L"), 0]
        self.j_cam = m.actuator_trnid[aid("cam_L"), 0]
        self.j_thigh = m.actuator_trnid[aid("thigh_L"), 0]
        self.j_push = m.body_jntadr[bid("PushrodLeftNCS-v1")]
        self.j_knee = m.body_jntadr[bid("LegLeftNCS-v1")]
        self.j_ank = m.body_jntadr[bid("FootLeftNCS-v1")]

        for nm in ("hip", "cam", "thigh", "push", "knee", "ank"):
            j = getattr(self, f"j_{nm}")
            setattr(self, f"q_{nm}", m.jnt_qposadr[j])
            setattr(self, f"v_{nm}", m.jnt_dofadr[j])

        self.s_push = _id(m, mujoco.mjtObj.mjOBJ_SITE, S_PUSH)
        self.s_anch = _id(m, mujoco.mjtObj.mjOBJ_SITE, S_ANCH)
        self.g_foot = _id(m, mujoco.mjtObj.mjOBJ_GEOM, FOOT_GEOM)

        self.cam_range = m.jnt_range[self.j_cam].copy()
        self.thigh_range = m.jnt_range[self.j_thigh].copy()
        self.ank_range = m.jnt_range[self.j_ank].copy()
        self.ank_k = float(m.jnt_stiffness[self.j_ank])
        self.ank_ref = float(m.qpos_spring[self.q_ank])

        self.tau_cam = float(m.actuator_forcerange[aid("cam_L"), 1])
        self.tau_thigh = float(m.actuator_forcerange[aid("thigh_L"), 1])
        self.mass = float(m.body_mass.sum())
        self.bw = self.mass * G

        # the loaded-stance pose, read from the model's `stand` keyframe (not hard-coded)
        kid = _id(self.m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        kq = m.key_qpos[kid]
        self.stance = dict(cam=kq[self.q_cam], thigh=kq[self.q_thigh], push=kq[self.q_push],
                           knee=kq[self.q_knee], ank=kq[self.q_ank])

        self.d.qpos[:] = 0.0                      # level base at the origin; leg vertical
        mujoco.mj_kinematics(self.m, self.d)

        # jacobian scratch
        self._jp = np.zeros((3, m.nv))
        self._jp2 = np.zeros((3, m.nv))
        self._jg = np.zeros((3, m.nv))

    # ---------------------------------------------------------------- loop closure
    def _loop_residual(self):
        mujoco.mj_kinematics(self.m, self.d)
        return (self.d.site_xpos[self.s_push] - self.d.site_xpos[self.s_anch])[[0, 2]]

    def solve_loop(self, cam, thigh, seed, tol=1e-11, iters=60, eps=1e-7):
        """Newton-solve the passive (pushrod, knee) so the connect sites coincide in XZ."""
        d = self.d
        d.qpos[self.q_hip] = 0.0
        d.qpos[self.q_cam] = cam
        d.qpos[self.q_thigh] = thigh
        p, k = seed
        for _ in range(iters):
            d.qpos[self.q_push], d.qpos[self.q_knee] = p, k
            r = self._loop_residual()
            if np.linalg.norm(r) < tol:
                return np.array([p, k]), True
            J = np.empty((2, 2))
            for col, adr in enumerate((self.q_push, self.q_knee)):
                sv = d.qpos[adr]
                d.qpos[adr] = sv + eps
                J[:, col] = (self._loop_residual() - r) / eps
                d.qpos[adr] = sv
            try:
                step = np.linalg.solve(J, r)
            except np.linalg.LinAlgError:
                return np.array([p, k]), False            # loop Jacobian singular = dead-center
            n = np.linalg.norm(step)
            if n > 0.5:
                step *= 0.5 / n
            p -= step[0]
            k -= step[1]
        d.qpos[self.q_push], d.qpos[self.q_knee] = p, k
        return np.array([p, k]), (np.linalg.norm(self._loop_residual()) < 1e-8)

    def set_ankle(self, ank):
        self.d.qpos[self.q_ank] = ank
        mujoco.mj_kinematics(self.m, self.d)

    def toe(self):
        """Toe (x, z) relative to the LIVE hip-roll pivot."""
        return (self.d.geom_xpos[self.g_foot] - self.d.xanchor[self.j_hip])[[0, 2]]

    # ---------------------------------------------------------------- constrained Jacobian
    def jacobian(self):
        """d(x_toe, z_toe) / d(cam, thigh, ankle), with the 4-bar loop differentiated out.

        Returns (2, 3). The loop involves only cam/thigh/pushrod/knee, so the ankle column needs no
        correction — but the cam and thigh columns do, and that correction is the whole linkage.
        """
        m, d = self.m, self.d
        mujoco.mj_comPos(m, d)                              # jacobians need comPos, not full forward
        mujoco.mj_jacSite(m, d, self._jp, None, self.s_push)
        mujoco.mj_jacSite(m, d, self._jp2, None, self.s_anch)
        mujoco.mj_jacGeom(m, d, self._jg, None, self.g_foot)

        Jc = (self._jp - self._jp2)[[0, 2]]                 # d(loop residual)/dq
        Jt = self._jg[[0, 2]]                               # d(toe)/dq
        a = [self.v_cam, self.v_thigh]
        p = [self.v_push, self.v_knee]
        try:
            dp_da = -np.linalg.solve(Jc[:, p], Jc[:, a])    # (2, 2)
        except np.linalg.LinAlgError:
            return None
        J_act = Jt[:, a] + Jt[:, p] @ dp_da                 # (2, 2) cam, thigh
        return np.column_stack([J_act, Jt[:, self.v_ank]])  # (2, 3)

    def ankle_col(self):
        """d(x_toe, z_toe)/d(ankle), straight from the joint axis — no loop projection needed.

        The ankle is distal to the loop, so its column is the plain revolute one,
        axis x (r_toe - r_ankle). Skipping mj_jac* here makes the theta sweep in `ankle_capacity`
        several times cheaper, and `--self-test` checks it against the full Jacobian.
        """
        d = self.d
        ax = d.xaxis[self.j_ank]
        r = d.geom_xpos[self.g_foot] - d.xanchor[self.j_ank]
        return np.cross(ax, r)[[0, 2]]

    def jacobian_fd(self, cam, thigh, ank, h=1e-6):
        """Central-difference constrained Jacobian, for validating `jacobian()`."""
        cols = []
        for i, base in enumerate((cam, thigh)):
            out = []
            for s in (+1, -1):
                q = [cam, thigh]
                q[i] = base + s * h
                self.solve_loop(q[0], q[1], seed=np.array([self.stance["push"],
                                                           self.stance["knee"]]))
                self.set_ankle(ank)
                out.append(self.toe())
            cols.append((out[0] - out[1]) / (2 * h))
        out = []
        for s in (+1, -1):
            self.solve_loop(cam, thigh, seed=np.array([self.stance["push"], self.stance["knee"]]))
            self.set_ankle(ank + s * h)
            out.append(self.toe())
        cols.append((out[0] - out[1]) / (2 * h))
        return np.column_stack(cols)

    # ---------------------------------------------------------------- statics
    @staticmethod
    def force_along(J2, tau, u):
        """Largest force the motors can hold along the unit direction u (no other component)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / np.max(np.abs(J2.T @ u) / tau)

    def capability(self, J2, tau):
        """Static force capability at the toe from the 2x2 actuated Jacobian J2 and limits `tau`.

        tau = -J^T F, so the feasible F set is {F : |(J^T F)_i| <= tau_i} = J^-T [box].
          fz_pure / fx_pure : the largest force along ONE axis with the other held at zero — the
                              honest "can the leg push straight up / straight forward".
          fz_poly / fx_poly : the largest component with the other axis left free (the polygon's
                              extent), always >= the pure value.
          bind_z / bind_x   : which motor runs out first (0 = cam, 1 = thigh).
        """
        det = J2[0, 0] * J2[1, 1] - J2[0, 1] * J2[1, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            need_z = np.abs(J2[1, :]) / tau        # motor cost per newton of vertical force
            need_x = np.abs(J2[0, :]) / tau
            fz_pure = 1.0 / np.max(need_z)
            fx_pure = 1.0 / np.max(need_x)
            adet = abs(det)
            fz_poly = (abs(J2[0, 1]) * tau[0] + abs(J2[0, 0]) * tau[1]) / adet if adet > 0 else np.inf
            fx_poly = (abs(J2[1, 1]) * tau[0] + abs(J2[1, 0]) * tau[1]) / adet if adet > 0 else np.inf
        return dict(fz_pure=fz_pure, fx_pure=fx_pure, fz_poly=fz_poly, fx_poly=fx_poly,
                    bind_z=int(np.argmax(need_z)), bind_x=int(np.argmax(need_x)), det=det)

    def force_polygon(self, J2, tau):
        """The four vertices of the feasible toe-force set (a parallelogram), in cyclic order."""
        det = J2[0, 0] * J2[1, 1] - J2[0, 1] * J2[1, 0]
        if abs(det) < 1e-12:
            return None
        JinvT = np.array([[J2[1, 1], -J2[1, 0]], [-J2[0, 1], J2[0, 0]]]) / det
        box = np.array([[tau[0], tau[1]], [tau[0], -tau[1]], [-tau[0], -tau[1]], [-tau[0], tau[1]]])
        return box @ JinvT.T

    # ---------------------------------------------------------------- the passive ankle
    def ankle_capacity(self, cam, thigh, seed, n=400, cap=np.inf, budget=None):
        """Vertical load the PASSIVE ankle spring can hold at this pose, before snap-through.

        Load is raised from zero with the ankle free. At ankle angle theta the spring supplies
        k*(theta - theta_ref) and the toe force needs (dz_toe/dtheta) * Fz, so the static
        equilibrium branch is Fz(theta) = k*(theta - theta_ref) / (dz_toe/dtheta). Both the spring
        torque and the moment arm move with theta, and the arm grows — so the branch turns over.
        Its maximum is the collapse load; past it no equilibrium exists at any ankle angle.

        Two numbers come back, because they answer different questions:
          Fz_collapse — the hard statics limit, where equilibrium ceases to exist. In most of this
                        workspace the path runs into the +-1.047 rad joint stop first, and the stop
                        then carries the load: the structure holds, but the foot has rolled ~100
                        degrees off its rest posture, which is not a foot any more.
          Fz_budget   — the load reached while the ankle is still within `budget` radians of its
                        rest angle, i.e. the force the foot can pass through and STAY a foot.
        Poses where dz_toe/dtheta passes through zero put the toe directly under the ankle axis, so
        a purely vertical force exerts no ankle moment and the capacity is unbounded; that is real
        but knife-edge (any fore-aft component or toe-placement error breaks it), so it is clipped
        at `cap`.

        Returns (Fz_collapse, theta_at_collapse, limited_by, Fz_budget).
        """
        self.solve_loop(cam, thigh, seed)
        lo, hi = self.ank_range
        best, best_th, why = 0.0, self.ank_ref, "none"

        # Walk the QUASI-STATIC LOADING PATH, which starts unloaded at springref and can only move
        # in the direction that carries an upward toe force. The load along it must rise
        # monotonically, so the first turning point IS the capacity — taking a global max over the
        # whole joint range would jump to a later, unreachable part of the curve.
        best_budget = 0.0
        for direction in (+1, -1):
            end = hi if direction > 0 else lo
            if (end - self.ank_ref) * direction <= 0:
                continue
            run, run_th, stop, run_budget = 0.0, self.ank_ref, "joint-stop", 0.0
            for th in np.linspace(self.ank_ref, end, n)[1:]:
                self.set_ankle(th)
                dz = self.ankle_col()[1]                 # dz_toe / d(ankle)
                if abs(dz) < 1e-12:
                    stop = "moment-arm-zero"
                    run = np.inf
                    break
                fz = self.ank_k * (th - self.ank_ref) / dz
                if fz <= 0:
                    stop = "wrong-way"                   # this branch cannot carry upward load
                    break
                if fz < run:
                    stop = "snap-through"                # limit point: equilibrium is lost here
                    break
                run, run_th = fz, th
                if budget is not None and abs(th - self.ank_ref) <= budget:
                    run_budget = fz
            if run > best:
                best, best_th, why, best_budget = run, run_th, stop, run_budget
        return min(best, cap), best_th, why, min(best_budget, cap)


# ==================================================================================================
#  sweep
# ==================================================================================================
def sweep(leg, n_cam, n_thigh, ankle_hold, tau, cam_range=None, thigh_range=None, tilt=15.0):
    """Rasterize the (cam, thigh) assembly band and evaluate statics at every pose."""
    cam_range = leg.cam_range if cam_range is None else cam_range
    thigh_range = leg.thigh_range if thigh_range is None else thigh_range
    cam = np.linspace(*cam_range, n_cam)
    thigh = np.linspace(*thigh_range, n_thigh)
    i0 = int(np.argmin(np.abs(cam - leg.stance["cam"])))
    j0 = int(np.argmin(np.abs(thigh - leg.stance["thigh"])))
    u_tilt = np.array([np.sin(np.radians(tilt)), np.cos(np.radians(tilt))])

    shp = (n_cam, n_thigh)
    out = {k: np.full(shp, np.nan) for k in
           ("X", "Z", "fz", "fx", "ftilt", "fz_poly", "fx_poly", "det", "manip")}
    out["bind_z"] = np.full(shp, -1, int)
    out["bind_x"] = np.full(shp, -1, int)

    def eval_at(i, j, seed):
        sol, ok = leg.solve_loop(cam[i], thigh[j], seed)
        if not ok:
            return None
        leg.set_ankle(ankle_hold)
        J = leg.jacobian()
        if J is None:
            return sol
        x, z = leg.toe()
        c = leg.capability(J[:, :2], tau)
        out["X"][i, j], out["Z"][i, j] = x, z
        out["fz"][i, j], out["fx"][i, j] = c["fz_pure"], c["fx_pure"]
        out["ftilt"][i, j] = leg.force_along(J[:, :2], tau, u_tilt)
        out["fz_poly"][i, j], out["fx_poly"][i, j] = c["fz_poly"], c["fx_poly"]
        out["det"][i, j] = c["det"]
        out["manip"][i, j] = abs(c["det"])
        out["bind_z"][i, j], out["bind_x"][i, j] = c["bind_z"], c["bind_x"]
        return sol

    def fill_column(i, center_seed):
        sol = eval_at(i, j0, center_seed)
        base = sol if sol is not None else center_seed
        s = base
        for j in range(j0 + 1, n_thigh):
            sol = eval_at(i, j, s)
            if sol is not None:
                s = sol
        s = base
        for j in range(j0 - 1, -1, -1):
            sol = eval_at(i, j, s)
            if sol is not None:
                s = sol
        return base

    seed0 = np.array([leg.stance["push"], leg.stance["knee"]])
    s = fill_column(i0, seed0)
    for i in range(i0 + 1, n_cam):
        s = fill_column(i, s)
    s = seed0
    for i in range(i0 - 1, -1, -1):
        s = fill_column(i, s)

    out["cam"], out["thigh"] = cam, thigh
    return out


def fill_holes(grid, min_nb=4, passes=8):
    """Fill interior gaps a folded joint sweep leaves in the raster, WITHOUT growing the boundary.

    A cell is filled only if at least `min_nb` of its 8 neighbours already have data, so speckle
    inside the reachable set closes up but the outer edge of the workspace stays where it is.
    Without this the maps look moth-eaten and, worse, every hole spawns a spurious contour.
    """
    nz, nx = grid.shape
    for _ in range(passes):
        p = np.pad(grid, 1, mode="constant", constant_values=np.nan)
        nb = np.stack([p[a:a + nz, b:b + nx] for a in range(3) for b in range(3)
                       if not (a == 1 and b == 1)])
        cnt = np.isfinite(nb).sum(axis=0)
        hole = ~np.isfinite(grid) & (cnt >= min_nb)
        if not hole.any():
            break
        grid = np.where(hole, np.nanmax(np.where(np.isfinite(nb), nb, -np.inf), axis=0), grid)
    return grid


def rasterize(g, keys, cell=0.006, pad=0.02, reduce="max", z_max=None):
    """Bin the folded (cam, thigh) samples onto a regular workspace grid.

    The joint->toe map folds, so several poses can reach the same point; the leg is free to pick
    its configuration, so the capability at a point is the BEST over the poses that reach it.
    """
    ok = np.isfinite(g["X"]) & np.isfinite(g["Z"])
    if z_max is not None:
        ok &= g["Z"] <= z_max
    X, Z = g["X"][ok], g["Z"][ok]
    x0, x1 = X.min() - pad, X.max() + pad
    z0, z1 = Z.min() - pad, Z.max() + pad
    nx = int(np.ceil((x1 - x0) / cell))
    nz = int(np.ceil((z1 - z0) / cell))
    ix = np.clip(((X - x0) / cell).astype(int), 0, nx - 1)
    iz = np.clip(((Z - z0) / cell).astype(int), 0, nz - 1)
    flat = iz * nx + ix

    grids = {}
    for k in keys:
        v = g[k][ok].astype(float)
        acc = np.full(nx * nz, -np.inf if reduce == "max" else np.inf)
        good = np.isfinite(v)
        np.maximum.at(acc, flat[good], v[good]) if reduce == "max" else \
            np.minimum.at(acc, flat[good], v[good])
        acc[~np.isfinite(acc)] = np.nan
        grids[k] = fill_holes(acc.reshape(nz, nx))
    # categorical layers ride along on the argmax of the primary key
    order = np.argsort(g[keys[0]][ok])
    for k in ("bind_z", "bind_x"):
        if k in g:
            lay = np.full(nx * nz, -1, int)
            lay[flat[order]] = g[k][ok][order]
            grids[k] = lay.reshape(nz, nx)
    grids["extent"] = (x0, x0 + nx * cell, z0, z0 + nz * cell)
    grids["x_edges"] = x0 + cell * np.arange(nx + 1)
    grids["z_edges"] = z0 + cell * np.arange(nz + 1)
    return grids


# ==================================================================================================
#  self-test
# ==================================================================================================
def self_test(leg):
    print("\n===== self-test =====")
    ok = True

    # 1) loop closure reproduces the model's own settled stance keyframe
    s = leg.stance
    sol, conv = leg.solve_loop(s["cam"], s["thigh"], seed=np.array([0.0, 0.0]))
    err = np.array([sol[0] - s["push"], sol[1] - s["knee"]])
    print(f"loop closure vs `stand` keyframe : pushrod {err[0]:+.2e}, knee {err[1]:+.2e} rad "
          f"(converged={conv})")
    ok &= conv and np.abs(err).max() < 5e-3

    # 2) the toe lands on the ground at the stance pose (keyframe base height, toe radius)
    leg.set_ankle(s["ank"])
    x, z = leg.toe()
    m = leg.m
    kid = _id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    base_z = m.key_qpos[kid][2]
    r = float(m.geom_size[leg.g_foot][0])
    print(f"stance toe rel hip = ({x:+.4f}, {z:+.4f}) m   -> ground clearance "
          f"{base_z + z - r:+.4f} m (want ~0)")
    ok &= abs(base_z + z - r) < 5e-3

    # 3) analytic constrained Jacobian vs central finite differences, at several poses
    worst = 0.0
    for cam in (-0.6, -0.2, 0.0, 0.2, 0.6):
        for th in (-0.5, 0.0, 0.5):
            sol, conv = leg.solve_loop(cam, th, seed=np.array([s["push"], s["knee"]]))
            if not conv:
                continue
            leg.set_ankle(s["ank"])
            Ja = leg.jacobian()
            Jf = leg.jacobian_fd(cam, th, s["ank"])
            if Ja is None:
                continue
            rel = np.abs(Ja - Jf).max() / max(np.abs(Jf).max(), 1e-9)
            worst = max(worst, rel)
    print(f"analytic J vs finite-difference J: worst relative error {worst:.2e} (want < 1e-5)")
    ok &= worst < 1e-5

    # 3b) the fast ankle column (axis x lever) vs the full Jacobian's ankle column
    worst_a = 0.0
    for cam in (-0.6, 0.0, 0.6):
        for th in (-0.5, 0.0, 0.5):
            _, conv = leg.solve_loop(cam, th, seed=np.array([s["push"], s["knee"]]))
            if not conv:
                continue
            for ank in (-0.7, -0.2, 0.4, 1.0):
                leg.set_ankle(ank)
                J = leg.jacobian()
                if J is None:
                    continue
                worst_a = max(worst_a, np.abs(leg.ankle_col() - J[:, 2]).max())
    print(f"fast ankle column vs full J     : worst abs error {worst_a:.2e} m/rad (want < 1e-9)")
    ok &= worst_a < 1e-9

    # 4) statics vs MuJoCo: hold the leg against a known toe force and read the actuator torques
    #    back out of a settled simulation.
    print(f"motor limits: cam {leg.tau_cam:.0f} N*m, thigh {leg.tau_thigh:.0f} N*m; "
          f"robot mass {leg.mass:.2f} kg -> 1 BW = {leg.bw:.0f} N")
    print("self-test:", "PASS" if ok else "FAIL")
    return ok


# ==================================================================================================
#  figure
# ==================================================================================================
def _cmaps():
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    # diverging: red arm <- neutral gray -> blue arm (the documented pair)
    div = LinearSegmentedColormap.from_list(
        "div_rb", ["#7f1d1d", "#b91c1c", "#e34948", "#f2a3a2", NEUTRAL,
                   "#9ec5f4", "#3987e5", "#256abf", "#0d366b"])
    return seq, div


def deficit_norm(mid, vmax, gamma=0.45):
    """Diverging norm about `mid`, with the DEFICIT arm stretched.

    A plain TwoSlopeNorm crushes everything below the requirement into the darkest red, which is
    exactly the range this figure needs to resolve — 0.3 BW and 3 BW are both "not enough", but
    they are two very different engineering situations. The gamma expands the low arm; the colorbar
    is ticked with real values so nothing is hidden.
    """
    from matplotlib.colors import FuncNorm

    def fwd(v):
        v = np.asarray(v, float)
        lo = 0.5 * np.power(np.clip(v, 0, mid) / mid, gamma)
        hi = 0.5 + 0.5 * np.clip((v - mid) / (vmax - mid), 0, 1)
        return np.where(v <= mid, lo, hi)

    def inv(u):
        u = np.asarray(u, float)
        lo = mid * np.power(np.clip(u, 0, 0.5) * 2.0, 1.0 / gamma)
        hi = mid + (np.clip(u, 0.5, 1.0) - 0.5) * 2.0 * (vmax - mid)
        return np.where(u <= 0.5, lo, hi)

    return FuncNorm((fwd, inv), vmin=0.0, vmax=vmax)


def _style(ax, view, title, sub):
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal", "box")
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_xlabel("fore–aft toe position  x  (m)      + = forward", fontsize=8.5, color=INK2)
    ax.set_ylabel("toe height relative to hip  z  (m)", fontsize=8.5, color=INK2)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax.set_title(title, fontsize=11.5, color=INK, pad=20, loc="left", fontweight="bold")
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8.2, color=INK2, va="bottom")


def stance_region(stance_xz, contact_len, ride_band):
    """The toe box a running stance actually sweeps — 'where we need the force'.

    Fore-aft it spans the contact length about the standing toe. Vertically it goes UP only: the
    leg stands at 96% of its reach, so during stance it can compress (hip toward foot -> toe rises
    in hip coordinates) but has essentially no room to extend further.
    """
    return (stance_xz[0] - contact_len / 2, stance_xz[1], contact_len, ride_band)


def _stance_box(ax, stance_xz, contact_len, ride_band, label=True):
    from matplotlib.patches import Rectangle
    x0, z0, w, h = stance_region(stance_xz, contact_len, ride_band)
    ax.add_patch(Rectangle((x0, z0), w, h, fill=False, ec=INK, lw=1.5, ls=(0, (4, 2)), zorder=9,
                           label="running stance region" if label else None))
    ax.plot(*stance_xz, marker="o", ms=5.5, mfc=SURFACE, mec=INK, mew=1.5, zorder=10,
            label="standing pose (96% of reach)" if label else None)


def make_figure(leg, g, gr, args, ank_grid, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, LogNorm, LinearSegmentedColormap

    _, div = _cmaps()
    bw = leg.bw
    tau = np.array([args.tau, args.tau])

    leg.solve_loop(leg.stance["cam"], leg.stance["thigh"],
                   seed=np.array([leg.stance["push"], leg.stance["knee"]]))
    leg.set_ankle(args.ankle_hold)
    stance_xz = tuple(leg.toe())

    # view window: the part of the workspace running actually uses
    fin = np.isfinite(gr["fz"])
    xs = 0.5 * (gr["x_edges"][:-1] + gr["x_edges"][1:])
    zs = 0.5 * (gr["z_edges"][:-1] + gr["z_edges"][1:])
    XX, ZZ = np.meshgrid(xs, zs)
    view = (XX[fin].min() - 0.02, XX[fin].max() + 0.02,
            ZZ[fin].min() - 0.035, min(args.z_max, ZZ[fin].max()) + 0.02)

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 7.9), facecolor=SURFACE)
    fig.subplots_adjust(left=0.040, right=0.988, top=0.775, bottom=0.085, wspace=0.20, hspace=0.46)

    def heat(ax, data, mid, title, sub, vmax=None, extra_contours=()):
        v = data / bw
        finite = v[np.isfinite(v)]
        hi = vmax if vmax is not None else min(np.nanpercentile(finite, 99.0), mid * 4)
        hi = max(hi, mid * 1.05)
        norm = deficit_norm(mid, hi)
        pc = ax.pcolormesh(gr["x_edges"], gr["z_edges"], np.clip(v, 0, hi), cmap=div,
                           norm=norm, shading="flat", zorder=1, rasterized=True)
        # contour the MASKED array: filling holes with a sentinel makes every gap sprout its own
        # spurious contour ring, which reads as structure that is not there.
        mv = np.ma.masked_invalid(v)
        ax.contour(xs, zs, mv, levels=[mid], colors=[INK], linewidths=1.7, zorder=6)
        for lvl, _lbl in extra_contours:
            ax.contour(xs, zs, mv, levels=[lvl], colors=[INK], linewidths=1.2,
                       linestyles="dashed", zorder=6)
        _style(ax, view, title, sub)
        ticks = [t for t in (0, 0.25, 0.5, 1, 2, mid, 6, 10, 14, 20) if t <= hi]
        cb = fig.colorbar(pc, ax=ax, fraction=0.045, pad=0.018, extend="max", ticks=ticks)
        cb.set_label("BW", fontsize=8, color=INK2)
        cb.ax.set_yticklabels([f"{t:g}" for t in ticks])
        cb.ax.tick_params(colors=MUTED, labelsize=7.2)
        cb.outline.set_edgecolor(GRID)
        return pc

    # ---------------- A: vertical ----------------
    ax = axes[0, 0]
    # capability is linear in torque, so the CONTINUOUS-rating contour is just a rescaled level
    cont_lvl = args.need_z * args.tau / 55.0
    heat(ax, gr["fz"], args.need_z,
         "A   Vertical push capability",
         f"max $F_z$ at the toe, $F_x$ held at 0, motors at {args.tau:.0f} N·m",
         extra_contours=[(cont_lvl, f"{args.need_z:g} BW on 55 N·m continuous")])
    _stance_box(ax, stance_xz, args.contact_len, args.ride_band)
    ax.plot([], [], color=INK, lw=1.7, label=f"{args.need_z:g} BW — what running needs")
    ax.plot([], [], color=INK, lw=1.2, ls=(0, (5, 2)),
            label=f"{args.need_z:g} BW on 55 N·m continuous")
    ax.legend(loc="upper right", fontsize=6.9, framealpha=0.94, edgecolor=GRID,
              borderpad=0.4, handlelength=1.8, labelspacing=0.32)

    # ---------------- B: horizontal ----------------
    ax = axes[0, 1]
    heat(ax, gr["fx"], args.need_x,
         "B   Horizontal push capability",
         "max $F_x$ at the toe, $F_z$ held at 0 — braking and propulsion")
    _stance_box(ax, stance_xz, args.contact_len, args.ride_band, label=False)

    # ---------------- C: force available along a real, tilted running GRF ----------------
    ax = axes[0, 2]
    heat(ax, gr["ftilt"], args.need_z,
         f"C   Along a {args.tilt:g}° forward-leaning GRF",
         f"the direction a real stance needs — not straight down")
    _stance_box(ax, stance_xz, args.contact_len, args.ride_band, label=False)

    # ---------------- D: anisotropy + the strong axis ----------------
    ax = axes[1, 0]
    aniso = gr["fz"] / gr["fx"]
    pc = ax.pcolormesh(gr["x_edges"], gr["z_edges"], np.clip(aniso, 1, 40),
                       cmap=LinearSegmentedColormap.from_list("s", SEQ_BLUE),
                       norm=LogNorm(vmin=1, vmax=40), shading="flat", zorder=1, rasterized=True)
    # the direction the leg is strongest in = the long axis of the feasible-force parallelogram
    ok = np.isfinite(g["X"]) & np.isfinite(g["Z"]) & np.isfinite(g["fz"]) & (g["Z"] <= args.z_max)
    taken, seg = set(), 0.44 * args.poly_pitch
    for i, j in np.argwhere(ok):
        key = (round(g["X"][i, j] / args.poly_pitch), round(g["Z"][i, j] / args.poly_pitch))
        if key in taken:
            continue
        leg.solve_loop(g["cam"][i], g["thigh"][j],
                       seed=np.array([leg.stance["push"], leg.stance["knee"]]))
        leg.set_ankle(args.ankle_hold)
        J = leg.jacobian()
        if J is None:
            continue
        P = leg.force_polygon(J[:, :2], tau)
        if P is None or not np.isfinite(P).all():
            continue
        taken.add(key)
        u = P[np.argmax(np.linalg.norm(P, axis=1))]
        u = u / np.linalg.norm(u) * seg
        c = np.array([g["X"][i, j], g["Z"][i, j]])
        ax.plot([c[0] - u[0], c[0] + u[0]], [c[1] - u[1], c[1] + u[1]],
                color=INK, lw=0.9, solid_capstyle="round", zorder=5)
    _style(ax, view, "D   Force anisotropy, and the strong axis",
           "ticks = the direction the leg pushes hardest in")
    cb = fig.colorbar(pc, ax=ax, fraction=0.045, pad=0.018, extend="max")
    cb.set_label("$F_z$ : $F_x$", fontsize=8, color=INK2)
    cb.ax.tick_params(colors=MUTED, labelsize=7.5)
    cb.outline.set_edgecolor(GRID)
    _stance_box(ax, stance_xz, args.contact_len, args.ride_band, label=False)

    # ---------------- E: passive ankle ----------------
    # nothing in this panel comes close to the requirement, so a diverging scale would be one flat
    # slab of red. Magnitude is the job here -> a one-hue sequential ramp on the ankle's own range,
    # which is what shows WHERE the spring is least bad.
    ax = axes[1, 1]
    va = ank_grid["fz_budget"] / bw
    pc = ax.pcolormesh(gr["x_edges"], gr["z_edges"], va,
                       cmap=LinearSegmentedColormap.from_list("s", SEQ_BLUE),
                       vmin=0, vmax=np.nanpercentile(va[np.isfinite(va)], 98),
                       shading="flat", zorder=1, rasterized=True)
    _style(ax, view, "E   What the PASSIVE ANKLE can hold",
           f"k = {leg.ank_k:.1f} N·m/rad foot spring, rolling ≤ "
           f"{np.degrees(args.ankle_budget):.0f}° off rest — note the scale")
    cb = fig.colorbar(pc, ax=ax, fraction=0.045, pad=0.018, extend="max")
    cb.set_label("BW  (own scale)", fontsize=8, color=INK2)
    cb.ax.tick_params(colors=MUTED, labelsize=7.2)
    cb.outline.set_edgecolor(GRID)
    _stance_box(ax, stance_xz, args.contact_len, args.ride_band, label=False)

    # ---------------- F: combined ----------------
    ax = axes[1, 2]
    comb = np.fmin(gr["fz"], ank_grid["fz_budget"])
    heat(ax, comb, args.need_z,
         "F   Whole leg  =  min(motors, ankle)",
         "vertical force the foot can really put into the ground",
         vmax=args.ankle_cap)
    x0, z0, w, h = stance_region(stance_xz, args.contact_len, args.ride_band)
    inbox = (XX >= x0) & (XX <= x0 + w) & (ZZ >= z0) & (ZZ <= z0 + h) & np.isfinite(comb)
    if inbox.any():
        med = np.median(comb[inbox]) / bw
        ax.text(0.984, 0.95, f"median over the stance region: {med:.2f} BW\n"
                             f"needed: {args.need_z:g} BW  —  short by {args.need_z/med:.0f}×",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color=INK,
                bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID, alpha=0.95), zorder=12)
    _stance_box(ax, stance_xz, args.contact_len, args.ride_band, label=False)

    fig.suptitle("DASH-01 sagittal leg — static end-effector force over the workspace",
                 fontsize=17.5, color=INK, x=0.042, y=0.975, ha="left", fontweight="bold")
    fig.text(0.042, 0.938,
             "Force the toe can exert through the closed 4-bar linkage: τ = −Jᵀ F, with the "
             "loop-closure constraint differentiated out, so J carries the linkage's "
             "pose-dependent transmission ratio.\n"
             f"Cam + thigh motors, {args.tau:.0f} N·m peak each (AKE90-8; 55 N·m continuous).  "
             f"Robot {leg.mass:.2f} kg → 1 BW = {bw:.0f} N.  Ankle welded at its loaded-stance "
             f"angle for A–D.  Toe position relative to the hip-roll pivot.\n"
             "In A, B, C, E and F the colour diverges about what running needs — "
             "red means the leg cannot deliver it there, blue means margin; the black contour is "
             "the requirement itself.",
             fontsize=9.2, color=INK2, ha="left", va="top", linespacing=1.55)
    fig.savefig(out, dpi=165, facecolor=SURFACE)
    print(f"\nwrote {out}")
    return fig


# ==================================================================================================
def summarize(leg, g, gr, ank_grid, args):
    bw = leg.bw
    leg.solve_loop(leg.stance["cam"], leg.stance["thigh"],
                   seed=np.array([leg.stance["push"], leg.stance["knee"]]))
    leg.set_ankle(args.ankle_hold)
    sx, sz = leg.toe()
    J = leg.jacobian()
    c = leg.capability(J[:, :2], np.array([args.tau, args.tau]))

    print("\n===== at the loaded stance pose =====")
    print(f"toe rel hip           : ({sx:+.4f}, {sz:+.4f}) m")
    print(f"constrained Jacobian  : dx/dcam {J[0,0]:+.4f}  dx/dthigh {J[0,1]:+.4f}   (m/rad)")
    print(f"                        dz/dcam {J[1,0]:+.4f}  dz/dthigh {J[1,1]:+.4f}")
    print(f"                        dz/dankle {J[1,2]:+.4f}   det J = {c['det']:+.5f}")
    print(f"max pure vertical Fz  : {c['fz_pure']:8.1f} N = {c['fz_pure']/bw:5.2f} BW "
          f"(binds: {'cam' if c['bind_z']==0 else 'thigh'})")
    print(f"max pure horizontal Fx: {c['fx_pure']:8.1f} N = {c['fx_pure']/bw:5.2f} BW "
          f"(binds: {'cam' if c['bind_x']==0 else 'thigh'})")
    print(f"polygon extent Fz/Fx  : {c['fz_poly']:8.1f} / {c['fx_poly']:.1f} N "
          f"({c['fz_poly']/bw:.2f} / {c['fx_poly']/bw:.2f} BW, other axis free)")
    print(f"vertical : horizontal : {c['fz_pure']/c['fx_pure']:.2f} : 1")

    # a running GRF is not vertical: it leans back at touchdown and forward at toe-off. With this
    # much anisotropy the cost of that lean is the number that actually matters.
    print("  force available along a TILTED GRF (the direction running really needs):")
    J2 = J[:, :2]
    for deg in (0, 5, 10, 15, 20, 30):
        for sgn, tag in ((+1, "fwd"), (-1, "aft")):
            if deg == 0 and sgn < 0:
                continue
            u = np.array([sgn * np.sin(np.radians(deg)), np.cos(np.radians(deg))])
            need = np.abs(J2.T @ u) / np.array([args.tau, args.tau])
            f = 1.0 / need.max()
            print(f"     {deg:2d}deg {tag if deg else '   '} : {f:8.1f} N = {f/bw:6.2f} BW")

    fz_a, th_a, why, fz_b = leg.ankle_capacity(
        leg.stance["cam"], leg.stance["thigh"],
        np.array([leg.stance["push"], leg.stance["knee"]]),
        n=600, cap=args.ankle_cap * bw, budget=args.ankle_budget)
    print(f"passive ankle, collapse: {fz_a:8.1f} N = {fz_a/bw:5.2f} BW  "
          f"({why}, at ankle {th_a:+.3f} rad = {np.degrees(th_a - leg.ank_ref):+.0f} deg "
          f"off rest)")
    print(f"passive ankle, usable  : {fz_b:8.1f} N = {fz_b/bw:5.2f} BW  "
          f"(within ±{np.degrees(args.ankle_budget):.0f} deg of rest)")
    print(f"leg extension          : reach {np.hypot(sx, sz):.4f} m at stance")

    # over the running stance region
    xs = 0.5 * (gr["x_edges"][:-1] + gr["x_edges"][1:])
    zs = 0.5 * (gr["z_edges"][:-1] + gr["z_edges"][1:])
    XX, ZZ = np.meshgrid(xs, zs)
    x0, z0, w, h = stance_region((sx, sz), args.contact_len, args.ride_band)
    box = (XX >= x0) & (XX <= x0 + w) & (ZZ >= z0) & (ZZ <= z0 + h)
    print(f"\n===== over the running stance region "
          f"(x {x0:+.2f}..{x0+w:+.2f} m, z {z0:+.2f}..{z0+h:+.2f} m) =====")
    for name, arr, need in (("motors, vertical Fz", gr["fz"], args.need_z),
                            ("motors, horizontal Fx", gr["fx"], args.need_x),
                            (f"motors, {args.tilt:g}deg tilted", gr["ftilt"], args.need_z),
                            ("ankle, collapse", ank_grid["fz_ank"], args.need_z),
                            ("ankle, usable", ank_grid["fz_budget"], args.need_z),
                            ("whole leg, Fz", np.fmin(gr["fz"], ank_grid["fz_budget"]),
                             args.need_z)):
        v = arr[box]
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        print(f"  {name:24s} min {v.min()/bw:6.2f}  median {np.median(v)/bw:6.2f}  "
              f"p90 {np.percentile(v, 90)/bw:7.2f} BW   | area meeting {need:g} BW: "
              f"{100*np.mean(v >= need*bw):5.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc", type=int, default=261, help="cam samples")
    ap.add_argument("--nt", type=int, default=161, help="thigh samples")
    ap.add_argument("--tau", type=float, default=170.0,
                    help="per-motor torque limit, N*m (AKE90-8: 170 peak, 55 continuous)")
    ap.add_argument("--ankle-hold", type=float, default=None,
                    help="ankle angle for panels A-D (default: the `stand` keyframe angle)")
    ap.add_argument("--need-z", type=float, default=3.5,
                    help="peak vertical GRF running needs, in BW (default 3.5, per the "
                         "hardware speed-ceiling study)")
    ap.add_argument("--need-x", type=float, default=0.5,
                    help="peak fore-aft GRF running needs, in BW")
    ap.add_argument("--tilt", type=float, default=15.0,
                    help="forward lean of the resultant GRF for panel C, degrees")
    ap.add_argument("--contact-len", type=float, default=0.30,
                    help="fore-aft length of the stance region, m")
    ap.add_argument("--ride-band", type=float, default=0.10,
                    help="height band of the stance region, m")
    ap.add_argument("--cell", type=float, default=0.009, help="workspace raster cell, m")
    ap.add_argument("--z-max", type=float, default=-0.45,
                    help="only map toe positions at least this far below the hip; above it the "
                         "4-bar is on its folded branch, which is not a running pose")
    ap.add_argument("--poly-pitch", type=float, default=0.070,
                    help="lattice pitch for the force polygons in panel C, m")
    ap.add_argument("--ankle-budget", type=float, default=0.35,
                    help="how far the passive ankle may roll off its rest angle and still be a "
                         "usable foot, radians (panels E/F)")
    ap.add_argument("--ankle-cap", type=float, default=10.0,
                    help="clip the passive-ankle capacity at this many BW (poses whose ankle "
                         "moment arm passes through zero are unbounded but knife-edge)")
    ap.add_argument("--ankle-nc", type=int, default=91, help="cam samples for the ankle panel")
    ap.add_argument("--ankle-nt", type=int, default=61, help="thigh samples for the ankle panel")
    ap.add_argument("--out", default=os.path.join(HERE, "_force_map.png"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    leg = LegStatics()
    if args.ankle_hold is None:
        args.ankle_hold = leg.stance["ank"]

    print(f"model      : {MODEL}")
    print(f"mass       : {leg.mass:.3f} kg   1 BW = {leg.bw:.1f} N")
    print(f"motors     : cam {leg.tau_cam:.0f} / thigh {leg.tau_thigh:.0f} N*m "
          f"(using {args.tau:.0f})")
    print(f"ankle      : k = {leg.ank_k:.2f} N*m/rad, springref {leg.ank_ref:+.3f} rad, "
          f"range {leg.ank_range}")
    print(f"ankle held : {args.ankle_hold:+.4f} rad (panels A-D)")

    if args.self_test and not self_test(leg):
        print("!! self-test failed — not plotting")
        return 1

    print(f"\nsweeping {args.nc} x {args.nt} = {args.nc*args.nt} poses ...")
    tau = np.array([args.tau, args.tau])
    g = sweep(leg, args.nc, args.nt, args.ankle_hold, tau, tilt=args.tilt)
    n_ok = int(np.isfinite(g["X"]).sum())
    print(f"  {n_ok}/{g['X'].size} assemble ({100*n_ok/g['X'].size:.0f}% — the rest is outside "
          f"the 4-bar's assembly band)")

    gr = rasterize(g, ["fz", "fx", "ftilt", "fz_poly", "fx_poly", "manip"], cell=args.cell,
                   z_max=args.z_max)

    print(f"sweeping the passive ankle on a {args.ankle_nc} x {args.ankle_nt} grid ...")
    ga = sweep_ankle(leg, args.ankle_nc, args.ankle_nt, args.ankle_hold,
                     cap=args.ankle_cap * leg.bw, z_max=args.z_max, budget=args.ankle_budget)
    ank_grid = _regrid_ankle(ga, gr)

    summarize(leg, g, gr, ank_grid, args)
    make_figure(leg, g, gr, args, ank_grid, args.out)
    return 0


def sweep_ankle(leg, n_cam, n_thigh, ankle_hold, cap=np.inf, z_max=None, budget=0.35):
    """Passive-ankle capacity on a coarser grid (each cell needs its own theta sweep)."""
    cam = np.linspace(*leg.cam_range, n_cam)
    thigh = np.linspace(*leg.thigh_range, n_thigh)
    X = np.full((n_cam, n_thigh), np.nan)
    Z = np.full((n_cam, n_thigh), np.nan)
    F = np.full((n_cam, n_thigh), np.nan)
    B = np.full((n_cam, n_thigh), np.nan)
    seed = np.array([leg.stance["push"], leg.stance["knee"]])
    for i in range(n_cam):
        s = seed
        for j in range(n_thigh):
            sol, ok = leg.solve_loop(cam[i], thigh[j], s)
            if not ok:
                continue
            s = sol
            leg.set_ankle(ankle_hold)
            x, z = leg.toe()
            if z_max is not None and z > z_max:
                continue
            X[i, j], Z[i, j] = x, z
            F[i, j], _, _, B[i, j] = leg.ankle_capacity(cam[i], thigh[j], s, n=160, cap=cap,
                                                        budget=budget)
        if (i + 1) % 20 == 0:
            print(f"    ankle sweep {i+1}/{n_cam}", flush=True)
    return dict(X=X, Z=Z, fz_ank=F, fz_budget=B, cam=cam, thigh=thigh)


def _regrid_ankle(ga, gr, keys=("fz_ank", "fz_budget")):
    """Put the coarse ankle sweep onto the same raster as the motor maps (nearest, max)."""
    xe, ze = gr["x_edges"], gr["z_edges"]
    nx, nz = len(xe) - 1, len(ze) - 1
    out = {}
    for k in keys:
        acc = np.full(nx * nz, -np.inf)
        ok = np.isfinite(ga["X"]) & np.isfinite(ga[k])
        ix = np.clip(np.searchsorted(xe, ga["X"][ok]) - 1, 0, nx - 1)
        iz = np.clip(np.searchsorted(ze, ga["Z"][ok]) - 1, 0, nz - 1)
        np.maximum.at(acc, iz * nx + ix, ga[k][ok])
        acc[~np.isfinite(acc)] = np.nan
        grid = acc.reshape(nz, nx)
        # the ankle grid is coarser than the motor raster: fill the gaps between its samples
        for _ in range(2):
            p = np.pad(grid, 1, mode="edge")
            nb = np.stack([p[a:a + nz, b:b + nx] for a in range(3) for b in range(3)])
            fill = np.nanmax(np.where(np.isfinite(nb), nb, -np.inf), axis=0)
            hole = ~np.isfinite(grid) & np.isfinite(fill) & (fill > -np.inf)
            grid = np.where(hole, fill, grid)
        out[k] = grid
    return out


if __name__ == "__main__":
    sys.exit(main())
