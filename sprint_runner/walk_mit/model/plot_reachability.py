"""Plot the sagittal (X-Z) reachability of the LEFT foot tip for DASH-01.

REDUCED 2-DOF STUDY. Only the two LEFT hip motors move:
  - cam   (HipLeftNCS-v1_Revolution-3) -> drives the knee through the parallel pushrod loop
          (knee is NOT a free input). The cam is a CRANK: the 4-bar assembles at EVERY cam
          angle (the MJCF's +-1.5 rad joint range is a CAD guess; recorded hardware sweeps
          span ~245 deg), so the default sweep covers the full circle [-90, 270] deg.
  - thigh (HipLeftNCS-v1_Revolution-5, range +-1.047) -> swings the thigh
Hip abduction (roll) is held at 0  -> leg stays vertical / in the sagittal plane.
The foot is held PARALLEL TO THE THIGH (the clean reading of "what the ankle spring does"),
so the only end-effector freedom comes from the two motors.

End-effector = center of the toe collision sphere `foot_L_col` (the "sphere at the end of
the foot"). Positions are reported RELATIVE TO THE HIP-ROLL pivot (the leg's attachment).

KEY FINDING -- the two motors are NOT independently free.
  cam, thigh, the passive pushrod and the passive knee form a planar CLOSED LOOP (a 4-bar):
  2 actuated + 2 passive joints, 2 loop constraints -> mobility 2. Loop closure means two
  circles (pushrod tip about its pivot; shin anchor about the knee) must intersect; they only
  do so over a DIAGONAL BAND of (cam, thigh). Outside the band the linkage cannot assemble --
  those motor combinations are mechanically impossible, not merely unused. The band boundary
  is the 4-bar DEAD-CENTER = the parallel-mechanism SINGULARITY (where det of the FK Jacobian
  d(x,z)/d(cam,thigh) also vanishes). That singular curve bounds the reachable foot area.

WHY THIS BEATS A BLIND SWEEP
  Because the leg is a 2-DOF mechanism, the reachable set is the IMAGE of the feasible band;
  its outline is the image of the band boundary (= the singularity) plus the few motor-limit
  edges -- no volumetric search is needed. We still rasterize the band cheaply to expose
  self-collision holes, flag where the ankle would exceed its range, and measure the area.

THE LOOP, SOLVED EXACTLY
  The `connect` equality ties site `pushrod_tip_L` to site `leg_anchor_L`. Every sagittal
  joint rotates about Y, so the sites never move in Y and loop closure is a planar 2-eq /
  2-unknown problem in (pushrod, knee) -- Newton-solved per (cam, thigh) on the validated
  model. We track the single PHYSICAL assembly branch (the one containing the rest pose).

Run:  .venv/Scripts/python.exe mujoco/dash01/plot_reachability.py
      (optional)  --nc 361 --nt 131 --ankle-mode parallel-thigh --cam-lo -86 --cam-hi 86 --show
"""
import argparse
import numpy as np
import mujoco

MODEL = "mujoco/dash01/dash01.xml"
FOOT_GEOM = "foot_L_col"                            # toe sphere = end effector
S_PUSH, S_ANCH = "pushrod_tip_L", "leg_anchor_L"    # the closed-loop connect sites
VIEW_X = -1.0                                        # mirror horizontally: view the leg from the other side


def _aid(m, n):  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
def _bid(m, n):  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
def _sid(m, n):  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
def _gid(m, n):  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)


def _wrap(a):
    """wrap angle(s) to (-pi, pi]"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _build_collidable_model():
    """Recompile the model with the visual leg meshes made collidable AT COMPILE TIME
    (a runtime contype flip does NOT build the mesh collision hulls), floor turned off."""
    spec = mujoco.MjSpec.from_file(MODEL)
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.contype = g.conaffinity = 1
        if g.name == "floor":
            g.contype = g.conaffinity = 0
    return spec.compile()


class Leg:
    """Forward kinematics of the reduced left-leg mechanism, backed by the MuJoCo model.

    cam_range: optional (lo, hi) RADIANS override of the model's cam joint range. The MJCF's
    +-1.5 rad cam range is a CAD guess; the real cam is a CRANK — the 4-bar assembles at EVERY
    cam angle (verified by sweeping loop closure over the full circle), and recorded hardware
    sweeps span ~245 deg of cam. Pass e.g. np.radians([-90, 270]) to study the true workspace."""

    def __init__(self, ankle_mode="parallel-thigh", cam_range=None):
        self.m = _build_collidable_model()
        self.d = mujoco.MjData(self.m)
        self.ankle_mode = ankle_mode
        m = self.m

        # --- joints (sidestep the non-ASCII joint names via actuators + body->joint) ---
        self.j_hip   = m.actuator_trnid[_aid(m, "hip_roll_L"), 0]
        self.j_cam   = m.actuator_trnid[_aid(m, "cam_L"),     0]
        self.j_thigh = m.actuator_trnid[_aid(m, "thigh_L"),   0]
        self.j_push  = m.body_jntadr[_bid(m, "PushrodLeftNCS-v1")]   # passive pushrod
        self.j_knee  = m.body_jntadr[_bid(m, "LegLeftNCS-v1")]       # passive knee
        self.j_ank   = m.body_jntadr[_bid(m, "FootLeftNCS-v1")]      # passive ankle
        self.q_hip   = m.jnt_qposadr[self.j_hip]
        self.q_cam   = m.jnt_qposadr[self.j_cam]
        self.q_thigh = m.jnt_qposadr[self.j_thigh]
        self.q_push  = m.jnt_qposadr[self.j_push]
        self.q_knee  = m.jnt_qposadr[self.j_knee]
        self.q_ank   = m.jnt_qposadr[self.j_ank]

        self.cam_range   = (np.asarray(cam_range, float) if cam_range is not None
                            else m.jnt_range[self.j_cam].copy())
        self.thigh_range = m.jnt_range[self.j_thigh].copy()
        self.ank_range   = m.jnt_range[self.j_ank].copy()

        self.s_push = _sid(m, S_PUSH)
        self.s_anch = _sid(m, S_ANCH)
        self.g_foot = _gid(m, FOOT_GEOM)
        self.ank_springref = float(m.qpos_spring[self.q_ank])   # spring-rest target

        # base at world origin, identity quat; right leg & everything else stay at qpos 0
        self.d.qpos[:] = 0.0
        self.d.qpos[3] = 1.0
        self.d.qpos[self.q_hip] = 0.0                            # leg vertical (no abduction)
        mujoco.mj_forward(self.m, self.d)

        # baseline = pairs touching BY DESIGN at the assembled rest pose (qpos 0): the
        # pushrod<->shin loop, hip<->pushrod, etc. Anything outside this set appearing later
        # is a genuine self-collision. (Adjacent links are auto-excluded by filterparent.)
        self.baseline = self._contact_pairs()

    def _contact_pairs(self):
        d = self.d
        return {tuple(sorted((d.contact[i].geom1, d.contact[i].geom2))) for i in range(d.ncon)}

    # ---- loop closure: solve passive (pushrod, knee) so the connect sites coincide ----
    def _loop_residual(self):
        mujoco.mj_kinematics(self.m, self.d)
        return (self.d.site_xpos[self.s_push] - self.d.site_xpos[self.s_anch])[[0, 2]]

    def solve_loop(self, cam, thigh, seed, eps=1e-7, tol=1e-9, iters=60):
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
            for col, qi in enumerate((self.q_push, self.q_knee)):
                save = d.qpos[qi]
                d.qpos[qi] = save + eps
                J[:, col] = (self._loop_residual() - r) / eps
                d.qpos[qi] = save
            try:
                step = np.linalg.solve(J, r)
            except np.linalg.LinAlgError:
                return np.array([p, k]), False           # loop Jacobian singular = lockup
            n = np.linalg.norm(step)
            if n > 0.5:
                step *= 0.5 / n                          # damp big steps near singularity
            p -= step[0]; k -= step[1]
        d.qpos[self.q_push], d.qpos[self.q_knee] = p, k
        return np.array([p, k]), (np.linalg.norm(self._loop_residual()) < 1e-6)

    # ---- set the ankle so the foot is parallel to the thigh (or at spring rest) ----
    def _set_ankle(self):
        d = self.d
        if self.ankle_mode == "spring-rest":
            d.qpos[self.q_ank] = self.ank_springref
            mujoco.mj_kinematics(self.m, self.d)
            return self.ank_springref
        # parallel-thigh: align foot axis (ankle->toe) with thigh axis (thigh pivot->knee).
        # Foot pitch is linear in the ankle angle (both rotate about Y) -> solve in one shot.
        v_thigh = (d.xanchor[self.j_knee] - d.xanchor[self.j_thigh])[[0, 2]]
        phi_thigh = np.arctan2(v_thigh[0], v_thigh[1])
        da = 1e-3
        d.qpos[self.q_ank] = 0.0
        mujoco.mj_kinematics(self.m, self.d)
        f0 = (d.geom_xpos[self.g_foot] - d.xanchor[self.j_ank])[[0, 2]]
        phi_f0 = np.arctan2(f0[0], f0[1])
        d.qpos[self.q_ank] = da
        mujoco.mj_kinematics(self.m, self.d)
        f1 = (d.geom_xpos[self.g_foot] - d.xanchor[self.j_ank])[[0, 2]]
        slope = _wrap(np.arctan2(f1[0], f1[1]) - phi_f0) / da
        a_star = _wrap(phi_thigh - phi_f0) / slope
        d.qpos[self.q_ank] = a_star
        mujoco.mj_kinematics(self.m, self.d)
        return a_star

    # ---- full forward map for one (cam, thigh) ----
    def fk(self, cam, thigh, seed):
        sol, ok = self.solve_loop(cam, thigh, seed)
        if not ok:
            return None, sol
        a_star = self._set_ankle()
        mujoco.mj_forward(self.m, self.d)            # contacts + final geom_xpos
        # tip relative to the LIVE hip pivot (robust to base pose, not a cached value)
        tip = (self.d.geom_xpos[self.g_foot] - self.d.xanchor[self.j_hip])[[0, 2]]
        collide = len(self._contact_pairs() - self.baseline) > 0
        ank_oob = not (self.ank_range[0] <= a_star <= self.ank_range[1])
        return dict(tip=tip, collide=collide, ank=a_star, ank_oob=ank_oob,
                    y=self.d.geom_xpos[self.g_foot][1]), sol

    # =====================================================================================
    #  NAIVE singularity search: force two links collinear, then solve the rest of the loop.
    #    cam-pushrod dead-center  : cam link  ||  pushrod link
    #    knee (leg extension)     : thigh link ||  shin link
    #  All link directions are read straight from the model (joint anchors + sites), in XZ.
    # =====================================================================================
    def _link_dirs(self):
        d = self.d
        return dict(
            cam=(d.xanchor[self.j_push] - d.xanchor[self.j_cam])[[0, 2]],     # cam pivot->pushrod jt
            push=(d.site_xpos[self.s_push] - d.xanchor[self.j_push])[[0, 2]],  # pushrod jt->tip
            thigh=(d.xanchor[self.j_knee] - d.xanchor[self.j_thigh])[[0, 2]],  # thigh pivot->knee
            shin=(d.xanchor[self.j_ank] - d.xanchor[self.j_knee])[[0, 2]])     # knee->ankle

    @staticmethod
    def _cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def close_loop_for(self, adr_a, adr_b, seed, eps=1e-7, iters=80):
        """Newton-solve qpos[adr_a], qpos[adr_b] so pushrod_tip == shin_anchor (X,Z).
        The OTHER joints must already be set. Returns ((a,b), ok)."""
        d = self.d
        a, b = seed
        for _ in range(iters):
            d.qpos[adr_a], d.qpos[adr_b] = a, b
            r = self._loop_residual()
            if np.linalg.norm(r) < 1e-11:
                return np.array([a, b]), True
            J = np.empty((2, 2))
            for col, adr in enumerate((adr_a, adr_b)):
                sv = d.qpos[adr]
                d.qpos[adr] = sv + eps
                J[:, col] = (self._loop_residual() - r) / eps
                d.qpos[adr] = sv
            try:
                step = np.linalg.solve(J, r)
            except np.linalg.LinAlgError:
                return np.array([a, b]), False
            n = np.linalg.norm(step)
            if n > 0.5:
                step *= 0.5 / n
            a -= step[0]; b -= step[1]
        d.qpos[adr_a], d.qpos[adr_b] = a, b
        return np.array([a, b]), (np.linalg.norm(self._loop_residual()) < 1e-6)

    def _roots_1d(self, f, lo, hi, n=1440, refine=44):
        """All sign-change roots of scalar f on [lo,hi] (bisection-refined)."""
        xs = np.linspace(lo, hi, n)
        fv = np.array([f(x) for x in xs])
        out = []
        for i in range(n - 1):
            u, v = fv[i], fv[i + 1]
            if not (np.isfinite(u) and np.isfinite(v)) or np.sign(u) == np.sign(v):
                continue
            a, b = xs[i], xs[i + 1]
            for _ in range(refine):
                m = 0.5 * (a + b)
                if np.sign(f(m)) == np.sign(u): a = m
                else: b = m
            out.append(0.5 * (a + b))
        return out

    def pushrod_align_roots(self, cam):
        """Pushrod-joint angles where the cam link and pushrod link are collinear, for this cam."""
        self.d.qpos[self.q_hip] = 0.0
        self.d.qpos[self.q_cam] = cam

        def f(p):
            self.d.qpos[self.q_push] = p
            mujoco.mj_kinematics(self.m, self.d)
            L = self._link_dirs()
            return self._cross(L["cam"], L["push"])
        return self._roots_1d(f, -np.pi, np.pi)

    def knee_straight_angle(self):
        """Knee-joint angle where the thigh and shin links are collinear AND extended (dot>0).
        Independent of the thigh angle (the thigh/shin relative angle is set by the knee)."""
        self.d.qpos[self.q_hip] = 0.0
        self.d.qpos[self.q_thigh] = 0.0

        def f(k):
            self.d.qpos[self.q_knee] = k
            mujoco.mj_kinematics(self.m, self.d)
            L = self._link_dirs()
            return self._cross(L["thigh"], L["shin"])
        best = None
        for k in self._roots_1d(f, -np.pi, np.pi):
            self.d.qpos[self.q_knee] = k
            mujoco.mj_kinematics(self.m, self.d)
            L = self._link_dirs()
            if np.dot(L["thigh"], L["shin"]) > 0:        # extended (straight leg), not folded
                best = k
        return best

    def tip_after_ankle(self):
        """With the loop already closed, set the foot || thigh and return (tip (X,Z) rel. hip,
        ankle_out_of_range). ank_oob=True means the foot CANNOT be held parallel within the
        +-60 deg ankle limit, i.e. this configuration is not physically reachable."""
        a_star = self._set_ankle()
        tip = (self.d.geom_xpos[self.g_foot] - self.d.xanchor[self.j_hip])[[0, 2]]
        oob = not (self.ank_range[0] <= a_star <= self.ank_range[1])
        return tip, oob


def sweep(leg, nc, nt):
    cam = np.linspace(*leg.cam_range, nc)
    thigh = np.linspace(*leg.thigh_range, nt)
    i0 = int(np.argmin(np.abs(cam)))             # column nearest cam = 0
    j0 = int(np.argmin(np.abs(thigh)))           # row nearest thigh = 0

    X = np.full((nc, nt), np.nan); Z = np.full((nc, nt), np.nan)
    COL = np.zeros((nc, nt), bool); OOB = np.zeros((nc, nt), bool)
    Y = np.full((nc, nt), np.nan)

    def store(i, j, res):
        if res is None:
            return
        X[i, j], Z[i, j] = res["tip"]
        COL[i, j], OOB[i, j], Y[i, j] = res["collide"], res["ank_oob"], res["y"]

    def fill_column(i, center_seed):
        """Solve the whole thigh column at cam[i], chaining the seed outward from thigh=0."""
        res, seed = leg.fk(cam[i], thigh[j0], center_seed)
        store(i, j0, res)
        base = seed if res is not None else center_seed
        s = base
        for j in range(j0 + 1, nt):
            res, sd = leg.fk(cam[i], thigh[j], s)
            store(i, j, res)
            if res is not None: s = sd
        s = base
        for j in range(j0 - 1, -1, -1):
            res, sd = leg.fk(cam[i], thigh[j], s)
            store(i, j, res)
            if res is not None: s = sd
        return base

    seed0 = np.array([0.0, 0.0])                 # loop is assembled at qpos 0
    s = fill_column(i0, seed0)
    for i in range(i0 + 1, nc):                  # cam ascending, chained center seeds
        s = fill_column(i, s)
    s = seed0
    for i in range(i0 - 1, -1, -1):              # cam descending
        s = fill_column(i, s)

    return dict(cam=cam, thigh=thigh, i0=i0, j0=j0, X=X, Z=Z, COL=COL, OOB=OOB, Y=Y)


def union_area(X, Z, mask, h=0.01, sub=3):
    """Area of the UNION of the reachable region (robust to the folding map): rasterize each
    valid grid quad into h x h pixels by bilinear subsampling and count distinct pixels."""
    occ = set()
    nc, nt = X.shape
    u = (np.arange(sub) + 0.5) / sub
    for i in range(nc - 1):
        for j in range(nt - 1):
            pts = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if not all(mask[p] and np.isfinite(X[p]) for p in pts):
                continue
            x00, x10, x11, x01 = (X[p] for p in pts)
            z00, z10, z11, z01 = (Z[p] for p in pts)
            for a in u:
                for b in u:
                    x = (1 - a) * ((1 - b) * x00 + b * x01) + a * ((1 - b) * x10 + b * x11)
                    z = (1 - a) * ((1 - b) * z00 + b * z01) + a * ((1 - b) * z10 + b * z11)
                    occ.add((int(np.floor(x / h)), int(np.floor(z / h))))
    return len(occ) * h * h


def linkage_points(leg):
    """XZ geometry of the current linkage pose (relative to the live hip pivot)."""
    d = leg.d
    hip = d.xanchor[leg.j_hip]
    xz = lambda p: (p - hip)[[0, 2]]
    return dict(
        cam=xz(d.xanchor[leg.j_cam]),   thigh=xz(d.xanchor[leg.j_thigh]),
        push=xz(d.xanchor[leg.j_push]), knee=xz(d.xanchor[leg.j_knee]),
        ank=xz(d.xanchor[leg.j_ank]),   ptip=xz(d.site_xpos[leg.s_push]),
        ee=xz(d.geom_xpos[leg.g_foot]))


def random_poses(leg, n=50, seed=0):
    """Solve n random FEASIBLE + collision-free poses (foot kept || thigh) and return each
    linkage's XZ geometry for drawing the mechanism."""
    rng = np.random.default_rng(seed)
    (lo_c, hi_c), (lo_t, hi_t) = leg.cam_range, leg.thigh_range
    poses, tries = [], 0
    while len(poses) < n and tries < n * 400:
        tries += 1
        cam, thigh = rng.uniform(lo_c, hi_c), rng.uniform(lo_t, hi_t)
        res, _ = leg.fk(cam, thigh, np.array([0.0, 0.0]))
        if res is None or res["collide"] or res["ank_oob"]:
            continue                                     # keep only the reachable (green) set
        poses.append(linkage_points(leg))
    if len(poses) < n:
        print(f"  (only {len(poses)}/{n} random poses were reachable)")
    return poses


def compute_overlays(leg, g, n_cam=361, n_thigh=361):
    """NAIVE singularity sweep. Instead of any Jacobian/boundary trick, force a pair of links
    collinear and solve the rest of the loop physics from there:

      (loop) cam-pushrod dead-center : for each cam, put the pushrod collinear with the cam
             (root of cross(cam-link, pushrod-link)), then solve (thigh, knee) that close the
             loop. Keep it if thigh is within its ctrl range.
      (knee) leg extension           : fix the knee at the thigh||shin (straight-leg) angle,
             then for each thigh solve (cam, pushrod) that close the loop. Keep if cam in range.

    Each accepted config is a genuine posed configuration, so its end effector (same FK, same
    hip reference, same mirror) sits exactly where the linkage puts it.
    """
    X, Z, COL, OOB = g["X"], g["Z"], g["COL"], g["OOB"]
    feas = np.isfinite(X)
    valid = feas & ~COL & ~OOB            # reachable: foot stays parallel & no self-collision
    (cam_lo, cam_hi), (th_lo, th_hi) = leg.cam_range, leg.thigh_range

    def reset():
        leg.d.qpos[:] = 0.0
        leg.d.qpos[3] = 1.0
        leg.d.qpos[leg.q_hip] = 0.0

    # ---- (loop) cam(l2) || pushrod(l3) : sweep cam, solve (thigh, knee) ----
    #   Keep only configs the leg can actually reach with the foot parallel within the +-60 deg
    #   ankle limit. (The thigh||shin "leg extension" singularity is omitted entirely -- it needs
    #   a 90 deg ankle to hold the foot parallel, so it is physically impossible to reach.)
    lx, lz, loop_cfg, loop_poses = [], [], [], []
    for c in np.linspace(cam_lo, cam_hi, n_cam):
        for p in leg.pushrod_align_roots(c):
            reset()
            leg.d.qpos[leg.q_cam] = c
            leg.d.qpos[leg.q_push] = p
            (th, kn), ok = leg.close_loop_for(leg.q_thigh, leg.q_knee, seed=(0.0, 0.0))
            if ok and th_lo <= th <= th_hi:
                (x, z), oob = leg.tip_after_ankle()
                if not oob:                          # ankle within +-60 deg -> actually reachable
                    lx.append(x); lz.append(z); loop_cfg.append((c, th))
                    # capture the EXACT singular pose now (the aligned pushrod branch); the frame
                    # must draw THIS, not re-solve via fk() which would pick a different branch.
                    loop_poses.append(linkage_points(leg))

    # drop singular configs whose foot is > 20 cm from the reachable region (far, folded branches)
    lx, lz = np.array(lx), np.array(lz)
    reach = np.column_stack([X[valid], Z[valid]])
    if len(lx):
        d2 = ((reach[:, 0][None, :] - lx[:, None]) ** 2 +
              (reach[:, 1][None, :] - lz[:, None]) ** 2).min(axis=1)
        keep = np.flatnonzero(d2 <= 0.20 ** 2)
    else:
        keep = np.array([], int)
    lx, lz = lx[keep], lz[keep]
    loop_cfg = [loop_cfg[i] for i in keep]
    loop_poses = [loop_poses[i] for i in keep]

    return dict(valid=valid, feas=feas, lx=lx, lz=lz,
                loop_cfg=loop_cfg, loop_poses=loop_poses)


def draw_map(ax, g, ov):
    """Draw the clean, MIRRORED reachability map: reachable region, singularity, refs."""
    vx = VIEW_X
    X, Z, v = g["X"], g["Z"], ov["valid"]
    ax.scatter(vx * X[v], Z[v], s=9, c="#2c9e3f", alpha=0.5, lw=0,
               label="reachable, tarsometatarsus || femur", zorder=4)
    ax.scatter(vx * ov["lx"], ov["lz"], s=8, c="darkorange", lw=0, zorder=6,
               label="singularity: l2 || l3 (aligned)")
    ax.plot(0, 0, "ks", ms=9, zorder=8, label="hip pivot (origin)")
    ax.set_xlabel("X  (forward, m)  — mirrored")
    ax.set_ylabel("Z  (up, m)")
    # display-only: flip the sign of the horizontal-axis tick labels to match the user's
    # convention, WITHOUT moving any plotted data.
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{(-v) + 0.0:g}"))
    ax.set_aspect("equal", "box")
    ax.grid(True, ls=":", alpha=0.4)


def draw_pose(ax, p):
    """Draw ONE full linkage pose (mirrored) over the map + mark its end effector."""
    vx = VIEW_X
    x = lambda n: vx * p[n][0]
    ax.plot([x("thigh"), x("knee"), x("ank"), x("ee")],                  # serial leg
            [p["thigh"][1], p["knee"][1], p["ank"][1], p["ee"][1]],
            color="#1f6fb0", lw=2.6, zorder=10, solid_capstyle="round",
            label="leg (femur–tibiotarsus–tarsometatarsus)")
    ax.plot([x("cam"), x("push"), x("ptip")],                            # parallel actuation
            [p["cam"][1], p["push"][1], p["ptip"][1]],
            color="#c0392b", lw=2.6, zorder=10, solid_capstyle="round",
            label="l2 + l3 loop")
    ax.plot([x("cam"), x("thigh")], [p["cam"][1], p["thigh"][1]],        # rigid hip block
            color="#555555", lw=3.2, zorder=9)
    joints = np.array([[vx * p[n][0], p[n][1]]
                       for n in ("cam", "thigh", "push", "knee", "ank", "ptip")])
    ax.scatter(joints[:, 0], joints[:, 1], s=30, c="white", edgecolor="black", lw=1.2,
               zorder=11)
    ax.scatter([x("ee")], [p["ee"][1]], s=95, c="black", zorder=12, edgecolor="white",
               lw=1.0, label="end effector")


def _fig():
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(8.5, 9.2))


def plot(g, ov, out, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = _fig()
    draw_map(ax, g, ov)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")
    if show:
        plt.show()
    plt.close(fig)


def render_pose_frames(g, ov, poses, outdir):
    """One image per pose: the complete reachability map + that pose's full linkage."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    for k, p in enumerate(poses):
        fig, ax = _fig()
        draw_map(ax, g, ov)
        draw_pose(ax, p)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"pose_{k:02d}.png"), dpi=130)
        plt.close(fig)
    print(f"saved {len(poses)} pose frames to {outdir}/")


def render_singularity_frames(leg, g, ov, outdir, per_family=40):
    """One image per SINGULAR configuration: the full linkage drawn AT each singularity, so its
    end effector (black dot) lands on the mirrored singularity curve. loop_* = l2||l3 (cam-pushrod)
    dead-center. (The femur||tibiotarsus 'leg extension' family is omitted -- unreachable.)"""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    total = 0
    for tag, cfgs, poses, desc in [
            ("loop", ov["loop_cfg"], ov["loop_poses"], "l2-l3 dead-center (aligned)")]:
        idx = list(range(len(cfgs)))
        if len(idx) > per_family:                        # even subsample across the curve
            idx = list(dict.fromkeys(np.linspace(0, len(idx) - 1, per_family).round().astype(int)))
        for k, i in enumerate(idx):
            (c, t), pose = cfgs[i], poses[i]
            fig, ax = _fig()
            draw_map(ax, g, ov)
            draw_pose(ax, pose)                          # the EXACT captured singular pose
            ax.text(0.02, 0.02, f"{desc}\nl2 = {c:+.3f}   femur = {t:+.3f} rad",
                    transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.85))
            ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{tag}_{k:02d}.png"), dpi=130)
            plt.close(fig)
            total += 1
    print(f"saved {total} singularity-pose frames to {outdir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc", type=int, default=241, help="cam samples")
    ap.add_argument("--nt", type=int, default=101, help="thigh samples")
    ap.add_argument("--cam-lo", type=float, default=-90.0, metavar="DEG",
                    help="cam sweep start in DEGREES. The real cam is a CRANK (the 4-bar "
                         "assembles at every angle; the model's ±86° joint range is a CAD "
                         "guess), so the default sweeps the full circle [-90, 270]")
    ap.add_argument("--cam-hi", type=float, default=270.0, metavar="DEG",
                    help="cam sweep end in DEGREES")
    ap.add_argument("--ankle-mode", choices=["parallel-thigh", "spring-rest"],
                    default="spring-rest",
                    help="spring-rest matches the real preloaded passive ankle (and the web "
                         "UI's FK LUT); parallel-thigh is the geometric idealization")
    ap.add_argument("--out", default="mujoco/dash01/_reachability.png",
                    help="clean reachability map (no linkage overlay)")
    ap.add_argument("--poses", type=int, default=50,
                    help="how many single-pose frames to render (0 = none)")
    ap.add_argument("--pose-seed", type=int, default=0)
    ap.add_argument("--frames-dir", default="mujoco/dash01/_reachability_poses",
                    help="folder for the per-pose frames")
    ap.add_argument("--sing-frames", type=int, default=40,
                    help="max singularity-pose frames per family (0 = none)")
    ap.add_argument("--sing-dir", default="mujoco/dash01/_reachability_singularities",
                    help="folder for the singularity-pose frames")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    leg = Leg(ankle_mode=args.ankle_mode, cam_range=np.radians([args.cam_lo, args.cam_hi]))
    print(f"cam range   = {leg.cam_range}   ({args.nc} samples)")
    print(f"thigh range = {leg.thigh_range} ({args.nt} samples)")
    print(f"ankle mode  = {args.ankle_mode}  (ankle phys range {leg.ank_range})")
    print("sweeping ...")
    g = sweep(leg, args.nc, args.nt)

    feas = np.isfinite(g["X"]); free = feas & ~g["COL"]
    valid = free & ~g["OOB"]; n = g["X"].size
    print("\n----- reachability summary (tip relative to hip pivot) -----")
    print(f"feasible poses (loop assembles) : {feas.sum():5d} / {n}"
          f"   ({100*feas.sum()/n:.0f}% — rest is outside the linkage's assembly band)")
    print(f"  self-colliding (excluded)     : {(feas & g['COL']).sum():5d}")
    print(f"  ankle past +/-{leg.ank_range[1]:.3f} (foot can't stay parallel): {(free & g['OOB']).sum():5d}")
    if valid.any():
        print(f"X (forward) span : [{g['X'][valid].min():+.3f}, {g['X'][valid].max():+.3f}] m")
        print(f"Z (up)      span : [{g['Z'][valid].min():+.3f}, {g['Z'][valid].max():+.3f}] m")
        reach = np.hypot(g["X"][valid], g["Z"][valid])
        print(f"reach from hip   : [{reach.min():.3f}, {reach.max():.3f}] m")
        print(f"Y spread (should be ~const): "
              f"{np.nanmax(g['Y'][feas]) - np.nanmin(g['Y'][feas]):.4f} m")
        print(f"area, foot || thigh & collision-free : {union_area(g['X'], g['Z'], valid):.4f} m^2")
        print(f"area, whole assembled band          : {union_area(g['X'], g['Z'], feas):.4f} m^2")

    ov = compute_overlays(leg, g)
    plot(g, ov, args.out, args.show)                 # the clean map
    if args.poses:
        poses = random_poses(leg, args.poses, args.pose_seed)
        render_pose_frames(g, ov, poses, args.frames_dir)   # one full-linkage frame per pose
    if args.sing_frames:
        render_singularity_frames(leg, g, ov, args.sing_dir, args.sing_frames)


if __name__ == "__main__":
    main()
