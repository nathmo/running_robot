"""
DASH-01 sagittal leg -- reachable foot workspace and end-effector FORCE AUTHORITY.

    python plot_reachability.py                 # writes _reachability.png
    python plot_reachability.py --self-test     # verify the model against known values
    python plot_reachability.py --show          # open an interactive window

STANDALONE BY DESIGN. This script reads no model file, no meshes and does not import
MuJoCo. The linkage is written out numerically in section 1 as six 2-D vectors and two
frame offsets, each annotated with the line of `dash01.xml` it was transcribed from. Only
numpy and matplotlib are required. (The previous version of this file drove a MuJoCo
model through Newton iterations; it needed `dash01.xml` plus 13 STL meshes on a hard-coded
relative path that no longer exists in this repo, and it carried a silent bug -- it wrote
`qpos[3] = 1.0` intending a free-joint quaternion, but this model's base is six scalar
joints, so `qpos[3]` is `base_roll` and every published number was measured with the leg
rolled 57 degrees out of the sagittal plane. Neither failure mode can exist here: there is
no external state and no base to mis-set.)


WHAT IS BEING MODELLED
----------------------
One leg of DASH-01, in the sagittal (X-Z) plane, with hip abduction held at zero so the
leg cannot leave that plane. Two motors move it:

    cam    -- a CRANK on the hip. It does not drive a joint directly; it drives the knee
              through a pushrod, so the knee is an OUTPUT, not an input.
    thigh  -- swings the thigh (femur) about the hip.

cam, thigh, the passive pushrod and the passive knee form a planar CLOSED LOOP -- a
four-bar. Four joints, two of them actuated, minus two loop-closure equations, leaves
mobility 2. That is the single fact that makes this leg interesting and it drives
everything below:

  * The two motors are NOT independently free. Loop closure requires two circles (the
    pushrod tip about the pushrod joint; the shin's anchor stub about the knee) to
    intersect. They only do so over a diagonal BAND of (cam, thigh). Outside that band the
    linkage physically cannot be assembled -- those motor commands are impossible, not
    merely unused.
  * The band boundary is the four-bar's DEAD CENTRE: the two circles are tangent, the
    pushrod and the anchor stub are collinear, and the loop Jacobian drops rank. It is
    also exactly where det(d(toe)/d(cam,thigh)) vanishes, so it bounds the reachable foot
    area AND is where force authority blows up to infinity and velocity to zero.
  * Because the leg is a 2-DOF mechanism, the reachable set is the IMAGE of that band. No
    volumetric search is needed, and the outline is the image of the boundary curve.

The end effector is the centre of the toe collision sphere. Everything is reported
RELATIVE TO THE HIP PIVOT, so the results do not depend on how tall the robot stands.


THE MATH, AND WHY IT IS ALL CLOSED FORM
---------------------------------------
1. LOOP CLOSURE.  Given cam, the pushrod joint P is fixed; given thigh, the knee K is
   fixed. The pushrod tip must lie at distance L_PUSH from P and L_ANCHOR from K -- a
   circle-circle intersection, solved algebraically (`_circle_circle`). No Newton
   iteration, no seeding, no convergence failures, no branch drift. The half-chord `h` of
   that intersection is the distance to the dead centre: h = 0 IS the singularity.
   The two roots are the two assembly branches; we keep the one containing the rest pose.

2. THE SINGULARITY CURVE.  Rather than root-finding on a collinearity residual, note that
   assembly requires |K - P| to lie in [|L_PUSH - L_ANCHOR|, L_PUSH + L_ANCHOR]. K lies on
   a circle about the thigh pivot, so the thigh angles that put |K - P| exactly on either
   bound are ANOTHER circle-circle intersection. The boundary is therefore exact and costs
   two closed-form solves per cam angle.

3. THE JACOBIAN.  Analytic, via implicit differentiation of the loop constraint
   g(q_active, q_passive) = 0:

        dq_passive/dq_active = -(dg/dq_passive)^-1 (dg/dq_active)
        J = d(toe)/d(q_active) = partial_active(toe) + partial_passive(toe) * the above

   Every partial of a planar revolute chain is `perp(point - pivot)`, so each block is two
   lines. This carries the four-bar's transmission ratio, which is the whole point -- a
   naive open-chain Jacobian would be wrong by that ratio. det(dg/dq_passive) vanishes
   precisely at h = 0, tying (3) back to (1) and (2). `--self-test` checks the analytic
   J against central differences of the forward map.

4. FORCE AUTHORITY.  Statics: to hold a force F at the toe the motors must supply
   tau = J^T F. Inside the symmetric torque box |tau_i| <= tau_max_i, the largest force
   along a unit direction u is

        f_max(u) = min_i  tau_max_i / |(J^T u)_i|

   The script maps u = vertical and u = fore-aft. The full achievable force set is the
   image of the torque box under J^-T -- a parallelogram, drawn in panel (d).

   Note (J^T u)_i is just the moment arm of that motor about the direction u, so the joint
   with the LARGER arm is the one that binds.


CONVENTIONS
-----------
X is forward, Z is up, origin at the hip pivot; the plot is the robot's left leg seen
from the left, so X runs to the right of the page. Angles in this file are ordinary CCW
angles in the (x, z) plane. MuJoCo's sagittal joints rotate about +-Y, and a +Y rotation
is CLOCKWISE in (x, z), so each joint carries an explicit sign S_* below that converts it.
Getting that sign wrong mirrors the workspace, which is why they are named and asserted
rather than folded into the constants.


MODELLING ASSUMPTIONS (all are choices, all are visible)
--------------------------------------------------------
* Hip abduction is exactly 0. Out-of-plane (true lateral, +-Y) force comes only from the
  hip roll motor and is OUTSIDE this 2-D study. Where this script says "lateral" it means
  fore-aft (X) -- the in-plane horizontal direction that propulsion and braking use.
* The ankle is PASSIVE (a spring in the MJCF, a welded strut on the current hardware), so
  it is not an input. It is held at a fixed angle -- see `--ankle`. The default is the
  loaded-stance angle from the model's own `stand` keyframe, because that is the geometry
  the leg actually has when it is carrying the robot.
* Statics only. No inertia, no motor speed limit, no thermal limit. This BOUNDS
  capability; it says nothing about control, and a force this map calls available may be
  unreachable at speed (torque falls off with rotor speed) or unholdable for long.

LIMITATIONS vs the MuJoCo-backed predecessor (in git history, and in the training/ and
walk_mit/ copies of this filename)
* No SELF-COLLISION check. That needed the mesh hulls. Cells where the pushrod passes
  through the thigh in projection are therefore still drawn as reachable. In practice the
  links sit in slightly different Y planes, so a projected crossing is not automatically a
  collision -- but this script cannot tell you which is which.
* No contact, no gravity, no floor.
* The passive ankle's own strength is NOT modelled here. Under the shipped k=28.65 N.m/rad
  spring, prior statics found the ankle -- not the motors -- is the binding constraint for
  vertical force by roughly 8x. So panel (b) is an upper bound on the leg as a whole
  unless the ankle is a rigid strut. `--ankle-deflect 20` reports that ceiling.
"""

import argparse
import numpy as np

# =============================================================================
# 1.  THE LINKAGE, DEFINED IN 2-D
# =============================================================================
# Transcribed from dash01.xml, LEFT leg, projected onto (X, Z). Y components are dropped:
# every sagittal joint rotates about Y, so no Y coordinate ever changes and the loop
# constraint is a purely planar 2-equation problem. The comment on each line is the MJCF
# element it came from, so this block can be re-derived or re-checked against the CAD.
#
# Frame chain:  hip pivot -> {cam pivot, thigh pivot} -> ... -> toe

HIP_TO_CAM     = np.array([-0.149,     0.0      ])  # <body CamLeftNCS-v1   pos="-0.149 .0519 0">
HIP_TO_THIGH   = np.array([ 0.101,     0.0      ])  # <body ThighLeftNCS-v1 pos="0.101 .0461 0">
CAM_TO_PUSHJ   = np.array([ 0.120,     0.0      ])  # <body PushrodLeftNCS-v1 pos="0.12 .002 0">
PUSHJ_TO_TIP   = np.array([ 0.000743214, -0.397969])  # <site pushrod_tip_L>, in pushrod frame
THIGH_TO_KNEE  = np.array([ 0.0,      -0.350    ])  # <body LegLeftNCS-v1   pos="0 .015 -.35">
KNEE_TO_ANCHOR = np.array([-0.0660502,  0.026337])  # <site leg_anchor_L>,  in shin frame
KNEE_TO_ANKLE  = np.array([-0.500,     0.0      ])  # <body FootLeftNCS-v1  pos="-.5 -.0016 0">
ANKLE_TO_TOE   = np.array([ 0.0,      -0.3115   ])  # <geom foot_L_col      pos="0 0 -.3115">
TOE_RADIUS     = 0.025                              # <geom foot_L_col      size="0.025">

# Fixed frame twists baked into the CAD (MJCF `euler="0 phi 0"`). A +Y euler of phi is a
# CCW rotation of -phi in our (x, z) convention.
PUSH_FRAME = -(-0.18047963198315084)   # pushrod body   euler="0 -0.180480 0"
SHIN_FRAME = -(-1.00038993994430810)   # shin (Leg) body euler="0 -1.000390 0"

# Joint sign: MuJoCo axis "0 1 0" (+Y) is clockwise in (x, z) -> -1;  "0 -1 0" -> +1.
S_CAM   = -1.0   # HipLeftNCS-v1_Revolution-3   axis "0 1 0"
S_THIGH = -1.0   # HipLeftNCS-v1_Revolution-5   axis "0 1 0"
S_KNEE  = +1.0   # ThighLeftNCS-v1_Revolution-7 axis "0 -1 0"   (passive)
S_ANKLE = +1.0   # LegLeftNCS-v1_Revolution-9   axis "0 -1 0"   (passive)
S_PUSH  = +1.0   # CamLeftNCS-v1_Revolution-11  axis "0 -1 0"   (passive; used only by
                 # the self-test -- the forward map solves the tip position geometrically
                 # and never needs the pushrod's own angle)

# Joint travel, from the MJCF `range=` attributes.
CAM_LIMIT   = 1.5      # rad -- but see below: this is a CAD guess, the cam is a crank
THIGH_LIMIT = 1.047    # rad
ANKLE_LIMIT = 1.047    # rad

# Actuator torque box, from <position ... forcerange="-144.5 144.5">. Both hip motors are
# the same AK-series unit, so the box is square.
TAU_CAM   = 144.5      # N.m
TAU_THIGH = 144.5      # N.m

# Passive ankle, from <joint LegLeftNCS-v1_Revolution-9 stiffness="28.65" springref="-0.7">
ANKLE_STIFFNESS = 28.65   # N.m/rad
ANKLE_SPRINGREF = -0.7    # rad, unloaded rest

# The `stand` keyframe. Left-leg qpos slice is (hip_roll, cam, pushrod, thigh, knee, ankle)
# in tree order; ctrl commands thigh to 0.12 and the cam sags to -0.0335 under load.
STAND_CAM, STAND_THIGH = -0.033502, 0.117592
STAND_KNEE, STAND_ANKLE = 0.175174, -0.25218
STAND_PUSH = -0.13113     # the passive pushrod angle in that same keyframe
STAND_BASE_Z = 1.02348    # hip pivot height in that keyframe

# Total model mass: the sum of every <inertial mass=...> in dash01.xml (both legs, torso
# and the four lumped motor bodies). Used only to express force in body weights.
MASS_KG = 12.8275
GRAVITY = 9.81

# Peak vertical ground reaction force a running gait needs, in body weights. Drawn as a
# requirement contour on the force panels.
GRF_TARGET_BW = 3.5

# ---- derived scalars: lengths and in-frame direction angles ------------------
_len = lambda v: float(np.hypot(v[0], v[1]))
_dir = lambda v: float(np.arctan2(v[1], v[0]))

L_CRANK,  A_CRANK  = _len(CAM_TO_PUSHJ),   _dir(CAM_TO_PUSHJ)     # 0.1200 m
L_PUSH,   A_PUSH   = _len(PUSHJ_TO_TIP),   _dir(PUSHJ_TO_TIP)     # 0.3980 m
L_THIGH,  A_THIGH  = _len(THIGH_TO_KNEE),  _dir(THIGH_TO_KNEE)    # 0.3500 m
L_ANCHOR, A_ANCHOR = _len(KNEE_TO_ANCHOR), _dir(KNEE_TO_ANCHOR)   # 0.0711 m
L_SHIN,   A_SHIN   = _len(KNEE_TO_ANKLE),  _dir(KNEE_TO_ANKLE)    # 0.5000 m
L_FOOT,   A_FOOT   = _len(ANKLE_TO_TOE),   _dir(ANKLE_TO_TOE)     # 0.3115 m

# Which of the two circle-circle roots is the physical assembly. Fixed at import by
# `_pick_branch()` below: the branch is a global property of the mechanism (you can only
# swap branches by passing through a dead centre), so one sign is valid everywhere.
BRANCH = -1.0


# =============================================================================
# 2.  PLANAR VECTOR HELPERS
# =============================================================================
# All operate on trailing-axis-2 arrays, so every routine below is vectorised over an
# entire (cam, thigh) grid at once.

def _e(a):
    """Unit vector at CCW angle `a`. Shape (..., 2)."""
    a = np.asarray(a, float)
    return np.stack([np.cos(a), np.sin(a)], axis=-1)


def _perp(v):
    """Rotate by +90 deg: (x, z) -> (-z, x).  Note d/dtheta of _e(theta) is _perp(_e(theta)),
    which is why every revolute partial derivative in section 4 is a `_perp`."""
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _norm(v):
    return np.hypot(v[..., 0], v[..., 1])


def _angle(v):
    return np.arctan2(v[..., 1], v[..., 0])


def _wrap(a):
    """Wrap to (-pi, pi]."""
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def _circle_circle(c1, r1, c2, r2, branch):
    """Intersect circle(c1, r1) with circle(c2, r2) in the plane.

    Returns (point, ok, half_chord). `ok` is False where the circles miss (the mechanism
    cannot be assembled). `half_chord` is 0 exactly at tangency -- for the leg's loop that
    is the four-bar DEAD CENTRE, so it doubles as a signed-free distance-to-singularity.
    `branch` (+1/-1) selects which of the two roots to return.
    """
    delta = c2 - c1
    d = _norm(delta)
    ok = (d <= r1 + r2) & (d >= abs(r1 - r2)) & (d > 1e-12)
    safe_d = np.where(ok, d, 1.0)                      # keep the arithmetic finite
    a = (r1 * r1 - r2 * r2 + safe_d * safe_d) / (2.0 * safe_d)
    half_chord = np.sqrt(np.maximum(r1 * r1 - a * a, 0.0))
    u = delta / safe_d[..., None]
    pt = c1 + a[..., None] * u + branch * half_chord[..., None] * _perp(u)
    return pt, ok, np.where(ok, half_chord, np.nan)


# =============================================================================
# 3.  FORWARD KINEMATICS (vectorised, exact)
# =============================================================================

ANKLE_MODES = ("stance", "spring-rest", "parallel-thigh")


def ankle_fixed_angle(mode):
    """The held ankle angle for the two fixed modes."""
    return {"stance": STAND_ANKLE, "spring-rest": ANKLE_SPRINGREF}[mode]


def forward(q_cam, q_thigh, ankle_mode="stance", ankle_angle=None, branch=None):
    """Full pose of the leg for arbitrary-shaped arrays of (cam, thigh) joint angles.

    Returns a dict of (..., 2) points -- `cam_pivot`, `pushrod_joint`, `pushrod_tip`,
    `thigh_pivot`, `knee`, `ankle`, `toe` -- plus:
        ok        (...)  bool, the loop assembles here
        h         (...)  half-chord of the loop closure; 0 at the dead centre
        R_shin    (...)  absolute CCW rotation of the shin body
        q_knee    (...)  the passive knee angle, in MuJoCo's sign convention
        q_ankle   (...)  the held (or solved) ankle angle
        ankle_ok  (...)  the ankle angle is inside its +-1.047 rad travel

    Ankle modes:
        "stance"         hold the loaded-stance angle from the `stand` keyframe (default)
        "spring-rest"    hold the unloaded spring rest angle (-0.7 rad)
        "parallel-thigh" solve the ankle so the foot stays parallel to the thigh. A clean
                         geometric idealisation, but it demands ankle angles that the real
                         joint often cannot reach -- watch `ankle_ok`.
    """
    branch = BRANCH if branch is None else branch
    q_cam, q_thigh = np.broadcast_arrays(np.asarray(q_cam, float),
                                         np.asarray(q_thigh, float))

    # Crank: cam pivot -> pushrod joint. The cam body's own rotation is S_CAM * q_cam.
    P = HIP_TO_CAM + L_CRANK * _e(S_CAM * q_cam + A_CRANK)
    # Femur: thigh pivot -> knee.
    K = HIP_TO_THIGH + L_THIGH * _e(S_THIGH * q_thigh + A_THIGH)

    # Loop closure. The pushrod tip is L_PUSH from the pushrod joint and, because it is
    # pinned to the shin's anchor stub, L_ANCHOR from the knee.
    T, ok, h = _circle_circle(P, L_PUSH, K, L_ANCHOR, branch)

    # The tip's bearing from the knee fixes the shin's absolute orientation -- that is how
    # the cam reaches through the loop and sets the knee.
    R_shin = _angle(T - K) - A_ANCHOR
    ankle = K + L_SHIN * _e(R_shin + A_SHIN)

    if ankle_mode == "parallel-thigh":
        R_foot = S_THIGH * q_thigh              # foot parallel to femur
        q_ankle = _wrap((R_foot - R_shin) / S_ANKLE)
    else:
        held = ankle_fixed_angle(ankle_mode) if ankle_angle is None else float(ankle_angle)
        q_ankle = np.full(R_shin.shape, held)
        R_foot = R_shin + S_ANKLE * q_ankle

    toe = ankle + L_FOOT * _e(R_foot + A_FOOT)
    q_knee = _wrap((R_shin - S_THIGH * q_thigh - SHIN_FRAME) / S_KNEE)

    nan = np.where(ok, 0.0, np.nan)[..., None]
    return dict(
        q_cam=q_cam, q_thigh=q_thigh, ok=ok, h=h,
        cam_pivot=np.broadcast_to(HIP_TO_CAM, P.shape),
        thigh_pivot=np.broadcast_to(HIP_TO_THIGH, K.shape),
        pushrod_joint=P, pushrod_tip=T + nan, knee=K, ankle=ankle + nan, toe=toe + nan,
        R_shin=R_shin, q_knee=q_knee, q_ankle=q_ankle,
        ankle_ok=np.abs(q_ankle) <= ANKLE_LIMIT,
    )


def _pick_branch():
    """Determine which circle-circle root is the assembled machine, by testing which one
    reproduces the CAD rest pose (all joints at zero, where the MJCF's `connect` equality
    is satisfied and the knee angle is 0 by construction)."""
    for b in (-1.0, +1.0):
        fk = forward(0.0, 0.0, branch=b)
        if bool(fk["ok"]) and abs(float(fk["q_knee"])) < 0.02:
            return b
    raise RuntimeError("neither assembly branch reproduces the CAD rest pose")


BRANCH = _pick_branch()


# =============================================================================
# 4.  CONSTRAINED JACOBIAN
# =============================================================================

def jacobian(fk, ankle_mode="stance"):
    """d(toe)/d(cam, thigh) with loop closure differentiated out. Shape (..., 2, 2), with
    J[..., i, j] = d(toe_i)/d(q_j) for q = (cam, thigh) in MuJoCo's sign convention.

    Derivation. Take the passive coordinates to be psi (the pushrod link's absolute
    bearing) and R (the shin body's absolute rotation). Loop closure says the pushrod tip
    and the shin's anchor stub are the same point:

        g(cam, thigh, psi, R) = [P(cam) + L_PUSH*e(psi)] - [K(thigh) + L_ANCHOR*e(R+A_ANCHOR)] = 0

    Every partial is a `perp` of a lever arm, because rotating a planar link by dtheta
    moves its tip by perp(tip - pivot) dtheta:

        dg/dpsi = perp(T - P)             dg/dR     = -perp(T - K)
        dg/dcam = S_CAM*perp(P - cam_pivot)   dg/dthigh = -S_THIGH*perp(K - thigh_pivot)

    Then dq_passive/dq_active = -(dg/dq_passive)^-1 (dg/dq_active), and the chain rule
    gives J. Note det(dg/dq_passive) = -cross(T - P, T - K), which vanishes exactly when
    the pushrod and the anchor stub are collinear -- the dead centre, i.e. h = 0. So the
    Jacobian's rank loss and the workspace boundary are the same event, checked in
    `--self-test`.
    """
    P, K, T = fk["pushrod_joint"], fk["knee"], fk["pushrod_tip"]
    toe, ankle = fk["toe"], fk["ankle"]

    dg_dpsi = _perp(T - P)
    dg_dR = -_perp(T - K)
    dg_dcam = S_CAM * _perp(P - HIP_TO_CAM)
    dg_dthigh = -S_THIGH * _perp(K - HIP_TO_THIGH)

    # Batched 2x2 solve. Columns of the passive block are (dg_dpsi, dg_dR).
    a, c = dg_dpsi[..., 0], dg_dpsi[..., 1]
    b, d = dg_dR[..., 0], dg_dR[..., 1]
    det = a * d - b * c
    with np.errstate(divide="ignore", invalid="ignore"):
        # second row of -inv(passive) @ rhs, i.e. dR/dq_active
        dR_dcam = -(-c * dg_dcam[..., 0] + a * dg_dcam[..., 1]) / det
        dR_dthigh = -(-c * dg_dthigh[..., 0] + a * dg_dthigh[..., 1]) / det

    if ankle_mode == "parallel-thigh":
        # The foot's orientation is tied to the thigh, so rotating the shin about the knee
        # carries only the shin; the foot segment instead moves with the thigh angle.
        dtoe_dR = _perp(ankle - K)
        dtoe_dthigh_direct = S_THIGH * (_perp(K - HIP_TO_THIGH) + _perp(toe - ankle))
    else:
        # Fixed ankle: everything distal to the knee is one rigid body rotating with R.
        dtoe_dR = _perp(toe - K)
        dtoe_dthigh_direct = S_THIGH * _perp(K - HIP_TO_THIGH)

    J = np.empty(toe.shape[:-1] + (2, 2), float)
    J[..., :, 0] = dtoe_dR * dR_dcam[..., None]
    J[..., :, 1] = dtoe_dthigh_direct + dtoe_dR * dR_dthigh[..., None]
    return J


# =============================================================================
# 5.  FORCE AUTHORITY (statics)
# =============================================================================

def axis_force(J, tau_max=(TAU_CAM, TAU_THIGH)):
    """Largest PURE fore-aft and PURE vertical toe force inside the motor torque box.

    Holding a toe force F costs tau = J^T F. For F = f*u along a unit direction u, the
    i-th motor sees f * (J^T u)_i, and (J^T u)_i = sum_k J[k,i] u_k is exactly that
    motor's moment arm about u. So the motor with the LARGER arm saturates first:

        f_max(u) = min_i  tau_max_i / |(J^T u)_i|

    "Pure" matters: this is the force with NO component in the other axis. A leg can
    usually push much harder along its strong axis than in either pure direction.

    Returns (f_x, f_z, bind_x, bind_z) in newtons; bind_* is 0 where the cam binds and 1
    where the thigh binds.
    """
    tau = np.asarray(tau_max, float)
    arms_x = np.abs(J[..., 0, :])      # d(toe_x)/dq  -- arms against a pure fore-aft force
    arms_z = np.abs(J[..., 1, :])      # d(toe_z)/dq  -- arms against a pure vertical force
    with np.errstate(divide="ignore", invalid="ignore"):
        # A zero arm means that motor is not loaded by this force at all, so it imposes no
        # limit: +inf is the right neutral element for the min, and it keeps the whole
        # expression NaN-free so the reductions below need no nan-aware variants.
        cap_x = np.where(arms_x > 0, tau / arms_x, np.inf)
        cap_z = np.where(arms_z > 0, tau / arms_z, np.inf)
    cap_x = np.where(np.isnan(cap_x), np.inf, cap_x)
    cap_z = np.where(np.isnan(cap_z), np.inf, cap_z)
    return (cap_x.min(axis=-1), cap_z.min(axis=-1),
            cap_x.argmin(axis=-1), cap_z.argmin(axis=-1))


def force_polygon(J, tau_max=(TAU_CAM, TAU_THIGH)):
    """The complete set of toe forces the motors can hold: the image of the torque box
    under J^-T, i.e. a parallelogram with corners J^-T (+-tau_cam, +-tau_thigh).
    Returns (..., 4, 2), corners in order."""
    tau = np.asarray(tau_max, float)
    corners = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float) * tau
    JT = np.swapaxes(J, -1, -2)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.linalg.inv(JT)
    return np.einsum("...ij,kj->...ki", inv, corners)


def ankle_vertical_ceiling(fk, deflect_rad):
    """Vertical toe force the PASSIVE ankle can react within a given deflection budget.

    The available ankle torque is the spring's: tau = ANKLE_STIFFNESS * deflect_rad. The
    budget matters because the ankle is a spring, not a stop -- asking it for more torque
    means accepting more foot roll, and past roughly 20 degrees the stance geometry the
    rest of this script assumes no longer holds.

    The ankle is not actuated, so whatever the hip motors can do, the foot can only
    transmit what the ankle joint can hold. The toe force moments the ankle with arm
    perp(toe - ankle_pivot), so f_z <= tau_ankle / |perp(toe-ankle)_z| = tau_ankle /
    |toe_x - ankle_x|.

    This is the LINEARISED ceiling, evaluated at the undeflected geometry. The true
    passive-spring limit is lower: deflecting the ankle grows the moment arm, which grows
    the required torque -- positive feedback, so the equilibrium branch turns over before
    the spring reaches its nominal torque.
    """
    tau_ankle = ANKLE_STIFFNESS * deflect_rad
    arm = np.abs(fk["toe"][..., 0] - fk["ankle"][..., 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(arm > 0, tau_ankle / arm, np.inf)


# =============================================================================
# 6.  THE ASSEMBLY BAND BOUNDARY (= dead centre = singularity), in closed form
# =============================================================================

def singularity_curve(cam_lo, cam_hi, n=1441, thigh_limit=THIGH_LIMIT, **fk_kw):
    """Trace the dead-centre curve exactly.

    Assembly requires |K(thigh) - P(cam)| in [|L_PUSH - L_ANCHOR|, L_PUSH + L_ANCHOR]; the
    bounds are where the two loop circles are tangent. K rides a circle of radius L_THIGH
    about the thigh pivot, so for a given cam the thigh angles hitting either bound are the
    intersections of circle(thigh_pivot, L_THIGH) with circle(P(cam), bound) -- closed
    form again. Returns the toe positions along the curve, shape (m, 2).
    """
    cams = np.linspace(cam_lo, cam_hi, n)
    P = HIP_TO_CAM + L_CRANK * _e(S_CAM * cams + A_CRANK)
    out = []
    for bound in (L_PUSH + L_ANCHOR, abs(L_PUSH - L_ANCHOR)):
        for br in (-1.0, +1.0):
            Kstar, ok, _ = _circle_circle(np.broadcast_to(HIP_TO_THIGH, P.shape), L_THIGH,
                                          P, bound, br)
            thigh = _wrap((_angle(Kstar - HIP_TO_THIGH) - A_THIGH) / S_THIGH)
            keep = ok & (np.abs(thigh) <= thigh_limit)
            if not keep.any():
                continue
            fk = forward(cams[keep], thigh[keep], **fk_kw)
            out.append(fk["toe"])
    return np.concatenate(out, axis=0) if out else np.zeros((0, 2))


# =============================================================================
# 7.  SWEEP AND RASTERISATION
# =============================================================================

def sweep(cam_lo, cam_hi, nc, nt, ankle_mode, thigh_limit=THIGH_LIMIT):
    """Evaluate the whole (cam, thigh) grid at once. Everything downstream is a view of
    this one result."""
    cam = np.linspace(cam_lo, cam_hi, nc)[:, None]
    thigh = np.linspace(-thigh_limit, thigh_limit, nt)[None, :]
    fk = forward(cam, thigh, ankle_mode=ankle_mode)
    J = jacobian(fk, ankle_mode=ankle_mode)
    fx, fz, bx, bz = axis_force(J)
    with np.errstate(divide="ignore", invalid="ignore"):
        det = np.abs(J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0])
    fk.update(J=J, f_x=fx, f_z=fz, bind_x=bx, bind_z=bz, det=det,
              valid=fk["ok"] & fk["ankle_ok"])
    return fk


def rasterise(xy, values, mask, pix, extent=None):
    """Bin scattered samples into square Cartesian cells, keeping the MAX per cell.

    The joint-space grid folds: a single toe position can be reachable in more than one
    posture, with different force authority. Taking the max answers the question a
    designer actually asks -- "what is the best this leg can do at this point?" -- and is
    well defined regardless of how the fold is oriented. `count` reports how many samples
    landed in each cell, so folds are visible rather than silent.
    """
    x, z = xy[..., 0][mask], xy[..., 1][mask]
    v = np.asarray(values)[mask] if np.ndim(values) else np.full(x.shape, float(values))
    good = np.isfinite(x) & np.isfinite(z)
    x, z, v = x[good], z[good], v[good]
    if extent is None:
        extent = (x.min() - pix, x.max() + pix, z.min() - pix, z.max() + pix)
    x0, x1, z0, z1 = extent
    nx, nz = max(int(np.ceil((x1 - x0) / pix)), 1), max(int(np.ceil((z1 - z0) / pix)), 1)
    ix = np.clip(((x - x0) / pix).astype(int), 0, nx - 1)
    iz = np.clip(((z - z0) / pix).astype(int), 0, nz - 1)
    grid = np.full((nz, nx), -np.inf)
    count = np.zeros((nz, nx), int)
    np.maximum.at(grid, (iz, ix), np.nan_to_num(v, nan=-np.inf))
    np.add.at(count, (iz, ix), 1)
    grid[count == 0] = np.nan
    grid[~np.isfinite(grid)] = np.nan
    return grid, count, (x0, x0 + nx * pix, z0, z0 + nz * pix)


# =============================================================================
# 8.  PLOTTING
# =============================================================================
# Colour follows the job each panel does, not taste:
#   * panels (b) and (c) encode MAGNITUDE -> one-hue sequential ramps, light to dark.
#     Two sequential contexts on one figure, so the second takes the next hue. No rainbow.
#   * panel (d) encodes POLARITY about a meaningful midpoint (equal strength in both
#     axes, ratio 1) -> diverging, two hues with a neutral grey middle.
#   * the linkage overlay is CATEGORICAL -> fixed hue slots, never cycled.
# Force spans several decades (it diverges at the dead centre), so the magnitude panels
# are log-normed and clipped, with the clip stated in the colourbar label.

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SURFACE, BAND = "#fcfcfb", "#dfe9f3"
C_LEG, C_LOOP, C_BLOCK, C_SING = "#2a78d6", "#eb6834", "#52514e", "#e34948"

RAMP_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RAMP_ORANGE = ["#fde4d8", "#f9c3ac", "#f5a37f", "#eb6834", "#c9521f", "#98400f", "#5f2708"]
RAMP_DIVERGING = ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#e97676", "#c22e2d", "#6d1615"]


def _cmap(name, hexes):
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list(name, hexes)
    cm.set_bad(SURFACE)
    return cm


def _style(ax, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal", "box")
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("X  forward (m)", color=MUTED, fontsize=8)
    ax.set_ylabel("Z  up (m)", color=MUTED, fontsize=8)
    # Placed by hand rather than via set_title: the axes use adjustable="box", so their
    # drawn height shrinks to satisfy the equal aspect and a padded title would float.
    ax.text(0, 1.050, title, transform=ax.transAxes, color=INK, fontsize=11,
            weight="bold", va="bottom")
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, color=MUTED,
                fontsize=8, va="bottom")


def draw_linkage(ax, fk, label=True):
    """Stick figure of one pose. Categorical hues: the serial leg chain in slot 1, the
    parallel actuation loop in slot 2, the rigid hip block in neutral ink."""
    g = lambda k: np.asarray(fk[k], float).reshape(2)
    hip = np.zeros(2)
    cam_p, thigh_p = g("cam_pivot"), g("thigh_pivot")
    pj, tip, knee, ank, toe = (g("pushrod_joint"), g("pushrod_tip"),
                               g("knee"), g("ankle"), g("toe"))

    ax.plot(*np.array([cam_p, hip, thigh_p]).T, color=C_BLOCK, lw=4.0, zorder=9,
            solid_capstyle="round", label="rigid hip block" if label else None)
    ax.plot(*np.array([cam_p, pj, tip, knee]).T, color=C_LOOP, lw=2.6, zorder=10,
            solid_capstyle="round", label="cam crank + pushrod (the 4-bar)" if label else None)
    ax.plot(*np.array([thigh_p, knee, ank, toe]).T, color=C_LEG, lw=2.8, zorder=10,
            solid_capstyle="round", label="femur - tibia - foot" if label else None)

    joints = np.array([cam_p, thigh_p, pj, tip, knee, ank])
    ax.scatter(*joints.T, s=26, c="white", edgecolor=INK, lw=1.1, zorder=11)
    ax.scatter(*hip, s=52, marker="s", c=INK, zorder=12,
               label="hip pivot (origin)" if label else None)
    from matplotlib.patches import Circle
    ax.add_patch(Circle(toe, TOE_RADIUS, fc=C_LEG, ec=INK, lw=0.8, alpha=0.35, zorder=11))
    ax.scatter(*toe, s=34, c=INK, zorder=13, label="toe (end effector)" if label else None)


def _panel_field(fig, ax, grid, extent, cmap, vlo, vhi, cbar_label, diverging=False):
    """One map panel plus its colourbar.

    Both encodings are logarithmic, because force spans two decades across the workspace
    and diverges at the dead centre. A RATIO panel (`diverging=True`) is plotted as
    log10(ratio) on symmetric arms about 0, so "twice as strong vertically" and "twice as
    strong fore-aft" are the same distance from the neutral midpoint -- on a linear ratio
    scale the whole fore-aft-dominant half would collapse into the bottom 2% of the bar.
    The tick labels are converted back to plain ratios.
    """
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    if diverging:
        span = max(abs(np.log10(vlo)), abs(np.log10(vhi)))
        data = np.log10(np.clip(grid, 10.0 ** -span, 10.0 ** span))
        norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    else:
        span, data = None, np.clip(grid, vlo, vhi)
        norm = LogNorm(vmin=vlo, vmax=vhi)
    im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap,
                   norm=norm, interpolation="nearest", zorder=2)
    # Anchor the colourbar to the AXES box, not the subplot cell. With adjustable="box"
    # the equal-aspect axes are much shorter than their cell, and a `fig.colorbar(ax=...)`
    # would be sized to the cell instead -- the bar would tower over the panel.
    cb = fig.colorbar(im, cax=ax.inset_axes([1.035, 0.0, 0.03, 1.0]))
    if diverging:
        ticks = [-span, -span / 2, 0.0, span / 2, span]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{10 ** t:.3g}" if t >= 0 else f"1/{10 ** -t:.3g}"
                           for t in ticks])
    cb.set_label(cbar_label, color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_edgecolor(GRID)
    return im


def make_figure(g, args, out, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    bw = MASS_KG * GRAVITY
    valid = g["valid"]
    toe = g["toe"]
    pix = args.pix

    reach, _, extent = rasterise(toe, 1.0, valid, pix)
    fz_grid, count, _ = rasterise(toe, g["f_z"] / bw, valid, pix, extent)
    fx_grid, _, _ = rasterise(toe, g["f_x"] / bw, valid, pix, extent)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_grid, _, _ = rasterise(toe, g["f_z"] / np.maximum(g["f_x"], 1e-9),
                                     valid, pix, extent)

    sing = singularity_curve(args.cam_lo, args.cam_hi, ankle_mode=args.ankle)
    ref = forward(STAND_CAM, STAND_THIGH, ankle_mode=args.ankle)
    floor_z = float(ref["toe"][1])

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.6), facecolor=SURFACE)
    fig.suptitle("DASH-01 sagittal leg  |  reachable workspace and end-effector force authority",
                 color=INK, fontsize=14, weight="bold", x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.958,
             f"2-DOF four-bar, hip abduction locked at 0.  Toe position relative to the hip "
             f"pivot.  Ankle held at {np.degrees(float(np.ravel(ref['q_ankle'])[0])):+.1f} deg "
             f"({args.ankle}).  Motor box {args.tau_cam:.0f}/{args.tau_thigh:.0f} N.m,  "
             f"1 BW = {bw:.0f} N ({MASS_KG:.2f} kg).",
             color=MUTED, fontsize=9, ha="left", va="top")

    # ---- (a) workspace -------------------------------------------------------
    ax = axes[0, 0]
    _style(ax, "(a)  Reachable workspace",
           "pale field = image of the four-bar's assembly band;  red = dead-centre fold lines")
    ax.imshow(np.where(np.isfinite(reach), 1.0, np.nan), origin="lower", extent=extent,
              cmap=_cmap("band", [BAND, BAND]), zorder=2, interpolation="nearest")
    if len(sing):
        ax.scatter(sing[:, 0], sing[:, 1], s=2.5, c=C_SING, lw=0, zorder=5,
                   label="dead centre (det J = 0)")
    ax.axhline(floor_z, color=MUTED, lw=1.0, ls="--", zorder=4)
    ax.text(extent[1], floor_z, " floor at stand height ", color=MUTED, fontsize=7.5,
            va="bottom", ha="right")
    draw_linkage(ax, ref)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.94, edgecolor=GRID,
              facecolor=SURFACE, labelcolor=INK)

    # ---- (b) vertical force --------------------------------------------------
    ax = axes[0, 1]
    _style(ax, "(b)  Vertical force authority",
           "largest PURE vertical toe force the motors can hold;  red = the 3.5 BW peak "
           "running GRF")
    _panel_field(fig, ax, fz_grid, extent, _cmap("seq_b", RAMP_BLUE),
                 args.f_lo, args.f_hi, f"body weights (clipped {args.f_lo:g}-{args.f_hi:g})")
    cs = ax.contour(np.clip(fz_grid, args.f_lo, args.f_hi), levels=[GRF_TARGET_BW],
                    colors=[C_SING], linewidths=1.6, extent=extent, zorder=6)
    for t in ax.clabel(cs, fmt={GRF_TARGET_BW: f"{GRF_TARGET_BW:g} BW"},
                       fontsize=7.5, colors=C_SING):
        t.set_path_effects(halo)

    # ---- (c) fore-aft force --------------------------------------------------
    ax = axes[1, 0]
    _style(ax, "(c)  Fore-aft force authority",
           "largest PURE horizontal toe force -- propulsion and braking.  Same colour "
           "scale as (b)")
    _panel_field(fig, ax, fx_grid, extent, _cmap("seq_c", RAMP_ORANGE),
                 args.f_lo, args.f_hi, f"body weights (clipped {args.f_lo:g}-{args.f_hi:g})")
    cs = ax.contour(np.clip(fx_grid, args.f_lo, args.f_hi), levels=[1.0],
                    colors=[C_SING], linewidths=1.6, extent=extent, zorder=6)
    for t in ax.clabel(cs, fmt={1.0: "1 BW"}, fontsize=7.5, colors=C_SING):
        t.set_path_effects(halo)

    # ---- (d) anisotropy + force polygons ------------------------------------
    ax = axes[1, 1]
    _style(ax, "(d)  Anisotropy and the achievable force set",
           "colour = vertical/fore-aft ratio;  outline = the force set's shape, each "
           "normalised to one size")
    _panel_field(fig, ax, ratio_grid, extent, _cmap("div_d", RAMP_DIVERGING),
                 args.r_lo, args.r_hi, "F_vertical / F_fore-aft", diverging=True)

    # The achievable force set spans two decades across the workspace, so drawing the
    # parallelograms to a common newton-per-metre scale makes all but a few invisible.
    # Each is instead normalised to the same plotted size: the OUTLINE carries the shape
    # and the orientation of the strong axis, the COLOUR underneath carries the ratio.
    step_c = max(g["q_cam"].shape[0] // 15, 1)
    step_t = max(g["q_thigh"].shape[1] // 10, 1)
    sub = (slice(None, None, step_c), slice(None, None, step_t))
    poly = force_polygon(g["J"][sub], (args.tau_cam, args.tau_thigh))
    anchors, keep = toe[sub], valid[sub] & np.isfinite(poly).all(axis=(-1, -2))
    radius = np.linalg.norm(poly, axis=-1).max(axis=-1)
    from matplotlib.collections import PolyCollection
    verts = [anchors[i] + poly[i] / radius[i] * args.polygon_size
             for i in zip(*np.nonzero(keep & (radius > 0)))]
    # A white halo: the diverging ramp is dark at BOTH ends, so a plain dark outline
    # disappears over the strongly-anisotropic and the strongly-isotropic regions alike.
    ax.add_collection(PolyCollection(
        verts, facecolors="none", edgecolors=INK, linewidths=0.8, zorder=7,
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")]))

    for ax in axes.ravel()[1:]:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    # Panel (a) also has to fit the mechanism itself, which reaches above the workspace.
    axes[0, 0].set_xlim(min(extent[0], HIP_TO_CAM[0] - 0.06), max(extent[1], 0.2))
    axes[0, 0].set_ylim(extent[2], max(extent[3], 0.07))

    fig.tight_layout(rect=(0, 0, 1, 0.952))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"\nsaved {out}")
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# 9.  SELF-TEST
# =============================================================================

def self_test():
    """Five checks. The first two are the ones that matter: they compare this file's
    hand-transcribed 2-D linkage against values produced by the full 3-D MuJoCo model, so
    they catch a mis-copied offset, a wrong joint sign or a flipped frame twist."""
    ok = True

    def check(name, got, want, tol, unit=""):
        nonlocal ok
        err = abs(got - want)
        good = err <= tol
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}]  {name:<46s} "
              f"{got:+.6f} vs {want:+.6f} {unit}  (err {err:.2e}, tol {tol:g})")

    print("\n1. CAD rest pose -- the MJCF `connect` equality at qpos = 0")
    fk = forward(0.0, 0.0)
    resid = float(np.linalg.norm(fk["pushrod_tip"] - (fk["knee"] + L_ANCHOR *
                                                      _e(fk["R_shin"] + A_ANCHOR))))
    check("loop residual (closed form, must be exact)", resid, 0.0, 1e-12, "m")
    check("knee angle at rest (CAD assembles at 0)", float(fk["q_knee"]), 0.0, 2e-3, "rad")
    print(f"     assembly branch selected at import: {BRANCH:+.0f}")

    print("\n2. `stand` keyframe -- ground truth from the 3-D MuJoCo solve")
    # Independent two-sided check of the frame chain. Build the pushrod tip from the
    # keyframe's own (cam, pushrod) angles and the shin's anchor stub from its own
    # (thigh, knee) angles, using no loop solve at all. If both frame chains are correct
    # the two points coincide -- and this is the ONLY check that exercises PUSH_FRAME,
    # A_PUSH and S_PUSH, which the forward map bypasses.
    P = HIP_TO_CAM + L_CRANK * _e(S_CAM * STAND_CAM + A_CRANK)
    tip = P + L_PUSH * _e(S_CAM * STAND_CAM + PUSH_FRAME + S_PUSH * STAND_PUSH + A_PUSH)
    K = HIP_TO_THIGH + L_THIGH * _e(S_THIGH * STAND_THIGH + A_THIGH)
    anchor = K + L_ANCHOR * _e(S_THIGH * STAND_THIGH + SHIN_FRAME
                               + S_KNEE * STAND_KNEE + A_ANCHOR)
    gap = float(np.linalg.norm(tip - anchor))
    check("loop gap from the keyframe's own passive angles", gap, 0.0, 1e-3, "m")

    fk = forward(STAND_CAM, STAND_THIGH, ankle_mode="stance")
    # Tolerance note: MuJoCo's `connect` is a SOFT equality (solref="0.005 1"), so under
    # stance load the loop stretches by the `gap` measured just above -- 0.31 mm, which is
    # gap/L_ANCHOR = 4.3 mrad of knee. This script closes the loop exactly (rigid), so a
    # knee offset of that order is the expected difference between a rigid and a compliant
    # model, not an error in the transcription. It is bounded here at 3x the implied
    # offset so a genuine geometry mistake still trips the check.
    check("solved knee vs keyframe qpos (rigid vs compliant)",
          float(fk["q_knee"]), STAND_KNEE, 3 * gap / L_ANCHOR, "rad")
    toe_z = float(fk["toe"][1])
    check("toe height below hip vs keyframe", toe_z, TOE_RADIUS - STAND_BASE_Z, 5e-3, "m")
    r_hip = float(np.linalg.norm(fk["toe"]))
    r_thigh = float(np.linalg.norm(fk["toe"] - HIP_TO_THIGH))
    print(f"     stance leg extension: {r_hip:.4f} m from the hip pivot, "
          f"{r_thigh:.4f} m from the thigh pivot")

    print("\n3. Analytic Jacobian vs central differences")
    rng = np.random.default_rng(0)
    worst = 0.0
    for mode in ANKLE_MODES:
        for _ in range(400):
            c, t = rng.uniform(-1.4, 1.4), rng.uniform(-1.0, 1.0)
            f0 = forward(c, t, ankle_mode=mode)
            if not bool(f0["ok"]) or float(f0["h"]) < 0.01:
                continue
            Ja = jacobian(f0, ankle_mode=mode)
            eps = 1e-6
            Jn = np.empty((2, 2))
            for j, (dc, dt) in enumerate(((eps, 0.0), (0.0, eps))):
                fp = forward(c + dc, t + dt, ankle_mode=mode)["toe"]
                fm = forward(c - dc, t - dt, ankle_mode=mode)["toe"]
                Jn[:, j] = (fp - fm) / (2 * eps)
            worst = max(worst, float(np.abs(Ja - Jn).max() / max(np.abs(Jn).max(), 1e-9)))
    check("worst relative error over 1200 poses", worst, 0.0, 1e-5)

    print("\n4. det(loop Jacobian) vanishes at the dead centre")
    cams = np.linspace(-1.5, 1.5, 400)
    dets, hs = [], []
    for c in cams:
        for br in (-1.0, +1.0):
            Kstar, good, _ = _circle_circle(HIP_TO_THIGH, L_THIGH,
                                            HIP_TO_CAM + L_CRANK * _e(S_CAM * c + A_CRANK),
                                            L_PUSH + L_ANCHOR, br)
            if not good:
                continue
            th = float(_wrap((_angle(Kstar - HIP_TO_THIGH) - A_THIGH) / S_THIGH))
            if abs(th) > THIGH_LIMIT:
                continue
            f = forward(c, th)
            dets.append(abs(float(_cross(f["pushrod_tip"] - f["pushrod_joint"],
                                         f["pushrod_tip"] - f["knee"]))))
            hs.append(float(f["h"]))
    # Both quantities are pure float round-off here: the curve is solved in closed form, so
    # the residual is set by cancellation in the tangency, not by any iteration tolerance.
    # det has units of m^2, hence the looser bound.
    check("max |det(dg/dq_passive)| on the traced curve", max(dets) if dets else 1.0,
          0.0, 1e-7, "m^2")
    check("max half-chord h on the traced curve", max(hs) if hs else 1.0, 0.0, 1e-6, "m")

    print("\n5. Statics round trip -- tau = J^T F saturates exactly one motor")
    fk = forward(STAND_CAM, STAND_THIGH)
    J = jacobian(fk)
    fx, fz, bx, bz = axis_force(J)
    tau = J.T @ np.array([0.0, float(fz)])
    check("binding motor torque at the pure-vertical limit",
          float(np.abs(tau).max()), TAU_CAM, 1e-6, "N.m")
    print(f"     stance pure-vertical  {float(fz):8.1f} N  "
          f"({float(fz)/(MASS_KG*GRAVITY):5.2f} BW, binds on "
          f"{'cam' if bz == 0 else 'thigh'})")
    print(f"     stance pure-fore-aft  {float(fx):8.1f} N  "
          f"({float(fx)/(MASS_KG*GRAVITY):5.2f} BW, binds on "
          f"{'cam' if bx == 0 else 'thigh'})")
    print(f"     anisotropy            {float(fz)/float(fx):8.1f} : 1")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


# =============================================================================
# 10.  SUMMARY AND CLI
# =============================================================================

def summarise(g, args):
    bw = MASS_KG * GRAVITY
    v = g["valid"]
    n = v.size
    toe = g["toe"]
    reach = np.hypot(toe[..., 0], toe[..., 1])

    print("\n" + "=" * 78)
    print("DASH-01 sagittal leg -- reachability and force authority")
    print("=" * 78)
    print(f"linkage   crank {L_CRANK:.4f}  pushrod {L_PUSH:.4f}  anchor {L_ANCHOR:.4f}  "
          f"femur {L_THIGH:.4f}  tibia {L_SHIN:.4f}  foot {L_FOOT:.4f}  [m]")
    print(f"sweep     cam [{np.degrees(args.cam_lo):+.0f}, {np.degrees(args.cam_hi):+.0f}] deg "
          f"x {g['q_cam'].shape[0]} | thigh +-{np.degrees(THIGH_LIMIT):.0f} deg "
          f"x {g['q_thigh'].shape[1]} | ankle mode '{args.ankle}'")
    print(f"assembly  {g['ok'].sum():6d} / {n} grid points assemble "
          f"({100 * g['ok'].sum() / n:.0f}%) -- the rest are outside the four-bar's band")
    print(f"          {(g['ok'] & ~g['ankle_ok']).sum():6d} assemble but need an ankle angle "
          f"past +-{np.degrees(ANKLE_LIMIT):.0f} deg")

    area, count, _ = rasterise(toe, 1.0, v, args.pix)
    cells = int(np.isfinite(area).sum())
    print(f"\nworkspace X span   [{toe[..., 0][v].min():+.3f}, {toe[..., 0][v].max():+.3f}] m")
    print(f"          Z span   [{toe[..., 1][v].min():+.3f}, {toe[..., 1][v].max():+.3f}] m")
    print(f"          reach    [{reach[v].min():.3f}, {reach[v].max():.3f}] m from the hip pivot")
    print(f"          area     {cells * args.pix ** 2:.4f} m^2 "
          f"({cells} cells of {args.pix * 1000:.0f} mm; {100 * (count > 1).sum() / max(cells, 1):.0f}% "
          f"reachable in more than one posture)")

    ref = forward(STAND_CAM, STAND_THIGH, ankle_mode=args.ankle)
    Jr = jacobian(ref, ankle_mode=args.ankle)
    fx, fz, bx, bz = axis_force(Jr, (args.tau_cam, args.tau_thigh))
    fx, fz = float(fx), float(fz)
    print(f"\nat the `stand` pose (cam {STAND_CAM:+.4f}, thigh {STAND_THIGH:+.4f} rad, "
          f"{float(np.linalg.norm(ref['toe'])):.4f} m from the hip):")
    print(f"          vertical  {fz:8.1f} N  = {fz / bw:6.2f} BW   "
          f"(binds on {'cam' if bz == 0 else 'thigh'})")
    print(f"          fore-aft  {fx:8.1f} N  = {fx / bw:6.2f} BW   "
          f"(binds on {'cam' if bx == 0 else 'thigh'})")
    print(f"          anisotropy {fz / fx:7.1f} : 1  -- the leg is a strut, not a manipulator")

    fzv, fxv = g["f_z"][v] / bw, g["f_x"][v] / bw
    print(f"\nover the whole reachable set (median / 10th pct):")
    print(f"          vertical  {np.nanmedian(fzv):7.2f} / {np.nanpercentile(fzv, 10):.2f} BW  "
          f"-- {100 * np.nanmean(fzv >= GRF_TARGET_BW):.0f}% clears the {GRF_TARGET_BW:g} BW "
          f"running requirement")
    print(f"          fore-aft  {np.nanmedian(fxv):7.2f} / {np.nanpercentile(fxv, 10):.2f} BW  "
          f"-- {100 * np.nanmean(fxv >= 1.0):.0f}% clears 1 BW")

    if args.ankle_deflect > 0:
        rad = np.radians(args.ankle_deflect)
        ceil = ankle_vertical_ceiling(g, rad)[v] / bw
        print(f"\npassive ankle ceiling, {args.ankle_deflect:.0f} deg of spring deflection "
              f"({ANKLE_STIFFNESS * rad:.1f} N.m, linearised): "
              f"median {np.nanmedian(ceil):.2f} BW, "
              f"{100 * np.nanmean(ceil >= GRF_TARGET_BW):.0f}% clears {GRF_TARGET_BW:g} BW")
        print("          -> where this is below panel (b), the ANKLE is the binding "
              "constraint, not the motors")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc", type=int, default=1081, help="cam samples (raster fill)")
    ap.add_argument("--nt", type=int, default=541, help="thigh samples (raster fill)")
    ap.add_argument("--cam-lo", type=float, default=-90.0, metavar="DEG",
                    help="cam sweep start. The MJCF's +-86 deg range is a CAD guess: the "
                         "four-bar assembles at EVERY cam angle and recorded hardware "
                         "sweeps span ~245 deg, so the default covers the full circle")
    ap.add_argument("--cam-hi", type=float, default=270.0, metavar="DEG")
    ap.add_argument("--ankle", choices=ANKLE_MODES, default="stance",
                    help="how the passive ankle is held (see the module docstring)")
    ap.add_argument("--tau-cam", type=float, default=TAU_CAM, metavar="NM")
    ap.add_argument("--tau-thigh", type=float, default=TAU_THIGH, metavar="NM")
    ap.add_argument("--ankle-deflect", type=float, default=0.0, metavar="DEG",
                    help="if > 0, also report the vertical force the PASSIVE ankle spring "
                         "can react within this much deflection -- usually the real "
                         "binding constraint. 20 deg is the conventional usable budget")
    ap.add_argument("--pix", type=float, default=0.01, metavar="M",
                    help="raster cell size for the maps")
    ap.add_argument("--f-lo", type=float, default=0.3, help="force colour scale, BW")
    ap.add_argument("--f-hi", type=float, default=60.0, help="force colour scale, BW")
    ap.add_argument("--r-lo", type=float, default=0.025,
                    help="anisotropy colour scale (kept symmetric with --r-hi)")
    ap.add_argument("--r-hi", type=float, default=40.0, help="anisotropy colour scale")
    ap.add_argument("--polygon-size", type=float, default=0.05, metavar="M",
                    help="plotted size of the normalised force-set outlines in panel (d)")
    ap.add_argument("--out", default="_reachability.png")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the hand-transcribed linkage and the Jacobian, then exit")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    args.cam_lo, args.cam_hi = np.radians(args.cam_lo), np.radians(args.cam_hi)
    g = sweep(args.cam_lo, args.cam_hi, args.nc, args.nt, args.ankle)
    summarise(g, args)
    make_figure(g, args, args.out, args.show)


if __name__ == "__main__":
    main()
