"""Hand-designed walking controller v2 for DASH-01 — capture-step FSM, no RL, no network.

Second attempt, designed against the two measured verdicts from v1 (scripted_walk.py):

  * v1's m3 autopsy: the robot FREE-FALLS from the keyframe (survival == the analytic topple time
    across a 5x gravity sweep). The keyframe stands with the toes 9.2 cm AHEAD of the CoM (a 5.7
    deg backward lean at t=0), the point-toe contact has zero CoP authority, and a free-running
    gait clock places feet on ITS schedule, not when the fall demands one — while pitch sweeps the
    toe away at ~L*theta_dot, twice as fast as the old 0.8 Hz drive could answer.
  * the plant REBUILD (2026-08-10): position-loop kp x3 moved the sagittal drive 0.8 -> 3.0 Hz
    (tau 199 -> 53 ms), the ankle is a rigid carbon tube, the shins lost 209 g each. Total
    command latency ~78 ms against a 306 ms topple constant — ratio 0.25, inside classical
    stabilizability for the first time. v1's "feedback destabilizes" result was measured at
    0.8 Hz and does NOT carry over.

What is different from v1, each tied to a measured failure:

  1. NO GLOBAL CLOCK. A per-leg finite-state machine: swings are triggered by a nominal timer OR
     EARLY, the moment the instantaneous capture point (ICP) leaves a margin box around the stance
     toe. v1's clock kept sweeping on schedule while the robot fell between beats.
  2. FEET UNDER THE COM BY CONSTRUCTION. The stance posture carries a fixed fore-aft offset x0
     (~-0.14 m, tuned) so the neutral stance is balanced — the keyframe's 5.7 deg lean is
     commanded away through the 3 Hz drive in the first ~150 ms instead of being inherited.
  3. PLACEMENT IN THE GRAVITY FRAME. Foot targets are computed against world-vertical (via the
     measured gravity vector), then converted to base-frame joint targets with a -L*sin(pitch)
     correction. v1 commanded in the BASE frame, so every degree of pitch silently moved the real
     foothold the wrong way — the mechanism measured to outrun placement 2:1.
  4. CONTINUOUS SWING RETARGETING. The swing foot tracks the CURRENT capture point until
     touchdown; v1 latched the target once at liftoff, 300+ ms of fall before it landed.
  5. LEAD RESIZED TO THE NEW PLANT: tau = 1/(2*pi*3.0) = 53 ms, one-third of a swing, so a
     commanded step is ~99% delivered by touchdown.

Same legitimacy rules as v1: the controller reaches the plant ONLY through the env's residual
channel (make_env zeroes the Fourier generator and both built-in reflexes), so the drive filter,
transport delay, per-joint velocity caps, torque limits, contact model, workspace kill and
termination rules are byte-identical to what every RL run is graded on. Sensing is onboard-only:
encoders (FK), IMU gravity + gyro, and leg odometry for velocity ('truth' oracle kept as an
ablation).

Usage
  # smoke test / evaluate on the current ladder plant (walk_fwd_m3 = 3 Hz drive, rigid ankle)
  python training/scripted_walk2.py --milestone m3 --episodes 3
  # tune (coordinate descent), starting from the defaults
  python training/scripted_walk2.py --tune --milestone m3 --seeds 3 --passes 3
  # the deliverable: stand -> forward -> stand -> backward -> stand in ONE episode, video + stats
  python training/scripted_walk2.py --milestone m3 --schedule demo --episodes 3 \\
      --video milestones/scripted2_m3.mp4 --stats milestones/scripted2_m3.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import mujoco

import cpg_gait
import fourier_gait
from scripted_walk import make_env, LegOdometry, smoothstep, VIDEO_FPS
from env import DashEnv

HIP_ROLL_L, CAM_L, THIGH_L, HIP_ROLL_R, CAM_R, THIGH_R = range(6)
G = 9.81
LEAD_LP_S = 0.02        # low-pass on the lead differentiator (drive tau is 53 ms; keep 2.5x faster)
LEAD_CLIP = 0.20        # rad cap on the lead contribution (amplifier, therefore bounded)
DX_SLEW = 1.0           # m/s cap on any foot target's fore-aft rate (snap guard)
Z_RAMP_S = 0.40         # s to fade the z_off leg-extension feedforward in (step input = a hop)
STANCE, SWING = 0, 1

GAINS = dict(
    # ---- posture ------------------------------------------------------------------------------
    x0=-0.140,        # stance fore-aft offset (m): feet UNDER the CoM (keyframe alone is -0 and
    #                   stands 9.2 cm toe-ahead = the measured 5.7 deg backward lean)
    # Weight support is JOINT-SPACE gravity compensation, not "leg extension": the leg stands at
    # 96% reach, so toe-space extension does not exist — commanding dz=-0.065 through the IK was
    # MEASURED to move the toe +0.085 m FORWARD and 0.008 m down, silently re-creating the very
    # toes-ahead-of-CoM lean the balance loop fights. The keyframe droop is servo P-error under
    # gravity (~stand_torque/kp = 0.15 rad), so cancel it there: bias = stand_torque/kp seeded at
    # reset (onboard-knowable: motor current + kp), refined by a slow per-joint integrator.
    ki=3.0,           # 1/s integrator on (target - measured) joint error, stance legs only
    int_clip=0.30,    # rad cap on the total integrator + gravity bias
    # CROUCH. The leg stands at 96% reach: there is NO down-reach at the keyframe, and a capture
    # step MUST reach down — the swung foot hovered 1-5 cm above ground (measured: 0.0 N on the
    # rear foot for an entire fall; joints ON target) from rear-region IK error plus the pitch
    # lifting the rear foot. Standing at z_home of crouch turns the LUT's lift-only workspace
    # into +-z_home of two-sided reach. Verified: cmd (0,+0.05) -> toe +0.051 measured.
    z_home=0.050,     # crouch depth (m of toe lift at stance)
    seek_v=0.30,      # m/s ground-seek descent rate at the end of a swing (until contact)
    # PER-LEG RIDE-HEIGHT LOOP, closed on FK (which sees the passive joints). The base sinks
    # ~15 cm during stepping with every ACTUATED joint on target: the 4-bar loop-closure
    # equalities yield under load, so no position-space feedforward can hold height — but the
    # measured toe position includes the stretch, so a height loop on FK does. v1 measured this
    # exact loop DESTABILIZING through the 0.8 Hz drive; at 3 Hz, re-tested, it holds. Only
    # possible at a crouch: the loop needs down-range, and z_home is what provides it.
    k_h=1.00,         # m of stance-leg dz per m of height error
    k_hd=0.10,        # ... per m/s of height rate (damping)
    # ---- gait timing (FSM, no global clock) ---------------------------------------------------
    T_sw=0.30,        # swing duration (s) ~ 5.7x drive tau -> command fully delivered in flight
    T_ds=0.14,        # nominal double-support dwell between swings (s)
    clearance=0.050,  # swing apex lift (m); reachable ceiling is 0.10
    # ---- foot placement (capture point, gravity frame) ----------------------------------------
    k_cap=1.0,        # gain on v_hat/omega (1.0 = the textbook capture point)
    k_dv=0.20,        # extra placement per m/s of velocity ERROR (speed regulation)
    k_theta=0.90,     # gravity-frame compensation: m of base-frame correction per unit grav_x
    #                   (geometrically = toe depth below base, 0.98 m; tuned around that)
    dx_clip=0.26,     # placement clip (m); measured reachable box is +-0.30
    # ---- step-on-demand -----------------------------------------------------------------------
    # Stepping is the SECOND responder: it fires when the ICP error exceeds what the stance
    # shift can absorb (margin ~ trim_clip). While standing balanced the robot takes NO steps.
    icp_margin=0.12,  # |ICP - support| (m) beyond which a swing fires
    t_ds_min=0.08,    # never re-step faster than this after a touchdown (s)
    # ---- continuous stance trims (viable again at 3 Hz) ---------------------------------------
    # THE PRIMARY BALANCE LOOP. In double support, shifting both loaded feet fore-aft moves the
    # support under the ICP directly — it is the CoP authority the point toe doesn't have. Driven
    # by the ICP error, NOT by pitch: the keyframe's imbalance is geometric (toes ahead of the
    # CoM with the base perfectly level), so a grav_x-driven trim reads zero exactly when the
    # robot most needs correcting (measured: that is the first 150 ms of every episode).
    k_icp=0.80,       # m of stance shift per m of ICP offset (1.0 = park the support ON the ICP)
    k_pitch=0.10,     # residual attitude leveling (m per unit grav_x)
    k_pitchd=0.04,    # ... per rad/s of pitch rate (damping)
    trim_clip=0.12,   # cap (m); this is the stance loop's total authority
    # ---- lateral (inert while roll is railed, m2..m4) -----------------------------------------
    w_shift=0.10,     # rad of hip_roll weight shift toward the stance side during swing
    k_roll=0.50,      # rad of hip_roll per unit grav_y (PD damping)
    k_rolld=0.10,     # ... per rad/s of roll rate
    k_vy=0.20,        # ... per m/s of lateral velocity
    roll_clip=0.35,   # cap (rad)
    # ---- drive inversion + estimation ---------------------------------------------------------
    lead_s=0.053,     # feedforward lead (s); 0.053 = exact inversion of the measured 3 Hz pole
    est_tau=0.060,    # leg-odometry low-pass (s)
)

GAIN_SPACE = dict(
    x0=(-0.22, -0.04), ki=(0.0, 8.0), int_clip=(0.10, 0.45),
    z_home=(0.02, 0.08), seek_v=(0.10, 0.60), k_h=(0.0, 2.5), k_hd=(-0.1, 0.5),
    T_sw=(0.18, 0.45), T_ds=(0.04, 0.30), clearance=(0.02, 0.09),
    k_cap=(0.3, 1.8), k_dv=(-0.3, 0.8), k_theta=(0.0, 1.6), dx_clip=(0.14, 0.30),
    icp_margin=(0.04, 0.20), t_ds_min=(0.02, 0.10),
    k_icp=(0.0, 1.6), k_pitch=(-0.6, 0.6), k_pitchd=(-0.2, 0.2), trim_clip=(0.04, 0.20),
    w_shift=(0.0, 0.30), k_roll=(-1.2, 1.2), k_rolld=(-0.4, 0.4), k_vy=(-0.8, 0.8),
    roll_clip=(0.0, 0.45),
    lead_s=(0.0, 0.12), est_tau=(0.015, 0.10),
)
TUNE_ORDER = ["x0", "z_home", "k_h", "k_hd", "seek_v", "ki", "k_icp", "trim_clip", "icp_margin",
              "T_sw", "T_ds", "clearance",
              "k_cap", "k_theta", "k_dv", "dx_clip", "t_ds_min",
              "k_pitch", "k_pitchd", "lead_s", "est_tau",
              "w_shift", "k_roll", "k_rolld", "k_vy", "roll_clip"]

GAINS_FILE = PKG_DIR / "scripted2_gains.json"

# CoM position in the BASE frame, measured on the rebuilt plant at the settled keyframe
# (subtree_com in base coords). On hardware this is a CAD/mass-model constant, not a sensor.
COM_BASE = np.array([0.008, -0.095])          # (x, z relative to base origin)
TOE_DEPTH = 0.98                              # m the toe hangs below the base origin


class CaptureStepWalker:
    """Per-leg FSM + capture-point foot placement. Pure numpy over onboard measurements."""

    IK_PROBE = 0.02
    JZ_TRUST = 0.10     # |dx| band where the LUT's lift Jacobian is trustworthy (v1 finding:
    #                     it SIGN-FLIPS past dx < -0.14, exactly where x0 now lives)

    def __init__(self, env: DashEnv, gains=None, est="odom", v_des=0.0):
        self.env = env
        self.g = dict(GAINS)
        if gains:
            self.g.update(gains)
        self.est_mode = est
        self.v_des = float(v_des)
        self.dt = env.control_dt
        self.lut = cpg_gait.load_lut(env.cfg.cpg_lut)
        self.ik0 = np.asarray(cpg_gait.foot_ik(0.0, 0.0, self.lut), float)
        self.nominal = np.asarray(env._nominal6, float).copy()
        self.res_off = fourier_gait.spec_dim(env.cfg.n_harmonics, env.n_steer)
        self.res_scale = float(env.cfg.residual_scale)
        self.omega = np.sqrt(G / float(env.height_target))
        self._lp = float(np.exp(-self.dt / LEAD_LP_S))
        self.reset()

    def reset(self):
        g = self.g
        self.t = 0.0
        self.state = np.array([STANCE, STANCE])       # per-leg
        self.t_state = np.zeros(2)                    # time in current state
        self.swing_leg = -1                           # -1 = double support
        self.next_leg = 0                             # who steps next in the alternation
        self.t_since_td = 1e9                         # time since the last touchdown
        # per-leg fore-aft toe offsets (base frame, relative to nominal_toe): where each foot's
        # current stance is anchored / its swing started. Seeded at the MEASURED toe positions,
        # NOT at x0 — commanding x0 outright at t=0 is a 0.14 m jerk on two loaded feet through a
        # 3 Hz drive, and it measurably throws the base (0.41 s survival vs 0.99 s doing nothing).
        # The keyframe's lean is instead walked off: the ICP check sees the stance 9 cm off
        # balance on the very first tick and fires an on-demand step toward x0.
        ref_x = float(np.asarray(self.lut["nominal_toe"])[0])
        toes0 = self._toes_body()
        seed = np.array([toes0[0][0] - ref_x, toes0[1][0] - ref_x])
        self.x_cur = seed.copy()
        self.x_lift = seed.copy()
        self.x_td = seed.copy()
        self.dx_prev = seed.copy()          # for the per-leg command slew guard
        self.dz_sw = np.full(2, g["z_home"])  # per-leg swing height state (for the ground seek)
        self.h0 = float(-np.min(toes0[:, 2]))  # standing height above the toes at reset
        self.h_filt = self.h0
        self.h_prev = self.h0
        self._hlp = float(np.exp(-self.dt / 0.03))
        # joint-space gravity compensation: stand_torque/kp is the servo's known P-droop under
        # the DOUBLE-support standing load (recomputed at env.reset for the episode's plant).
        # Kept separate from the integrator because it is load-scheduled: in single support the
        # stance leg carries twice the load, and waiting for an integrator to discover that
        # (measured) sinks the base ~15 cm per swing and trips the workspace kill.
        kp = np.maximum(self.env.model.actuator_gainprm[:6, 0], 1e-6)
        self.grav_bias = np.clip(np.asarray(self.env._stand_torque[:6], float) / kp,
                                 -g["int_clip"], g["int_clip"])
        self.integ = np.zeros(6)
        self.odom = LegOdometry(g["est_tau"], self.dt)
        self.odom.VMAX = 1.5                # tighter than the class default: a mis-selected
        #                                     stance sample is worth m/s, and placement multiplies it
        self.odom.reset()
        self.v_hat = np.zeros(3)
        self.q_prev = None
        self.dq_filt = np.zeros(6)
        self.n_steps = 0
        self.n_demand = 0
        self.dbg = {}

    # ---------- sensing ----------
    def _toes_body(self):
        e = self.env
        R = e.data.xmat[e.base_id].reshape(3, 3)
        base = e.data.xpos[e.base_id]
        return np.array([R.T @ (e.data.geom_xpos[g] - base) for g in e.foot_gids])

    def _ik(self, dx, dz):
        """Measured IK COMPOSED as: sagittal curve at dz=0 + a lift Jacobian probed only inside
        the trustworthy |dx| band, applied for ALL dz. The raw table's rear region (dx < -0.14) is
        branch-selection garbage — its lift direction sign-flips — and x0 puts the working range
        exactly there, so v2 never reads the table off the dz=0 row."""
        j0 = np.asarray(cpg_gait.foot_ik(dx, 0.0, self.lut), float)
        dxp = float(np.clip(dx, -self.JZ_TRUST, self.JZ_TRUST))
        jp = np.asarray(cpg_gait.foot_ik(dxp, 0.0, self.lut), float)
        jz = (np.asarray(cpg_gait.foot_ik(dxp, self.IK_PROBE, self.lut), float) - jp) / self.IK_PROBE
        return (j0 - self.ik0) + jz * dz

    # ---------- the control law ----------
    def _placement(self, v_x, grav_x):
        """Touchdown target in base-frame toe units. Capture point + speed regulation, computed
        against world-vertical and mapped back with the -k_theta*grav_x geometric correction."""
        g = self.g
        T_st = g["T_sw"] + 2.0 * g["T_ds"]            # nominal single-leg stance duration
        X = (g["k_cap"] * v_x / self.omega            # arrest the measured motion
             + g["k_dv"] * (v_x - self.v_des)         # regulate toward the command
             + 0.5 * self.v_des * T_st)               # neutral point for steady progression
        dx = g["x0"] + X - g["k_theta"] * grav_x
        return float(np.clip(dx, g["x0"] - g["dx_clip"], g["x0"] + g["dx_clip"]))

    def _icp_offset(self, toes, v_x, grav_x):
        """ICP position relative to the support (m, + = ahead), in the gravity frame.
        Support reference = the stance toe (single support) or the mid-point (double)."""
        st = [i for i in (0, 1) if self.state[i] == STANCE]
        # world-frame fore-aft of toe relative to CoM: base-frame dx + small-angle tilt term
        xs = [float(toes[i][0] - COM_BASE[0] + grav_x * (-(toes[i][2] - COM_BASE[1])))
              for i in st]
        x_sup = float(np.mean(xs))
        return v_x / self.omega - x_sup               # ICP rel. CoM  minus  support rel. CoM

    def __call__(self):
        e, g = self.env, self.g
        grav = e._gravity_body()
        gyro = e._ang_vel_body()
        pitch, roll = float(grav[0]), float(grav[1])
        pitch_rate, roll_rate = float(gyro[1]), float(gyro[0])
        toes = self._toes_body()
        v_odom = self.odom.update(toes, gyro)
        self.v_hat = e._vel_body().copy() if self.est_mode == "truth" else v_odom
        v_x, v_y = float(self.v_hat[0]), float(self.v_hat[1])

        s_icp = self._icp_offset(toes, v_x, pitch)
        demand = abs(s_icp) > g["icp_margin"]

        # ---- FSM transitions ------------------------------------------------------------------
        self.t_since_td += self.dt
        if self.swing_leg < 0:
            # double support: swings alternate on the T_ds timer, or fire EARLY on ICP demand.
            # The timer runs even at v_des = 0 — a point-toe biped with pitch free has no static
            # stand (nothing holds the sagittal DOF between steps), so "stay in place" is
            # STEPPING in place with the capture law centering each footfall. This was tested the
            # other way first: gating the timer off while balanced fell in 0.5 s every seed.
            if self.t_since_td >= g["t_ds_min"] and (self.t_since_td >= g["T_ds"] or demand):
                i = self.next_leg
                self.swing_leg = i
                self.state[i] = SWING
                self.t_state[i] = 0.0
                self.x_lift[i] = self.x_cur[i]
                self.n_steps += 1
                if demand and self.t_since_td < g["T_ds"]:
                    self.n_demand += 1
        else:
            i = self.swing_leg
            self.t_state[i] += self.dt
            u = self.t_state[i] / g["T_sw"]
            j = 1 - i
            # strike test in the WORLD frame (base-frame toe heights lie by dx*grav_x under
            # pitch): the swing toe has reached the stance toe's ground plane
            dz_w = (toes[i][2] - toes[j][2]) - (toes[i][0] - toes[j][0]) * pitch
            struck = (u > 0.5 and dz_w <= 0.002)
            # GROUND-SEEK TIMEOUT. After the flight phase the foot descends at seek_v until it
            # actually touches (measured, not modelled — rear-region IK error and pitch both move
            # the real ground several cm from where any model puts it; a capture step that ends
            # hovering is a capture step that never happened, measured as 0.0 N on the rear foot
            # through an entire fall). The timeout accepts a floor-limited descent as stance.
            seek_budget = (g["z_home"] + 0.02) / g["seek_v"] + 0.10
            if struck or self.t_state[i] >= g["T_sw"] + seek_budget:
                self.state[i] = STANCE
                # anchor at the dx actually COMMANDED at strike (an early strike lands short of
                # x_td); dx_prev carries it — minus the stance trim that will be re-added
                self.x_cur[i] = float(self.dx_prev[i])
                self.swing_leg = -1
                self.next_leg = j
                self.t_since_td = 0.0

        # ---- stance feet: sweep at the commanded ground speed + the ICP-driven stance shift ---
        # s_icp < 0 = ICP behind the support = falling backward -> shift the loaded feet BACK
        # (which moves the CoM forward over them). k_icp is the CoP-substitute loop.
        trim = float(np.clip(g["k_icp"] * s_icp
                             + g["k_pitch"] * pitch + g["k_pitchd"] * pitch_rate,
                             -g["trim_clip"], g["trim_clip"]))
        for i in (0, 1):
            if self.state[i] == STANCE:
                self.x_cur[i] -= self.v_des * self.dt          # anchored foot moves back as the
                #                                                body advances at v_des
                lo, hi = g["x0"] - g["dx_clip"], g["x0"] + g["dx_clip"]
                self.x_cur[i] = float(np.clip(self.x_cur[i], lo, hi))

        # ---- swing foot: capture step with a ground-seeking landing ---------------------------
        # the crouch is ramped in over Z_RAMP_S: commanded as a step it drops the body 5 cm onto
        # its feet at t=0 (the mirror image of the z_off hop v2 already had to remove)
        zh = g["z_home"] * min(1.0, self.t / Z_RAMP_S)
        # ride-height loop: FK height above the loaded (lowest) toe, PD toward the crouched
        # reference. Output extends the stance legs (dz below zh) when the plant's soft 4-bar
        # loop lets the base sink.
        st_feet = [i for i in (0, 1) if self.state[i] == STANCE]
        h = float(-np.min([toes[i][2] for i in st_feet])) if st_feet else self.h_filt
        self.h_filt = self._hlp * self.h_filt + (1.0 - self._hlp) * h
        h_rate = (self.h_filt - self.h_prev) / self.dt
        self.h_prev = self.h_filt
        dz_corr = float(np.clip(-g["k_h"] * ((self.h0 - zh) - self.h_filt) + g["k_hd"] * h_rate,
                                -zh - 0.02, 0.05))
        dxz = np.zeros((2, 2))
        for i in (0, 1):                                       # stance feet live at the crouch
            if self.state[i] == STANCE:
                dxz[i] = (self.x_cur[i] + trim, zh + dz_corr)
        for i in (0, 1):
            if self.state[i] == SWING:
                u = self.t_state[i] / g["T_sw"]
                if u < 1.0:                                    # FLIGHT: fly to the capture point
                    if u < 0.8:                                # (frozen late so the seek descends
                        self.x_td[i] = self._placement(v_x, pitch)      # vertically)
                    dx = self.x_lift[i] + (self.x_td[i] - self.x_lift[i]) * smoothstep(u)
                    self.dz_sw[i] = zh + g["clearance"] * float(np.sin(np.pi * u))
                else:                                          # SEEK: descend until real contact
                    dx = self.x_td[i]
                    self.dz_sw[i] = max(0.0, self.dz_sw[i] - g["seek_v"] * self.dt)
                dxz[i] = (dx, self.dz_sw[i])    # placement is absolute — no trim on the swing foot

        # command slew guard: no foot target moves faster than DX_SLEW regardless of what the
        # trims/retargeting add up to — a snap on a loaded foot is an impulse on the base
        step = DX_SLEW * self.dt
        dxz[:, 0] = np.clip(dxz[:, 0], self.dx_prev - step, self.dx_prev + step)
        self.dx_prev = dxz[:, 0].copy()

        # ---- assemble the 6 joint targets -----------------------------------------------------
        q = self.nominal.copy()
        for i, (sgn, cam_j, th_j) in enumerate(((+1, CAM_L, THIGH_L), (-1, CAM_R, THIGH_R))):
            dcam, dthigh = self._ik(float(dxz[i][0]), float(dxz[i][1]))
            q[cam_j] += sgn * dcam
            q[th_j] += sgn * dthigh
        # lateral: weight shift toward the stance side while a leg is in the air, plus PD damping.
        # hip_roll axes are mirrored: +u/-u shifts both feet the same physical way.
        shift = 0.0
        if self.swing_leg == 0:
            shift = -g["w_shift"]                     # left leg swings -> lean onto the right
        elif self.swing_leg == 1:
            shift = +g["w_shift"]
        u_lat = float(np.clip(g["k_roll"] * roll + g["k_rolld"] * roll_rate + g["k_vy"] * v_y
                              + shift, -g["roll_clip"], g["roll_clip"]))
        q[HIP_ROLL_L] += u_lat
        q[HIP_ROLL_R] -= u_lat

        # ---- gravity compensation, load-scheduled + integrator-trimmed ------------------------
        # single support doubles the stance leg's load; schedule its bias x2 instead of waiting
        # for the integrator to find 0.15 rad of extra droop mid-swing
        bias = self.grav_bias.copy()
        if self.swing_leg >= 0:
            st = 1 - self.swing_leg
            for jj in ((CAM_L, THIGH_L), (CAM_R, THIGH_R))[st]:
                bias[jj] *= 2.0
            for jj in ((CAM_L, THIGH_L), (CAM_R, THIGH_R))[self.swing_leg]:
                bias[jj] = 0.0                    # the airborne leg carries nothing
        q_act = self.env.data.qpos[self.env.act_qadr][:6]
        err = q - q_act
        for i, (cam_j, th_j) in enumerate(((CAM_L, THIGH_L), (CAM_R, THIGH_R))):
            if self.state[i] == STANCE:
                for jj in (cam_j, th_j):
                    self.integ[jj] += g["ki"] * err[jj] * self.dt
        self.integ[HIP_ROLL_L] += g["ki"] * err[HIP_ROLL_L] * self.dt
        self.integ[HIP_ROLL_R] += g["ki"] * err[HIP_ROLL_R] * self.dt
        self.integ = np.clip(self.integ, -g["int_clip"], g["int_clip"])

        # ---- drive inversion: lead the measured 3 Hz pole -------------------------------------
        if self.q_prev is None:
            self.q_prev = q.copy()
        dq = (q - self.q_prev) / self.dt
        self.dq_filt = self._lp * self.dq_filt + (1.0 - self._lp) * dq
        q_cmd = q + bias + self.integ + np.clip(g["lead_s"] * self.dq_filt, -LEAD_CLIP, LEAD_CLIP)
        self.q_prev = q.copy()

        act = np.zeros(e.action_dim, np.float32)
        act[self.res_off:self.res_off + 6] = np.clip(
            (q_cmd - self.nominal) / self.res_scale, -1.0, 1.0)

        self.t += self.dt
        self.dbg = dict(v_hat=self.v_hat.copy(), pitch=pitch, roll=roll, s_icp=s_icp,
                        state=self.state.copy(), dx=dxz[:, 0].copy(), dz=dxz[:, 1].copy(),
                        x_td=self.x_td.copy(), trim=trim, u_lat=u_lat,
                        q_des=q.copy(), q_cmd=q_cmd.copy(), demand=demand)
        return act


# ------------------------------------------------------------------------------------------------
# rollout with a COMMAND SCHEDULE (the deliverable is stand -> fwd -> stand -> back -> stand)
# ------------------------------------------------------------------------------------------------
def demo_schedule(v_fwd=0.15, v_back=-0.10):
    return [(0.0, 0.0), (5.0, v_fwd), (15.0, 0.0), (20.0, v_back), (30.0, 0.0)]


def rollout(env, gains, seed, schedule=None, v_des=0.0, est="odom", record=False, on_step=None):
    """One episode. `schedule` = [(t_start, v_cmd), ...] overrides the constant v_des."""
    sched = sorted(schedule) if schedule else [(0.0, float(v_des))]
    ctl = CaptureStepWalker(env, gains, est=est, v_des=sched[0][1])
    obs, _ = env.reset(seed=int(seed))
    env.set_command(sched[0][1], 0.0)
    ctl.reset()
    x0 = float(env.data.qpos[0])
    keys = ("t", "vx", "vy", "vx_hat", "pitch", "roll", "contact", "state", "dx", "dz",
            "x_td", "s_icp", "trim", "u_lat", "q_des", "q_act", "base_z", "v_cmd", "demand")
    tel = {k: [] for k in keys}
    seg_err = {}                                     # per-command-segment |v - v_cmd| sums
    n, k_sched, terminated = 0, 0, False
    while True:
        if k_sched + 1 < len(sched) and ctl.t >= sched[k_sched + 1][0]:
            k_sched += 1
            ctl.v_des = float(sched[k_sched][1])
            env.set_command(ctl.v_des, 0.0)
        act = ctl()
        if record:
            d = ctl.dbg
            v_true = env._vel_body()
            tel["t"].append(ctl.t)
            tel["vx"].append(float(v_true[0])); tel["vy"].append(float(v_true[1]))
            tel["vx_hat"].append(float(d["v_hat"][0]))
            tel["pitch"].append(d["pitch"]); tel["roll"].append(d["roll"])
            tel["contact"].append(env._foot_contacts().copy())
            tel["state"].append(d["state"]); tel["dx"].append(d["dx"]); tel["dz"].append(d["dz"])
            tel["x_td"].append(d["x_td"]); tel["s_icp"].append(d["s_icp"])
            tel["trim"].append(d["trim"]); tel["u_lat"].append(d["u_lat"])
            tel["q_des"].append(d["q_cmd"]); tel["q_act"].append(env.data.qpos[env.act_qadr].copy())
            tel["base_z"].append(float(env.data.qpos[2]))
            tel["v_cmd"].append(ctl.v_des); tel["demand"].append(d["demand"])
        obs, r, terminated, truncated, info = env.step(act)
        err = abs(float(env._vel_body()[0]) - ctl.v_des)
        s = seg_err.setdefault(k_sched, [0.0, 0])
        s[0] += err; s[1] += 1
        n += 1
        if on_step is not None:
            on_step()
        if terminated or truncated:
            break
    t_alive = n * env.control_dt
    segs = [(sched[k][1], v[0] / max(v[1], 1)) for k, v in sorted(seg_err.items())]
    out = dict(t_alive=t_alive, dist=float(env.data.qpos[0]) - x0, fell=bool(terminated),
               steps=n, survived=bool(not terminated), n_foot_steps=ctl.n_steps,
               n_demand=ctl.n_demand, seg_err=segs,
               v_err=float(np.mean([e_ for _, e_ in segs])))
    if record:
        out["tel"] = {k: np.array(v) for k, v in tel.items() if v}
    return out


def score(res, sched_span=1.0):
    """Survival seconds, tripled when the command is tracked — v1's anti-exploit objective (a
    plain distance term was measurably gamed: the tuner ran away at 1.6 m/s to bank distance)."""
    out = []
    for r in res:
        track = np.mean([max(0.0, 1.0 - e / max(abs(v), 0.12)) for v, e in r["seg_err"]])
        out.append(r["t_alive"] * (1.0 + 2.0 * track))
    return float(np.mean(out))


# ------------------------------------------------------------------------------------------------
def tune(args):
    seeds = list(range(args.seeds))
    env = make_env(args.milestone, dr=args.dr, pushes=args.pushes, episode_s=args.tune_seconds)
    sched = [(0.0, 0.0), (args.tune_seconds / 3, args.v_des),
             (2 * args.tune_seconds / 3, -args.v_des * 2 / 3)]
    g = dict(GAINS)
    if args.start_gains:
        g.update(json.loads(Path(args.start_gains).read_text()))

    def sc(gv):
        return score([rollout(env, gv, s, schedule=sched, est=args.est) for s in seeds])

    best = sc(g)
    print(f"[tune2] {args.milestone} seeds={args.seeds} ep={args.tune_seconds}s "
          f"sched=stand/{args.v_des}/{-args.v_des * 2 / 3:.2f}  start {best:.2f}")
    t0 = time.time()
    for it in range(args.passes):
        improved = False
        for k in TUNE_ORDER:
            if k in args.freeze:
                continue
            lo, hi = GAIN_SPACE[k]
            cur, span = g[k], (hi - lo) * (0.5 ** it)
            cand = [np.clip(cur + d * span, lo, hi) for d in (-0.5, -0.2, 0.2, 0.5)]
            if it == 0:
                cand += [lo, 0.5 * (lo + hi), hi]
            for v in cand:
                if abs(v - cur) < 1e-9:
                    continue
                s = sc(dict(g, **{k: float(v)}))
                if s > best + 1e-6:
                    best, g[k], improved = s, float(v), True
            print(f"  p{it} {k:>10s} = {g[k]:+.4f}   score {best:7.2f}   ({time.time() - t0:.0f}s)")
        if not improved:
            break
    res = [rollout(env, g, s, schedule=sched, est=args.est) for s in seeds]
    print(f"[tune2] final {best:.2f}  t_alive {np.mean([r['t_alive'] for r in res]):.2f}s  "
          f"survived {sum(r['survived'] for r in res)}/{len(res)}")
    out = Path(args.save_gains or GAINS_FILE)
    out.write_text(json.dumps(g, indent=2, sort_keys=True))
    print(f"[tune2] wrote {out}")
    return g


# ------------------------------------------------------------------------------------------------
def plot_stats(tel, gains, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tel["t"]
    fig, ax = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle(title, fontsize=13)

    a = ax[0, 0]
    a.plot(t, tel["vx"], lw=1.0, label="v_x true")
    a.plot(t, tel["vx_hat"], lw=0.8, alpha=.8, label="v_x leg-odometry")
    a.plot(t, tel["v_cmd"], "k--", lw=1.2, label="command")
    a.set_ylabel("m/s"); a.set_title("forward speed vs the command schedule")
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[0, 1]
    a.plot(t, np.degrees(np.arcsin(np.clip(tel["pitch"], -1, 1))), lw=1.0, label="pitch")
    a.plot(t, np.degrees(np.arcsin(np.clip(tel["roll"], -1, 1))), lw=1.0, label="roll")
    a.plot(t, (tel["base_z"] - tel["base_z"][0]) * 100.0, lw=0.8, label="base dz (cm)")
    a.set_ylabel("deg / cm"); a.set_title("attitude + ride height"); a.legend(fontsize=8)
    a.grid(alpha=.3)

    a = ax[1, 0]
    for i, (lbl, off) in enumerate((("L", 0), ("R", 1.2))):
        a.fill_between(t, off, off + 0.45, where=tel["state"][:, i] == SWING, step="mid",
                       alpha=.35, label=f"{lbl} commanded swing")
        a.fill_between(t, off + 0.5, off + 0.95, where=tel["contact"][:, i], step="mid",
                       alpha=.75, label=f"{lbl} measured contact")
    dem = np.asarray(tel["demand"], bool)
    if dem.any():
        a.plot(t[dem], np.full(dem.sum(), 2.35), "r|", ms=8, label="ICP demand")
    a.set_yticks([]); a.set_title("gait diagram + step-on-demand events")
    a.legend(fontsize=7, ncol=3); a.grid(alpha=.3)

    a = ax[1, 1]
    a.plot(t, tel["s_icp"], lw=1.0, label="ICP offset from support")
    a.axhline(gains["icp_margin"], ls=":", c="r", lw=0.8, label="step-demand margin")
    a.axhline(-gains["icp_margin"], ls=":", c="r", lw=0.8)
    a.plot(t, tel["x_td"][:, 0] - gains["x0"], lw=0.7, alpha=.7, label="x_td - x0 (L)")
    a.plot(t, tel["trim"], lw=0.7, alpha=.7, label="stance pitch trim")
    a.set_ylabel("m"); a.set_title("capture point vs the step trigger")
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[2, 0]
    a.plot(t, tel["q_des"][:, THIGH_L], lw=1.0, label="thigh_L commanded (with lead)")
    a.plot(t, tel["q_act"][:, THIGH_L], lw=1.0, label="thigh_L actual")
    a.set_ylabel("rad"); a.set_xlabel("s")
    a.set_title(f"drive tracking through the 3 Hz pole (lead_s={gains['lead_s']:.3f} s)")
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[2, 1]
    st = tel["state"]
    for i, mk in ((0, "o"), (1, "s")):
        td = np.flatnonzero((st[:-1, i] == SWING) & (st[1:, i] == STANCE)) + 1
        if td.size:
            a.plot(t[td], tel["dx"][td, i] - gains["x0"], mk + "-", ms=3, lw=.7,
                   label=f"touchdown dx - x0 ({'LR'[i]})")
    a.axhline(0, c="k", lw=.6)
    a.set_ylabel("m relative to balanced stance"); a.set_xlabel("s")
    a.set_title("where the feet actually land"); a.legend(fontsize=8); a.grid(alpha=.3)

    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


def print_stats(res, tel, gains, est):
    print("\n" + "=" * 78)
    print("CAPTURE-STEP CONTROLLER v2 — debug stats")
    print("=" * 78)
    n = len(res)
    print(f"episodes            {n}   survived {sum(r['survived'] for r in res)}/{n}")
    print(f"time alive          {np.mean([r['t_alive'] for r in res]):6.2f} s "
          f"+/- {np.std([r['t_alive'] for r in res]):.2f}   "
          f"(min {min(r['t_alive'] for r in res):.2f})")
    print(f"distance            {np.mean([r['dist'] for r in res]):+6.2f} m")
    print(f"steps taken         {np.mean([r['n_foot_steps'] for r in res]):5.0f}   "
          f"of which ON-DEMAND {np.mean([r['n_demand'] for r in res]):4.0f} "
          f"({100 * np.mean([r['n_demand'] / max(r['n_foot_steps'], 1) for r in res]):.0f}%)")
    for v_cmd, err in res[0]["seg_err"]:
        print(f"  segment v_cmd {v_cmd:+.2f}:  mean |v err| {err:.3f} m/s")
    if tel is not None:
        con = tel["contact"]
        st = tel["state"]
        td = np.flatnonzero((st[:-1, 0] == SWING) & (st[1:, 0] == STANCE)) + 1
        cad = 1.0 / np.mean(np.diff(tel["t"][td])) if td.size > 2 else float("nan")
        v_e = tel["vx_hat"] - tel["vx"]
        amp_d = tel["q_des"][:, THIGH_L].max() - tel["q_des"][:, THIGH_L].min()
        amp_a = tel["q_act"][:, THIGH_L].max() - tel["q_act"][:, THIGH_L].min()
        print("-" * 78)
        print(f"cadence (L)         {cad:6.2f} Hz    duty L {con[:, 0].mean():.2f} "
              f"R {con[:, 1].mean():.2f}    double support {con.all(axis=1).mean():.2f}")
        print(f"pitch rms           {np.degrees(np.std(tel['pitch'])):6.2f} deg   "
              f"roll rms {np.degrees(np.std(tel['roll'])):.2f} deg")
        print(f"drive delivery      thigh_L {amp_a:.3f}/{amp_d:.3f} = {amp_a / max(amp_d, 1e-9):.2f}")
        print(f"estimator ({est})    bias {np.mean(v_e):+.3f}  rms {np.sqrt(np.mean(v_e ** 2)):.3f} m/s")
    print("-" * 78)
    print("gains: " + "  ".join(f"{k}={v:+.4g}" for k, v in gains.items()))
    print("=" * 78 + "\n")


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", default="m3", choices=["m2", "m3", "m4", "m5", "m6"])
    ap.add_argument("--v-des", type=float, default=0.15)
    ap.add_argument("--schedule", default=None, choices=[None, "demo"],
                    help="demo = stand/fwd/stand/back/stand in one 35 s episode")
    ap.add_argument("--est", default="odom", choices=["odom", "truth"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--dr", type=float, default=0.0)
    ap.add_argument("--pushes", action="store_true")
    ap.add_argument("--gains", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--tune-seconds", type=float, default=18.0)
    ap.add_argument("--freeze", nargs="*", default=[])
    ap.add_argument("--start-gains", default=None)
    ap.add_argument("--save-gains", default=None)
    args = ap.parse_args()

    if args.tune:
        tune(args)
        return

    gains = dict(GAINS)
    gf = Path(args.gains) if args.gains else GAINS_FILE
    if gf.exists():
        gains.update(json.loads(gf.read_text()))
        print(f"[scripted2] gains from {gf}")

    sched = demo_schedule() if args.schedule == "demo" else None
    ep_s = args.seconds or (35.0 if sched else 20.0)
    env = make_env(args.milestone, dr=args.dr, pushes=args.pushes, episode_s=ep_s)
    print(f"[scripted2] {args.milestone}  drive {env.drive_bandwidth_hz:.2f} Hz  ankle "
          f"{env.ankle_mode}  control {1 / env.control_dt:.0f} Hz  dr={args.dr}  "
          f"sched={'demo' if sched else f'const {args.v_des}'}")

    writer = renderer = None
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.distance, cam.elevation = 2.5, -15
    vid_every = max(1, int(round(1.0 / (env.control_dt * VIDEO_FPS))))
    if args.video:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(env.model, 480, 640)
        writer = imageio.get_writer(args.video, fps=int(round(1.0 / (env.control_dt * vid_every))))
    grab = {"on": False, "n": 0, "i": 0}

    def on_step():
        grab["i"] += 1
        if grab["on"] and writer is not None and grab["i"] % vid_every == 0:
            cam.lookat[:] = env.data.qpos[:3]
            renderer.update_scene(env.data, cam)
            writer.append_data(renderer.render())
            grab["n"] += 1

    res, tel = [], None
    for ep in range(args.episodes):
        grab["on"] = (ep == 0 and writer is not None)
        r = rollout(env, gains, ep, schedule=sched, v_des=args.v_des, est=args.est,
                    record=(ep == 0), on_step=on_step)
        if ep == 0:
            tel = r.pop("tel", None)
        res.append(r)
        print(f"  ep{ep}: {r['t_alive']:6.2f} s  {r['dist']:+6.2f} m  "
              f"{r['n_foot_steps']:3d} steps ({r['n_demand']} on-demand)  "
              f"{'FELL' if r['fell'] else 'survived'}")
    if writer is not None:
        writer.close()
        print(f"wrote {args.video} ({grab['n']} frames, "
              f"{grab['n'] * env.control_dt * vid_every:.1f} s at real time)")
    print_stats(res, tel, gains, args.est)
    if args.stats and tel is not None:
        plot_stats(tel, gains, args.stats,
                   f"capture-step v2 — {args.milestone}, 3 Hz drive, rigid ankle, "
                   f"est={args.est}, dr={args.dr}")


if __name__ == "__main__":
    main()
