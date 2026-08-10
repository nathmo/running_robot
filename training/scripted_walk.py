"""A HAND-DESIGNED walking controller for DASH-01 — no neural network, no training.

The question this answers: can classical control walk this robot through the MEASURED drive
(0.8 Hz first-order pole + 25 ms transport delay), or does the plant genuinely require RL?

Structure — deliberately the same three layers every hand-built biped controller has used since
Raibert (1986), and the same three layers the Fourier action space already exposes:

  1. GAIT GENERATOR (feedforward). A fixed clock drives a per-leg foot trajectory in FOOT space
     (fore-aft travel + a swing-phase lift), mapped to cam/thigh by the measured IK table
     (cpg_gait.foot_ik). Not a Fourier series here — a stance ramp plus a smooth swing return, which
     is the same shape with two interpretable parameters instead of fourteen coefficients.

  2. FOOT PLACEMENT (the balance loop). The touchdown point is chosen ONCE per step, at the instant
     the leg enters swing, from the estimated body velocity:

         x_td = 0.5*v*T_stance  +  k_cap*(v - v_des)/omega  +  x_bias,     omega = sqrt(g/h)

     The first term is the neutral point (the foot lands where it will be under the hip at
     mid-stance); the second is the Raibert/capture-point correction. This is the ONLY high-authority
     sagittal channel this robot has: the passive ankle delivers ~0.42 BW against the ~3.5 BW an
     ankle strategy needs, so CoP regulation is not available at any gain.

     WHY ONCE PER SWING AND NOT CONTINUOUSLY. The drive is a 0.8 Hz pole, tau = 199 ms, against an
     inverted-pendulum divergence time constant T_c = sqrt(h/g) = 311 ms at the measured 1.01 m base
     height. A continuous loop on pitch would need gain crossover above 2/T_c ~ 1.0 Hz THROUGH a
     0.8 Hz actuator pole -- roughly 60 deg of phase spent before the controller does anything.
     Latching the target at swing entry instead gives the drive a whole swing phase (~0.35 s at the
     tuned cadence) to converge, which is ~1.8 tau, so ~83% of the commanded placement is delivered
     by touchdown. That converts a bandwidth problem into a lead-time problem, which this drive can
     actually do. It is also why the tuned cadence lands near 1 Hz rather than the 2-4 Hz the RL
     presets reach for: at 2 Hz the measured drive returns 0.38 of the commanded amplitude.

  3. CONTINUOUS TRIM (low gain, by necessity). A pitch PD adds a small SYMMETRIC fore-aft shift to
     both feet, and a roll PD + lateral-velocity term drives the hip_roll abduction pair. These
     cannot be the primary loop (see above) but they cover the intra-step interval.

  Plus one thing an RL policy structurally cannot do: FEEDFORWARD DRIVE INVERSION. The drive is a
  known, measured LTI lag, so the command carries a lead term  q_cmd = q + lead_s * dq/dt.
  lead_s = tau = 0.199 s is exact inversion; the tuner picks a fraction of that, because full
  inversion of the gait fundamental needs command amplitude the joints do not have.

HOW IT REACHES THE PLANT. The controller computes 6 PD targets directly and injects them through
the env's per-step residual channel with the Fourier coefficients, the abduction reflex and the
built-in pitch reflex all zeroed -- so fourier_gait.assemble() returns exactly `nominal6` and
target6 = nominal6 + residual_scale*residual. Nothing about the plant is bypassed: the same drive
filter, the same 25 ms delay, the same domain randomization, torque limits, contact model,
workspace kill and termination rules as every RL run. That is what makes the comparison legitimate.

STATE ESTIMATION is a first-class part of the design, not an afterthought: an RL policy is handed
raw proprioception and learns whatever it needs, but a placement law needs body velocity in metres
per second. --est odom is the honest one (leg odometry from encoder FK + gyro, exactly what the Pi
could run); --est truth is the oracle ablation that separates "the controller is wrong" from "the
estimator is wrong". The debug plot shows both.

MEASURED RESULT (2026-08-10, tuned gains in scripted_gains_m2.json, dr=0, est=odom)

  m2 (X+Z free; pitch/roll/yaw/Y railed)   60.0 s, 5/5 seeds, 16.9 m, 0.28 m/s
    RL reference on the same rung, runs/ladder_m2_s0:  ep_len 4066 +/- 4320 = 20.3 +/- 21.6 s
    So 21 hand-set numbers beat a 100M-step policy on this rung, and beat it on VARIANCE by more
    than on the mean -- the RL run is bimodal, some episodes fall immediately.

  m3 (+ pitch free)                        1.0 s, 0/5 seeds
    RL reference on the same rung, runs/ladder_m3_s0: ep_len 230 +/- 95 = 1.15 s.
    THE SAME WALL. Ruled out as causes, each by direct measurement: the gait (falls identically
    with the gait switched off), all 17 gains (coordinate descent cannot move the score off 1.14 s
    at any setting), the drive (survival is flat at 0.86-1.12 s as the pole is swept 0.8 -> 35 Hz),
    and foot placement (flat over x_bias -0.18 .. 0 m, i.e. the whole reachable correction).
    Whatever kills m3 is a property of the rung, not of the controller, and it defeats RL too.

HONEST LIMITATIONS of the m2 result -- what it does NOT show:
  * m2 rails pitch AND roll, so it does not test balance at all. The est=truth ablation makes this
    explicit: swapping the biased leg odometry for the oracle makes tracking WORSE (0.187 vs 0.279
    m/s), which is only possible if the placement feedback is not doing meaningful work. m2 is a
    treadmill, and the 60 s number is a gait result, not a balance result.
  * The speed command barely functions. Commanded 0.1 -> 0.8 m/s all produce 0.28-0.32 m/s actual.
    The tuner landed on a gait whose natural speed is ~0.3, and the 0.05 m/s "tracking error" at
    v_des=0.3 is that coincidence, not regulation.
  * The gait chatters: measured foot-contact rate 9.7 Hz against a commanded 1.87 Hz cadence. Same
    family as the documented ~6 Hz footfall limit cycle.
  * Robustness is weak, because the gains were tuned on the nominal plant only: 5/5 at dr=0,
    2/5 at dr=0.5, 1/5 at dr=1.0. Re-tuning with --dr 1.0 is the obvious next step.
  * The estimator sees CLEAN encoders and gyro. Sensor noise rides the same dr curriculum, so at
    dr=0 it is off; the odometry has not been tested against the measurement model.

Usage
  # tune (coordinate descent over the gain table), then hold the result
  python training/scripted_walk.py --tune --milestone m3 --seeds 3
  # evaluate + video + debug stats with the tuned gains
  python training/scripted_walk.py --milestone m6 --episodes 5 \
      --video milestones/scripted_m6.mp4 --stats milestones/scripted_m6.png
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
from config import get_config
from env import DashEnv

HIP_ROLL_L, CAM_L, THIGH_L, HIP_ROLL_R, CAM_R, THIGH_R = range(6)
G = 9.81
LEAD_LP_S = 0.05    # low-pass on the lead term's derivative (see the drive-inversion note below)
LEAD_CLIP = 0.35    # hard cap on the lead contribution per joint (rad) -- see __call__
HEIGHT_LP_S = 0.05  # low-pass on the kinematic ride-height measurement
VIDEO_FPS = 50      # mp4 playback rate; control runs at 200 Hz, so frames are decimated 4:1

# ----------------------------------------------------------------------------------------------
# THE GAIN TABLE. Fifteen numbers, all with units and all hand-interpretable -- which is the entire
# point of the exercise. Defaults below are the tuned m6 values (see --tune); GAIN_SPACE gives the
# search interval and is also the documentation of what each knob is allowed to be.
# ----------------------------------------------------------------------------------------------
GAINS = dict(
    f_hz=0.90,        # gait cadence (Hz). Near 1 by design: the 0.8 Hz drive returns 0.60 of the
    #                   commanded amplitude at 1.1 Hz and only 0.38 at 2 Hz.
    duty=0.72,        # stance fraction of the cycle (>0.5 = a walk, with double support)
    k_sweep=1.00,     # stance fore-aft sweep as a multiple of v_des*T_stance (1 = no foot scuffing)
    sweep0=0.02,      # minimum stance sweep (m) so stepping in place still shifts weight
    clearance=0.035,  # swing foot lift (m); measured reachable ceiling is 0.10
    x_bias=0.00,      # static fore-aft trim on the touchdown point (m)
    # RIDE HEIGHT. The stand keyframe's ctrl is NOT a static equilibrium: commanding it exactly
    # (a zero action) collapses this robot to the floor in ~1.5 s on every preset, because the
    # keyframe records the settled qpos while the position servo needs a further ~6 cm of foot
    # extension to hold that qpos against 15 kg. RL policies discover this; a hand-built controller
    # has to be told. z_off is the static bias, k_h/k_hd the loop that keeps it there.
    # MEASURED: the static bias alone stands 3/3 seeds for the full 20 s, while adding the height
    # FEEDBACK at k_h=0.8 drops it to 7.8 s. That is the same fundamental limit the foot-placement
    # note above describes, showing up in a second channel: a continuous loop closed through a
    # 0.8 Hz actuator against a plant that diverges in 311 ms destabilizes rather than helps.
    # Feedforward holds this robot up; feedback does not. k_h/k_hd stay in the tuner's search space
    # so the claim keeps getting re-tested, but they default OFF.
    z_off=-0.07,      # static leg-extension bias (m of toe offset; NEGATIVE = longer leg)
    k_h=0.00,         # ride-height P: m of extra extension per m of height error
    k_hd=0.00,        # ... D, on the measured height rate
    z_clip=0.06,      # authority cap on the height loop's correction about z_off (m)
    k_cap=0.35,       # Raibert/capture gain on (v - v_des), in units of 1/omega
    dx_clip=0.26,     # touchdown-point clip (m); the measured reachable box is +-0.30
    k_pitch=0.30,     # continuous pitch trim: m of symmetric foot shift per unit grav_x
    k_pitchd=0.06,    # ... per rad/s of pitch rate
    trim_clip=0.06,   # authority cap on the continuous sagittal trim (m)
    k_roll=0.60,      # lateral: rad of hip_roll per unit grav_y
    k_rolld=0.10,     # ... per rad/s of roll rate
    k_vy=0.25,        # ... per m/s of lateral velocity
    roll_clip=0.30,   # authority cap on the abduction command (rad)
    lead_s=0.10,      # feedforward drive inversion (s of lead). tau_drive = 0.199 s = exact.
    est_tau=0.06,     # velocity-estimator low-pass time constant (s)
)

# (lo, hi) search interval per gain. Signs are left open where the physical sign convention of the
# axis is not something to assume -- the tuner is allowed to discover it, exactly as the learned
# abduction reflex does.
GAIN_SPACE = dict(
    f_hz=(0.7, 2.0), duty=(0.55, 0.80), k_sweep=(0.4, 1.4), sweep0=(0.0, 0.10),
    clearance=(0.02, 0.09), x_bias=(-0.08, 0.08), k_cap=(0.0, 1.2), dx_clip=(0.10, 0.30),
    z_off=(-0.13, -0.02), k_h=(0.0, 2.5), k_hd=(-0.20, 0.20), z_clip=(0.0, 0.10),
    k_pitch=(-0.8, 0.8), k_pitchd=(-0.25, 0.25), trim_clip=(0.0, 0.15),
    k_roll=(-1.5, 1.5), k_rolld=(-0.4, 0.4), k_vy=(-1.0, 1.0), roll_clip=(0.0, 0.45),
    lead_s=(0.0, 0.20), est_tau=(0.02, 0.15),
)
# Order matters only for coordinate descent's sweep order: cadence/geometry first (they set the
# operating point), then the balance loop, then the trims.
TUNE_ORDER = ["z_off", "k_h", "k_hd", "z_clip",
              "f_hz", "duty", "clearance", "k_sweep", "sweep0", "x_bias",
              "k_cap", "dx_clip", "lead_s",
              "k_pitch", "k_pitchd", "trim_clip",
              "k_roll", "k_rolld", "k_vy", "roll_clip", "est_tau"]

GAINS_FILE = PKG_DIR / "scripted_gains.json"


def smoothstep(u):
    """C1 0->1 ramp on [0,1]; used for the swing-phase fore-aft return so the foot arrives at the
    touchdown point with zero fore-aft velocity (no scuffing, no landing impulse fore-aft)."""
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


class LegOdometry:
    """Body-frame velocity from leg kinematics + gyro -- the estimator the Pi would actually run.

    A foot in stance is stationary in the world, so with p = the stance foot's position in the BASE
    frame (pure forward kinematics from the encoders),

        v_body = -( dp/dt + omega x p )

    WHICH FOOT. The robot has no foot switches, so contact has to be inferred. Taking it from the
    gait clock does not work: the moment the plant stops tracking the clock -- which is the entire
    situation the estimator exists to detect -- the "stance" foot is actually swinging and the
    estimate becomes the swing trajectory, off by more than the speed being measured. Instead the
    reference is the LOWEST toe, which is the planted one by construction and needs no sensor the
    robot does not have. Ties (double support) average.

    Raw per-sample velocities are clipped to VMAX before filtering: a single mis-selected sample
    during a handover is worth several m/s, and an EMA has no way to reject it after the fact.
    """
    VMAX = 3.0

    def __init__(self, tau, dt):
        self.alpha = float(np.exp(-dt / max(tau, 1e-3)))
        self.dt = dt
        self.v = np.zeros(3)
        self.p_prev = None

    def reset(self):
        self.v[:] = 0.0
        self.p_prev = None

    def update(self, p_feet, omega, _unused=None):
        """p_feet: (2,3) toe positions in the BASE frame. omega: gyro (body frame)."""
        p = np.asarray(p_feet, float)
        if self.p_prev is None:
            self.p_prev = p.copy()
            return self.v.copy()
        z = p[:, 2]
        use = [0, 1] if abs(z[0] - z[1]) < 5e-3 else [int(np.argmin(z))]
        raw = np.zeros(3)
        for i in use:
            dp = (p[i] - self.p_prev[i]) / self.dt
            raw += -(dp + np.cross(omega, p[i]))
        raw /= len(use)
        self.p_prev = p.copy()
        raw = np.clip(raw, -self.VMAX, self.VMAX)
        self.v = self.alpha * self.v + (1.0 - self.alpha) * raw
        return self.v.copy()


class ScriptedWalker:
    """The controller. Pure numpy, no env internals beyond reading sensors -- Pi-shareable."""

    def __init__(self, env: DashEnv, gains=None, est="odom", v_des=0.30, start_ramp_s=1.0):
        self.env = env
        self.g = dict(GAINS)
        if gains:
            self.g.update(gains)
        self.est_mode = est
        self.v_des = float(v_des)
        self.dt = env.control_dt
        self.start_ramp_s = float(start_ramp_s)
        self.lut = cpg_gait.load_lut(env.cfg.cpg_lut)
        # The LUT was measured on the m7_freq stance; this plant's `bar` ankle settles somewhere
        # slightly different, so foot_ik(0,0) is NOT zero (~[-0.021, +0.050] rad). Re-origin the
        # table on its own zero so "dx=dz=0" means "the joints this env calls nominal" -- otherwise
        # every episode starts with a constant spurious cam/thigh offset that x_bias then has to
        # cancel by accident.
        self.ik0 = np.asarray(cpg_gait.foot_ik(0.0, 0.0, self.lut), float)
        self.nominal = np.asarray(env._nominal6, float).copy()
        self.res_off = fourier_gait.spec_dim(env.cfg.n_harmonics, env.n_steer)
        self.res_scale = float(env.cfg.residual_scale)
        self.omega = np.sqrt(G / float(env.height_target))
        self._lp = float(np.exp(-self.dt / LEAD_LP_S))
        self._hlp = float(np.exp(-self.dt / HEIGHT_LP_S))
        self.odom = LegOdometry(self.g["est_tau"], self.dt)
        self.reset()

    # ---------- gait clock ----------
    @property
    def T_cycle(self):
        return 1.0 / self.g["f_hz"]

    @property
    def T_stance(self):
        return self.g["duty"] * self.T_cycle

    def reset(self):
        self.phase = 0.0
        self.t = 0.0
        # x_td_cur = the touchdown point this leg's CURRENT stance is sweeping from;
        # x_td_next = the point its CURRENT swing is aiming at (latched at swing entry).
        # SEEDED AT x_bias, not at zero. The first stance of an episode never passes through a
        # swing-entry latch, so a zero seed pins it at dx = 0 for the whole first step no matter
        # what x_bias says -- and on this robot the nominal stance already puts the toes 0.091 m
        # AHEAD of the CoM, so that first step is exactly where a backward topple gets started.
        xb = float(self.g["x_bias"])
        self.x_td_cur = np.full(2, xb)
        self.x_td_next = np.full(2, xb)
        self.x_lo = np.full(2, xb)
        self._was_swing = np.zeros(2, bool)
        self.q_prev = self.nominal.copy()
        self.dq_filt = np.zeros(6)
        # ride-height reference: the standing height the plant actually settles at, captured on the
        # first call so it tracks per-episode plant draws rather than a hard-coded constant
        self.h_ref = float(-np.min(self._toes_body()[:, 2]))
        self.h_filt = self.h_ref
        self.h_prev = self.h_ref
        self.lift = 0.0
        self.odom = LegOdometry(self.g["est_tau"], self.dt)
        self.odom.reset()
        self.v_hat = np.zeros(3)
        self.dbg = {}

    # ---------- sensing ----------
    def _toes_body(self):
        """Toe positions in the base frame. In sim this reads the geom; on hardware it is forward
        kinematics from the six encoders -- the same quantity either way."""
        e = self.env
        R = e.data.xmat[e.base_id].reshape(3, 3)
        base = e.data.xpos[e.base_id]
        return np.array([R.T @ (e.data.geom_xpos[g] - base) for g in e.foot_gids])

    IK_PROBE = 0.02      # finite-difference step for the local lift Jacobian (m)
    JZ_TRUST = 0.10      # |dx| band over which that Jacobian is trustworthy (see _ik)

    def _ik(self, dx, dz):
        """Measured foot IK, re-origined, and EXTENDED BELOW THE TABLE.

        cpg_foot_lut.npz only covers dz in [0, 0.10] -- lift, because the CPG only ever needed to
        pick a foot up. Holding ride height needs the opposite sign (a LONGER leg than nominal), so
        for dz < 0 we take the table's value at dz = 0 and extrapolate along the locally measured
        lift Jacobian. Linear extrapolation is honest over the ~7 cm involved: the column measured
        at the stance pose is [-2.99, +6.20] rad per m and varies slowly with dx.
        """
        j0 = np.asarray(cpg_gait.foot_ik(dx, 0.0, self.lut), float)
        if dz >= 0.0:
            return np.asarray(cpg_gait.foot_ik(dx, dz, self.lut), float) - self.ik0
        # The lift Jacobian is probed at a CLAMPED dx. Measured on the table, d(cam,thigh)/d(dz)
        # holds near (-3.0, +6.2) rad/m for dx in [-0.10, +0.10] and then falls apart: +2.85 at
        # dx=-0.14, SIGN-FLIPPED to -3.49 at -0.18, and ~0 (no lift authority at all) at -0.26.
        # That is the redundant 4-bar's branch choice degrading in the rear workspace, not real
        # kinematics, and probing there makes the leg shorten when told to extend. Since the lift
        # direction genuinely varies little with fore-aft position on a near-straight leg, take it
        # from the well-conditioned band.
        dxp = float(np.clip(dx, -self.JZ_TRUST, self.JZ_TRUST))
        jp = np.asarray(cpg_gait.foot_ik(dxp, 0.0, self.lut), float)
        jz = (np.asarray(cpg_gait.foot_ik(dxp, self.IK_PROBE, self.lut), float) - jp) / self.IK_PROBE
        return (j0 - self.ik0) + jz * dz

    def _leg_phase(self, i):
        return self.phase if i == 0 else (self.phase + np.pi) % (2.0 * np.pi)

    def _in_stance(self, i):
        return self._leg_phase(i) < 2.0 * np.pi * self.g["duty"]

    # ---------- the control law ----------
    def _placement(self, v_x, sweep):
        """Raibert / capture-point touchdown target, in metres of fore-aft toe offset.

        Neutral term is HALF THE COMMANDED SWEEP, not 0.5*v_measured*T_stance. Same quantity when
        the robot is tracking (sweep = v*T_st is the no-scuff condition), but computing it from the
        command keeps the stance arc centred under the hip from the very first step -- seeded from
        the measured velocity it starts at zero, plants the whole first stance behind the CoM, and
        pitches the robot over before the estimator has converged.
        """
        g = self.g
        x = 0.5 * sweep + g["k_cap"] * (v_x - self.v_des) / self.omega + g["x_bias"]
        return float(np.clip(x, -g["dx_clip"], g["dx_clip"]))

    def _foot_xz(self, i, sweep):
        """(dx, dz) of leg i's toe, relative to the nominal stance toe, at the current phase."""
        g = self.g
        th = self._leg_phase(i)
        thr = 2.0 * np.pi * g["duty"]
        if th < thr:                                   # STANCE: ramp back at ground speed
            s = th / max(thr, 1e-6)
            dx = self.x_td_cur[i] + (self.x_lo[i] - self.x_td_cur[i]) * s
            dz = 0.0
        else:                                          # SWING: smooth return to the latched target
            u = (th - thr) / max(2.0 * np.pi - thr, 1e-6)
            dx = self.x_lo[i] + (self.x_td_next[i] - self.x_lo[i]) * smoothstep(u)
            dz = self.lift * (0.5 - 0.5 * np.cos(2.0 * np.pi * u))
        return dx, dz

    def __call__(self):
        e, g = self.env, self.g
        # ---- sense (everything here is an onboard measurement) ----
        grav = e._gravity_body()
        gyro = e._ang_vel_body()
        pitch, roll = float(grav[0]), float(grav[1])
        pitch_rate, roll_rate = float(gyro[1]), float(gyro[0])
        stance = np.array([self._in_stance(0), self._in_stance(1)])
        toes = self._toes_body()
        v_odom = self.odom.update(toes, gyro)
        self.v_hat = e._vel_body().copy() if self.est_mode == "truth" else v_odom
        v_x, v_y = float(self.v_hat[0]), float(self.v_hat[1])

        # ---- soft start: fade the stride in over start_ramp_s so the very first swing does not
        # yank a foot out from under a standing robot ----
        ramp = 1.0 if self.start_ramp_s <= 0 else min(1.0, self.t / self.start_ramp_s)
        self.lift = ramp * g["clearance"]

        # ---- ride-height loop. h is the base's height above the LOWEST stance toe, measured along
        # the body z axis -- pure encoder forward kinematics, no altimeter, so it is available on
        # hardware exactly as it is here. ----
        # min over BOTH toes, not over the stance set: the swing foot is by construction the raised
        # one, so the minimum picks the planted leg automatically and stays CONTINUOUS across the
        # stance handover. Selecting by the stance flag instead steps the signal at every touchdown,
        # and that step goes straight through the lead term as a multi-radian command spike.
        h = float(-np.min(toes[:, 2]))
        self.h_filt = self._hlp * self.h_filt + (1.0 - self._hlp) * h
        hdot = (self.h_filt - self.h_prev) / self.dt
        self.h_prev = self.h_filt
        z_cmd = g["z_off"] + float(np.clip(-g["k_h"] * (self.h_ref - self.h_filt) + g["k_hd"] * hdot,
                                           -g["z_clip"], g["z_clip"]))
        z_cmd *= ramp        # fade the extension in rather than snapping the legs straight at t=0

        # ---- 2. foot placement, latched ONCE at swing entry ----
        sgn = 1.0 if self.v_des >= 0 else -1.0
        sweep = ramp * (g["k_sweep"] * self.v_des * self.T_stance + sgn * g["sweep0"])
        for i in (0, 1):
            sw = not self._in_stance(i)
            if sw and not self._was_swing[i]:                  # liftoff: choose the next foothold
                self.x_td_next[i] = self._placement(v_x, sweep)
                self.x_lo[i] = self.x_td_cur[i] - sweep
            if (not sw) and self._was_swing[i]:                # touchdown: the target becomes current
                self.x_td_cur[i] = self.x_td_next[i]
            self._was_swing[i] = sw

        # ---- 1. gait generator + 3. continuous trims ----
        trim = float(np.clip(g["k_pitch"] * pitch + g["k_pitchd"] * pitch_rate,
                             -g["trim_clip"], g["trim_clip"]))
        u_lat = float(np.clip(g["k_roll"] * roll + g["k_rolld"] * roll_rate + g["k_vy"] * v_y,
                              -g["roll_clip"], g["roll_clip"]))
        q = self.nominal.copy()
        dxz = []
        for i in (0, 1):
            dx, dz = self._foot_xz(i, sweep)
            dx = float(np.clip(dx + trim, -g["dx_clip"], g["dx_clip"]))
            dz = dz + z_cmd                       # the swing lift rides ON TOP of the extension
            dxz.append((dx, dz))
            dcam, dthigh = self._ik(dx, dz)
            if i == 0:
                q[CAM_L] += dcam
                q[THIGH_L] += dthigh
            else:                                              # right leg: mirrored axes
                q[CAM_R] -= dcam
                q[THIGH_R] -= dthigh
        q[HIP_ROLL_L] += u_lat
        q[HIP_ROLL_R] -= u_lat

        # ---- feedforward drive inversion: the measured 0.8 Hz pole is a KNOWN lag, so lead it ----
        # The derivative is low-passed at LEAD_LP_S before use. The IK table is piecewise bilinear
        # over an 81x25 grid, so dq/dt carries step-sized quantization noise that a raw lead gain of
        # lead_s/dt = 20 turns into a 1.2 rad command spike -- 4x the whole gait amplitude. The
        # filter is 4x faster than the gait fundamental, so it costs the lead nothing.
        # It is also HARD CAPPED at LEAD_CLIP. An inverse filter is an amplifier, and an amplifier
        # with no bound will happily emit a 4 rad command off one bad sample; the gait's own
        # amplitude is ~0.5 rad, so anything past a third of a radian of lead is a fault, not
        # compensation.
        dq = (q - self.q_prev) / self.dt
        self.dq_filt = self._lp * self.dq_filt + (1.0 - self._lp) * dq
        q_cmd = q + np.clip(g["lead_s"] * self.dq_filt, -LEAD_CLIP, LEAD_CLIP)
        self.q_prev = q.copy()

        # ---- pack into the env's residual channel (the gait spec stays identically zero) ----
        act = np.zeros(e.action_dim, np.float32)
        act[self.res_off:self.res_off + 6] = np.clip(
            (q_cmd - self.nominal) / self.res_scale, -1.0, 1.0)

        # ---- advance the clock, record debug ----
        self.phase = (self.phase + 2.0 * np.pi * g["f_hz"] * self.dt) % (2.0 * np.pi)
        self.t += self.dt
        self.dbg = dict(v_hat=self.v_hat.copy(), pitch=pitch, roll=roll, stance=stance.copy(),
                        dx=np.array([dxz[0][0], dxz[1][0]]), dz=np.array([dxz[0][1], dxz[1][1]]),
                        x_td=self.x_td_next.copy(), trim=trim, u_lat=u_lat, z_cmd=z_cmd,
                        h=self.h_filt,
                        q_des=q.copy(), q_cmd=q_cmd.copy(), phase=self.phase)
        return act


# ----------------------------------------------------------------------------------------------
# plant setup + rollout
# ----------------------------------------------------------------------------------------------
def make_env(milestone="m6", dr=0.0, pushes=False, episode_s=None, res_scale=1.5, drive_hz=None):
    """Build the measured-plant env (200 Hz, 0.8 Hz drive + 25 ms, bar ankle) with the gait
    generator DISABLED, so the scripted controller owns the 6 targets outright.

    Zeroing cam_amp/thigh_amp/pitch_clip is what makes fourier_gait.assemble() return exactly
    `nominal6` for the all-zero gait spec this controller emits. Nothing else about the plant is
    touched -- drive filter, transport delay, torque limits, contact, workspace kill and the
    termination rules are all the ones every RL run in runs_dl/ was graded on.
    """
    cfg = get_config(f"walk_fwd_{milestone}")
    cfg.residual_scale = float(res_scale)   # direct rad authority on all six PD targets
    cfg.cam_amp = cfg.thigh_amp = 0.0       # no Fourier waveform
    cfg.pitch_clip = 0.0                    # no built-in pitch reflex (we bring our own)
    cfg.reflex_kp_scale = cfg.reflex_kd_scale = cfg.reflex_bias_scale = 0.0
    if not pushes:
        cfg.push_interval_s = 0.0
        cfg.trip_prob = 0.0
    if episode_s:
        cfg.episode_s = float(episode_s)
    if drive_hz:
        # COUNTERFACTUAL DRIVE. Not a plant we own -- an instrument. Sweeping the actuator pole
        # while holding the controller, the gait and the milestone fixed is the only way to
        # separate "this controller cannot balance" from "no controller can balance through a
        # 0.8 Hz pole", and the same sweep applies to the RL policies for free.
        cfg.drive_bandwidth_hz = float(drive_hz)
    env = DashEnv(cfg)
    env.set_dr_scale(dr)                    # 0 = nominal plant + clean sensors
    return env


def rollout(env, gains, seed, v_des=0.30, est="odom", record=False, on_step=None):
    """One episode. Returns a metrics dict (+ per-step telemetry when record=True)."""
    ctl = ScriptedWalker(env, gains, est=est, v_des=v_des)
    obs, _ = env.reset(seed=int(seed))
    if env.command_mode:
        env.set_command(v_des, 0.0)
    ctl.reset()
    x0 = float(env.data.qpos[0])
    tel = {k: [] for k in ("t", "vx", "vy", "vx_hat", "vy_hat", "pitch", "roll", "contact",
                           "stance", "dx", "dz", "x_td", "trim", "u_lat", "q_des", "q_act",
                           "phase", "base_z", "z_cmd", "h")}
    n, v_err, terminated = 0, [], False
    while True:
        act = ctl()
        if record:
            d = ctl.dbg
            v_true = env._vel_body()
            tel["t"].append(ctl.t)
            tel["vx"].append(float(v_true[0])); tel["vy"].append(float(v_true[1]))
            tel["vx_hat"].append(float(d["v_hat"][0])); tel["vy_hat"].append(float(d["v_hat"][1]))
            tel["pitch"].append(d["pitch"]); tel["roll"].append(d["roll"])
            tel["contact"].append(env._foot_contacts().copy())
            tel["stance"].append(d["stance"]); tel["dx"].append(d["dx"]); tel["dz"].append(d["dz"])
            tel["x_td"].append(d["x_td"]); tel["trim"].append(d["trim"]); tel["u_lat"].append(d["u_lat"])
            tel["z_cmd"].append(d["z_cmd"]); tel["h"].append(d["h"])
            tel["q_des"].append(d["q_cmd"]); tel["phase"].append(d["phase"])
            tel["q_act"].append(env.data.qpos[env.act_qadr].copy())
            tel["base_z"].append(float(env.data.qpos[2]))
        obs, r, terminated, truncated, info = env.step(act)
        v_err.append(abs(float(env._vel_body()[0]) - v_des))
        n += 1
        if on_step is not None:
            on_step()
        if terminated or truncated:
            break
    dist = float(env.data.qpos[0]) - x0
    t_alive = n * env.control_dt
    out = dict(t_alive=t_alive, dist=dist, fell=bool(terminated), steps=n, v_des=v_des,
               v_mean=dist / max(t_alive, 1e-9), v_err=float(np.mean(v_err)),
               survived=bool(not terminated))
    if record:
        out["tel"] = {k: np.array(v) for k, v in tel.items() if v}
    return out


def score(res):
    """Tuning objective: seconds upright, TRIPLED when the commanded speed is actually tracked.

    Deliberately not `t_alive + w*distance`. That form was tried first and it is exploitable in
    exactly the way a badly-shaped RL reward is: at m2 the pitch and roll rails mean running away
    costs nothing, so the tuner drove the robot to 1.6 m/s against a 0.3 m/s command and banked the
    distance term. Scaling survival by tracking quality removes the incentive -- distance is now a
    consequence of tracking the command for a long time, not a separately payable quantity.
    """
    v_ref = max(abs(res[0].get("v_des", 0.3)), 0.25)
    return float(np.mean([r["t_alive"] * (1.0 + 2.0 * max(0.0, 1.0 - r["v_err"] / v_ref))
                          for r in res]))


def evaluate(gains, milestone, seeds, v_des, est, dr, pushes, episode_s, env=None):
    own = env is None
    if own:
        env = make_env(milestone, dr=dr, pushes=pushes, episode_s=episode_s)
    res = [rollout(env, gains, s, v_des=v_des, est=est) for s in seeds]
    return res


# ----------------------------------------------------------------------------------------------
# hand tuning: coordinate descent over the gain table
# ----------------------------------------------------------------------------------------------
def tune(args):
    seeds = list(range(args.seeds))
    env = make_env(args.milestone, dr=args.dr, pushes=args.pushes, episode_s=args.tune_seconds)
    g = dict(GAINS)
    if args.start_gains:
        g.update(json.loads(Path(args.start_gains).read_text()))

    def sc(gv):
        return score([rollout(env, gv, s, v_des=args.v_des, est=args.est) for s in seeds])

    best = sc(g)
    print(f"[tune] milestone={args.milestone} seeds={args.seeds} ep={args.tune_seconds}s "
          f"v_des={args.v_des}  start score {best:.2f}")
    t0 = time.time()
    for it in range(args.passes):
        improved = False
        for k in TUNE_ORDER:
            if k in args.freeze:
                continue
            lo, hi = GAIN_SPACE[k]
            cur = g[k]
            # local bracket around the incumbent, shrinking each pass, plus the interval ends on
            # the first pass so a badly-placed default cannot trap the search in its own basin
            span = (hi - lo) * (0.5 ** it)
            cand = [np.clip(cur + d * span, lo, hi) for d in (-0.5, -0.2, 0.2, 0.5)]
            if it == 0:
                cand += [lo, 0.5 * (lo + hi), hi]
            for v in cand:
                if abs(v - cur) < 1e-9:
                    continue
                trial = dict(g); trial[k] = float(v)
                s = sc(trial)
                if s > best + 1e-6:
                    best, g[k], improved = s, float(v), True
            print(f"  pass{it} {k:>10s} = {g[k]:+.4f}   score {best:7.2f}   "
                  f"({time.time() - t0:.0f}s)")
        if not improved:
            print(f"[tune] pass {it}: no improvement, stopping")
            break
    res = [rollout(env, g, s, v_des=args.v_des, est=args.est) for s in seeds]
    print(f"[tune] final score {best:.2f}  "
          f"t_alive {np.mean([r['t_alive'] for r in res]):.2f}s  "
          f"dist {np.mean([r['dist'] for r in res]):.2f}m  "
          f"survived {sum(r['survived'] for r in res)}/{len(res)}")
    out = Path(args.save_gains or GAINS_FILE)
    out.write_text(json.dumps(g, indent=2, sort_keys=True))
    print(f"[tune] wrote {out}")
    return g


# ----------------------------------------------------------------------------------------------
# debug stats
# ----------------------------------------------------------------------------------------------
def plot_stats(tel, gains, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tel["t"]
    fig, ax = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle(title, fontsize=13)

    # 1 — forward speed: truth vs the onboard estimator vs the command
    a = ax[0, 0]
    a.plot(t, tel["vx"], lw=1.0, label="v_x true")
    a.plot(t, tel["vx_hat"], lw=1.0, label="v_x leg-odometry")
    a.axhline(np.mean(tel["vx"]), ls=":", c="k", lw=0.8, label=f"mean {np.mean(tel['vx']):.2f}")
    a.set_ylabel("m/s"); a.set_title("forward speed + estimator"); a.legend(fontsize=8)
    a.grid(alpha=.3)

    # 2 — attitude
    a = ax[0, 1]
    a.plot(t, np.degrees(np.arcsin(np.clip(tel["pitch"], -1, 1))), lw=1.0, label="pitch")
    a.plot(t, np.degrees(np.arcsin(np.clip(tel["roll"], -1, 1))), lw=1.0, label="roll")
    a.plot(t, (tel["h"] - tel["h"][0]) * 100.0, lw=1.0, label="ride height, kinematic (cm)")
    a.plot(t, tel["z_cmd"] * 100.0, lw=0.8, ls="--", label="leg-extension command (cm)")
    a.set_ylabel("deg / cm"); a.set_title("attitude + ride height"); a.legend(fontsize=8)
    a.grid(alpha=.3)

    # 3 — gait diagram: commanded stance window vs measured contact
    a = ax[1, 0]
    for i, (lbl, off) in enumerate((("L", 0), ("R", 1.2))):
        a.fill_between(t, off, off + 0.45, where=tel["stance"][:, i], step="mid",
                       alpha=.35, label=f"{lbl} commanded stance")
        a.fill_between(t, off + 0.5, off + 0.95, where=tel["contact"][:, i], step="mid",
                       alpha=.75, label=f"{lbl} measured contact")
    a.set_yticks([]); a.set_title("gait diagram — commanded vs measured")
    a.legend(fontsize=7, ncol=2); a.grid(alpha=.3)

    # 4 — the balance loop: latched touchdown target vs the instantaneous capture point
    a = ax[1, 1]
    omega = np.sqrt(G / float(np.mean(tel["h"])))
    a.plot(t, tel["x_td"][:, 0], lw=1.0, label="x_td latched (L)")
    a.plot(t, tel["vx_hat"] / omega, lw=0.8, alpha=.7, label="capture point v/omega")
    a.plot(t, tel["trim"], lw=0.8, label="continuous pitch trim")
    a.plot(t, tel["dx"][:, 0], lw=0.6, alpha=.6, label="dx foot L (commanded)")
    a.set_ylabel("m"); a.set_title("foot placement law"); a.legend(fontsize=8); a.grid(alpha=.3)

    # 5 — what the 0.8 Hz drive actually delivers
    a = ax[1 + 1, 0]
    a.plot(t, tel["q_des"][:, THIGH_L], lw=1.0, label="thigh_L commanded (with lead)")
    a.plot(t, tel["q_act"][:, THIGH_L], lw=1.0, label="thigh_L actual")
    a.set_ylabel("rad"); a.set_xlabel("s")
    a.set_title(f"drive tracking through the 0.8 Hz pole (lead_s={gains['lead_s']:.3f} s)")
    a.legend(fontsize=8); a.grid(alpha=.3)

    # 6 — the m3-diagnostic metric: where the foot lands relative to the base
    a = ax[2, 1]
    td = np.flatnonzero((~tel["stance"][:-1, 0]) & tel["stance"][1:, 0]) + 1
    if td.size:
        a.plot(t[td], tel["dx"][td, 0], "o-", ms=3, lw=.8, label="foot dx at touchdown (L)")
    td_r = np.flatnonzero((~tel["stance"][:-1, 1]) & tel["stance"][1:, 1]) + 1
    if td_r.size:
        a.plot(t[td_r], tel["dx"][td_r, 1], "s-", ms=3, lw=.8, label="foot dx at touchdown (R)")
    a.axhline(0, c="k", lw=.6)
    a.set_ylabel("m ahead of nominal"); a.set_xlabel("s")
    a.set_title("touchdown point vs the base (RL m3 policy landed ~-0.08 m)")
    a.legend(fontsize=8); a.grid(alpha=.3)

    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


def print_stats(res, tel, gains, est):
    print("\n" + "=" * 78)
    print("HAND-DESIGNED CONTROLLER — debug stats")
    print("=" * 78)
    n = len(res)
    print(f"episodes            {n}   survived {sum(r['survived'] for r in res)}/{n}")
    print(f"time alive          {np.mean([r['t_alive'] for r in res]):6.2f} s "
          f"+/- {np.std([r['t_alive'] for r in res]):.2f}   "
          f"(min {min(r['t_alive'] for r in res):.2f}, max {max(r['t_alive'] for r in res):.2f})")
    print(f"distance            {np.mean([r['dist'] for r in res]):6.2f} m "
          f"+/- {np.std([r['dist'] for r in res]):.2f}")
    print(f"mean speed          {np.mean([r['v_mean'] for r in res]):6.3f} m/s   "
          f"tracking error {np.mean([r['v_err'] for r in res]):.3f} m/s")
    print(f"ep_len (ctrl steps) {np.mean([r['steps'] for r in res]):6.0f}")
    if tel is None:
        return
    dt = tel["t"][1] - tel["t"][0]
    con = tel["contact"]
    td = np.flatnonzero((~con[:-1, 0]) & con[1:, 0]) + 1
    cad = 1.0 / np.mean(np.diff(tel["t"][td])) if td.size > 2 else float("nan")
    duty = con.mean(axis=0)
    q_des, q_act = tel["q_des"], tel["q_act"]
    amp_des = q_des[:, THIGH_L].max() - q_des[:, THIGH_L].min()
    amp_act = q_act[:, THIGH_L].max() - q_act[:, THIGH_L].min()
    v_e = tel["vx_hat"] - tel["vx"]
    print("-" * 78)
    print(f"cadence             {cad:6.2f} Hz commanded {gains['f_hz']:.2f} Hz")
    print(f"duty factor         L {duty[0]:.2f}  R {duty[1]:.2f}   commanded {gains['duty']:.2f}")
    print(f"step length         {np.mean([r['v_mean'] for r in res]) / max(cad, 1e-6):6.3f} m")
    print(f"flight fraction     {float(np.mean(~con.any(axis=1))):6.3f} "
          f"  double support {float(np.mean(con.all(axis=1))):.3f}")
    print(f"drive delivery      thigh_L amplitude {amp_act:.3f} / {amp_des:.3f} commanded "
          f"= {amp_act / max(amp_des, 1e-9):.2f}")
    print(f"touchdown dx        {np.mean(tel['dx'][td, 0]) if td.size else float('nan'):+.3f} m "
          f"(RL m3 policy: -0.080 m, behind the CoM)")
    print(f"attitude            pitch rms {np.degrees(np.std(tel['pitch'])):.2f} deg   "
          f"roll rms {np.degrees(np.std(tel['roll'])):.2f} deg")
    print(f"estimator ({est:>5s})   v_x bias {np.mean(v_e):+.3f} m/s   rms {np.sqrt(np.mean(v_e ** 2)):.3f} m/s")
    print("-" * 78)
    print("gains: " + "  ".join(f"{k}={v:+.4g}" for k, v in gains.items()))
    print("=" * 78 + "\n")


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", default="m6", choices=["m2", "m3", "m4", "m5", "m6"],
                    help="base-DOF rail: m2 = pitch+roll locked ... m6 = fully free")
    ap.add_argument("--v-des", type=float, default=0.30, help="commanded forward speed (m/s)")
    ap.add_argument("--est", default="odom", choices=["odom", "truth"],
                    help="odom = onboard leg odometry (honest); truth = oracle ablation")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=None, help="episode length override")
    ap.add_argument("--dr", type=float, default=0.0, help="domain-randomization width 0..1")
    ap.add_argument("--pushes", action="store_true", help="enable shoves + trips")
    ap.add_argument("--gains", default=None, help="json gain file (default: scripted_gains.json)")
    ap.add_argument("--video", default=None)
    ap.add_argument("--stats", default=None, help="png path for the debug plot")
    ap.add_argument("--viewer", action="store_true")
    # tuning
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--tune-seconds", type=float, default=12.0)
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
        print(f"[scripted] gains from {gf}")

    env = make_env(args.milestone, dr=args.dr, pushes=args.pushes, episode_s=args.seconds)
    print(f"[scripted] {args.milestone}  drive {env.drive_bandwidth_hz:.2f} Hz "
          f"(action_filter {env.cfg.action_filter:.4f}, delay {env.cfg.action_delay_steps} steps) "
          f"control {1 / env.control_dt:.0f} Hz  dr={args.dr}")

    if args.viewer:
        from mujoco import viewer as mjviewer
        ctl = ScriptedWalker(env, gains, est=args.est, v_des=args.v_des)
        env.reset(seed=0)
        env.set_command(args.v_des, 0.0)
        ctl.reset()
        with mjviewer.launch_passive(env.model, env.data) as v:
            while v.is_running():
                _, _, term, trunc, _ = env.step(ctl())
                v.sync()
                time.sleep(env.control_dt)
                if term or trunc:
                    env.reset(); env.set_command(args.v_des, 0.0); ctl.reset()
        return

    # ---- video: stream frames of episode 0 with a follow cam (same recipe as evaluate.py) ----
    writer = renderer = None
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.distance, cam.elevation = 2.5, -15
    # Decimate to ~VIDEO_FPS so the mp4 plays back at REAL TIME. Writing one frame per control step
    # would be a 200 fps file: correct, unplayable, and 4x the frames.
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
        r = rollout(env, gains, ep, v_des=args.v_des, est=args.est,
                    record=(ep == 0), on_step=on_step)
        if ep == 0:
            tel = r.pop("tel", None)
        res.append(r)
        print(f"  ep{ep}: {r['t_alive']:6.2f} s  {r['dist']:6.2f} m  "
              f"{r['v_mean']:+.3f} m/s  {'FELL' if r['fell'] else 'survived'}")
    if writer is not None:
        writer.close()
        print(f"wrote {args.video} ({grab['n']} frames, "
              f"{grab['n'] * env.control_dt * vid_every:.1f} s at real time)")
    print_stats(res, tel, gains, args.est)
    if args.stats and tel is not None:
        plot_stats(tel, gains, args.stats,
                   f"scripted walk — {args.milestone}, v_des={args.v_des} m/s, "
                   f"0.8 Hz drive, est={args.est}, dr={args.dr}")


if __name__ == "__main__":
    main()
