"""The deployed 100 Hz BALANCE control law (v3 bundle, BalanceRL/): sensors -> observation -> the MIT
frame. A line-by-line mirror of `BalanceRL/env.py BalanceEnv._step_one` with the physics removed.

There is no gait, no clock, no latch and no task channel. The policy reads its own proprioception
and writes, per joint and per tick, a position target, Kp and Kd:

    frame   = [pos - q_stance 6, vel 6, tau 6, grav 3, gyro 3, previous action 18] x obs_scales
    obs     = the last 10 frames, 20 ms apart (hist_idx), oldest first, then the SLOW BLOCK:
              leaky integrals of the measured signals: [g_xy (tau slow) 2, g_xy (tau mid) 2,
              w_xy 2, q - q_stance 6, tau/peak 6].
              They are what lets the policy infer the DC lean that cancels a CoM offset -- the
              calibration error is as large as the standing basin, so one frame cannot show it.
    action  = clip(mean of the policy, -1, 1)                                  (18)
    target  = clip(nominal + q_scale * a[0:6] + reflex, q_lo, q_hi) -> no-load slew cap
    reflex  = the stabilising PRIOR the policy was trained on top of, in joint space, from the
              newest frame's measured gravity and gyro: thigh -/+ (kp_p g_x + kd_p w_y), hip roll
              -/+ (kp_r g_y + kd_r w_x), each clipped to +-reflex_clip_rad. It is NOT a safety
              feature and not tunable here: it is part of the trained control law, so its gains
              come from the bundle and nothing else may touch them.
    kp      = kp0 (kp_hi/kp0)^a  (a >= 0)  |  kp0 (kp0/kp_lo)^a  (a < 0)        a = a[6:12]
    kd      = the same map on a[12:18] with kd0, kd_lo, kd_hi

NO TORQUE FEED-FORWARD. The drives' torque span is unidentified (mit.py rule 2), so the policy was
trained without one and the wire carries tau_ff = 0 exactly, as for every other bundle.

NO SOFTWARE DELAY. The sim delays the COMMAND inside the plant (drive.live_command), which models the
real 12 ms CAN transport the robot already has -- the same as v2.

The ORDER inside step() is the sim's: the newest frame carries the PREVIOUS action (the one that
produced the state being measured), the policy acts on the stacked history, and the action it
returns becomes the previous action of the next frame. start() fills the whole history with one
frame built from the measured pose with zero velocity, zero torque and a zero previous action,
exactly as `_reset_one` does.

The interface is the v2 controller's (start / step -> an object with target, kp, kd, action,
vel_est, ...), so the runner and the webui daemon drive it with the same code. It has no command:
`command_kind` is "none", set_run/set_speed refuse, and there is nothing for the operator to press
but the stop.
"""
import numpy as np


class CommandBalance:
    __slots__ = ("target", "kp", "kd", "action", "phase", "freq", "vel_est", "saturated",
                 "target_prefilter", "run_flag", "speed_cmd", "commit", "spec", "residual",
                 "brake_frac", "reflex")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def gain_map(a, k0, k_lo, k_hi):
    """a in [-1, 1] -> gain; a = 0 is k0 exactly (BalanceRL/env.py gain_map, numpy)."""
    a = np.asarray(a, np.float64)
    up = k0 * (k_hi / k0) ** np.clip(a, 0.0, 1.0)
    dn = k0 * (k0 / k_lo) ** np.clip(a, -1.0, 0.0)
    return np.where(a >= 0.0, up, dn)


class PolicyControllerBalance:
    command_kind = "none"
    stop_flag = False
    brake = None
    resync_active = False
    v_max = v_min = 0.0
    cmd_slew_mps2 = 0.0
    v_trained = (0.0, 0.0)

    def __init__(self, bundle, net=None):
        from policy_net import PolicyNet
        b = self.b = bundle
        if getattr(b, "version", 1) != 3:
            raise ValueError("PolicyControllerBalance runs v3 (balance) bundles; this one is v{}"
                             .format(getattr(b, "version", 1)))
        m, a = b.meta, b.a
        self.net = net if net is not None else PolicyNet(b)
        self.nu = int(m["nu"])
        self.action_dim = int(m["action_dim"])
        self.frame_dim = int(m["frame_dim"])
        self.history_len = int(m["history_len"])
        self.hist_idx = np.asarray(a["hist_idx"], int)
        self.hist_raw_len = int(self.hist_idx.max()) + 1
        self.actor_dim = int(m["actor_dim"])
        self.control_dt = float(m["control_dt"])
        if self.frame_dim * self.history_len + int(m["once_dim"]) != self.actor_dim:
            raise ValueError("balance bundle: {} frames x {} + a {}-wide slow block is {}, but "
                             "actor_dim says {}".format(self.history_len, self.frame_dim, m["once_dim"],
                                                        self.frame_dim * self.history_len
                                                        + int(m["once_dim"]), self.actor_dim))
        if self.action_dim != 3 * self.nu or self.frame_dim != 3 * self.nu + 6 + self.action_dim:
            raise ValueError("balance bundle: action {} / frame {} do not fit {} joints".format(
                self.action_dim, self.frame_dim, self.nu))
        self.nominal = np.asarray(a["nominal_ctrl"], float)
        self.default_motor_pos = np.asarray(a["default_motor_pos"], float)
        self.q_lo = np.asarray(a["q_lo"], float)
        self.q_hi = np.asarray(a["q_hi"], float)
        self.q_scale = np.asarray(a["q_scale"], float)
        self.kp0 = np.asarray(a["drive_kp"], float)
        self.kd0 = np.asarray(a["drive_kd"], float)
        g = m["gains"]
        self.kp_lo, self.kp_hi = float(g["kp_lo"]), float(g["kp_hi"])
        self.kd_lo, self.kd_hi = float(g["kd_lo"]), float(g["kd_hi"])
        vl = np.asarray(a["motor_vel_limit"], float)
        self.vel_limit = np.where(vl > 0.0, vl, np.inf)
        self.accel_limit = float(m["motor_accel_limit"])
        self.once_dim = int(m["once_dim"])
        sl = m.get("slow") or {}
        self.a_slow = float(np.exp(-self.control_dt / float(sl["slow_s"])))
        self.a_mid = float(np.exp(-self.control_dt / float(sl["mid_s"])))
        self.a_fast = float(np.exp(-self.control_dt / float(sl["fast_s"])))
        self.tau_peak = np.asarray(a["forcerange"], float)
        self.a_filt = (float(np.exp(-self.control_dt / float(m["action_filter_tau_s"])))
                       if float(m["action_filter_tau_s"]) > 0 else 0.0)
        r = m.get("reflex") or {}
        self.reflex_on = bool(r.get("enable", False))
        self.rk = (float(r.get("kp_pitch", 0.0)), float(r.get("kd_pitch", 0.0)),
                   float(r.get("kp_roll", 0.0)), float(r.get("kd_roll", 0.0)))
        self.reflex_clip = float(r.get("clip_rad", 0.0))
        s = m["obs_scales"]
        self.s_pos, self.s_vel = float(s["motor_pos"]), float(s["motor_vel"])
        self.s_trq, self.s_grav = float(s["motor_torque"]), float(s["gravity"])
        self.s_angv = float(s["ang_vel"])
        self._alloc()

    def _alloc(self):
        self._history = np.zeros((self.hist_raw_len, self.frame_dim), np.float32)
        self._once = np.zeros(self.once_dim, np.float32)      # [g_xy 2, w_xy 2, q err 6, tau 6]
        self._prev_action = np.zeros(self.action_dim, np.float32)
        self._prev_target = self.nominal.copy()
        self._prev_target_vel = np.zeros(self.nu)
        self._primed = False
        self.n_steps = 0

    # ------------------------------------------------------------------ no command channel
    def set_run(self, run):
        return False

    def set_speed(self, v_mps, immediate=False):
        return False, 0.0

    @property
    def run(self):
        return False

    @property
    def braking(self):
        return False

    def start_brake(self):
        return False, "a balance bundle has no brake: it stands still by construction"

    def cancel_brake(self):
        return False

    def zero_heading(self):
        pass

    # ------------------------------------------------------------------ observation
    def _frame(self, motor_pos, motor_vel, motor_tau, grav, gyro):
        f = np.concatenate([
            (np.asarray(motor_pos, float) - self.default_motor_pos) * self.s_pos,
            np.asarray(motor_vel, float) * self.s_vel,
            np.asarray(motor_tau, float) * self.s_trq,
            np.asarray(grav, float) * self.s_grav,
            np.asarray(gyro, float) * self.s_angv,
            self._prev_action,
        ]).astype(np.float32)
        if f.size != self.frame_dim:
            raise ValueError("built a {}-wide frame, bundle says {}".format(f.size, self.frame_dim))
        return f

    def _push(self, frame):
        self._history[:-1] = self._history[1:]
        self._history[-1] = frame

    def obs(self):
        return np.concatenate([self._history[self.hist_idx].reshape(-1), self._once])

    def _update_once(self, motor_pos, motor_tau, grav, gyro):
        """One tick of the slow channels (BalanceRL/env.py BalanceEnv._ema), measured signals only."""
        o, n = self._once, self.nu
        g, w = np.asarray(grav, float), np.asarray(gyro, float)
        o[0:2] = self.a_slow * o[0:2] + (1.0 - self.a_slow) * g[:2]
        o[2:4] = self.a_mid * o[2:4] + (1.0 - self.a_mid) * g[:2]
        o[4:6] = self.a_fast * o[4:6] + (1.0 - self.a_fast) * w[:2]
        o[6:6 + n] = self.a_slow * o[6:6 + n] + (1.0 - self.a_slow) * (
            np.asarray(motor_pos, float) - self.default_motor_pos)
        o[6 + n:6 + 2 * n] = self.a_slow * o[6 + n:6 + 2 * n] + (1.0 - self.a_slow) * (
            np.asarray(motor_tau, float) / self.tau_peak)

    def reflex(self):
        """The prior's joint offsets, from the NEWEST frame (BalanceRL/env.py BalanceEnv.reflex)."""
        n = self.nu
        if not self.reflex_on:
            return np.zeros(n)
        f = self._history[-1]
        g = np.asarray(f[3 * n:3 * n + 3], float) / self.s_grav
        w = np.asarray(f[3 * n + 3:3 * n + 6], float) / self.s_angv
        kp_p, kd_p, kp_r, kd_r = self.rk
        lean = float(np.clip(-(kp_p * g[0] + kd_p * w[1]), -self.reflex_clip, self.reflex_clip))
        roll = float(np.clip(-(kp_r * g[1] + kd_r * w[0]), -self.reflex_clip, self.reflex_clip))
        return np.array([roll, 0.0, lean, -roll, 0.0, -lean])

    @staticmethod
    def _unit(grav):
        grav = np.asarray(grav, float)
        n = np.linalg.norm(grav)
        return grav / n if n > 1e-6 else grav

    # ------------------------------------------------------------------ lifecycle
    def start(self, motor_pos, motor_vel, motor_tau, grav, gyro, exact=False):
        """Start from the robot's CURRENT measured pose: one frame with zero velocity, zero torque
        and a zero previous action fills the whole history (`_reset_one`, whose first frame is the
        noise model applied to a standing robot's zero velocity and zero last torque). The first
        target is slewed from the STANCE command: the caller has crawled the robot there.

        exact=True uses the given velocity and torque instead of zeros -- for the parity test,
        which replays the sim's own noisy first frame."""
        self._alloc()
        n = self.nu
        v = np.asarray(motor_vel, float) if exact else np.zeros(n)
        t = np.asarray(motor_tau, float) if exact else np.zeros(n)
        g = self._unit(grav)
        self._history[:] = self._frame(motor_pos, v, t, g, gyro)
        # the slow channels start AT the first measurement (the sim's _reset_one does the same):
        # gravity and the joint error as measured, the rate and torque integrals at zero
        self._once[:] = 0.0
        self._once[0:2] = np.asarray(g, float)[:2]
        self._once[2:4] = np.asarray(g, float)[:2]
        self._once[6:6 + n] = np.asarray(motor_pos, float) - self.default_motor_pos
        self._primed = True
        return self.obs()

    reset = start

    def step(self, motor_pos, motor_vel, motor_tau, grav, gyro, override_action=None):
        # 1) measurement -> newest frame (carrying the previous action); skipped once after start()
        if self._primed:
            self._primed = False
        else:
            g = self._unit(grav)
            self._push(self._frame(motor_pos, motor_vel, motor_tau, g, gyro))
            self._update_once(motor_pos, motor_tau, g, gyro)
        obs = self.obs()
        # 2) policy: the mean, clipped
        action, vel_est = self.net(obs)
        action = action.astype(np.float32)
        drive = action if override_action is None else np.clip(
            np.asarray(override_action, np.float32), -1.0, 1.0)
        # 3) the MIT frame
        raw = self.nominal + self.q_scale * drive[0:self.nu] + self.reflex()
        tgt = np.minimum(np.maximum(raw, self.q_lo), self.q_hi)
        saturated = bool(np.any(tgt != raw))
        kp = gain_map(drive[self.nu:2 * self.nu], self.kp0, self.kp_lo, self.kp_hi)
        kd = gain_map(drive[2 * self.nu:3 * self.nu], self.kd0, self.kd_lo, self.kd_hi)
        # 4) the one-pole filter on the target, then the no-load slew cap (drive.slew_limit)
        tgt = self.a_filt * self._prev_target + (1.0 - self.a_filt) * tgt
        v_des = (tgt - self._prev_target) / self.control_dt
        if self.accel_limit > 0.0:
            dv = self.accel_limit * self.control_dt
            v_des = np.clip(v_des, self._prev_target_vel - dv, self._prev_target_vel + dv)
        v_des = np.clip(v_des, -self.vel_limit, self.vel_limit)
        target = self._prev_target + v_des * self.control_dt
        self._prev_target, self._prev_target_vel = target.copy(), v_des
        self._prev_action = drive.astype(np.float32)
        self.n_steps += 1
        return CommandBalance(reflex=self.reflex(), target=target, kp=kp, kd=kd, action=action, phase=0.0, freq=0.0,
                              vel_est=vel_est, saturated=saturated, target_prefilter=tgt,
                              run_flag=0.0, speed_cmd=0.0, commit=False, spec=None,
                              residual=drive, brake_frac=0.0)
