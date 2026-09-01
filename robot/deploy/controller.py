"""The deployed 200 Hz control law: sensors -> observation -> action -> joint targets + impedance.

This is a line-by-line mirror of walk_mit/env.py DashEnv.reset/step with the physics removed.
Everything that shapes the observation or the command is reproduced in the same ORDER, because
the order is load-bearing:

  * the observation frame pushed at the end of sim step n carries the phase the NEXT action will be
    assembled at (the phase is advanced BEFORE the frame is built), and prev_action from step n-1
    (the frame is built BEFORE _prev_action is updated). So the newest frame the policy sees at
    tick n holds phase_n and a_{n-2}. Both offsets are reproduced here exactly; getting either
    wrong feeds the network an observation off the training manifold, and the failure mode is a
    policy that looks "almost right" and falls over.
  * the target is EMA-filtered, then clipped to ctrlrange, then slew-limited to the motor velocity
    limit -- in that order, with the filter state carried across ticks.
  * the action rides a delay buffer (action_delay_steps, from the measured 7 ms MIT transport
    delay) before ANY of it is used, impedance gains included: on hardware they travel in the same
    CAN frame, which is why the sim delays them together.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not touch CAN, does not clamp for safety, and does not know about motors, temperatures or
calibration. It is a pure function of (measurements, command) -> (targets, gains), so it can be run
inside MuJoCo and diffed against the torch policy (verify_export.py) with nothing mocked. Safety
lives downstream in safety.py, deliberately: a limiter the controller can see is a limiter the
controller can be tuned against, and this controller is not being tuned.

FRAME CONVENTIONS (identical to the model, see env._gravity_body)
-----------------------------------------------------------------
  motor_pos/vel : MODEL joint angles, radians, actuator order
                  [hip_roll_L, cam_L, thigh_L, hip_roll_R, cam_R, thigh_R]
  motor_tau     : joint torque, N*m, same order and sign convention as the joint angle
  grav          : world DOWN expressed in body axes -- upright is [0, 0, -1].
                  The Sense HAT publishes world UP (up_body), so grav = -up_body.
  gyro          : body angular rate, rad/s, body axes (X fwd, Y left, Z up)
"""
import numpy as np

# vendored BYTE-FOR-BYTE from walk_mit/fourier_gait.py (a hash test in tests/ fails on drift).
# Named for its source rather than `gait` because robot/fixed_gait/gait.py is also on the Pi's
# sys.path -- the webui's paths bootstrap puts it there -- and the shorter name silently imported
# the hand-built demo gait instead of this one.
import fourier_gait as gait


class Command:
    """One tick of controller output. Plain attributes -- this is read at 200 Hz."""
    __slots__ = ("target", "kp", "kd", "action", "applied", "phase", "freq", "vel_est",
                 "target_prefilter", "residual", "saturated")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class PolicyController:
    def __init__(self, bundle, net=None):
        from policy_net import PolicyNet
        self.b = bundle
        self.net = net if net is not None else PolicyNet(bundle)
        m, a = bundle.meta, bundle.a
        self.nu = int(m["nu"])
        self.action_dim = int(m["action_dim"])
        self.frame_dim = int(m["frame_dim"])
        self.history_len = int(m["history_len"])
        self.hist_raw_len = int(m["hist_raw_len"])
        self.hist_idx = np.asarray(a["hist_idx"], int)
        self.task_dim = int(m["task_dim"])
        self.obs_base_vel = bool(m["obs_base_vel"])
        self.control_dt = float(m["control_dt"])
        self.n_steer = int(m["n_steer"])
        self.imp_dim = int(m["imp_dim"])
        self.imp_action_start = int(m["imp_action_start"])
        self.gait_action_dim = int(m["gait_action_dim"])

        self.nominal = np.asarray(a["nominal_ctrl"], float)
        self.default_motor_pos = np.asarray(a["default_motor_pos"], float)
        self.ctrl_lo = np.asarray(a["ctrl_lo"], float)
        self.ctrl_hi = np.asarray(a["ctrl_hi"], float)
        self.kp_base = np.asarray(a["imp_kp_base"], float)
        self.kd_base = np.asarray(a["imp_kd_base"], float)
        self.imp_leg_ix = np.asarray(a["imp_leg_ix"], int)

        vl = np.asarray(a["motor_vel_limit"], float)
        self.vel_limit = np.where(vl > 0.0, vl, np.inf)
        self.accel_limit = float(m["motor_accel_limit"])
        self.vel_accel_limited = bool(m["vel_accel_limited"])

        s = m["obs_scales"]
        self.s_pos, self.s_vel = float(s["motor_pos"]), float(s["motor_vel"])
        self.s_trq, self.s_grav = float(s["motor_torque"]), float(s["gravity"])
        self.s_angv, self.s_bvel = float(s["ang_vel"]), float(s["base_vel"])

        self.filter_a = float(m["action_filter"])
        self.residual_scale = float(m["residual_scale"])
        self.delay_steps = int(m["action_delay_steps"])
        self.gcfg = bundle.cfg_view()
        self.n_harm = int(self.gcfg.n_harmonics)
        self.freq_range = tuple(self.gcfg.gait_freq_hz)
        self.pitch_lp = float(self.gcfg.pitch_reflex_rate_lp)

        self.imp_log_kp = (np.log(float(m["imp_kp_up"])), np.log(float(m["imp_kp_dn"])))
        self.imp_log_kd = (np.log(float(m["imp_kd_up"])), np.log(float(m["imp_kd_dn"])))

        self.cmd_v_norm = float(m["cmd_v_norm"])
        self.cmd_yaw_norm = float(m["cmd_yaw_norm"])
        self.cmd_deadband = float(m["cmd_deadband"])
        self.cmd_yaw_deadband = float(m["cmd_yaw_deadband"])

        self._alloc()

    def _alloc(self):
        self._history = np.zeros((self.hist_raw_len, self.frame_dim), np.float32)
        self._task = np.zeros(self.task_dim, np.float32)
        self._delay_buf = [np.zeros(self.action_dim, np.float32) for _ in range(self.delay_steps)]
        self._filt_target = self.nominal.copy()
        self._prev_cmd_pos = self.nominal.copy()
        self._prev_cmd_vel = np.zeros(self.nu)
        self._phase = 0.0
        self._reflex_prate = 0.0
        # env keeps ONE _prev_action, but the sim's frame is built before it is updated, so the
        # frame lags the action by two steps once you unroll it against a robot loop that measures
        # first. Two explicit slots, so the offset is visible instead of implied.
        self._a_prev1 = np.zeros(self.action_dim, np.float32)   # a_{n-1}
        self._a_prev2 = np.zeros(self.action_dim, np.float32)   # a_{n-2}, the one the frame carries
        self._v_cmd = 0.0
        self._yaw_cmd = 0.0
        self._standing = True
        # start() latches the first measurement as the WHOLE history, exactly as DashEnv.reset
        # does, so the first action sees a stationary robot instead of a buffer of zeros. The
        # first step() must therefore NOT push again: in the sim the first action is computed
        # from a history of frame_0 alone, with no second measurement in it. Pushing here would
        # put the policy one frame ahead of where it was trained on tick 1 of every run.
        self._primed = False
        self.n_steps = 0

    # ------------------------------------------------------------------ command channel
    def set_command(self, v_cmd, yaw_cmd=0.0):
        """Joystick command, m/s and rad/s. Applies the SAME deadband the training sampler used --
        a command below it was snapped to zero in training, so the policy has never been asked to
        resolve one, and passing it through would put the task channel off-manifold."""
        v = float(v_cmd)
        y = float(yaw_cmd)
        if abs(v) < self.cmd_deadband:
            v = 0.0
        if abs(y) < self.cmd_yaw_deadband:
            y = 0.0
        self._v_cmd, self._yaw_cmd = v, y
        self._standing = (v == 0.0 and y == 0.0)

    def _update_task(self):
        self._task[0] = self._v_cmd / self.cmd_v_norm
        self._task[1] = self._yaw_cmd / self.cmd_yaw_norm if self.cmd_yaw_norm else 0.0
        self._task[2] = 1.0 if self._standing else 0.0

    # ------------------------------------------------------------------ observation
    def _proprio(self, motor_pos, motor_vel, motor_tau, grav, gyro, base_vel=None):
        """One measurement frame, in the exact layout of DashEnv._proprio.

        There is no sensor-noise model here: on the robot the measurement IS the noisy one. The
        reflex below reads the same measured gravity that goes into this frame, whereas the sim
        reads a clean copy for the reflex and a corrupted one for the observation. That is an
        inherent sim2real difference, and it sits inside the band this run trained against
        (noise_grav 0.02 plus a 0.02 bias, noise_gyro 0.02 plus bias and random walk)."""
        parts = [(np.asarray(motor_pos, float) - self.default_motor_pos) * self.s_pos,
                 np.asarray(motor_vel, float) * self.s_vel,
                 np.asarray(motor_tau, float) * self.s_trq,
                 np.asarray(grav, float) * self.s_grav,
                 np.asarray(gyro, float) * self.s_angv]
        if self.obs_base_vel:
            if base_vel is None:
                raise ValueError("this bundle was trained with the PRIVILEGED base velocity in "
                                 "the observation (obs_base_vel=True). A robot cannot produce it. "
                                 "Deploy a policy trained with obs_base_vel=False.")
            parts.append(np.asarray(base_vel, float) * self.s_bvel)
        parts.append(np.array([np.sin(self._phase), np.cos(self._phase)]))
        parts.append(self._task)
        parts.append(self._a_prev2)
        f = np.concatenate(parts).astype(np.float32)
        if f.size != self.frame_dim:
            raise ValueError("built a {}-wide observation frame, bundle says {} -- the sensor "
                             "vector widths are wrong".format(f.size, self.frame_dim))
        return f

    def _push_frame(self, frame):
        self._history[:-1] = self._history[1:]
        self._history[-1] = frame

    def _obs(self):
        return self._history[self.hist_idx].reshape(-1)

    # ------------------------------------------------------------------ lifecycle
    def start(self, motor_pos, motor_vel, motor_tau, grav, gyro, v_cmd=0.0, yaw_cmd=0.0):
        """Start (or restart) the control law from the robot's CURRENT measured pose.

        DashEnv.reset fills the whole history with copies of the first frame, so the policy's first
        action sees a stationary robot rather than a buffer of zeros. Same here. _filt_target and
        _prev_cmd_pos start at the STANCE, not at the measured pose -- that is what the sim does,
        and it is also what you want: the first commanded target is the stance the policy was
        trained to hold. The caller is responsible for having already moved the robot near that
        stance (the runner has an approach phase for exactly this); starting the filter at the
        measured pose instead would make the first tick's target depend on where the legs happened
        to be sagging."""
        self._alloc()
        self.set_command(v_cmd, yaw_cmd)
        self._update_task()
        frame = self._proprio(motor_pos, motor_vel, motor_tau, grav, gyro)
        self._history[:] = frame
        self._primed = True
        return self._obs()

    # backwards-compatible alias; `start` is the honest name (it does not reset a robot)
    reset = start

    def step(self, motor_pos, motor_vel, motor_tau, grav, gyro, v_cmd=None, yaw_cmd=None,
             override_action=None):
        """One control tick. Measurements in, joint targets plus per-joint impedance out.

        override_action: use this action INSTEAD of the network's own output for everything
        downstream (delay buffer, gait, impedance, and the prev_action the next observation
        carries), while still reporting what the network wanted in `Command.action`. Two uses,
        both non-production:
          * verify_export.py -- a shadow run driven by the torch action, so the comparison
            measures the port and not the compounding of float32 noise around the prev_action
            feedback loop;
          * blackbox replay -- re-running a recorded action stream through the control law to
            reproduce a fault offline.
        The runner never passes it.
        """
        if v_cmd is not None or yaw_cmd is not None:
            self.set_command(self._v_cmd if v_cmd is None else v_cmd,
                             self._yaw_cmd if yaw_cmd is None else yaw_cmd)
        self._update_task()
        # 1) measurement -> newest history frame (carries phase_n and a_{n-2}; see the class doc).
        #    Skipped exactly once after start(), which already latched this measurement into every
        #    row of the history.
        if self._primed:
            self._primed = False
        else:
            self._push_frame(self._proprio(motor_pos, motor_vel, motor_tau, grav, gyro))
        obs = self._obs()

        # 2) policy. Mean action, clipped -- the robot never samples.
        action, vel_est = self.net(obs)
        action = action.astype(np.float32)
        drive = action if override_action is None else np.asarray(override_action, np.float32)

        # 3) actuation delay: the whole action, gains included, arrives one CAN frame late
        self._delay_buf.append(drive)
        applied = self._delay_buf.pop(0) if self._delay_buf else drive

        # 4) gait reconstruction at the CURRENT phase
        cam_c, thigh_c, freq_raw, reflex, steer, residual = gait.decode(
            applied, self.n_harm, self.n_steer)
        f = gait.frequency(freq_raw, self.freq_range)
        grav = np.asarray(grav, float)
        gyro = np.asarray(gyro, float)
        roll, roll_rate = float(grav[1]), float(gyro[0])
        pitch, pitch_rate = float(grav[0]), float(gyro[1])
        if self.pitch_lp > 0.0:
            self._reflex_prate = (self.pitch_lp * self._reflex_prate
                                  + (1.0 - self.pitch_lp) * pitch_rate)
            pitch_rate = self._reflex_prate
        target = gait.assemble(cam_c, thigh_c, reflex, self._phase, roll, roll_rate,
                               self.nominal, self.gcfg, pitch=pitch, pitch_rate=pitch_rate,
                               steer=steer if self.n_steer else None)
        target = target + self.residual_scale * residual
        target_prefilter = target.copy()

        # 5) per-leg impedance. Exponential map with asymmetric headroom, neutral 0 -> 1.0.
        #    For this bundle the span works out to kp 40-500 N*m/rad and kd 1.0-5.0 N*m*s/rad,
        #    which lands exactly inside the CubeMars force-control Kp 0-500 / Kd 0-5 wire ranges,
        #    so nothing here has to be re-scaled to be commandable. safety.py asserts it anyway.
        if self.imp_dim:
            ia = applied[self.imp_action_start:]
            up_p, dn_p = self.imp_log_kp
            up_d, dn_d = self.imp_log_kd
            kp_leg = np.exp(np.where(ia[0::2] >= 0.0, ia[0::2] * up_p, ia[0::2] * dn_p))
            kd_leg = np.exp(np.where(ia[1::2] >= 0.0, ia[1::2] * up_d, ia[1::2] * dn_d))
            kp = self.kp_base * kp_leg[self.imp_leg_ix]
            kd = self.kd_base * kd_leg[self.imp_leg_ix]
        else:
            kp, kd = self.kp_base.copy(), self.kd_base.copy()

        # 6) the actuation chain the sim runs inside _run_physics, in the same order
        self._filt_target = self.filter_a * self._filt_target + (1.0 - self.filter_a) * target
        # np.clip routes through fromnumeric.clip -> _wrapfunc -> _methods._clip before it reaches
        # the same two ufuncs these call directly. On six-element arrays that dispatch costs more
        # than the arithmetic. Identical semantics, NaN included: maximum/minimum propagate NaN
        # exactly as clip does.
        tgt = np.minimum(np.maximum(self._filt_target, self.ctrl_lo), self.ctrl_hi)
        saturated = bool(np.any(tgt != self._filt_target))
        if self.vel_accel_limited:
            dt = self.control_dt
            v_des = (tgt - self._prev_cmd_pos) / dt
            if self.accel_limit > 0.0:
                dv = self.accel_limit * dt
                v_des = np.minimum(np.maximum(v_des, self._prev_cmd_vel - dv),
                                   self._prev_cmd_vel + dv)
            np.minimum(np.maximum(v_des, -self.vel_limit, out=v_des), self.vel_limit, out=v_des)
            tgt = self._prev_cmd_pos + v_des * dt
            self._prev_cmd_vel = v_des
            self._prev_cmd_pos = tgt.copy()

        # 7) advance the clock and roll the action pipeline (both AFTER the command is formed)
        self._phase = (self._phase + 2.0 * np.pi * f * self.control_dt) % (2.0 * np.pi)
        self._a_prev2 = self._a_prev1
        self._a_prev1 = drive
        self.n_steps += 1

        return Command(target=tgt, kp=kp, kd=kd, action=action, applied=applied,
                       phase=self._phase, freq=float(f), vel_est=vel_est,
                       target_prefilter=target_prefilter, residual=residual,
                       saturated=saturated)
