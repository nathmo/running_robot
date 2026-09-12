"""The deployed 100 Hz v2 control law: sensors -> observation -> action -> joint targets + gains.

This is a line-by-line mirror of `walk_v2/env.py DashEnvV2._step_one` with the physics removed.
Everything that shapes the observation or the command is reproduced in the same ORDER, because the
order is load-bearing. The four places a naive port goes wrong, all of them silent:

  * THE LATCH. 44 of the 50 action dims are a gait SPEC that reaches the generator only on a tick
    whose observation carried `commit = 1`; on every other tick they are thrown away and the spec
    from the last commit is replayed. The clock wrapping is what sets `commit` for the NEXT tick,
    and the flag the policy reads is the last entry of the observation. Latch on every tick and you
    get a policy rewriting its own gait at 100 Hz -- which is exactly the CPG-gaming pathology this
    design was built to stop.
  * THE PHASE IS SHARED. The phase in the newest observation frame and the phase the generator
    assembles at are the SAME number (the sim advances the clock at the end of the tick and writes
    the advanced value into the frame it publishes). One variable here; two would drift by a tick.
  * cos THEN sin. `walk_v2/env.py` writes `[cos phi, sin phi]` into the frame. (The CPU arm's
    V2_CONTRACT.md documents the opposite order for its own implementation -- a known, recorded
    difference between the two arms. This runtime deploys walk_v2 policies, so: cos, sin.)
  * NO SOFTWARE ACTUATION DELAY. v1's controller carried the action on a delay buffer because
    walk_mit delayed the ACTION. v2 delays the COMMAND inside the substep loop
    (`drive.live_command`), which is a model of the real 12 ms CAN transport -- the robot HAS that
    delay in hardware. Re-applying it here would double it.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not touch CAN, does not clamp for safety, and does not know about motors, temperatures or
calibration. It is a pure function of (measurements, command) -> (targets, gains). Safety lives
downstream in safety.py, deliberately.

TWO WAYS TO ASK THIS ROBOT TO STOP, AND ONE OF THEM IS MEASURED TO DROP IT
--------------------------------------------------------------------------
v2's task vector is `[run, distance_to_go]`. `run` is the green light: 1 while the policy is being
asked to travel, 0 for the red-light phases of the stop curriculum and for everything past the
finish line. The obvious deployment of a STOP button is therefore "set run = 0", and that is what
this runtime did until the cluster measured it (walk_v2/README.md, 2026-09-11 17:00):

    [brake] CONTROL (schedule = cruise): upright 512/512   <- policy NOT told it has finished
    [brake] CONTROL (schedule = cruise): upright   3/512   <- same control, told it has finished

Ten training configurations across two months failed to teach this lineage to brake, and that pair
of numbers is why: **the command channel IS the disturbance.** Dropping `run` does not ask the
policy to slow down, it puts it in a state where it accelerates and falls, so no amount of reward
shaping on the response could ever have helped. What works is to brake the BODY without telling the
policy anything: hold `run = 1`, and override the latched gait spec with a fitted open-loop
schedule while the policy keeps contributing its per-tick residual (which is 45-95% of joint motion
on this lineage -- zeroing it removes the stabiliser and everything falls).

So this runtime has two, and they are not interchangeable:

  `start_brake()`  THE BRAKE. Freezes the spec at its current (cruise) value, then drives four
                   channels over a fixed window from a 12-number schedule fitted offline for THIS
                   checkpoint (`walk_v2/tools/brake_search.py`). `run` stays 1 throughout. This is
                   what the panel's STOP button does, and it needs `bundle.meta["brake"]`.
  `set_run(False)` THE FLAG. The literal task input. Kept because it is what the stop curriculum
                   trains and it has to stay testable, but on every checkpoint measured so far it
                   is the fall, not the stop. Nothing in the panel reaches it by default.

A run starts with `run = 0` and no brake -- see `start()`. That is a THIRD thing again: at rest,
before anything has moved, the flag is simply the truthful state of the world.

THE JOYSTICK (objective='joystick'), WHICH HAS NEITHER OF THOSE
----------------------------------------------------------------
The retrained lineage does away with the green light. `task[0]` stops being a flag and becomes the
commanded speed, normalised:

    task[0] = clip(v_cmd / v_max, v_min / v_max, 1)     task[1] = 1 (reserved)

with `v_min` 0 while the trainer clips there -- forward only -- and negative once walking backwards
is trained. Two things follow, and both are why the retrain is worth deploying. The policy has seen
this input at every value in between, so asking for 0 is an operating point rather than a step into
a state it only ever meets at the finish line; and nothing in the task vector is odometry any more
(v2's `task[1]` was clip((line - d)/8, 0, 1), computed in the sim from ground-truth world x, which
this robot can only guess at). `set_speed()` is the whole interface. There is no run flag to press
and no fitted brake to fire: 0 m/s IS the stop, and the panel's slider sits there until someone
moves it.

THE SLIDER IS SLEWED, DELIBERATELY. Dragged from 0 to top speed, a browser delivers a step change
on exactly the input the pair of numbers above says this machine falls over -- and a step is not
what the trainer showed it. So `set_speed()` sets a TARGET and the applied command walks toward it
at `cmd_slew_mps2`: the trainer's own command rate when the bundle records one, otherwise the whole
range in DEFAULT_CMD_SLEW_S seconds. `set_speed(v, immediate=True)` defeats it, for the bench.

THE CLOCK FREE-RUNS ON THE ROBOT
--------------------------------
In simulation the gait clock is nudged toward the measured touchdown phase (`resync_kappa`) each
time a foot lands inside a +-0.15-cycle window. DASH-01 has no foot contact sensor, so nothing here
can produce that event and the clock free-runs. That is a real sim2real difference, and it is a
small one by construction: kappa is held at 0 for the first 3 cycles of every episode, it is
randomised over U[0.3, 0.7] per episode, and the correction it applies is bounded by the window. If
a contact signal ever exists, feed it to `note_contact()` and the machinery below is already the
sim's. `resync_active` says which of the two is running.

FRAME CONVENTIONS (identical to the model, see env._grav_body)
--------------------------------------------------------------
  motor_pos/vel : MODEL joint angles, radians, actuator order
                  [hip_roll_L, cam_L, thigh_L, hip_roll_R, cam_R, thigh_R]
  motor_tau     : joint torque, N*m, same order and sign convention as the joint angle
  grav          : world DOWN expressed in body axes -- upright is [0, 0, -1]. The Sense HAT
                  publishes world UP (up_body), so grav = -up_body. Normalised here, because the
                  sim's sensor model normalises it and the policy has only ever seen a unit vector.
  gyro          : body angular rate, rad/s, body axes (X fwd, Y left, Z up)
"""
import numpy as np

import gait_v2 as gait

TWO_PI = 2.0 * np.pi

# Seconds a 'speed' command takes to cross the WHOLE commandable range when the bundle does not
# record the rate the trainer moved it at. A number, not a philosophy: two seconds is slow enough
# that no drag of the panel's slider is a step input, and fast enough that an operator who wants
# the robot slowed now does not stand there waiting for it.
DEFAULT_CMD_SLEW_S = 2.0


class CommandV2:
    """One tick of controller output. Plain attributes -- this is read at 100 Hz."""
    __slots__ = ("target", "kp", "kd", "action", "phase", "freq", "vel_est", "residual",
                 "spec", "commit", "run_flag", "saturated", "target_prefilter", "brake_frac",
                 "speed_cmd")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class BrakeSchedule:
    """The 12 numbers `walk_v2/tools/brake_search.py` fits per checkpoint, and how they are applied.

    Four channels x three knots, piecewise linear in time across a window of fixed length. Read
    `brake_search.brake_rollout` next to this: the application has to be identical or the schedule
    is being replayed onto a different control law from the one it was fitted to.

      freq_scale   multiplies the cruise spec's frequency entry IN SPEC UNITS, then clips to [-1, 1]
                   (not the frequency in Hz -- the map from spec unit to Hz is not linear in the
                   same way, and the search multiplied the spec entry)
      amp_scale    multiplies the cam AND thigh Fourier series
      o_cam        SET (not scaled) -- the fore-aft foot placement, i.e. the capture step
      o_thigh      SET -- the lean

    Everything else in the spec is the frozen cruise value, and the policy's per-tick residual is
    untouched.

    THE THREE THINGS THAT DECIDE WHETHER A SCHEDULE REPRODUCES (all measured, README 17:00):
      * it is DURATION-specific. Replaying a 12 s fit over 16 s is not a gentler brake, it is a
        worse one: 0/512 stopped. `window_s` is part of the schedule, not a runtime choice.
      * it is fitted at a SPEED and a state. The 6 s-fitted schedule replayed at 88 m topples.
      * it is CHECKPOINT-specific. It rides on that policy's residual.
    """
    __slots__ = ("theta", "window_s", "channels", "knots", "meta")

    CHANNELS = ("freq_scale", "amp_scale", "o_cam", "o_thigh")
    KNOTS = 3

    def __init__(self, theta, window_s, meta=None):
        self.theta = np.asarray(theta, float).reshape(len(self.CHANNELS), self.KNOTS)
        self.window_s = float(window_s or 0.0)
        if self.window_s <= 0.0:
            raise ValueError("a brake schedule needs the window length it was fitted over; "
                             "replaying one over a different window is a different brake")
        self.meta = dict(meta or {})

    @classmethod
    def from_meta(cls, d):
        """From `bundle.meta['brake']`, or None if the bundle carries no schedule."""
        if not d:
            return None
        th = d.get("theta")
        if th is None or len(th) != len(cls.CHANNELS) * cls.KNOTS:
            raise ValueError("the bundle's brake block needs a {}-number theta; got {}".format(
                len(cls.CHANNELS) * cls.KNOTS, None if th is None else len(th)))
        return cls(th, d.get("window_s") or d.get("brake_s"),
                   {k: v for k, v in d.items() if k not in ("theta",)})

    def at(self, frac):
        """The four channel values at `frac` in [0, 1] -- piecewise linear across the knots."""
        x = min(max(frac, 0.0), 1.0) * (self.KNOTS - 1)
        i0 = min(int(np.floor(x)), self.KNOTS - 2)
        w = x - i0
        return self.theta[:, i0] + w * (self.theta[:, i0 + 1] - self.theta[:, i0])

    def apply(self, cruise_spec, frac):
        """The braked spec: the frozen cruise spec with four entries driven by the schedule."""
        ch = self.at(frac)
        spec = np.array(cruise_spec, np.float32, copy=True)
        spec[gait.I_FREQ] = np.clip(cruise_spec[gait.I_FREQ] * ch[0], -1.0, 1.0)
        spec[gait.I_S_CAM] *= ch[1]
        spec[gait.I_S_THIGH] *= ch[1]
        spec[gait.I_O.start] = np.clip(ch[2], -1.0, 1.0)          # o_cam
        spec[gait.I_O.start + 1] = np.clip(ch[3], -1.0, 1.0)      # o_thigh
        return spec


class PolicyControllerV2:
    def __init__(self, bundle, net=None):
        from policy_net import PolicyNet
        b = self.b = bundle
        if getattr(b, "version", 1) != 2:
            raise ValueError("PolicyControllerV2 runs v2 bundles; this one is v{} -- use "
                             "controller.PolicyController".format(getattr(b, "version", 1)))
        m, a = b.meta, b.a
        if m["spec_source"] != "policy":
            # The library variant's spec comes from a gait library plus a Raibert prior evaluated
            # at every commit, neither of which is in the bundle. Refused rather than approximated.
            raise ValueError("this bundle was trained with spec_source={!r}. Only the 'policy' "
                             "variant carries its whole gait spec in the action; the library "
                             "variant needs gait_lib/library.json and the Raibert prior, which "
                             "this runtime does not ship.".format(m["spec_source"]))
        self.net = net if net is not None else PolicyNet(b)
        self.gp = b.gait_params()
        self.nu = int(m["nu"])
        self.action_dim = int(m["action_dim"])
        self.frame_dim = int(m["frame_dim"])
        self.history_len = int(m["history_len"])
        self.hist_idx = np.asarray(a["hist_idx"], int)
        self.hist_raw_len = int(self.hist_idx.max()) + 1
        self.once_dim = int(m["once_dim"])
        self.actor_dim = int(m["actor_dim"])
        self.control_dt = float(m["control_dt"])
        self.spec_dim = self.action_dim - gait.N_RESIDUAL
        if self.spec_dim != gait.SPEC_DIM or int(m["n_harmonics"]) != gait.N_HARMONICS:
            raise ValueError("this bundle's gait is {} spec dims / {} harmonics; the vendored "
                             "generator is {} / {}".format(self.spec_dim, m["n_harmonics"],
                                                           gait.SPEC_DIM, gait.N_HARMONICS))

        self.nominal = np.asarray(a["nominal_ctrl"], float)
        # The generator's hot path, with the plant constants bound. Same algebra as gait.assemble
        # and pinned to it bit for bit by the tests -- it is 5.2 ms of an 8.2 ms tick on the Pi 3B
        # otherwise, almost all of it numpy dispatch on 6-vectors rather than arithmetic.
        self.gait_eval = gait.GaitEval(self.gp, self.nominal)
        self.default_motor_pos = np.asarray(a["default_motor_pos"], float)
        self.q_lo = np.asarray(a["q_lo"], float)
        self.q_hi = np.asarray(a["q_hi"], float)
        vl = np.asarray(a["motor_vel_limit"], float)
        self.vel_limit = np.where(vl > 0.0, vl, np.inf)
        self.accel_limit = float(m["motor_accel_limit"])

        s = m["obs_scales"]
        self.s_pos, self.s_vel = float(s["motor_pos"]), float(s["motor_vel"])
        self.s_trq, self.s_grav = float(s["motor_torque"]), float(s["gravity"])
        self.s_angv = float(s["ang_vel"])
        self.lp_yaw_a = float(np.exp(-self.control_dt / float(m["lp_yaw_tau_s"])))
        # v3 adds a HEADING channel to the frame: the robot's own dead-reckoned heading, from
        # integrating the same gyro z the low-pass above reads. Older (v2) bundles do not export
        # `heading_scale`, so they build the 33-wide frame exactly as before and this costs them
        # nothing -- one control law, both bundles, and the frame width check below catches any
        # mismatch rather than letting a wrong-width observation reach the net.
        self.s_heading = float(s.get("heading", 0.0)) if isinstance(s, dict) else 0.0
        self.has_heading = "heading" in s and int(m["frame_dim"]) > 33
        self.heading_cap = float(m.get("heading_cap_rad", 0.5 * np.pi))
        self.pitch_lp = float(m["pitch_reflex_rate_lp"])

        # the task channel. Three command kinds, one per objective -- see the module docstring.
        self.objective = str(m["objective"])
        self.command_kind = "speed" if self.objective == "joystick" else "run_stop"
        self.task_is_constant = (self.objective == "speed")
        self.task_brake_m = float(m["task_brake_m"])
        self.sprint_dist_m = float(m.get("sprint_dist_m", 0.0))
        self.v_max = float(m.get("v_max") or 0.0)
        self.v_min = float(m.get("v_min") or 0.0)
        if self.command_kind == "speed":
            if not self.v_max > 0.0:
                raise ValueError(
                    "this bundle was trained with objective='joystick', whose task[0] is the "
                    "commanded speed divided by v_max -- but it carries no v_max, so there is no "
                    "way to turn the operator's m/s into the number the policy reads. Re-export "
                    "it with a walk_v2/export.py that writes the command block.")
            if self.v_min >= self.v_max:
                raise ValueError("this bundle's speed range is [{}, {}] m/s, which is empty"
                                 .format(self.v_min, self.v_max))
            # the rate the panel's slider is allowed to move the command at
            rate = float(m.get("v_cmd_rate") or 0.0)
            self.cmd_slew_mps2 = rate if rate > 0.0 else (self.v_max - self.v_min) / DEFAULT_CMD_SLEW_S
            # what THIS checkpoint has actually been commanded (the curriculum widens the draw band
            # downward from the warm-start parent's one speed), so the panel can say when the
            # bottom of the slider is a speed the policy has never been asked for
            self.v_trained = tuple(float(x) for x in getattr(b, "v_trained", (self.v_min,
                                                                              self.v_max)))
        else:
            self.cmd_slew_mps2 = 0.0
            self.v_trained = (0.0, 0.0)

        # the fitted open-loop brake, if this checkpoint has one
        self.brake = BrakeSchedule.from_meta(m.get("brake"))
        self.brake_ticks = (0 if self.brake is None
                            else max(1, int(round(self.brake.window_s / self.control_dt))))

        # the clock resync, off unless a contact signal is supplied (see the module docstring)
        r = m.get("resync", {})
        self.resync_kappa = float(r.get("kappa", 0.0)) if r.get("enable", False) else 0.0
        self.resync_window = float(r.get("window_cycle", 0.0)) * TWO_PI
        self.resync_ema_cycles = float(r.get("ema_cycles", 1.0)) or 1.0
        self.resync_warmup = int(r.get("warmup_cycles", 0))
        self.resync_active = False           # flips true the first time note_contact() is called

        self._alloc()

    def _alloc(self):
        self._history = np.zeros((self.hist_raw_len, self.frame_dim), np.float32)
        self._once = np.zeros(self.once_dim, np.float32)
        # --- the sim's EnvState, minus the plant ---
        self._phase = 0.0                                   # the phase THIS tick assembles at
        self._spec = np.zeros(self.spec_dim, np.float32)    # live latched spec
        self._commit = True                                 # this tick's action latches (reset: 1)
        self._cycle_n = 0
        self._prev_residual = np.zeros(gait.N_RESIDUAL, np.float32)
        self._prev_target = self.nominal.copy()
        self._prev_target_vel = np.zeros(self.nu)
        self._lp_yaw = 0.0
        # heading is measured from where the operator was pointing when the policy started: zero
        # here, and zero again on any `zero_heading()` the operator asks for
        self._yaw_est = 0.0
        self._reflex_prate = 0.0
        self._phi_td_hat = np.array([0.0, np.pi])
        self._resynced = np.zeros(2, bool)
        self._grounded_prev = np.zeros(2, bool)
        self._pending_contact = None
        # the operator's command: a run ALWAYS starts stopped (see start()). For a joystick
        # bundle "stopped" is 0 m/s -- a speed it was trained at, and the one it walks in place at.
        self._run = False
        self._d_to_go = 1.0
        self._v_want = 0.0                                  # where the slider is
        self._v_cmd = 0.0                                   # where the slewed command has got to
        # the brake: None = not braking; otherwise the frozen cruise spec and the tick counter
        self._brake_spec = None
        self._brake_i = 0
        # start() latches the first measurement as the WHOLE history, exactly as _reset_one does,
        # so the first action sees a stationary robot instead of a buffer of zeros. The first
        # step() must therefore NOT push again: in the sim the first action is computed from a
        # history of frame_0 alone, with no second measurement in it.
        self._primed = False
        self.n_steps = 0

    # ------------------------------------------------------------------ command channel
    def set_run(self, run):
        """The green light. False = the red-light / past-the-line phase: the speed income stops
        and a stoplight-trained policy tracks a deceleration ramp down to standing still.

        Returns False if the flag is not something this bundle's policy can see -- a constant task
        channel (objective='speed'), or a joystick bundle, whose task[0] is a speed and has no
        flag in it at all -- in which case the caller should say so rather than pretend."""
        if self.command_kind == "speed":
            return False
        self._run = bool(run)
        return not self.task_is_constant

    @property
    def run(self):
        return self._run

    def set_speed(self, v_mps, immediate=False):
        """THE JOYSTICK. Ask for `v_mps` metres per second; 0 is walk in place.

        Clamped to the range this checkpoint was trained over, because task[0] is clipped in the
        sim too: commanding 6 m/s on a 3 m/s policy does not ask for 6, it asks for 3 while the
        panel says 6. Returns (ok, applied_target_mps); ok is False on a bundle whose command is
        not a speed, and then nothing moves.

        What arrives here is a TARGET. `speed_cmd` walks toward it at `cmd_slew_mps2` on every
        step(), so a slider dragged across its whole travel in one browser event is still a ramp
        on the wire. immediate=True writes it straight through -- for the bench and for tests,
        never for the panel."""
        if self.command_kind != "speed":
            return False, 0.0
        self._v_want = float(np.clip(float(v_mps), self.v_min, self.v_max))
        if immediate:
            self._v_cmd = self._v_want
        return True, self._v_want

    @property
    def speed_target(self):
        """Where the operator put the slider, m/s."""
        return self._v_want

    @property
    def speed_cmd(self):
        """What the policy is being asked for THIS tick, m/s -- the slewed value, which is what
        actually went into the observation."""
        return self._v_cmd

    def _advance_speed(self):
        """One tick of the command slew. Called at the top of step(), before the observation is
        built, because task[0] is read while that observation is assembled."""
        if self.command_kind != "speed":
            return
        step = self.cmd_slew_mps2 * self.control_dt
        d = self._v_want - self._v_cmd
        self._v_cmd = self._v_want if abs(d) <= step else self._v_cmd + np.sign(d) * step

    # ------------------------------------------------------------------ the brake
    def start_brake(self):
        """Begin the fitted open-loop brake. Returns (ok, why).

        Freezes the spec where it is -- that frozen value IS the schedule's baseline, exactly as
        the search froze `state.spec` at the brake point -- and hands the four scheduled channels
        the window. `run` is deliberately not touched: telling the policy it has finished is the
        measured cause of the fall this is here to avoid."""
        if self.command_kind == "speed":
            return False, ("this bundle's command is a SPEED: to stop it, ask for 0 m/s. The "
                           "fitted brake exists because the run/stop lineage could not be told to "
                           "slow down without falling over; a joystick policy is trained at every "
                           "speed in its range and 0 is one of them.")
        if self.brake is None:
            return False, ("this bundle carries no brake schedule. A stop on this lineage is an "
                           "open-loop schedule fitted per checkpoint (walk_v2/tools/"
                           "brake_search.py), not a flag: dropping the run flag instead is "
                           "measured at 3/512 upright against 512/512 for not dropping it.")
        if self._brake_spec is not None:
            return True, ""                       # already braking; pressing again is not a restart
        self._brake_spec = np.array(self._spec, np.float32, copy=True)
        self._brake_i = 0
        return True, ""

    def cancel_brake(self):
        """Hand the spec back to the policy's latch at the next commit tick."""
        was = self._brake_spec is not None
        self._brake_spec = None
        self._brake_i = 0
        return was

    @property
    def braking(self):
        return self._brake_spec is not None

    @property
    def brake_frac(self):
        """Progress through the window in [0, 1]; 1.0 means the schedule has run out and its final
        knot is being held. Holding is deliberate: the window ends with the robot at rest, and the
        only other option is handing a standing robot back to a policy that knows one speed."""
        if self._brake_spec is None or not self.brake_ticks:
            return 0.0
        return min(1.0, self._brake_i / float(self.brake_ticks))

    def zero_heading(self):
        """Declare THIS direction to be straight ahead.

        The heading channel is an integral, so it only means anything relative to an origin. The
        origin is set when the policy starts; call this to move it -- after turning the robot by
        hand, say, or if a long run has accumulated visible drift. Nothing else resets it: a
        heading that silently re-zeroed itself would make the robot veer."""
        self._yaw_est = 0.0

    def heading_deg(self):
        """The robot's own estimate of how far it has turned since the origin, in degrees."""
        return float(np.degrees((self._yaw_est + np.pi) % (2.0 * np.pi) - np.pi))

    def set_distance_to_go(self, frac):
        """task[1], already normalised: clip(metres_to_the_line / task_brake_m, 0, 1). 1.0 means
        'the line is more than task_brake_m away', which is what a bench run should say."""
        self._d_to_go = float(np.clip(frac, 0.0, 1.0))

    def _task(self):
        """The two task entries, exactly as `DashEnvV2._task` builds them."""
        if self.command_kind == "speed":
            # task[1] is reserved (a yaw command later) so the width stays 2 -- held at ONE, not
            # zero. Under v2 semantics this channel is the distance-to-go ramp: 1.0 means "the line
            # is far away", 0 means "brake now". A joystick bundle that shipped 0 here would hold
            # the policy in a permanent stop request; the trainer pins it at 1.0 and this must match
            # bit for bit or the robot runs a different controller than the one that was trained.
            return float(np.clip(self._v_cmd / self.v_max, self.v_min / self.v_max, 1.0)), 1.0
        if self.task_is_constant:
            return 1.0, 1.0
        return (1.0 if self._run else 0.0), self._d_to_go

    # ------------------------------------------------------------------ observation
    def _proprio(self, motor_pos, motor_vel, motor_tau, grav, gyro):
        """One measurement frame, in the exact layout of DashEnvV2._frame.

        There is no sensor-noise model here: on the robot the measurement IS the noisy one. The
        reflexes below read the same measured gravity that goes into this frame, whereas the sim
        reads a clean copy for the reflexes and a corrupted one for the observation. That is an
        inherent sim2real difference and it sits inside the band this run trained against."""
        self._lp_yaw = self.lp_yaw_a * self._lp_yaw + (1.0 - self.lp_yaw_a) * float(gyro[2])
        self._yaw_est += float(gyro[2]) * self.control_dt
        tail = []
        if self.has_heading:
            # wrap to (-pi, pi] then clip, matching walk_v3/env.py exactly: the policy was trained
            # on a saturating heading error, not a wrapping one
            y = (self._yaw_est + np.pi) % (2.0 * np.pi) - np.pi
            tail = [[float(np.clip(y, -self.heading_cap, self.heading_cap)) * self.s_heading]]
        f = np.concatenate([
            (np.asarray(motor_pos, float) - self.default_motor_pos) * self.s_pos,
            np.asarray(motor_vel, float) * self.s_vel,
            np.asarray(motor_tau, float) * self.s_trq,
            np.asarray(grav, float) * self.s_grav,
            np.asarray(gyro, float) * self.s_angv,
            [self._lp_yaw * self.s_angv],
            [np.cos(self._phase), np.sin(self._phase)],
            self._prev_residual,
            *tail,
        ]).astype(np.float32)
        if f.size != self.frame_dim:
            raise ValueError("built a {}-wide observation frame, bundle says {} -- the sensor "
                             "vector widths are wrong".format(f.size, self.frame_dim))
        return f

    def _push_frame(self, frame):
        self._history[:-1] = self._history[1:]
        self._history[-1] = frame

    def _obs(self):
        """history (stride-sampled, newest last) ++ once-block [spec 44, task 2, commit 1]."""
        run, d = self._task()
        self._once[:self.spec_dim] = self._spec
        self._once[self.spec_dim] = run
        self._once[self.spec_dim + 1] = d
        self._once[self.spec_dim + 2] = 1.0 if self._commit else 0.0
        return np.concatenate([self._history[self.hist_idx].reshape(-1), self._once])

    # ------------------------------------------------------------------ the clock
    def note_contact(self, grounded):
        """OPTIONAL: two booleans, (left foot down, right foot down), measured this tick.

        Calling this switches the touchdown resync on and reproduces `_step_one`'s clock
        correction exactly: on a rising edge inside +-resync_window of the foot's expected
        touchdown phase, and at most once per foot per cycle, the habit estimate moves toward the
        observed phase and the clock is pulled by kappa * error. DASH-01 has no such sensor today;
        this exists so that adding one is a wiring change and not a control-law change."""
        self.resync_active = True
        self._pending_contact = np.asarray(grounded, bool)

    def _advance_clock(self, freq_hz):
        """Free-run the phase one tick, then apply a touchdown resync if one was reported.

        Returns True if the cycle wrapped -- which is the NEXT tick's commit flag."""
        phi_adv = self._phase + TWO_PI * freq_hz * self.control_dt
        wrapped = phi_adv >= TWO_PI
        phi = phi_adv % TWO_PI
        self._cycle_n += int(wrapped)
        if wrapped:
            self._resynced[:] = False
        grounded = getattr(self, "_pending_contact", None)
        if grounded is not None:
            self._pending_contact = None
            rising = grounded & ~self._grounded_prev
            err = gait.wrap_pi(self._phi_td_hat - phi)
            can = rising & (np.abs(err) <= self.resync_window) & ~self._resynced
            self._phi_td_hat = self._phi_td_hat + np.where(
                can, gait.wrap_pi(phi - self._phi_td_hat) / self.resync_ema_cycles, 0.0)
            kappa = self.resync_kappa if self._cycle_n >= self.resync_warmup else 0.0
            shift = float(np.sum(np.where(can, kappa * err, 0.0)))
            phi2 = phi + shift
            if phi2 >= TWO_PI:                 # a resync can carry the clock over the wrap
                wrapped = True
                self._cycle_n += 1
            phi = max(phi2, 0.0) % TWO_PI      # a backward pull never uncrosses
            self._resynced = self._resynced | can
            self._grounded_prev = grounded
        self._phase = float(phi)
        return bool(wrapped)

    # ------------------------------------------------------------------ lifecycle
    def start(self, motor_pos, motor_vel, motor_tau, grav, gyro):
        """Start (or restart) the control law from the robot's CURRENT measured pose, STOPPED.

        `_reset_one` fills the whole history with copies of the first frame, and that first frame
        carries a ZERO torque channel and a zero previous residual at phase 0 -- not the measured
        torque. Same here, so the policy's first observation is the one it has seen 300 million
        times. `_prev_target` starts at the STANCE, not at the measured pose: the first commanded
        target is the stance the policy was trained to hold, and the caller is responsible for
        having already crawled the robot to it (the runner's approach phase does exactly this).

        The command starts at rest -- the run flag at 0, the joystick at 0 m/s. A policy that
        comes up already asked to travel would take its first step before anyone had pressed
        anything."""
        self._alloc()
        frame = self._proprio(motor_pos, np.zeros(self.nu), np.zeros(self.nu), grav, gyro)
        self._history[:] = frame
        self._primed = True
        return self._obs()

    reset = start           # backwards-compatible alias; `start` is the honest name

    def step(self, motor_pos, motor_vel, motor_tau, grav, gyro, override_action=None):
        """One control tick. Measurements in, joint targets plus per-joint impedance out.

        override_action: use this action INSTEAD of the network's own output for everything
        downstream (the latch, the generator, and the previous residual the next observation
        carries), while still reporting what the network wanted in `CommandV2.action`. For
        offline replay of a recorded action stream; the runner never passes it."""
        grav = np.asarray(grav, float)
        n = np.linalg.norm(grav)
        if n > 1e-6:
            grav = grav / n          # the sim's sensor model publishes a UNIT gravity vector
        gyro = np.asarray(gyro, float)

        # 0) the operator's command, one slew step closer to where the slider is. Before the
        #    observation, because task[0] is read while the observation is being assembled.
        self._advance_speed()

        # 1) measurement -> newest history frame. It carries the phase THIS tick will assemble at
        #    and the PREVIOUS tick's residual; both offsets are the sim's. Skipped exactly once
        #    after start(), which already latched this measurement into every row.
        if self._primed:
            self._primed = False
        else:
            self._push_frame(self._proprio(motor_pos, motor_vel, motor_tau, grav, gyro))
        obs = self._obs()

        # 2) policy. Mean action, clipped -- the robot never samples.
        action, vel_est = self.net(obs)
        action = action.astype(np.float32)
        drive = action if override_action is None else np.clip(
            np.asarray(override_action, np.float32), -1.0, 1.0)

        # 3) THE LATCH: the 44 spec dims commit only on a commit tick; the 6 residual dims are live
        commit = self._commit
        if commit:
            self._spec = drive[:self.spec_dim].copy()
        residual = drive[self.spec_dim:]
        spec = self._spec
        # 3b) THE BRAKE overrides the latched spec -- and ONLY the latched spec. The policy's
        #     residual above is kept exactly as it is: it is the stabiliser, and the search
        #     established that zeroing it makes everything fall regardless of the schedule.
        #     The latch keeps running underneath, so cancel_brake() hands back a live spec.
        brake_frac = 0.0
        if self._brake_spec is not None:
            brake_frac = self.brake_frac
            spec = self.brake.apply(self._brake_spec, brake_frac)
            self._brake_i += 1
        freq = float(gait.frequency(spec[gait.I_FREQ], self.gp))

        # 4) the reflexes, on the measurement just taken. The pitch rate is EMA-filtered with the
        #    state carried across ticks; alpha 0 means the raw rate (both configs exist).
        roll, roll_rate = float(grav[1]), float(gyro[0])
        pitch, pitch_rate = float(grav[0]), float(gyro[1])
        if self.pitch_lp > 0.0:
            self._reflex_prate = (self.pitch_lp * self._reflex_prate
                                  + (1.0 - self.pitch_lp) * pitch_rate)
            pitch_rate = self._reflex_prate

        # 5) the generator, at the CURRENT phase
        target, kp, kd, _q_ref = self.gait_eval(spec, residual, self._phase,
                                                roll, roll_rate, pitch, pitch_rate)
        target_prefilter = target.copy()

        # 6) the actuation chain the sim runs before the substeps, in the same order: joint range,
        #    then the no-load slew cap. There is NO action filter in v2 (walk_mit's EMA is gone --
        #    the pitch reflex was retuned for its absence).
        tgt = np.minimum(np.maximum(target, self.q_lo), self.q_hi)
        saturated = bool(np.any(tgt != target))
        tgt, tvel = gait.slew_limit(tgt, self._prev_target, self._prev_target_vel,
                                    self.vel_limit, self.accel_limit, self.control_dt)
        self._prev_target, self._prev_target_vel = tgt.copy(), tvel

        # 7) advance the clock; wrapping is the NEXT tick's commit flag
        self._commit = self._advance_clock(freq)
        self._prev_residual = residual
        self.n_steps += 1

        return CommandV2(target=tgt, kp=kp, kd=kd, action=action, phase=self._phase, freq=freq,
                         vel_est=vel_est, residual=residual, spec=spec, commit=commit,
                         run_flag=self._task()[0], saturated=saturated,
                         target_prefilter=target_prefilter, brake_frac=brake_frac,
                         speed_cmd=self._v_cmd)
