"""Dash01Env — a Gymnasium environment for command-conditioned biped locomotion.

The agent sees only what the real robot can measure (motor pos/vel/torque, IMU-derived gravity
direction + angular velocity, its own previous action, and the joystick command), stacked over a
short history. It outputs 6 PD position targets; the knee follows the parallel linkage and the
ankle follows its spring. Reward tracks the commanded body-frame velocity + yaw rate while staying
upright. No foot-contact sensor is used anywhere.
"""
from types import SimpleNamespace

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .config import Config

# per-frame observation groups and sizes (see _proprio); total = 32
FRAME_DIM = 6 + 6 + 6 + 3 + 3 + 6 + 2


class Dash01Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: Config = None, render_mode: str = None):
        self.cfg = cfg or Config()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(self.cfg.model_path)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = float(self.model.opt.timestep)
        self.control_dt = self.sim_dt * self.cfg.control_decimation
        self.max_steps = int(round(self.cfg.episode_s / self.control_dt))
        self._resample_every = max(1, int(round(self.cfg.cmd_resample_s / self.control_dt)))

        # actuator -> joint qpos/dof addresses (actuator order = ctrl/action order)
        self.nu = self.model.nu
        self.act_qadr, self.act_dadr = [], []
        for a in range(self.nu):
            jid = self.model.actuator_trnid[a, 0]
            self.act_qadr.append(self.model.jnt_qposadr[jid])
            self.act_dadr.append(self.model.jnt_dofadr[jid])
        self.act_qadr = np.array(self.act_qadr)
        self.act_dadr = np.array(self.act_dadr)

        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bodyNCS-v1")
        self._gyro_adr = self._sensor_adr("imu_gyro")

        # base-DOF locks: 6 <equality><joint> constraints lock_{x,y,z,roll,pitch,yaw}, inactive by
        # default. cfg.base_lock ([X,Y,Z,roll,pitch,yaw], 1=locked) selects which to activate at reset
        # (data.eq_active). Each single-joint equality pins qpos[joint] = eq_data[k,0]: 0 is correct
        # for X/Y (origin) and roll/pitch/yaw (level), but base_z must be pinned at its ride-height,
        # so we set model.eq_data[lock_z,0] at reset. The loop equalities (ids 0,1) are left untouched.
        self.base_lock = np.asarray(self.cfg.base_lock, dtype=np.int32)
        self.lock_eq_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"lock_{n}")
            for n in ("x", "y", "z", "roll", "pitch", "yaw")], dtype=int)
        self.lock_z_eq_id = int(self.lock_eq_ids[2])
        self.z_locked = bool(self.base_lock[2])
        # leg hinges begin at the first qpos address of any non-base joint (base joints are qpos 0..5)
        self.hinge_qadr_start = int(min(
            self.model.jnt_qposadr[j] for j in range(self.model.njnt)
            if self.model.jnt_bodyid[j] != self.base_id))
        # base X/Y translational dof addresses, for the (free-axis-only) random pushes
        self._base_x_dadr = int(self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_x")])
        self._base_y_dadr = int(self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_y")])
        # ride-height -> leg-posture table for M1's per-episode random rail height (see measure_ride_band.py)
        self._lut = None
        if self.cfg.z_rail_randomize:
            _d = np.load(self.cfg.ride_height_lut)
            self._lut = dict(H=_d["H"], hinges=_d["hinges"], ctrl=_d["ctrl"])

        # foot spheres + floor, for the gait-shaping rewards (sim contact, reward-only). The TOE
        # sphere is the walking contact (gait logic keys on it); the HEEL sphere is a passive floor
        # stop (clears the toe-down stance, catches the floor only if the leg folds / foot flattens
        # so the foot can never clip through the ground — see build_model.py).
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_gids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_col")
                          for s in "LR"]
        self.foot_gids_arr = np.array(self.foot_gids)
        self._toe_r = float(self.model.geom_size[self.foot_gids[0]][0])
        # all foot collision spheres (toe + heel), for the deep-penetration safety check
        self._col_gids = {}
        for s in "LR":
            for kind in ("foot", "heel"):
                g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{kind}_{s}_col")
                if g >= 0:
                    self._col_gids[g] = float(self.model.geom_size[g][0])
        self._air_time = np.zeros(2, np.float32)     # continuous seconds NOT grounded, per foot
        self._contact_time = np.zeros(2, np.float32)  # continuous seconds grounded, per foot
        self._grounded_prev = np.zeros(2, bool)
        self._prev_toe_xy = np.zeros((2, 2))
        self._push_countdown = 0

        # nominal standing pose / targets from the keyframe
        self.key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.cfg.keyframe)
        self.default_qpos = self.model.key_qpos[self.key_id].copy()
        self.nominal_ctrl = self.model.key_ctrl[self.key_id].copy()
        self.default_motor_pos = self.default_qpos[self.act_qadr]
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

        # height target derived from the keyframe (so it can never go stale relative to the model)
        self.height_target = float(self.default_qpos[2])
        # standing-baseline holding torque: the torque penalty prices torque ABOVE this, so
        # single-support stance (which must carry the full weight on one leg) isn't taxed into
        # being strictly worse than two-feet-planted skating. The keyframe IS the settled loaded
        # stance (build_model settles it under gravity), so one mj_forward gives the exact PD
        # holding torque — deterministic, no re-settle to drift.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        self.data.ctrl[:] = self.nominal_ctrl
        mujoco.mj_forward(self.model, self.data)
        self._stand_torque = self.data.actuator_force[:self.nu].copy()
        # hip-roll (lateral) actuators, for the neutral-pose regularization that prevents crossing
        self.hip_roll_idx = np.array(
            [a for a in range(self.nu)
             if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "")
             .startswith("hip_roll")], dtype=int)

        # action representation: "pd" = 6 per-step PD targets (default); "fourier" = per-cycle gait
        # coefficients (cam+thigh Fourier + frequency + abduction reflex gains); "fourier_step" =
        # the same 18 coefficients, re-emitted at every 50 Hz control step. See rl/fourier_gait.py.
        if self.cfg.action_mode in ("fourier", "fourier_step"):
            from . import fourier_gait
            self._fourier = fourier_gait
            self.action_dim = fourier_gait.action_dim(self.cfg.n_harmonics)
        else:
            self._fourier = None
            self.action_dim = self.nu
        self.action_space = spaces.Box(-1.0, 1.0, (self.action_dim,), np.float32)
        self._prev_action = np.zeros(self.action_dim, np.float32)     # policy output (obs); coeffs in fourier
        self._prev_motor_cmd = np.zeros(self.nu, np.float32)          # normalized 6 motor targets, for action_rate
        self._command = np.zeros(2, np.float32)
        # fourier_step: the last APPLIED gait spec + the phase-gated change penalty state
        # (billed = sum((applied - prev_applied)^2) * sin(phase/2)^2 — see cfg.w_coef_rate)
        self._prev_applied = np.zeros(self.action_dim, np.float32)
        self._coef_rate_gated = 0.0
        self._phase = 0.0            # gait phase (fourier modes): continuous, kept in [0, 2*pi);
        #                              defined BEFORE the obs-module probe below (the facade exposes it)

        # framework module injection ("" = built-in): an experiment can override the
        # reward and/or observation with a local python module (framework/compile.py sets
        # cfg.reward_module / cfg.obs_module). The stock library modules are pinned
        # numerically identical to the built-in paths by tests/test_module_parity.py.
        self._reward_fn = None
        self._obs_fn = None
        if self.cfg.reward_module or self.cfg.obs_module:
            from framework.modules import load_callable
            if self.cfg.reward_module:
                self._reward_fn = load_callable(self.cfg.reward_module,
                                                self.cfg.experiment_dir, "reward")
            if self.cfg.obs_module:
                self._obs_fn = load_callable(self.cfg.obs_module,
                                             self.cfg.experiment_dir, "features")

        # per-frame obs = motor pos/vel/trq (6+6+6) + gravity(3) + gyro(3) + prev_action + command(2)
        # (+ [sin(phase), cos(phase)] in fourier_step mode, appended after the command);
        # an injected observation module defines its own frame (sized by a probe call —
        # self.data holds the forwarded keyframe at this point)
        if self._obs_fn is not None:
            self.frame_dim = int(np.asarray(self._obs_fn(self._obs_state(), self.cfg)).shape[0])
        else:
            self.frame_dim = 6 + 6 + 6 + 3 + 3 + self.action_dim + 2
            if self.cfg.action_mode == "fourier_step":
                self.frame_dim += 2
        obs_dim = self.frame_dim * self.cfg.history_len
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

        self._history = np.zeros((self.cfg.history_len, self.frame_dim), np.float32)
        self._filt_target = self.nominal_ctrl.copy()
        # delay buffer holds POLICY actions (action_dim: == nu for pd, 18 for fourier_step;
        # the per-cycle fourier macro path doesn't use it)
        self._delay_buf = [np.zeros(self.action_dim, np.float32)
                           for _ in range(self.cfg.action_delay_steps)]
        # integrated command-pose target (world xy + heading); re-anchored to the actual pose on
        # reset and on every command resample so tracking error stays bounded.
        self._des_xy = np.zeros(2, np.float64)
        self._des_yaw = 0.0
        self._step = 0
        self._elapsed_t = 0.0        # cumulative sim time this episode (fourier time-based truncation)
        # optional zero-arg hook fired once per 50 Hz CONTROL step, after the gait state
        # (_grounded_prev/_air_time) has been updated. Rollout scripts (evaluate/gait_probe/
        # joystick) must sample HERE, not per env.step(): in fourier mode one step() is a whole
        # gait cycle, so per-step() sampling strobes at cycle rate (twitching videos, garbage
        # duty/swing metrics).
        self.on_control_step = None

    # ---------- helpers ----------
    def set_cmd_vx_frac(self, frac):
        """Curriculum hook: set the sampled forward-command fraction (called via VecEnv.env_method so
        it reaches SubprocVecEnv worker processes too). Takes effect on the next command resample."""
        self.cfg.cmd_vx_frac = float(frac)

    def set_sprint_dist(self, d):
        """Curriculum hook (VecEnv.env_method): move the sprint finish line. Applies from the NEXT
        reset — the line never moves mid-dash."""
        self.cfg.sprint_dist_m = float(d)

    def _sensor_adr(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return self.model.sensor_adr[sid]

    def _base_rot(self):
        return self.data.xmat[self.base_id].reshape(3, 3)

    def _base_yaw(self):
        R = self._base_rot()
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _anchor_command_pose(self):
        """Re-anchor the integrated command-pose target onto the robot's current base pose."""
        self._des_xy[:] = self.data.qpos[0:2]
        self._des_yaw = self._base_yaw()

    def _gravity_body(self):
        return self._base_rot().T @ np.array([0.0, 0.0, -1.0])

    def _ang_vel_body(self):
        return self.data.sensordata[self._gyro_adr:self._gyro_adr + 3].copy()

    def _foot_contacts(self):
        """Which foot-tip spheres touch the floor (sim contact; used only for the reward)."""
        c = np.zeros(2, bool)
        n = self.data.ncon
        if n == 0:
            return c
        g1 = self.data.contact.geom1[:n]
        g2 = self.data.contact.geom2[:n]
        floor = (g1 == self.floor_gid) | (g2 == self.floor_gid)
        other = np.where(g1 == self.floor_gid, g2, g1)[floor]
        for fi, fg in enumerate(self.foot_gids):
            c[fi] = bool(np.any(other == fg))
        return c

    def _toe_heights(self):
        """Height of each toe sphere's BOTTOM above the floor."""
        return self.data.geom_xpos[self.foot_gids_arr, 2] - self._toe_r

    def _grounded(self, contact):
        """A foot counts as grounded if it has contact OR its sphere bottom is within grounded_h
        of the floor — a toe hovered 1-2 mm up (contact broken, translation continuing) must not
        escape the contact-gated penalties; on a real floor that drag band is a trip hazard."""
        return contact | (self._toe_heights() < self.cfg.grounded_h)

    def _foot_lateral_sep(self):
        """Body-frame lateral separation of the two toe spheres, sep = y_left - y_right.
        Nominal stance is ~0.40 m; sep shrinking toward 0 (or going negative) means the legs are
        coming together / crossing. Used by the anti-crossing stance reward."""
        R = self._base_rot()
        base = self.data.qpos[0:3]
        y = [(R.T @ (self.data.geom_xpos[fg] - base))[1] for fg in self.foot_gids]  # [L, R]
        return float(y[0] - y[1])

    def _floor_violation(self):
        """Safety check: a foot collision sphere (toe or heel) has sunk deep into the floor
        (penetration past half its radius) -> the solver is being driven through the ground.
        The foot can no longer clip through the floor by geometry alone (the toe + heel spheres
        physically stop it), so the old foot/shin mesh-vertex scan is gone: it fired on every
        normal leg-fold when the ghost foot mesh dipped a centimetre below the floor, capping
        episodes at ~1 s and blocking all learning."""
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = (con.geom1, con.geom2)
            if self.floor_gid in pair:
                for g, r in self._col_gids.items():
                    if g in pair and con.dist < -0.5 * r:
                        return True
        return False

    def _obs_state(self):
        """State facade an injected observation module reads (see _lib/obs/standard.py)."""
        return SimpleNamespace(
            motor_pos=self.data.qpos[self.act_qadr] - self.default_motor_pos,
            motor_vel=self.data.qvel[self.act_dadr].copy(),
            motor_torque=self.data.actuator_force[:self.nu].copy(),
            gravity_body=self._gravity_body(),
            ang_vel=self._ang_vel_body(),
            command=self._command.copy(),
            prev_action=self._prev_action.copy(),
            gait_phase=self._phase,
        )

    def _proprio(self):
        if self._obs_fn is not None:
            return np.asarray(self._obs_fn(self._obs_state(), self.cfg), dtype=np.float32)
        s = self.cfg.obs_scales
        motor_pos = (self.data.qpos[self.act_qadr] - self.default_motor_pos) * s["motor_pos"]
        motor_vel = self.data.qvel[self.act_dadr] * s["motor_vel"]
        motor_trq = self.data.actuator_force[:self.nu] * s["motor_torque"]
        grav = self._gravity_body() * s["gravity"]
        angv = self._ang_vel_body() * s["ang_vel"]
        parts = [motor_pos, motor_vel, motor_trq, grav, angv,
                 self._prev_action, self._command]
        if self.cfg.action_mode == "fourier_step":     # + gait phase (the policy needs to know
            parts.append(np.array([np.sin(self._phase), np.cos(self._phase)]))  # where in the cycle
        return np.concatenate(parts).astype(np.float32)                         # its spec lands)

    def _obs(self):
        return self._history.reshape(-1).astype(np.float32)

    def _push_frame(self, frame):
        self._history[:-1] = self._history[1:]
        self._history[-1] = frame

    def _sample_command(self):
        c = self.cfg
        if c.sprint_mode:                       # sprint: full throttle until the line, then stop.
            self._command[0] = 0.0 if self._sprint_crossed else 1.0
            self._command[1] = 0.0              # ([1,0] == the speed_mode training distribution,
            return                              #  so a speed policy fine-tunes into sprint cleanly)
        if c.speed_mode:                        # max-speed milestones: always command forward-max
            self._command[0] = 1.0
            self._command[1] = 0.0
            return
        if self.np_random.random() < c.p_stand:
            self._command[:] = 0.0
        else:
            # move commands have a MINIMUM magnitude: the 0..0.1 m/s band is where sliding beats
            # stepping, and training there first is how the skating basin got consolidated.
            hi = max(c.cmd_vx_frac, c.cmd_vx_min_frac)
            mag = self.np_random.uniform(c.cmd_vx_min_frac, hi)
            sign = 1.0 if c.cmd_forward_only else (1.0 if self.np_random.random() < 0.5 else -1.0)
            self._command[0] = sign * mag
            self._command[1] = self.np_random.uniform(-c.cmd_yaw_frac, c.cmd_yaw_frac)

    def _advance_command_pose(self):
        """Move the integrated command-pose target by one control step of the current command."""
        cmd_vx = self._command[0] * self.cfg.vx_max
        cmd_yaw = self._command[1] * self.cfg.yaw_max
        self._des_yaw += cmd_yaw * self.control_dt
        self._des_xy[0] += self.control_dt * cmd_vx * np.cos(self._des_yaw)
        self._des_xy[1] += self.control_dt * cmd_vx * np.sin(self._des_yaw)

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        # activate this milestone's base-DOF locks (the loop equalities are left active)
        self.data.eq_active[self.lock_eq_ids] = self.base_lock
        # M1: rail Z at a per-episode RANDOM ride-height and seat the legs to reach the floor there
        # (from the ride-height LUT), so the episode starts in a valid on-floor stance. Otherwise a
        # locked Z is pinned at the natural stance height. base_z's lock target lives in eq_data.
        if self.z_locked:
            if self._lut is not None:
                H = float(self.np_random.uniform(*self.cfg.z_rail_range))
                k = int(np.argmin(np.abs(self._lut["H"] - H)))
                self.data.qpos[self.hinge_qadr_start:] = self._lut["hinges"][k]
                self.nominal_ctrl = self._lut["ctrl"][k].astype(np.float64).copy()
            else:
                H = float(self.height_target)
            self.model.eq_data[self.lock_z_eq_id, 0] = H
            self.data.qpos[2] = H
        # small random perturbation to the leg hinges (base qpos occupy 0..hinge_qadr_start-1)
        n = self.cfg.reset_joint_noise
        self.data.qpos[self.hinge_qadr_start:] += self.np_random.uniform(
            -n, n, self.model.nq - self.hinge_qadr_start)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._prev_action[:] = 0.0
        self._prev_motor_cmd[:] = 0.0
        self._prev_applied[:] = 0.0
        self._coef_rate_gated = 0.0
        self._phase = 0.0
        self._elapsed_t = 0.0
        self._filt_target[:] = self.nominal_ctrl
        self._delay_buf = [np.zeros(self.action_dim, np.float32)
                           for _ in range(self.cfg.action_delay_steps)]
        self._air_time[:] = 0.0
        self._contact_time[:] = 0.0
        self._grounded_prev = self._grounded(self._foot_contacts())
        self._prev_toe_xy = self.data.geom_xpos[self.foot_gids_arr, 0:2].copy()
        self._push_countdown = self._next_push_in()
        self._step = 0
        # sprint state: the finish line is frozen per episode (curriculum moves it between dashes)
        self._x0 = float(self.data.qpos[0])
        self._sprint_D = float(self.cfg.sprint_dist_m)
        self._sprint_crossed = False
        self._sprint_t_line = None
        self._sprint_d = 0.0
        self._stop_hold = 0.0
        if self.cfg.sprint_mode:
            self._resample_every = 10 ** 9    # the dash owns the command (run=[1,0] / stop=[0,0])
        self._sample_command()
        self._anchor_command_pose()                   # target starts at the actual standing pose
        frame = self._proprio()
        self._history[:] = frame                      # fill history with the initial frame
        return self._obs(), {}

    def _next_push_in(self):
        c = self.cfg
        if c.push_interval_s <= 0:
            return 10 ** 9
        s = c.push_interval_s * self.np_random.uniform(0.7, 1.3)
        return max(1, int(round(s / self.control_dt)))

    def _update_sprint(self):
        """One control step of sprint bookkeeping (call after physics): distance, the line-crossing
        latch (freezes the dash time + flips the policy's command to 'stop'), and the stopped-hold
        detector. Returns True when the robot has come to rest past the line (success)."""
        self._sprint_d = float(self.data.qpos[0]) - self._x0
        if not self._sprint_crossed and self._sprint_d >= self._sprint_D:
            self._sprint_crossed = True                 # latched: recrossing backward doesn't un-finish
            self._sprint_t_line = self._elapsed_t
            self._command[:] = 0.0                      # the stop signal the policy observes
        if self._sprint_crossed:
            vx = float((self._base_rot().T @ self.data.qvel[0:3])[0])
            if abs(vx) <= self.cfg.stop_speed_eps:
                self._stop_hold += self.control_dt
                if self._stop_hold >= self.cfg.stop_hold_s:
                    return True
            else:
                self._stop_hold = 0.0
        return False

    def _run_physics(self, target6):
        """One control step of plant: EMA-filter the target, clip to ctrlrange, and run
        control_decimation sim substeps (OR-accumulating foot contact across them so a sub-20 ms hop
        can't pass as continuous flight/contact at the 50 Hz boundary). Shared by PD and fourier."""
        c = self.cfg
        self._filt_target = c.action_filter * self._filt_target + (1 - c.action_filter) * target6
        self.data.ctrl[:] = np.clip(self._filt_target, self.ctrl_lo, self.ctrl_hi)
        contact_acc = np.zeros(2, bool)
        for _ in range(c.control_decimation):
            mujoco.mj_step(self.model, self.data)
            if not contact_acc.all():
                contact_acc |= self._foot_contacts()
        return contact_acc

    def step(self, action):
        if self.cfg.action_mode == "fourier":
            return self._step_fourier(action)
        if self.cfg.action_mode == "fourier_step":
            return self._step_fourier_step(action)
        c = self.cfg
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        # fixed actuation delay (plant truth: Pi inference + moteus/CAN is ~one 50 Hz step).
        # (append-then-pop is a no-op passthrough when action_delay_steps == 0)
        self._delay_buf.append(action)
        applied = self._delay_buf.pop(0)
        target6 = self.nominal_ctrl + c.action_scale * applied

        # gentle random shove BEFORE the physics runs, so its effect is integrated by the dynamics
        # and reaches this step's sensors/reward as real state (not a bare qvel edit).
        self._push_countdown -= 1
        if self._push_countdown <= 0:
            ang = self.np_random.uniform(0.0, 2.0 * np.pi)
            if not self.base_lock[0]:                       # only shove the FREE translational axes
                self.data.qvel[self._base_x_dadr] += c.push_dv * np.cos(ang)
            if not self.base_lock[1]:
                self.data.qvel[self._base_y_dadr] += c.push_dv * np.sin(ang)
            self._push_countdown = self._next_push_in()

        contact_acc = self._run_physics(target6)
        self._elapsed_t += self.control_dt
        finished = c.sprint_mode and self._update_sprint()

        self._step += 1
        if self._step % self._resample_every == 0:
            self._sample_command()
            self._anchor_command_pose()               # new command -> re-anchor target to here
            # the gait clocks belong to the OLD command: a stand phase must not arrive at a move
            # command with 4 s of banked contact time (instant stance-cap penalty at the switch)
            self._contact_time[:] = 0.0
            self._air_time[:] = 0.0
        else:
            self._advance_command_pose()              # else advance target by the command

        self._push_frame(self._proprio())
        reward, terms = self._compute_reward(action, contact_acc)   # PD: motor_cmd == action
        terminated = self._fallen()
        if terminated:
            reward -= c.fall_penalty
        elif finished:                     # dash complete: stopped past the line
            terminated = True
            reward += c.finish_bonus
        truncated = self._step >= self.max_steps
        self._prev_action[:] = action
        self._prev_motor_cmd[:] = action
        if self.on_control_step is not None:
            self.on_control_step()
        info = {"reward_terms": terms, "command": self._command.copy()}
        if c.sprint_mode:
            info["sprint"] = self._sprint_info(finished)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def _step_fourier(self, action):
        """MACRO-step = one full gait cycle. The action is the per-cycle gait spec (cam+thigh Fourier
        coeffs + frequency + abduction reflex gains, see rl/fourier_gait.py). We replay the cycle at
        50 Hz — reconstructing the 6 PD targets each control step (abduction driven by the reflex on
        current roll/roll-rate) — and return the MEAN per-control-step reward (keeps the per-step
        reward tuning scale). One obs frame is emitted at the cycle boundary."""
        c = self.cfg
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        cam_c, thigh_c, freq_raw, reflex = self._fourier.decode(action, c.n_harmonics)
        f = self._fourier.frequency(freq_raw, c.gait_freq_hz)
        K = max(1, int(round(1.0 / (f * self.control_dt))))     # control steps per gait cycle
        dphi = 2.0 * np.pi / K

        cyc_reward, term_sum, fell, finished, n = 0.0, None, False, False, 0
        for _ in range(K):
            roll = float(self._gravity_body()[1])               # ~roll angle (small-angle: grav_y)
            roll_rate = float(self._ang_vel_body()[0])          # roll rate (gyro x)
            target6 = self._fourier.assemble(cam_c, thigh_c, reflex, self._phase,
                                             roll, roll_rate, self.nominal_ctrl, c)
            motor_cmd = ((target6 - self.nominal_ctrl) / c.action_scale).astype(np.float32)
            contact_acc = self._run_physics(target6)
            self._elapsed_t += self.control_dt
            finished = c.sprint_mode and self._update_sprint()
            reward, terms = self._compute_reward(motor_cmd, contact_acc)
            self._prev_motor_cmd[:] = motor_cmd
            cyc_reward += reward
            term_sum = dict(terms) if term_sum is None else {k: term_sum[k] + terms[k] for k in terms}
            n += 1
            self._phase = (self._phase + dphi) % (2.0 * np.pi)
            if self.on_control_step is not None:
                self.on_control_step()
            if finished:                                        # dash complete: stop mid-cycle
                break
            if self._fallen():
                fell = True
                break

        macro_reward = cyc_reward / n
        if fell:
            macro_reward -= c.fall_penalty
        elif finished:
            macro_reward += c.finish_bonus
        terminated = fell or finished
        truncated = self._elapsed_t >= c.episode_s
        mean_terms = {k: v / n for k, v in term_sum.items()}
        self._prev_action[:] = action
        self._push_frame(self._proprio())                       # one obs frame per cycle
        self._step += 1
        info = {"reward_terms": mean_terms, "command": self._command.copy()}
        if c.sprint_mode:
            info["sprint"] = self._sprint_info(finished)
        return self._obs(), float(macro_reward), bool(terminated), bool(truncated), info

    def _step_fourier_step(self, action):
        """Per-STEP Fourier override: same 18-dim gait-spec action as the per-cycle mode, but
        re-emitted at EVERY 50 Hz control step — the policy can instantly rewrite the coefficients
        mid-cycle and PD tracks the freshly reconstructed setpoint (mirrors the pd step structure,
        NOT the macro loop). Abrupt rewrites are priced by the phase-gated coef_rate penalty,
        which is ZERO at the cycle boundary (sin(phase/2)^2 gate). With a constant action this
        reproduces the per-cycle trajectory exactly (tests/test_fourier_step.py)."""
        c = self.cfg
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        # fixed actuation delay, like pd: the DELAYED spec is what the plant actually runs
        self._delay_buf.append(action)
        applied = self._delay_buf.pop(0)
        cam_c, thigh_c, freq_raw, reflex = self._fourier.decode(applied, c.n_harmonics)
        f = self._fourier.frequency(freq_raw, c.gait_freq_hz)
        phase_used = self._phase
        # phase-gated coefficient-change penalty state (read by _reward as term "coef_rate"):
        # changing the gait spec exactly at a cycle boundary (phase ~ 0 == 2pi) is FREE
        self._coef_rate_gated = (float(np.sum((applied - self._prev_applied) ** 2))
                                 * float(np.sin(phase_used / 2.0) ** 2))
        self._prev_applied = applied.copy()
        roll = float(self._gravity_body()[1])               # ~roll angle (small-angle: grav_y)
        roll_rate = float(self._ang_vel_body()[0])          # roll rate (gyro x)
        target6 = self._fourier.assemble(cam_c, thigh_c, reflex, phase_used,
                                         roll, roll_rate, self.nominal_ctrl, c)
        motor_cmd = ((target6 - self.nominal_ctrl) / c.action_scale).astype(np.float32)

        # gentle random shove BEFORE the physics runs (free translational axes only), like pd
        self._push_countdown -= 1
        if self._push_countdown <= 0:
            ang = self.np_random.uniform(0.0, 2.0 * np.pi)
            if not self.base_lock[0]:
                self.data.qvel[self._base_x_dadr] += c.push_dv * np.cos(ang)
            if not self.base_lock[1]:
                self.data.qvel[self._base_y_dadr] += c.push_dv * np.sin(ang)
            self._push_countdown = self._next_push_in()

        contact_acc = self._run_physics(target6)
        self._elapsed_t += self.control_dt
        finished = c.sprint_mode and self._update_sprint()
        # advance the gait phase AFTER assembly (the obs frame below carries the NEXT step's phase)
        self._phase = (self._phase + 2.0 * np.pi * f * self.control_dt) % (2.0 * np.pi)

        self._step += 1
        if self._step % self._resample_every == 0:
            self._sample_command()
            self._anchor_command_pose()               # new command -> re-anchor target to here
            # the gait clocks belong to the OLD command (see the pd path)
            self._contact_time[:] = 0.0
            self._air_time[:] = 0.0
        else:
            self._advance_command_pose()              # else advance target by the command

        self._push_frame(self._proprio())
        reward, terms = self._compute_reward(motor_cmd, contact_acc)
        terminated = self._fallen()
        if terminated:
            reward -= c.fall_penalty
        elif finished:                     # dash complete: stopped past the line
            terminated = True
            reward += c.finish_bonus
        truncated = self._step >= self.max_steps
        self._prev_action[:] = action
        self._prev_motor_cmd[:] = motor_cmd
        if self.on_control_step is not None:          # this mode: 1 env.step == 1 control step
            self.on_control_step()
        info = {"reward_terms": terms, "command": self._command.copy()}
        if c.sprint_mode:
            info["sprint"] = self._sprint_info(finished)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def _gait_reward(self, terms, progress_frac, contact_acc):
        """Gait-shaping terms (all sim-side, reward-only). The measured failure mode of v2 was
        skating: both feet grounded 57-66% of the time, translation entirely from loaded-foot
        slip (0.11-0.24 m/s), zero real swings. Each term below removes one leg of that strategy:
        slip makes it pay, the stance cap forces every foot to cycle, clearance makes the first
        honest swings pay immediately, and the one-sided touchdown credit no longer punishes them."""
        c = self.cfg
        dt = self.control_dt
        toe_pos = self.data.geom_xpos[self.foot_gids_arr].copy()   # (2,3), one fetch per step
        heights = toe_pos[:, 2] - self._toe_r
        grounded = contact_acc | (heights < c.grounded_h)
        grounded_recent = grounded | self._grounded_prev   # 2-step debounce for the gates

        # foot slip: horizontal toe-CENTER speed over the control step (catches sphere rolling
        # too), penalized only if the foot was grounded at BOTH ends of the interval — a landing
        # foot arrives with legitimate swing speed and must not be billed for it (that would
        # re-teach 'don't step'). Push-off micro-hops can't hide in the gap: contact is substep-
        # accumulated, so a foot that touched at all during the interval stays 'grounded'.
        slip_v = np.linalg.norm(toe_pos[:, 0:2] - self._prev_toe_xy, axis=1) / dt
        slip = np.where(grounded & self._grounded_prev,
                        np.maximum(0.0, slip_v - c.slip_deadband) ** 2, 0.0)
        terms["foot_slip"] = -min(c.w_foot_slip * float(slip.sum()), c.penalty_term_cap)

        if c.sprint_mode:      # run phase = full-speed gait shaping; stop phase = standing, no gait
            cmd_speed = 0.0 if self._sprint_crossed else c.v_ceiling
        elif c.speed_mode:
            cmd_speed = c.v_ceiling
        else:
            cmd_speed = abs(float(self._command[0])) * c.vx_max
        gait_on = cmd_speed >= c.gait_cmd_gate

        # air/stance clocks + one-sided capped touchdown credit (computed BEFORE the clocks
        # advance, so the landing step is not counted as swing).
        air = 0.0
        for i in range(2):
            if grounded[i]:
                if self._air_time[i] > 0 and gait_on:
                    air += c.w_air_time * float(np.clip(
                        self._air_time[i] - c.foot_air_time_min, 0.0, c.air_credit_cap_s))
                self._air_time[i] = 0.0
                self._contact_time[i] += dt
            else:
                self._air_time[i] += dt
                self._contact_time[i] = 0.0
        terms["air_time"] = air

        # per-foot stance-time cap: any foot grounded longer than the allowance pays per step.
        if gait_on:
            cap = c.stance_cap_s if cmd_speed >= c.stance_slow_speed else c.stance_cap_slow_s
            over = np.minimum(np.maximum(self._contact_time - cap, 0.0), 1.0)
            terms["stance_time"] = -min(c.w_stance_time * float(over.sum()), c.penalty_term_cap)
        else:
            terms["stance_time"] = 0.0

        # swing clearance: pays only for FRESH swings, above the ghost-drag band, scaled by
        # progress so marching in place earns little.
        clear = 0.0
        if gait_on:
            for i in range(2):
                if not grounded_recent[i] and 0.0 < self._air_time[i] <= c.swing_fresh_s:
                    frac = np.clip((float(heights[i]) - c.clearance_dead_m)
                                   / c.clearance_scale_m, 0.0, 1.0)
                    clear += c.w_clearance * float(frac) * (0.3 + 0.7 * progress_frac)
        terms["clearance"] = clear

        self._grounded_prev = grounded
        self._prev_toe_xy = toe_pos[:, 0:2]
        return terms["foot_slip"] + air + terms["stance_time"] + clear

    # ---------- reward & termination ----------
    def _compute_reward(self, motor_cmd, contact_acc):
        """One control step of reward: the built-in _reward + _gait_reward pair, or an
        injected experiment reward module fed the same state. Both paths advance the gait
        clocks exactly once; tests/test_module_parity.py pins the stock library module
        numerically identical to the built-in path."""
        if self._reward_fn is None:
            reward, terms, progress_frac = self._reward(motor_cmd)
            reward += self._gait_reward(terms, progress_frac, contact_acc)
            return reward, terms
        state, cmd = self._reward_state(motor_cmd, contact_acc)
        terms = self._reward_fn(state, cmd, self.cfg)
        return float(sum(terms.values())), terms

    def _reward_state(self, motor_cmd, contact_acc):
        """Assemble the reward-module state facade AND advance the gait clocks — the
        module path's equivalent of _gait_reward's bookkeeping (mutates _air_time /
        _contact_time / _grounded_prev / _prev_toe_xy identically to the built-in path).
        Field reference: experiments/_lib/rewards/gait_speed_v3.py docstring."""
        c = self.cfg
        dt = self.control_dt
        R = self._base_rot()
        v_body = R.T @ self.data.qvel[0:3]
        # NOTE dtype fidelity: the built-in path mixes float64 state with float32
        # command products and clocks; NumPy's promotion rules make the rounding depend
        # on which side is a "weak" python float. The facade hands the module the SAME
        # dtypes so both paths round identically — do not "clean up" with float() here
        # (tests/test_module_parity.py is the guard).
        vx = v_body[0]                                   # np.float64, like _reward
        angv = self._ang_vel_body()
        grav = self._gravity_body()
        cmd_vx_phys = self._command[0] * c.vx_max        # float32 product, like _reward
        cmd_yaw_phys = self._command[1] * c.yaw_max

        # mode-specific quantities (mirrors _reward)
        pos_err = yaw_err = sprint_overrun = 0.0
        if c.sprint_mode:
            progress_frac = 0.0 if self._sprint_crossed else float(np.clip(vx / c.v_ceiling, 0.0, 1.0))
            sprint_overrun = max(0.0, self._sprint_d - (self._sprint_D + c.sprint_brake_m))
            gate_vx = 0.0 if self._sprint_crossed else c.v_ceiling
        elif c.speed_mode:
            progress_frac = float(np.clip(vx / c.v_ceiling, 0.0, 1.0))
            gate_vx = c.v_ceiling
        else:
            progress_frac = (float(np.clip(vx / cmd_vx_phys, 0.0, 1.0))
                             if abs(cmd_vx_phys) > 1e-6 else 0.0)
            pos_err = float(np.linalg.norm(self.data.qpos[0:2] - self._des_xy))
            yaw_err = np.arctan2(np.sin(self._base_yaw() - self._des_yaw),
                                 np.cos(self._base_yaw() - self._des_yaw))
            gate_vx = cmd_vx_phys

        # gait bookkeeping (mirrors _gait_reward; advances the same clocks)
        toe_pos = self.data.geom_xpos[self.foot_gids_arr].copy()
        heights = toe_pos[:, 2] - self._toe_r
        grounded = contact_acc | (heights < c.grounded_h)
        grounded_recent = grounded | self._grounded_prev
        slip_v = np.linalg.norm(toe_pos[:, 0:2] - self._prev_toe_xy, axis=1) / dt
        feet = []
        for i in range(2):
            billable = bool(grounded[i] and self._grounded_prev[i])
            just_landed = bool(grounded[i] and self._air_time[i] > 0)
            air_basis = self._air_time[i]                # float32 scalar (clock dtype)
            if grounded[i]:
                self._air_time[i] = 0.0
                self._contact_time[i] += dt
            else:
                self._air_time[i] += dt
                self._contact_time[i] = 0.0
            feet.append(SimpleNamespace(
                grounded=bool(grounded[i]),
                slip_speed=float(slip_v[i]) if billable else 0.0,
                just_landed=just_landed,
                air_time=air_basis,
                stance_time=self._contact_time[i],       # float32 scalar (clock dtype)
                fresh_swing=bool(not grounded_recent[i]
                                 and 0.0 < self._air_time[i] <= c.swing_fresh_s),
                clearance=float(heights[i]),
            ))
        self._grounded_prev = grounded
        self._prev_toe_xy = toe_pos[:, 0:2]

        hr = (self.data.qpos[self.act_qadr[self.hip_roll_idx]]
              - self.default_motor_pos[self.hip_roll_idx])
        state = SimpleNamespace(
            vx=vx, vy=v_body[1],
            vz=self.data.qvel[2], height=self.data.qpos[2],
            height_target=self.height_target, z_locked=self.z_locked,
            gravity_body=grav, ang_vel=angv, yaw_rate=angv[2],
            pos_err=pos_err, yaw_err=yaw_err, progress_frac=progress_frac,
            motor_torque=self.data.actuator_force[:self.nu].copy(),
            stand_torque=np.abs(self._stand_torque),
            action_rate_sq=np.sum((motor_cmd - self._prev_motor_cmd) ** 2),  # float32, like _reward
            coef_rate_gated=self._coef_rate_gated,   # phase-gated gait-spec change (fourier_step)
            hip_roll=hr, foot_sep=self._foot_lateral_sep(),
            sprint_crossed=bool(getattr(self, "_sprint_crossed", False)),
            sprint_overrun_m=sprint_overrun,
            footL=feet[0], footR=feet[1],
        )
        cmd = SimpleNamespace(vx=gate_vx, yaw=cmd_yaw_phys)
        return state, cmd

    def _pen(self, v):
        """Floor a penalty term: reward normalization is off, so these raw scales reach PPO
        directly, and no reachable state may make dying cheaper than living (suicide-proofing)."""
        return max(float(v), -self.cfg.penalty_term_cap)

    def _reward(self, motor_cmd):
        # motor_cmd = the normalized 6-vector motor command this control step (PD: == the action;
        # fourier: (reconstructed target - nominal)/action_scale). Only action_rate uses it.
        c = self.cfg
        R = self._base_rot()
        v_body = R.T @ self.data.qvel[0:3]
        vx = v_body[0]
        angv = self._ang_vel_body()
        yaw_rate = angv[2]
        cmd_vx = self._command[0] * c.vx_max
        cmd_yaw = self._command[1] * c.yaw_max
        grav = self._gravity_body()

        t = {}
        if c.sprint_mode:
            # 100 m dash. Run phase: dense speed income (same monotone term as speed_mode).
            # Stop phase (line crossed): income flips to 'be stationary', with a free braking
            # zone then a per-meter overrun penalty. Both phases pay the constant clock cost —
            # sum(vx)*dt integrates to the DISTANCE whatever the pace, so the clock term
            # (-w_time * T total) is what actually prices the dash time; it also replaces
            # w_alive, which would pay the policy per second of dawdling.
            for k in ("track_vx", "track_yaw", "progress", "track_pos",
                      "track_heading", "pos_pen", "yaw_rate", "heading_pen"):
                t[k] = 0.0
            t["time"] = -c.w_time
            if not self._sprint_crossed:
                t["fwd_speed"] = c.w_fwd_speed * float(np.clip(vx, 0.0, c.v_ceiling))
                t["stop"] = 0.0
                t["overrun"] = 0.0
                progress_frac = float(np.clip(vx / c.v_ceiling, 0.0, 1.0))
            else:
                t["fwd_speed"] = 0.0
                t["stop"] = c.w_stop_vel * float(np.exp(-((vx / c.stop_sigma) ** 2)))
                over = max(0.0, self._sprint_d - (self._sprint_D + c.sprint_brake_m))
                t["overrun"] = self._pen(-c.w_overrun * over)
                progress_frac = 0.0
        elif c.speed_mode:
            # maximum forward speed: one monotone linear reward, clip(vx, 0, v_ceiling) -- "faster is
            # strictly better" up to a physical cap. The command-tracking terms are switched off
            # (straightness comes from the base locks + the surviving lat_vel/heading penalties).
            t["fwd_speed"] = c.w_fwd_speed * float(np.clip(vx, 0.0, c.v_ceiling))
            for k in ("track_vx", "track_yaw", "progress", "track_pos",
                      "track_heading", "pos_pen", "yaw_rate", "heading_pen"):
                t[k] = 0.0
            progress_frac = float(np.clip(vx / c.v_ceiling, 0.0, 1.0))
        else:
            t["fwd_speed"] = 0.0
            t["track_vx"] = c.w_track_vx * np.exp(-((cmd_vx - vx) ** 2) / c.track_sigma_vx ** 2)
            t["track_yaw"] = c.w_track_yaw * np.exp(-((cmd_yaw - yaw_rate) ** 2) / c.track_sigma_yaw ** 2)
            # forward progress along the commanded heading: fraction of the commanded speed actually
            # achieved (clipped to [0,1] so it can't be gamed by lunging), SQUARED so cruising at a
            # fraction of the command is not a plateau. Zero when no motion is commanded.
            if abs(cmd_vx) > 1e-6:
                progress_frac = float(np.clip(vx / cmd_vx, 0.0, 1.0))
            else:
                progress_frac = 0.0
            t["progress"] = c.w_progress * progress_frac ** 2
            # integrated command-pose tracking: be where the command says (kills spin, wander, and the
            # "fake slow shuffle" gaming, because position/heading error accumulates over a few seconds)
            pos_err = float(np.linalg.norm(self.data.qpos[0:2] - self._des_xy))
            t["track_pos"] = c.w_track_pos * np.exp(-pos_err ** 2 / c.track_sigma_pos ** 2)
            yaw_err = np.arctan2(np.sin(self._base_yaw() - self._des_yaw),
                                 np.cos(self._base_yaw() - self._des_yaw))
            t["track_heading"] = c.w_track_heading * np.exp(-(yaw_err ** 2) / c.track_sigma_heading ** 2)
            # linear companion to the saturating pos kernel: sustained under-speed keeps costing even
            # once the exp kernel has flatlined (the v2 policy parked in exactly that plateau).
            t["pos_pen"] = self._pen(-c.w_pos_l1 * pos_err)
            # quadratic companions (bounded, see _pen) that keep a restoring gradient far from target,
            # where the exp-kernel tracking rewards saturate to ~0 gradient.
            t["yaw_rate"] = self._pen(-c.w_yaw_rate * (yaw_rate - cmd_yaw) ** 2)
            t["heading_pen"] = self._pen(-c.w_heading_pen * yaw_err ** 2)
        t["lat_vel"] = self._pen(-c.w_lat_vel * v_body[1] ** 2)         # go straight, don't wander
        t["ang_xy"] = self._pen(-c.w_angvel_xy * (angv[0] ** 2 + angv[1] ** 2))
        t["upright"] = self._pen(-c.w_upright * (grav[0] ** 2 + grav[1] ** 2))
        # height/vz are meaningless when Z is railed (the base can't move vertically); neutralize them
        # so a per-episode ride-height isn't billed as a constant off-target penalty.
        if self.z_locked:
            t["height"] = 0.0
            t["vz"] = 0.0
        else:
            t["height"] = self._pen(-c.w_height * (self.data.qpos[2] - self.height_target) ** 2)
            t["vz"] = self._pen(-c.w_vz * self.data.qvel[2] ** 2)
        t["action_rate"] = self._pen(-c.w_action_rate * np.sum((motor_cmd - self._prev_motor_cmd) ** 2))
        # fourier_step only: the phase-gated gait-spec change penalty (state set in
        # _step_fourier_step). CONDITIONAL key — old modes must not gain it (golden term-key sets).
        if c.action_mode == "fourier_step":
            t["coef_rate"] = self._pen(-c.w_coef_rate * self._coef_rate_gated)
        # torque ABOVE the standing baseline magnitude, one-sided: holding the stance is free,
        # relaxing (a swing leg unloading toward zero) is free, only exceeding the baseline pays.
        # (Raw tau^2 made one-leg support cost 2x two-leg support by construction.)
        exc = np.maximum(np.abs(self.data.actuator_force[:self.nu])
                         - np.abs(self._stand_torque), 0.0)
        t["torque"] = self._pen(-c.w_torque * np.sum(exc ** 2))
        # anti-crossing: penalize the feet getting closer than stance_min_sep (and, hard, crossing);
        # one-sided so a normal/wide stance is free.
        sep = self._foot_lateral_sep()
        t["stance"] = self._pen(-c.w_no_cross * max(0.0, c.stance_min_sep - sep) ** 2)
        # keep the hip-roll (lateral) joints near neutral so the legs don't roll inward to cross
        hr = self.data.qpos[self.act_qadr[self.hip_roll_idx]] - self.default_motor_pos[self.hip_roll_idx]
        t["hip_roll"] = self._pen(-c.w_hip_roll * float(np.sum(hr ** 2)))
        t["alive"] = c.w_alive
        return sum(t.values()), t, progress_frac

    def _sprint_info(self, finished):
        """Dash telemetry for eval tooling: distance covered, elapsed time, time at the line."""
        return dict(d=round(self._sprint_d, 2), t=round(self._elapsed_t, 2),
                    t_line=None if self._sprint_t_line is None else round(self._sprint_t_line, 2),
                    dist_target=self._sprint_D, finished=bool(finished))

    def _fallen(self):
        if not np.all(np.isfinite(self.data.qpos)):
            return True
        if self.data.qpos[2] < self.cfg.term_height:
            return True
        if self._gravity_body()[2] > self.cfg.term_gravity_z:   # tipped past ~60 deg
            return True
        if self._floor_violation():                             # foot sank through the floor
            return True
        return False
