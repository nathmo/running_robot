"""SpiderBotEnv — a Gymnasium environment for command-conditioned biped locomotion.

The agent sees only what the real robot can measure (motor pos/vel/torque, IMU-derived gravity
direction + angular velocity, its own previous action, and the joystick command), stacked over a
short history. It outputs 6 PD position targets; the knee follows the parallel linkage and the
ankle follows its spring. Reward tracks the commanded body-frame velocity + yaw rate while staying
upright. No foot-contact sensor is used anywhere.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .config import Config

# per-frame observation groups and sizes (see _proprio); total = 32
FRAME_DIM = 6 + 6 + 6 + 3 + 3 + 6 + 2


class SpiderBotEnv(gym.Env):
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

        self.action_space = spaces.Box(-1.0, 1.0, (self.nu,), np.float32)
        obs_dim = FRAME_DIM * self.cfg.history_len
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

        self._history = np.zeros((self.cfg.history_len, FRAME_DIM), np.float32)
        self._prev_action = np.zeros(self.nu, np.float32)
        self._filt_target = self.nominal_ctrl.copy()
        self._delay_buf = [np.zeros(self.nu, np.float32)
                           for _ in range(self.cfg.action_delay_steps)]
        self._command = np.zeros(2, np.float32)
        # integrated command-pose target (world xy + heading); re-anchored to the actual pose on
        # reset and on every command resample so tracking error stays bounded.
        self._des_xy = np.zeros(2, np.float64)
        self._des_yaw = 0.0
        self._step = 0

    # ---------- helpers ----------
    def set_cmd_vx_frac(self, frac):
        """Curriculum hook: set the sampled forward-command fraction (called via VecEnv.env_method so
        it reaches SubprocVecEnv worker processes too). Takes effect on the next command resample."""
        self.cfg.cmd_vx_frac = float(frac)

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

    def _proprio(self):
        s = self.cfg.obs_scales
        motor_pos = (self.data.qpos[self.act_qadr] - self.default_motor_pos) * s["motor_pos"]
        motor_vel = self.data.qvel[self.act_dadr] * s["motor_vel"]
        motor_trq = self.data.actuator_force[:self.nu] * s["motor_torque"]
        grav = self._gravity_body() * s["gravity"]
        angv = self._ang_vel_body() * s["ang_vel"]
        return np.concatenate([motor_pos, motor_vel, motor_trq, grav, angv,
                               self._prev_action, self._command]).astype(np.float32)

    def _obs(self):
        return self._history.reshape(-1).astype(np.float32)

    def _push_frame(self, frame):
        self._history[:-1] = self._history[1:]
        self._history[-1] = frame

    def _sample_command(self):
        c = self.cfg
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
        # small random perturbation to the hinge joints (qpos[7:] are the 12 hinges)
        n = self.cfg.reset_joint_noise
        self.data.qpos[7:] += self.np_random.uniform(-n, n, self.model.nq - 7)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._prev_action[:] = 0.0
        self._filt_target[:] = self.nominal_ctrl
        self._delay_buf = [np.zeros(self.nu, np.float32)
                           for _ in range(self.cfg.action_delay_steps)]
        self._air_time[:] = 0.0
        self._contact_time[:] = 0.0
        self._grounded_prev = self._grounded(self._foot_contacts())
        self._prev_toe_xy = self.data.geom_xpos[self.foot_gids_arr, 0:2].copy()
        self._push_countdown = self._next_push_in()
        self._step = 0
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

    def step(self, action):
        c = self.cfg
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        # fixed actuation delay (plant truth: Pi inference + moteus/CAN is ~one 50 Hz step).
        # The observation's prev_action stays the policy's own output, like on hardware.
        # (append-then-pop is a no-op passthrough when action_delay_steps == 0)
        self._delay_buf.append(action)
        applied = self._delay_buf.pop(0)
        raw_target = self.nominal_ctrl + c.action_scale * applied
        self._filt_target = c.action_filter * self._filt_target + (1 - c.action_filter) * raw_target
        self.data.ctrl[:] = np.clip(self._filt_target, self.ctrl_lo, self.ctrl_hi)

        # gentle random shove BEFORE the physics runs, so its effect is integrated by the
        # dynamics and reaches this step's sensors/reward as real state — never as a bare qvel
        # edit that the reward would bill to (or credit against) the pre-push action.
        self._push_countdown -= 1
        if self._push_countdown <= 0:
            ang = self.np_random.uniform(0.0, 2.0 * np.pi)
            self.data.qvel[0] += c.push_dv * np.cos(ang)
            self.data.qvel[1] += c.push_dv * np.sin(ang)
            self._push_countdown = self._next_push_in()

        # contact is OR-accumulated across the sim substeps so a sub-20 ms hop cannot pass as
        # continuous flight (or continuous contact) at the 50 Hz sampling boundary.
        contact_acc = np.zeros(2, bool)
        for _ in range(c.control_decimation):
            mujoco.mj_step(self.model, self.data)
            if not contact_acc.all():
                contact_acc |= self._foot_contacts()

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
        reward, terms, progress_frac = self._reward(action)
        reward += self._gait_reward(terms, progress_frac, contact_acc)
        terminated = self._fallen()
        if terminated:
            reward -= c.fall_penalty
        truncated = self._step >= self.max_steps
        self._prev_action[:] = action
        info = {"reward_terms": terms, "command": self._command.copy()}
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
    def _pen(self, v):
        """Floor a penalty term: reward normalization is off, so these raw scales reach PPO
        directly, and no reachable state may make dying cheaper than living (suicide-proofing)."""
        return max(float(v), -self.cfg.penalty_term_cap)

    def _reward(self, action):
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
        t["height"] = self._pen(-c.w_height * (self.data.qpos[2] - self.height_target) ** 2)
        t["vz"] = self._pen(-c.w_vz * self.data.qvel[2] ** 2)
        t["action_rate"] = self._pen(-c.w_action_rate * np.sum((action - self._prev_action) ** 2))
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
