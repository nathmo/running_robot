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

        # foot-tip spheres + floor, for the air-time stepping reward (sim contact, reward-only)
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_gids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_col")
                          for s in "LR"]
        self._air_time = np.zeros(2, np.float32)

        # Parts that must NOT dip below the floor (only the toe spheres may touch). Precompute a
        # subsampled set of each mesh's vertices so we can cheaply test true floor penetration.
        self._nofloor = []
        for nm in ("FootLeftNCS-v1_geom", "FootRightNCS-v1_geom",
                   "LegLeftNCS-v1_geom", "LegRightNCS-v1_geom"):
            g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, nm)
            mid = self.model.geom_dataid[g]
            a, k = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            v = self.model.mesh_vert[a:a + k].reshape(-1, 3)
            if len(v) > 256:
                v = v[np.linspace(0, len(v) - 1, 256).astype(int)]
            self._nofloor.append((g, v.copy()))

        # nominal standing pose / targets from the keyframe
        self.key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.cfg.keyframe)
        self.default_qpos = self.model.key_qpos[self.key_id].copy()
        self.nominal_ctrl = self.model.key_ctrl[self.key_id].copy()
        self.default_motor_pos = self.default_qpos[self.act_qadr]
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

        # height target derived from the keyframe (so it can never go stale relative to the model)
        self.height_target = float(self.default_qpos[2])
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
        self._command = np.zeros(2, np.float32)
        self._step = 0

    # ---------- helpers ----------
    def _sensor_adr(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return self.model.sensor_adr[sid]

    def _base_rot(self):
        return self.data.xmat[self.base_id].reshape(3, 3)

    def _gravity_body(self):
        return self._base_rot().T @ np.array([0.0, 0.0, -1.0])

    def _ang_vel_body(self):
        return self.data.sensordata[self._gyro_adr:self._gyro_adr + 3].copy()

    def _foot_contacts(self):
        """Which foot-tip spheres touch the floor (sim contact; used only for the reward)."""
        c = np.zeros(2, bool)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = (con.geom1, con.geom2)
            if self.floor_gid in pair:
                for fi, fg in enumerate(self.foot_gids):
                    if fg in pair:
                        c[fi] = True
        return c

    def _foot_lateral_sep(self):
        """Body-frame lateral separation of the two toe spheres, sep = y_left - y_right.
        Nominal stance is ~0.40 m; sep shrinking toward 0 (or going negative) means the legs are
        coming together / crossing. Used by the anti-crossing stance reward."""
        R = self._base_rot()
        base = self.data.qpos[0:3]
        y = [(R.T @ (self.data.geom_xpos[fg] - base))[1] for fg in self.foot_gids]  # [L, R]
        return float(y[0] - y[1])

    def _floor_violation(self):
        """Only the LOWER half of each toe sphere may touch the floor (resting on top of it).
        Forbidden (kill the episode) if either:
          (a) a toe-sphere contact lands on its UPPER hemisphere -> the sphere sank below the floor;
          (b) any other foot/shin point dips below the floor -> the 'rest of the foot' is traversing.
        Both would damage the real robot, so they must never be a usable strategy."""
        # (a) toe sphere contacted from above its center => sunk
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = (con.geom1, con.geom2)
            if self.floor_gid in pair:
                for fg in self.foot_gids:
                    if fg in pair and con.pos[2] > self.data.geom_xpos[fg][2] + 1e-3:
                        return True
        # (b) any foot/shin mesh vertex below the floor
        tol = self.cfg.floor_penetration_tol
        for g, v in self._nofloor:
            R = self.data.geom_xmat[g].reshape(3, 3)
            if (v @ R.T + self.data.geom_xpos[g])[:, 2].min() < -tol:
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
            self._command[0] = self.np_random.uniform(-c.cmd_vx_frac, c.cmd_vx_frac)
            self._command[1] = self.np_random.uniform(-c.cmd_yaw_frac, c.cmd_yaw_frac)

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
        self._air_time[:] = 0.0
        self._step = 0
        self._sample_command()
        frame = self._proprio()
        self._history[:] = frame                      # fill history with the initial frame
        return self._obs(), {}

    def step(self, action):
        c = self.cfg
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        raw_target = self.nominal_ctrl + c.action_scale * action
        self._filt_target = c.action_filter * self._filt_target + (1 - c.action_filter) * raw_target
        self.data.ctrl[:] = np.clip(self._filt_target, self.ctrl_lo, self.ctrl_hi)

        for _ in range(c.control_decimation):
            mujoco.mj_step(self.model, self.data)

        self._step += 1
        if self._step % self._resample_every == 0:
            self._sample_command()

        self._push_frame(self._proprio())
        reward, terms = self._reward(action)
        # stepping reward (air-time): credit each foot at touchdown by (air_time - min); strides
        # shorter than the minimum score negative, punishing tiny chattering steps.
        contact = self._foot_contacts()
        first_contact = contact & (self._air_time > 0)
        self._air_time += self.control_dt
        air = c.w_air_time * float(np.sum((self._air_time - c.foot_air_time_min) * first_contact))
        self._air_time[contact] = 0.0
        terms["air_time"] = air
        reward += air
        terminated = self._fallen()
        if terminated:
            reward -= c.fall_penalty
        truncated = self._step >= self.max_steps
        self._prev_action[:] = action
        info = {"reward_terms": terms, "command": self._command.copy()}
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    # ---------- reward & termination ----------
    def _reward(self, action):
        c = self.cfg
        R = self._base_rot()
        v_body = R.T @ self.data.qvel[0:3]
        vx = v_body[0]
        yaw_rate = self._ang_vel_body()[2]
        cmd_vx = self._command[0] * c.vx_max
        cmd_yaw = self._command[1] * c.yaw_max
        grav = self._gravity_body()

        t = {}
        t["track_vx"] = c.w_track_vx * np.exp(-((cmd_vx - vx) ** 2) / c.track_sigma_vx ** 2)
        t["track_yaw"] = c.w_track_yaw * np.exp(-((cmd_yaw - yaw_rate) ** 2) / c.track_sigma_yaw ** 2)
        t["upright"] = -c.w_upright * (grav[0] ** 2 + grav[1] ** 2)
        t["height"] = -c.w_height * (self.data.qpos[2] - self.height_target) ** 2
        t["vz"] = -c.w_vz * self.data.qvel[2] ** 2
        t["action_rate"] = -c.w_action_rate * np.sum((action - self._prev_action) ** 2)
        t["torque"] = -c.w_torque * np.sum(self.data.actuator_force[:self.nu] ** 2)
        # anti-crossing: penalize the feet getting closer than stance_min_sep (and, hard, crossing);
        # one-sided so a normal/wide stance is free.
        sep = self._foot_lateral_sep()
        t["stance"] = -c.w_no_cross * max(0.0, c.stance_min_sep - sep) ** 2
        # keep the hip-roll (lateral) joints near neutral so the legs don't roll inward to cross
        hr = self.data.qpos[self.act_qadr[self.hip_roll_idx]] - self.default_motor_pos[self.hip_roll_idx]
        t["hip_roll"] = -c.w_hip_roll * float(np.sum(hr ** 2))
        t["alive"] = c.w_alive
        return sum(t.values()), t

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
