"""DashEnv — Gymnasium environment for the DASH-01 100 m sprint (and max-speed debug) task.

One env.step == one 50 Hz control step. The action is the per-step Fourier gait spec + residuals
(see fourier_gait.py): the policy re-parameterizes a phase-driven gait generator every step
(CPG-RL-style) and adds small direct target corrections (PMTG-style). The observation is
proprioception the real robot can measure (motor pos/vel/torque, IMU gravity + gyro) PLUS
privileged sim state that the reward is built on (base velocity; sprint distance-to-go) — those
two need estimators on hardware and are the documented sim2real debt.

The base can be partially railed (cfg.base_lock) for the m1..m6 milestone curriculum; the model's
<equality> joint locks are activated at reset. m1 rails Z at a per-episode random ride height,
seated from the measured ride-height->posture LUT.
"""
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from config import Config
import fourier_gait

PKG_DIR = Path(__file__).resolve().parent


def _resolve(p):
    """Resolve a config path: as given if it exists (absolute or CWD-relative), else relative
    to this package directory — so scripts work from any working directory."""
    q = Path(p)
    if q.exists():
        return str(q)
    return str(PKG_DIR / p)


class DashEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: Config = None, render_mode: str = None):
        self.cfg = cfg or Config()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(_resolve(self.cfg.model_path))
        self.data = mujoco.MjData(self.model)
        self.sim_dt = float(self.model.opt.timestep)
        self.control_dt = self.sim_dt * self.cfg.control_decimation
        self.max_steps = int(round(self.cfg.episode_s / self.control_dt))
        # rate-invariance: the reward is hand-balanced in raw PER-STEP units at 50 Hz (0.02 s) with
        # normalization OFF, while the fall/finish bonuses are per-EVENT. Scaling the summed per-step
        # reward by control_dt/0.02 makes the per-SECOND reward invariant to the control rate, so the
        # suicide-proofing / stop-farm balance holds at any Hz. Exactly 1.0 at 50 Hz (a no-op).
        self._reward_dt_scale = self.control_dt / 0.02

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
        # lower-CoM experiment: shift EVERY body's inertial CoM down by com_lower (the base alone is
        # only ~14% of mass, so shifting just it barely moves the real CoM). Moving all mass down
        # com_lower in-frame lowers the whole-robot CoM by ~com_lower. Done once at load, pre-forward.
        if self.cfg.com_lower != 0.0:
            self.model.body_ipos[1:, 2] -= float(self.cfg.com_lower)
        self._gyro_adr = self._sensor_adr("imu_gyro")

        # base-DOF locks: 6 <equality><joint> constraints lock_{x,y,z,roll,pitch,yaw}, inactive by
        # default; cfg.base_lock selects which to activate at reset (data.eq_active). Each pins
        # qpos[joint] = eq_data[k,0]: 0 is correct for X/Y and roll/pitch/yaw, but base_z must be
        # pinned at its ride height, so eq_data[lock_z,0] is set at reset.
        self.base_lock = np.asarray(self.cfg.base_lock, dtype=np.int32)
        self.lock_eq_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"lock_{n}")
            for n in ("x", "y", "z", "roll", "pitch", "yaw")], dtype=int)
        self.lock_z_eq_id = int(self.lock_eq_ids[2])
        self.z_locked = bool(self.base_lock[2])
        # leg hinges begin at the first qpos address of any non-base joint (base joints are 0..5)
        self.hinge_qadr_start = int(min(
            self.model.jnt_qposadr[j] for j in range(self.model.njnt)
            if self.model.jnt_bodyid[j] != self.base_id))
        self._base_x_dadr = int(self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_x")])
        self._base_y_dadr = int(self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_y")])
        _pitch_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_pitch")
        self._base_pitch_qadr = int(self.model.jnt_qposadr[_pitch_jid])
        self._base_pitch_dadr = int(self.model.jnt_dofadr[_pitch_jid])
        # ankle (foot) joints = the only ones with a spring (passive ankle); dof addr for the L/R
        # ankle-torque reflex. Sorted by qpos addr so [0]=Left, [1]=Right (L body precedes R).
        _ankle_j = sorted((j for j in range(self.model.njnt) if self.model.jnt_stiffness[j] > 0),
                          key=lambda j: self.model.jnt_qposadr[j])
        self._ankle_dadr = [int(self.model.jnt_dofadr[j]) for j in _ankle_j]
        # ride-height -> leg-posture table for m1's per-episode random rail height
        self._lut = None
        if self.cfg.z_rail_randomize:
            _d = np.load(_resolve(self.cfg.ride_height_lut))
            self._lut = dict(H=_d["H"], hinges=_d["hinges"], ctrl=_d["ctrl"])

        # foot spheres + floor (sim contact, reward-only). The TOE sphere is the walking contact;
        # the HEEL sphere is a passive floor stop so the foot can never clip through the ground.
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_gids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_col")
                          for s in "LR"]
        self.foot_gids_arr = np.array(self.foot_gids)
        self._toe_r = float(self.model.geom_size[self.foot_gids[0]][0])
        self._col_gids = {}
        for s in "LR":
            for kind in ("foot", "heel"):
                g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{kind}_{s}_col")
                if g >= 0:
                    self._col_gids[g] = float(self.model.geom_size[g][0])
        self._air_time = np.zeros(2, np.float32)      # continuous seconds NOT grounded, per foot
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
        # torque-budget curriculum: keep the model's ORIGINAL forcerange so the callback can scale
        # it down (and restore) via set_torque_limit. 1.0 = full torque.
        self._orig_forcerange = self.model.actuator_forcerange.copy()
        self._torque_scale = 1.0
        self.height_target = float(self.default_qpos[2])
        # optional STIFFER passive ankle spring (m3 sagittal-balance experiment): a firmer foot
        # lever = more passive pitch-restoring torque in stance. The standing ankle sits well off
        # the spring's rest angle (loaded ~12.8 N*m), so raising k alone would balloon that preload
        # and topple the robot -> also shift springref (model.qpos_spring) to PRESERVE the standing
        # preload k*(q_stand - ref), leaving posture unchanged while only the restoring gain rises.
        # Applied before _stand_torque below so the holding-torque baseline reflects the new spring.
        if self.cfg.ankle_stiffness > 0.0:
            k_new = float(self.cfg.ankle_stiffness)
            for j in _ankle_j:
                qadr = int(self.model.jnt_qposadr[j])
                k_old = float(self.model.jnt_stiffness[j])
                ref_old = float(self.model.qpos_spring[qadr])
                q_stand = float(self.default_qpos[qadr])
                self.model.qpos_spring[qadr] = q_stand - (k_old / k_new) * (q_stand - ref_old)
                self.model.jnt_stiffness[j] = k_new
        if self.cfg.ankle_damping > 0.0:
            for j in _ankle_j:
                self.model.dof_damping[int(self.model.jnt_dofadr[j])] = float(self.cfg.ankle_damping)
        # standing-baseline holding torque: the torque penalty prices torque ABOVE this, so
        # single-support stance isn't taxed into being strictly worse than double-support skating.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        self.data.ctrl[:] = self.nominal_ctrl
        mujoco.mj_forward(self.model, self.data)
        self._stand_torque = self.data.actuator_force[:self.nu].copy()
        self.hip_roll_idx = np.array(
            [a for a in range(self.nu)
             if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "")
             .startswith("hip_roll")], dtype=int)

        # ----- action / observation spaces -----
        self.action_dim = fourier_gait.action_dim(self.cfg.n_harmonics)
        self.spec_dim = fourier_gait.spec_dim(self.cfg.n_harmonics)
        self.action_space = spaces.Box(-1.0, 1.0, (self.action_dim,), np.float32)
        self._prev_action = np.zeros(self.action_dim, np.float32)   # policy output (obs)
        self._prev_applied = np.zeros(self.action_dim, np.float32)  # post-delay, for coef_rate
        self._prev_motor_cmd = np.zeros(self.nu, np.float32)        # normalized targets, action_rate
        self._prev_residual = np.zeros(self.nu, np.float32)         # for the residual-rate penalty
        self._residual_rate_sq = 0.0
        self._coef_rate_gated = 0.0
        self._phase = 0.0                # gait phase, continuous, kept in [0, 2*pi)
        self._phase_reward = 0.0         # the phase the current step's targets were assembled at
        self._task = np.zeros(2, np.float32)   # [run_flag, dist_to_go/100]

        # frame: pos6 vel6 trq6 grav3 gyro3 vbody3 phase2 task2 prev_action
        self.frame_dim = 6 + 6 + 6 + 3 + 3 + 3 + 2 + 2 + self.action_dim
        obs_dim = self.frame_dim * self.cfg.history_len
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self._history = np.zeros((self.cfg.history_len, self.frame_dim), np.float32)

        self._filt_target = self.nominal_ctrl.copy()
        # motor velocity/accel limiter state: the slew limiter in _run_physics tracks the previously
        # COMMANDED target position + velocity so it can cap joint velocity/acceleration.
        self._vel_accel_limited = (self.cfg.motor_vel_limit > 0.0
                                   or self.cfg.motor_accel_limit > 0.0)
        self._prev_cmd_pos = self.nominal_ctrl.copy()
        self._prev_cmd_vel = np.zeros(self.nu)
        self._delay_buf = [np.zeros(self.action_dim, np.float32)
                           for _ in range(self.cfg.action_delay_steps)]
        self._step_n = 0
        self._elapsed_t = 0.0
        # curriculum state — defaults are the FINAL curriculum point (hardest task), correct for
        # standalone/eval use; during training the RampCallbacks overwrite these via env_method
        # before the very first rollout (evaluate.py additionally restores a mid-training run's
        # values from its curriculum.json).
        self._stance_ratio = float(self.cfg.stance_ratio_final)
        self._eff_scale = 1.0
        # pitch-assist scale: 0 = final/hardest (no training-wheel). The RampCallback drives it
        # 1 -> 0 during training when the preset enables it; eval/standalone leaves it at 0.
        self._pitch_assist = 0.0
        self._assist_torque = 0.0        # the N*m the assist applied this step (for w_assist_penalty)
        # pitch slow-motion: extra armature on the base pitch DOF, faded 1 -> 0 by the curriculum.
        # 0 = final/hardest (real dynamics); the RampCallback drives the scale during training.
        self._base_pitch_armature = float(self.model.dof_armature[self._base_pitch_dadr])
        self._armature_scale = 0.0
        # sim2real control-timing randomization (curriculum-driven; 0 = off). jitter is in sim
        # substeps (sim_dt = 1 ms, so substeps == ms); drop is the per-step hold-last-action prob.
        self._ctrl_jitter_substeps = 0
        self._ctrl_drop_prob = 0.0
        # optional zero-arg hook fired once per control step (frame capture / metrics / pacing)
        self.on_control_step = None

    # ---------- curriculum hooks (VecEnv.env_method reaches SubprocVecEnv workers) ----------
    def set_sprint_dist(self, d):
        """Move the sprint finish line. Applies from the NEXT reset — never mid-dash."""
        self.cfg.sprint_dist_m = float(d)

    def set_stance_ratio(self, r):
        """Set the expected stance duty factor of the phase-gated contact schedule (< 0.5 opens
        a double-swing flight window). Takes effect immediately (reward-only)."""
        self._stance_ratio = float(r)

    def set_efficiency_scale(self, s):
        """Set the 0..1 multiplier on the efficiency terms (torque/motor_vel/energy)."""
        self._eff_scale = float(np.clip(s, 0.0, 1.0))

    def set_pitch_assist(self, s):
        """Set the 0..1 scale on the decaying pitch-assist training-wheel (1 = full help at the
        start of the m2->m3 bridge, 0 = off / self-sufficient). Takes effect immediately."""
        self._pitch_assist = float(np.clip(s, 0.0, 1.0))

    def set_pitch_armature(self, s):
        """Set the 0..1 scale on the extra base-pitch armature (slow-motion curriculum): 1 = full
        extra rotor inertia (sluggish fall), 0 = real dynamics. Writes model.dof_armature so the
        next mj_step's mass matrix picks it up. NOT a crutch (adds inertia, never holds position)."""
        self._armature_scale = float(np.clip(s, 0.0, 1.0))
        self.model.dof_armature[self._base_pitch_dadr] = \
            self._base_pitch_armature + self._armature_scale * self.cfg.pitch_armature

    def set_torque_limit(self, scale):
        """Scale the actuator torque budget (forcerange) to `scale` x the model's original limits
        (clamped to [torque_limit_floor, 1]). 1.0 = full torque; <1 = tighter budget (the torque-
        efficiency curriculum). A reduced budget is a real motor constraint — MuJoCo clips the
        actuator force to the new range from the next mj_step."""
        self._torque_scale = float(np.clip(scale, self.cfg.torque_limit_floor, 1.0))
        self.model.actuator_forcerange[:] = self._orig_forcerange * self._torque_scale

    def set_ctrl_jitter(self, ms):
        """Set the +- control-timing jitter (ms; sim_dt=1 ms so this is +- substeps per control step)."""
        self._ctrl_jitter_substeps = int(round(max(0.0, float(ms))))

    def set_ctrl_drop(self, p):
        """Set the per-control-step probability of a DROPPED inference (hold the last action)."""
        self._ctrl_drop_prob = float(np.clip(p, 0.0, 1.0))

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

    def _vel_body(self):
        return self._base_rot().T @ self.data.qvel[0:3]

    def _foot_contacts(self):
        """Which foot-tip spheres touch the floor (sim contact; reward-only)."""
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
        return self.data.geom_xpos[self.foot_gids_arr, 2] - self._toe_r

    def _foot_lateral_sep(self):
        """Body-frame lateral separation of the toe spheres, sep = y_left - y_right (~0.40 m
        nominal; sep < stance_min_sep means the legs are coming together / crossing)."""
        R = self._base_rot()
        base = self.data.qpos[0:3]
        y = [(R.T @ (self.data.geom_xpos[fg] - base))[1] for fg in self.foot_gids]  # [L, R]
        return float(y[0] - y[1])

    def _floor_violation(self):
        """A foot collision sphere has sunk past half its radius into the floor — the solver is
        being driven through the ground."""
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = (con.geom1, con.geom2)
            if self.floor_gid in pair:
                for g, r in self._col_gids.items():
                    if g in pair and con.dist < -0.5 * r:
                        return True
        return False

    # ---------- observation ----------
    def _proprio(self):
        s = self.cfg.obs_scales
        motor_pos = (self.data.qpos[self.act_qadr] - self.default_motor_pos) * s["motor_pos"]
        motor_vel = self.data.qvel[self.act_dadr] * s["motor_vel"]
        motor_trq = self.data.actuator_force[:self.nu] * s["motor_torque"]
        grav = self._gravity_body() * s["gravity"]
        angv = self._ang_vel_body() * s["ang_vel"]
        vbody = self._vel_body() * s["base_vel"]
        phase = np.array([np.sin(self._phase), np.cos(self._phase)])
        return np.concatenate([motor_pos, motor_vel, motor_trq, grav, angv, vbody,
                               phase, self._task, self._prev_action]).astype(np.float32)

    def _obs(self):
        return self._history.reshape(-1).astype(np.float32)

    def _push_frame(self, frame):
        self._history[:-1] = self._history[1:]
        self._history[-1] = frame

    def _update_task(self):
        """Refresh the 2-dim task channel: [run_flag, dist_to_go/100]. The run->stop flip at the
        line is the policy's stop signal; dist_to_go lets it SEE the line coming (plan braking)
        and gives the value function the state its return actually depends on."""
        if self.cfg.objective == "sprint":
            if self._sprint_crossed:
                self._task[:] = 0.0
            else:
                self._task[0] = 1.0
                self._task[1] = np.clip((self._sprint_D - self._sprint_d) / 100.0, 0.0, 1.0)
        else:                       # speed: run forever
            self._task[:] = 1.0

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        # activate this milestone's base-DOF locks (loop-closure equalities stay untouched)
        self.data.eq_active[self.lock_eq_ids] = self.base_lock
        # m1: rail Z at a per-episode RANDOM ride height, legs seated from the LUT so the episode
        # starts in a valid on-floor stance; otherwise a locked Z pins at the natural stance height.
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
        # per-EPISODE standing-torque baseline (captured on the CLEAN stance, before reset noise):
        # at an m1 LUT-seated ride height the holding torques differ from the keyframe's, and a
        # stale baseline would bill normal stance once the efficiency terms ramp in.
        self.data.ctrl[:] = self.nominal_ctrl
        mujoco.mj_forward(self.model, self.data)
        self._stand_torque = self.data.actuator_force[:self.nu].copy()
        n = self.cfg.reset_joint_noise
        self.data.qpos[self.hinge_qadr_start:] += self.np_random.uniform(
            -n, n, self.model.nq - self.hinge_qadr_start)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._prev_action[:] = 0.0
        self._prev_applied[:] = 0.0
        self._prev_motor_cmd[:] = 0.0
        self._prev_residual[:] = 0.0
        self._coef_rate_gated = 0.0
        self._phase = 0.0
        self._phase_reward = 0.0
        self._elapsed_t = 0.0
        self._filt_target[:] = self.nominal_ctrl
        self._prev_cmd_pos[:] = self.nominal_ctrl
        self._prev_cmd_vel[:] = 0.0
        self._delay_buf = [np.zeros(self.action_dim, np.float32)
                           for _ in range(self.cfg.action_delay_steps)]
        self._air_time[:] = 0.0
        self._contact_time[:] = 0.0
        self._grounded_prev = self._foot_contacts() | (self._toe_heights() < self.cfg.grounded_h)
        self._prev_toe_xy = self.data.geom_xpos[self.foot_gids_arr, 0:2].copy()
        self._push_countdown = self._next_push_in()
        self._step_n = 0
        # sprint state: the finish line is frozen per episode (curriculum moves it between dashes)
        self._x0 = float(self.data.qpos[0])
        self._sprint_D = float(self.cfg.sprint_dist_m)
        self._sprint_crossed = False
        self._sprint_t_line = None
        self._sprint_d = 0.0
        self._stop_hold = 0.0
        self._update_task()
        frame = self._proprio()
        self._history[:] = frame
        return self._obs(), {}

    def _next_push_in(self):
        c = self.cfg
        if c.push_interval_s <= 0:
            return 10 ** 9
        s = c.push_interval_s * self.np_random.uniform(0.7, 1.3)
        return max(1, int(round(s / self.control_dt)))

    def _update_sprint(self):
        """One control step of sprint bookkeeping: distance, the line-crossing latch (freezes the
        dash time + flips the task channel to 'stop'), the stopped-hold success detector."""
        self._sprint_d = float(self.data.qpos[0]) - self._x0
        if not self._sprint_crossed and self._sprint_d >= self._sprint_D:
            self._sprint_crossed = True          # latched: recrossing backward doesn't un-finish
            self._sprint_t_line = self._elapsed_t
        if self._sprint_crossed:
            vx = float(self._vel_body()[0])
            if abs(vx) <= self.cfg.stop_speed_eps:
                self._stop_hold += self.control_dt
                if self._stop_hold >= self.cfg.stop_hold_s:
                    return True
            else:
                self._stop_hold = 0.0
        return False

    def _run_physics(self, target6):
        """One control step of plant: EMA-filter the target, clip to ctrlrange, run
        control_decimation sim substeps (OR-accumulating foot contact so a sub-20 ms hop can't
        pass as continuous flight/contact at the 50 Hz boundary)."""
        c = self.cfg
        self._filt_target = c.action_filter * self._filt_target + (1 - c.action_filter) * target6
        tgt = np.clip(self._filt_target, self.ctrl_lo, self.ctrl_hi)
        # motor velocity + acceleration limits: slew-limit the commanded target so joint velocity
        # <= motor_vel_limit and its rate of change <= motor_accel_limit (a velocity/accel-bounded
        # position servo = the real moteus limits). Trapezoidal profile via the previous commanded
        # velocity; result stays inside ctrlrange (it interpolates between two in-range targets).
        if self._vel_accel_limited:
            dt = self.control_dt
            v_des = (tgt - self._prev_cmd_pos) / dt
            if c.motor_accel_limit > 0.0:
                dv = c.motor_accel_limit * dt
                v_des = np.clip(v_des, self._prev_cmd_vel - dv, self._prev_cmd_vel + dv)
            if c.motor_vel_limit > 0.0:
                np.clip(v_des, -c.motor_vel_limit, c.motor_vel_limit, out=v_des)
            tgt = self._prev_cmd_pos + v_des * dt
            self._prev_cmd_vel = v_des
            self._prev_cmd_pos = tgt.copy()
        self.data.ctrl[:] = tgt
        # sim2real timing jitter: vary the substep count (control period) by +-jitter ms. The gait
        # phase clock still advances by the NOMINAL control_dt in step() -> models the real mismatch
        # between the Pi's fixed-rate gait clock and its jittery actual loop timing.
        n = c.control_decimation
        if self._ctrl_jitter_substeps > 0:
            n = max(1, n + int(self.np_random.integers(
                -self._ctrl_jitter_substeps, self._ctrl_jitter_substeps + 1)))
        contact_acc = np.zeros(2, bool)
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if not contact_acc.all():
                contact_acc |= self._foot_contacts()
        return contact_acc

    def step(self, action):
        c = self.cfg
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        # dropped inference (sim2real): with prob ctrl_drop_prob the Pi missed its deadline this
        # step, so no new command is produced -> hold the last policy output (moteus keeps its target).
        if self._ctrl_drop_prob > 0.0 and self.np_random.random() < self._ctrl_drop_prob:
            action = self._prev_action.copy()
        # fixed actuation delay (plant truth: Pi inference + moteus/CAN is ~one 50 Hz step)
        self._delay_buf.append(action)
        applied = self._delay_buf.pop(0)
        cam_c, thigh_c, freq_raw, reflex, residual = fourier_gait.decode(applied, c.n_harmonics)
        f = fourier_gait.frequency(freq_raw, c.gait_freq_hz)
        phase_used = self._phase
        self._phase_reward = phase_used
        # phase-gated gait-SPEC change penalty state: rewriting the spec exactly at the cycle
        # boundary (phase ~ 0 == 2pi) is FREE; mid-cycle rewrites pay. Residual dims are per-step
        # by design and NOT billed here.
        d_spec = applied[:self.spec_dim] - self._prev_applied[:self.spec_dim]
        self._coef_rate_gated = float(np.sum(d_spec ** 2)) * float(np.sin(phase_used / 2.0) ** 2)
        self._prev_applied = applied.copy()
        grav = self._gravity_body()
        angv = self._ang_vel_body()
        roll = float(grav[1])            # ~roll angle (small-angle: grav_y)
        roll_rate = float(angv[0])       # roll rate (gyro x)
        pitch = float(grav[0])           # ~pitch angle (grav_x ~ sin(pitch), + = nose-down)
        pitch_rate = float(angv[1])      # pitch rate (gyro y)
        target6 = fourier_gait.assemble(cam_c, thigh_c, reflex, phase_used,
                                        roll, roll_rate, self.nominal_ctrl, c,
                                        pitch=pitch, pitch_rate=pitch_rate)
        target6 = target6 + c.residual_scale * residual  # the per-step fast-feedback channel
        motor_cmd = ((target6 - self.nominal_ctrl) / c.action_scale).astype(np.float32)
        self._residual_sq = float(np.sum(residual ** 2))
        # per-step residual CHANGE (for the residual-rate penalty that suppresses fast chatter)
        self._residual_rate_sq = float(np.sum((residual - self._prev_residual) ** 2))
        self._prev_residual = residual.copy()

        # gentle random shove BEFORE the physics runs (free translational axes only)
        self._push_countdown -= 1
        if self._push_countdown <= 0:
            ang = self.np_random.uniform(0.0, 2.0 * np.pi)
            if not self.base_lock[0]:
                self.data.qvel[self._base_x_dadr] += c.push_dv * np.cos(ang)
            if not self.base_lock[1]:
                self.data.qvel[self._base_y_dadr] += c.push_dv * np.sin(ang)
            self._push_countdown = self._next_push_in()

        # decaying pitch-assist (m2->m3 bridge): external spring-damper torque on the base pitch
        # joint toward level, scaled by the curriculum (1 -> 0 over training). Written EVERY step
        # (0 when faded/disabled) so a stale qfrc_applied can never linger; held across the physics
        # substeps. Sim-only helper -> the final assist=0 policy is hardware-valid.
        if c.pitch_assist_kp > 0.0:
            pq = float(self.data.qpos[self._base_pitch_qadr])
            pqd = float(self.data.qvel[self._base_pitch_dadr])
            self._assist_torque = -self._pitch_assist * (c.pitch_assist_kp * pq
                                                         + c.pitch_assist_kd * pqd)
            self.data.qfrc_applied[self._base_pitch_dadr] = self._assist_torque

        # ankle-torque reflex (emulates an ACTUATED ankle): a pitch-restoring torque at the ankle
        # joints, applied only to a GROUNDED foot (ankle strategy works only in stance). Mirrored
        # L/R axes -> +u on L, -u on R. Written every step (0 when off/airborne) so no stale torque.
        if c.ankle_kp > 0.0:
            u_ank = -float(np.clip(c.ankle_kp * pitch + c.ankle_kd * pitch_rate,
                                   -c.ankle_clip, c.ankle_clip))
            gnd = self._foot_contacts()
            self.data.qfrc_applied[self._ankle_dadr[0]] = u_ank if gnd[0] else 0.0
            self.data.qfrc_applied[self._ankle_dadr[1]] = -u_ank if gnd[1] else 0.0

        contact_acc = self._run_physics(target6)
        self._elapsed_t += self.control_dt
        finished = c.objective == "sprint" and self._update_sprint()
        # advance the gait phase AFTER assembly (the obs frame carries the NEXT step's phase)
        self._phase = (self._phase + 2.0 * np.pi * f * self.control_dt) % (2.0 * np.pi)

        self._step_n += 1
        self._update_task()
        self._push_frame(self._proprio())
        reward, terms = self._reward(motor_cmd, contact_acc)
        # rate-invariance: scale the summed per-step reward by control_dt/0.02 so the per-SECOND
        # income/penalty is the same at any control rate, while the fall/finish EVENTS below stay
        # fixed. No-op at 50 Hz. (The individual `terms` stay raw per-step for the gate/plot logic.)
        reward *= self._reward_dt_scale
        # global per-step floor (2nd level of suicide-proofing, see config.py): per-term caps
        # bound each term but not the SUM — unfloored, a standing pre-locomotion policy's ~-2.2/step
        # made diving value-optimal. Applied BEFORE the terminal bonus/penalty (also dt-scaled).
        reward = max(reward, -c.step_reward_floor * self._reward_dt_scale)
        terminated = self._fallen()
        if terminated:
            reward -= c.fall_penalty
        elif finished:
            terminated = True
            reward += c.finish_bonus
        truncated = self._step_n >= self.max_steps
        self._prev_action[:] = action
        self._prev_motor_cmd[:] = motor_cmd
        if self.on_control_step is not None:
            self.on_control_step()
        info = {"reward_terms": terms}
        # mean actuator torque utilization |tau|/limit (for the torque-budget curriculum callback)
        _lim = self.model.actuator_forcerange[:self.nu, 1]
        info["torque_util"] = float(np.mean(
            np.abs(self.data.actuator_force[:self.nu]) / np.maximum(_lim, 1e-6)))
        if c.objective == "sprint":
            info["sprint"] = self._sprint_info(finished)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    # ---------- reward ----------
    def _pen(self, v):
        """Floor a penalty term (reward normalization is OFF; no reachable state may make dying
        cheaper than living — suicide-proofing against the raw fall penalty)."""
        return max(float(v), -self.cfg.penalty_term_cap)

    def _reward(self, motor_cmd, contact_acc):
        c = self.cfg
        v_body = self._vel_body()
        vx = v_body[0]
        angv = self._ang_vel_body()
        grav = self._gravity_body()
        run_phase = c.objective == "speed" or not self._sprint_crossed
        t = {}

        # ----- objective income -----
        if run_phase:
            # SYMMETRIC clip: backward motion pays negative income (a one-sided clip makes
            # shuttling in front of the line strictly out-value crossing it — see config.py)
            speed_income = c.w_fwd_speed * float(np.clip(vx, -c.v_ceiling, c.v_ceiling))
            if c.speed_upright_gate and speed_income > 0.0:
                # gate ONLY forward income by uprightness: a toppling robot must not bank speed
                # reward on the way down. -grav[2] is 1 upright, falling toward c0 (=-term_gravity_z)
                # at the ~60 deg tip-over termination, so income fades smoothly to 0 as it tips.
                # Backward income is left fully negative (gating it would shrink a topple's penalty).
                u = np.clip((-grav[2] - c.speed_upright_c0) / (1.0 - c.speed_upright_c0), 0.0, 1.0)
                speed_income *= float(u) ** c.speed_upright_k
            t["fwd_speed"] = speed_income
            t["stop"] = 0.0
            t["overrun"] = 0.0
            progress_frac = float(np.clip(vx / c.v_ceiling, 0.0, 1.0))
        else:                       # sprint stop phase: income flips to 'be stationary'
            t["fwd_speed"] = 0.0
            t["stop"] = c.w_stop_vel * float(np.exp(-((vx / c.stop_sigma) ** 2)))
            over = max(0.0, self._sprint_d - (self._sprint_D + c.sprint_brake_m))
            t["overrun"] = self._pen(-c.w_overrun * over)
            progress_frac = 0.0
        # the clock: what actually prices the dash TIME (sum(vx)*dt integrates to the distance
        # whatever the pace; sum(-w_time) = -w_time * T). Paid in BOTH sprint phases; never in speed.
        t["time"] = -c.w_time if c.objective == "sprint" else 0.0
        t["alive"] = c.w_alive

        # ----- gait shaping (anti-skate + phase schedule) -----
        cmd_speed = c.v_ceiling if run_phase else 0.0
        gait_on = cmd_speed >= c.gait_cmd_gate
        dt = self.control_dt
        toe_pos = self.data.geom_xpos[self.foot_gids_arr].copy()
        heights = toe_pos[:, 2] - self._toe_r
        grounded = contact_acc | (heights < c.grounded_h)
        grounded_recent = grounded | self._grounded_prev

        # foot slip: horizontal toe speed over the control step, billed only if grounded at BOTH
        # ends (a landing foot arrives with legitimate swing speed and must not be billed for it)
        slip_v = np.linalg.norm(toe_pos[:, 0:2] - self._prev_toe_xy, axis=1) / dt
        slip = np.where(grounded & self._grounded_prev,
                        np.maximum(0.0, slip_v - c.slip_deadband) ** 2, 0.0)
        t["foot_slip"] = -min(c.w_foot_slip * float(slip.sum()), c.penalty_term_cap)

        # air/stance clocks + one-sided capped touchdown credit (before the clocks advance)
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
        t["air_time"] = air

        # per-foot stance-time cap: any foot grounded longer than the allowance pays per step
        if gait_on:
            cap = c.stance_cap_s if cmd_speed >= c.stance_slow_speed else c.stance_cap_slow_s
            over = np.minimum(np.maximum(self._contact_time - cap, 0.0), 1.0)
            t["stance_time"] = -min(c.w_stance_time * float(over.sum()), c.penalty_term_cap)
        else:
            t["stance_time"] = 0.0

        # swing clearance: fresh swings only, above the ghost-drag band, scaled by progress
        clear = 0.0
        if gait_on:
            for i in range(2):
                if not grounded_recent[i] and 0.0 < self._air_time[i] <= c.swing_fresh_s:
                    frac = np.clip((float(heights[i]) - c.clearance_dead_m)
                                   / c.clearance_scale_m, 0.0, 1.0)
                    clear += c.w_clearance * float(frac) * (0.3 + 0.7 * progress_frac)
        t["clearance"] = clear

        # phase-gated contact schedule (Siekmann): each foot pays for being grounded during its
        # expected SWING window. The windows use the SAME phase + antiphase convention as the
        # action, so the demanded schedule is exactly the one the gait generator is producing.
        # With stance_ratio < 0.5 the swing windows overlap -> ground contact by EITHER foot in
        # the overlap pays -> a flight phase is demanded (this is the term that asks for running).
        if gait_on and c.w_phase_contact > 0.0:
            sr = self._stance_ratio
            sw_L = 1.0 - fourier_gait.stance_indicator(self._phase_reward, sr)
            sw_R = 1.0 - fourier_gait.stance_indicator(self._phase_reward + np.pi, sr)
            pen = sw_L * float(grounded[0]) + sw_R * float(grounded[1])
            t["phase_contact"] = -min(c.w_phase_contact * pen, c.penalty_term_cap)
        else:
            t["phase_contact"] = 0.0

        # foot-placement ahead of CoM (capture step): credit a foot that LANDS ahead of the whole-
        # robot CoM in the heading direction (world +x; yaw is locked m3..m5). Touchdown-only (fresh
        # air->ground, grounded & ~grounded_prev) so it rewards actively stepping the foot out to
        # catch the CoM, not a static forward-foot lean (which a held reward would breed). CoM from
        # subtree_com[0] (whole model, valid after the physics step). Off (0) for m1/m2 by default.
        if c.w_foot_ahead > 0.0:
            com_x = float(self.data.subtree_com[0][0])
            td = grounded & (~self._grounded_prev)
            ahead = 0.0
            for i in range(2):
                if td[i]:
                    ahead += min(max(float(toe_pos[i, 0] - com_x), 0.0), c.foot_ahead_cap_m)
            t["foot_ahead"] = c.w_foot_ahead * ahead
        else:
            t["foot_ahead"] = 0.0

        # cadence / anti-chatter: penalize each foot that flips grounded<->airborne this control
        # step -> fewer, longer steps (minimise stepping frequency). phase_contact still demands
        # swing, so the equilibrium is a slower gait, not a skate.
        if c.w_contact_switch > 0.0:
            t["step_rate"] = self._pen(-c.w_contact_switch
                                       * float(np.sum(grounded != self._grounded_prev)))
        else:
            t["step_rate"] = 0.0

        self._grounded_prev = grounded
        self._prev_toe_xy = toe_pos[:, 0:2]

        # ----- efficiency (Cassie-100m recipe; ramped in by the curriculum callback) -----
        tau = self.data.actuator_force[:self.nu]
        qd = self.data.qvel[self.act_dadr]
        exc = np.maximum(np.abs(tau) - np.abs(self._stand_torque), 0.0)
        es = self._eff_scale
        t["torque"] = self._pen(-es * c.w_torque * float(np.sum(exc ** 2)))
        t["motor_vel"] = self._pen(-es * c.w_motor_vel * float(np.sum(qd ** 2)))
        t["energy"] = self._pen(-es * c.w_energy * float(np.sum(np.maximum(tau * qd, 0.0))))

        # ----- smoothness -----
        t["action_rate"] = self._pen(-c.w_action_rate
                                     * float(np.sum((motor_cmd - self._prev_motor_cmd) ** 2)))
        t["coef_rate"] = self._pen(-c.w_coef_rate * self._coef_rate_gated)
        t["residual"] = self._pen(-c.w_residual * self._residual_sq)
        t["residual_rate"] = self._pen(-c.w_residual_rate * self._residual_rate_sq)
        # anti-crutch: pay for the assist torque the policy provokes (0 when it balances itself, so
        # the assist becomes a safety net the policy is pushed to stop relying on). 0 when disabled.
        t["assist_pen"] = self._pen(-c.w_assist_penalty * self._assist_torque ** 2)

        # ----- posture -----
        t["upright"] = self._pen(-c.w_upright * (grav[0] ** 2 + grav[1] ** 2))
        if self.z_locked:           # height/vz are meaningless when Z is railed
            t["height"] = 0.0
            t["vz"] = 0.0
        else:
            t["height"] = self._pen(-c.w_height * (self.data.qpos[2] - self.height_target) ** 2)
            t["vz"] = self._pen(-c.w_vz * self.data.qvel[2] ** 2)
        t["lat_vel"] = self._pen(-c.w_lat_vel * v_body[1] ** 2)
        t["ang_xy"] = self._pen(-c.w_angvel_xy * (angv[0] ** 2 + angv[1] ** 2))
        # centroidal angular-momentum regulation (mj_subtreeVel -> subtree_angmom about the CoM,
        # world frame): penalize whole-robot pitch-axis (world Y) angular momentum so the gait's
        # foot impulses average out to a body that isn't tumbling. Pitch component only while
        # yaw/roll are locked (m3..m5); for m6 (yaw free) also add L[0]^2. Guarded so m1/m2 skip
        # the O(nbody) call and pay nothing.
        if c.w_angmom > 0.0:
            mujoco.mj_subtreeVel(self.model, self.data)
            L = self.data.subtree_angmom[self.base_id]
            t["angmom"] = self._pen(-c.w_angmom * float(L[1] ** 2))
        else:
            t["angmom"] = 0.0
        sep = self._foot_lateral_sep()
        t["stance"] = self._pen(-c.w_no_cross * max(0.0, c.stance_min_sep - sep) ** 2)
        hr = self.data.qpos[self.act_qadr[self.hip_roll_idx]] \
            - self.default_motor_pos[self.hip_roll_idx]
        t["hip_roll"] = self._pen(-c.w_hip_roll * float(np.sum(hr ** 2)))

        return float(sum(t.values())), t

    def _sprint_info(self, finished):
        """Dash telemetry for eval tooling."""
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
        if self._floor_violation():
            return True
        return False
