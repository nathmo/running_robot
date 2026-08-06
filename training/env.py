"""DashEnv — Gymnasium environment for DASH-01.

Three objectives share one plant:
  "sprint"  — the 100 m dash (stand, run the line, stop past it). The original task.
  "speed"   — endless max-forward-speed (gait-shaping / debug).
  "command" — JOYSTICK teleoperation: track a commanded forward speed and yaw rate, and stand
              still (stepping in place is allowed) when the stick is centred. This is the mode
              built for hardware demos, and it is the only one that is sim2real-honest:
              privileged observations are off, the plant is randomized every episode and the
              proprioception is corrupted by a measurement model (see domain_rand.py).

One env.step == one control step. The action is the per-step Fourier gait spec + steering
asymmetry + residuals (see fourier_gait.py): the policy re-parameterizes a phase-driven gait
generator every step (CPG-RL-style) and adds small direct target corrections (PMTG-style).

Observation. In command mode the frame is proprioception the real robot can actually measure —
motor pos/vel/torque, IMU gravity + gyro — plus the gait phase (computed onboard, not measured),
the command, and the previous action. Base linear velocity, which every earlier milestone fed the
policy as privileged sim state, is REMOVED: it is not measurable without an estimator, and a
policy that has only ever seen ground truth has never seen the signal it will be given. What
replaces it is a longer strided history (cfg.history_len x cfg.history_stride), from which the
velocity is observable in principle. Set cfg.obs_base_vel=True to put the oracle back for an
ablation baseline.

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
import cpg_gait
from domain_rand import PlantRandomizer, SensorNoise

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
        # The MEASURED drive, expressed in Hz and converted at THIS control rate (see the
        # drive_bandwidth_hz note in config.py). Resolved before anything reads action_filter /
        # action_delay_steps, and written back onto cfg so resolved_config.json records what
        # actually ran rather than the 0 sentinel.
        if self.cfg.drive_bandwidth_hz > 0.0:
            tau = 1.0 / (2.0 * np.pi * float(self.cfg.drive_bandwidth_hz))
            self.cfg.action_filter = float(np.exp(-self.control_dt / tau))
        if self.cfg.drive_delay_ms > 0.0:
            self.cfg.action_delay_steps = int(round(
                float(self.cfg.drive_delay_ms) * 1e-3 / self.control_dt))
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
        self._duty_ema = np.full(2, 0.5, np.float64)  # per-foot grounded-fraction EMA (duty_sym term)
        self._ws_out_t = np.zeros(2)                  # per-foot continuous time outside the workspace
        self._ws_ref = None                           # LUT nominal_toe (base frame) for workspace-kill
        if self.cfg.workspace_kill:
            _lut = np.load(str(PKG_DIR / "model" / "cpg_foot_lut.npz"), allow_pickle=True)
            self._ws_ref = np.asarray(_lut["nominal_toe"], float)
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
        self._sag_scale = 1.0      # bus-voltage droop (dr_torque_sag)
        self._sag_state = 0.0
        self.height_target = float(self.default_qpos[2])
        # optional STIFFER passive ankle spring (m3 sagittal-balance experiment): a firmer foot
        # lever = more passive pitch-restoring torque in stance. The standing ankle sits well off
        # the spring's rest angle (loaded ~12.8 N*m), so raising k alone would balloon that preload
        # and topple the robot -> also shift springref (model.qpos_spring) to PRESERVE the standing
        # preload k*(q_stand - ref), leaving posture unchanged while only the restoring gain rises.
        # Applied before _stand_torque below so the holding-torque baseline reflects the new spring.
        self._setup_ankle(_ankle_j)
        if self.cfg.ankle_resettle:
            self._resettle_keyframe()
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
        # Steering is OPT-IN: with steer_enable False the action has no steering dims at all, so
        # every pre-steering preset keeps its 24-dim action and its checkpoints keep loading.
        # Two gait generators, selected by cfg.action_mode. They have different action widths (and,
        # via prev_action + the phase channel, different obs widths), so a checkpoint never crosses
        # between them — a CPG run only ever warm-starts from a CPG run.
        self.cpg_mode = (self.cfg.action_mode == "cpg")
        if self.cpg_mode:
            self.n_steer = cpg_gait.N_STEER if self.cfg.steer_enable else 0
            self.action_dim = cpg_gait.action_dim(self.n_steer, self.cfg.cpg_residual)
            self.spec_dim = cpg_gait.spec_dim(self.n_steer)
            self._cpg_lut = cpg_gait.load_lut(self.cfg.cpg_lut)
            # the oscillator IS observable state (it lives in the controller, not the plant), so the
            # policy sees both leg phases and both amplitudes rather than one global clock
            self.phase_obs_dim = 6
        else:
            self.n_steer = fourier_gait.N_STEER if self.cfg.steer_enable else 0
            self.action_dim = fourier_gait.action_dim(self.cfg.n_harmonics, self.n_steer)
            self.spec_dim = fourier_gait.spec_dim(self.cfg.n_harmonics, self.n_steer)
            self._cpg_lut = None
            self.phase_obs_dim = 2
        # ACTIVE ANKLE: 2 extra action dims APPENDED after the gait generator's own. Both decoders
        # slice from the front and take exactly what they need, so a tail extension leaves every
        # existing preset's layout byte-identical -- the ankle is a separate channel bolted on, not
        # a change to the generator. That is what makes passive-vs-active a clean comparison: the
        # gait generator, the reward and the curriculum are the same in both arms; only the ankle
        # differs. (The action WIDTH still changes, so active runs are their own warm-start lineage.)
        self.gait_action_dim = self.action_dim
        self.action_dim += self.n_ankle_act
        self.action_space = spaces.Box(-1.0, 1.0, (self.action_dim,), np.float32)
        self._prev_action = np.zeros(self.action_dim, np.float32)   # policy output (obs)
        self._prev_applied = np.zeros(self.action_dim, np.float32)  # post-delay, for coef_rate
        self._prev_motor_cmd = np.zeros(self.nu, np.float32)        # normalized targets, action_rate
        # the residual is per GAIT actuator (the active ankle has its own channel, not a residual)
        self._prev_residual = np.zeros(self.n_gait_act, np.float32)  # for the residual-rate penalty
        self._residual_rate_sq = 0.0
        self._reflex_prate_filt = 0.0                               # pitch-reflex rate low-pass state
        self._coef_rate_gated = 0.0
        self._phase = 0.0                # fourier: the single global gait clock, kept in [0, 2*pi)
        self._phase_reward = 0.0         # the phase the current step's targets were assembled at
        # per-leg phases the reward's contact schedule is graded against. Fourier hard-codes the
        # right leg at +pi; the CPG carries two independent phases held near antiphase by coupling,
        # which is precisely the freedom being tested, so the reward must read them separately.
        self._phase_reward_R = np.pi
        # CPG oscillator state (r, rdot, theta), each [left, right]; unused in fourier mode
        self._cpg = (np.zeros(2), np.zeros(2), np.array([0.0, np.pi]))
        # ----- task / command channel -----
        # sprint+speed: [run_flag, dist_to_go/100]   command: [v_cmd/norm, yaw_cmd/norm, stand_flag]
        self.command_mode = (self.cfg.objective == "command")
        self.task_dim = 3 if self.command_mode else 2
        self._task = np.zeros(self.task_dim, np.float32)
        self._v_cmd = 0.0                # commanded forward speed, m/s (body x)
        self._yaw_cmd = 0.0              # commanded yaw rate, rad/s (body z)
        self._standing = False           # command is centred -> hold position
        self._stand_anchor = np.zeros(2)  # base xy latched when the stand command began
        self._cmd_countdown = 10 ** 9
        self._cmd_scale = 1.0            # 0..1 command-RANGE curriculum (set by the callback)
        self._track_err_sum, self._track_err_n = 0.0, 0

        # frame: pos_nu vel_nu trq_nu grav3 gyro3 [vbody3] phase2 task prev_action
        # (nu is 6 on the passive plants and 8 with actuated ankles — the ankle servo's encoder and
        # current are real onboard measurements, so the policy sees them like any other joint.)
        self.obs_base_vel = bool(self.cfg.obs_base_vel)
        self.frame_dim = (3 * self.nu + 3 + 3 + (3 if self.obs_base_vel else 0)
                          + self.phase_obs_dim + self.task_dim + self.action_dim)
        obs_dim = self.frame_dim * self.cfg.history_len
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        # strided history: keep (history_len-1)*stride+1 raw frames but expose only every stride-th,
        # so a fixed obs width can span a much longer window. At 200 Hz, len=10 x stride=4 = 200 ms,
        # which is what makes base velocity inferable once the privileged channel is removed.
        self._hist_stride = max(1, int(self.cfg.history_stride))
        self._hist_raw_len = (self.cfg.history_len - 1) * self._hist_stride + 1
        self._history = np.zeros((self._hist_raw_len, self.frame_dim), np.float32)
        self._hist_idx = ((self._hist_raw_len - 1)
                          - (np.arange(self.cfg.history_len) * self._hist_stride)[::-1])
        # measurement chain + per-episode plant draw (both inert unless enabled in the config)
        self._noise = SensorNoise(self.cfg, self.nu, control_dt=self.control_dt)
        _leg_dofs = [int(self.model.jnt_dofadr[j]) for j in range(self.model.njnt)
                     if self.model.jnt_bodyid[j] != self.base_id]
        _loop_sites = [s for s in (mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
                                   for n in ("pushrod_tip_L", "leg_anchor_L",
                                             "pushrod_tip_R", "leg_anchor_R")) if s >= 0]
        self._dr = PlantRandomizer(self.model, self.cfg, _ankle_j, _leg_dofs, _loop_sites)
        self._dr.stand_qpos = self.default_qpos
        self._dr_torque_scale = 1.0
        self._prev_vel_body = np.zeros(3)   # for the accelerometer-leak noise model
        self._obs_delay_buf = []
        self._ep_delay_steps = int(self.cfg.action_delay_steps)
        self._trip_left, self._trip_body, self._trip_force = 0, 0, 0.0
        self._fixed_base_h = None       # set by set_fixed_base() for the in-air test rig
        # foot BODY ids, for the trip disturbance (a toe catching an unseen obstacle)
        self._foot_bids = [int(self.model.geom_bodyid[g]) for g in self.foot_gids]

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

    # ---------- ankle-spring study ----------
    ANKLE_MODES = ("passive", "free", "rigid", "active", "active_spring", "bar")
    SHIN_BODIES = ("LegLeftNCS-v1", "LegRightNCS-v1")
    BAR_SAT_RAD = 0.035     # travel over which the strut reaches its full buckling load (~2 deg)

    def _ankle_inertia(self, dadr, tau=5.0, n=10):
        """Effective inertia the ankle sees IN LOADED STANCE, measured by impulse response.

        NOT the mass-matrix diagonal. M[dadr,dadr] is the inertia of the subtree distal to the
        joint — i.e. just the foot (~0.006 kg*m^2), which is the SWING inertia and has nothing to
        do with the resonance this damping has to kill. The m7 cadence bug was a LOADED spring
        ringing in stance, where the ankle is reacting against the ground and the inertia is the
        robot's, not the foot's; sizing damping off M[i,i] there lands ~40x under-damped, which is
        exactly the 1.6-N*m*s/rad mistake the 2026-07-24 sweep made.

        So measure it: apply a known torque at the ankle for a few ms from the settled stance and
        difference the resulting velocity against a zero-torque baseline (which cancels gravity,
        spring preload and contact transients), then I = tau*dt / dv. Contact, the closed leg loop
        and the rest of the robot are all included because they are all still in the sim."""
        def _vel(t):
            d = mujoco.MjData(self.model)
            mujoco.mj_resetDataKeyframe(self.model, d, self.key_id)
            mujoco.mj_forward(self.model, d)
            for _ in range(n):
                d.qfrc_applied[dadr] = t
                mujoco.mj_step(self.model, d)
            return float(d.qvel[dadr])

        dv = _vel(tau) - _vel(0.0)
        if abs(dv) < 1e-12:                      # welded/constrained ankle — no meaningful inertia
            return float('inf')
        return abs(tau * n * self.sim_dt / dv)

    def _setup_ankle(self, ankle_j):
        """Configure the ankle as an experimental variable: spring / no spring / welded / actuated.

        Deliberately FAILS LOUDLY on a mode/plant mismatch. Silently running "rigid" on a model with
        no lock equalities (or "active" on the 6-actuator plant) would produce a plausible-looking
        curve for the wrong arm, which is the one failure mode this study cannot survive."""
        c = self.cfg
        mode = str(c.ankle_mode)
        if mode not in self.ANKLE_MODES:
            raise ValueError(f"ankle_mode {mode!r} not in {self.ANKLE_MODES}")
        self.ankle_mode = mode

        self.ankle_act_idx = np.array(
            [a for a in range(self.nu)
             if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "")
             .startswith("ankle_")], dtype=int)
        self.n_ankle_act = int(len(self.ankle_act_idx))
        actuated = mode in ("active", "active_spring")
        if actuated and self.n_ankle_act != 2:
            raise ValueError(
                f"ankle_mode={mode!r} needs the actuated-ankle plant (2 'ankle_*' actuators, found "
                f"{self.n_ankle_act}). Set model_path='model/dash01_active.xml' "
                f"(generate it with `python -m model.make_ankle_variants`).")
        if not actuated and self.n_ankle_act:
            raise ValueError(
                f"ankle_mode={mode!r} but model_path has {self.n_ankle_act} ankle actuators — that "
                f"plant carries the ankle motors' mass, which would silently penalise a passive arm.")

        self._ankle_lock_eq = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"lock_ankle_{s}")
            for s in ("L", "R")], dtype=int)
        if mode == "rigid" and (self._ankle_lock_eq < 0).any():
            raise ValueError(
                "ankle_mode='rigid' needs the lock_ankle_L/R equalities — run "
                "`python -m model.make_ankle_variants` and point model_path at the patched model.")

        # ----- the tension-only strut ("bar") -----
        # Set up BEFORE stiffness so the k=0 branch below finds the geometry already resolved.
        # Sign convention, derived from the plant rather than hard-coded: the shipped spring pushes
        # the ankle TOWARDS springref, so the direction the ground loads the joint is the one AWAY
        # from springref, i.e. sign(q_stand - springref). The strut is in TRACTION on that side.
        self._bar_sign = np.zeros(len(ankle_j))
        self._bar_q0 = np.zeros(len(ankle_j))
        for i, j in enumerate(ankle_j):
            qadr = int(self.model.jnt_qposadr[j])
            q_stand = float(self.default_qpos[qadr])
            self._bar_q0[i] = q_stand              # taut at the flat-foot angle (= the lock angle)
            self._bar_sign[i] = np.sign(q_stand - float(self.model.qpos_spring[qadr])) or 1.0
        if mode == "bar":
            # Traction side = a HARD STOP: an inextensible strut, so the joint simply cannot travel
            # past the flat-foot angle under load. Implemented as a one-sided joint limit (the range
            # stays open on the compression side, where the buckling law below takes over).
            for i, j in enumerate(ankle_j):
                lo, hi = self.model.jnt_range[j]
                if self._bar_sign[i] > 0:
                    self.model.jnt_range[j] = (lo, self._bar_q0[i])
                else:
                    self.model.jnt_range[j] = (self._bar_q0[i], hi)
                self.model.jnt_limited[j] = 1
                # MuJoCo's DEFAULT limit softness (solref 0.02) lets body weight push 0.059 rad
                # (3.4 deg) past the stop — measured. That is not an inextensible bar, it is a
                # rubber one, and it would have quietly given the strut arm a compliant ankle:
                # the exact confound this study exists to avoid. Stiffened to the same solref/
                # solimp the lock_ankle equalities use, which brings penetration to ~0.
                self.model.jnt_solref[j] = (0.005, 1.0)
                self.model.jnt_solimp[j] = (0.95, 0.99, 0.001, 0.5, 2.0)

        # ----- stiffness -----
        # "free"/"active" mean k=0 EXACTLY. ankle_stiffness cannot express that (0 there is the
        # legacy "keep the model's 28.65" sentinel, kept so every m1..m7 preset keeps its meaning).
        # "rigid" zeroes it too: the joint cannot move, so a spring there is not physics, it is just
        # ~14 N*m of stance preload for the lock constraint to fight (and solver noise to explain).
        # "bar" has no spring at all -- its restoring law is the strut, applied per substep.
        if mode in ("free", "active", "rigid", "bar"):
            k_new = 0.0
        elif c.ankle_stiffness > 0.0:
            k_new = float(c.ankle_stiffness)
        else:
            k_new = None                              # keep whatever the model ships
        zero_preload = str(c.ankle_preload) == "zero"
        if zero_preload and mode in ("passive", "active_spring") and k_new is None:
            k_new = float(self.model.jnt_stiffness[ankle_j[0]])   # keep k, but re-reference it
        if k_new is not None:
            for j in ankle_j:
                qadr = int(self.model.jnt_qposadr[j])
                if k_new == 0.0:
                    self.model.jnt_stiffness[j] = 0.0
                    continue
                q_stand = float(self.default_qpos[qadr])
                if zero_preload:
                    # NO preload: the spring is at free length with the foot flat and unloaded, so
                    # it makes zero torque there and only resists deflection from it. Strictly
                    # weaker at stance than the same k preloaded -- that is the point of the arm.
                    self.model.qpos_spring[qadr] = q_stand
                else:
                    # preload-preserving: shift springref so k*(q_stand - ref) is unchanged, i.e.
                    # only the restoring GAIN rises and the standing posture does not move. Raising
                    # k alone balloons the ~14 N*m stance preload and flips the robot (2026-07-24).
                    k_old = float(self.model.jnt_stiffness[j])
                    ref_old = float(self.model.qpos_spring[qadr])
                    self.model.qpos_spring[qadr] = q_stand - (k_old / k_new) * (q_stand - ref_old)
                self.model.jnt_stiffness[j] = k_new

        # ----- distal mass: delete the spring assembly with the spring -----
        # Only legitimate when there IS no spring; charging a passive arm for hardware it needs (or
        # crediting it for hardware it still carries) is the one confound this study cannot survive.
        if c.ankle_spring_mass_kg > 0.0:
            if mode not in ("bar", "free", "rigid", "active"):
                raise ValueError(
                    f"ankle_spring_mass_kg={c.ankle_spring_mass_kg} with ankle_mode={mode!r} — that "
                    "arm still has a spring, so its mass cannot be removed.")
            for name in self.SHIN_BODIES:
                b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if b < 0:
                    raise ValueError(f"shin body {name!r} not found in {c.model_path}")
                m_old = float(self.model.body_mass[b])
                m_new = m_old - float(c.ankle_spring_mass_kg)
                if m_new <= 0.0:
                    raise ValueError(
                        f"ankle_spring_mass_kg={c.ankle_spring_mass_kg} exceeds the {name} mass "
                        f"({m_old:.3f} kg)")
                # inertia scaled by the mass ratio, exactly as apply_measured_masses.py does. The
                # CoM is left alone: we do not know where in the shin the spring sat.
                self.model.body_mass[b] = m_new
                self.model.body_inertia[b] *= m_new / m_old
        self.shin_mass = float(
            self.model.body_mass[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.SHIN_BODIES[0])])

        # ----- damping -----
        # ankle_zeta ties damping to the CURRENT k, so a stiffness sweep no longer also sweeps the
        # damping ratio (the confound that produced the m7 6 Hz spring-ring). Skipped at k=0, where
        # a ratio is undefined and the honest model of a floppy ankle is the joint's own friction.
        if c.ankle_zeta > 0.0:
            for j in ankle_j:
                dadr = int(self.model.jnt_dofadr[j])
                k = float(self.model.jnt_stiffness[j])
                if k <= 0.0:
                    continue
                self.model.dof_damping[dadr] = (
                    2.0 * float(c.ankle_zeta) * np.sqrt(k * self._ankle_inertia(dadr)))
        elif c.ankle_damping > 0.0:
            for j in ankle_j:
                self.model.dof_damping[int(self.model.jnt_dofadr[j])] = float(c.ankle_damping)

        self.ankle_k = float(self.model.jnt_stiffness[ankle_j[0]])
        self.ankle_b = float(self.model.dof_damping[int(self.model.jnt_dofadr[ankle_j[0]])])
        self.n_gait_act = self.nu - self.n_ankle_act
        self._ankle_dof = np.array([int(self.model.jnt_dofadr[j]) for j in ankle_j], dtype=int)
        self._ankle_qpos = np.array([int(self.model.jnt_qposadr[j]) for j in ankle_j], dtype=int)
        # Compression-side gain of the tension strut. It is a rigid bar, not a spring, so this is
        # only the numerical ramp that makes the saturating law continuous: full buckling load is
        # reached within BAR_SAT_RAD of the taut angle. The physics is the SATURATION, not the gain.
        self._bar_k = float(c.ankle_bar_buckle_nm) / self.BAR_SAT_RAD
        self._bar_active = (mode == "bar")
        if self._bar_active and c.ankle_kp > 0.0:
            # both write qfrc_applied on the ankle dofs; the reflex would silently overwrite the
            # strut and the arm would be measuring an actuated ankle instead of a passive one.
            raise ValueError("ankle_mode='bar' is incompatible with the ankle_kp pitch reflex — "
                             "both drive qfrc_applied on the ankle joints.")
        # torque-speed envelope: only meaningful when there IS a motor and a finite no-load speed
        self._ankle_ts_curve = bool(self.n_ankle_act and c.ankle_motor_noload_rads > 0.0)
        self._ankle_peak_w = 0.0        # peak |ankle speed| this control step (substep-resolved)

    # nominal_ctrl is rebound at reset by the m1 ride-height LUT, so these stay views rather than
    # snapshots — a stale copy would silently command the previous episode's posture.
    @property
    def _nominal6(self):
        """The gait generator's slice of the nominal control (it only ever knows 6 joints)."""
        return self.nominal_ctrl[:self.n_gait_act]

    @property
    def _nominal_ankle(self):
        """Settled stance angle of each ankle servo — the active ankle commands relative to this."""
        return self.nominal_ctrl[self.n_gait_act:]

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
        self._apply_torque_limit()

    def _apply_torque_limit(self):
        """forcerange = original x curriculum scale x this episode's domain-randomization draw.
        Both factors go through here so neither can clobber the other (the curriculum writes
        between resets, the DR draw writes at reset)."""
        self.model.actuator_forcerange[:] = (self._orig_forcerange * self._torque_scale
                                             * self._dr_torque_scale * self._sag_scale)

    def _update_torque_sag(self):
        """Bus-voltage droop: a real pack loses volts (and therefore torque) under sustained
        current and recovers when the draw stops, which a per-EPISODE torque scale cannot express.
        First-order lag on delivered mechanical power, normalised by the actuator's own peak."""
        c = self.cfg
        if c.dr_torque_sag <= 0.0:
            return
        p = float(np.sum(np.abs(self.data.actuator_force[:self.nu] * self.data.qvel[self.act_dadr])))
        p_ref = float(np.sum(self._orig_forcerange[:self.nu, 1])) * 5.0    # ~peak torque x a brisk rate
        a = self.control_dt / max(c.dr_torque_sag_tau_s, 1e-6)
        self._sag_state += a * (min(p / max(p_ref, 1e-9), 1.0) - self._sag_state)
        self._sag_scale = 1.0 - c.dr_torque_sag * self._sag_state
        self._apply_torque_limit()

    def _apply_ankle_torque_speed(self):
        """Clamp the ankle servos to a real motor's TORQUE-SPEED curve, re-evaluated every substep.

        A constant forcerange would let the idealized ankle deliver peak torque at any speed, which
        no motor does: available torque falls roughly linearly to zero at the no-load speed. Since
        the study's whole purpose is to find out whether an ankle motor is worth it AND what
        performance it would need, the envelope has to be the realistic part even when the mass is
        not — otherwise a win could just mean "an impossible actuator wins".

        Multiplies the same curriculum/DR scaling _apply_torque_limit applies, so the torque-budget
        curriculum still reaches the ankle instead of being silently overwritten here."""
        w = np.abs(self.data.qvel[self._ankle_dof])
        frac = np.clip(1.0 - w / self.cfg.ankle_motor_noload_rads, 0.0, 1.0)
        lim = (self._orig_forcerange[self.ankle_act_idx, 1]
               * self._torque_scale * self._dr_torque_scale * frac)
        self.model.actuator_forcerange[self.ankle_act_idx, 0] = -lim
        self.model.actuator_forcerange[self.ankle_act_idx, 1] = lim

    def _search_stance(self):
        """Find THIS arm's best standing posture before re-settling into it.

        `nominal_ctrl` ([0,0,0.12,0,0,-0.12]) is the stance the robot was tuned to with the shipped
        stiff, preloaded spring. A softer ankle does not stand there: measured, the k=41.4 no-preload
        arm settles with the ankle 26.6 deg past flat, far enough that the foot body grazes the floor
        and `_floor_violation` ends the episode in 3 steps -- from the RESET pose, for any policy.
        Screening that arm against a posture it cannot hold would answer a question nobody asked.

        So each arm gets the same small symmetric (cam, thigh) search, walked in order of INCREASING
        deviation from the design stance and stopping at the first pose that settles without a floor
        violation. "Keep the design posture unless this ankle cannot hold it, and then change it as
        little as possible." Identical procedure everywhere, so a stiff arm simply keeps the nominal
        stance (deviation 0 is tried first) and only a collapsing ankle is forced to crouch -- and
        HOW FAR it is forced to crouch is itself a reported result.

        Scoring on ankle deflection instead was tried and is wrong: it drags every arm into a deep
        crouch that unloads the ankle at the cost of ride height, and it is meaningless for `rigid`,
        whose welded ankle reads ~0 deflection at every pose. Returns the winning ctrl."""
        grid = (0.0, -0.05, 0.05, -0.10, 0.10)
        cands = sorted(((dc, dt) for dc in grid for dt in grid), key=lambda p: (abs(p[0]) + abs(p[1])))
        for dc, dt in cands:
            ctrl = self.nominal_ctrl.copy()
            ctrl[1] += dc; ctrl[2] += dt                  # cam_L, thigh_L
            ctrl[4] -= dc; ctrl[5] -= dt                  # mirrored R (flipped sagittal axes)
            qpos, viol = self._settle(ctrl, t_s=1.0)
            if qpos is not None and not viol:
                self.stance_search_delta = (float(dc), float(dt))
                return ctrl
        raise RuntimeError(
            f"ankle arm {self.ankle_mode!r} (k={self.ankle_k}) has NO standing posture in the search "
            "grid that settles without the foot going through the floor — it cannot stand at all.")

    def _settle(self, ctrl, t_s=2.0):
        """Gravity-settle from the keyframe with base x/y/roll/pitch/yaw held and Z free, motors
        holding `ctrl`. Runs on self.data (construction-time only; reset() re-initialises it) so the
        floor check sees the same contact state the episode will.

        Returns (qpos, floor_violated), or (None, True) on divergence. floor_violated is watched over
        the SETTLED TAIL, not just the final pose: the violation flickers on and off as the foot
        grazes the ground, so a single end-of-settle sample reads clean on a pose that terminates
        2 steps into an episode -- measured, on exactly the k=41.4 no-preload arm this matters for."""
        held = np.array([0, 1, 3, 4, 5])
        d = self.data
        mujoco.mj_resetDataKeyframe(self.model, d, self.key_id)
        d.ctrl[:] = ctrl
        if self.ankle_mode == "rigid":
            d.eq_active[self._ankle_lock_eq] = 1
        base_q = d.qpos[held].copy()
        n = int(t_s / self.sim_dt)
        tail = int(0.7 * n)
        viol = False
        for i in range(n):
            if self._bar_active:
                e = self._bar_sign * (d.qpos[self._ankle_qpos] - self._bar_q0)
                d.qfrc_applied[self._ankle_dof] = self._bar_sign * np.clip(
                    np.clip(-e, 0.0, None) * self._bar_k, 0.0, self.cfg.ankle_bar_buckle_nm)
            mujoco.mj_step(self.model, d)
            d.qpos[held] = base_q
            d.qvel[held] = 0.0
            if i >= tail and not viol:
                viol = bool(self._floor_violation())
        d.qfrc_applied[:] = 0.0
        d.eq_active[self._ankle_lock_eq] = 0
        if not np.all(np.isfinite(d.qpos)):
            return None, True
        return d.qpos.copy(), viol

    def _resettle_keyframe(self, t_s=2.0):
        """Re-settle the `stand` keyframe against THIS arm's ankle law.

        The shipped keyframe is a gravity-settled equilibrium of the k=28.65 preloaded spring. Any
        arm that changes the ankle law starts off that equilibrium and lurches for the first few
        control steps of every episode — which would show up as a handicap on exactly the soft arms
        the study is about, and would also bias `_stand_torque` (the torque penalty's baseline) and
        `height_target`. So each arm gets its own settled stance.

        Settled the same way the record's loaded-stance numbers were measured: base x/y/roll/pitch/
        yaw held, Z free, motors holding the nominal stance targets. Holding the 5 base DOFs is what
        makes it a plant measurement rather than a balance test — a floppy ankle would simply topple
        with the base free, and toppling is the RL question, not the keyframe question."""
        z_before = float(self.model.key_qpos[self.key_id][2])
        # posture first, then settle into it (the search itself settles each candidate)
        self.nominal_ctrl[:] = self._search_stance()
        self.model.key_ctrl[self.key_id] = self.nominal_ctrl
        qpos, _ = self._settle(self.nominal_ctrl, t_s=t_s)
        if qpos is None:
            raise RuntimeError(f"ankle arm {self.ankle_mode!r} diverged while re-settling the stance")
        self.model.key_qpos[self.key_id] = qpos
        self.default_qpos = qpos.copy()
        self.default_motor_pos = self.default_qpos[self.act_qadr]
        self.height_target = float(self.default_qpos[2])
        # how far this ankle sags relative to the shipped k=28.65 preloaded stance. Reported by the
        # statics tool: a large sag IS the answer for a soft arm, not a nuisance to be normalized.
        self.settle_sag_m = z_before - self.height_target
        self.settle_ankle = self.default_qpos[self._ankle_qpos].copy()
        # Re-reference the workspace box to THIS arm's settled stance.
        #
        # workspace_kill measures the toe in the BASE frame against the LUT's nominal_toe, so it
        # cannot tell a collapsed ANKLE from a folded 4-BAR — a sagging base lifts the toe relative
        # to the base exactly as a parked leg does. That conflation is harmless in the wskill
        # lineage, where every run has the same ankle. Across an ankle STUDY it is fatal: measured,
        # the no-preload arm stands at dz=+0.180 m against a +0.14 ceiling, so the kill would fire
        # 0.1 s into every episode from the reset pose and the arm would score zero for a reason
        # that has nothing to do with whether it can be controlled.
        # Re-centering on each arm's own stance (same half-widths, applied identically to every arm,
        # including the k350 control) restores the question the box was built to ask — how far has
        # this foot travelled from where this robot stands. The 4-bar's real reachability is still
        # enforced independently by the loop-closure equality and the joint ranges.
        if self._ws_ref is not None:
            d = self.data
            mujoco.mj_resetDataKeyframe(self.model, d, self.key_id)
            mujoco.mj_forward(self.model, d)
            base, R = d.xpos[self.base_id], d.xmat[self.base_id].reshape(3, 3)
            self._ws_ref = np.array([R.T @ (d.geom_xpos[g] - base) for g in self.foot_gids])

    def _apply_ankle_bar(self):
        """Tension-only strut, re-evaluated every substep (ankle_mode='bar').

        The traction side is a joint LIMIT (set in _setup_ankle) — an inextensible bar cannot let
        the ankle travel past the flat-foot angle under load, and MuJoCo's limit constraint is a
        better model of that than any stiff spring we could write here.

        This handles the other side. A real strut is not a one-way constraint: it pushes back until
        it BUCKLES, and past that its capacity is gone (Euler collapse), so the honest law is a
        SATURATION at ankle_bar_buckle_nm rather than a hard stop or nothing at all. That matters
        more than it looks — the saturated torque (0.2-0.6 N*m over the plausible lever arms) is
        larger than the foot's own gravity torque (~0.13 N*m), so the strut carries the unloaded
        foot near flat through swing instead of letting it flop to the joint stop and slam on
        touchdown. It only gives way when something pushes harder than the buckling load."""
        q = self.data.qpos[self._ankle_qpos]
        e = self._bar_sign * (q - self._bar_q0)          # >0 traction (the limit handles it), <0 compression
        tau = np.clip(-e, 0.0, None) * self._bar_k
        np.clip(tau, 0.0, self.cfg.ankle_bar_buckle_nm, out=tau)
        self.data.qfrc_applied[self._ankle_dof] = self._bar_sign * tau

    def set_dr_scale(self, s):
        """0..1 curriculum on the WIDTH of every domain-randomization range (applies from the next
        reset). Measured on teleop_v3: on the nominal plant the policy survives 196 s and never
        falls; at full-width DR it survives 4.8 s — randomization was ~20x more destructive than
        pushes, trips and sensor noise combined, and unlike all of those it had no curriculum.
        A policy cannot learn to be robust to a plant it cannot stand up on."""
        self._dr.scale = float(np.clip(s, 0.0, 1.0))

    def set_cmd_scale(self, s):
        """0..1 command-RANGE curriculum: interpolates the sampled command box from
        (cmd_v_fwd_start, cmd_v_back_start, cmd_yaw_start) at 0 to the full
        (cmd_v_fwd_max, cmd_v_back_max, cmd_yaw_max) at 1.

        The policy observes the command in PHYSICAL UNITS scaled by a FIXED constant (cmd_v_norm /
        cmd_yaw_norm), never by this curriculum value — if the normalizer moved with the curriculum
        then 'obs = 1.0' would mean 0.5 m/s early and 1.8 m/s late, the same input would mean
        different things at different times, and nothing learned early would still be true. Only
        the SAMPLING DISTRIBUTION widens here; the command's meaning never changes."""
        self._cmd_scale = float(np.clip(s, 0.0, 1.0))

    def _cmd_box(self):
        """(v_fwd_max, v_back_max, yaw_max) at the current curriculum scale."""
        c, s = self.cfg, self._cmd_scale
        return (c.cmd_v_fwd_start + s * (c.cmd_v_fwd_max - c.cmd_v_fwd_start),
                c.cmd_v_back_start + s * (c.cmd_v_back_max - c.cmd_v_back_start),
                c.cmd_yaw_start + s * (c.cmd_yaw_max - c.cmd_yaw_start))

    def set_fixed_base(self, clearance=0.25):
        """Clamp ALL six base DOFs and hang the robot `clearance` metres higher than its stance
        height — the test-rig configuration: bolted to a stand, legs cycling in the air.

        This is deliberately a runtime override rather than a preset, because it is not a training
        condition: it is how the FIRST hardware bring-up will be run, and the point is to preview
        in sim exactly what that rig will show before committing to it. Note what it does to the
        policy's inputs — with the base clamped upright, gravity really is constant and the gyro
        really is zero, so those two observations are CORRECT rather than out-of-distribution.
        What is missing is ground contact, so nothing the legs do feeds back. Expect the nominal
        gait for the commanded speed and no closed-loop balance behaviour; that is the honest
        limit of what an in-air test can tell you. Applies from the next reset."""
        self.base_lock[:] = 1
        self.z_locked = True
        self._lut = None                      # the m1 ride-height LUT seats feet ON the floor
        self._fixed_base_h = float(self.height_target) + float(clearance)

    def set_command(self, v_cmd, yaw_cmd=0.0):
        """Drive the robot directly (teleop / evaluation). Suspends the automatic resampling —
        once something outside is holding the stick, nothing inside should be moving it."""
        self._v_cmd = float(v_cmd)
        self._yaw_cmd = float(yaw_cmd)
        self._cmd_countdown = 10 ** 9
        was = self._standing
        self._standing = (abs(self._v_cmd) < 1e-6 and abs(self._yaw_cmd) < 1e-6)
        if self._standing and not was:
            self._stand_anchor[:] = self.data.qpos[0:2]
        self._update_task()

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
        """One RAW measurement frame, then corrupted by the sensor model. Everything in here is a
        quantity the hardware can actually produce (encoder, motor current, IMU) except the gait
        phase, which is computed onboard, and the optional privileged base velocity."""
        s = self.cfg.obs_scales
        motor_pos = self.data.qpos[self.act_qadr] - self.default_motor_pos
        motor_vel = self.data.qvel[self.act_dadr].copy()
        motor_trq = self.data.actuator_force[:self.nu].copy()
        grav = self._gravity_body()
        angv = self._ang_vel_body()
        # body-frame linear acceleration, for the accelerometer-leak term of the gravity model
        v_now = self._vel_body()
        accel_body = (v_now - self._prev_vel_body) / self.control_dt
        self._prev_vel_body = v_now
        if self._noise.enabled:
            self._noise.step_bias(self.np_random)
            motor_pos, motor_vel, motor_trq, grav, angv = self._noise.apply(
                self.np_random, motor_pos, motor_vel, motor_trq, grav, angv, accel_body)
        parts = [motor_pos * s["motor_pos"], motor_vel * s["motor_vel"],
                 motor_trq * s["motor_torque"], grav * s["gravity"], angv * s["ang_vel"]]
        if self.obs_base_vel:                       # privileged; off in command mode
            parts.append(v_now * s["base_vel"])
        if self.cpg_mode:
            r, _, th = self._cpg
            phase_ch = np.array([np.sin(th[0]), np.cos(th[0]),
                                 np.sin(th[1]), np.cos(th[1]), r[0], r[1]])
        else:
            phase_ch = np.array([np.sin(self._phase), np.cos(self._phase)])
        parts += [phase_ch, self._task, self._prev_action]
        return np.concatenate(parts).astype(np.float32)

    def _obs(self):
        return self._history[self._hist_idx].reshape(-1).astype(np.float32)

    def _push_frame(self, frame):
        """Append a measurement to the history, optionally after a fixed sensor delay (staleness
        of the CAN read, on top of the action delay that models inference + actuation)."""
        if self.cfg.obs_delay_steps > 0:
            self._obs_delay_buf.append(frame)
            frame = self._obs_delay_buf.pop(0)
        self._history[:-1] = self._history[1:]
        self._history[-1] = frame

    def _update_task(self):
        """Refresh the task channel.

        sprint : [run_flag, dist_to_go/100]. The run->stop flip at the line is the policy's stop
                 signal; dist_to_go lets it SEE the line coming (plan braking) and gives the value
                 function the state its return actually depends on.
        command: [v_cmd/cmd_v_norm, yaw_cmd/cmd_yaw_norm, stand_flag]. FIXED normalizers — see
                 set_cmd_scale for why they must never track the curriculum. The explicit
                 stand_flag makes 'hold position' a distinct mode rather than something the policy
                 has to infer from two near-zero floats.
        """
        c = self.cfg
        if self.command_mode:
            self._task[0] = self._v_cmd / c.cmd_v_norm
            self._task[1] = self._yaw_cmd / c.cmd_yaw_norm
            self._task[2] = 1.0 if self._standing else 0.0
        elif c.objective == "sprint":
            if self._sprint_crossed:
                self._task[:] = 0.0
            else:
                self._task[0] = 1.0
                self._task[1] = np.clip((self._sprint_D - self._sprint_d) / 100.0, 0.0, 1.0)
        else:                       # speed: run forever
            self._task[:] = 1.0

    # ---------- joystick command ----------
    def _sample_command(self):
        """Draw a new command from the current curriculum box. A fixed fraction of draws are
        EXACTLY zero (stand still) rather than merely small: standing is a mode the demo needs to
        do well and cleanly, and it will not be learned from the tail of a uniform distribution.
        Small non-zero draws are snapped to zero by the deadband for the same reason — a real
        joystick has one too, and a command the robot cannot resolve is a command it should not
        be graded on."""
        rng, c = self.np_random, self.cfg
        v_fwd, v_back, yaw = self._cmd_box()
        if rng.random() < c.cmd_zero_prob:
            self._v_cmd, self._yaw_cmd = 0.0, 0.0
        else:
            self._v_cmd = float(rng.uniform(-v_back, v_fwd))
            self._yaw_cmd = float(rng.uniform(-yaw, yaw))
            if abs(self._v_cmd) < c.cmd_deadband:
                self._v_cmd = 0.0
            if abs(self._yaw_cmd) < c.cmd_yaw_deadband:
                self._yaw_cmd = 0.0
        was = self._standing
        self._standing = (self._v_cmd == 0.0 and self._yaw_cmd == 0.0)
        if self._standing and not was:
            self._stand_anchor[:] = self.data.qpos[0:2]
        # resample on a randomized interval: the policy must handle the stick MOVING, which is the
        # whole point of teleop, and a fixed interval is something it can learn to anticipate.
        s = c.cmd_resample_s * rng.uniform(0.7, 1.3)
        self._cmd_countdown = max(1, int(round(s / self.control_dt)))

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # per-episode plant + measurement draw, BEFORE mj_forward so the new masses/inertias are
        # in this episode's mass matrix and the standing-torque baseline below reflects them.
        ep = self._dr.resample(self.model, self.np_random)
        self._dr_torque_scale = ep["torque_scale"]
        self._sag_scale, self._sag_state = 1.0, 0.0
        self._apply_torque_limit()
        self._ep_delay_steps = int(ep["action_delay_steps"])
        self._noise.reset(self.np_random)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        # activate this milestone's base-DOF locks (loop-closure equalities stay untouched)
        self.data.eq_active[self.lock_eq_ids] = self.base_lock
        # ankle_mode="rigid": weld both ankles at their stance angle for the whole episode
        if self.ankle_mode == "rigid":
            self.data.eq_active[self._ankle_lock_eq] = 1
        # m1: rail Z at a per-episode RANDOM ride height, legs seated from the LUT so the episode
        # starts in a valid on-floor stance; otherwise a locked Z pins at the natural stance height.
        if self.z_locked:
            if self._lut is not None:
                H = float(self.np_random.uniform(*self.cfg.z_rail_range))
                k = int(np.argmin(np.abs(self._lut["H"] - H)))
                self.data.qpos[self.hinge_qadr_start:] = self._lut["hinges"][k]
                lut_ctrl = self._lut["ctrl"][k].astype(np.float64)
                # the ride-height LUT was generated on the 6-actuator plant; on the actuated-ankle
                # plant keep the ankle servos at their stance angle rather than truncating nu.
                if lut_ctrl.size < self.nu:
                    lut_ctrl = np.concatenate([lut_ctrl, self.model.key_ctrl[self.key_id][lut_ctrl.size:]])
                self.nominal_ctrl = lut_ctrl.copy()
            else:
                H = float(self.height_target)
            if self._fixed_base_h is not None:      # in-air test rig (set_fixed_base)
                H = self._fixed_base_h
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
        self._reflex_prate_filt = 0.0
        self._coef_rate_gated = 0.0
        self._phase = 0.0
        self._phase_reward = 0.0
        self._phase_reward_R = np.pi
        # start the oscillators at rest, already in antiphase: r = 0 is a standing robot, so the
        # gait has to be started by the policy raising mu rather than being handed a running start
        self._cpg = (np.zeros(2), np.zeros(2), np.array([0.0, np.pi]))
        self._elapsed_t = 0.0
        self._filt_target[:] = self.nominal_ctrl
        self._prev_cmd_pos[:] = self.nominal_ctrl
        self._prev_cmd_vel[:] = 0.0
        self._delay_buf = [np.zeros(self.action_dim, np.float32)
                           for _ in range(self._ep_delay_steps)]
        self._obs_delay_buf = [np.zeros(self.frame_dim, np.float32)
                               for _ in range(self.cfg.obs_delay_steps)]
        self._air_time[:] = 0.0
        self._contact_time[:] = 0.0
        self._grounded_prev = self._foot_contacts() | (self._toe_heights() < self.cfg.grounded_h)
        self._prev_toe_xy = self.data.geom_xpos[self.foot_gids_arr, 0:2].copy()
        self._duty_ema[:] = 0.5          # neutral start: above duty_floor, so no penalty until the
        #                                  gait actually parks a foot in the air (EMA then decays)
        self._ws_out_t[:] = 0.0          # reset the per-foot outside-workspace timers
        self._push_countdown = self._next_push_in()
        self._step_n = 0
        self._prev_vel_body[:] = 0.0
        self._track_err_sum, self._track_err_n = 0.0, 0
        # sprint state: the finish line is frozen per episode (curriculum moves it between dashes)
        self._x0 = float(self.data.qpos[0])
        self._sprint_D = float(self.cfg.sprint_dist_m)
        self._sprint_crossed = False
        self._sprint_t_line = None
        self._sprint_d = 0.0
        self._stop_hold = 0.0
        # joystick command: draw the first one (also latches the stand anchor)
        if self.command_mode:
            self._standing = False
            self._sample_command()
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

    def _command_income(self, t, vx, v_body, angv, grav):
        """Joystick objective: track the commanded forward speed and yaw rate, or hold position.
        Fills the income terms in `t` and returns (cmd_speed, progress_frac) for the gait shaping.

        Tracking uses a Gaussian kernel with a RELATIVE tolerance. With a fixed sigma, holding
        1.8 m/s to +-0.15 is far harder than holding 0.5 m/s to +-0.15, so a fixed-sigma reward is
        quietly a bribe to stay slow — the policy maximizes it by living at the bottom of the
        command range. sigma = max(sigma_min, sigma_rel*|cmd|) grades every speed on equal terms.
        """
        c = self.cfg
        yaw_rate = float(angv[2])
        # --- linear speed tracking ---
        sig = max(c.track_sigma_min, c.track_sigma_rel * abs(self._v_cmd))
        e_lin = (float(vx) - self._v_cmd) / sig
        lin = c.w_track_lin * float(np.exp(-e_lin * e_lin))
        # uprightness gate, same argument as the sprint speed gate: a toppling robot must not be
        # able to bank tracking reward on the way down (it can hit any velocity while falling).
        if c.speed_upright_gate:
            u = np.clip((-grav[2] - c.speed_upright_c0) / (1.0 - c.speed_upright_c0), 0.0, 1.0)
            lin *= float(u) ** c.speed_upright_k
        t["track_lin"] = lin
        # --- yaw rate tracking (gyro z: measurable on hardware, unlike a curvature radius) ---
        sigw = max(c.track_yaw_sigma_min, c.track_sigma_rel * abs(self._yaw_cmd))
        e_yaw = (yaw_rate - self._yaw_cmd) / sigw
        t["track_yaw"] = c.w_track_yaw * float(np.exp(-e_yaw * e_yaw))
        # --- stand still ---
        # Stepping in place is explicitly allowed (the plant cannot stand passively — it needs an
        # active gait for height), so this grades the BASE, not the feet: near-zero body velocity
        # plus a penalty on drifting away from where the stand command was given. Without the
        # anchor term a slow constant creep costs almost nothing per step and the robot walks off.
        if self._standing:
            vmag = float(np.linalg.norm(v_body[0:2]))
            t["stand"] = c.w_stand * float(np.exp(-((vmag / c.stand_sigma) ** 2)))
            drift = float(np.linalg.norm(self.data.qpos[0:2] - self._stand_anchor))
            t["stand_drift"] = self._pen(-c.w_stand_drift * max(0.0, drift - c.stand_drift_free_m) ** 2)
        else:
            t["stand"] = 0.0
            t["stand_drift"] = 0.0
        t["fwd_speed"] = 0.0
        t["stop"] = 0.0
        t["overrun"] = 0.0
        # tracking error, for the command-range curriculum callback (top-of-range competence)
        self._track_err_sum += abs(float(vx) - self._v_cmd)
        self._track_err_n += 1
        cmd_speed = abs(self._v_cmd)
        progress_frac = float(np.clip(cmd_speed / max(c.cmd_v_fwd_max, 1e-6), 0.0, 1.0))
        return cmd_speed, progress_frac

    def _run_physics(self, target):
        """One control step of plant: EMA-filter the target, clip to ctrlrange, run
        control_decimation sim substeps (OR-accumulating foot contact so a sub-20 ms hop can't
        pass as continuous flight/contact at the 50 Hz boundary)."""
        c = self.cfg
        self._filt_target = c.action_filter * self._filt_target + (1 - c.action_filter) * target
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
        # HOMING error, command side. The encoder reads theta - delta (applied in SensorNoise), so
        # the drive closes its loop on that and parks the TRUE joint at target + delta. Applying it
        # to only one side would model a robot that does not exist.
        if self.cfg.dr_joint_zero_deg > 0.0:
            tgt = tgt + self._noise.zero_offset[:len(tgt)]
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
            if self._ankle_ts_curve:
                self._apply_ankle_torque_speed()
            if self._bar_active:
                self._apply_ankle_bar()
            mujoco.mj_step(self.model, self.data)
            if not contact_acc.all():
                contact_acc |= self._foot_contacts()
            self._ankle_peak_w = max(self._ankle_peak_w,
                                     float(np.max(np.abs(self.data.qvel[self._ankle_dof]))))
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
        if self.cpg_mode:
            mu_raw, freq_raw, psi_raw, reflex, steer, residual = cpg_gait.decode(
                applied, self.n_steer, c.cpg_residual)
            cam_c = thigh_c = None
            cpg_mu = cpg_gait.amplitude_setpoint(mu_raw, c)
            f = cpg_gait.frequency(freq_raw, c.gait_freq_hz)     # per-leg, 2-vector
            cpg_psi = c.cpg_psi_range * float(np.clip(psi_raw, -1.0, 1.0))
            # the left oscillator plays the role the global clock plays in fourier mode: it gates
            # the spec-change penalty and anchors the phase the reward's contact schedule reads
            phase_used = float(self._cpg[2][0])
            self._phase_reward = phase_used
            self._phase_reward_R = float(self._cpg[2][1])
        else:
            cam_c, thigh_c, freq_raw, reflex, steer, residual = fourier_gait.decode(
                applied, c.n_harmonics, self.n_steer)
            f = fourier_gait.frequency(freq_raw, c.gait_freq_hz)
            phase_used = self._phase
            self._phase_reward = phase_used
            self._phase_reward_R = phase_used + np.pi
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
        if c.pitch_reflex_rate_lp > 0.0:  # low-pass the rate the reflex sees: keep the slow real-
            # tilt response, drop the fast gait-bob the D-term was rectifying into ~6 Hz chatter
            self._reflex_prate_filt = (c.pitch_reflex_rate_lp * self._reflex_prate_filt
                                       + (1.0 - c.pitch_reflex_rate_lp) * pitch_rate)
            pitch_rate = self._reflex_prate_filt
        if self.cpg_mode:
            target6 = cpg_gait.assemble(self._cpg, reflex, roll, roll_rate, self._nominal6, c,
                                        stance_ratio=self._stance_ratio,
                                        pitch=pitch, pitch_rate=pitch_rate, steer=steer,
                                        lut=self._cpg_lut)
        else:
            target6 = fourier_gait.assemble(cam_c, thigh_c, reflex, phase_used,
                                            roll, roll_rate, self._nominal6, c,
                                            pitch=pitch, pitch_rate=pitch_rate, steer=steer)
        target6 = target6 + c.residual_scale * residual  # the per-step fast-feedback channel
        if self.n_ankle_act:
            # ACTIVE ANKLE: the tail dims are a position command about the settled stance angle.
            # Deliberately NOT routed through the gait generator — the ankle gets no clock, no
            # Fourier series and no phase, only per-step feedback authority, so "active" tests an
            # ankle STRATEGY the policy has to learn rather than a second scripted waveform. This
            # is also the channel the 2026-07-23 fixed PD reflex could not provide: that one was
            # phase-blind by construction and failed for exactly that reason.
            ankle_cmd = (self._nominal_ankle
                         + c.ankle_action_scale * applied[self.gait_action_dim:])
            target = np.concatenate([target6, ankle_cmd])
        else:
            target = target6
        motor_cmd = ((target - self.nominal_ctrl) / c.action_scale).astype(np.float32)
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

        # TRIP: a swinging toe catches something that isn't in the map. Modelled as a brief force
        # opposing the swing rather than as terrain geometry, because the point is not to teach
        # the policy one particular obstacle — it is to make "my foot stopped moving and my torso
        # is rotating over it" a state the policy has recovered from thousands of times. That is
        # the RL-native version of a hand-written raise-the-foot reflex, and unlike a detector it
        # cannot fail to fire.
        self.data.xfrc_applied[:] = 0.0
        if self._trip_left > 0:
            self.data.xfrc_applied[self._trip_body, 0] = self._trip_force
            self._trip_left -= 1
        elif c.trip_prob > 0.0 and self.np_random.random() < c.trip_prob:
            air = ~(self._foot_contacts() | (self._toe_heights() < c.grounded_h))
            cand = np.flatnonzero(air)
            if cand.size:
                i = int(self.np_random.choice(cand))
                self._trip_body = self._foot_bids[i]
                # opposes travel, so a forward-running robot gets caught forward-on (the case that
                # actually matters); sign taken from base velocity, +x when standing still
                vx_now = float(self._vel_body()[0])
                self._trip_force = -(1.0 if vx_now >= 0.0 else -1.0) * float(
                    self.np_random.uniform(*c.trip_force_range))
                self._trip_left = max(1, int(round(c.trip_duration_s / self.control_dt)))

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

        contact_acc = self._run_physics(target)
        self._update_torque_sag()
        self._elapsed_t += self.control_dt
        finished = c.objective == "sprint" and self._update_sprint()
        # advance the gait phase AFTER assembly (the obs frame carries the NEXT step's phase)
        if self.cpg_mode:
            self._cpg = cpg_gait.integrate(self._cpg, cpg_mu, f, cpg_psi, self.control_dt, c)
        else:
            self._phase = (self._phase + 2.0 * np.pi * f * self.control_dt) % (2.0 * np.pi)

        self._step_n += 1
        reward, terms = self._reward(motor_cmd, contact_acc)
        # joystick: age the command and redraw when it expires — AFTER the reward, so this step is
        # always graded against the command that actually produced it, and BEFORE the obs frame, so
        # the policy sees the new command on the same step the grading switches to it.
        if self.command_mode:
            self._cmd_countdown -= 1
            if self._cmd_countdown <= 0:
                self._sample_command()
        self._update_task()
        self._push_frame(self._proprio())
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
        if self.command_mode:
            # the command-range curriculum grades competence at the TOP of the current box, so it
            # needs both the error and the command that produced it, not a rollout-wide average
            info["cmd_v"] = self._v_cmd
            info["cmd_yaw"] = self._yaw_cmd
            info["track_err"] = abs(float(self._vel_body()[0]) - self._v_cmd)
            info["track_yaw_err"] = abs(float(self._ang_vel_body()[2]) - self._yaw_cmd)
            info["cmd_scale"] = self._cmd_scale
        # mean actuator torque utilization |tau|/limit (for the torque-budget curriculum callback)
        _lim = self.model.actuator_forcerange[:self.nu, 1]
        info["torque_util"] = float(np.mean(
            np.abs(self.data.actuator_force[:self.nu]) / np.maximum(_lim, 1e-6)))
        if c.objective == "sprint":
            info["sprint"] = self._sprint_info(finished)
        info.update(self._ankle_info())
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def _ankle_info(self):
        """Per-step ankle telemetry — the numbers that decide whether a winning arm is BUILDABLE.

        A stiffness that wins in sim is only useful if a real spring can survive it, and an active
        ankle that wins is only useful if a real motor can deliver it. So log the peak demands, not
        just the score: spring torque and stored energy (does the part exist?), motor torque and
        mechanical power (does the motor exist?). |q - springref| also catches an arm that is
        silently living on the joint's +-1.047 travel limit rather than on its spring."""
        q = self.data.qpos[self._ankle_qpos]
        qd = self.data.qvel[self._ankle_dof]
        defl = q - self.model.qpos_spring[self._ankle_qpos]
        out = {"ankle_defl": float(np.max(np.abs(defl))),
               "ankle_spring_trq": float(np.max(np.abs(self.ankle_k * defl))),
               # 1/2 k x^2 per side, summed: the elastic energy the structure has to store
               "ankle_spring_energy": float(np.sum(0.5 * self.ankle_k * defl ** 2))}
        if self.n_ankle_act:
            # THE SPEC READOUT. If the active arm wins, these four numbers are the answer to "what
            # performance do we need from an ankle motor" — which is half the point of the study,
            # so they are logged per step rather than reconstructed from video afterwards.
            tau = self.data.actuator_force[self.ankle_act_idx]
            out["ankle_motor_trq"] = float(np.max(np.abs(tau)))          # -> peak torque needed
            out["ankle_motor_w"] = self._ankle_peak_w                    # -> no-load speed needed
            out["ankle_motor_power"] = float(np.sum(np.abs(tau * qd)))   # -> peak power needed
            # thermal: a motor may hit peak torque briefly but must live below continuous. This is
            # the fraction of ankles over the continuous rating right now; averaged over a run it
            # says whether the duty cycle is survivable or whether the motor cooks.
            out["ankle_motor_over_cont"] = float(np.mean(
                np.abs(tau) > self.cfg.ankle_motor_cont_nm))
            # utilization against the CURRENT (speed-derated) limit: ~1.0 means the torque-speed
            # curve, not the policy, is what is capping the ankle
            out["ankle_motor_util"] = float(np.max(
                np.abs(tau) / np.maximum(self.model.actuator_forcerange[self.ankle_act_idx, 1], 1e-6)))
            self._ankle_peak_w = 0.0
        return out

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
        if self.command_mode:
            cmd_speed, progress_frac = self._command_income(t, vx, v_body, angv, grav)
        elif run_phase:
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
        # In command mode the gait terms key off the COMMANDED speed, not a constant: with the old
        # `cmd_speed = v_ceiling` a walk command would still be graded under running rules (stance
        # caps, flight-phase demand), which is exactly backwards.
        if not self.command_mode:
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
            sw_R = 1.0 - fourier_gait.stance_indicator(self._phase_reward_R, sr)
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

        # duty-symmetry / anti-one-legged: EMA each foot's grounded fraction and penalize (linearly)
        # any foot whose duty sinks below duty_floor, so a foot that never bears load is expensive ->
        # forces both legs to share stance instead of one-legged pattering (the slow_gait failure).
        if c.w_duty_sym > 0.0:
            a = min(1.0, dt / max(c.duty_sym_tau_s, 1e-3))
            self._duty_ema += a * (grounded.astype(np.float64) - self._duty_ema)
            deficit = np.maximum(0.0, c.duty_floor - self._duty_ema)
            t["duty_sym"] = self._pen(-c.w_duty_sym * float(deficit.sum()))
        else:
            t["duty_sym"] = 0.0

        self._grounded_prev = grounded
        self._prev_toe_xy = toe_pos[:, 0:2]

        # ----- efficiency (Cassie-100m recipe; ramped in by the curriculum callback) -----
        # The ankle servos are billed here like every other actuator (nu is 8 on the active plant),
        # so an active arm cannot buy stability with free energy and win the study for the wrong
        # reason. ankle_torque_billed=False exempts them, which exists only as a sensitivity check:
        # "does active still lose once its energy is free?"
        n_eff = self.nu if (c.ankle_torque_billed or not self.n_ankle_act) else self.n_gait_act
        tau = self.data.actuator_force[:n_eff]
        qd = self.data.qvel[self.act_dadr[:n_eff]]
        exc = np.maximum(np.abs(tau) - np.abs(self._stand_torque[:n_eff]), 0.0)
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
        if self._workspace_violation():
            return True
        return False

    def _workspace_violation(self):
        """Terminate when a foot's toe leaves the MEASURED real-robot workspace, sustained for
        workspace_grace_s -- kills the one-legged gait's parked/folded leg (a sim-only exploit the
        physical 4-bar cannot do). Toe (dx fore-aft, dz lift) in the BASE frame relative to the LUT
        nominal_toe, exactly the frame build_cpg_lut measured the reachable box in. Per-foot grace
        timer: a foot parked outside fires; a transient swing overshoot resets and does not."""
        c = self.cfg
        if not c.workspace_kill or self._ws_ref is None:
            return False
        base = self.data.xpos[self.base_id]
        R = self.data.xmat[self.base_id].reshape(3, 3)
        fired = False
        # _ws_ref is the LUT's single nominal_toe by default, or -- when the ankle law has been
        # changed and the stance re-settled -- this arm's own per-foot settled toe (see
        # _resettle_keyframe for why that re-referencing is necessary and what it costs).
        ref = self._ws_ref if self._ws_ref.ndim == 2 else np.broadcast_to(self._ws_ref, (2, 3))
        for fi, g in enumerate(self.foot_gids):
            tb = R.T @ (self.data.geom_xpos[g] - base)
            dx = tb[0] - ref[fi][0]
            dz = tb[2] - ref[fi][2]
            if abs(dx) > c.workspace_dx_max or dz > c.workspace_dz_max or dz < c.workspace_dz_min:
                self._ws_out_t[fi] += self.control_dt
                if self._ws_out_t[fi] >= c.workspace_grace_s:
                    fired = True
            else:
                self._ws_out_t[fi] = 0.0
        return fired
