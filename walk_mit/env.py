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
import gait_v2
from raibert import RaibertPrior
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
    # width of the privileged critic tail (see obs_privileged_critic in __init__). A class
    # constant because train.py and asym_policy.py must agree with the env on where the actor's
    # slice of the observation ends, and three hardcoded 6s is how that stops being true.
    PRIV_DIM = 6
    # v2 (action_mode="latched") privileged tail, artifact §03: true base velocity 3, contacts 2,
    # height above the kill line 1, lateral offset y 1, heading 1, true base acceleration 3,
    # normal contact force L/R 2, DR draw 5 (mass, CoM, friction, kp, torque), drawn delay 1,
    # winding dT/dT_max x6 = 25. Instance attribute `priv_dim` is what train.py must read.
    PRIV_DIM_V2 = 25

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
            # start at the EASY end when a curriculum is configured; the callback tightens it
            hz0 = (self.cfg.drive_bandwidth_start_hz
                   if (self.cfg.drive_bandwidth_start_hz > 0.0
                       and self.cfg.drive_curriculum_steps > 0)
                   else self.cfg.drive_bandwidth_hz)
            self.set_drive_bandwidth_log10(np.log10(hz0))
        if self.cfg.drive_delay_ms > 0.0 and not self.cfg.drive_delay_substep:
            self.cfg.action_delay_steps = int(round(
                float(self.cfg.drive_delay_ms) * 1e-3 / self.control_dt))
        if self.cfg.drive_delay_substep:
            # v2: the transport delay rides the PD targets + gains at 1 kHz substep granularity
            # (see _run_physics), drawn per episode in ms; the whole-action delay-in-steps is off
            self.cfg.action_delay_steps = 0
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
        # v2 drive: joint armature per motor family (hip_roll, cam/thigh) from the Bode fit
        # (model/fit_drive.py). Written before the DR snapshot so the randomizer's nominal has it.
        if self.cfg.drive_armature:
            a_hip, a_ct = (float(x) for x in self.cfg.drive_armature)
            for a in range(self.nu):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
                if name.startswith("ankle"):
                    continue
                self.model.dof_armature[self.act_dadr[a]] = a_hip if name.startswith("hip_roll") else a_ct
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
        # The ankle (foot) joints are the hinges of the Foot bodies, identified by BODY NAME rather
        # than "has a spring": the v2 plant carries a stiff series spring on each pushrod slide
        # joint (model/make_v2_plant.py), which a jnt_stiffness > 0 test would take for an ankle.
        _ankle_j = sorted((j for j in range(self.model.njnt)
                           if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                 int(self.model.jnt_bodyid[j])) or "").startswith("Foot")
                           and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE),
                          key=lambda j: self.model.jnt_qposadr[j])
        if not _ankle_j:                          # legacy fallback: the sprung joints
            _ankle_j = sorted((j for j in range(self.model.njnt) if self.model.jnt_stiffness[j] > 0),
                              key=lambda j: self.model.jnt_qposadr[j])
        self._ankle_dadr = [int(self.model.jnt_dofadr[j]) for j in _ankle_j]
        # reset-noise targets: the non-base HINGE joints only (a 30 kN/m slide joint must not be
        # handed 0.03 m of reset noise). Same count as the legacy qpos[hinge_qadr_start:] slice on
        # every legacy plant, so the RNG stream and the values are unchanged there.
        self._noise_qadr = np.array([int(self.model.jnt_qposadr[j]) for j in range(self.model.njnt)
                                     if self.model.jnt_bodyid[j] != self.base_id
                                     and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE], dtype=int)
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
        # geom_size[0] is the radius ONLY for a sphere. The foot-shape study (model/
        # make_foot_variants.py) also runs a 30x100x10 mm plate and a 100 mm lateral cylinder, where
        # size[0] is the fore-aft half-length (15 mm) and the cylinder radius respectively — reading
        # it as "toe radius" would put the plate's sole 15 mm below where it is and mis-scale every
        # clearance/air-time term. `_sole_offsets` is the true support point, orientation included.
        # Needs a kinematics pass: a fresh MjData has geom_xmat all zeros, which would read every
        # sole offset as 0 rather than raising.
        mujoco.mj_forward(self.model, self.data)
        self._toe_r = float(self._sole_offsets()[0])          # nominal-pose value, for reporting
        # The shipped point toe's radius: the fixed penetration budget every foot shape is graded
        # against below, so the floor check is the same test on every arm of the foot study.
        self._toe_r_nominal = 0.025
        self._col_gids = {}
        for s in "LR":
            for kind in ("foot", "heel"):
                g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{kind}_{s}_col")
                if g >= 0:
                    # Penetration budget for _floor_violation. For the shipped SPHERES this is the
                    # radius, exactly as before, so the control plant's terminations are unchanged.
                    #
                    # For a shaped foot it is NOT derived from the geometry, deliberately. Scaling
                    # it by the part's own thinnest half-extent was tried and is wrong: the 10 mm
                    # plate got a 2.5 mm budget against the point toe's 12.5 mm, and terminated on
                    # STEP 1 of every episode on 2 of 6 seeds — a foot graded five times harder
                    # than the foot it is being compared against. The budget is a property of the
                    # CHECK ("is the solver being driven through the floor"), not of the part, so
                    # every foot gets the same absolute depth as the toe it replaces and a
                    # termination means the same thing on every arm.
                    self._col_gids[g] = (
                        float(self.model.geom_size[g][0])
                        if int(self.model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
                        else float(self._toe_r_nominal))
        self._col_gids_side = {g: (0 if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                                       .endswith("_L_col") else 1) for g in self._col_gids}
        self._weight_n = float(np.sum(self.model.body_mass)) * float(-self.model.opt.gravity[2])
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
        self._push_axis = None          # None = random direction; "x"/"y" = the eval protocol

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
        # height_target_offset_m: crouch — see config.py. Applied to the reward TARGET only; the
        # keyframe (and so every reset) still starts at the settled stance and the policy earns
        # its way down. term_height stays absolute, so the kill line does not follow the crouch.
        self.height_target = float(self.default_qpos[2]) - float(cfg.height_target_offset_m)
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
        self.latched = (self.cfg.action_mode == "latched")
        self.cpg_mode = (self.cfg.action_mode == "cpg")
        if self.latched:
            self._init_latched_action()
        elif self.cpg_mode:
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
        if self.latched and (self.n_ankle_act or self.cfg.imp_enable or self.cfg.steer_enable):
            raise ValueError("action_mode='latched' carries impedance and steering INSIDE the spec "
                             "and has no active-ankle tail: imp_enable/steer_enable must be False "
                             "and the plant passive")
        self.action_dim += self.n_ankle_act
        # PER-STEP IMPEDANCE: per-leg kp/kd multipliers appended after the ankle tail, same
        # bolt-on contract as the ankle (decoders slice from the front; the generator never sees
        # these). Order [kp_L, kd_L, kp_R, kd_R], neutral 0 -> scale 1.0. Rides the same
        # actuation delay as the rest of the action (same CAN frame on hardware).
        self.imp_dim = 4 if self.cfg.imp_enable else 0
        self.imp_action_start = self.action_dim
        self.action_dim += self.imp_dim
        self.action_space = spaces.Box(-1.0, 1.0, (self.action_dim,), np.float32)
        self._prev_action = np.zeros(self.action_dim, np.float32)   # policy output (obs)
        self._prev_applied = np.zeros(self.action_dim, np.float32)  # post-delay, for coef_rate
        self._prev_motor_cmd = np.zeros(self.nu, np.float32)        # normalized targets, action_rate
        # the residual is per GAIT actuator (the active ankle has its own channel, not a residual)
        self._prev_residual = np.zeros(self.n_gait_act, np.float32)  # for the residual-rate penalty
        self._residual_rate_sq = 0.0
        # impedance-channel state. _imp_pristine is captured ONCE from the freshly built model and
        # restored at every reset BEFORE the DR draw, so per-step gain writes can never leak across
        # episodes even on arms where the randomizer no-ops (dr_enable=False).
        self._imp_prev_a = np.zeros(4, np.float32)
        self._imp_rate_sq = 0.0
        self._imp_pristine = (self.model.actuator_gainprm[:self.n_gait_act, 0].copy(),
                              self.model.actuator_biasprm[:self.n_gait_act, 1].copy(),
                              self.model.actuator_biasprm[:self.n_gait_act, 2].copy())
        self._imp_base = tuple(p.copy() for p in self._imp_pristine)
        self._imp_leg_ix = np.array([0, 0, 0, 1, 1, 1])[:self.n_gait_act]
        # anti-shuffle swing-floor state: per-foot EMA airborne fraction (see w_swing_floor).
        # Seeded AT the floor so a fresh episode starts penalty-free.
        self._swing_ema = np.full(2, self.cfg.swing_floor_frac, np.float32)
        self._swing_ema_coef = float(np.exp(-self.control_dt
                                            / max(self.cfg.swing_floor_tau_s, 1e-6)))
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
        if self.latched:
            # v2 per-tick FRAME (33): motor pos/vel/torque 18, gravity 3, gyro 3, LP yaw 1,
            # phase 2, previous residual 6 -- only what changes every tick is stacked. The latched
            # spec (or the library reference) and the task are observed ONCE, outside the stack,
            # in the once-block, together with the commit flag (its last entry = wrap_index).
            self.frame_dim = 3 * self.nu + 3 + 3 + 1 + self.phase_obs_dim + gait_v2.N_RESIDUAL
            self.once_dim = self._once_dim() + self.task_dim + 1
        else:
            self.frame_dim = (3 * self.nu + 3 + 3 + (3 if self.obs_base_vel else 0)
                              + self.phase_obs_dim + self.task_dim + self.action_dim)
            self.once_dim = 0
        obs_dim = self.frame_dim * self.cfg.history_len + self.once_dim
        self.n_actor_obs = int(obs_dim)
        self.wrap_index = int(obs_dim - 1) if self.latched else -1
        # PRIVILEGED CRITIC TAIL (asymmetric actor-critic, standard in the SOTA velocity-command
        # stacks): sim-only ground truth appended AFTER the history block, so the ACTOR's slice is
        # simply obs[:frame_dim*history_len] and the tail never has to exist on hardware. The value
        # function is estimating the return of a velocity-TRACKING task; without this it does so
        # while blind to velocity (R^2 0.807 recoverable from history — the remaining ~20% is pure
        # variance in every advantage estimate). The tail is ground truth on purpose: no noise, no
        # delay — the critic never runs on the robot. Layout (PRIV_DIM entries):
        #   [0:3] true base linear velocity, body frame, * obs_scales.base_vel
        #         (doubles as the supervised TARGET for the velocity-estimator head)
        #   [3:5] per-foot ground contact (toe OR heel), {0,1}
        #   [5]   base height error vs the settled stance (m)
        self.priv_dim = ((self.PRIV_DIM_V2 if self.latched else self.PRIV_DIM)
                         if self.cfg.obs_privileged_critic else 0)
        obs_dim += self.priv_dim
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
        # motor_vel_limit is a scalar OR a per-joint 6-tuple (the AKE90-8 cam/thigh and the
        # AK60-39 hip-roll differ by 2.1x in no-load output speed, so one number cannot serve both).
        # Broadcast either form to a per-actuator array once, here, so the hot path stays a clip.
        _vl = np.broadcast_to(np.asarray(self.cfg.motor_vel_limit, float),
                              (self.nu,)).astype(float).copy()
        if _vl.size != self.nu:
            raise ValueError(f"motor_vel_limit must be a scalar or {self.nu} values, got {_vl.size}")
        self._motor_vel_limit = np.where(_vl > 0.0, _vl, np.inf)
        self._vel_accel_limited = (bool(np.isfinite(self._motor_vel_limit).any())
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
        # v2 state (latch, resync, thermal, wind, delay ring, library) -- inert on legacy plants
        if self.latched:
            self._init_latched_state()
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
        # gait-actuator torque-speed curve: needs a measured phase resistance, so it stays OFF
        # until motor_r_ohm is filled in (see the config note -- R is not measured on this robot)
        self._motor_ts_curve = bool(c.motor_r_ohm) and c.motor_bus_volts > 0.0
        if self._motor_ts_curve:
            self._motor_kt = np.asarray(c.motor_kt_joint, float)[:self.nu]
            self._motor_r = np.asarray(c.motor_r_ohm, float)[:self.nu]
            if self._motor_kt.size != self.nu or self._motor_r.size != self.nu:
                raise ValueError(f"motor_kt_joint/motor_r_ohm must have {self.nu} entries "
                                 f"(got {self._motor_kt.size}/{self._motor_r.size})")
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
        self._sag_scale = 1.0 - c.dr_torque_sag * self._dr.scale * self._sag_state
        self._apply_torque_limit()

    def _apply_motor_torque_speed(self):
        """Real torque-speed envelope for the six GAIT actuators, re-evaluated every substep.

        A brushless drive is CURRENT-limited at low speed and VOLTAGE-limited past the corner:

            tau(w) = min( tau_peak,  Kt_j * (V_bus - Kt_j * w_joint) / R )

        Back-EMF referred to the joint is exactly Kt_j*w_joint, since Kt_motor = Kt_j/G and
        w_motor = w_joint*G. NOT the linear-from-zero derate the ankle uses -- that shape ignores
        the flat branch and under-reports torque exactly where this robot operates. At 48 V the
        corner is 4.7-10.2 rad/s (hip_roll) and 6.8-20.5 rad/s (cam/thigh) against observed peaks
        of 1.48 and 5.40, so this is currently a no-op; it exists for the fast-running case.

        Multiplies the same curriculum / DR / sag scaling _apply_torque_limit applies, so it
        composes with the torque-budget curriculum instead of silently overwriting it.
        """
        w = np.abs(self.data.qvel[self.act_dadr])
        kt = self._motor_kt
        v_avail = np.maximum(self.cfg.motor_bus_volts - kt * w, 0.0)
        tau_volt = kt * v_avail / self._motor_r
        base = (self._orig_forcerange[:self.nu, 1] * self._torque_scale
                * self._dr_torque_scale * self._sag_scale)
        lim = np.minimum(base, tau_volt)
        self.model.actuator_forcerange[:self.nu, 0] = -lim
        self.model.actuator_forcerange[:self.nu, 1] = lim

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
        self.height_target = float(self.default_qpos[2]) - float(self.cfg.height_target_offset_m)
        # how far this ankle sags relative to the shipped k=28.65 preloaded stance. Reported by the
        # statics tool: a large sag IS the answer for a soft arm, not a nuisance to be normalized.
        # (measured against the SETTLED height, not the crouch target)
        self.settle_sag_m = z_before - float(self.default_qpos[2])
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

    def set_drive_bandwidth_log10(self, x):
        """Curriculum setter: x = log10(bandwidth in Hz) -> the target-filter coefficient.

        The drive's position loop is a first-order lag, so a bandwidth f maps to an EMA retention
        of exp(-control_dt / tau), tau = 1/(2*pi*f). Expressed in Hz (and ramped in log-Hz) so the
        same curriculum means the same physical drive at any control rate -- see the
        drive_bandwidth_hz note in config.py."""
        hz = float(10.0 ** float(x))
        tau = 1.0 / (2.0 * np.pi * max(hz, 1e-6))
        self.cfg.action_filter = float(np.exp(-self.control_dt / tau))
        self.drive_bandwidth_hz = hz

    def set_dr_scale(self, s):
        """0..1 curriculum on the WIDTH of every domain-randomization range (applies from the next
        reset). Measured on teleop_v3: on the nominal plant the policy survives 196 s and never
        falls; at full-width DR it survives 4.8 s — randomization was ~20x more destructive than
        pushes, trips and sensor noise combined, and unlike all of those it had no curriculum.
        A policy cannot learn to be robust to a plant it cannot stand up on."""
        self._dr.scale = float(np.clip(s, 0.0, 1.0))
        # The measured sim2real calibration axes (homing zero, IMU mount rotation, IMU dropout,
        # bus sag) ride the SAME ramp -- see SensorNoise.scale for why they must not stand at full
        # width from step 0.
        self._noise.scale = self._dr.scale

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

    def _sole_offsets(self):
        """Per foot, the distance from the collision geom's CENTRE down to its lowest point, in the
        geom's current orientation. Constant (= the radius) for the shipped spheres; for the plate
        and the blade it depends on how the foot is tilted right now, which is the whole point of
        those variants — a plate on its edge is 15 mm from centre to floor, flat it is 5 mm."""
        m, d = self.model, self.data
        out = np.empty(2)
        T = mujoco.mjtGeom
        for i, g in enumerate(self.foot_gids):
            a = d.geom_xmat[g].reshape(3, 3)[2, :]     # world-z row of the geom's rotation
            s = m.geom_size[g]
            t = int(m.geom_type[g])
            if t == T.mjGEOM_SPHERE:
                out[i] = s[0]
            elif t == T.mjGEOM_BOX:
                out[i] = abs(a[0]) * s[0] + abs(a[1]) * s[1] + abs(a[2]) * s[2]
            elif t == T.mjGEOM_CYLINDER:
                out[i] = s[0] * np.hypot(a[0], a[1]) + s[1] * abs(a[2])
            elif t == T.mjGEOM_CAPSULE:
                out[i] = s[0] + s[1] * abs(a[2])
            else:
                out[i] = m.geom_rbound[g]
        return out

    def _toe_heights(self):
        """Height of each foot's SOLE above the floor (0 = touching)."""
        return self.data.geom_xpos[self.foot_gids_arr, 2] - self._sole_offsets()

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
        if self.latched:
            # v2 FRAME (33). LP yaw from the MEASURED gyro (the Pi computes the same EMA); the
            # previous residual is the only part of the old prev_action whose history carries
            # information; the spec/task live in the once-block (see _obs).
            self._yaw_lp_meas = (self._yaw_lp_a * self._yaw_lp_meas
                                 + (1.0 - self._yaw_lp_a) * float(angv[2]))
            self._accel_body_last = accel_body
            phase_ch = np.array([np.sin(self._phase), np.cos(self._phase)])
            parts = [motor_pos * s["motor_pos"], motor_vel * s["motor_vel"],
                     motor_trq * s["motor_torque"], grav * s["gravity"], angv * s["ang_vel"],
                     np.array([self._yaw_lp_meas * s["ang_vel"]]), phase_ch, self._prev_residual]
            return np.concatenate(parts).astype(np.float32)
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
        hist = self._history[self._hist_idx].reshape(-1)
        if self.latched:
            parts = [hist, self._once_block()]
            if self.priv_dim:
                parts.append(self._priv_tail_v2())
            return np.concatenate(parts).astype(np.float32)
        if not self.priv_dim:
            return hist.astype(np.float32)
        # critic-only ground truth; layout documented at the priv_dim definition in __init__
        tail = np.empty(self.PRIV_DIM)
        tail[0:3] = self._vel_body() * self.cfg.obs_scales["base_vel"]
        tail[3:5] = (self._foot_contacts()
                     | (self._toe_heights() < self.cfg.grounded_h)).astype(float)
        tail[5] = float(self.data.qpos[2]) - self.height_target
        return np.concatenate([hist, tail]).astype(np.float32)

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
                self._task[1] = np.clip((self._sprint_D - self._sprint_d)
                                        / float(c.sprint_task_scale_m), 0.0, 1.0)
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
        # undo any per-step impedance gain writes from the previous episode BEFORE the DR draw:
        # PlantRandomizer restores its own snapshot when it runs, but on dr_enable=False arms it
        # cannot be relied on to clean up after the impedance channel.
        if self.imp_dim or self.latched:
            self.model.actuator_gainprm[:self.n_gait_act, 0] = self._imp_pristine[0]
            self.model.actuator_biasprm[:self.n_gait_act, 1] = self._imp_pristine[1]
            self.model.actuator_biasprm[:self.n_gait_act, 2] = self._imp_pristine[2]
        # per-episode plant + measurement draw, BEFORE mj_forward so the new masses/inertias are
        # in this episode's mass matrix and the standing-torque baseline below reflects them.
        ep = self._dr.resample(self.model, self.np_random)
        self._dr_torque_scale = ep["torque_scale"]
        self._sag_scale, self._sag_state = 1.0, 0.0
        self._apply_torque_limit()
        if self.imp_dim or self.latched:
            # THIS episode's base gains (post-DR draw): the per-step channel scales these
            self._imp_base = (self.model.actuator_gainprm[:self.n_gait_act, 0].copy(),
                              self.model.actuator_biasprm[:self.n_gait_act, 1].copy(),
                              self.model.actuator_biasprm[:self.n_gait_act, 2].copy())
            self._imp_prev_a[:] = 0.0
            self._imp_rate_sq = 0.0
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
        self.data.qpos[self._noise_qadr] += self.np_random.uniform(-n, n, self._noise_qadr.size)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if self.latched:
            self._reset_latched(ep)
        self._prev_action[:] = 0.0
        self._prev_applied[:] = 0.0
        self._prev_motor_cmd[:] = 0.0
        self._prev_residual[:] = 0.0
        self._swing_ema[:] = self.cfg.swing_floor_frac
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
        yaw = c.w_track_yaw * float(np.exp(-e_yaw * e_yaw))
        # anti-stand-subsidy: a policy standing still under a speed command tracks yaw_cmd ~0
        # perfectly and banks the whole term (imp_m3b: 2.0/step for 200M while refusing to move).
        # Couple yaw income to LINEAR competence so it only flows while the speed command is
        # being followed. Stand commands keep the uncoupled term (v=0 is tracked by standing).
        if c.track_yaw_couple and not self._standing:
            yaw *= float(np.exp(-e_lin * e_lin))
        t["track_yaw"] = yaw
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

    def _run_physics(self, target, gains=None):
        """One control step of plant: EMA-filter the target, clip to ctrlrange, run
        control_decimation sim substeps (OR-accumulating foot contact so a sub-20 ms hop can't
        pass as continuous flight/contact at the 50 Hz boundary)."""
        if self.latched:
            return self._run_physics_latched(target, gains)
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
            np.clip(v_des, -self._motor_vel_limit, self._motor_vel_limit, out=v_des)
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
            if self._motor_ts_curve:
                self._apply_motor_torque_speed()
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

    def _pre_physics_forces(self, pitch, pitch_rate):
        """Everything that writes qvel / xfrc_applied / qfrc_applied BEFORE the physics runs:
        pushes, trips, the pitch-assist wheel, the ankle-torque reflex and (v2) the wind. Shared
        by the legacy and the latched step, in this exact order (the RNG stream is part of the
        legacy bit-identity)."""
        c = self.cfg
        # gentle random shove BEFORE the physics runs (free translational axes only).
        # Scaled by the DR curriculum: measured on the walk_fwd_easy walker with paired seeds,
        # pushes alone take it from 3/12 surviving episodes to 0/12 -- the same cost as full-width
        # plant DR, which HAS a ramp. Leaving them at full width from step 0 is what pinned
        # walk_fwd2 at 2 s for 300 M steps.
        adv = self._dr.scale if self.cfg.adversity_curriculum else 1.0
        self._push_countdown -= 1
        if self._push_countdown <= 0:
            ang = self.np_random.uniform(0.0, 2.0 * np.pi)
            if self._push_axis is not None:          # eval protocol: one body axis, random sign
                ang = (0.0 if self._push_axis == "x" else 0.5 * np.pi) + (np.pi if ang > np.pi else 0.0)
            dv = c.push_dv
            if c.push_dv_range[1] > c.push_dv_range[0]:      # v2: |dv| drawn per push
                dv = float(self.np_random.uniform(*c.push_dv_range))
            if not self.base_lock[0]:
                self.data.qvel[self._base_x_dadr] += adv * dv * np.cos(ang)
            if not self.base_lock[1]:
                self.data.qvel[self._base_y_dadr] += adv * dv * np.sin(ang)
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
        elif c.trip_prob > 0.0 and self.np_random.random() < adv * c.trip_prob:
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

        if self.latched:
            self._apply_wind()

    def step(self, action):
        if self.latched:
            return self._step_latched(action)
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
        if self.imp_dim:
            # per-leg impedance from the (delayed) action tail. Exp map with asymmetric headroom,
            # neutral 0 -> 1.0; imp_kd_up=1.0 makes the kd branch soften-only (log 1 = 0).
            ia = applied[self.imp_action_start:]
            kp_leg = np.exp(np.where(ia[0::2] >= 0.0, ia[0::2] * np.log(c.imp_kp_up),
                                     ia[0::2] * np.log(c.imp_kp_dn)))
            kd_leg = np.exp(np.where(ia[1::2] >= 0.0, ia[1::2] * np.log(c.imp_kd_up),
                                     ia[1::2] * np.log(c.imp_kd_dn)))
            kp_s = kp_leg[self._imp_leg_ix]
            kd_s = kd_leg[self._imp_leg_ix]
            ng = self.n_gait_act
            self.model.actuator_gainprm[:ng, 0] = self._imp_base[0] * kp_s
            self.model.actuator_biasprm[:ng, 1] = self._imp_base[1] * kp_s
            self.model.actuator_biasprm[:ng, 2] = self._imp_base[2] * kd_s
            self._imp_rate_sq = float(np.sum((ia - self._imp_prev_a) ** 2))
            self._imp_prev_a = ia.copy()

        self._pre_physics_forces(pitch, pitch_rate)

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
        # PER-FOOT airborne flag. Raw physics, not a reward term: the entropy competence gate keys
        # off this precisely so it cannot drift when a reward weight or the control rate changes
        # (see ent_gate_swing_frac in config.py).
        #
        # Per-foot, not a mean, because the mean cannot tell an alternating gait from a leg parked
        # in the air: MEASURED, m2drv_d12_s0 scores a higher mean swing fraction (0.47) than the arm
        # that actually walks (d3, 0.34) purely by folding one leg up until workspace_kill fires.
        # The gate takes the MIN across feet, so both feet have to leave the ground to open it.
        # Airborne uses the SAME definition as the air_time term above -- contact OR a toe below
        # grounded_h. _foot_contacts() alone only watches the toe sphere, so a foot resting on its
        # heel reads as airborne: in the settled m2 keyframe that misreports one foot as off the
        # ground while the base has not moved a millimetre.
        _air = ~(self._foot_contacts() | (self._toe_heights() < c.grounded_h))
        info["foot_air"] = _air.astype(np.float64)
        info["swing_frac"] = float(_air.mean())          # logged for continuity, not gated on
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
        # sprint income frame: pay world-x (the dash axis) instead of body-forward, so a yaw-free
        # policy cannot bank speed income by circling (measured on sprint_m6_mit — see config).
        # qvel[0] is the base's world-x DOF in every milestone (locked ones are just constrained).
        if c.sprint_world_speed and c.objective == "sprint":
            vx = float(self.data.qvel[0])
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
        # anti-circling, observable version: sustained gyro-z is what a circler cannot avoid and
        # what the policy CAN see (unlike absolute heading — see config.w_yaw_rate). 0 = off.
        if c.w_yaw_rate > 0.0:
            # v2: bill the LOW-PASSED yaw rate (drift, not gait wobble) -- the same filtered
            # signal the actor observes (§03/§10)
            yaw_sig = self._yaw_lp_true if (self.latched and c.yaw_lp_tau_s > 0.0) else float(angv[2])
            t["yaw_rate"] = self._pen(-c.w_yaw_rate * yaw_sig ** 2)
        else:
            t["yaw_rate"] = 0.0

        # ----- gait shaping (anti-skate + phase schedule) -----
        # In command mode the gait terms key off the COMMANDED speed, not a constant: with the old
        # `cmd_speed = v_ceiling` a walk command would still be graded under running rules (stance
        # caps, flight-phase demand), which is exactly backwards.
        if not self.command_mode:
            cmd_speed = c.v_ceiling if run_phase else 0.0
        gait_on = cmd_speed >= c.gait_cmd_gate
        dt = self.control_dt
        toe_pos = self.data.geom_xpos[self.foot_gids_arr].copy()
        heights = self._toe_heights()
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

        # ANTI-SHUFFLE swing floor (2026-08-28): imp_m3_long converged on symmetric shuffling
        # (worse foot airborne 3%) — duty_sym is blind to it and contact_switch rewards it. Bill
        # the WORSE foot's EMA airborne fraction below swing_floor_frac, only while a speed is
        # commanded (gait_on — standing stays legal, the stop-farm rule; worse foot not mean,
        # the k350 one-leg-patter rule).
        if c.w_swing_floor > 0.0:
            a_ema = self._swing_ema_coef
            self._swing_ema = (a_ema * self._swing_ema
                               + (1.0 - a_ema) * (~grounded).astype(np.float32))
            if gait_on:
                deficit = max(0.0, c.swing_floor_frac - float(self._swing_ema.min()))
                t["swing_floor"] = self._pen(-c.w_swing_floor * deficit ** 2)
            else:
                t["swing_floor"] = 0.0
        else:
            t["swing_floor"] = 0.0

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
        t["imp_rate"] = self._pen(-c.w_imp_rate * self._imp_rate_sq)
        # anti-crutch: pay for the assist torque the policy provokes (0 when it balances itself, so
        # the assist becomes a safety net the policy is pushed to stop relying on). 0 when disabled.
        t["assist_pen"] = self._pen(-c.w_assist_penalty * self._assist_torque ** 2)

        # ----- posture -----
        t["upright"] = self._pen(-c.w_upright * (grav[0] ** 2 + grav[1] ** 2))
        if self.z_locked:           # height/vz are meaningless when Z is railed
            t["height"] = 0.0
            t["vz"] = 0.0
        elif c.height_floor_m > 0.0:
            # v2 (§10): a fence around the validated ride band (LUT feasible band 0.81-1.04 m),
            # quadratic BELOW the floor only; no upper bound, which would tax the flight phase
            t["height"] = self._pen(-c.w_height
                                    * max(0.0, c.height_floor_m - float(self.data.qpos[2])) ** 2)
            t["vz"] = self._pen(-c.w_vz * self.data.qvel[2] ** 2)
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

        # ----- v2 terms (artifact §10); every one is exactly 0.0 at the legacy weights -----
        # lane keeping: the 1.22 m lane is the deliverable's real constraint (§08)
        t["lane"] = (self._pen(-c.w_lane * max(0.0, abs(float(self.data.qpos[1])) - c.lane_free_m) ** 2)
                     if c.w_lane > 0.0 else 0.0)
        # thermal budget: quadratic above thermal_pen_start of dT_max, per motor (§07)
        if c.w_thermal > 0.0 and self.latched and self._thermal_on:
            over = np.maximum(self._theta - c.thermal_pen_start, 0.0)
            t["thermal"] = self._pen(-c.w_thermal * float(np.sum(over ** 2)))
        else:
            t["thermal"] = 0.0
        # spec change billed ONCE per cycle at commit; no phase gate anywhere (the m3 audit rule)
        t["spec_cycle"] = (self._pen(-c.w_spec_cycle * self._spec_change_sq)
                           if c.w_spec_cycle > 0.0 and self.latched else 0.0)
        # standing price on the five relationship knobs: knobs at zero IS the mirror gait
        t["knob"] = (self._pen(-c.w_knob * float(np.sum(gait_v2.knob_vector(self._spec_live, self._lay) ** 2)))
                     if c.w_knob > 0.0 and self.latched else 0.0)
        # library variant: the stabilizer's small tracking bill against the reference (§09 st.3)
        if c.w_track > 0.0 and self.latched and self._q_ref is not None:
            dq = self.data.qpos[self.act_qadr[:self.n_gait_act]] - self._q_ref
            t["track"] = self._pen(-c.w_track * float(np.sum(dq ** 2)))
        else:
            t["track"] = 0.0

        return float(sum(t.values())), t


    # =====================================================================================
    # DASH-01 Walker v2 -- the LATCHED gait spec (artifact rev 2026-09-09). Everything in this
    # block is reached only when cfg.action_mode == "latched"; legacy plants never enter it.
    # =====================================================================================
    def _init_latched_action(self):
        c = self.cfg
        self.n_steer = 0
        self._cpg_lut = None
        self.phase_obs_dim = 2
        self._lay = gait_v2.Layout(c.n_harmonics)
        self.spec_source = str(c.spec_source)
        if self.spec_source not in ("policy", "library"):
            raise ValueError(f"spec_source {self.spec_source!r} not in ('policy', 'library')")
        if c.drive_bandwidth_hz > 0.0 or c.action_filter > 0.0:
            raise ValueError("latched mode has ONE drive lag (armature + substep delay): set "
                             "drive_bandwidth_hz=0 and action_filter=0 (no EMA target filter)")
        if self.spec_source == "policy":
            # [0:44] spec, latched at phi-wrap; [44:50] residual, every tick
            self.spec_dim = self._lay.spec_dim
            self.action_dim = self._lay.action_dim
            self.latched_slice = slice(0, self.spec_dim)
        else:
            # [0:6] residual, every tick; [6:9] optional latched mods (df/f, amplitude, lift)
            self.n_lib_latched = 3 if c.library_latched_dims else 0
            self.spec_dim = 0
            self.action_dim = gait_v2.N_RESIDUAL + self.n_lib_latched
            self.latched_slice = slice(gait_v2.N_RESIDUAL, self.action_dim)

    def _once_dim(self):
        """Width of the once-block MINUS task and commit: the live spec (policy variant) or the
        library reference q_ref(phi) 6 + qdot_ref(phi) 6 + q_ref(phi + 1/4 cycle) 6 + td_hat 2."""
        return self.spec_dim if self.spec_source == "policy" else 3 * self.n_gait_act + 2

    def _init_latched_state(self):
        c = self.cfg
        lay = self._lay
        self._spec_live = gait_v2.neutral_spec(lay)
        self._commit_flag = True             # cycle 0 opens at reset: the first spec commits
        self._cycle_n = 0
        self._spec_change_sq = 0.0
        self._f_hz = gait_v2.frequency(0.0, c.gait_freq_hz)
        self._q_ref = None
        # contact-triggered resync (§05)
        self._kappa = float(c.resync_kappa)
        self._td_hat = np.array([0.0, np.pi])   # the stance windows' opening phases (L, R)
        self._resync_done = np.zeros(2, bool)
        self._td_err_last = np.zeros(2)
        # LP yaw (true for the reward, measured for the obs)
        self._yaw_lp_a = (float(np.exp(-self.control_dt / c.yaw_lp_tau_s))
                          if c.yaw_lp_tau_s > 0.0 else 0.0)
        self._yaw_lp_true = 0.0
        self._yaw_lp_meas = 0.0
        self._accel_body_last = np.zeros(3)
        # substep-granular command ring: (target 6 | kp scale 6 | kd scale 6) per 1 ms substep
        self._ring_len = 64
        self._ring = np.zeros((self._ring_len, 3 * self.nu))
        self._ring_head = 0
        self._delay_ms = float(c.drive_delay_ms)
        self._delay_sub = int(round(self._delay_ms * 1e-3 / self.sim_dt))
        self._delay_override_ms = None
        self._sub_n = c.control_decimation
        # thermal (§07)
        self._thermal_on = bool(c.thermal_enable)
        self._tau_cont = np.asarray(c.thermal_tau_cont, dtype=float)[:self.nu]
        if self._thermal_on and self._tau_cont.size != self.nu:
            raise ValueError(f"thermal_tau_cont needs {self.nu} entries")
        self._theta = np.zeros(self.nu)
        self._theta_cmax = 1.0
        self._theta_override = None
        self._tau_sq_acc = np.zeros(self.nu)
        # wind (§08)
        self._wind_on = bool(c.wind_force_max_n > 0.0 or c.gust_force_n > 0.0)
        self._wind_f = np.zeros(2)
        self._wind_override = None
        self._gust_left = 0
        self._gust_countdown = 10 ** 9
        self._gust_vec = np.zeros(2)
        self._ep_draw = {}
        # library variant (§09)
        self._raibert = None
        self._lib_entries = None
        self._lib_override = None
        self._theta_ep = None
        self._lib_stand = None
        self._lib_v_ref = 0.0
        if self.spec_source == "library":
            self._lib_entries = self._load_library(c.library_path)
            if c.raibert_enable:
                self._raibert = RaibertPrior(c.raibert_kp, c.raibert_ki, c.raibert_imax,
                                             c.raibert_ky, c.raibert_kr, c.offset_max_rad,
                                             c.raibert_roll_tau_s)

    # ---- per-episode -----------------------------------------------------------------------
    def _reset_latched(self, ep):
        c = self.cfg
        self._ep_draw = dict(ep)
        self._spec_live[:] = gait_v2.neutral_spec(self._lay)
        self._commit_flag = True
        self._cycle_n = 0
        self._spec_change_sq = 0.0
        self._q_ref = None
        self._td_hat[:] = (0.0, np.pi)
        self._resync_done[:] = False
        self._kappa = float(ep.get("kappa", c.resync_kappa))
        self._yaw_lp_true = 0.0
        self._yaw_lp_meas = 0.0
        self._accel_body_last[:] = 0.0
        self._delay_ms = float(ep.get("delay_ms", c.drive_delay_ms))
        if self._delay_override_ms is not None:
            self._delay_ms = float(self._delay_override_ms)
        self._delay_sub = int(round(self._delay_ms * 1e-3 / self.sim_dt))
        if self._delay_sub >= self._ring_len:
            raise ValueError(f"delay {self._delay_ms} ms exceeds the command ring")
        self._theta[:] = np.asarray(ep.get("theta0", np.zeros(self.nu)), dtype=float)[:self.nu]
        if self._theta_override is not None:
            self._theta[:] = self._theta_override
        self._theta_cmax = float(ep.get("thermal_cmax", 1.0))
        self._tau_sq_acc[:] = 0.0
        if self._wind_on:
            w = float(c.wind_force_max_n)
            self._wind_f[:] = self.np_random.uniform(-w, w, 2) if w > 0 else 0.0
            if self._wind_override is not None:
                self._wind_f[:] = self._wind_override
            self._gust_left = 0
            self._gust_countdown = self._next_gust_in()
        if self.spec_source == "library":
            self._lib_reset()
        # the ring holds the last 64 ms of commands: at reset, the stance hold
        gains0 = (np.ones(self.nu), np.ones(self.nu))
        self.prime_command_ring(self._spec_live, reflex_free_hold=True)
        self._set_gains(*gains0)

    def _next_gust_in(self):
        lo, hi = self.cfg.gust_interval_s
        if self.cfg.gust_force_n <= 0.0 or hi <= 0.0:
            return 10 ** 9
        return max(1, int(round(float(self.np_random.uniform(lo, hi)) / self.control_dt)))

    # ---- the step -------------------------------------------------------------------------
    def _step_latched(self, action):
        c = self.cfg
        lay = self._lay
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self._ctrl_drop_prob > 0.0 and self.np_random.random() < self._ctrl_drop_prob:
            action = self._prev_action.copy()
        # ---- the LATCH REGISTER: the spec in this action reaches the generator only if this
        # tick is the first of a cycle (the commit flag the policy just observed); otherwise it
        # is discarded -- not penalised, discarded, so there is no gate to park the clock against
        committed = bool(self._commit_flag)
        self._spec_change_sq = 0.0
        if self.spec_source == "policy":
            if committed:
                new = action[:self.spec_dim].astype(np.float64)
                if self._cycle_n > 0:
                    d = new - self._spec_live
                    self._spec_change_sq = float(np.dot(d, d))
                self._spec_live[:] = new
            residual = action[lay.residual].astype(np.float64)
        else:
            residual = action[:gait_v2.N_RESIDUAL].astype(np.float64)
            if committed:
                self._lib_commit(action[gait_v2.N_RESIDUAL:])
        spec = self._spec_live
        f = gait_v2.frequency(spec[lay.freq], c.gait_freq_hz)
        self._f_hz = f
        phi = self._phase
        _, _, delta = gait_v2.leg_phases(phi, spec, c, lay)
        self._phase_reward = phi
        self._phase_reward_R = phi + np.pi - delta
        grav = self._gravity_body()
        angv = self._ang_vel_body()
        roll = float(grav[1])
        roll_rate = float(angv[0])
        pitch = float(grav[0])
        pitch_rate = float(angv[1])
        if c.pitch_reflex_rate_lp > 0.0:
            self._reflex_prate_filt = (c.pitch_reflex_rate_lp * self._reflex_prate_filt
                                       + (1.0 - c.pitch_reflex_rate_lp) * pitch_rate)
            pitch_rate = self._reflex_prate_filt
        if self._raibert is not None:
            self._raibert.filter_roll(roll, self.control_dt)
        target6 = gait_v2.assemble(spec, phi, roll, roll_rate, self._nominal6, c,
                                   pitch=pitch, pitch_rate=pitch_rate, layout=lay)
        # the reflex-free reference (a pure function of phi): the tracking bill and once-block
        self._q_ref = gait_v2.assemble(spec, phi, 0.0, 0.0, self._nominal6, c, layout=lay,
                                       reflexes=False) if (c.w_track > 0.0) else None
        target = target6 + c.residual_scale * residual
        kp_s, kd_s = gait_v2.gains(spec, phi, c, lay)
        motor_cmd = ((target - self.nominal_ctrl) / c.action_scale).astype(np.float32)
        self._residual_sq = float(np.sum(residual ** 2))
        self._residual_rate_sq = float(np.sum((residual - self._prev_residual) ** 2))
        self._prev_residual[:] = residual
        self._pre_physics_forces(pitch, pitch_rate)
        contact_acc = self._run_physics(target, gains=(kp_s, kd_s))
        self._update_torque_sag()
        if self._thermal_on:
            self._thermal_step()
        self._elapsed_t += self.control_dt
        finished = c.objective == "sprint" and self._update_sprint()
        wz = float(self._ang_vel_body()[2])
        self._yaw_lp_true = self._yaw_lp_a * self._yaw_lp_true + (1.0 - self._yaw_lp_a) * wz
        # advance the clock (+ the contact resync); this is what sets the NEXT tick's commit flag
        grounded_now = contact_acc | (self._toe_heights() < c.grounded_h)
        self._advance_phase_latched(f, grounded_now)
        self._step_n += 1
        reward, terms = self._reward(motor_cmd, contact_acc)
        self._update_task()
        self._push_frame(self._proprio())
        reward *= self._reward_dt_scale
        reward = max(reward, -c.step_reward_floor * self._reward_dt_scale)
        terminated = self._fallen()
        if terminated:
            reward -= c.fall_penalty
        elif finished:
            terminated = True
            reward += c.finish_bonus
        truncated = self._step_n >= self.max_steps
        self._prev_action[:] = action
        self._prev_applied[:] = action           # no action delay in latched mode (gait_diag)
        self._prev_motor_cmd[:] = motor_cmd
        if self.on_control_step is not None:
            self.on_control_step()
        info = {"reward_terms": terms}
        _lim = self.model.actuator_forcerange[:self.nu, 1]
        info["torque_util"] = float(np.mean(
            np.abs(self.data.actuator_force[:self.nu]) / np.maximum(_lim, 1e-6)))
        _air = ~(self._foot_contacts() | (self._toe_heights() < c.grounded_h))
        info["foot_air"] = _air.astype(np.float64)
        info["swing_frac"] = float(_air.mean())
        if c.objective == "sprint":
            info["sprint"] = self._sprint_info(finished)
        info.update(self._ankle_info())
        info["v2"] = dict(commit=committed, f_hz=float(f), cycle=int(self._cycle_n), phi=float(phi),
                          theta_max=float(self._theta.max()) if self._thermal_on else 0.0,
                          delay_ms=float(self._delay_ms), kappa=float(self._kappa),
                          td_err=self._td_err_last.copy(), spec_change=self._spec_change_sq)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def _advance_phase_latched(self, f, grounded_now):
        """Advance the gait clock at the LATCHED frequency and apply the contact-triggered resync
        (artifact §05): on the first touchdown of foot j this cycle, the clock closes a fraction
        kappa of the timing error against that foot's habitual touchdown phase (a slow EMA), but
        only inside a +-window around it -- a double contact or a stumble does not move the
        clock. A wrap (free-running or carried across 2pi by the resync) sets the commit flag the
        next observation carries; the spec then latches from THAT tick's action (+1 tick)."""
        c = self.cfg
        two_pi = 2.0 * np.pi
        phi = float(self._phase)
        td = grounded_now & ~self._grounded_prev
        corr = 0.0
        for j in range(2):
            if not td[j] or self._resync_done[j]:
                continue
            self._resync_done[j] = True
            err = (phi - self._td_hat[j] + np.pi) % two_pi - np.pi      # wrap(phi_raw - td_hat)
            self._td_err_last[j] = err
            inside = abs(err) <= c.resync_window_cycle * two_pi
            if inside and self._cycle_n >= c.resync_warmup_cycles and self._kappa > 0.0:
                corr += self._kappa * (-err)                           # kappa * wrap(td_hat - phi)
            # habit update from the PRE-resync phase; this is the gait's own touchdown timing
            self._td_hat[j] = (self._td_hat[j] + err / max(c.resync_n_ema, 1.0)) % two_pi
        phi = max(phi + corr, 0.0)          # never re-enter the previous cycle
        phi += two_pi * f * self.control_dt
        if phi >= two_pi:
            phi -= two_pi
            self._commit_flag = True
            self._cycle_n += 1
            self._resync_done[:] = False
        else:
            self._commit_flag = False
        self._phase = phi % two_pi

    # ---- plant pieces ---------------------------------------------------------------------
    def _set_gains(self, kp_s, kd_s):
        ng = self.n_gait_act
        self.model.actuator_gainprm[:ng, 0] = self._imp_base[0] * kp_s[:ng]
        self.model.actuator_biasprm[:ng, 1] = self._imp_base[1] * kp_s[:ng]
        self.model.actuator_biasprm[:ng, 2] = self._imp_base[2] * kd_s[:ng]

    def _run_physics_latched(self, target, gains):
        """The v2 drive: clip, slew-limit (no-load speed cap), homing offset, then push the
        (target, kp, kd) command through the substep delay ring so what the plant sees is the
        command from delay_ms ago -- one transport delay, at 1 kHz granularity, on the same
        frame the gains ride. Substep torque is accumulated for the thermal node."""
        c = self.cfg
        tgt = np.clip(target, self.ctrl_lo, self.ctrl_hi)
        if self._vel_accel_limited:
            dt = self.control_dt
            v_des = (tgt - self._prev_cmd_pos) / dt
            if c.motor_accel_limit > 0.0:
                dv = c.motor_accel_limit * dt
                v_des = np.clip(v_des, self._prev_cmd_vel - dv, self._prev_cmd_vel + dv)
            np.clip(v_des, -self._motor_vel_limit, self._motor_vel_limit, out=v_des)
            tgt = self._prev_cmd_pos + v_des * dt
            self._prev_cmd_vel = v_des
            self._prev_cmd_pos = tgt.copy()
        if c.dr_joint_zero_deg > 0.0:
            tgt = tgt + self._noise.zero_offset[:len(tgt)]
        kp_s, kd_s = gains
        cmd = np.concatenate([tgt, kp_s, kd_s])
        n = c.control_decimation
        if self._ctrl_jitter_substeps > 0:
            n = max(1, n + int(self.np_random.integers(
                -self._ctrl_jitter_substeps, self._ctrl_jitter_substeps + 1)))
        contact_acc = np.zeros(2, bool)
        ring, L, nu = self._ring, self._ring_len, self.nu
        for _ in range(n):
            ring[self._ring_head] = cmd
            ap = ring[(self._ring_head - self._delay_sub) % L]
            self._ring_head = (self._ring_head + 1) % L
            self.data.ctrl[:] = ap[:nu]
            self._set_gains(ap[nu:2 * nu], ap[2 * nu:])
            if self._motor_ts_curve:
                self._apply_motor_torque_speed()
            mujoco.mj_step(self.model, self.data)
            if not contact_acc.all():
                contact_acc |= self._foot_contacts()
            if self._thermal_on:
                self._tau_sq_acc += self.data.actuator_force[:nu] ** 2
        self._sub_n = n
        return contact_acc

    def prime_command_ring(self, spec, reflex_free_hold=False):
        """Fill the delay ring with what the generator WOULD have commanded over the last
        ring_len substeps at `spec` (phases phi = -k * 2 pi f dt_sub), so a periodic orbit's
        delayed commands are consistent at the section (the return-map solver) and an episode
        does not start with 64 ms of zeros. reflex_free_hold: the standing hold (targets at the
        nominal, gains neutral) -- what a fresh episode starts from."""
        c = self.cfg
        lay = self._lay
        f = gait_v2.frequency(spec[lay.freq], c.gait_freq_hz)
        nu = self.nu
        for k in range(self._ring_len):
            if reflex_free_hold:
                tgt = np.asarray(self.nominal_ctrl, dtype=float).copy()
                kp_s, kd_s = np.ones(nu), np.ones(nu)
            else:
                ph = -(self._ring_len - k) * 2.0 * np.pi * f * self.sim_dt
                tgt = gait_v2.assemble(spec, ph, 0.0, 0.0, self._nominal6, c, layout=lay,
                                       reflexes=False)
                kp_s, kd_s = gait_v2.gains(spec, ph, c, lay)
            self._ring[k] = np.concatenate([np.clip(tgt, self.ctrl_lo, self.ctrl_hi), kp_s, kd_s])
        self._ring_head = 0
        # the slew limiter's memory: the last two ticks of the same trajectory
        if reflex_free_hold:
            self._prev_cmd_pos[:] = self.nominal_ctrl
            self._prev_cmd_vel[:] = 0.0
        else:
            dphi = 2.0 * np.pi * f * self.control_dt
            t1 = gait_v2.assemble(spec, -dphi, 0.0, 0.0, self._nominal6, c, layout=lay, reflexes=False)
            t2 = gait_v2.assemble(spec, -2 * dphi, 0.0, 0.0, self._nominal6, c, layout=lay, reflexes=False)
            self._prev_cmd_pos[:] = np.clip(t1, self.ctrl_lo, self.ctrl_hi)
            self._prev_cmd_vel[:] = (t1 - t2) / self.control_dt

    def _thermal_step(self):
        """One control tick of the single-node winding model per motor (artifact §07):
        tau_th * dtheta/dt = (tau_rms / tau_cont)^2 / c_max - theta, theta = dT / dT_max."""
        c = self.cfg
        n = max(int(self._sub_n), 1)
        q = (self._tau_sq_acc / n) / (self._tau_cont ** 2) / max(self._theta_cmax, 1e-6)
        self._theta += (self.control_dt / c.thermal_tau_s) * (q - self._theta)
        self._tau_sq_acc[:] = 0.0

    def _apply_wind(self):
        """Constant body-frame force (steady wind) + gust steps, on the base (artifact §08)."""
        if not self._wind_on:
            return
        c = self.cfg
        adv = self._dr.scale if c.adversity_curriculum else 1.0
        f_body = self._wind_f.copy()
        if self._gust_left > 0:
            f_body += self._gust_vec
            self._gust_left -= 1
        else:
            self._gust_countdown -= 1
            if self._gust_countdown <= 0 and c.gust_force_n > 0.0:
                axis = int(self.np_random.integers(0, 2))
                sign = 1.0 if self.np_random.random() < 0.5 else -1.0
                self._gust_vec[:] = 0.0
                self._gust_vec[axis] = sign * c.gust_force_n
                self._gust_left = max(1, int(round(c.gust_duration_s / self.control_dt)))
                self._gust_countdown = self._next_gust_in()
                f_body += self._gust_vec
        if not np.any(f_body):
            return
        R = self._base_rot()
        self.data.xfrc_applied[self.base_id, 0:3] += adv * (R @ np.array([f_body[0], f_body[1], 0.0]))

    def _contact_normal_forces(self):
        """Per-foot normal ground force (N), toe + heel, from the constraint solver."""
        out = np.zeros(2)
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = (con.geom1, con.geom2)
            if self.floor_gid not in pair:
                continue
            g = pair[1] if pair[0] == self.floor_gid else pair[0]
            if g not in self._col_gids:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, f6)
            side = self._col_gids_side[g]
            out[side] += abs(float(f6[0]))
        return out

    # ---- observation pieces ----------------------------------------------------------------
    def _once_block(self):
        c = self.cfg
        lay = self._lay
        commit = np.array([1.0 if self._commit_flag else 0.0], dtype=np.float64)
        if self.spec_source == "policy":
            # the LIVE latched spec: what is driving the legs, not what the policy last emitted
            return np.concatenate([self._spec_live, self._task, commit])
        s = c.obs_scales
        spec = self._spec_live
        phi = float(self._phase)
        f = gait_v2.frequency(spec[lay.freq], c.gait_freq_hz)
        nom = np.asarray(self._nominal6, dtype=float)
        q0 = gait_v2.assemble(spec, phi, 0.0, 0.0, nom, c, layout=lay, reflexes=False)
        eps = 1e-3
        qp = gait_v2.assemble(spec, phi + eps, 0.0, 0.0, nom, c, layout=lay, reflexes=False)
        qm = gait_v2.assemble(spec, phi - eps, 0.0, 0.0, nom, c, layout=lay, reflexes=False)
        qdot = (qp - qm) / (2.0 * eps) * (2.0 * np.pi * f)
        q4 = gait_v2.assemble(spec, phi + 0.5 * np.pi, 0.0, 0.0, nom, c, layout=lay, reflexes=False)
        td = np.array([(self._td_hat[0] + np.pi) % (2 * np.pi) - np.pi,
                       (self._td_hat[1] - np.pi + np.pi) % (2 * np.pi) - np.pi]) / np.pi
        return np.concatenate([(q0 - nom) * s["motor_pos"], qdot * s["motor_vel"],
                               (q4 - nom) * s["motor_pos"], td, self._task, commit])

    def _priv_tail_v2(self):
        c = self.cfg
        s = c.obs_scales
        d = self._ep_draw
        tail = np.empty(self.PRIV_DIM_V2)
        tail[0:3] = self._vel_body() * s["base_vel"]
        tail[3:5] = (self._foot_contacts() | (self._toe_heights() < c.grounded_h)).astype(float)
        tail[5] = float(self.data.qpos[2]) - c.term_height
        tail[6] = float(self.data.qpos[1])
        tail[7] = float(self.data.qpos[5])
        tail[8:11] = self._accel_body_last / 9.81
        tail[11:13] = self._contact_normal_forces() / max(self._weight_n, 1e-6)
        tail[13] = float(d.get("mass_scale", 1.0))
        tail[14] = 10.0 * float(d.get("com_x", 0.0))
        tail[15] = float(d.get("friction", 1.0))
        tail[16] = float(d.get("kp_scale", 1.0))
        tail[17] = float(d.get("torque_scale", 1.0))
        tail[18] = self._delay_ms / 10.0
        tail[19:25] = self._theta[:6]
        return tail

    def mirror_perm_sign(self):
        """(perm, sign) of the L/R mirror on the FULL observation, obs_m = sign * obs[perm]:
        swap+negate every joint block (the FK sign rule), negate y-ish quantities (grav_y,
        gyro_x/z, LP yaw, y, heading, vy, ay), shift the phase by pi, negate the five knobs and
        the reflex bias, leave S / profiles / frequency / task / commit alone. Used by the
        equivariance loss (sym_ppo.py); an involution."""
        nu = self.nu
        fp, fs = [], []
        swap = list(gait_v2.MIRROR_PERM6)
        for _ in range(3):                                  # pos, vel, torque
            fp += [len(fp) - len(fp) % nu + i for i in swap]
            fs += [-1.0] * nu
        base = len(fp)
        fp += [base, base + 1, base + 2]; fs += [1.0, -1.0, 1.0]            # gravity
        base = len(fp)
        fp += [base, base + 1, base + 2]; fs += [-1.0, 1.0, -1.0]           # gyro
        fp += [len(fp)]; fs += [-1.0]                                       # LP yaw
        base = len(fp)
        fp += [base, base + 1]; fs += [-1.0, -1.0]                          # (sin, cos) -> phi + pi
        base = len(fp)
        fp += [base + i for i in swap]; fs += [-1.0] * 6                    # previous residual
        assert len(fp) == self.frame_dim, (len(fp), self.frame_dim)
        perm, sign = [], []
        H = self.cfg.history_len
        for k in range(H):
            perm += [k * self.frame_dim + i for i in fp]
            sign += fs
        base = H * self.frame_dim
        lay = self._lay
        if self.spec_source == "policy":
            op = list(range(lay.spec_dim))
            os_ = [1.0] * lay.spec_dim
            for i in lay.knobs:
                os_[i] = -1.0
            os_[lay.reflex.start + 2] = -1.0
        else:
            op, os_ = [], []
            for _ in range(3):
                b = len(op)
                op += [b + i for i in swap]; os_ += [-1.0] * 6
            b = len(op)
            op += [b + 1, b]; os_ += [1.0, 1.0]                             # td_hat L<->R
        perm += [base + i for i in op]; sign += os_
        base = len(perm)
        perm += list(range(base, base + self.task_dim + 1)); sign += [1.0] * (self.task_dim + 1)
        if self.priv_dim:
            base = len(perm)
            tp = [0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 12, 11, 13, 14, 15, 16, 17, 18] + [19 + i for i in swap]
            ts = [1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [1.0] * 6
            perm += [base + i for i in tp]; sign += [float(x) for x in ts]
        perm = np.asarray(perm, dtype=np.int64)
        sign = np.asarray(sign, dtype=np.float32)
        assert perm.size == self.observation_space.shape[0], (perm.size, self.observation_space.shape)
        return perm, sign

    # ---- library variant (§09) -------------------------------------------------------------
    @staticmethod
    def default_library(cfg):
        """A placeholder library until library/library_solve.py writes a real one: a symmetric
        3 Hz sinusoidal gait (cam-led, thigh a quarter cycle behind for lift) and the stand."""
        lay = gait_v2.Layout(cfg.n_harmonics)
        run = gait_v2.neutral_spec(lay, freq_raw=gait_v2.freq_raw_of(3.0, cfg.gait_freq_hz))
        run[lay.s_cam.start + 1] = 0.6           # cam a1 (cos)
        run[lay.s_thigh.start + 2] = 0.35        # thigh b1 (sin)
        stand = gait_v2.neutral_spec(lay, freq_raw=gait_v2.freq_raw_of(2.0, cfg.gait_freq_hz))
        return [dict(v=2.0, theta=run.tolist(), x_star=None, lambda_max=None, source="placeholder"),
                dict(v=0.0, theta=stand.tolist(), x_star=None, lambda_max=None, source="placeholder")]

    def _load_library(self, path):
        import json
        if not path:
            entries = self.default_library(self.cfg)
        else:
            d = json.loads(Path(_resolve(path)).read_text())
            entries = d["entries"] if isinstance(d, dict) else d
        out = []
        for e in entries:
            th = np.clip(np.asarray(e["theta"], dtype=float), -1.0, 1.0)
            if th.size != self._lay.spec_dim:
                raise ValueError(f"library entry has {th.size} spec dims, need {self._lay.spec_dim}")
            xs = e.get("x_star")
            out.append(dict(v=float(e.get("v", 0.0)), theta=th,
                            x_star=None if xs is None else np.asarray(xs, dtype=float)))
        if not out:
            raise ValueError("empty gait library")
        return out

    def set_library_entry(self, theta, v=None, x_star=None):
        """Override the library with ONE running entry (the search / absorption tools). Applies
        from the next reset; None restores the file/default library."""
        if theta is None:
            self._lib_override = None
            return
        self._lib_override = dict(v=float(v if v is not None else 1.0),
                                  theta=np.clip(np.asarray(theta, dtype=float), -1.0, 1.0),
                                  x_star=None if x_star is None else np.asarray(x_star, dtype=float))

    def _lib_pick(self):
        if self._lib_override is not None:
            return self._lib_override
        c = self.cfg
        run = [e for e in self._lib_entries if e["v"] > 0.0] or self._lib_entries
        if c.library_entry == "random":
            return run[int(self.np_random.integers(0, len(run)))]
        if c.library_entry == "fastest":
            return max(run, key=lambda e: e["v"])
        v = float(c.library_entry)
        return min(run, key=lambda e: abs(e["v"] - v))

    def _lib_reset(self):
        c = self.cfg
        lay = self._lay
        rng = self.np_random
        entry = self._lib_pick()
        th = entry["theta"].copy()
        # the per-episode box around the entry: amplitudes, frequency, knobs
        if c.library_box_amp > 0.0:
            for sl in (lay.s_cam, lay.s_thigh, lay.s_hip):
                th[sl] *= float(rng.uniform(1.0 - c.library_box_amp, 1.0 + c.library_box_amp))
        if c.library_box_f_hz > 0.0:
            f = gait_v2.frequency(th[lay.freq], c.gait_freq_hz) + float(
                rng.uniform(-c.library_box_f_hz, c.library_box_f_hz))
            th[lay.freq] = gait_v2.freq_raw_of(f, c.gait_freq_hz)
        if c.library_box_knob > 0.0:
            th[lay.knobs] += rng.uniform(-c.library_box_knob, c.library_box_knob, len(lay.knobs))
        self._theta_ep = np.clip(th, -1.0, 1.0)
        self._lib_v_ref = float(entry["v"])
        stand = [e for e in self._lib_entries if e["v"] == 0.0]
        if stand:
            self._lib_stand = stand[0]["theta"].copy()
        else:                                    # stopping is an amplitude decision: S -> 0
            self._lib_stand = self._theta_ep.copy()
            for sl in (lay.s_cam, lay.s_thigh, lay.s_hip):
                self._lib_stand[sl] = 0.0
        self._spec_live[:] = self._theta_ep
        if self._raibert is not None:
            self._raibert.reset()
        # start ON the orbit when the entry carries its section state (Stage-3 episodes)
        if c.library_reset_on_orbit and entry.get("x_star") is not None \
                and not self.z_locked and self._fixed_base_h is None:
            self.set_section_state(entry["x_star"], self._theta_ep, keep_xy=True)
            n = c.reset_joint_noise
            self.data.qpos[self._noise_qadr] += rng.uniform(-n, n, self._noise_qadr.size)
            mujoco.mj_forward(self.model, self.data)

    def _lib_commit(self, mods):
        """Per-cycle spec for the library variant: the episode's theta (or the stand entry in
        the stop phase), the optional latched mods, then the Raibert prior on o_cam / o_hip."""
        c = self.cfg
        lay = self._lay
        run_phase = c.objective == "speed" or not self._sprint_crossed
        spec = (self._theta_ep if run_phase else self._lib_stand).copy()
        v_ref = self._lib_v_ref if run_phase else 0.0
        mods = np.asarray(mods, dtype=float)
        if mods.size >= 3:
            sc = c.library_latched_scale
            f = gait_v2.frequency(spec[lay.freq], c.gait_freq_hz) * (1.0 + sc[0] * mods[0])
            spec[lay.freq] = gait_v2.freq_raw_of(f, c.gait_freq_hz)
            amp = 1.0 + sc[1] * mods[1]
            for sl in (lay.s_cam, lay.s_thigh, lay.s_hip):
                spec[sl] *= amp
            lift = sc[2] * mods[2]
            spec[lay.offset.start] += lift
            spec[lay.offset.start + 1] += lift
        if self._raibert is not None:
            v_hat = self._vel_body().copy()
            if c.raibert_v_noise > 0.0:
                v_hat += self.np_random.normal(0.0, c.raibert_v_noise, 3)
            T = 1.0 / max(gait_v2.frequency(spec[lay.freq], c.gait_freq_hz), 1e-6)
            spec[lay.offset] = self._raibert.commit(spec[lay.offset], v_hat, v_ref, T)
        self._spec_live[:] = np.clip(spec, -1.0, 1.0)

    # ---- section / return-map hooks (library/fixed_point.py) -------------------------------
    def section_dims(self):
        """qpos indices kept in the section state (all but the cyclic x, y, yaw) + every qvel."""
        keep = [i for i in range(self.model.nq) if i not in (0, 1, 5)]
        return np.asarray(keep, dtype=int), self.model.nv

    def section_state(self):
        keep, _ = self.section_dims()
        return np.concatenate([self.data.qpos[keep], self.data.qvel])

    def set_section_state(self, x, spec, keep_xy=False):
        """Put the plant at section state x (phi = 0+, start of a cycle) under `spec`, with the
        command ring and slew memory primed as if the previous cycle of the same gait had just
        ended. keep_xy leaves the base x/y/yaw where they are (episode resets)."""
        keep, nv = self.section_dims()
        x = np.asarray(x, dtype=float)
        if not keep_xy:
            self.data.qpos[[0, 1, 5]] = 0.0
        self.data.qpos[keep] = x[:len(keep)]
        self.data.qvel[:] = x[len(keep):len(keep) + nv]
        self._spec_live[:] = np.clip(np.asarray(spec, dtype=float), -1.0, 1.0)
        self._phase = 0.0
        self._commit_flag = True
        self._resync_done[:] = False
        self.prime_command_ring(self._spec_live)
        # controller / solver memory that is not part of x: the reflex rate filter, the thermal
        # accumulator, and MuJoCo's constraint warm-start (a history-dependent initial guess that
        # would make P(x) differ at the 1e-6 level between two calls from the same x)
        self._reflex_prate_filt = 0.0
        self._tau_sq_acc[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._grounded_prev = self._foot_contacts() | (self._toe_heights() < self.cfg.grounded_h)
        self._prev_toe_xy = self.data.geom_xpos[self.foot_gids_arr, 0:2].copy()

    def run_cycle(self, action=None, record=None):
        """Step until the clock wraps once (P(x; theta)); returns the section state after it.
        `record(env)` is called after every tick (envelope flags)."""
        a = np.zeros(self.action_dim, np.float32) if action is None else action
        n_max = int(4.0 / (max(self._f_hz, 0.5) * self.control_dt)) + 8
        for _ in range(n_max):
            self.step(a)
            if record is not None:
                record(self)
            if self._commit_flag:
                break
        return self.section_state()

    # ---- eval / margin-protocol setters (tools/eval_envelope.py) ----------------------------
    def set_delay_ms(self, ms):
        """Fix the drive transport delay (ms) for every following episode; None = the draw."""
        self._delay_override_ms = None if ms is None else float(ms)

    def set_wind(self, fx, fy):
        """Fix the constant body-frame force (N) for every following episode; None = the draw."""
        self._wind_override = None if fx is None else np.array([float(fx), float(fy)])
        if self._wind_override is not None:
            self._wind_on = True

    def set_thermal_hot(self, theta0):
        """Fix the hot-start winding state (fraction of dT_max, all motors); None = the draw."""
        self._theta_override = None if theta0 is None else float(theta0)

    def set_push_axis(self, axis):
        """Restrict pushes to one body axis ("x"/"y", random sign) for the margin protocol."""
        self._push_axis = None if axis is None else str(axis)

    def set_resync_kappa(self, kappa):
        self.cfg.resync_kappa = float(kappa)
        self.cfg.dr_resync_kappa_range = (0.0, 0.0)
        self._kappa = float(kappa)

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
