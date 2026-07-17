"""Compile a framework.schema.Experiment onto the trainer's rl.config.Config.

This is the capability gate: the schema can EXPRESS more than the runtime can RUN
(per-joint torque/PID mixing, soft-limit DOFs, tunable L/R phase, ...). Everything the
current Dash01Env supports maps here; everything else raises CompileError with an
explicit "not supported by the runtime yet" message instead of silently degrading.

Contract (enforced by tests/test_preset_parity.py): every preset in rl/config.py,
re-encoded as an experiment under experiments/presets/, compiles to a Config that is
FIELD-IDENTICAL to get_config(<preset>). The compiler starts from Config() defaults and
only overwrites mapped fields, so inert defaults can never drift.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from rl.config import Config

from . import loader
from .schema import (
    Axis, AXIS_ORDER, Bind, CommandMode, ControllerKind, Device, DofMode,
    Drives, Experiment, PatternGen, PatternUpdate, PhaseMode, PositionPD, Reflex,
    ReflexInput, SafetyEvent,
)


class CompileError(ValueError):
    """The experiment is valid per the schema but not runnable by the current runtime."""


# Config fields a reward module may configure through reward.params. These are the
# primary sweep axes. (fall_penalty lives in safety; sprint geometry in command.sprint.)
REWARD_PARAMS = {
    "penalty_term_cap",
    "w_track_vx", "w_track_yaw", "w_progress", "w_yaw_rate", "w_heading_pen", "w_pos_l1",
    "w_upright", "w_height", "w_vz", "w_action_rate", "w_torque", "w_alive",
    "w_fwd_speed", "v_ceiling",
    "w_time", "w_stop_vel", "stop_sigma", "w_overrun", "finish_bonus",
    "w_foot_slip", "slip_deadband",
    "w_stance_time", "stance_cap_s", "stance_cap_slow_s", "stance_slow_speed",
    "w_clearance", "clearance_dead_m", "clearance_scale_m", "swing_fresh_s", "gait_cmd_gate",
    "w_air_time", "foot_air_time_min", "air_credit_cap_s", "w_coef_rate",
    "w_no_cross", "stance_min_sep", "w_hip_roll",
    "w_track_pos", "track_sigma_pos", "w_track_heading", "track_sigma_heading",
    "w_lat_vel", "w_angvel_xy", "track_sigma_vx", "track_sigma_yaw",
}

# reward modules that ARE the built-in env reward (no module injection needed)
BUILTIN_REWARDS = {"gait_speed_v3", "_lib/rewards/gait_speed_v3.py"}
BUILTIN_OBS = {"standard", "_lib/obs/standard.py"}

# the one fourier controller shape the runtime implements (rl/fourier_gait.py):
# left cam+thigh pattern generators + a left hip-roll reflex, mirrored antiphase.
FOURIER_JOINTS = {"cam_L", "thigh_L", "hip_roll_L"}
FOURIER_MIRROR = {"cam_R": "cam_L", "thigh_R": "thigh_L", "hip_roll_R": "hip_roll_L"}


@dataclass
class Compiled:
    """The trainer inputs derived from an Experiment."""
    config: Config
    name: str
    description: str
    device: str                       # "cpu" | "gpu"
    n_envs_spec: object               # int or "auto" (resolved by the orchestrator)
    warm_start: str | None            # run id / checkpoint ref (resolved by the caller)
    sim2sim_gate: str | None
    meta: dict = field(default_factory=dict)


def _local(module: str) -> bool:
    return module.startswith("./") or module.startswith(".\\")


def compile_experiment(exp: Experiment, exp_dir: str | None = None) -> Compiled:
    errors: list[str] = []
    cfg = Config()
    exp_dir = loader.experiment_dir(exp) if exp_dir is None else exp_dir

    def gate(msg: str):
        errors.append(msg)

    # ---- ① plant --------------------------------------------------------------
    cfg.model_path = exp.plant.model
    cfg.keyframe = exp.plant.keyframe
    if exp.plant.control_hz != 50:
        gate(f"plant.control_hz={exp.plant.control_hz}: runtime is fixed at 50 Hz "
             "(1 kHz sim / decimation 20) for now")

    # ---- ④ base DOF -----------------------------------------------------------
    lock = [0, 0, 0, 0, 0, 0]
    for i, axis in enumerate(AXIS_ORDER):
        dof = exp.base_dof.get(axis)
        if dof is None or dof.mode is DofMode.FREE:
            continue
        if dof.mode is DofMode.LOCK:
            lock[i] = 1
            if dof.value != 0.0:
                gate(f"base_dof.{axis.value}: lock at a custom value ({dof.value}) is not "
                     "supported — locks pin the reset pose (0 / natural stance height)")
        elif dof.mode is DofMode.RAIL:
            lock[i] = 1
            if axis is Axis.Z:
                cfg.z_rail_randomize = True
                cfg.z_rail_range = tuple(dof.range)
            else:
                gate(f"base_dof.{axis.value}: rail randomization is only implemented for Z "
                     "(the M1 ride-height rail)")
        else:
            gate(f"base_dof.{axis.value}: mode '{dof.mode.value}' not supported by the runtime yet")
    cfg.base_lock = tuple(lock)

    # ---- ② + ③ controllers & symmetry ------------------------------------------
    if not exp.joints:
        pass                                        # default: 6-joint PD ("pd" action mode)
    elif all(isinstance(c, PositionPD) for c in exp.joints.values()):
        first = next(iter(exp.joints.values()))
        if any((c.action_scale, c.filter, c.delay_steps) !=
               (first.action_scale, first.filter, first.delay_steps) for c in exp.joints.values()):
            gate("joints: per-joint PD parameters differ — the runtime applies one global "
                 "action_scale/filter/delay")
        cfg.action_scale = first.action_scale
        cfg.action_filter = first.filter
        cfg.action_delay_steps = first.delay_steps
        if exp.symmetry:
            gate("symmetry groups are not implemented for the pd action mode")
    elif set(exp.joints) == FOURIER_JOINTS:
        cam, thigh = exp.joints["cam_L"], exp.joints["thigh_L"]
        reflex = exp.joints["hip_roll_L"]
        if not (isinstance(cam, PatternGen) and isinstance(thigh, PatternGen)
                and isinstance(reflex, Reflex)):
            gate("joints: the fourier shape needs pattern_gen on cam_L/thigh_L and a reflex "
                 "on hip_roll_L")
        else:
            if cam.n_harmonics != thigh.n_harmonics:
                gate("joints: cam_L and thigh_L must share n_harmonics (one gait spectrum)")
            if tuple(cam.freq_hz) != tuple(thigh.freq_hz):
                gate("joints: cam_L and thigh_L must share freq_hz (one cadence)")
            if cam.drives is not Drives.POSITION or thigh.drives is not Drives.POSITION:
                gate("joints: pattern_gen drives=torque is not supported (PD position targets only)")
            if cam.update is not thigh.update:
                gate("joints: cam_L and thigh_L must share update (one gait-spec rewrite cadence)")
            # per_cycle -> the macro-step "fourier" mode; per_step -> "fourier_step" (the policy
            # re-emits the coefficients at every 50 Hz control step; phase-gated w_coef_rate penalty)
            cfg.action_mode = ("fourier" if cam.update is PatternUpdate.PER_CYCLE
                               else "fourier_step")
            cfg.n_harmonics = cam.n_harmonics
            cfg.gait_freq_hz = tuple(cam.freq_hz)
            cfg.cam_amp = cam.amp
            cfg.thigh_amp = thigh.amp
            if reflex.inputs != [ReflexInput.ROLL, ReflexInput.ROLL_RATE]:
                gate("hip_roll_L reflex: runtime implements inputs=[roll, roll_rate] exactly")
            if reflex.gains != "learned":
                gate("hip_roll_L reflex: fixed gains are not implemented (gains=learned only)")
            cfg.reflex_kp_scale = reflex.kp_scale
            cfg.reflex_kd_scale = reflex.kd_scale
            cfg.reflex_bias_scale = reflex.bias_scale
        pairs = list(exp.symmetry.values())
        if len(pairs) != 1 or pairs[0].mirror != FOURIER_MIRROR or pairs[0].sign != -1:
            gate("symmetry: the fourier runtime implements exactly the L->R mirror "
                 f"{FOURIER_MIRROR} with sign=-1")
        elif pairs[0].phase.mode is not PhaseMode.ANTIPHASE or pairs[0].phase.tune is not Bind.FIXED:
            gate("symmetry.phase: only mode=antiphase with tune=fixed is implemented "
                 "(the network-tuned L/R phase delta is future work in rl/fourier_gait.py)")
    else:
        gate(f"joints: unsupported controller set {sorted(exp.joints)} — the runtime "
             "implements all-PD (empty joints block) or the cam/thigh/hip_roll fourier shape")

    # ---- ⑤ observation ----------------------------------------------------------
    if exp.observation.module in BUILTIN_OBS:
        cfg.obs_module = ""
    elif _local(exp.observation.module):
        cfg.obs_module = exp.observation.module
    else:
        gate(f"observation.module '{exp.observation.module}': unknown library module "
             "(library: 'standard'; local: './observation.py')")
    cfg.history_len = exp.observation.history_len
    cfg.obs_scales = dict(exp.observation.scales)

    # ---- ⑥ command / task --------------------------------------------------------
    cmd = exp.command
    cfg.vx_max = cmd.vx_max
    cfg.yaw_max = cmd.yaw_max
    cfg.p_stand = cmd.p_stand
    cfg.cmd_forward_only = cmd.forward_only
    cfg.cmd_resample_s = cmd.resample_s
    cfg.episode_s = cmd.episode_s
    if cmd.mode is CommandMode.JOYSTICK:
        cfg.cmd_vx_frac = cmd.vx_frac
        cfg.cmd_vx_min_frac = cmd.vx_min_frac
        cfg.cmd_yaw_frac = cmd.yaw_frac
    elif cmd.mode in (CommandMode.SPEED, CommandMode.SPRINT):
        # the max-speed convention (rl/config._speed): command pinned at the ceiling
        if not cmd.forward_only:
            gate(f"command.mode={cmd.mode.value} requires forward_only=true")
        cfg.speed_mode = True
        cfg.cmd_vx_frac = 1.0
        cfg.cmd_vx_min_frac = 1.0
        cfg.cmd_yaw_frac = 0.0
        if cmd.mode is CommandMode.SPRINT:
            sp = cmd.sprint
            # NOTE speed_mode stays True: the presets build sprint via _speed(**_SPRINT)
            # and the env checks sprint_mode first, so the flag is inert but set.
            cfg.sprint_mode = True
            cfg.sprint_dist_m = sp.dist_m
            cfg.sprint_dist_start_m = sp.dist_start_m
            cfg.sprint_curriculum_steps = sp.curriculum_steps
            cfg.sprint_brake_m = sp.brake_m
            cfg.stop_speed_eps = sp.stop_speed_eps
            cfg.stop_hold_s = sp.stop_hold_s
            cfg.curriculum_steps = 0        # guard: the cmd ramp must not clobber curriculum.json
    else:
        gate("command.mode=none is not supported for DASH-01 (pendulum-only; future work)")

    # ---- ⑦ reward -----------------------------------------------------------------
    if exp.reward.module in BUILTIN_REWARDS:
        cfg.reward_module = ""
    elif _local(exp.reward.module):
        cfg.reward_module = exp.reward.module
    else:
        gate(f"reward.module '{exp.reward.module}': unknown library reward "
             "(library: 'gait_speed_v3'; local: './reward.py')")
    for k, v in exp.reward.params.items():
        if k not in REWARD_PARAMS:
            gate(f"reward.params.{k}: not a known reward parameter (see framework.compile.REWARD_PARAMS)")
        else:
            setattr(cfg, k, type(getattr(cfg, k))(v))

    # ---- ⑧ curriculum ---------------------------------------------------------------
    if exp.curriculum is not None:
        cur = exp.curriculum
        if cur.module == "cmd_ramp":
            allowed = {"start", "steps"}
            if set(cur.params) - allowed:
                gate(f"curriculum cmd_ramp: unknown params {sorted(set(cur.params) - allowed)} "
                     "(takes start, steps; target is command.vx_frac)")
            cfg.cmd_vx_frac_start = float(cur.params.get("start", 0.0))
            cfg.curriculum_steps = int(cur.params.get("steps", cfg.curriculum_steps))
        elif _local(cur.module):
            cfg.curriculum_module = cur.module
        else:
            gate(f"curriculum.module '{cur.module}': unknown library curriculum "
                 "(library: 'cmd_ramp'; local: './curriculum.py')")

    # ---- ⑨ network / PPO --------------------------------------------------------------
    if exp.policy.activation != "tanh":
        gate(f"policy.activation={exp.policy.activation}: the trainer builds tanh MLPs "
             "(use policy.module for a custom architecture — also not wired yet)")
    cfg.policy_hidden = list(exp.policy.hidden)
    if exp.policy.module is not None:
        if _local(exp.policy.module):
            cfg.network_module = exp.policy.module
        else:
            gate(f"policy.module '{exp.policy.module}': only local './network.py' is supported")
    p = exp.ppo
    cfg.total_steps = p.total_steps
    cfg.n_steps = p.n_steps
    cfg.batch_size = p.batch_size
    cfg.n_epochs = p.n_epochs
    cfg.gamma = p.gamma
    cfg.gae_lambda = p.gae_lambda
    cfg.learning_rate = p.learning_rate
    cfg.lr_final = p.lr_final
    cfg.clip_range = p.clip_range
    cfg.target_kl = p.target_kl
    cfg.ent_coef = p.entropy.coef
    cfg.ent_final = p.entropy.final
    cfg.ent_anneal_steps = p.entropy.anneal_steps
    cfg.ent_gate_air_time = p.entropy.gate_air_time
    cfg.max_log_std = p.entropy.max_log_std
    cfg.seed = p.seed

    # ---- ⑩ backend ----------------------------------------------------------------------
    if isinstance(exp.backend.n_envs, int):
        cfg.n_envs = exp.backend.n_envs

    # ---- ⑪ run / safety / dr / steering ---------------------------------------------------
    if exp.run.checkpoint_every != 200_000:
        gate(f"run.checkpoint_every={exp.run.checkpoint_every}: the trainer checkpoints every "
             "200k env-steps (making this configurable is queued work)")
    if set(exp.run.outputs) != {"csv", "plots", "tensorboard"}:
        gate("run.outputs: the trainer always writes csv+plots+tensorboard (subsetting is queued work)")

    s = exp.safety
    if set(s.terminate_on) != {SafetyEvent.FALL, SafetyEvent.TILT_60DEG, SafetyEvent.FLOOR_BLOWUP}:
        gate("safety.terminate_on: the runtime hardcodes {fall, tilt_60deg, floor_blowup}")
    cfg.term_height = s.term_height
    cfg.term_gravity_z = s.term_gravity_z
    cfg.grounded_h = s.grounded_h
    cfg.fall_penalty = s.fall_penalty
    cfg.reset_joint_noise = s.reset_joint_noise
    cfg.push_interval_s = s.push_interval_s
    cfg.push_dv = s.push_dv

    d = exp.dr
    cfg.dr_enabled = d.enabled
    cfg.dr_mass = d.mass
    cfg.dr_friction = tuple(d.friction)
    cfg.dr_motor_strength = d.motor_strength
    cfg.dr_pd_gain = d.pd_gain
    cfg.dr_latency_steps = d.latency_steps
    cfg.dr_imu_gyro_noise = d.imu_gyro_noise
    cfg.dr_motor_pos_noise = d.motor_pos_noise
    cfg.dr_push_interval_s = d.push_interval_s

    if exp.steering is not None:
        if _local(exp.steering):
            cfg.steering_module = exp.steering
        else:
            gate(f"steering '{exp.steering}': only local './steering.py' is supported")

    if any(_local(m) for m in (cfg.reward_module, cfg.obs_module,
                               cfg.curriculum_module, cfg.network_module, cfg.steering_module)):
        if not exp_dir:
            gate("experiment uses local ./modules but was not loaded from a folder "
                 "(no directory to resolve them against)")
        cfg.experiment_dir = exp_dir

    if errors:
        raise CompileError(
            f"experiment '{exp.name}' is valid but not runnable by the current runtime:\n  - "
            + "\n  - ".join(errors))

    return Compiled(
        config=cfg,
        name=exp.name,
        description=exp.description,
        device=exp.backend.device.value,
        n_envs_spec=exp.backend.n_envs,
        warm_start=exp.run.warm_start,
        sim2sim_gate=exp.safety.sim2sim_gate,
        meta={"deploy_target": exp.deploy.target.value},
    )
