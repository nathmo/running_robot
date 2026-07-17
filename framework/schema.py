"""Typed experiment schema (pydantic v2) — the single source of truth.

v1.1 — the completed version of the experiments/_lib/schema.py draft. Every categorical
is an Enum and every structured choice is a typed model, so the legal set of any field
is DISCOVERABLE without reading a loader or a comment:

    >>> list(ControllerKind)                 # every controller you can pick
    >>> Phase.model_json_schema()            # the phase grammar, machine-readable
    >>> Experiment.model_json_schema()       # the whole format, as JSON Schema

Changes vs the draft (see the pipeline plan):
  * `description` is a REQUIRED field — every experiment states why it exists.
  * `reward` / `curriculum` are typed specs `{module, params}` (plain-string shorthand
    still accepted); reward params are how the ~40 w_* weights are configured/swept.
  * `Command` carries the sampling fields (vx_frac, vx_min_frac, yaw_frac, resample_s),
    the episode length, and the sprint block (Command.mode == SPRINT).
  * `Safety` carries the live termination/perturbation knobs; `dr` is the domain-
    randomization block; `steering` (eval-only command script) is a legal field.
  * YAML shorthands (positional base_dof list, `free` / `{lock: 0}` / `{rail: [a,b]}`,
    bare reward strings) are normalized by before-validators, so the authoring forms in
    experiments/README.md actually validate.

The runtime split: this file defines WHAT an experiment is; framework/compile.py maps
it onto the trainer (rl.config.Config) and is where "supported by the runtime today"
is enforced (capability gating). Keep semantics out of this file.

Requires: pydantic>=2.
"""
from __future__ import annotations
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enums — the closed vocabularies. "What else besides X?" == list(TheEnum).
# =============================================================================
class Axis(str, Enum):
    X = "X"; Y = "Y"; Z = "Z"; ROLL = "roll"; PITCH = "pitch"; YAW = "yaw"


AXIS_ORDER = (Axis.X, Axis.Y, Axis.Z, Axis.ROLL, Axis.PITCH, Axis.YAW)


class ControllerKind(str, Enum):
    LOCKED = "locked"; SPRING_DAMPER = "spring_damper"; POSITION = "position"
    PID = "pid"; TORQUE = "torque"; PATTERN_GEN = "pattern_gen"; REFLEX = "reflex"


class Drives(str, Enum):            # what a controller's output actuates
    POSITION = "position"; TORQUE = "torque"


class PatternUpdate(str, Enum):    # when the policy may rewrite a pattern_gen's coefficients
    PER_CYCLE = "per_cycle"        # latched for a whole gait cycle (macro-step)
    PER_STEP = "per_step"          # re-emitted every 50 Hz control step (instant override)


class Bind(str, Enum):             # where a tunable value comes from
    FIXED = "fixed"; ACTION = "action"; OBS = "obs"


class PhaseMode(str, Enum):        # <-- the fix for "pi + delta"
    INPHASE = "inphase"; ANTIPHASE = "antiphase"; OFFSET = "offset"


class DofMode(str, Enum):
    FREE = "free"; LOCK = "lock"; RAIL = "rail"; SOFT_LIMIT = "soft_limit"


class ReflexInput(str, Enum):
    ROLL = "roll"; ROLL_RATE = "roll_rate"; PITCH = "pitch"; PITCH_RATE = "pitch_rate"


class CommandMode(str, Enum):
    NONE = "none"; JOYSTICK = "joystick"; SPEED = "speed"; SPRINT = "sprint"


class Device(str, Enum):
    CPU = "cpu"; GPU = "gpu"          # gpu == MJX (does not support pattern_gen/sprint)


class DeployTarget(str, Enum):
    MOTEUS = "moteus"; SIM = "sim"


class SafetyEvent(str, Enum):
    FALL = "fall"; TILT_60DEG = "tilt_60deg"; FLOOR_BLOWUP = "floor_blowup"
    ENVELOPE_EXIT = "envelope_exit"   # left the recorded safe space (runtime watchdog)


# =============================================================================
# ② + ③ controllers (discriminated union on `controller`)
# =============================================================================
class PatternGen(BaseModel):
    controller: Literal[ControllerKind.PATTERN_GEN] = ControllerKind.PATTERN_GEN
    n_harmonics: int = Field(3, ge=1, le=8)          # "number of weights"
    amp: float = Field(..., gt=0)                    # rad deviation from nominal (required)
    freq_hz: tuple[float, float] = (0.5, 3.0)        # learnable cadence band
    drives: Drives = Drives.POSITION
    update: PatternUpdate = PatternUpdate.PER_CYCLE  # coefficient rewrite cadence


class Reflex(BaseModel):
    controller: Literal[ControllerKind.REFLEX] = ControllerKind.REFLEX
    inputs: list[ReflexInput]
    gains: Literal["learned", "fixed"] = "learned"
    kp_scale: float = 0.5
    kd_scale: float = 0.1
    bias_scale: float = 0.2


class PositionPD(BaseModel):
    controller: Literal[ControllerKind.POSITION] = ControllerKind.POSITION
    action_scale: float = 0.5        # rad about nominal
    filter: float = Field(0.2, ge=0, le=1)
    delay_steps: int = 1


class Torque(BaseModel):
    controller: Literal[ControllerKind.TORQUE] = ControllerKind.TORQUE
    scale_nm: float = Field(..., gt=0)
    cap_nm: float = Field(..., gt=0)


class PID(BaseModel):
    controller: Literal[ControllerKind.PID] = ControllerKind.PID
    kp: float; ki: float = 0.0; kd: float = 0.0
    integral_clamp: float = 1.0
    torque_cap_nm: float = Field(..., gt=0)
    gravity_ff: bool = False


class SpringDamper(BaseModel):
    controller: Literal[ControllerKind.SPRING_DAMPER] = ControllerKind.SPRING_DAMPER
    stiffness: float; damping: float; friction: float = 0.0
    rest_rad: float = 0.0; breakaway_nm: float = 0.0


class Locked(BaseModel):
    controller: Literal[ControllerKind.LOCKED] = ControllerKind.LOCKED


Controller = Annotated[
    Union[PatternGen, Reflex, PositionPD, Torque, PID, SpringDamper, Locked],
    Field(discriminator="controller"),
]


# =============================================================================
# ③ symmetry — phase decomposed into mode + tune (no magic strings)
# =============================================================================
class Phase(BaseModel):
    mode: PhaseMode = PhaseMode.ANTIPHASE
    offset_rad: Optional[float] = None      # required iff mode == OFFSET
    tune: Bind = Bind.FIXED                 # FIXED | ACTION (network-tuned) | OBS

    @model_validator(mode="after")
    def _offset_needs_value(self):
        if self.mode is PhaseMode.OFFSET and self.offset_rad is None:
            raise ValueError("phase.mode=offset requires phase.offset_rad")
        return self


class MirrorPair(BaseModel):
    mirror: dict[str, str]                  # {derived_joint: source_joint}
    sign: Literal[-1, 1] = -1
    phase: Phase = Phase()


# =============================================================================
# ④ per-DOF restriction (discriminated union on `mode`)
# =============================================================================
class DofFree(BaseModel):
    mode: Literal[DofMode.FREE] = DofMode.FREE


class DofLock(BaseModel):
    mode: Literal[DofMode.LOCK] = DofMode.LOCK
    value: float = 0.0


class DofRail(BaseModel):
    mode: Literal[DofMode.RAIL] = DofMode.RAIL
    range: tuple[float, float]              # per-episode randomized lock target


class DofSoftLimit(BaseModel):
    mode: Literal[DofMode.SOFT_LIMIT] = DofMode.SOFT_LIMIT
    range: tuple[float, float]


Dof = Annotated[Union[DofFree, DofLock, DofRail, DofSoftLimit], Field(discriminator="mode")]


def _normalize_dof(v):
    """Accept the authoring shorthands for one DOF entry:
    "free" | "lock" | {"lock": 0.3} | {"rail": [a, b]} | {"soft_limit": [a, b]} | typed dict."""
    if isinstance(v, str):
        return {"mode": v}
    if isinstance(v, dict) and "mode" not in v and len(v) == 1:
        k, val = next(iter(v.items()))
        if k == "lock":
            return {"mode": "lock", "value": val}
        if k in ("rail", "soft_limit"):
            return {"mode": k, "range": val}
    return v


# =============================================================================
# the remaining axes
# =============================================================================
class Plant(BaseModel):
    model: str = "mujoco/dash01/dash01.xml"
    keyframe: str = "stand"
    control_hz: int = 50


class Observation(BaseModel):
    module: str = "standard"                # bare name -> library; ./x.py -> local override
    history_len: int = Field(5, ge=1)
    scales: dict[str, float] = Field(default_factory=lambda: dict(
        motor_pos=1.0, motor_vel=0.1, motor_torque=0.01, gravity=1.0, ang_vel=0.25))


class Sprint(BaseModel):
    """The 100 m dash geometry (Command.mode == SPRINT). The sprint reward weights
    (w_time, w_stop_vel, finish_bonus, ...) are reward params, not command fields."""
    dist_m: float = 100.0
    dist_start_m: float = 25.0              # distance-curriculum start (line ramps to dist_m)
    curriculum_steps: int = 0               # 0 = no ramp
    brake_m: float = 5.0
    stop_speed_eps: float = 0.15
    stop_hold_s: float = 1.0


class Command(BaseModel):
    mode: CommandMode = CommandMode.JOYSTICK
    vx_max: float = 1.5
    yaw_max: float = 2.0
    forward_only: bool = False
    p_stand: float = Field(0.2, ge=0, le=1)
    # sampling bounds for JOYSTICK mode (SPEED/SPRINT pin the command; see compile.py):
    vx_frac: float = 0.0                    # sampled |vx| <= vx_frac * vx_max
    vx_min_frac: float = 0.0                # minimum move-command magnitude (anti-skate)
    yaw_frac: float = 0.0
    resample_s: float = 4.0
    episode_s: float = 20.0                 # episode truncation (task duration)
    sprint: Optional[Sprint] = None

    @model_validator(mode="after")
    def _sprint_block(self):
        if self.mode is CommandMode.SPRINT and self.sprint is None:
            self.sprint = Sprint()
        if self.mode is not CommandMode.SPRINT and self.sprint is not None:
            raise ValueError("command.sprint is only valid with command.mode=sprint")
        return self


class RewardSpec(BaseModel):
    """A named reward module + its parameters. `module` is a library name
    (experiments/_lib/rewards/<name>.py) or a local './reward.py'. `params` configure
    the module — for the stock reward they map 1:1 onto the rl.config.Config weight
    fields (w_foot_slip, gait_cmd_gate, ...; see framework/compile.py REWARD_PARAMS),
    which makes them the primary sweep axes."""
    module: str = "gait_speed_v3"
    params: dict[str, Union[float, int, bool]] = Field(default_factory=dict)


class CurriculumSpec(BaseModel):
    """Either the library command ramp (`cmd_ramp`, params start/steps — target is
    command.vx_frac) or a local './curriculum.py' exposing stages(cfg)."""
    module: str
    params: dict[str, Union[float, int, bool]] = Field(default_factory=dict)


def _normalize_spec(v):
    return {"module": v} if isinstance(v, str) else v


class Policy(BaseModel):
    hidden: list[int] = Field(default_factory=lambda: [256, 256])
    activation: Literal["tanh", "relu", "elu"] = "tanh"
    module: Optional[str] = None            # architecture escape hatch (./network.py)


class Entropy(BaseModel):
    coef: float = 0.01; final: float = 0.002; anneal_steps: int = 4_000_000
    gate_air_time: float = 0.02; max_log_std: float = 0.0


class PPO(BaseModel):
    total_steps: int = 20_000_000
    n_steps: int = 1024; batch_size: int = 4096; n_epochs: int = 4
    gamma: float = Field(0.995, gt=0, lt=1); gae_lambda: float = 0.95
    learning_rate: float = 3e-4; lr_final: float = 1e-4
    clip_range: float = 0.2; target_kl: float = 0.03
    entropy: Entropy = Entropy(); seed: int = 0


class Backend(BaseModel):
    device: Device = Device.CPU
    n_envs: Union[int, Literal["auto"]] = "auto"


class DomainRand(BaseModel):
    """Domain randomization (sim2real). Ranges multiplicative unless noted."""
    enabled: bool = False
    mass: float = 0.15
    friction: tuple[float, float] = (0.6, 1.2)
    motor_strength: float = 0.15
    pd_gain: float = 0.15
    latency_steps: int = 2
    imu_gyro_noise: float = 0.1
    motor_pos_noise: float = 0.01
    push_interval_s: float = 0.0


class Calibration(BaseModel):
    """The recorded real->sim map, validated by the L0 telemetry cross-check."""
    file: Optional[str] = None              # e.g. deploy/calib_dash01.npz
    joint_offset_rad: dict[str, float] = Field(default_factory=dict)
    joint_sign: dict[str, Literal[-1, 1]] = Field(default_factory=dict)
    imu_accel_units: Literal["g", "m/s2"] = "m/s2"
    gyro_units: Literal["rad/s", "deg/s"] = "rad/s"


class Safety(BaseModel):
    terminate_on: list[SafetyEvent] = Field(
        default_factory=lambda: [SafetyEvent.FALL, SafetyEvent.TILT_60DEG, SafetyEvent.FLOOR_BLOWUP])
    term_height: float = 0.45
    term_gravity_z: float = -0.5            # body-frame gravity z above this -> tipped > 60 deg
    grounded_h: float = 0.005               # contact-debounce height for the gait gates
    fall_penalty: float = 100.0
    reset_joint_noise: float = 0.03
    push_interval_s: float = 0.0            # gentle random base shoves (0 = off)
    push_dv: float = 0.4
    safe_envelope: Optional[str] = None     # recorded safe-space file; runtime watchdog + deploy gate
    sim2sim_gate: Optional[str] = "gait_probe"


class Deploy(BaseModel):
    export: Literal["onnx"] = "onnx"
    target: DeployTarget = DeployTarget.MOTEUS
    per_joint: Literal["from_controllers"] = "from_controllers"
    calibration: Calibration = Calibration()
    bringup_max_level: int = Field(0, ge=0, le=4)   # ladder rung authorized for this policy


class Run(BaseModel):
    warm_start: Optional[str] = None        # a run id ("<id>" | "<id>:latest" | "<id>:ppo_400000")
    #                                         or a checkpoint path (legacy form)
    checkpoint_every: int = 200_000
    outputs: list[Literal["csv", "plots", "tensorboard"]] = Field(
        default_factory=lambda: ["csv", "plots", "tensorboard"])


# =============================================================================
# the experiment
# =============================================================================
class Experiment(BaseModel):
    model_config = {"extra": "forbid"}      # a typo'd key is an error, not a silent ignore

    name: str
    description: str = Field(..., min_length=8)     # REQUIRED: why this experiment exists
    inherits: Optional[str] = None          # library base to deep-merge under this
    plant: Plant = Plant()
    base_dof: dict[Axis, Dof] = Field(default_factory=dict)
    joints: dict[str, Controller] = Field(default_factory=dict)   # actuated, emitting joints only
    symmetry: dict[str, MirrorPair] = Field(default_factory=dict)
    observation: Observation = Observation()
    command: Command = Command()
    reward: RewardSpec = RewardSpec()
    curriculum: Optional[CurriculumSpec] = None
    steering: Optional[str] = None          # eval-only scripted command profile (./steering.py)
    policy: Policy = Policy()
    ppo: PPO = PPO()
    backend: Backend = Backend()
    dr: DomainRand = DomainRand()
    run: Run = Run()
    deploy: Deploy = Deploy()
    safety: Safety = Safety()

    # ---- authoring-shorthand normalization (YAML forms from experiments/README.md) ----
    @field_validator("base_dof", mode="before")
    @classmethod
    def _base_dof_shorthand(cls, v):
        if isinstance(v, (list, tuple)):                        # positional [X..yaw] 6-list
            if len(v) != 6:
                raise ValueError("positional base_dof needs exactly 6 entries [X,Y,Z,roll,pitch,yaw]")
            v = {axis.value: entry for axis, entry in zip(AXIS_ORDER, v)}
        if isinstance(v, dict):
            return {k: _normalize_dof(e) for k, e in v.items()}
        return v

    @field_validator("reward", mode="before")
    @classmethod
    def _reward_shorthand(cls, v):
        return _normalize_spec(v)

    @field_validator("curriculum", mode="before")
    @classmethod
    def _curriculum_shorthand(cls, v):
        return _normalize_spec(v)

    @field_validator("steering", mode="before")
    @classmethod
    def _steering_shorthand(cls, v):
        # the m3_ft draft wrote `steering: {module: ./steering.py}`
        if isinstance(v, dict) and set(v) == {"module"}:
            return v["module"]
        return v

    @model_validator(mode="after")
    def _backend_supports_controllers(self):
        kinds = {c.controller for c in self.joints.values()}
        if self.backend.device is Device.GPU and (
                ControllerKind.PATTERN_GEN in kinds or self.command.mode is CommandMode.SPRINT):
            raise ValueError("MJX/GPU backend does not support pattern_gen or sprint — use device=cpu")
        return self


if __name__ == "__main__":
    # discoverability demo — the answer to "how would I know what's legal?"
    print("controllers:", [c.value for c in ControllerKind])
    print("phase modes:", [p.value for p in PhaseMode], "| tune:", [b.value for b in Bind])
    print("dof modes:  ", [d.value for d in DofMode])
    # full machine-readable format: Experiment.model_json_schema()
