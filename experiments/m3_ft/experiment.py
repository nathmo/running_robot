"""m3_ft as a typed Python object (config-as-code).

The SAME experiment as experiment.yaml, but authored through the schema. This is the
discoverability payoff of the pydantic schema: ctrl-click `PatternGen` to see its
fields; type `PhaseMode.` and the editor lists INPHASE / ANTIPHASE / OFFSET; a wrong
value is a red squiggle before you run. No magic strings, no "how would I know?".

Author an experiment EITHER way — this file, or experiment.yaml (both validate through
framework/schema.py). Pick whichever you prefer; a coder usually wants this one.
"""
from _lib.schema import (
    Experiment, Plant, Axis, DofFree, DofLock, PatternGen, Reflex, ReflexInput,
    MirrorPair, Phase, PhaseMode, Bind, Observation, Command, CommandMode,
    Policy, PPO, Backend, Device, Run, Deploy, Safety, SafetyEvent,
)

EXPERIMENT = Experiment(
    name="m3_ft",
    description="M3 Fourier fine-tune (typed form): every axis spelled out, "
                "all module escape hatches demonstrated.",
    inherits="dash01_base",

    # ① plant
    plant=Plant(model="mujoco/dash01/dash01.xml", keyframe="stand", control_hz=50),

    # ④ base DOF — X, Z, pitch free; Y, roll, yaw railed (M3)
    base_dof={
        Axis.X:     DofFree(),
        Axis.Y:     DofLock(value=0.0),
        Axis.Z:     DofFree(),
        Axis.ROLL:  DofLock(value=0.0),
        Axis.PITCH: DofFree(),
        Axis.YAW:   DofLock(value=0.0),
    },

    # ② + ③ per-joint controllers — only the emitting (left) joints
    joints={
        "cam_L":      PatternGen(n_harmonics=3, amp=0.30, freq_hz=(0.5, 3.0)),
        "thigh_L":    PatternGen(n_harmonics=3, amp=0.35, freq_hz=(0.5, 3.0)),
        "hip_roll_L": Reflex(inputs=[ReflexInput.ROLL, ReflexInput.ROLL_RATE],
                             gains="learned", kp_scale=0.5, kd_scale=0.1, bias_scale=0.2),
    },
    symmetry={
        "gait_pair": MirrorPair(
            mirror={"cam_R": "cam_L", "thigh_R": "thigh_L", "hip_roll_R": "hip_roll_L"},
            sign=-1,
            # the runtime implements antiphase with a FIXED pi offset; tune=Bind.ACTION
            # (network-learned L/R phase delta) is future work — the compiler gates it
            phase=Phase(mode=PhaseMode.ANTIPHASE, tune=Bind.FIXED),
        ),
    },

    # ⑤ / ⑥
    observation=Observation(module="./observation.py", history_len=5),
    command=Command(mode=CommandMode.SPEED, vx_max=2.5, forward_only=True, p_stand=0.0),

    # ⑦ / ⑧  (strings coerce to RewardSpec / CurriculumSpec)
    reward="./reward.py",
    curriculum="./curriculum.py",
    steering="./steering.py",

    # ⑨
    policy=Policy(hidden=[256, 256], activation="tanh", module="./network.py"),
    ppo=PPO(gamma=0.93, n_steps=256, batch_size=512, total_steps=800_000),

    # ⑩
    backend=Backend(device=Device.CPU, n_envs="auto"),

    # ⑪
    run=Run(warm_start="runs/m2_fourier/final_model.zip", checkpoint_every=200_000),
    deploy=Deploy(),   # onnx -> moteus; calibration + bringup ladder configured at deploy time
    safety=Safety(terminate_on=[SafetyEvent.FALL, SafetyEvent.TILT_60DEG, SafetyEvent.FLOOR_BLOWUP],
                  sim2sim_gate="gait_probe"),
)
