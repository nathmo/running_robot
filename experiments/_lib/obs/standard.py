"""standard — the stock DASH-01 observation frame, as a framework module.

VERBATIM extraction of Dash01Env._proprio (kept identical by
tests/test_module_parity.py). One frame = 6+6+6+3+3+action_dim+2 numbers; the env
stacks the last cfg.history_len frames.

CONTRACT
--------
`features(state, cfg) -> np.ndarray` returns ONE frame for the current control step.
Copy into an experiment folder as ./observation.py to change what the policy sees.

`state`:
    .motor_pos      motor angles MINUS the standing pose (6,)
    .motor_vel      motor velocities (6,)
    .motor_torque   applied torque (6,)
    .gravity_body   gravity dir in body frame (3,)
    .ang_vel        IMU gyro (3,)
    .command        the joystick command in [-1,1] (2,)
    .prev_action    last policy output (6 pd / 18 fourier)
    .gait_phase     the env's gait phase in [0, 2*pi) (fourier modes; 0 otherwise) — appended
                    as a final [sin, cos] pair only in action_mode "fourier_step"

DELIBERATELY EXCLUDED from the actor: true base linear velocity and any foot-contact
signal (no such sensor on the real robot).
"""
import numpy as np


def features(state, cfg) -> np.ndarray:
    s = cfg.obs_scales
    parts = [
        state.motor_pos * s["motor_pos"],
        state.motor_vel * s["motor_vel"],
        state.motor_torque * s["motor_torque"],
        state.gravity_body * s["gravity"],
        state.ang_vel * s["ang_vel"],
        state.prev_action,
        state.command,
    ]
    if cfg.action_mode == "fourier_step":   # + [sin, cos] gait phase (guard keeps old-mode parity)
        parts.append(np.array([np.sin(state.gait_phase), np.cos(state.gait_phase)]))
    return np.concatenate(parts).astype(np.float32)
