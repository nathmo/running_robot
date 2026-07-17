"""Observation assembly for m3_ft — what the policy can see.

CONTRACT
--------
`features(state, cfg)` returns ONE frame (1-D float array) for the current control
step. The framework stacks the last `cfg.history_len` frames (oldest -> newest) into
the policy input and maintains the normalization (VecNormalize) for you.
`privileged(state, cfg)` returns extra features the CRITIC may see but the actor may
NOT (asymmetric actor-critic) — kept out of the deployed policy so sim2real is honest.

Current DASH-01 frame (44 numbers) -> x history_len(5) = 220, matching the trained
m3_ft policy:
    motor_pos    6   * scales.motor_pos
    motor_vel    6   * scales.motor_vel
    motor_torque 6   * scales.motor_torque
    gravity      3   body-frame gravity direction (replaces absolute orientation)
    ang_vel      3   IMU gyro  * scales.ang_vel
    command      2   [forward, yaw]
    prev_action 18   last policy output (fourier action dim)
    ------------------------------------------------------------------
    44 / frame

DELIBERATELY EXCLUDED from the actor: true base linear velocity (privileged only) and
any foot-contact signal (no such sensor on the real robot). If you move to the
fixed-rate hybrid loop, add `state.gait_phase` here (+1 -> 45/frame) so the fast
reflex is gait-aware.
"""
import numpy as np


def features(state, cfg) -> np.ndarray:
    s = cfg.obs_scales
    return np.concatenate([
        state.motor_pos    * s["motor_pos"],
        state.motor_vel    * s["motor_vel"],
        state.motor_torque * s["motor_torque"],
        state.gravity_body * s["gravity"],       # 3
        state.ang_vel      * s["ang_vel"],       # 3  (IMU gyro)
        state.command,                           # 2  [forward, yaw]
        state.prev_action,                       # 18 (fourier)
        # state.gait_phase[None],                # <- enable for the fixed-rate hybrid loop
    ]).astype(np.float32)


def privileged(state, cfg) -> np.ndarray:
    """Critic-only. Never reaches the deployed ONNX policy."""
    return np.array([state.vx, state.vy, state.vz], dtype=np.float32)
