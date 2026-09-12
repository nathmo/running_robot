"""A structurally real v2 policy bundle, with nonsense weights. TEST FIXTURE ONLY.

A trained bundle is ~5 MB of weights and is not in git, so every test that needs one either skips
or builds one. This builds one: every array width, every meta field and every derived alias is
what `walk_v2/export.py` writes, and `bundle.Bundle` validates it on the way out -- so a test can
exercise the whole deployment path (arm, approach, run, the run/stop command, the log) without a
training run on disk.

The WEIGHTS are random. Nothing here can tell you what a policy does; it tells you that the
plumbing carries whatever a policy says. Do not put one of these in `robot/deploy/bundles/` or
`data/policies/` on the robot -- a bundle that loads is a bundle the panel will offer to run.
"""
import json

import numpy as np

SPEC_DIM, N_RESIDUAL, ACTION_DIM = 44, 6, 50
FRAME_DIM, HIST_LEN, HIST_STRIDE = 33, 10, 2
ONCE_DIM = SPEC_DIM + 3                        # latched spec | task 2 | commit 1
ACTOR_DIM = FRAME_DIM * HIST_LEN + ONCE_DIM    # 377
# v3 adds one channel to the frame: the robot's own dead-reckoned heading (integrated gyro z).
# Everything else -- the once-block, the action, the gait vocabulary -- is unchanged, which is why
# one control law serves both and why these two constants are all a fixture needs to switch.
FRAME_DIM_V3 = 34
ACTOR_DIM_V3 = FRAME_DIM_V3 * HIST_LEN + ONCE_DIM    # 387
Q_LO = np.array([-0.785, -1.5, -1.047, -0.785, -1.5, -1.047])
VEL_LIMIT = np.array([10.30, 22.01, 22.01, 10.30, 22.01, 22.01])
FORCERANGE = np.array([61.2, 144.5, 144.5, 61.2, 144.5, 144.5])
DEFAULT_GAIT = dict(
    cam_amp=0.45, thigh_amp=0.45, roll_amp=0.20, delta_max=0.6, o_max=[0.06, 0.06, 0.15],
    imp_kp_up=2.5, imp_kp_dn=3.0, imp_kd_up=1.0, imp_kd_dn=4.0,
    reflex_kp_scale=0.5, reflex_kd_scale=0.1, reflex_bias_scale=0.2,
    pitch_kp=1.0, pitch_kd=0.1, pitch_bias=0.0, pitch_clip=0.25, residual_scale=0.20,
    freq_lo=0.5, freq_hi=5.0, drive_kp=[120.0, 200.0, 200.0, 120.0, 200.0, 200.0],
    drive_kd=[4.0, 5.0, 5.0, 4.0, 5.0, 5.0])


def hist_idx():
    raw = (HIST_LEN - 1) * HIST_STRIDE + 1
    return (raw - 1) - (np.arange(HIST_LEN) * HIST_STRIDE)[::-1]


def v2_arrays_and_meta(gait_params=None, nominal=None, default_motor_pos=None, pitch_lp=0.0,
                       objective="sprint", spec_source="policy", stoplight_prob=0.5, seed=0,
                       weight_scale=0.05, heading=False, **meta_over):
    """(arrays, meta) for a bundle. `heading=True` makes it a v3 (34-wide frame) bundle.

    `meta_over` overrides any meta field."""
    rng = np.random.default_rng(seed)
    FRAME, ACTOR = ((FRAME_DIM_V3, ACTOR_DIM_V3) if heading else (FRAME_DIM, ACTOR_DIM))
    gp = dict(DEFAULT_GAIT if gait_params is None else gait_params)
    nominal = (np.array([0.0, 0.0, 0.12, 0.0, 0.0, -0.12]) if nominal is None
               else np.asarray(nominal, float))
    dmp = nominal.copy() if default_motor_pos is None else np.asarray(default_motor_pos, float)
    n = lambda *s: (rng.standard_normal(s) * weight_scale).astype(np.float32)
    arrays = {
        "est_w0": n(128, ACTOR), "est_b0": n(128),
        "est_w1": n(64, 128), "est_b1": n(64),
        "est_w2": n(3, 64), "est_b2": n(3),
        "pi_w0": n(256, ACTOR + 3), "pi_b0": n(256),
        "pi_w1": n(256, 256), "pi_b1": n(256),
        "act_w": n(ACTION_DIM, 256), "act_b": n(ACTION_DIM),
        "obs_mean": np.zeros(ACTOR), "obs_var": np.ones(ACTOR),
        "nominal_ctrl": nominal, "default_motor_pos": dmp,
        "q_lo": Q_LO.copy(), "q_hi": -Q_LO.copy(),
        "motor_vel_limit": VEL_LIMIT.copy(), "forcerange": FORCERANGE.copy(),
        "drive_kp": np.asarray(gp["drive_kp"], float), "drive_kd": np.asarray(gp["drive_kd"], float),
        "stand_torque": np.zeros(6), "hist_idx": hist_idx().astype(np.int32),
        "latched_dims": np.array([1] * SPEC_DIM + [0] * N_RESIDUAL, np.int32),
        "log_std": np.full(ACTION_DIM, -1.5, np.float32),
    }
    meta = {
        "bundle_version": 2, "run": "v2_fixture", "checkpoint": "step_0", "step": 0, "nu": 6,
        "spec_source": spec_source, "control_dt": 0.01,
        "frame_dim": FRAME, "history_len": HIST_LEN, "history_stride": HIST_STRIDE,
        "once_dim": ONCE_DIM, "actor_dim": ACTOR, "action_dim": ACTION_DIM,
        "wrap_index": ACTOR - 1, "clip_obs": 10.0, "obs_eps": 1e-8,
        "obs_scales": dict({"motor_pos": 1.0, "motor_vel": 0.1, "motor_torque": 0.01,
                            "gravity": 1.0, "ang_vel": 0.25, "base_vel": 1.0},
                           **({"heading": 1.0} if heading else {})),
        **({"heading_cap_rad": 1.5708} if heading else {}),
        "lp_yaw_tau_s": 0.7, "est_hidden": [128, 64], "policy_hidden": [256, 256],
        "task_brake_m": 8.0, "sprint_dist_m": 100.0, "objective": objective,
        "gait": gp, "pitch_reflex_rate_lp": pitch_lp, "motor_accel_limit": 0.0, "n_harmonics": 3,
        "residual_scale": gp["residual_scale"], "action_scale": 0.5,
        "drive_delay_ms": 12.0, "motor_bus_volts": 48.0,
        "resync": {"kappa": 0.5, "window_cycle": 0.15, "ema_cycles": 5.0, "warmup_cycles": 3,
                   "enable": True},
        "stop": {"stoplight_prob_final": float(stoplight_prob),
                 "stoplight_curriculum_steps": 20000000, "stop_decel_s": 1.5,
                 "stop_speed_eps": 0.15, "stop_hold_s": 1.0},
        "term_height": 0.45, "term_gravity_z": -0.5, "height_stand": 1.0,
        "planar": False, "base_lock": [False] * 6,
        "actuator_names": ["hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R"],
        # the command channel, exactly as walk_v2/export.py writes it: the scale is ZERO unless the
        # command IS a speed, so that a run/stop bundle cannot be read as a joystick with a
        # plausible-looking range.
        "v_ceiling": 3.0, "v_cmd_rate": 0.0,
        "v_max": 3.0 if objective == "joystick" else 0.0,
        "v_min": 0.0,
        "command": {"kind": "speed_fraction" if objective == "joystick" else "run_flag_distance",
                    "zero_means": "step in place"},
    }
    meta.update(meta_over)
    return arrays, meta


def v2_joystick_bundle(**kw):
    """A loaded joystick bundle: task[0] is the commanded speed / v_max, task[1] reserved at 0."""
    kw.setdefault("objective", "joystick")
    return v2_bundle(**kw)


def v2_bundle(**kw):
    """A loaded `bundle.Bundle`, validated."""
    from bundle import Bundle
    arrays, meta = v2_arrays_and_meta(**kw)
    return Bundle(arrays, meta)


def write_v2_bundle(path, **kw):
    """Write one to disk in the on-the-wire format, and return the path."""
    arrays, meta = v2_arrays_and_meta(**kw)
    np.savez(str(path), meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    return str(path)
