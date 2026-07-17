"""Reward for m3_ft — LOCAL, editable copy of _lib/rewards/gait_speed_v3.py.

Drop-in per-experiment override: edit terms freely here without touching the env or the
library. The library file's docstring carries the full `state`/`cmd` field reference;
as shipped this copy is byte-identical math to the built-in reward (the library
version is pinned to the built-in path by tests/test_module_parity.py).
"""
import numpy as np


def pen(cfg, v) -> float:
    """Floor a penalty term at -cfg.penalty_term_cap (suicide-proofing; reward
    normalization is OFF, so raw scales reach PPO directly)."""
    return max(float(v), -cfg.penalty_term_cap)


def reward(state, cmd, cfg) -> dict:
    t = {}
    vx = state.vx

    # ---- objective terms per command mode (verbatim: Dash01Env._reward) ----
    if cfg.sprint_mode:
        for k in ("track_vx", "track_yaw", "progress", "track_pos",
                  "track_heading", "pos_pen", "yaw_rate", "heading_pen"):
            t[k] = 0.0
        t["time"] = -cfg.w_time
        if not state.sprint_crossed:
            t["fwd_speed"] = cfg.w_fwd_speed * float(np.clip(vx, 0.0, cfg.v_ceiling))
            t["stop"] = 0.0
            t["overrun"] = 0.0
        else:
            t["fwd_speed"] = 0.0
            t["stop"] = cfg.w_stop_vel * float(np.exp(-((vx / cfg.stop_sigma) ** 2)))
            t["overrun"] = pen(cfg, -cfg.w_overrun * state.sprint_overrun_m)
    elif cfg.speed_mode:
        t["fwd_speed"] = cfg.w_fwd_speed * float(np.clip(vx, 0.0, cfg.v_ceiling))
        for k in ("track_vx", "track_yaw", "progress", "track_pos",
                  "track_heading", "pos_pen", "yaw_rate", "heading_pen"):
            t[k] = 0.0
    else:
        t["fwd_speed"] = 0.0
        t["track_vx"] = cfg.w_track_vx * np.exp(-((cmd.vx - vx) ** 2) / cfg.track_sigma_vx ** 2)
        t["track_yaw"] = cfg.w_track_yaw * np.exp(
            -((cmd.yaw - state.yaw_rate) ** 2) / cfg.track_sigma_yaw ** 2)
        t["progress"] = cfg.w_progress * state.progress_frac ** 2
        t["track_pos"] = cfg.w_track_pos * np.exp(-state.pos_err ** 2 / cfg.track_sigma_pos ** 2)
        t["track_heading"] = cfg.w_track_heading * np.exp(
            -(state.yaw_err ** 2) / cfg.track_sigma_heading ** 2)
        t["pos_pen"] = pen(cfg, -cfg.w_pos_l1 * state.pos_err)
        t["yaw_rate"] = pen(cfg, -cfg.w_yaw_rate * (state.yaw_rate - cmd.yaw) ** 2)
        t["heading_pen"] = pen(cfg, -cfg.w_heading_pen * state.yaw_err ** 2)

    # ---- posture / effort (shared) ----
    t["lat_vel"] = pen(cfg, -cfg.w_lat_vel * state.vy ** 2)
    t["ang_xy"] = pen(cfg, -cfg.w_angvel_xy * (state.ang_vel[0] ** 2 + state.ang_vel[1] ** 2))
    t["upright"] = pen(cfg, -cfg.w_upright
                       * (state.gravity_body[0] ** 2 + state.gravity_body[1] ** 2))
    if state.z_locked:          # height/vz are meaningless when Z is railed
        t["height"] = 0.0
        t["vz"] = 0.0
    else:
        t["height"] = pen(cfg, -cfg.w_height * (state.height - state.height_target) ** 2)
        t["vz"] = pen(cfg, -cfg.w_vz * state.vz ** 2)
    t["action_rate"] = pen(cfg, -cfg.w_action_rate * state.action_rate_sq)
    exc = np.maximum(np.abs(state.motor_torque) - state.stand_torque, 0.0)
    t["torque"] = pen(cfg, -cfg.w_torque * np.sum(exc ** 2))
    t["stance"] = pen(cfg, -cfg.w_no_cross * max(0.0, cfg.stance_min_sep - state.foot_sep) ** 2)
    t["hip_roll"] = pen(cfg, -cfg.w_hip_roll * float(np.sum(np.square(state.hip_roll))))
    t["alive"] = cfg.w_alive

    # ---- gait battery: the v3 anti-skate terms (verbatim: Dash01Env._gait_reward) ----
    feet = (state.footL, state.footR)
    slip = sum(max(0.0, f.slip_speed - cfg.slip_deadband) ** 2 for f in feet)
    t["foot_slip"] = -min(cfg.w_foot_slip * float(slip), cfg.penalty_term_cap)

    gait_on = abs(cmd.vx) >= cfg.gait_cmd_gate
    air = 0.0
    if gait_on:
        air = sum(cfg.w_air_time
                  * float(np.clip(f.air_time - cfg.foot_air_time_min, 0.0, cfg.air_credit_cap_s))
                  for f in feet if f.just_landed)
    t["air_time"] = air

    if gait_on:
        cap = cfg.stance_cap_s if abs(cmd.vx) >= cfg.stance_slow_speed else cfg.stance_cap_slow_s
        over = sum(min(max(f.stance_time - cap, 0.0), 1.0) for f in feet)
        t["stance_time"] = -min(cfg.w_stance_time * float(over), cfg.penalty_term_cap)
    else:
        t["stance_time"] = 0.0

    clear = 0.0
    if gait_on:
        for f in feet:
            if f.fresh_swing:
                frac = np.clip((float(f.clearance) - cfg.clearance_dead_m)
                               / cfg.clearance_scale_m, 0.0, 1.0)
                clear += cfg.w_clearance * float(frac) * (0.3 + 0.7 * state.progress_frac)
    t["clearance"] = clear

    return t
