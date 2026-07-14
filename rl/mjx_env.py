"""Dash01MjxEnv -- a functional, vmap-able JAX/MJX port of Dash01Env (rl/env.py) for
GPU-parallel training.

This is a PORT, not a reimplementation: every reward/gait-shaping term below mirrors the CPU
env's private method of the same name, in the same order, so the two stay diffable. The reward
was hard-won (see rl/README.md's v2 "skating" post-mortem) -- do not retune weights here without
retuning rl/env.py too. rl/mjx_env_test.py is the automated parity gate between the two; run it
after touching either file.

Key differences from rl/env.py, all forced by JAX being functional/traced rather than
object-oriented/imperative:
  - No `self.` mutation. All per-episode state (history, air_time, command, RNG, ...) lives in an
    EnvState pytree threaded through reset()/step() and returned each call.
  - No implicit global RNG. Every random draw takes an explicit jax.random key, split off a
    per-env key carried in EnvState.rng.
  - step() performs its OWN auto-reset (blends in a fresh reset() when done) rather than relying
    on a wrapper, because Brax's stock AutoResetWrapper resets to the SAME fixed initial state
    captured once at the first reset() call, not a freshly-randomized one -- that would silently
    drop this env's reset-pose noise (reset_joint_noise) and command resampling every episode
    boundary, both real parts of the training recipe. See rl/mjx_train.py for the vmap/scan
    rollout loop that drives this.
  - cmd_vx_frac (curriculum) is a per-env EnvState field the TRAINING LOOP overwrites each
    iteration (`state.replace(cmd_vx_frac=...)`) rather than an env-instance attribute mutated by
    an SB3 callback via env_method.

Physics fidelity (closed-loop parallel-knee equality constraint, condim=6 elliptic-cone contact,
ankle spring) is validated against CPU MuJoCo in mujoco/dash01/validate_mjx.py -- read that
before changing anything physics-related here.

NOTE: an analytical (spring-damper) foot contact model was attempted here to cut Newton-solver
cost, but was reverted -- it's numerically unstable against this model's closed-loop parallel-
knee constraint (verified stable in isolation on a single free body, but diverges on the real
robot even with zero damping / no torque / no extra forward call, i.e. not a tuning problem).
MJX's real contact solver (condim=6, elliptic cone) is kept for the foot -- see the ~1.9x
GPU-vs-CPU throughput measured with it in rl/README.md's GPU training section.
"""
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
from flax import struct

from .config import Config

# per-frame observation groups and sizes (see proprio()); total = 32 -- must match rl/env.py
FRAME_DIM = 6 + 6 + 6 + 3 + 3 + 6 + 2

# every key reward_fn/gait_reward ever assigns, in assignment order -- reset() must pre-populate
# reward_terms with exactly these keys (zeros) so its pytree structure matches step()'s output;
# otherwise jax.lax.scan over repeated step() calls starting from a reset() carry raises a
# structure-mismatch error the first time it's used in a rollout loop.
TERM_NAMES = [
    "fwd_speed",
    "track_vx", "track_yaw", "progress", "track_pos", "track_heading", "pos_pen", "yaw_rate",
    "heading_pen", "lat_vel", "ang_xy", "upright", "height", "vz", "action_rate", "torque",
    "stance", "hip_roll", "alive", "foot_slip", "air_time", "stance_time", "clearance",
]


@struct.dataclass
class EnvState:
    data: mjx.Data
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    terminated: jnp.ndarray     # true fall/floor-violation -- GAE must NOT bootstrap past this
    truncated: jnp.ndarray      # episode-length cutoff -- GAE SHOULD bootstrap past this
    reward_terms: dict
    ep_return: jnp.ndarray      # accumulated reward THIS (carried, possibly in-progress) episode
    ep_length: jnp.ndarray      # accumulated steps THIS (carried, possibly in-progress) episode
    #                             -- these two are genuine carried state (must reset to 0/0 on
    #                             auto-reset, like `data`/`history`/etc, so the NEXT episode counts
    #                             from zero instead of accumulating forever); read the completed
    #                             episode's totals from final_ep_return/final_ep_length instead,
    #                             which -- like reward/done -- are protected from the auto-reset
    #                             blend and hold the TRUE final total exactly on the done step.
    final_ep_return: jnp.ndarray
    final_ep_length: jnp.ndarray
    step: jnp.ndarray
    history: jnp.ndarray
    prev_action: jnp.ndarray
    filt_target: jnp.ndarray
    delay_buf: jnp.ndarray
    air_time: jnp.ndarray
    contact_time: jnp.ndarray
    grounded_prev: jnp.ndarray
    prev_toe_xy: jnp.ndarray
    push_countdown: jnp.ndarray
    command: jnp.ndarray
    des_xy: jnp.ndarray
    des_yaw: jnp.ndarray
    cmd_vx_frac: jnp.ndarray
    rng: jnp.ndarray


def _pen(v, cap):
    return jnp.maximum(v, -cap)


def base_rot(data, base_id):
    return data.xmat[base_id]           # mjx stores xmat as (nbody,3,3) already -- no reshape


def base_yaw(data, base_id):
    R = base_rot(data, base_id)
    return jnp.arctan2(R[1, 0], R[0, 0])


def gravity_body(data, base_id):
    return base_rot(data, base_id).T @ jnp.array([0.0, 0.0, -1.0])


def ang_vel_body(data, gyro_adr):
    return data.sensordata[gyro_adr:gyro_adr + 3]      # gyro_adr is a static python int


def foot_contacts(data, foot_contact_slot):
    """Which foot-tip spheres touch the floor, via MJX's static per-model contact-pair slots
    (see Dash01MjxEnv.__init__ for how these slots are discovered)."""
    return data.contact.dist[foot_contact_slot] < 0.0


def toe_heights(data, foot_gids_arr, toe_r):
    return data.geom_xpos[foot_gids_arr, 2] - toe_r


def grounded_fn(contact, heights, grounded_h):
    return contact | (heights < grounded_h)


def foot_lateral_sep(data, base_id, foot_gids):
    """Body-frame lateral separation of the two toe spheres. foot_gids is a static 2-elem
    python list, so this unrolls to two ops at trace time -- matches rl/env.py exactly."""
    R = base_rot(data, base_id)
    base = data.qpos[0:3]
    y = [(R.T @ (data.geom_xpos[fg] - base))[1] for fg in foot_gids]
    return y[0] - y[1]


def floor_violation(data, floor_pairs):
    """floor_pairs: static list of (contact_slot, geom_radius) for every (floor, foot/heel)
    MJX contact-pair slot. Mirrors rl/env.py._floor_violation's deep-penetration safety check."""
    flags = jnp.array([data.contact.dist[idx] < -0.5 * r for idx, r in floor_pairs])
    return jnp.any(flags)


def proprio(data, cfg, act_qadr, act_dadr, default_motor_pos, nu, base_id, gyro_adr,
            prev_action, command):
    s = cfg.obs_scales
    motor_pos = (data.qpos[act_qadr] - default_motor_pos) * s["motor_pos"]
    motor_vel = data.qvel[act_dadr] * s["motor_vel"]
    motor_trq = data.actuator_force[:nu] * s["motor_torque"]
    grav = gravity_body(data, base_id) * s["gravity"]
    angv = ang_vel_body(data, gyro_adr) * s["ang_vel"]
    return jnp.concatenate([motor_pos, motor_vel, motor_trq, grav, angv, prev_action, command])


def sample_command(rng, cfg, cmd_vx_frac):
    k_stand, k_mag, k_sign, k_yaw = jax.random.split(rng, 4)
    is_stand = jax.random.uniform(k_stand) < cfg.p_stand
    hi = jnp.maximum(cmd_vx_frac, cfg.cmd_vx_min_frac)
    lo = cfg.cmd_vx_min_frac
    mag = jax.random.uniform(k_mag, minval=lo, maxval=jnp.maximum(hi, lo + 1e-6))
    if cfg.cmd_forward_only:
        sign = jnp.array(1.0)
    else:
        sign = jnp.where(jax.random.uniform(k_sign) < 0.5, 1.0, -1.0)
    vx = jnp.where(is_stand, 0.0, sign * mag)
    yaw = jnp.where(is_stand, 0.0,
                     jax.random.uniform(k_yaw, minval=-cfg.cmd_yaw_frac, maxval=cfg.cmd_yaw_frac))
    return jnp.array([vx, yaw])


def advance_command_pose(des_xy, des_yaw, command, cfg, control_dt):
    cmd_vx = command[0] * cfg.vx_max
    cmd_yaw = command[1] * cfg.yaw_max
    new_yaw = des_yaw + cmd_yaw * control_dt
    new_xy = des_xy + control_dt * cmd_vx * jnp.array([jnp.cos(new_yaw), jnp.sin(new_yaw)])
    return new_xy, new_yaw


def anchor_command_pose(data, base_id):
    return data.qpos[0:2], base_yaw(data, base_id)


def next_push_in(rng, cfg, control_dt):
    if cfg.push_interval_s <= 0:
        return jnp.array(10 ** 9, dtype=jnp.int32)
    s = cfg.push_interval_s * jax.random.uniform(rng, minval=0.7, maxval=1.3)
    return jnp.maximum(1, jnp.round(s / control_dt).astype(jnp.int32))


def reward_fn(data, cfg, action, prev_action, command, des_xy, des_yaw, base_id, gyro_adr,
              act_qadr, hip_roll_idx, default_motor_pos, height_target, stand_torque, nu,
              foot_gids):
    R = base_rot(data, base_id)
    v_body = R.T @ data.qvel[0:3]
    vx = v_body[0]
    angv = ang_vel_body(data, gyro_adr)
    yaw_rate = angv[2]
    cmd_vx = command[0] * cfg.vx_max
    cmd_yaw = command[1] * cfg.yaw_max
    grav = gravity_body(data, base_id)

    t = {}
    # speed_mode / Z-lock are STATIC (cfg fields are Python scalars), so these branches resolve at
    # trace time -- exact mirror of rl/env.py._reward. Dicts flatten by sorted key in JAX, so both
    # branches producing the same key SET keeps the reward_terms pytree structure stable.
    if cfg.speed_mode:
        t["fwd_speed"] = cfg.w_fwd_speed * jnp.clip(vx, 0.0, cfg.v_ceiling)
        for k in ("track_vx", "track_yaw", "progress", "track_pos",
                  "track_heading", "pos_pen", "yaw_rate", "heading_pen"):
            t[k] = jnp.asarray(0.0)
        progress_frac = jnp.clip(vx / cfg.v_ceiling, 0.0, 1.0)
    else:
        t["fwd_speed"] = jnp.asarray(0.0)
        t["track_vx"] = cfg.w_track_vx * jnp.exp(-((cmd_vx - vx) ** 2) / cfg.track_sigma_vx ** 2)
        t["track_yaw"] = cfg.w_track_yaw * jnp.exp(-((cmd_yaw - yaw_rate) ** 2) / cfg.track_sigma_yaw ** 2)
        progress_frac = jnp.where(jnp.abs(cmd_vx) > 1e-6, jnp.clip(vx / jnp.where(cmd_vx == 0, 1.0, cmd_vx), 0.0, 1.0), 0.0)
        t["progress"] = cfg.w_progress * progress_frac ** 2
        pos_err = jnp.linalg.norm(data.qpos[0:2] - des_xy)
        t["track_pos"] = cfg.w_track_pos * jnp.exp(-pos_err ** 2 / cfg.track_sigma_pos ** 2)
        yaw_err = jnp.arctan2(jnp.sin(base_yaw(data, base_id) - des_yaw), jnp.cos(base_yaw(data, base_id) - des_yaw))
        t["track_heading"] = cfg.w_track_heading * jnp.exp(-(yaw_err ** 2) / cfg.track_sigma_heading ** 2)
        t["pos_pen"] = _pen(-cfg.w_pos_l1 * pos_err, cfg.penalty_term_cap)
        t["yaw_rate"] = _pen(-cfg.w_yaw_rate * (yaw_rate - cmd_yaw) ** 2, cfg.penalty_term_cap)
        t["heading_pen"] = _pen(-cfg.w_heading_pen * yaw_err ** 2, cfg.penalty_term_cap)
    t["lat_vel"] = _pen(-cfg.w_lat_vel * v_body[1] ** 2, cfg.penalty_term_cap)
    t["ang_xy"] = _pen(-cfg.w_angvel_xy * (angv[0] ** 2 + angv[1] ** 2), cfg.penalty_term_cap)
    t["upright"] = _pen(-cfg.w_upright * (grav[0] ** 2 + grav[1] ** 2), cfg.penalty_term_cap)
    if bool(cfg.base_lock[2]):                        # Z railed -> height/vz meaningless, neutralize
        t["height"] = jnp.asarray(0.0)
        t["vz"] = jnp.asarray(0.0)
    else:
        t["height"] = _pen(-cfg.w_height * (data.qpos[2] - height_target) ** 2, cfg.penalty_term_cap)
        t["vz"] = _pen(-cfg.w_vz * data.qvel[2] ** 2, cfg.penalty_term_cap)
    t["action_rate"] = _pen(-cfg.w_action_rate * jnp.sum((action - prev_action) ** 2), cfg.penalty_term_cap)
    exc = jnp.maximum(jnp.abs(data.actuator_force[:nu]) - jnp.abs(stand_torque), 0.0)
    t["torque"] = _pen(-cfg.w_torque * jnp.sum(exc ** 2), cfg.penalty_term_cap)
    sep = foot_lateral_sep(data, base_id, foot_gids)
    t["stance"] = _pen(-cfg.w_no_cross * jnp.maximum(0.0, cfg.stance_min_sep - sep) ** 2, cfg.penalty_term_cap)
    hr = data.qpos[act_qadr[hip_roll_idx]] - default_motor_pos[hip_roll_idx]
    t["hip_roll"] = _pen(-cfg.w_hip_roll * jnp.sum(hr ** 2), cfg.penalty_term_cap)
    t["alive"] = jnp.asarray(cfg.w_alive)
    return sum(t.values()), t, progress_frac


def gait_reward(data, cfg, control_dt, command, air_time, contact_time, grounded_prev,
                 prev_toe_xy, foot_gids_arr, toe_r, foot_contact_slot, contact_acc, progress_frac,
                 terms):
    dt = control_dt
    toe_pos = data.geom_xpos[foot_gids_arr]
    heights = toe_pos[:, 2] - toe_r
    grounded = contact_acc | (heights < cfg.grounded_h)
    grounded_recent = grounded | grounded_prev

    slip_v = jnp.linalg.norm(toe_pos[:, 0:2] - prev_toe_xy, axis=1) / dt
    slip = jnp.where(grounded & grounded_prev,
                      jnp.maximum(0.0, slip_v - cfg.slip_deadband) ** 2, 0.0)
    terms["foot_slip"] = -jnp.minimum(cfg.w_foot_slip * jnp.sum(slip), cfg.penalty_term_cap)

    cmd_speed = cfg.v_ceiling if cfg.speed_mode else jnp.abs(command[0]) * cfg.vx_max
    gait_on = cmd_speed >= cfg.gait_cmd_gate

    # touchdown credit computed BEFORE clocks advance (landing step isn't counted as swing),
    # then clocks advance -- per-foot, vectorized equivalent of the CPU per-foot python loop.
    touchdown_credit = jnp.where(
        grounded & (air_time > 0) & gait_on,
        cfg.w_air_time * jnp.clip(air_time - cfg.foot_air_time_min, 0.0, cfg.air_credit_cap_s),
        0.0)
    air = jnp.sum(touchdown_credit)
    terms["air_time"] = air
    new_air_time = jnp.where(grounded, 0.0, air_time + dt)
    new_contact_time = jnp.where(grounded, contact_time + dt, 0.0)

    cap = jnp.where(cmd_speed >= cfg.stance_slow_speed, cfg.stance_cap_s, cfg.stance_cap_slow_s)
    over = jnp.clip(new_contact_time - cap, 0.0, 1.0)
    stance_pen = -jnp.minimum(cfg.w_stance_time * jnp.sum(over), cfg.penalty_term_cap)
    terms["stance_time"] = jnp.where(gait_on, stance_pen, 0.0)

    fresh_swing = (~grounded_recent) & (new_air_time > 0.0) & (new_air_time <= cfg.swing_fresh_s)
    clear_frac = jnp.clip((heights - cfg.clearance_dead_m) / cfg.clearance_scale_m, 0.0, 1.0)
    clear_term = jnp.where(fresh_swing, cfg.w_clearance * clear_frac * (0.3 + 0.7 * progress_frac), 0.0)
    clear = jnp.where(gait_on, jnp.sum(clear_term), 0.0)
    terms["clearance"] = clear

    total = terms["foot_slip"] + air + terms["stance_time"] + clear
    return total, terms, new_air_time, new_contact_time, grounded, toe_pos[:, 0:2]


def fallen(data, cfg, base_id, floor_pairs):
    bad = ~jnp.all(jnp.isfinite(data.qpos))
    bad = bad | (data.qpos[2] < cfg.term_height)
    bad = bad | (gravity_body(data, base_id)[2] > cfg.term_gravity_z)
    bad = bad | floor_violation(data, floor_pairs)
    return bad


class Dash01MjxEnv:
    """Bundles the static model/config; reset()/step() are pure functions of (state, ...) ->
    state, meant to be jax.vmap'd across parallel envs and jax.jit'd by the caller (see
    rl/mjx_train.py). No brax.envs.base.Env / wrapper dependency -- see the module docstring for
    why (Brax's stock AutoResetWrapper doesn't re-randomize the reset pose)."""

    def __init__(self, cfg: Config = None):
        self.cfg = cfg = cfg or Config()
        if getattr(cfg, "action_mode", "pd") != "pd":
            raise NotImplementedError(
                "Dash01MjxEnv only supports action_mode='pd'. The fourier per-cycle gait policy is "
                "CPU-only for now (variable-length cycle scan + per-cycle macro-step is deferred in "
                "MJX, like the GPU ride-height randomization). Train fourier presets with rl.train.")
        mj_model = mujoco.MjModel.from_xml_path(cfg.model_path)
        mj_data = mujoco.MjData(mj_model)
        self.sim_dt = float(mj_model.opt.timestep)
        self.control_dt = self.sim_dt * cfg.control_decimation
        self.max_steps = int(round(cfg.episode_s / self.control_dt))
        self._resample_every = max(1, int(round(cfg.cmd_resample_s / self.control_dt)))

        self.nu = mj_model.nu
        act_qadr, act_dadr = [], []
        for a in range(self.nu):
            jid = mj_model.actuator_trnid[a, 0]
            act_qadr.append(int(mj_model.jnt_qposadr[jid]))
            act_dadr.append(int(mj_model.jnt_dofadr[jid]))
        self.act_qadr = jnp.array(act_qadr)
        self.act_dadr = jnp.array(act_dadr)

        self.base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "bodyNCS-v1")
        gyro_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.gyro_adr = int(mj_model.sensor_adr[gyro_sid])

        self.floor_gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_gids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_col")
                          for s in "LR"]
        self.foot_gids_arr = jnp.array(self.foot_gids)
        self.toe_r = float(mj_model.geom_size[self.foot_gids[0]][0])
        col_gids = {}
        for s in "LR":
            for kind in ("foot", "heel"):
                g = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"{kind}_{s}_col")
                if g >= 0:
                    col_gids[g] = float(mj_model.geom_size[g][0])

        self.key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, cfg.keyframe)
        self.default_qpos = jnp.array(mj_model.key_qpos[self.key_id])
        self.nominal_ctrl = jnp.array(mj_model.key_ctrl[self.key_id])
        self.default_motor_pos = self.default_qpos[self.act_qadr]
        self.ctrl_lo = jnp.array(mj_model.actuator_ctrlrange[:, 0])
        self.ctrl_hi = jnp.array(mj_model.actuator_ctrlrange[:, 1])
        self.height_target = float(mj_model.key_qpos[self.key_id][2])

        mujoco.mj_resetDataKeyframe(mj_model, mj_data, self.key_id)
        mj_data.ctrl[:] = mj_model.key_ctrl[self.key_id]
        mujoco.mj_forward(mj_model, mj_data)
        self.stand_torque = jnp.array(mj_data.actuator_force[:self.nu])

        # template Data built via mjx.put_data from a real, forward-computed CPU MjData -- the
        # exact construction path validated in mujoco/dash01/validate_mjx.py (Phase 0). Not
        # mjx.make_data(): that's a different, unvalidated init path.
        self.data_template = mjx.put_data(mj_model, mj_data)

        hip_roll_idx = [a for a in range(self.nu)
                        if (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "")
                        .startswith("hip_roll")]
        self.hip_roll_idx = jnp.array(hip_roll_idx, dtype=jnp.int32)

        # base-DOF locks (mirror of rl/env.py): activate cfg.base_lock's subset via a STATIC
        # eq_active vector applied at reset. The mask is constant per run, so no per-env / batched-
        # model plumbing is needed. base_z's lock target (its neutral height) is baked into the
        # model's eq_data at the natural stance height. NOTE: the CPU env randomizes the M1
        # ride-height per episode; here it is fixed (per-episode Z randomization on GPU is a deferred
        # batched-eq_data refactor). Leg hinges start after the 6 base joints (qpos[6:]).
        self.base_lock = tuple(int(x) for x in cfg.base_lock)
        lock_eq_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, f"lock_{n}")
                       for n in ("x", "y", "z", "roll", "pitch", "yaw")]
        eq_active0 = mj_model.eq_active0.copy()
        for eid, lk in zip(lock_eq_ids, self.base_lock):
            eq_active0[eid] = lk
        # match the template's eq_active dtype (also fails fast here if this MJX build lacks the
        # Data.eq_active field -- the whole runtime-lock scheme depends on it, MJX >= 3.2.5).
        self.eq_active_vec = jnp.asarray(eq_active0, dtype=self.data_template.eq_active.dtype)
        if self.base_lock[2]:
            mj_model.eq_data[lock_eq_ids[2], 0] = self.height_target
        self.hinge_qadr_start = int(min(
            mj_model.jnt_qposadr[j] for j in range(mj_model.njnt)
            if mj_model.jnt_bodyid[j] != self.base_id))

        self.mjx_model = mjx.put_model(mj_model)

        # one-off (non-jitted) probe of MJX's static contact-pair layout for this model: the
        # candidate geom-pair SET and ORDER are a model-level property (broadphase filtering
        # done once at put_model time), identical for every mjx.Data derived from this model --
        # see mujoco/dash01/validate_mjx.py's introspection for how this was confirmed.
        probe = mjx.forward(self.mjx_model, self.data_template)
        pair_geoms = np.asarray(probe.contact.geom)

        def find_pair(g1, g2):
            for i, (a, b) in enumerate(pair_geoms):
                if {int(a), int(b)} == {g1, g2}:
                    return i
            raise ValueError(f"no MJX contact pair for geoms {g1},{g2} -- model changed?")

        self.foot_contact_slot = jnp.array([find_pair(self.floor_gid, fg) for fg in self.foot_gids])
        self.floor_pairs = [(find_pair(self.floor_gid, g), r) for g, r in col_gids.items()]

        self.obs_size = FRAME_DIM * cfg.history_len
        self.action_size = self.nu

    # ---------- reset ----------
    def reset(self, rng, cmd_vx_frac=0.0):
        cfg = self.cfg
        rng, k_noise, k_push, k_cmd = jax.random.split(rng, 4)

        h = self.hinge_qadr_start
        qpos = self.default_qpos.at[h:].add(
            jax.random.uniform(k_noise, (self.default_qpos.shape[0] - h,),
                               minval=-cfg.reset_joint_noise, maxval=cfg.reset_joint_noise))
        # activate the milestone's base-DOF locks (constant per run -> static vector)
        data = self.data_template.replace(
            qpos=qpos, qvel=jnp.zeros_like(self.data_template.qvel), ctrl=self.nominal_ctrl,
            eq_active=self.eq_active_vec)
        data = mjx.forward(self.mjx_model, data)

        command = sample_command(k_cmd, cfg, jnp.asarray(cmd_vx_frac))
        des_xy, des_yaw = anchor_command_pose(data, self.base_id)
        contact0 = foot_contacts(data, self.foot_contact_slot)
        heights0 = toe_heights(data, self.foot_gids_arr, self.toe_r)
        grounded0 = grounded_fn(contact0, heights0, cfg.grounded_h)
        prev_action = jnp.zeros(self.nu)

        frame = proprio(data, cfg, self.act_qadr, self.act_dadr, self.default_motor_pos, self.nu,
                        self.base_id, self.gyro_adr, prev_action, command)
        history = jnp.tile(frame, (cfg.history_len, 1))

        zero_terms = {k: jnp.asarray(0.0) for k in TERM_NAMES}
        return EnvState(
            data=data, obs=history.reshape(-1), reward=jnp.asarray(0.0), done=jnp.asarray(False),
            terminated=jnp.asarray(False), truncated=jnp.asarray(False),
            reward_terms=zero_terms, ep_return=jnp.asarray(0.0),
            ep_length=jnp.asarray(0, dtype=jnp.int32),
            final_ep_return=jnp.asarray(0.0), final_ep_length=jnp.asarray(0, dtype=jnp.int32),
            step=jnp.asarray(0, dtype=jnp.int32), history=history,
            prev_action=prev_action, filt_target=self.nominal_ctrl,
            delay_buf=jnp.zeros((cfg.action_delay_steps, self.nu)),
            air_time=jnp.zeros(2), contact_time=jnp.zeros(2), grounded_prev=grounded0,
            prev_toe_xy=data.geom_xpos[self.foot_gids_arr, 0:2], push_countdown=next_push_in(k_push, cfg, self.control_dt),
            command=command, des_xy=des_xy, des_yaw=jnp.asarray(des_yaw),
            cmd_vx_frac=jnp.asarray(cmd_vx_frac), rng=rng,
        )

    # ---------- step ----------
    def step(self, state, action):
        cfg = self.cfg
        action = jnp.clip(action, -1.0, 1.0)
        rng, k_push_dir, k_push_next, k_cmd, k_reset = jax.random.split(state.rng, 5)

        if cfg.action_delay_steps == 0:
            applied, delay_buf = action, state.delay_buf
        else:
            applied = state.delay_buf[0]
            delay_buf = jnp.concatenate([state.delay_buf[1:], action[None]], axis=0)

        raw_target = self.nominal_ctrl + cfg.action_scale * applied
        filt_target = cfg.action_filter * state.filt_target + (1 - cfg.action_filter) * raw_target
        ctrl = jnp.clip(filt_target, self.ctrl_lo, self.ctrl_hi)

        do_push = (state.push_countdown - 1) <= 0
        ang = jax.random.uniform(k_push_dir, minval=0.0, maxval=2 * jnp.pi)
        free_xy = jnp.array([1.0 - self.base_lock[0], 1.0 - self.base_lock[1]])   # only shove free axes
        push_dv = (jnp.where(do_push, cfg.push_dv, 0.0)
                   * jnp.array([jnp.cos(ang), jnp.sin(ang)]) * free_xy)
        data = state.data.replace(
            qvel=state.data.qvel.at[0:2].add(push_dv), ctrl=ctrl)
        new_countdown = jnp.where(do_push, next_push_in(k_push_next, cfg, self.control_dt),
                                   state.push_countdown - 1)

        def substep(carry, _):
            d, acc = carry
            d = mjx.step(self.mjx_model, d)
            acc = acc | foot_contacts(d, self.foot_contact_slot)
            return (d, acc), None

        (data, contact_acc), _ = jax.lax.scan(
            substep, (data, jnp.zeros(2, dtype=bool)), None, length=cfg.control_decimation)

        new_step = state.step + 1
        do_resample = (new_step % self._resample_every) == 0
        resampled_cmd = sample_command(k_cmd, cfg, state.cmd_vx_frac)
        resampled_xy, resampled_yaw = anchor_command_pose(data, self.base_id)
        advanced_xy, advanced_yaw = advance_command_pose(state.des_xy, state.des_yaw,
                                                         state.command, cfg, self.control_dt)
        command = jnp.where(do_resample, resampled_cmd, state.command)
        des_xy = jnp.where(do_resample, resampled_xy, advanced_xy)
        des_yaw = jnp.where(do_resample, resampled_yaw, advanced_yaw)
        air_time_in = jnp.where(do_resample, 0.0, state.air_time)
        contact_time_in = jnp.where(do_resample, 0.0, state.contact_time)

        frame = proprio(data, cfg, self.act_qadr, self.act_dadr, self.default_motor_pos, self.nu,
                        self.base_id, self.gyro_adr, state.prev_action, command)
        history = jnp.concatenate([state.history[1:], frame[None]], axis=0)

        reward, terms, progress_frac = reward_fn(
            data, cfg, action, state.prev_action, command, des_xy, des_yaw, self.base_id,
            self.gyro_adr, self.act_qadr, self.hip_roll_idx, self.default_motor_pos,
            self.height_target, self.stand_torque, self.nu, self.foot_gids)
        gait_total, terms, air_time, contact_time, grounded, toe_xy = gait_reward(
            data, cfg, self.control_dt, command, air_time_in, contact_time_in,
            state.grounded_prev, state.prev_toe_xy, self.foot_gids_arr, self.toe_r,
            self.foot_contact_slot, contact_acc, progress_frac, terms)
        reward = reward + gait_total

        terminated = fallen(data, cfg, self.base_id, self.floor_pairs)
        reward = reward - jnp.where(terminated, cfg.fall_penalty, 0.0)
        truncated = new_step >= self.max_steps
        done = terminated | truncated

        ep_return = state.ep_return + reward
        ep_length = state.ep_length + 1

        final = EnvState(
            data=data, obs=history.reshape(-1), reward=reward, done=done,
            terminated=terminated, truncated=truncated, reward_terms=terms,
            ep_return=ep_return, ep_length=ep_length,
            final_ep_return=ep_return, final_ep_length=ep_length,
            step=new_step, history=history, prev_action=action, filt_target=filt_target,
            delay_buf=delay_buf, air_time=air_time, contact_time=contact_time,
            grounded_prev=grounded, prev_toe_xy=toe_xy, push_countdown=new_countdown,
            command=command, des_xy=des_xy, des_yaw=jnp.asarray(des_yaw),
            cmd_vx_frac=state.cmd_vx_frac, rng=rng,
        )
        # NOTE: ep_return/ep_length are deliberately NOT in this protect-list -- they're carried
        # state and must follow the normal reset blend (0/0 on done) so the next episode counts
        # from zero. Only final_ep_return/final_ep_length (read for logging) are protected.
        reset_state = self.reset(k_reset, state.cmd_vx_frac).replace(
            reward=reward, done=done, terminated=terminated, truncated=truncated,
            reward_terms=terms, final_ep_return=ep_return, final_ep_length=ep_length)
        blended = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), reset_state, final)
        return blended
