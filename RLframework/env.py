"""DashEnvV2 — the DASH-01 Walker v2 environment on MJX, one jitted step for N envs.

One env.step == one 10 ms control tick == 10 substeps of 1 kHz MJX physics. Everything the
artifact specifies for the tick lives here, in the order of Fig. 1:

    action (50) -> latch register (spec commits only on the commit tick) -> gait generator
    (gait.py) -> drive (drive.py: kp(phi)/kd(phi) PD, torque-speed clamp, 6-18 ms delay at
    substep granularity, thermal node) -> plant (mjx, per-env randomized fields) -> sensor
    model (noise, bias, staleness) -> frame 34 -> history 10 x stride 2 + once-block +
    privileged tail (train only).

State is a flax struct; reset/step are pure functions of (state, action, params) and are
vmapped over the batch by `DashEnvV2`. Auto-reset: an env that ends is reset inside the same
step and the returned obs is the new episode's first. `EnvParams` carries the curriculum
values the trainer moves between rollouts (dr_scale, sprint line, stance ratio, ...).

Observation layouts (actor slice first, privileged tail last):
    policy variant : 340 history + 44 latched spec + 2 task + 1 commit = 387 | + 25 = 412
    library variant: 340 history + 23 once-block                       = 363 | + 25 = 388
The commit flag is an explicit once-block entry (artifact §05, rev 2026-09-09). Under v2 it had to
be, because a touchdown resync could move the wrap; with the resync off (v3) the flag is inferable
from the phase, but it stays explicit so the width is stable across both and because the log-prob
mask reads the same flag (info["commit"]) so rollout and update agree bit for bit.
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from flax import struct
from mujoco import mjx

import gait
import drive
from gait import GaitParams
from plant import Plant, PlantDraw, PlantFields, Override, draw_plant, model_with

FRAME_DIM = 34            # the actor's per-tick frame, incl. the integrated-gyro heading estimate
# The critic's privileged tail: 25 as before, plus the two sole pads per foot.  The actor's frame
# is deliberately NOT widened by them -- DASH-01 has no foot contact sensing, so a contact bit in
# the actor's input would be a sensor the robot does not have.
PRIV_DIM = 29
TASK_DIM = 2
TWO_PI = 2.0 * np.pi


def heading_ema(prev, yaw_est, a, cap):
    """The heading error v4 bills: a SIGNED moving average (a = exp(-dt / heading_avg_s)) of the
    robot's own integrated-gyro heading, the channel the actor reads -- so the set point is always
    one it can see and hold. Signed, so a stride's yaw wobble and the start-up transient average
    out and cost little, while a held offset reaches the bill in full within a few time constants.
    Clipped like the observation (a sideways robot has failed either way)."""
    return a * prev + (1.0 - a) * jnp.clip(gait.wrap_pi(yaw_est), -cap, cap)


def joystick_income(vx, v_cmd, sigma, w_track, w_fwd, w_speed, v_ceiling):
    """The joystick's income: a Laplace kernel on the tracking error PLUS a monotone term in the
    speed actually achieved, capped by what was commanded.

    The kernel alone is flat where the policy lives. Measured on v4_one_s1's keeper: held at a
    reachable 1.8 m/s it earns 2.74/tick, held at 3.58 m/s (which it misses by 1.7) it earns 0.31
    against 1.81 of cost -- the reward floor, where no behaviour changes the return, so nothing pays
    for the next 0.1 m/s while the slip and contact penalties still charge for it. The second term
    is w * clip(vx, 0, v_cmd) / v_ceiling: always increasing in achieved speed, so there is a
    gradient uphill at any command, and capped at the command so under-stick behaviour is unchanged
    (running faster than asked still pays less, via the kernel)."""
    kernel = ((w_track + w_fwd * jnp.clip(v_cmd, 0.0, v_ceiling))
              * jnp.exp(-jnp.abs(vx - v_cmd) / jnp.maximum(sigma, 1e-3)))
    return kernel + w_speed * jnp.clip(vx, 0.0, v_cmd) / v_ceiling


def heading_rate(w, g, euler):
    """The rate integrated into the robot's heading estimate, from the MEASURED gyro w and gravity g
    (body frame; g = R^T [0, 0, -1] = [sin p, -sin r cos p, -cos r cos p]).

    Body gyro z is the heading rate only while the body is level. Measured on v4_nolane_s1 at
    0.5-1.6 m/s (pitch -2 deg, roll sd 2 deg): the integrated gyro z drifted 10 deg from the true
    heading in 15 s, identical with and without sensor noise, while the ZYX Euler rate
    (w_y sin r + w_z cos r) / cos p stayed within 0.8 deg -- geometry, not noise. euler=False is the
    v3 estimate. controller/deploy/controller_v2.py computes the same thing in numpy."""
    if not euler:
        return w[2]
    g = g / jnp.maximum(jnp.linalg.norm(g), 1e-6)
    sp = jnp.clip(g[0], -0.95, 0.95)
    roll = jnp.arctan2(-g[1], -g[2])
    return (w[1] * jnp.sin(roll) + w[2] * jnp.cos(roll)) / jnp.sqrt(1.0 - sp * sp)


class EnvParams(NamedTuple):
    """Curriculum / runtime values, traced. Defaults = the FINAL (hardest) curriculum point."""
    dr_scale: float = 1.0
    # THE SURVIVAL BONUS IS A CURRICULUM, NOT A CONSTANT.
    #
    # w_alive has two jobs that pull opposite ways. Early it is the ONLY reachable reward -- a cold
    # policy earns 0.051/tick of income, so without it LIVING clamps to the step floor and living
    # ties with dying (measured 2026-09-17; removing it outright gave alive_frac 0.19/0.30/0.00 and
    # froze every curriculum at progress 0.00). Late it is a flat subsidy worth 121% of a stander's
    # whole income and only 21% of a walker's, which compresses walking's 5.8x income advantage to
    # 3.1x and pays for the standing basin the first campaign settled into.
    #
    # So: full weight while the policy learns to balance, then weaned off so income has to come
    # from the task. 1.0 = cfg.w_alive, and it decays toward cfg.alive_scale_final.
    alive_scale: float = 1.0
    sprint_dist_m: float = 100.0
    stance_ratio: float = 0.42
    eff_scale: float = 1.0
    ctrl_jitter_ms: float = 0.0
    ctrl_drop_prob: float = 0.0
    bringup_scale: float = 1.0   # 0 = mild starts, 1 = the full drop / tilt bands
    bringup_p_drop: float = -1.0 # share of episodes DROPPED (<0 = cfg.bringup_drop_frac * scale)
    bringup_p_held: float = -1.0 # share HELD-misaligned then released (<0 = cfg * scale); the eval
                                 # pins these to force every episode to start dirty
    cmd_zero_p: float = 0.25     # share of command draws that are exactly zero (ramped)
    cmd_stop_p: float = 0.0      # share of command draws that are STOP (cfg.stop_flag; ramped)
    cmd_lo: float = 0.0          # joystick: fraction-of-v_max band the command is drawn from
    cmd_hi: float = 1.0          # (curriculum widens it down from cmd_range_start to cmd_range)
    track_sigma: float = 0.6     # Laplace width of the tracking income, m/s (annealed from wider)
    shape_scale: float = 1.0     # weight on the gait-quality penalties (ramped from shape_scale_start)
    stoplight_prob: float = 0.0
    gait_freq_lo: float = 0.0        # curriculum lower rail of the gait clock (0 = the config value)
    hold_s: float = 0.0          # bring-up probe (cfg.hold_enable): seconds the base is held
    hold_z: float = 0.0          # base height while held (<= 0: the keyframe height)
    hold_pitch: float = 0.0      # base pitch while held (rad, + = nose down)
    hold_roll: float = 0.0       # base roll while held (rad)
    start_red_s: float = 0.0     # >0: episode starts on a RED light (task[0]=0, the deployed
                                 # runtime's STOPPED bring-up) and turns green after this long
    release_vx: float = 0.0      # base velocity added on the release tick (m/s): the operator's
    release_vy: float = 0.0      # hand is not a clean let-go -- fore-aft, lateral and vertical
    release_vz: float = 0.0      # (negative = still moving the robot down as it is released)

    @classmethod
    def final(cls, cfg):
        return cls(dr_scale=float(getattr(cfg, "dr_scale_final", 1.0)), sprint_dist_m=float(cfg.sprint_dist_m),
                   alive_scale=float(getattr(cfg, "alive_scale_final", 1.0)),
                   cmd_zero_p=float(cfg.cmd_zero_frac),
                   cmd_stop_p=(float(getattr(cfg, "cmd_stop_frac", 0.0))
                               if getattr(cfg, "stop_flag", False) else 0.0),
                   cmd_lo=float(cfg.cmd_range[0]), cmd_hi=float(cfg.cmd_range[1]),
                   stance_ratio=float(cfg.stance_ratio_final), eff_scale=float(cfg.efficiency_target),
                   ctrl_jitter_ms=float(cfg.ctrl_jitter_ms_final),
                   ctrl_drop_prob=float(cfg.ctrl_drop_prob_final), stoplight_prob=0.0,
                   gait_freq_lo=float(cfg.gait_freq_hz[0]), track_sigma=float(cfg.track_sigma),
                   shape_scale=1.0)


@struct.dataclass
class EnvState:
    data: mjx.Data
    draw: PlantDraw
    key: jnp.ndarray
    step_n: jnp.ndarray
    t: jnp.ndarray
    # the clock and the latch
    phase: jnp.ndarray
    spec: jnp.ndarray            # (44,) live latched spec
    commit: jnp.ndarray          # bool: THIS tick's action latches the spec (in the once-block)
    cycle_n: jnp.ndarray
    phi_td_hat: jnp.ndarray      # (2,) expected touchdown phase per foot
    resynced: jnp.ndarray        # (2,) bool, one resync per foot per cycle
    # drive
    cmd_buf: jnp.ndarray         # (3, 18) [target6 kp6 kd6] current, previous, two back
    prev_target: jnp.ndarray     # (6,)
    prev_target_vel: jnp.ndarray
    prev_action: jnp.ndarray
    prev_residual: jnp.ndarray
    prev_motor_cmd: jnp.ndarray
    thermal_x: jnp.ndarray       # (6,)
    # signals
    prev_vel_body: jnp.ndarray
    lp_yaw_true: jnp.ndarray
    lp_yaw_obs: jnp.ndarray
    yaw_est: jnp.ndarray         # v3: heading from INTEGRATED gyro z -- the robot's own estimate
    heading_avg: jnp.ndarray     # v4: yaw_est averaged over heading_avg_s -- what heading bills when on
    roll_lp: jnp.ndarray
    hist: jnp.ndarray            # (hist_raw_len, FRAME_DIM)
    gyro_bias: jnp.ndarray
    stale_left: jnp.ndarray
    stale_frame: jnp.ndarray     # (6,) held gravity + gyro
    # reward bookkeeping
    air_time: jnp.ndarray
    contact_time: jnp.ndarray
    grounded_prev: jnp.ndarray
    prev_toe_xy: jnp.ndarray
    duty_ema: jnp.ndarray
    ws_out_t: jnp.ndarray
    swing_ema: jnp.ndarray
    push_countdown: jnp.ndarray
    trip_left: jnp.ndarray
    trip_foot: jnp.ndarray
    trip_force: jnp.ndarray
    gust_left: jnp.ndarray
    gust_countdown: jnp.ndarray
    gust_dir: jnp.ndarray
    # sprint
    x0: jnp.ndarray
    sprint_d: jnp.ndarray
    crossed: jnp.ndarray
    t_line: jnp.ndarray
    stop_hold: jnp.ndarray
    light_red: jnp.ndarray       # stop curriculum: red phase active (before the line)
    light_left: jnp.ndarray      # seconds left in the current light phase (inf = no lights this episode)
    light_v0: jnp.ndarray        # forward speed when the last red / the line started (decel target start)
    light_t: jnp.ndarray         # seconds since that switch
    light_floor: jnp.ndarray     # speed the ramp descends TO (0 = stop, > 0 = amber, a slow run)
    # joystick (objective='joystick'): the operator's commanded speed, redrawn on a timer so the
    # policy sees the command MOVE from step 0 of training -- the v2 failure was a command pinned
    # at 1 for a whole run and then flipped, which arrives as a disturbance, not as an input
    hold_s: jnp.ndarray          # per-episode seconds the operator's hand stays on (0 = none)
    grace_left: jnp.ndarray      # seconds of fall-termination grace left (a drop needs to land)
    v_cmd: jnp.ndarray           # m/s, absolute; task[0] carries v_cmd / v_max
    cmd_left: jnp.ndarray        # seconds until the command is redrawn
    stop_cmd: jnp.ndarray        # 1.0 = the operator's switch is on STOP (cfg.stop_flag); task[1] = 1 - this
    # library variant
    theta: jnp.ndarray           # (44,) this episode's gait (box-perturbed library entry)
    v_ref: jnp.ndarray
    raibert_i: jnp.ndarray
    # episode stats
    ep_return: jnp.ndarray
    ep_len: jnp.ndarray


def _impl(d):
    return getattr(d, "_impl", d)


class DashEnvV2:
    """Batched env. Construct once; `reset(keys, params)` and `step(state, action, params)`
    are jitted and vmapped over the leading axis."""

    def __init__(self, cfg, n_envs=None):
        self.cfg = cfg
        self.plant = Plant(cfg)
        self.gp = GaitParams.from_cfg(cfg)
        self.n_envs = int(n_envs or cfg.n_envs)
        self.control_dt = self.plant.control_dt
        self.max_steps = int(round(cfg.episode_s / self.control_dt))
        self.reward_dt_scale = self.control_dt / 0.02      # walk_mit's rate invariance
        tau_h = getattr(cfg, "heading_avg_s", 0.0)
        self._heading_a = float(np.exp(-self.control_dt / tau_h)) if tau_h > 0 else 0.0
        self.hist_stride = int(cfg.history_stride)
        self.hist_raw_len = (cfg.history_len - 1) * self.hist_stride + 1
        self.hist_idx = np.array((self.hist_raw_len - 1)
                                 - (np.arange(cfg.history_len) * self.hist_stride)[::-1])
        self._touch = None
        if cfg.bringup_enable:
            # base z with the lowest toe on the floor, per (pitch, roll). Solved offline by
            # tools/make_touch_table.py because it is a MuJoCo root-find and the reset is jitted.
            tt = Path(__file__).resolve().parent / "model" / "touch_height.npz"
            if not tt.exists():
                raise FileNotFoundError(f"bringup_enable needs {tt}; run tools/make_touch_table.py")
            z = np.load(tt)
            self._touch = (jnp.asarray(z["pitch_rad"]), jnp.asarray(z["roll_rad"]), jnp.asarray(z["z"]))
        self.library_mode = (cfg.spec_source == "library")
        self.action_dim = gait.LIB_ACTION_DIM if self.library_mode else gait.ACTION_DIM
        self.once_dim = 23 if self.library_mode else (gait.SPEC_DIM + TASK_DIM + 1)
        self.actor_dim = FRAME_DIM * cfg.history_len + self.once_dim
        self.obs_dim = self.actor_dim + PRIV_DIM
        self.priv_dim = PRIV_DIM
        # log-prob mask template: which action dims are latched (scored on commit ticks only)
        mask = np.ones(self.action_dim, bool)
        if self.library_mode:
            mask[gait.LIB_N_RESIDUAL:] = False
        else:
            mask[:gait.SPEC_DIM] = False
        self.latched_dims = ~mask
        self.wrap_index = self.actor_dim - 1          # commit flag = last once-block entry
        self._library = self._load_library()
        # jitted, batched entry points
        self._reset_v = jax.jit(jax.vmap(self._reset_one, in_axes=(0, None, 0)))
        self._step_v = jax.jit(jax.vmap(self._step_one, in_axes=(0, 0, None)))

    # ------------------------------------------------------------------ library
    def _load_library(self):
        """Library entries for spec_source='library': theta (44), v* (m/s), sorted by speed.
        Missing file -> one neutral entry (theta 0 = the mirror-symmetric mid-range gait)."""
        import json
        from pathlib import Path
        p = Path(self.cfg.library_path)
        p = p if p.exists() else Path(__file__).resolve().parent / self.cfg.library_path
        if not self.library_mode or not p.exists():
            return dict(theta=np.zeros((1, gait.SPEC_DIM), np.float32), v=np.zeros(1, np.float32))
        d = json.loads(p.read_text())
        ent = sorted(d["entries"], key=lambda e: e["v_star"])
        return dict(theta=np.array([e["theta"] for e in ent], np.float32),
                    v=np.array([e["v_star"] for e in ent], np.float32))

    # ------------------------------------------------------------------ helpers
    def _base_rot(self, data):
        return data.xmat[self.plant.base_bid].reshape(3, 3)

    def _grav_body(self, data):
        R = self._base_rot(data)
        return R.T @ jnp.array([0.0, 0.0, -1.0])

    def _gyro(self, data):
        a = self.plant.gyro_adr
        return data.sensordata[a:a + 3]

    def _vel_world(self, data):
        p = self.plant
        vx = data.qvel[p.base_d["x"]]
        vy = data.qvel[p.base_d["y"]] if not p.planar else 0.0 * vx
        vz = data.qvel[p.base_d["z"]]
        return jnp.stack([vx, vy, vz])

    def _vel_body(self, data):
        return self._base_rot(data).T @ self._vel_world(data)

    def _base_pos(self, data):
        return data.xpos[self.plant.base_bid]

    def _yaw(self, data):
        p = self.plant
        return data.qpos[p.base_q["yaw"]] if not p.planar else 0.0 * data.qpos[0]

    def _y(self, data):
        p = self.plant
        return data.qpos[p.base_q["y"]] if not p.planar else 0.0 * data.qpos[0]

    def _contacts(self, data):
        """(toe_down[2], heel_down[2], normal_force[2], floor_violation) from the MJX contact array.

        The sole is a 3 mm pad split into a front and a rear box, so each foot reports TWO contact
        bits instead of one.  That distinction is what a flat foot buys and the point-foot robot
        this lineage started from could not express: toe-down, heel-down and flat are different
        states of the same foot, and the gait is billed on which one it is in.

        A penetration past `floor_viol_m` is a floor violation: the solver has lost the contact
        and the foot is inside the ground.  It is deliberately much larger than the pad -- a soft
        contact legitimately sinks a millimetre or two under load, and judging it against half the
        3 mm pad terminated 17 of 60 ticks on a robot that was simply standing."""
        con = _impl(data).contact
        p, c = self.plant, self.cfg
        g, dist = con.geom, con.dist
        efc = _impl(data).efc_force
        adr = con.efc_address
        fn_all = jnp.where(adr >= 0, efc[jnp.maximum(adr, 0)], 0.0)
        bite = c.floor_viol_m

        def pad(gid):
            pair = ((g[:, 0] == p.floor_gid) & (g[:, 1] == gid)) |                    ((g[:, 1] == p.floor_gid) & (g[:, 0] == gid))
            touching = pair & (dist < 0.0)
            fn = jnp.sum(jnp.where(touching, jnp.maximum(fn_all, 0.0), 0.0))
            return touching.any(), fn, (touching & (dist < -bite)).any()

        toe, heel, force, viol = [], [], [], jnp.zeros((), bool)
        for i in range(2):
            t_on, t_f, t_v = pad(p.foot_gids[i])
            h_on, h_f, h_v = pad(p.heel_gids[i])
            toe.append(t_on)
            heel.append(h_on)
            force.append(t_f + h_f)
            viol = viol | t_v | h_v
        return jnp.stack(toe), jnp.stack(heel), jnp.stack(force), viol

    def _pad_heights(self, data, gids, sizes):
        """Height of each pad's lowest corner above the floor.

        A box's support below its origin is |R[2, :]| . half_extents, which depends on the foot's
        orientation -- it is not a radius.  Using the geom's first half-extent instead (what a
        sphere model would do) reads the pad's half LENGTH, twenty times too large."""
        R = data.geom_xmat[gids].reshape(-1, 3, 3)
        drop = jnp.sum(jnp.abs(R[:, 2, :]) * jnp.asarray(sizes), axis=-1)
        return data.geom_xpos[gids, 2] - drop

    def _toe_heights(self, data):
        return self._pad_heights(data, self.plant.foot_gids, self.plant.foot_size)

    def _foot_heights(self, data):
        """Lowest point of each foot: whichever of its two pads is nearer the ground."""
        return jnp.minimum(self._toe_heights(data),
                           self._pad_heights(data, self.plant.heel_gids, self.plant.heel_size))

    def _toe_pos(self, data):
        return data.geom_xpos[self.plant.foot_gids]

    def _foot_sep(self, data):
        R = self._base_rot(data)
        base = self._base_pos(data)
        y = [(R.T @ (data.geom_xpos[g] - base))[1] for g in self.plant.foot_gids]
        return y[0] - y[1]

    def _angmom_pitch(self, mx_i, data):
        try:
            d2 = mjx.subtree_vel(mx_i, data)
            L = _impl(d2).subtree_angmom[self.plant.base_bid]
            return L[1]
        except Exception:               # pragma: no cover - field layout drift
            return jnp.zeros(())

    def _workspace_out(self, data):
        p, c = self.plant, self.cfg
        R = self._base_rot(data)
        base = self._base_pos(data)
        outs = []
        for i, g in enumerate(p.foot_gids):
            tb = R.T @ (data.geom_xpos[g] - base)
            dx = tb[0] - p.ws_ref[i, 0]
            dz = tb[2] - p.ws_ref[i, 2]
            outs.append((jnp.abs(dx) > c.workspace_dx_max) | (dz > c.workspace_dz_max)
                        | (dz < c.workspace_dz_min))
        return jnp.stack(outs)

    # ------------------------------------------------------------------ observation
    def _frame(self, data, tau_last, residual, phase_next, state, key):
        """One 34-dim per-tick frame through the measurement chain. Returns (frame, new noise
        state pieces, accel_body, v_body)."""
        c, p, s = self.cfg, self.plant, self.cfg.obs_scales
        dr = state.draw
        q = data.qpos[p.act_qadr] - p.default_motor_pos
        qd = data.qvel[p.act_dadr]
        grav = self._grav_body(data)
        gyro = self._gyro(data)
        v_body = self._vel_body(data)
        accel = (v_body - state.prev_vel_body) / self.control_dt
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
        on = 1.0 if c.obs_noise_enable else 0.0
        gyro_bias = dr.gyro_bias0 + state.gyro_bias   # bias0 + random walk
        gyro_bias_walk = state.gyro_bias + on * c.noise_gyro_walk * jax.random.normal(k1, (3,))
        mp = q + dr.enc_offset - dr.joint_zero + on * c.noise_encoder * jax.random.normal(k2, (6,))
        mv = qd + dr.vel_bias + on * c.noise_motor_vel * jax.random.normal(k3, (6,))
        mt = (tau_last * (1.0 + on * c.noise_torque_gain * jax.random.normal(k4, (6,)))
              + dr.trq_bias + on * c.noise_torque * jax.random.normal(k5, (6,)))
        g = grav + dr.grav_bias + on * c.noise_grav * jax.random.normal(k6, (3,))
        g = g - dr.accel_leak * accel / 9.81
        g = g / jnp.maximum(jnp.linalg.norm(g), 1e-6)
        w = gyro + gyro_bias + on * c.noise_gyro * jax.random.normal(k7, (3,))
        g = dr.imu_R @ g
        w = dr.imu_R @ w
        # IMU stale window
        kdrop = jax.random.fold_in(key, 99)
        start = (state.stale_left <= 0) & (jax.random.uniform(kdrop) < c.dr_imu_dropout_prob * on)
        n_stale = int(max(1, round(c.dr_imu_dropout_s / self.control_dt)))
        stale_left = jnp.where(start, n_stale, jnp.maximum(state.stale_left - 1, 0))
        held = state.stale_left > 0
        stale_frame = jnp.where(start, jnp.concatenate([g, w]), state.stale_frame)
        g = jnp.where(held, state.stale_frame[:3], g)
        w = jnp.where(held, state.stale_frame[3:], w)
        lp = float(np.exp(-self.control_dt / c.lp_yaw_tau_s))
        lp_yaw_obs = lp * state.lp_yaw_obs + (1.0 - lp) * w[2]
        # HEADING, and the reason v3 can run straight. v2 gave the actor a low-passed yaw RATE and
        # billed the same rate in the reward, so neither could see accumulated heading: walk_mit
        # measured a 171 m path for 15-38 m of net progress -- a random walk, because a rate
        # controller has no set point. This integrates the NOISY gyro the policy actually reads
        # (w[2], after bias, drift, IMU rotation and the staleness window), so it is the robot's own
        # dead-reckoned heading, not the simulator's. That is affordable on this IMU: measured
        # 0.0097 dps/sqrt(Hz) with a 0.0024 dps bias floor (results/imu_noise.png), i.e. well under a
        # degree of drift over a 30 s episode, so the estimate is honest over any run we ask for.
        # Zero = the direction the operator was facing when they let go.
        # Clipped, not wrapped: +-pi/2 is smooth where we care and saturates once the robot is
        # sideways, which is a failure either way -- a wrapped angle would put a discontinuity in
        # the observation at the worst possible moment.
        yaw_est = state.yaw_est + heading_rate(w, g, getattr(c, "heading_euler", False)) * self.control_dt
        yaw_err = jnp.clip(gait.wrap_pi(yaw_est), -0.5 * np.pi, 0.5 * np.pi)
        frame = jnp.concatenate([
            mp * s["motor_pos"], mv * s["motor_vel"], mt * s["motor_torque"],
            g * s["gravity"], w * s["ang_vel"], jnp.array([lp_yaw_obs * s["ang_vel"]]),
            jnp.array([jnp.cos(phase_next), jnp.sin(phase_next)]), residual,
            jnp.array([yaw_err * s["heading"]]),
        ]).astype(jnp.float32)
        return frame, dict(gyro_bias=gyro_bias_walk, stale_left=stale_left, stale_frame=stale_frame,
                           lp_yaw_obs=lp_yaw_obs, yaw_est=yaw_est), accel, v_body

    def _task(self, state, params):
        if self.cfg.objective == "joystick":
            # THE JOYSTICK. task[0] is the commanded speed as a FRACTION of v_max, so 0.5 asks for
            # half of top speed and 1.0 asks for everything. Nothing here reads sprint_d, so the
            # actor carries no odometry at all -- under v2 task[1] was clip((line - d)/8, 0, 1),
            # computed from ground-truth world x, which the robot can only guess at. task[1] is
            # reserved (a yaw command later) so the width stays 2 and v2 checkpoints remain loadable.
            #
            # It is held at ONE, not zero. Zero is not a neutral filler here: under v2 semantics
            # task[1] is the distance-to-go ramp, 1.0 whenever the line is far away and 0 only to
            # demand a stop. Pinning it at 0 told a warm-started runner to brake on every tick --
            # measured 2026-09-11, the 3.27 m/s runner made 0.1 m/s under a 3.2 m/s command, earned
            # 0.05 of a possible 9.0 of tracking income and fell 100% of the time inside 86 ticks,
            # while the same checkpoint ran 600/600 upright under the sprint preset. 1.0 is what the
            # warm start saw for the whole run phase, i.e. "nothing to brake for".
            # ...unless stop_flag: then it is the RUN/STOP switch, 1 = run, 0 = stop (config.stop_flag).
            run = (1.0 - state.stop_cmd) if self.cfg.stop_flag else 1.0
            return jnp.stack([jnp.clip(state.v_cmd / self.cfg.v_max, 0.0, 1.0), run])
        stop_now = state.crossed | state.light_red
        if self.cfg.stop_cmd_continuous and self.cfg.stop_decel_s > 0:
            # the same ramp the stop reward tracks, recomputed from the state (v0 at the switch, time since)
            f = jnp.maximum(0.0, 1.0 - state.light_t / self.cfg.stop_decel_s)
            v_tgt = state.light_floor + (state.light_v0 - state.light_floor) * f
            run = jnp.where(stop_now, jnp.clip(v_tgt / self.cfg.v_ceiling, 0.0, 1.0), 1.0)
        else:
            run = jnp.where(stop_now, 0.0, 1.0)
        d_to_go = jnp.clip((params.sprint_dist_m - state.sprint_d) / self.cfg.task_brake_m, 0.0, 1.0)
        if self.cfg.objective == "speed":
            return jnp.array([1.0, 1.0])
        return jnp.stack([run, jnp.where(stop_now, 0.0, d_to_go)])

    def _once(self, state, params, phase_next, commit_next):
        if not self.library_mode:
            return jnp.concatenate([state.spec, self._task(state, params), jnp.array([commit_next])])
        p = self.plant
        f = gait.frequency(state.spec[gait.I_FREQ], self.gp)
        q_ref = gait.feedforward(state.spec, phase_next, p.nominal_ctrl, self.gp)
        qd_ref = gait.feedforward_dphi(state.spec, phase_next, self.gp) * TWO_PI * f
        q_ahead = gait.feedforward(state.spec, phase_next + 0.5 * np.pi, p.nominal_ctrl, self.gp)
        td = jnp.stack([gait.wrap_pi(state.phi_td_hat[0]), gait.wrap_pi(state.phi_td_hat[1] - np.pi)]) / np.pi
        return jnp.concatenate([q_ref - p.nominal_ctrl, qd_ref * 0.1, q_ahead - p.nominal_ctrl,
                                td, self._task(state, params), jnp.array([commit_next])])

    def _priv(self, data, state, grounded, fn, accel, v_body, toe=None, heel=None):
        c, p = self.cfg, self.plant
        dr = state.draw
        return jnp.concatenate([
            v_body * c.obs_scales["base_vel"], grounded.astype(jnp.float32),
            # Both pads per foot, so the critic can tell flat from rolled-over-the-toe.  This is
            # the PRIVILEGED tail and it stays privileged: DASH-01 has no foot contact sensing, so
            # putting a contact bit in the actor's input would train against a sensor the robot
            # does not have.  The policy has to infer its foot state from the IMU and the joints,
            # exactly as it will on the robot.
            (toe if toe is not None else grounded).astype(jnp.float32),
            (heel if heel is not None else grounded).astype(jnp.float32),
            jnp.array([self._base_pos(data)[2] - c.term_height, self._y(data), self._yaw(data)]),
            accel / 9.81, fn / p.bw,
            jnp.array([dr.mass_scale, dr.com_off_x * 10.0, dr.friction,
                       dr.kp_scale.mean(), dr.torque_scale, dr.delay_ms / 10.0]),
            state.thermal_x,
        ]).astype(jnp.float32)

    def _obs(self, state, params, data, grounded, fn, accel, v_body, phase_next, commit_next,
             toe=None, heel=None):
        hist = state.hist[self.hist_idx].reshape(-1)
        once = self._once(state, params, phase_next, commit_next)
        priv = self._priv(data, state, grounded, fn, accel, v_body, toe, heel)
        return jnp.concatenate([hist, once, priv]).astype(jnp.float32)

    # ------------------------------------------------------------------ reset
    def _reset_one(self, key, params: EnvParams, ov: Override):
        c, p, gp = self.cfg, self.plant, self.gp
        (k_draw, k_noise, k_pose, k_lib, k_push, k_gust, k_next, k_frame,
         k_light) = jax.random.split(key, 9)
        draw = draw_plant(k_draw, c, p, params.dr_scale, ov)
        mx_i = model_with(p, draw.fields)
        qpos = jnp.asarray(p.key_qpos)
        n = c.reset_joint_noise
        noise = n * jax.random.uniform(k_pose, (p.leg_hinge_qadr.size,), minval=-1.0, maxval=1.0)
        qpos = qpos.at[jnp.asarray(p.leg_hinge_qadr)].add(noise)
        data = p.data0.replace(qpos=qpos, qvel=jnp.zeros(p.nv), ctrl=jnp.zeros(p.nu),
                               qfrc_applied=jnp.zeros(p.nv),
                               xfrc_applied=jnp.zeros_like(p.data0.xfrc_applied), time=jnp.zeros(()))
        # library: the episode's gait = a box-perturbed entry, and v_ref
        lib = self._library
        i_lib = lib["theta"].shape[0] - 1
        theta0 = jnp.asarray(lib["theta"][i_lib])
        kl1, kl2, kl3 = jax.random.split(k_lib, 3)
        amp = 1.0 + c.library_box_amp * jax.random.uniform(kl1, (21,), minval=-1.0, maxval=1.0)
        theta = theta0.at[0:21].multiply(amp)
        f0 = gait.frequency(theta0[gait.I_FREQ], gp)
        f1 = jnp.clip(f0 + c.library_box_f_hz * jax.random.uniform(kl2, (), minval=-1.0, maxval=1.0),
                      gp.freq_lo, gp.freq_hi)
        theta = theta.at[gait.I_FREQ].set(2.0 * (f1 - gp.freq_lo) / (gp.freq_hi - gp.freq_lo) - 1.0)
        n_knob = len(gait.KNOB_IDX)
        theta = theta.at[gait.KNOB_IDX[0]:gait.KNOB_IDX[-1] + 1].add(
            c.library_box_knob * jax.random.uniform(kl3, (n_knob,), minval=-1.0, maxval=1.0))
        ov_theta = jnp.broadcast_to(jnp.asarray(ov.theta, jnp.float32), (gait.SPEC_DIM,))
        theta = jnp.where(jnp.isnan(ov_theta), theta, ov_theta)
        v_ref = jnp.asarray(lib["v"][i_lib])
        spec0 = jnp.clip(theta, -1.0, 1.0) if self.library_mode else jnp.zeros(gait.SPEC_DIM)
        data = mjx.forward(mx_i, data)
        nominal = jnp.asarray(p.nominal_ctrl)
        cmd0 = jnp.concatenate([nominal, jnp.asarray(gp.drive_kp), jnp.asarray(gp.drive_kd)])
        k_cmd0, k_bring = jax.random.split(k_noise)
        v_cmd0, cmd_left0, stop0 = self._draw_cmd(k_cmd0, params)
        hold_s0 = jnp.asarray(params.hold_s, jnp.float32)      # probe override wins when set
        grace0 = jnp.zeros(())
        if c.bringup_enable:
            bz, bpitch, broll, bhold, bgrace = self._draw_bringup(k_bring, params)
            qpos = qpos.at[p.base_q["z"]].set(bz)
            if p.base_q["pitch"] >= 0:
                qpos = qpos.at[p.base_q["pitch"]].set(bpitch)
            if p.base_q["roll"] >= 0:
                qpos = qpos.at[p.base_q["roll"]].set(broll)
            hold_s0 = jnp.where(params.hold_s > 0.0, params.hold_s, bhold)
            grace0 = bgrace
            data = data.replace(qpos=qpos)
            data = mjx.forward(mx_i, data)
        state = EnvState(
            data=data, draw=draw, key=k_next, step_n=jnp.zeros((), jnp.int32), t=jnp.zeros(()),
            phase=jnp.zeros(()), spec=spec0, commit=jnp.ones((), bool),
            cycle_n=jnp.zeros((), jnp.int32), phi_td_hat=jnp.array([0.0, np.pi]),
            resynced=jnp.zeros(2, bool),
            cmd_buf=jnp.stack([cmd0, cmd0, cmd0]), prev_target=nominal,
            prev_target_vel=jnp.zeros(6), prev_action=jnp.zeros(self.action_dim),
            prev_residual=jnp.zeros(6), prev_motor_cmd=jnp.zeros(6), thermal_x=draw.thermal_x0,
            prev_vel_body=jnp.zeros(3), lp_yaw_true=jnp.zeros(()), lp_yaw_obs=jnp.zeros(()),
            yaw_est=jnp.zeros(()), heading_avg=jnp.zeros(()),
            roll_lp=jnp.zeros(()),
            hist=jnp.zeros((self.hist_raw_len, FRAME_DIM), jnp.float32),
            gyro_bias=jnp.zeros(3), stale_left=jnp.zeros((), jnp.int32), stale_frame=jnp.zeros(6),
            air_time=jnp.zeros(2), contact_time=jnp.zeros(2), grounded_prev=jnp.zeros(2, bool),
            prev_toe_xy=jnp.zeros((2, 2)), duty_ema=jnp.full(2, 0.5), ws_out_t=jnp.zeros(2),
            swing_ema=jnp.full(2, c.swing_floor_frac),
            push_countdown=self._next_interval(k_push, c.push_interval_s),
            trip_left=jnp.zeros((), jnp.int32), trip_foot=jnp.zeros((), jnp.int32),
            trip_force=jnp.zeros(()), gust_left=jnp.zeros((), jnp.int32),
            gust_countdown=self._next_gust(k_gust), gust_dir=jnp.array([1.0, 0.0]),
            x0=data.qpos[p.base_q["x"]], sprint_d=jnp.zeros(()), crossed=jnp.zeros((), bool),
            t_line=jnp.full((), -1.0), stop_hold=jnp.zeros(()),
            light_red=(params.start_red_s > 0.0) if c.hold_enable else jnp.zeros((), bool),
            light_left=jnp.where(jax.random.uniform(k_light) < params.stoplight_prob,
                                 jax.random.uniform(k_next, (), minval=c.stoplight_green_s[0],
                                                    maxval=c.stoplight_green_s[1]), jnp.inf)
            if not c.hold_enable else
            jnp.where(params.start_red_s > 0.0, params.start_red_s,
                      jnp.where(jax.random.uniform(k_light) < params.stoplight_prob,
                                jax.random.uniform(k_next, (), minval=c.stoplight_green_s[0],
                                                   maxval=c.stoplight_green_s[1]), jnp.inf)),
            light_v0=jnp.zeros(()), light_t=jnp.zeros(()), light_floor=jnp.zeros(()),
            v_cmd=v_cmd0, cmd_left=cmd_left0, stop_cmd=stop0, hold_s=hold_s0, grace_left=grace0,
            theta=jnp.clip(theta, -1.0, 1.0), v_ref=v_ref, raibert_i=jnp.zeros(()),
            ep_return=jnp.zeros(()), ep_len=jnp.zeros((), jnp.int32),
        )
        toe_c, heel_c, fn, _ = self._contacts(data)
        grounded = toe_c | heel_c | (self._foot_heights(data) < c.grounded_h)
        frame, ns, accel, v_body = self._frame(data, jnp.zeros(6), jnp.zeros(6), 0.0, state, k_frame)
        state = state.replace(hist=jnp.tile(frame[None], (self.hist_raw_len, 1)),
                              grounded_prev=grounded, prev_toe_xy=self._toe_pos(data)[:, :2],
                              prev_vel_body=v_body, **{**ns, "yaw_est": jnp.zeros(())})
        obs = self._obs(state, params, data, grounded, fn, accel, v_body, 0.0, 1.0, toe_c, heel_c)
        return state, obs

    def _touch_z(self, pitch, roll):
        """Bilinear lookup of the touching height; clamped at the table edge."""
        pg, rg, z = self._touch
        i = jnp.clip(jnp.searchsorted(pg, pitch) - 1, 0, pg.size - 2)
        j = jnp.clip(jnp.searchsorted(rg, roll) - 1, 0, rg.size - 2)
        wp = jnp.clip((pitch - pg[i]) / (pg[i + 1] - pg[i]), 0.0, 1.0)
        wr = jnp.clip((roll - rg[j]) / (rg[j + 1] - rg[j]), 0.0, 1.0)
        z0 = z[i, j] * (1 - wr) + z[i, j + 1] * wr
        z1 = z[i + 1, j] * (1 - wr) + z[i + 1, j + 1] * wr
        return z0 * (1 - wp) + z1 * wp

    def _draw_bringup(self, key, params):
        """(base_z, pitch, roll, hold_s, grace_s) for one episode's start.

        Three modes: NOMINAL (the settled keyframe, as every run before v3), DROP (feet off the
        ground, released from a height above touching) and HELD (feet down, body not square, the
        hand on for a moment). The bands scale with params.bringup_scale so the curriculum can open
        them as the policy earns it."""
        c = self.cfg
        k_m, k_p, k_r, k_d, k_h = jax.random.split(key, 5)
        s = jnp.clip(params.bringup_scale, 0.0, 1.0)
        # scale the SHARE of off-nominal starts, not just how hard they are: at s = 0 nearly every
        # episode is the settled keyframe, which is what the warm start knows how to do.
        u = jax.random.uniform(k_m)
        p_drop = jnp.where(params.bringup_p_drop < 0.0, c.bringup_drop_frac * s, params.bringup_p_drop)
        p_held = jnp.where(params.bringup_p_held < 0.0, c.bringup_held_frac * s, params.bringup_p_held)
        drop = u < p_drop
        held = (~drop) & (u < p_drop + p_held)
        lerp = lambda a, b: a + (b - a) * s
        pit_hi = jnp.deg2rad(lerp(c.bringup_pitch_deg_start, c.bringup_pitch_deg))
        rol_hi = jnp.deg2rad(lerp(c.bringup_roll_deg_start, c.bringup_roll_deg))
        pitch = jax.random.uniform(k_p, (), minval=-pit_hi, maxval=pit_hi)
        roll = jax.random.uniform(k_r, (), minval=-rol_hi, maxval=rol_hi)
        d_lo = lerp(c.bringup_drop_m_start[0], c.bringup_drop_m[0])
        d_hi = lerp(c.bringup_drop_m_start[1], c.bringup_drop_m[1])
        drop_h = jax.random.uniform(k_d, (), minval=d_lo, maxval=d_hi)
        hold = jax.random.uniform(k_h, (), minval=c.bringup_hold_s[0], maxval=c.bringup_hold_s[1])
        active = drop | held
        pitch = jnp.where(active, pitch, 0.0)
        roll = jnp.where(active, roll, 0.0)
        z_touch = self._touch_z(pitch, roll)
        base_z = jnp.where(drop, z_touch + drop_h, z_touch)
        # a drop has no hand on it; a held start does, and only a drop needs landing grace
        return base_z, pitch, roll, jnp.where(held, hold, 0.0), jnp.where(drop, c.bringup_grace_s, 0.0)

    def _draw_cmd(self, key, params):
        """(v_cmd, seconds until the next redraw). A share of draws are exactly zero -- that is the
        joystick at rest, which the policy must answer by stepping in place rather than by stopping
        dead: this plant has no passive stance to stand on."""
        c = self.cfg
        k_v, k_z, k_t = jax.random.split(key, 3)
        lo, hi = params.cmd_lo, params.cmd_hi
        v = jax.random.uniform(k_v, (), minval=lo, maxval=hi) * c.v_max
        if c.cmd_binary:
            # RUN/STOP: the stick is either at rest or at full, and full means "as fast as you can"
            # (the linear income in _reward). The band curriculum has nothing to widen here; the
            # zero share below still ramps, so a cold start is all RUN until the gait exists.
            v = jnp.full((), c.v_max)
        # params.cmd_zero_p, not cfg.cmd_zero_frac: a fixed zero share would ignore the command
        # curriculum completely and ask a warm-started runner to stop a quarter of the time from
        # step 0, which drags the gait clock onto the slow rail it never comes back from.
        v = jnp.where(jax.random.uniform(k_z) < params.cmd_zero_p, 0.0, v)
        left = c.cmd_interval_s * jax.random.uniform(k_t, (), minval=0.6, maxval=1.4)
        # THE SWITCH. Drawn independently of the stick, which keeps whatever value it drew: on the
        # robot the operator flips STOP with the stick anywhere, so the flag has to win on its own.
        if c.stop_flag:
            stop = (jax.random.uniform(jax.random.fold_in(k_z, 7)) < params.cmd_stop_p).astype(jnp.float32)
        else:
            stop = jnp.zeros(())
        return v, left, stop

    def _next_interval(self, key, mean_s):
        if mean_s <= 0.0:
            return jnp.full((), 10 ** 9, jnp.int32)
        s = mean_s * jax.random.uniform(key, (), minval=0.7, maxval=1.3)
        return jnp.maximum(1, jnp.round(s / self.control_dt)).astype(jnp.int32)

    def _next_gust(self, key):
        lo, hi = self.cfg.wind_gust_interval_s
        if self.cfg.wind_gust_n <= 0.0:
            return jnp.full((), 10 ** 9, jnp.int32)
        s = jax.random.uniform(key, (), minval=lo, maxval=hi)
        return jnp.maximum(1, jnp.round(s / self.control_dt)).astype(jnp.int32)

    # ------------------------------------------------------------------ step
    def _library_spec(self, state, catch, v_body):
        """Stage 2 Raibert prior + the catch-event dims -> the spec to commit (library variant)."""
        c, gp = self.cfg, self.gp
        th = state.theta
        f0 = gait.frequency(th[gait.I_FREQ], gp)
        f1 = jnp.clip(f0 * (1.0 + c.library_catch_scale * catch[0]), gp.freq_lo, gp.freq_hi)
        spec = th.at[gait.I_FREQ].set(2.0 * (f1 - gp.freq_lo) / (gp.freq_hi - gp.freq_lo) - 1.0)
        spec = spec.at[0:14].multiply(1.0 + c.library_catch_scale * catch[1])
        spec = spec.at[14:21].multiply(1.0 + c.library_catch_scale * catch[2])
        e_v = v_body[0] - state.v_ref
        i_cam, i_hip = gait.I_O.start, gait.I_O.start + 2
        o_cam = th[i_cam] - (c.raibert_kp * e_v + c.raibert_ki * state.raibert_i) / gp.o_max[0]
        o_hip = th[i_hip] - (c.raibert_ky * v_body[1] + c.raibert_kr * state.roll_lp) / gp.o_max[2]
        spec = spec.at[i_cam].set(o_cam).at[i_hip].set(o_hip)
        return jnp.clip(spec, -1.0, 1.0)

    def _step_one(self, state: EnvState, action, params: EnvParams):
        c, p, gp = self.cfg, self.plant, self.gp
        if c.gait_freq_floor_steps > 0:      # static branch: presets without the curriculum are untouched
            gp = gp._replace(freq_lo=params.gait_freq_lo)
        dt = self.control_dt
        dr = state.draw
        key, k_drop, k_push, k_trip, k_gust, k_jit, k_frame, k_reset, k_int, k_light = jax.random.split(state.key, 10)
        action = jnp.clip(action, -1.0, 1.0)
        drop = jax.random.uniform(k_drop) < params.ctrl_drop_prob
        action = jnp.where(drop, state.prev_action, action)
        data = state.data
        # ---- current clean signals (the reward and the library's Raibert offsets read these; the
        # control law itself sees only the measured frame, since v4 has no reflex)
        grav = self._grav_body(data)
        gyro = self._gyro(data)
        v_body_pre = self._vel_body(data)
        # ---- the latch
        commit = state.commit
        if self.library_mode:
            residual = action[:gait.LIB_N_RESIDUAL]
            catch = action[gait.LIB_N_RESIDUAL:]
            spec_cmd = self._library_spec(state, catch, v_body_pre)
        else:
            residual = action[gait.SPEC_DIM:]
            spec_cmd = action[:gait.SPEC_DIM]
        spec = jnp.where(commit, spec_cmd, state.spec)
        spec_change = jnp.where(commit & (state.cycle_n > 0), jnp.sum((spec - state.spec) ** 2), 0.0)
        if c.brake_prior > 0.0:
            # the measured braking direction (tools/brake_search.py): MORE cadence, feet forward, lean back
            st_now = state.crossed | state.light_red
            rmp = jnp.maximum(0.0, 1.0 - state.light_t / c.stop_decel_s) if c.stop_decel_s > 0 else 0.0
            v_tg = state.light_floor + (state.light_v0 - state.light_floor) * rmp
            g = c.brake_prior * jnp.tanh(jnp.maximum(v_body_pre[0] - v_tg, 0.0)) * st_now
            spec = spec.at[gait.I_FREQ].add(0.20 * g)
            spec = spec.at[gait.I_O.start].add(0.35 * g)
            spec = spec.at[gait.I_O.start + 1].add(-0.50 * g)
            spec = jnp.clip(spec, -1.0, 1.0)
        f = gait.frequency(spec[gait.I_FREQ], gp)
        phi = state.phase
        target, kp, kd, q_ref = gait.assemble(spec, residual, phi, jnp.asarray(p.nominal_ctrl), gp)
        target = jnp.clip(target, jnp.asarray(p.q_lo), jnp.asarray(p.q_hi))
        target, tvel = drive.slew_limit(target, state.prev_target, state.prev_target_vel,
                                        jnp.asarray(c.motor_vel_limit), c.motor_accel_limit, dt)
        motor_cmd = (target - jnp.asarray(p.nominal_ctrl)) / c.action_scale
        cmd = jnp.concatenate([target + dr.joint_zero, kp * dr.kp_scale, kd * dr.kv_scale])
        cmd_buf = jnp.stack([cmd, state.cmd_buf[0], state.cmd_buf[1]])
        # ---- disturbances at tick start
        # ADVERSITY RIDES THE RAMP ABOVE THE FLOOR, NOT THE FLOOR ITSELF.
        #
        # dr_scale_start exists to stop the policy converging on a deterministic PLANT -- one set
        # of masses, gains and friction. Pushes, trips and wind are not that; they are a separate
        # difficulty, and reading them straight off dr_scale meant the floor switched them on at
        # step 0. Measured: with the floor at 0.15 the robot took a push every 4.0 s while its
        # episodes were ~0.8 s long, and early ep_len halved against the same run at dr_scale 0
        # (84 vs 175 at 14 M) with the cadence stuck on its 1.50 Hz floor the whole way.
        #
        # Subtracting the floor and renormalising keeps both properties: the plant varies from the
        # first rollout, and the disturbances still start at zero and reach full by dr_scale 1.
        _f = float(getattr(c, "dr_scale_start", 0.0))
        adv = (jnp.clip((params.dr_scale - _f) / max(1.0 - _f, 1e-6), 0.0, 1.0)
               if c.adversity_curriculum else 1.0)
        qvel = data.qvel
        push_now = state.push_countdown <= 1
        kp1, kp2, kp3 = jax.random.split(k_push, 3)
        ang = jax.random.uniform(kp1, (), minval=0.0, maxval=TWO_PI)
        dv = adv * jax.random.uniform(kp2, (), minval=c.push_dv_range[0], maxval=c.push_dv_range[1])
        dvx = jnp.where(push_now, dv * jnp.cos(ang), 0.0)
        dvy = jnp.where(push_now, dv * jnp.sin(ang), 0.0)
        qvel = qvel.at[p.base_d["x"]].add(dvx)
        if not p.planar:
            qvel = qvel.at[p.base_d["y"]].add(dvy)
        if c.hold_enable:
            # the release tick: the hand's own motion goes into the base as it opens
            rel = (state.hold_s > 0.0) & (state.t >= state.hold_s) & (state.t - dt < state.hold_s)
            qvel = qvel.at[p.base_d["x"]].add(jnp.where(rel, params.release_vx, 0.0))
            qvel = qvel.at[p.base_d["z"]].add(jnp.where(rel, params.release_vz, 0.0))
            if not p.planar:
                qvel = qvel.at[p.base_d["y"]].add(jnp.where(rel, params.release_vy, 0.0))
        push_countdown = jnp.where(push_now, self._next_interval(kp3, c.push_interval_s),
                                   state.push_countdown - 1)
        # trip: a brief force opposing travel on an airborne foot
        gr_pre = state.grounded_prev
        kt1, kt2, kt3 = jax.random.split(k_trip, 3)
        air = ~gr_pre
        n_air = air.sum()
        pick = jnp.where(air[0] & air[1], (jax.random.uniform(kt1) < 0.5).astype(jnp.int32),
                         jnp.where(air[1], 1, 0))
        trip_start = (state.trip_left <= 0) & (n_air > 0) & (jax.random.uniform(kt2) < adv * c.trip_prob)
        trip_force = jnp.where(trip_start, -jnp.sign(v_body_pre[0] + 1e-9)
                               * jax.random.uniform(kt3, (), minval=c.trip_force_range[0],
                                                    maxval=c.trip_force_range[1]), state.trip_force)
        trip_left = jnp.where(trip_start, int(max(1, round(c.trip_duration_s / dt))),
                              jnp.maximum(state.trip_left - 1, 0))
        trip_foot = jnp.where(trip_start, pick, state.trip_foot)
        trip_on = trip_left > 0
        # wind: constant + gusts
        kg1, kg2 = jax.random.split(k_gust)
        gust_start = state.gust_countdown <= 1
        gang = jax.random.uniform(kg1, (), minval=0.0, maxval=TWO_PI)
        gust_dir = jnp.where(gust_start, jnp.array([jnp.cos(gang), jnp.sin(gang)]), state.gust_dir)
        gust_left = jnp.where(gust_start, int(max(1, round(c.wind_gust_s / dt))),
                              jnp.maximum(state.gust_left - 1, 0))
        gust_countdown = jnp.where(gust_start, self._next_gust(kg2), state.gust_countdown - 1)
        gust = jnp.where(gust_left > 0, adv * c.wind_gust_n * gust_dir, jnp.zeros(2))
        if p.planar:
            gust = gust.at[1].set(0.0)
        wind = dr.wind + gust
        xfrc = jnp.zeros_like(data.xfrc_applied)
        xfrc = xfrc.at[p.base_bid, 0].set(wind[0]).at[p.base_bid, 1].set(wind[1])
        foot_b = jnp.asarray(p.foot_bids)[trip_foot]
        xfrc = xfrc.at[foot_b, 0].add(jnp.where(trip_on, trip_force, 0.0))
        data = data.replace(qvel=qvel, xfrc_applied=xfrc, qfrc_applied=jnp.zeros(p.nv))
        # ---- physics: 10 substeps at a jittered timestep (the Pi's loop vs the gait clock)
        jit_ms = params.ctrl_jitter_ms * jax.random.uniform(k_jit, (), minval=-1.0, maxval=1.0)
        ts = p.sim_dt * (1.0 + jit_ms / (p.decimation * p.sim_dt * 1e3))
        mx_i = model_with(p, dr.fields, timestep=ts)
        peak = jnp.asarray(p.tau_peak)
        kt = jnp.asarray(c.motor_kt_joint)
        r_ohm = jnp.asarray(c.motor_r_ohm)

        # bring-up hold: the base is on the operator's stand for the first hold_s seconds
        if c.hold_enable:
            names = [n for n in ("x", "y", "z", "roll", "pitch", "yaw") if p.base_q[n] >= 0]
            hold_qadr = np.array([p.base_q[n] for n in names])
            hold_dadr = np.array([p.base_d[n] for n in names])
            held_at = dict(x=p.key_qpos[p.base_q["x"]], y=0.0, yaw=0.0,
                           # hold it WHERE IT WAS PLACED, the way roll and pitch below do. The
                           # keyframe height is the height the feet touch at zero tilt; a bring-up
                           # episode is placed at `_touch_z(pitch, roll)`, which is a different
                           # number, so clamping z back to the keyframe presses the robot into the
                           # floor (or hangs it) for the whole hold. Small at the tilts stage 3
                           # uses -- 5 mm at 9 deg, inside this plant's own 5 mm stance deflection
                           # -- but 26 mm at the 20 deg stage 2 opens to, which is not.
                           z=jnp.where(params.hold_z > 0.0, params.hold_z,
                                       jnp.where(c.bringup_enable,
                                                 state.data.qpos[p.base_q["z"]],
                                                 p.key_qpos[p.base_q["z"]])),
                           roll=jnp.where(c.bringup_enable, state.data.qpos[p.base_q["roll"]]
                                          if p.base_q["roll"] >= 0 else 0.0, params.hold_roll),
                           pitch=jnp.where(c.bringup_enable, state.data.qpos[p.base_q["pitch"]]
                                           if p.base_q["pitch"] >= 0 else 0.0, params.hold_pitch))
            hold_qval = jnp.stack([jnp.asarray(held_at[n], jnp.float32) for n in names])
            hold = state.t < state.hold_s

        def substep(carry, k):
            d, tau_sq, con_acc, _ = carry
            live = drive.live_command(k, dr.delay_ms, cmd_buf)
            q = d.qpos[p.act_qadr]
            qd = d.qvel[p.act_dadr]
            lim = drive.torque_limit(qd, peak, dr.torque_scale, kt, r_ohm, c.motor_bus_volts)
            tau = drive.pd_torque(q, qd, live[:6], live[6:12], live[12:18], lim)
            d = d.replace(ctrl=tau)
            d = mjx.step(mx_i, d)
            if c.hold_enable:
                d = d.replace(qpos=jnp.where(hold, d.qpos.at[hold_qadr].set(hold_qval), d.qpos),
                              qvel=jnp.where(hold, d.qvel.at[hold_dadr].set(0.0), d.qvel))
            toe_c, heel_c, _, _ = self._contacts(d)
            return (d, tau_sq + tau ** 2, con_acc | toe_c | heel_c, tau), None

        (data, tau_sq, contact_acc, tau_last), _ = lax.scan(
            substep, (data, jnp.zeros(6), jnp.zeros(2, bool), jnp.zeros(6)),
            jnp.arange(p.decimation))
        # ---- thermal node
        thermal_x = drive.thermal_update(state.thermal_x, tau_sq / p.decimation, dt, c.thermal_tau_s,
                                         jnp.asarray(c.thermal_tau_cont), dr.thermal_scale) \
            if c.thermal_enable else state.thermal_x
        # ---- the clock: advance and wrap. FREE-RUNNING unless resync_enable (v2 only).
        phi_adv = phi + TWO_PI * f * dt
        wrapped = phi_adv >= TWO_PI
        phi_new = jnp.mod(phi_adv, TWO_PI)
        cycle_n = state.cycle_n + wrapped.astype(jnp.int32)
        toe_c, heel_c, fn, floor_viol = self._contacts(data)   # reward + privileged tail only
        heights = self._foot_heights(data)
        grounded = contact_acc | toe_c | heel_c | (heights < c.grounded_h)
        # foot-flat: both pads of a grounded foot on the floor. A foot down on one pad only is
        # rolling over its toe or its heel, which is what this robot does instead of using an
        # ankle it does not have.
        foot_flat = jnp.where(grounded, (toe_c & heel_c).astype(jnp.float32), 1.0)
        if c.resync_enable:
            # THE ONE CONTACT PATH INTO THE ACTOR, and the reason v3 turns this off. The pull moves
            # phi -- which the actor reads as [cos, sin] -- and `crossed_fwd` can carry the commit
            # flag over the wrap, so BOTH of those actor inputs become contact-derived. DASH-01 has
            # no foot contact sensor and cannot reproduce either. Measured 2026-09-11: removing it
            # changes neither the dash (103.1-104.9 m at 3.20-3.28 m/s vs 103.1-104.8 at 3.23-3.29)
            # nor braking (511/512 upright vs 476/512). Kept switchable so v2 runs still replay; when
            # off, this whole block is compiled out and no contact quantity reaches the clock at all.
            resynced = jnp.where(wrapped, jnp.zeros(2, bool), state.resynced)
            rising = grounded & ~state.grounded_prev
            W = c.resync_window_cycle * TWO_PI
            err = gait.wrap_pi(state.phi_td_hat - phi_new)          # (2,)
            can = rising & (jnp.abs(err) <= W) & ~resynced
            phi_td_hat = state.phi_td_hat + jnp.where(can, gait.wrap_pi(phi_new - state.phi_td_hat)
                                                      / c.resync_ema_cycles, 0.0)
            kappa = jnp.where(cycle_n >= c.resync_warmup_cycles, dr.kappa, 0.0)
            shift = jnp.sum(jnp.where(can, kappa * err, 0.0))
            phi2 = phi_new + shift
            crossed_fwd = phi2 >= TWO_PI
            phi2 = jnp.maximum(phi2, 0.0)                          # a backward pull never uncrosses
            phi2 = jnp.mod(phi2, TWO_PI)
            wrapped = wrapped | crossed_fwd
            cycle_n = cycle_n + crossed_fwd.astype(jnp.int32)
            resynced = resynced | can
        else:
            phi2 = phi_new
            phi_td_hat, resynced = state.phi_td_hat, state.resynced
            can = jnp.zeros(2, bool)        # diag/resync stays a channel, permanently 0
        commit_next = wrapped
        grace_left = jnp.maximum(state.grace_left - dt, 0.0)
        # ---- sprint bookkeeping
        t = state.t + dt
        x = data.qpos[p.base_q["x"]]
        sprint_d = x - state.x0
        v_world = self._vel_world(data)
        v_body = self._vel_body(data)
        newly_crossed = (~state.crossed) & (sprint_d >= params.sprint_dist_m)
        crossed = state.crossed | newly_crossed
        t_line = jnp.where(newly_crossed, t, state.t_line)
        vx_stop = v_body[0]
        stopped = jnp.abs(vx_stop) <= c.stop_speed_eps
        stop_hold = jnp.where(crossed & stopped, state.stop_hold + dt, 0.0)
        finished = crossed & (stop_hold >= c.stop_hold_s) if c.objective == "sprint" else jnp.zeros((), bool)
        # ---- the joystick: redraw the commanded speed on its own timer
        if c.objective == "joystick":
            k_cmd, k_light = jax.random.split(k_light)
            due = (state.cmd_left - dt) <= 0.0
            v_new, left_new, stop_new = self._draw_cmd(k_cmd, params)
            v_cmd = jnp.where(due, v_new, state.v_cmd)
            cmd_left = jnp.where(due, left_new, state.cmd_left - dt)
            stop_cmd = jnp.where(due, stop_new, state.stop_cmd)
        else:
            v_cmd, cmd_left, stop_cmd = state.v_cmd, state.cmd_left, state.stop_cmd
        # ---- stop curriculum: red light / green light phases (only before the line)
        kl1, kl2 = jax.random.split(k_light)
        light_left = state.light_left - dt
        toggle = (light_left <= 0.0) & (~crossed)
        green_dur = jax.random.uniform(kl1, (), minval=c.stoplight_green_s[0], maxval=c.stoplight_green_s[1])
        red_dur = jax.random.uniform(kl2, (), minval=c.stoplight_red_s[0], maxval=c.stoplight_red_s[1])
        light_red = jnp.where(toggle, ~state.light_red, state.light_red)
        light_left = jnp.where(toggle, jnp.where(state.light_red, green_dur, red_dur), light_left)
        go_red = (toggle & ~state.light_red) | newly_crossed
        light_v0 = jnp.where(go_red, v_body[0], state.light_v0)
        light_t = jnp.where(go_red, 0.0, state.light_t + dt)
        # amber: the ramp descends to a slow RUN instead of a standstill (the line always demands a stop)
        if c.amber_frac > 0.0:
            kl3, kl4 = jax.random.split(kl2)
            amber = (jax.random.uniform(kl3) < c.amber_frac) & (~newly_crossed)
            floor_new = jnp.where(amber, jax.random.uniform(kl4, (), minval=c.amber_speed_band[0],
                                                            maxval=c.amber_speed_band[1]), 0.0)
            light_floor = jnp.where(go_red, floor_new, state.light_floor)
            light_floor = jnp.where(newly_crossed, 0.0, light_floor)
        else:
            light_floor = jnp.zeros(())
        stop_now = crossed | light_red
        if c.stop_decel_s > 0:
            frac = jnp.maximum(0.0, 1.0 - light_t / c.stop_decel_s)
            v_target = light_floor + (light_v0 - light_floor) * frac
        else:
            v_target = jnp.zeros(())
        # ---- reward
        lp = float(np.exp(-dt / c.lp_yaw_tau_s))
        lp_yaw_true = lp * state.lp_yaw_true + (1.0 - lp) * gyro[2]
        roll_lp = 0.95 * state.roll_lp + 0.05 * grav[1]
        heading_avg = heading_ema(state.heading_avg, state.yaw_est, self._heading_a, c.heading_cap_rad)
        rw = self._reward(state, params, data, mx_i, spec, spec_change, residual, motor_cmd, tau_last,
                          grounded, heights, fn, v_body, grav, gyro, lp_yaw_true, crossed, sprint_d,
                          thermal_x, phi, stop_now, v_target, heading_avg=heading_avg,
                          foot_flat=foot_flat)
        reward, terms, book = rw
        reward = reward * self.reward_dt_scale
        reward = jnp.maximum(reward, -c.step_reward_floor * self.reward_dt_scale)
        # ---- termination
        finite = jnp.isfinite(data.qpos).all() & jnp.isfinite(data.qvel).all()
        ws_out = self._workspace_out(data)
        ws_out_t = jnp.where(ws_out, state.ws_out_t + dt, 0.0)
        ws_kill = (ws_out_t >= c.workspace_grace_s).any() if c.workspace_kill else jnp.zeros((), bool)
        term_low = self._base_pos(data)[2] < c.term_height
        term_tip = grav[2] > c.term_gravity_z
        fallen = (~finite) | term_low | term_tip | floor_viol | ws_kill
        if c.hold_enable:
            # held: the stand is what carries it, not a fall. NOTE this masks ~finite too, so a NaN
            # plant state during the hold is invisible -- watch diag for it rather than assuming none.
            fallen = fallen & ~hold
        if c.bringup_enable:
            # a dropped robot is legitimately below term_height and touching nothing while it falls;
            # without this every drop episode dies on the first tick. The grace is short and only a
            # drop gets one, so it cannot hide a genuine fall for long.
            fallen = fallen & (state.grace_left <= 0.0)
        reward = reward - c.fall_penalty * fallen + c.finish_bonus * (finished & ~fallen)
        # a non-finite plant state (an unconverged capped solver step can blow up) ends the episode
        # above; the reward computed from that state is NaN and would poison GAE, the value loss and
        # the params for good (v2c_s1_planar_fast_s0 died this way at 36.5 M) -- bill it as a fall
        reward = jnp.where(finite, reward, -c.fall_penalty)
        step_n = state.step_n + 1
        truncated = step_n >= self.max_steps
        done = fallen | finished | truncated
        # ---- observation
        frame, ns, accel, v_body_m = self._frame(data, tau_last, residual, phi2, state, k_frame)
        hist = jnp.concatenate([state.hist[1:], frame[None]], axis=0)
        new_state = state.replace(
            data=data, key=key, step_n=step_n, t=t, phase=phi2, spec=spec, commit=commit_next,
            cycle_n=cycle_n, phi_td_hat=phi_td_hat, resynced=resynced, cmd_buf=cmd_buf,
            prev_target=target, prev_target_vel=tvel, prev_action=action, prev_residual=residual,
            prev_motor_cmd=motor_cmd, thermal_x=thermal_x, prev_vel_body=v_body_m,
            lp_yaw_true=lp_yaw_true, roll_lp=roll_lp, hist=hist,
            heading_avg=heading_avg,
            air_time=book["air_time"], contact_time=book["contact_time"], grounded_prev=grounded,
            prev_toe_xy=self._toe_pos(data)[:, :2], duty_ema=book["duty_ema"], ws_out_t=ws_out_t,
            swing_ema=book["swing_ema"], push_countdown=push_countdown, trip_left=trip_left,
            trip_foot=trip_foot, trip_force=trip_force, gust_left=gust_left,
            gust_countdown=gust_countdown, gust_dir=gust_dir, sprint_d=sprint_d, crossed=crossed,
            t_line=t_line, stop_hold=stop_hold,
            light_red=light_red, light_left=light_left, light_v0=light_v0, light_t=light_t,
            light_floor=light_floor, v_cmd=v_cmd, cmd_left=cmd_left, stop_cmd=stop_cmd,
            hold_s=state.hold_s, grace_left=grace_left,
            raibert_i=jnp.clip(state.raibert_i + (v_body[0] - state.v_ref) * dt,
                               -c.raibert_imax, c.raibert_imax),
            ep_return=state.ep_return + reward, ep_len=state.ep_len + 1, **ns)
        obs = self._obs(new_state, params, data, grounded, fn, accel, v_body_m, phi2,
                        commit_next.astype(jnp.float32), toe_c, heel_c)
        info = dict(
            commit=commit, reward_terms=terms, foot_air=(~grounded).astype(jnp.float32),
            fallen=fallen, finished=finished, truncated=truncated,
            sprint_d=sprint_d, t_line=t_line, ep_return=new_state.ep_return,
            ep_len=new_state.ep_len, thermal_max=thermal_x.max(),
            torque_util=jnp.mean(jnp.abs(tau_last) / jnp.asarray(p.tau_peak)),
            freq_hz=f, spec_change=spec_change, resync=can.any(),
            residual_sat=jnp.mean(jnp.abs(residual) >= 0.95), lateral_y=self._y(data),
            # what the joystick actually commands: FORWARD speed in the base frame. The world-x
            # velocity is the wrong readout for a policy that tracks its own heading -- after a half
            # turn an obedient robot reads as running backwards -- and it is what the reward bills.
            v_body_x=v_body[0], yaw_true=self._yaw(data),
            # the EFFECTIVE target: zero under STOP, whatever the stick says
            v_cmd=state.v_cmd * (1.0 - state.stop_cmd), stop_cmd=state.stop_cmd,
            track_err=jnp.abs(v_body[0] - state.v_cmd * (1.0 - state.stop_cmd)),
            run_tick=1.0 - state.stop_cmd,
            term_low=term_low, term_tip=term_tip, term_floor=floor_viol, term_ws=ws_kill, term_nan=~finite,
            light_red=light_red.astype(jnp.float32),
            # the observation of the state the episode ENDED in, before the auto-reset overwrote it:
            # what a time-limit truncation has to bootstrap from (ppo.gae)
            obs_final=jnp.nan_to_num(obs),
        )
        # ---- auto-reset
        rs_state, rs_obs = self._reset_one(k_reset, params, Override())
        sel = lambda a, b: jnp.where(done, a, b)
        out_state = jax.tree_util.tree_map(sel, rs_state, new_state)
        out_obs = jnp.nan_to_num(jnp.where(done, rs_obs, obs))
        return out_state, out_obs, reward.astype(jnp.float32), done, info

    # ------------------------------------------------------------------ reward
    def _reward(self, state, params, data, mx_i, spec, spec_change, residual, motor_cmd, tau,
                grounded, heights, fn, v_body, grav, gyro, lp_yaw_true, crossed, sprint_d,
                thermal_x, phi, stop_now=None, v_target=None, heading_avg=0.0,
                foot_flat=None):
        c, p, gp = self.cfg, self.plant, self.gp
        if stop_now is None:
            stop_now, v_target = crossed, jnp.zeros(())
        dt = self.control_dt
        cap = c.penalty_term_cap
        pen = lambda v: jnp.maximum(v, -cap)
        vx = v_body[0]
        if c.sprint_world_speed:
            vx = self._vel_world(data)[0]
        run_phase = (c.objective in ("speed", "joystick")) | (~stop_now)
        t = {}
        # ---- objective income
        if c.objective == "joystick":
            # TRACK THE COMMAND -- but the income must be worth as much as the dash's was, or staying
            # alive stops paying. Measured 2026-09-11 (tools/reward_budget.py, same checkpoint, same
            # plant): the sprint objective earns income 5.11/tick against 2.69 of cost, net +1.32,
            # and survives 600/600 ticks. A BOUNDED tracking income (w_track * shape, ceiling 3.0)
            # earned 0.57 against the same 2.50 of cost -- net -0.55/tick after the step floor. With
            # fall_penalty 100 and reward_dt_scale 0.5, the break-even horizon is ~200 ticks against
            # 3000-tick episodes, so dying at once was worth ~15x living: both v3 seeds railed the clock to its
            # 1.5 Hz floor, dropped residual saturation to 0.06 and fell early. Costs were never the
            # problem (2.69 vs 2.50); deleting fwd_speed removed 94% of the income.
            #
            # So scale the income by what was ASKED, not by what was achieved. `shape` still peaks
            # only at the commanded speed -- running faster than commanded pays less, which is the
            # property a bare speed income lacks -- while the magnitude tracks the cost of the
            # commanded gait, exactly as the dash's income tracked the cost of sprinting. w_track is
            # the part that survives at v_cmd = 0, so holding station still pays (and drifting does
            # not), which a purely proportional income would have zeroed out.
            # Laplace, not Gaussian: measured, a Gaussian pays 0.006 at the ~1.8 m/s error this
            # lineage sits at, i.e. it is flat exactly where the policy lives.
            # The Laplace width is a CURRICULUM, not a constant. A cold policy runs at 0 m/s, so at
            # the shipped sigma of 0.6 a 1.5 m/s command pays exp(-2.5) = 0.08 of its income and a
            # 3.2 m/s one pays 0.005 -- flat, which is how four cold seeds ended up parked on the
            # 1.5 Hz clock rail with nothing to climb. Starting wide makes the first metre of speed
            # worth something; it then tightens to 0.6 so the finished policy is held to the stick.
            v_tgt_j = state.v_cmd * (1.0 - state.stop_cmd)      # STOP asks for zero, whatever the stick says
            income = joystick_income(vx, v_tgt_j, params.track_sigma, c.w_track, c.w_fwd_speed,
                                     getattr(c, "w_speed_income", 0.0), c.v_ceiling)
            if c.run_income_linear:
                # RUN/STOP (walk_v4). The Laplace kernel above pays a robot that walks in place under
                # a 3.6 m/s command 0.06 of its income -- not zero, and flat -- so across the command
                # band the in-place basin earned ~2.8/tick with alive against ~7.2 for a perfect
                # tracker, and won whenever the runner fell before ~11 s. Under RUN the income is the
                # sprint objective's instead: linear in forward speed, paying from the first cm/s and
                # nothing for standing, which is the only income this project ever trained a fast
                # runner on. At rest the kernel stays: holding station is what STOP asks for.
                income = jnp.where(state.v_cmd > 0.0,
                                   c.w_fwd_speed * jnp.clip(vx, -c.v_ceiling, c.v_ceiling), income)
            if c.speed_upright_gate:
                u = jnp.clip((-grav[2] - c.speed_upright_c0) / (1.0 - c.speed_upright_c0), 0.0, 1.0)
                # gate the positive side only: a tilted robot running BACKWARDS must not get a discount
                income = jnp.where(income > 0.0, income * u ** c.speed_upright_k, income)
            t["track"] = income
            t["fwd_speed"] = jnp.zeros(())
        else:
            income = c.w_fwd_speed * jnp.clip(vx, -c.v_ceiling, c.v_ceiling)
            if c.speed_upright_gate:
                u = jnp.clip((-grav[2] - c.speed_upright_c0) / (1.0 - c.speed_upright_c0), 0.0, 1.0)
                income = jnp.where(income > 0.0, income * u ** c.speed_upright_k, income)
            t["fwd_speed"] = jnp.where(run_phase, income, 0.0)
        # stop term: track the deceleration target while it is > 0 (stop_decel_s), then be still
        # capture step: feet ahead of the CoM while the robot is above its commanded speed
        if c.w_brake_foot > 0.0:
            ahead = self._toe_pos(data)[:, 0] - data.qpos[p.base_q["x"]]
            # NOT clipped at zero: this lineage plants its feet ~8 cm BEHIND the CoM (walk_mit m3), so a
            # one-sided reward is identically zero there and teaches nothing -- the same flat-region
            # mistake as the Gaussian tracking term. Signed, so moving the foot forward always pays.
            ahead = jnp.sum(jnp.where(grounded, jnp.clip(ahead, -c.brake_foot_max_m, c.brake_foot_max_m), 0.0))                 / jnp.maximum(jnp.sum(grounded), 1.0)
            need = jnp.tanh(jnp.maximum(vx - v_target, 0.0))
            t["brake_foot"] = jnp.where(run_phase, 0.0, c.w_brake_foot * need * ahead / c.brake_foot_max_m)
        else:
            t["brake_foot"] = 0.0
        sig = jnp.where(v_target > 0.0, c.decel_sigma, c.stop_sigma)
        err = jnp.abs(vx - v_target)
        track = jnp.exp(-err / sig) if c.stop_track_laplace else jnp.exp(-(err / sig) ** 2)
        t["stop"] = jnp.where(run_phase, 0.0, c.w_stop_vel * track)
        over = jnp.maximum(0.0, sprint_d - (params.sprint_dist_m + c.sprint_brake_m))
        t["overrun"] = jnp.where(run_phase, 0.0, pen(-c.w_overrun * over))
        t["time"] = -c.w_time if c.objective == "sprint" else 0.0   # joystick: obeying "stop" is not a sin
        t["alive"] = c.w_alive * params.alive_scale
        t["yaw_rate"] = pen(-c.w_yaw_rate * lp_yaw_true ** 2)
        # HEADING: the set point the yaw-rate term never had. Billed on the TRUE base yaw (a
        # reward may be privileged; the actor gets the integrated-gyro estimate instead), and
        # saturated at the same +-pi/2 the observation is, so a robot already sideways is not
        # paying a penalty that dwarfs everything else it could still do about it.
        # v4 (heading_avg_s > 0): billed instead on the robot's OWN integrated-gyro heading, averaged
        # -- see heading_ema.
        if c.heading_avg_s > 0:
            t["heading"] = pen(-c.w_heading * heading_avg ** 2)
        else:
            yaw_true = jnp.clip(gait.wrap_pi(self._yaw(data)), -c.heading_cap_rad, c.heading_cap_rad)
            t["heading"] = pen(-c.w_heading * yaw_true ** 2)
        y = self._y(data)
        t["lane"] = pen(-c.w_lane * jnp.maximum(jnp.abs(y) - c.lane_free_m, 0.0) ** 2)
        progress = jnp.where(run_phase, jnp.clip(vx / c.v_ceiling, 0.0, 1.0), 0.0)
        # Under the continuous stop command the commanded speed FOLLOWS THE RAMP instead of dropping to
        # zero the instant the light turns. With the binary version the whole gait block (air-time
        # credit, swing floor, stance time, clearance, phase contact -- everything below `gait_on`)
        # switched off for the entire deceleration, i.e. exactly while the robot has to hold a gait
        # together through 2.5 -> 2.0 -> 1.0 -> 0 m/s, the regime it has never been shaped in. Now the
        # shaping tracks the command down and only lets go below gait_cmd_gate (a genuine standstill).
        standing_cmd = jnp.zeros((), bool)
        if c.objective == "joystick":
            # the commanded speed IS the joystick, and the gait block stays on all the way down to
            # zero: a zero command means step in place, not stand still, because this plant has no
            # passive stance to hold (bring-up probe: it topples in 0.7-1.0 s with no gait)
            cmd_speed = state.v_cmd * (1.0 - state.stop_cmd)
            # ...unless stand_at_zero: the flat-foot robot HAS a stance, so at exactly zero stick the
            # gait block lets go and the stand bill below takes over (config.stand_at_zero). With
            # stop_flag the condition is the operator's SWITCH, not the stick (config.stop_flag).
            if c.stop_flag:
                standing_cmd = state.stop_cmd > 0.5
            else:
                standing_cmd = (state.v_cmd <= 1e-3) if c.stand_at_zero else jnp.zeros((), bool)
            gait_on = ~standing_cmd
        elif c.stop_cmd_continuous and c.stop_decel_s > 0:
            cmd_speed = jnp.where(run_phase, c.v_ceiling, jnp.clip(v_target, 0.0, c.v_ceiling))
            gait_on = cmd_speed >= c.gait_cmd_gate
        else:
            cmd_speed = jnp.where(run_phase, c.v_ceiling, 0.0)
            gait_on = cmd_speed >= c.gait_cmd_gate
        # ---- gait shaping
        toe = self._toe_pos(data)
        gprev = state.grounded_prev
        slip_v = jnp.linalg.norm(toe[:, :2] - state.prev_toe_xy, axis=1) / dt
        slip = jnp.where(grounded & gprev, jnp.maximum(0.0, slip_v - c.slip_deadband) ** 2, 0.0)
        t["foot_slip"] = -jnp.minimum(c.w_foot_slip * slip.sum(), cap)
        air_credit = jnp.where(grounded & (state.air_time > 0.0) & gait_on,
                               c.w_air_time * jnp.clip(state.air_time - c.foot_air_time_min, 0.0,
                                                       c.air_credit_cap_s), 0.0)
        if c.air_credit_cmd_scaled:
            air_credit = air_credit * jnp.clip(cmd_speed / max(c.v_max, 1e-6), 0.0, 1.0)
        t["air_time"] = air_credit.sum()
        air_time = jnp.where(grounded, 0.0, state.air_time + dt)
        contact_time = jnp.where(grounded, state.contact_time + dt, 0.0)
        a_ema = float(np.exp(-dt / max(c.swing_floor_tau_s, 1e-6)))
        swing_ema = a_ema * state.swing_ema + (1.0 - a_ema) * (~grounded).astype(jnp.float32)
        deficit = jnp.maximum(0.0, c.swing_floor_frac - swing_ema.min())
        t["swing_floor"] = jnp.where(gait_on, pen(-c.w_swing_floor * deficit ** 2), 0.0)
        capst = jnp.where(cmd_speed >= c.stance_slow_speed, c.stance_cap_s, c.stance_cap_slow_s)
        over_st = jnp.minimum(jnp.maximum(contact_time - capst, 0.0), 1.0)
        t["stance_time"] = jnp.where(gait_on, -jnp.minimum(c.w_stance_time * over_st.sum(), cap), 0.0)
        grounded_recent = grounded | gprev
        fresh = (~grounded_recent) & (air_time > 0.0) & (air_time <= c.swing_fresh_s)
        frac = jnp.clip((heights - c.clearance_dead_m) / c.clearance_scale_m, 0.0, 1.0)
        t["clearance"] = jnp.where(gait_on, jnp.sum(jnp.where(fresh, c.w_clearance * frac
                                                               * (0.3 + 0.7 * progress), 0.0)), 0.0)
        phi_l, phi_r = gait.phases(spec, phi, gp)
        sw_l = 1.0 - gait.stance_indicator(phi_l, params.stance_ratio)
        sw_r = 1.0 - gait.stance_indicator(phi_r, params.stance_ratio)
        pc = sw_l * grounded[0] + sw_r * grounded[1]
        t["phase_contact"] = jnp.where(gait_on, -jnp.minimum(c.w_phase_contact * pc, cap), 0.0)
        t["step_rate"] = pen(-c.w_contact_switch * jnp.sum(grounded != gprev))
        a_d = min(1.0, dt / max(c.duty_sym_tau_s, 1e-3))
        duty_ema = state.duty_ema + a_d * (grounded.astype(jnp.float32) - state.duty_ema)
        t["duty_sym"] = pen(-c.w_duty_sym * jnp.sum(jnp.maximum(0.0, c.duty_floor - duty_ema)))
        # ---- efficiency
        qd = data.qvel[p.act_dadr]
        exc = jnp.maximum(jnp.abs(tau) - jnp.abs(jnp.asarray(p.stand_torque)), 0.0)
        es = params.eff_scale
        t["torque"] = pen(-es * c.w_torque * jnp.sum(exc ** 2))
        t["motor_vel"] = pen(-es * c.w_motor_vel * jnp.sum(qd ** 2))
        t["energy"] = pen(-es * c.w_energy * jnp.sum(jnp.maximum(tau * qd, 0.0)))
        # ---- smoothness, the latch's billing, the standing knob price, thermal
        t["action_rate"] = pen(-c.w_action_rate * jnp.sum((motor_cmd - state.prev_motor_cmd) ** 2))
        # Billed ONCE at commit, which means a policy that halves its cadence halves what it pays
        # per SECOND for rewriting the spec -- a standing discount collected by parking the clock on
        # its floor, and this lineage has parked the clock before (the m3 clock-warp exploit, and
        # four cold v2 seeds that sat on 1.5 Hz and never left). Normalising by cadence makes the
        # cost per second independent of frequency, so slowing down buys nothing here. Bounded
        # either way: f is confined to gait_freq_hz, so the factor lives in ~[0.7, 1.8].
        f_nom = 0.5 * (c.gait_freq_hz[0] + c.gait_freq_hz[1])
        f_now = gait.frequency(spec[gait.I_FREQ], gp)
        rate = (f_nom / jnp.maximum(f_now, 1e-3)) if c.spec_cycle_rate_invariant else 1.0
        t["spec_cycle"] = pen(-c.w_spec_cycle * spec_change * rate)
        t["knob"] = pen(-c.w_knob * jnp.sum(gait.knobs(spec, gp) ** 2))
        t["residual"] = pen(-c.w_residual * jnp.sum(residual ** 2))
        t["residual_rate"] = pen(-c.w_residual_rate * jnp.sum((residual - state.prev_residual) ** 2))
        t["thermal"] = pen(-c.w_thermal * jnp.sum(jnp.maximum(thermal_x - c.thermal_penalty_frac, 0.0) ** 2)) \
            if c.thermal_enable else 0.0
        # Foot-flat.  A grounded foot resting on one pad is rolling over its toe or its heel,
        # and on this robot that is not a choice the ankle makes -- there is no ankle, so the
        # sole's angle is the leg's angle, and a foot that lands on an edge is a leg that arrived
        # at the wrong angle.  Billed per foot, per tick, and rides the shaping ramp with the rest
        # of the gait-quality terms so a cold policy is not taxed for it before it can walk.
        if c.w_foot_flat > 0.0 and foot_flat is not None:
            t["foot_flat"] = pen(-c.w_foot_flat * jnp.sum(1.0 - foot_flat))

        # ---- posture
        t["upright"] = pen(-c.w_upright * (grav[0] ** 2 + grav[1] ** 2))
        z = self._base_pos(data)[2]
        t["height"] = pen(-c.w_height * jnp.maximum(c.height_floor_m - z, 0.0) ** 2)
        t["vz"] = pen(-c.w_vz * self._vel_world(data)[2] ** 2)
        t["lat_vel"] = pen(-c.w_lat_vel * v_body[1] ** 2)
        t["ang_xy"] = pen(-c.w_angvel_xy * (gyro[0] ** 2 + gyro[1] ** 2))
        if c.w_angmom > 0.0:
            t["angmom"] = pen(-c.w_angmom * self._angmom_pitch(mx_i, data) ** 2)
        else:
            t["angmom"] = 0.0
        # ---- the stand: zero stick only. Measured pose against the standing stance, fading in as the
        # body comes to rest so braking steps are free (config.stand_at_zero).
        if c.stand_at_zero and (c.w_stand_pose > 0.0 or c.w_stand_vel > 0.0 or c.w_stop_amp > 0.0):
            dq_stand = data.qpos[p.act_qadr] - jnp.asarray(p.default_motor_pos)
            still = jnp.exp(-(jnp.linalg.norm(v_body[:2]) / c.stand_still_mps) ** 2)
            fade = (c.stop_bill_floor + (1.0 - c.stop_bill_floor) * still) if c.stop_flag else still
            bill = pen(-c.w_stand_pose * jnp.sum(dq_stand ** 2)) + pen(-c.w_stand_vel * jnp.sum(qd ** 2))
            t["stand"] = jnp.where(standing_cmd, fade * bill, 0.0)
            if c.w_stop_amp > 0.0:
                # what the POLICY is commanding: the oscillating coefficients of the live latched spec,
                # in rad^2 (coefficient 0 of each family, the offset, is free: it may need it to balance)
                wk2 = jnp.asarray(gait.WEIGHTS)[1:] ** 2
                osc = 0.0
                for _sl, _amp in ((gait.I_S_CAM, c.cam_amp), (gait.I_S_THIGH, c.thigh_amp),
                                  (gait.I_S_HIP, c.roll_amp)):
                    _co = jnp.clip(spec[_sl], -1.0, 1.0)
                    osc = osc + (_amp ** 2) * jnp.sum(wk2 * (_co[1::2] ** 2 + _co[2::2] ** 2))
                t["stop_amp"] = jnp.where(standing_cmd, fade * pen(-c.w_stop_amp * osc), 0.0)
            else:
                t["stop_amp"] = jnp.zeros(())
        else:
            t["stand"] = jnp.zeros(())
            t["stop_amp"] = jnp.zeros(())
        sep = self._foot_sep(data)
        t["stance"] = pen(-c.w_no_cross * jnp.maximum(0.0, c.stance_min_sep - sep) ** 2)
        hr = data.qpos[p.act_qadr[p.hip_roll_idx]] - jnp.asarray(p.default_motor_pos)[p.hip_roll_idx]
        t["hip_roll"] = pen(-c.w_hip_roll * jnp.sum(hr ** 2))
        if self.library_mode:
            q_ref = gait.feedforward(spec, phi, jnp.asarray(p.nominal_ctrl), gp)
            t["track_ref"] = pen(-c.w_track_ref * jnp.sum((data.qpos[p.act_qadr] - q_ref) ** 2))
        # ---- the shaping ramp
        # Measured 2026-09-12 with tools/reward_budget.py on two cold runs (one per plant): at the
        # start of the curriculum LIVING is NEGATIVE -- income 1.87, cost 2.34, net -0.236/tick --
        # so against a one-time fall_penalty of 100 dying immediately is worth 7x staying alive, and
        # the optimiser correctly hunts for the shortest episode. That is an objective bug, not a
        # policy failure, and it is the same one recorded in reward-budget-income-floor.
        #
        # The cost is not efficiency (torque, motor_vel and energy already ramp through eff_scale,
        # and were 0.0% of it). It is GAIT QUALITY, billed at full weight to a policy that has no
        # gait yet: residual 27.6%, phase_contact 15.5%, foot_slip 12.9% -- 56% of all cost between
        # them. "Do not fight your own clock" and "do not scuff your feet" are corrections to a
        # walk; charged before there is a walk they are a tax on trying.
        #
        # So these ramp from shape_scale_start to 1. On a CLOCK, not a competence gate: a gate would
        # be circular, since what it would measure is exactly what these penalties suppress.
        # NOT ramped: upright, height, alive, track, clearance, air_time, heading, lane -- the
        # income and the safety terms, which mean the same thing on day one as at the end.
        if c.shape_curriculum_steps > 0:
            sh = params.shape_scale
            # duty_sym, swing_floor and stance_time are deliberately NOT in this list. They are the
            # guards against the two pathologies this plant falls into -- the one-legged hop
            # (walk_mit measured 4.76 Hz, left duty 0.01) and the dragged stance -- and they were
            # 0.0% of the measured cost, so ramping them down buys nothing and risks letting a hop
            # establish itself early, which this lineage has found hard to unlearn.
            for _k in ("residual", "residual_rate", "action_rate", "knob", "spec_cycle",
                       "phase_contact", "foot_slip", "step_rate",
                       "angmom", "ang_xy", "vz", "hip_roll", "lat_vel"):
                if _k in t:
                    t[_k] = jnp.asarray(t[_k], jnp.float32) * sh
        total = sum(jnp.asarray(v, dtype=jnp.float32) for v in t.values())
        book = dict(air_time=air_time, contact_time=contact_time, duty_ema=duty_ema, swing_ema=swing_ema)
        terms = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in t.items()}
        return total, terms, book

    # ------------------------------------------------------------------ public API
    def reset(self, key, params: EnvParams, override: Override = None):
        keys = jax.random.split(key, self.n_envs)
        ov = override if override is not None else Override()
        def _bc(x):
            x = jnp.asarray(x, jnp.float32)
            if x.ndim >= 1 and x.shape[-1] == gait.SPEC_DIM:       # theta: (44,) or (N, 44)
                return jnp.broadcast_to(x, (self.n_envs, gait.SPEC_DIM))
            return jnp.broadcast_to(x, (self.n_envs,))
        ov = jax.tree_util.tree_map(_bc, ov)
        return self._reset_v(keys, params, ov)

    def step(self, state, action, params: EnvParams):
        return self._step_v(state, action, params)

    def actor_slice(self, obs):
        return obs[..., :self.actor_dim]

    def priv_slice(self, obs):
        return obs[..., self.actor_dim:]
