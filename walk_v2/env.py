"""DashEnvV2 — the DASH-01 Walker v2 environment on MJX, one jitted step for N envs.

One env.step == one 10 ms control tick == 10 substeps of 1 kHz MJX physics. Everything the
artifact specifies for the tick lives here, in the order of Fig. 1:

    action (50) -> latch register (spec commits only on the commit tick) -> gait generator
    (gait.py) -> drive (drive.py: kp(phi)/kd(phi) PD, torque-speed clamp, 6-18 ms delay at
    substep granularity, thermal node) -> plant (mjx, per-env randomized fields) -> sensor
    model (noise, bias, staleness) -> frame 33 -> history 10 x stride 2 + once-block +
    privileged tail (train only).

State is a flax struct; reset/step are pure functions of (state, action, params) and are
vmapped over the batch by `DashEnvV2`. Auto-reset: an env that ends is reset inside the same
step and the returned obs is the new episode's first. `EnvParams` carries the curriculum
values the trainer moves between rollouts (dr_scale, sprint line, stance ratio, ...).

Observation layouts (actor slice first, privileged tail last):
    policy variant : 330 history + 44 latched spec + 2 task + 1 commit = 377 | + 25 = 402
    library variant: 330 history + 23 once-block                       = 353 | + 25 = 378
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

FRAME_DIM = 33
PRIV_DIM = 25
TASK_DIM = 2
TWO_PI = 2.0 * np.pi


class EnvParams(NamedTuple):
    """Curriculum / runtime values, traced. Defaults = the FINAL (hardest) curriculum point."""
    dr_scale: float = 1.0
    sprint_dist_m: float = 100.0
    stance_ratio: float = 0.42
    eff_scale: float = 1.0
    ctrl_jitter_ms: float = 0.0
    ctrl_drop_prob: float = 0.0
    bringup_scale: float = 1.0   # 0 = mild starts, 1 = the full drop / tilt bands
    cmd_zero_p: float = 0.25     # share of command draws that are exactly zero (ramped)
    cmd_lo: float = 0.0          # joystick: fraction-of-v_max band the command is drawn from
    cmd_hi: float = 1.0          # (curriculum widens it down from cmd_range_start to cmd_range)
    pitch_assist: float = 0.0
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
        return cls(dr_scale=1.0, sprint_dist_m=float(cfg.sprint_dist_m),
                   cmd_zero_p=float(cfg.cmd_zero_frac),
                   cmd_lo=float(cfg.cmd_range[0]), cmd_hi=float(cfg.cmd_range[1]),
                   stance_ratio=float(cfg.stance_ratio_final), eff_scale=float(cfg.efficiency_target),
                   ctrl_jitter_ms=float(cfg.ctrl_jitter_ms_final),
                   ctrl_drop_prob=float(cfg.ctrl_drop_prob_final), pitch_assist=0.0, stoplight_prob=0.0,
                   gait_freq_lo=float(cfg.gait_freq_hz[0]))


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
    reflex_prate: jnp.ndarray
    roll_lp: jnp.ndarray
    hist: jnp.ndarray            # (hist_raw_len, 33)
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
        """(grounded_by_contact[2], normal_force[2], floor_violation) from the MJX contact
        array: foot (toe) sphere vs floor, dist < 0."""
        c = _impl(data).contact
        p = self.plant
        g = c.geom
        dist = c.dist
        efc = _impl(data).efc_force
        adr = c.efc_address
        fn_all = jnp.where(adr >= 0, efc[jnp.maximum(adr, 0)], 0.0)
        out_c, out_f, viol = [], [], jnp.zeros((), bool)
        for i in range(2):
            fg = p.foot_gids[i]
            pair = ((g[:, 0] == p.floor_gid) & (g[:, 1] == fg)) | ((g[:, 1] == p.floor_gid) & (g[:, 0] == fg))
            touching = pair & (dist < 0.0)
            out_c.append(touching.any())
            out_f.append(jnp.sum(jnp.where(touching, jnp.maximum(fn_all, 0.0), 0.0)))
            viol = viol | (touching & (dist < -0.5 * p.toe_r)).any()
            hg = p.heel_gids[i]
            hpair = ((g[:, 0] == p.floor_gid) & (g[:, 1] == hg)) | ((g[:, 1] == p.floor_gid) & (g[:, 0] == hg))
            viol = viol | (hpair & (dist < -0.5 * p.col_r[int(hg)])).any()
        return jnp.stack(out_c), jnp.stack(out_f), viol

    def _toe_heights(self, data):
        return data.geom_xpos[self.plant.foot_gids, 2] - self.plant.toe_r

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
        """One 33-dim per-tick frame through the measurement chain. Returns (frame, new noise
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
        frame = jnp.concatenate([
            mp * s["motor_pos"], mv * s["motor_vel"], mt * s["motor_torque"],
            g * s["gravity"], w * s["ang_vel"], jnp.array([lp_yaw_obs * s["ang_vel"]]),
            jnp.array([jnp.cos(phase_next), jnp.sin(phase_next)]), residual,
        ]).astype(jnp.float32)
        return frame, dict(gyro_bias=gyro_bias_walk, stale_left=stale_left, stale_frame=stale_frame,
                           lp_yaw_obs=lp_yaw_obs), accel, v_body

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
            return jnp.stack([jnp.clip(state.v_cmd / self.cfg.v_max, 0.0, 1.0), 1.0])
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

    def _priv(self, data, state, grounded, fn, accel, v_body):
        c, p = self.cfg, self.plant
        dr = state.draw
        return jnp.concatenate([
            v_body * c.obs_scales["base_vel"], grounded.astype(jnp.float32),
            jnp.array([self._base_pos(data)[2] - c.term_height, self._y(data), self._yaw(data)]),
            accel / 9.81, fn / p.bw,
            jnp.array([dr.mass_scale, dr.com_off_x * 10.0, dr.friction,
                       dr.kp_scale.mean(), dr.torque_scale, dr.delay_ms / 10.0]),
            state.thermal_x,
        ]).astype(jnp.float32)

    def _obs(self, state, params, data, grounded, fn, accel, v_body, phase_next, commit_next):
        hist = state.hist[self.hist_idx].reshape(-1)
        once = self._once(state, params, phase_next, commit_next)
        priv = self._priv(data, state, grounded, fn, accel, v_body)
        return jnp.concatenate([hist, once, priv]).astype(jnp.float32)

    # ------------------------------------------------------------------ reset
    def _reset_one(self, key, params: EnvParams, ov: Override):
        c, p, gp = self.cfg, self.plant, self.gp
        k_draw, k_noise, k_pose, k_lib, k_push, k_gust, k_next, k_frame, k_light = jax.random.split(key, 9)
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
        theta = theta.at[39:44].add(c.library_box_knob * jax.random.uniform(kl3, (5,), minval=-1.0, maxval=1.0))
        ov_theta = jnp.broadcast_to(jnp.asarray(ov.theta, jnp.float32), (gait.SPEC_DIM,))
        theta = jnp.where(jnp.isnan(ov_theta), theta, ov_theta)
        v_ref = jnp.asarray(lib["v"][i_lib])
        spec0 = jnp.clip(theta, -1.0, 1.0) if self.library_mode else jnp.zeros(gait.SPEC_DIM)
        data = mjx.forward(mx_i, data)
        nominal = jnp.asarray(p.nominal_ctrl)
        cmd0 = jnp.concatenate([nominal, jnp.asarray(gp.drive_kp), jnp.asarray(gp.drive_kd)])
        k_cmd0, k_bring = jax.random.split(k_noise)
        v_cmd0, cmd_left0 = self._draw_cmd(k_cmd0, params)
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
            reflex_prate=jnp.zeros(()), roll_lp=jnp.zeros(()),
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
            v_cmd=v_cmd0, cmd_left=cmd_left0, hold_s=hold_s0, grace_left=grace0,
            theta=jnp.clip(theta, -1.0, 1.0), v_ref=v_ref, raibert_i=jnp.zeros(()),
            ep_return=jnp.zeros(()), ep_len=jnp.zeros((), jnp.int32),
        )
        grounded_c, fn, _ = self._contacts(data)
        grounded = grounded_c | (self._toe_heights(data) < c.grounded_h)
        frame, ns, accel, v_body = self._frame(data, jnp.zeros(6), jnp.zeros(6), 0.0, state, k_frame)
        state = state.replace(hist=jnp.tile(frame[None], (self.hist_raw_len, 1)),
                              grounded_prev=grounded, prev_toe_xy=self._toe_pos(data)[:, :2],
                              prev_vel_body=v_body, **ns)
        obs = self._obs(state, params, data, grounded, fn, accel, v_body, 0.0, 1.0)
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
        p_drop = c.bringup_drop_frac * s
        p_held = c.bringup_held_frac * s
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
        # params.cmd_zero_p, not cfg.cmd_zero_frac: a fixed zero share would ignore the command
        # curriculum completely and ask a warm-started runner to stop a quarter of the time from
        # step 0, which drags the gait clock onto the slow rail it never comes back from.
        v = jnp.where(jax.random.uniform(k_z) < params.cmd_zero_p, 0.0, v)
        left = c.cmd_interval_s * jax.random.uniform(k_t, (), minval=0.6, maxval=1.4)
        return v, left

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
        o_cam = th[41] - (c.raibert_kp * e_v + c.raibert_ki * state.raibert_i) / gp.o_max[0]
        o_hip = th[43] - (c.raibert_ky * v_body[1] + c.raibert_kr * state.roll_lp) / gp.o_max[2]
        spec = spec.at[41].set(o_cam).at[43].set(o_hip)
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
        # ---- current clean signals for the reflexes
        grav = self._grav_body(data)
        gyro = self._gyro(data)
        v_body_pre = self._vel_body(data)
        roll, roll_rate, pitch = grav[1], gyro[0], grav[0]
        if c.pitch_reflex_rate_lp > 0.0:
            prate = c.pitch_reflex_rate_lp * state.reflex_prate + (1.0 - c.pitch_reflex_rate_lp) * gyro[1]
        else:
            prate = gyro[1]
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
        target, kp, kd, q_ref = gait.assemble(spec, residual, phi, roll, roll_rate, pitch, prate,
                                              jnp.asarray(p.nominal_ctrl), gp)
        target = jnp.clip(target, jnp.asarray(p.q_lo), jnp.asarray(p.q_hi))
        target, tvel = drive.slew_limit(target, state.prev_target, state.prev_target_vel,
                                        jnp.asarray(c.motor_vel_limit), c.motor_accel_limit, dt)
        motor_cmd = (target - jnp.asarray(p.nominal_ctrl)) / c.action_scale
        cmd = jnp.concatenate([target + dr.joint_zero, kp * dr.kp_scale, kd * dr.kv_scale])
        cmd_buf = jnp.stack([cmd, state.cmd_buf[0], state.cmd_buf[1]])
        # ---- disturbances at tick start
        adv = params.dr_scale if c.adversity_curriculum else 1.0
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
        # pitch assist (sim-only training wheel, faded by the curriculum)
        qfrc = jnp.zeros(p.nv)
        if c.pitch_assist_kp > 0.0:
            pq = data.qpos[p.base_q["pitch"]]
            pqd = data.qvel[p.base_d["pitch"]]
            assist = -params.pitch_assist * (c.pitch_assist_kp * pq + c.pitch_assist_kd * pqd)
            qfrc = qfrc.at[p.base_d["pitch"]].set(assist)
        else:
            assist = jnp.zeros(())
        if c.roll_assist_kp > 0.0 and p.base_d["roll"] >= 0:
            # S2 roll wheel: same fade scalar, own gains; billed with the pitch torque (assist_pen = w*(ap^2+ar^2))
            rq = data.qpos[p.base_q["roll"]]
            rqd = data.qvel[p.base_d["roll"]]
            assist_r = -params.pitch_assist * (c.roll_assist_kp * rq + c.roll_assist_kd * rqd)
            qfrc = qfrc.at[p.base_d["roll"]].set(assist_r)
            assist = jnp.sqrt(assist ** 2 + assist_r ** 2)
        if c.yaw_assist_kp > 0.0 and p.base_d["yaw"] >= 0:
            yq = data.qpos[p.base_q["yaw"]]
            yqd = data.qvel[p.base_d["yaw"]]
            assist_y = -params.pitch_assist * (c.yaw_assist_kp * yq + c.yaw_assist_kd * yqd)
            qfrc = qfrc.at[p.base_d["yaw"]].set(assist_y)
            assist = jnp.sqrt(assist ** 2 + assist_y ** 2)
        data = data.replace(qvel=qvel, xfrc_applied=xfrc, qfrc_applied=qfrc)
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
                           z=jnp.where(params.hold_z > 0.0, params.hold_z,
                                       p.key_qpos[p.base_q["z"]]),
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
            con, _, _ = self._contacts(d)
            return (d, tau_sq + tau ** 2, con_acc | con, tau), None

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
        grounded_c, fn, floor_viol = self._contacts(data)      # reward + privileged tail only
        heights = self._toe_heights(data)
        grounded = contact_acc | grounded_c | (heights < c.grounded_h)
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
            v_new, left_new = self._draw_cmd(k_cmd, params)
            v_cmd = jnp.where(due, v_new, state.v_cmd)
            cmd_left = jnp.where(due, left_new, state.cmd_left - dt)
        else:
            v_cmd, cmd_left = state.v_cmd, state.cmd_left
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
        rw = self._reward(state, params, data, mx_i, spec, spec_change, residual, motor_cmd, tau_last,
                          grounded, heights, fn, v_body, grav, gyro, lp_yaw_true, crossed, sprint_d,
                          thermal_x, assist, phi, stop_now, v_target)
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
            lp_yaw_true=lp_yaw_true, reflex_prate=prate, roll_lp=roll_lp, hist=hist,
            air_time=book["air_time"], contact_time=book["contact_time"], grounded_prev=grounded,
            prev_toe_xy=self._toe_pos(data)[:, :2], duty_ema=book["duty_ema"], ws_out_t=ws_out_t,
            swing_ema=book["swing_ema"], push_countdown=push_countdown, trip_left=trip_left,
            trip_foot=trip_foot, trip_force=trip_force, gust_left=gust_left,
            gust_countdown=gust_countdown, gust_dir=gust_dir, sprint_d=sprint_d, crossed=crossed,
            t_line=t_line, stop_hold=stop_hold,
            light_red=light_red, light_left=light_left, light_v0=light_v0, light_t=light_t,
            light_floor=light_floor, v_cmd=v_cmd, cmd_left=cmd_left,
            hold_s=state.hold_s, grace_left=grace_left,
            raibert_i=jnp.clip(state.raibert_i + (v_body[0] - state.v_ref) * dt,
                               -c.raibert_imax, c.raibert_imax),
            ep_return=state.ep_return + reward, ep_len=state.ep_len + 1, **ns)
        obs = self._obs(new_state, params, data, grounded, fn, accel, v_body_m, phi2, commit_next.astype(jnp.float32))
        info = dict(
            commit=commit, reward_terms=terms, foot_air=(~grounded).astype(jnp.float32),
            fallen=fallen, finished=finished, truncated=truncated,
            sprint_d=sprint_d, t_line=t_line, ep_return=new_state.ep_return,
            ep_len=new_state.ep_len, thermal_max=thermal_x.max(),
            torque_util=jnp.mean(jnp.abs(tau_last) / jnp.asarray(p.tau_peak)),
            freq_hz=f, spec_change=spec_change, resync=can.any(),
            residual_sat=jnp.mean(jnp.abs(residual) >= 0.95), lateral_y=self._y(data),
            term_low=term_low, term_tip=term_tip, term_floor=floor_viol, term_ws=ws_kill, term_nan=~finite,
            light_red=light_red.astype(jnp.float32),
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
                thermal_x, assist, phi, stop_now=None, v_target=None):
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
            err_cmd = jnp.abs(vx - state.v_cmd)
            income = ((c.w_track + c.w_fwd_speed * jnp.clip(state.v_cmd, 0.0, c.v_ceiling))
                      * jnp.exp(-err_cmd / c.track_sigma))
            if c.speed_upright_gate:
                u = jnp.clip((-grav[2] - c.speed_upright_c0) / (1.0 - c.speed_upright_c0), 0.0, 1.0)
                income = income * u ** c.speed_upright_k
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
        t["alive"] = c.w_alive
        t["yaw_rate"] = pen(-c.w_yaw_rate * lp_yaw_true ** 2)
        y = self._y(data)
        t["lane"] = pen(-c.w_lane * jnp.maximum(jnp.abs(y) - c.lane_free_m, 0.0) ** 2)
        progress = jnp.where(run_phase, jnp.clip(vx / c.v_ceiling, 0.0, 1.0), 0.0)
        # Under the continuous stop command the commanded speed FOLLOWS THE RAMP instead of dropping to
        # zero the instant the light turns. With the binary version the whole gait block (air-time
        # credit, swing floor, stance time, clearance, phase contact -- everything below `gait_on`)
        # switched off for the entire deceleration, i.e. exactly while the robot has to hold a gait
        # together through 2.5 -> 2.0 -> 1.0 -> 0 m/s, the regime it has never been shaped in. Now the
        # shaping tracks the command down and only lets go below gait_cmd_gate (a genuine standstill).
        if c.objective == "joystick":
            # the commanded speed IS the joystick, and the gait block stays on all the way down to
            # zero: a zero command means step in place, not stand still, because this plant has no
            # passive stance to hold (bring-up probe: it topples in 0.7-1.0 s with no gait)
            cmd_speed = state.v_cmd
            gait_on = jnp.ones((), bool)
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
        t["spec_cycle"] = pen(-c.w_spec_cycle * spec_change)
        t["knob"] = pen(-c.w_knob * jnp.sum(gait.knobs(spec, gp) ** 2))
        t["residual"] = pen(-c.w_residual * jnp.sum(residual ** 2))
        t["residual_rate"] = pen(-c.w_residual_rate * jnp.sum((residual - state.prev_residual) ** 2))
        t["thermal"] = pen(-c.w_thermal * jnp.sum(jnp.maximum(thermal_x - c.thermal_penalty_frac, 0.0) ** 2)) \
            if c.thermal_enable else 0.0
        t["assist_pen"] = pen(-c.w_assist_penalty * assist ** 2)
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
        sep = self._foot_sep(data)
        t["stance"] = pen(-c.w_no_cross * jnp.maximum(0.0, c.stance_min_sep - sep) ** 2)
        hr = data.qpos[p.act_qadr[p.hip_roll_idx]] - jnp.asarray(p.default_motor_pos)[p.hip_roll_idx]
        t["hip_roll"] = pen(-c.w_hip_roll * jnp.sum(hr ** 2))
        if self.library_mode:
            q_ref = gait.feedforward(spec, phi, jnp.asarray(p.nominal_ctrl), gp)
            t["track_ref"] = pen(-c.w_track_ref * jnp.sum((data.qpos[p.act_qadr] - q_ref) ** 2))
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
