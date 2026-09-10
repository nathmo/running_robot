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
The commit flag is in the once-block (artifact §05, rev 2026-09-09): the wrap can be moved by
a touchdown resync, so the actor cannot infer it from the phase alone; the log-prob mask reads
the same flag (info["commit"]) so rollout and update agree bit for bit.
"""
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
    pitch_assist: float = 0.0

    @classmethod
    def final(cls, cfg):
        return cls(dr_scale=1.0, sprint_dist_m=float(cfg.sprint_dist_m),
                   stance_ratio=float(cfg.stance_ratio_final), eff_scale=float(cfg.efficiency_target),
                   ctrl_jitter_ms=float(cfg.ctrl_jitter_ms_final),
                   ctrl_drop_prob=float(cfg.ctrl_drop_prob_final), pitch_assist=0.0)


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
        run = jnp.where(state.crossed, 0.0, 1.0)
        d_to_go = jnp.clip((params.sprint_dist_m - state.sprint_d) / self.cfg.task_brake_m, 0.0, 1.0)
        if self.cfg.objective == "speed":
            return jnp.array([1.0, 1.0])
        return jnp.stack([run, jnp.where(state.crossed, 0.0, d_to_go)])

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
        k_draw, k_noise, k_pose, k_lib, k_push, k_gust, k_next, k_frame = jax.random.split(key, 8)
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
        dt = self.control_dt
        dr = state.draw
        key, k_drop, k_push, k_trip, k_gust, k_jit, k_frame, k_reset, k_int = jax.random.split(state.key, 9)
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
        data = data.replace(qvel=qvel, xfrc_applied=xfrc, qfrc_applied=qfrc)
        # ---- physics: 10 substeps at a jittered timestep (the Pi's loop vs the gait clock)
        jit_ms = params.ctrl_jitter_ms * jax.random.uniform(k_jit, (), minval=-1.0, maxval=1.0)
        ts = p.sim_dt * (1.0 + jit_ms / (p.decimation * p.sim_dt * 1e3))
        mx_i = model_with(p, dr.fields, timestep=ts)
        peak = jnp.asarray(p.tau_peak)
        kt = jnp.asarray(c.motor_kt_joint)
        r_ohm = jnp.asarray(c.motor_r_ohm)

        def substep(carry, k):
            d, tau_sq, con_acc, _ = carry
            live = drive.live_command(k, dr.delay_ms, cmd_buf)
            q = d.qpos[p.act_qadr]
            qd = d.qvel[p.act_dadr]
            lim = drive.torque_limit(qd, peak, dr.torque_scale, kt, r_ohm, c.motor_bus_volts)
            tau = drive.pd_torque(q, qd, live[:6], live[6:12], live[12:18], lim)
            d = d.replace(ctrl=tau)
            d = mjx.step(mx_i, d)
            con, _, _ = self._contacts(d)
            return (d, tau_sq + tau ** 2, con_acc | con, tau), None

        (data, tau_sq, contact_acc, tau_last), _ = lax.scan(
            substep, (data, jnp.zeros(6), jnp.zeros(2, bool), jnp.zeros(6)),
            jnp.arange(p.decimation))
        # ---- thermal node
        thermal_x = drive.thermal_update(state.thermal_x, tau_sq / p.decimation, dt, c.thermal_tau_s,
                                         jnp.asarray(c.thermal_tau_cont), dr.thermal_scale) \
            if c.thermal_enable else state.thermal_x
        # ---- the clock: advance, wrap, contact resync
        phi_adv = phi + TWO_PI * f * dt
        wrapped = phi_adv >= TWO_PI
        phi_new = jnp.mod(phi_adv, TWO_PI)
        cycle_n = state.cycle_n + wrapped.astype(jnp.int32)
        resynced = jnp.where(wrapped, jnp.zeros(2, bool), state.resynced)
        grounded_c, fn, floor_viol = self._contacts(data)
        heights = self._toe_heights(data)
        grounded = contact_acc | grounded_c | (heights < c.grounded_h)
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
        commit_next = wrapped
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
        # ---- reward
        lp = float(np.exp(-dt / c.lp_yaw_tau_s))
        lp_yaw_true = lp * state.lp_yaw_true + (1.0 - lp) * gyro[2]
        roll_lp = 0.95 * state.roll_lp + 0.05 * grav[1]
        rw = self._reward(state, params, data, mx_i, spec, spec_change, residual, motor_cmd, tau_last,
                          grounded, heights, fn, v_body, grav, gyro, lp_yaw_true, crossed, sprint_d,
                          thermal_x, assist, phi)
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
                thermal_x, assist, phi):
        c, p, gp = self.cfg, self.plant, self.gp
        dt = self.control_dt
        cap = c.penalty_term_cap
        pen = lambda v: jnp.maximum(v, -cap)
        vx = v_body[0]
        if c.sprint_world_speed:
            vx = self._vel_world(data)[0]
        run_phase = (c.objective == "speed") | (~crossed)
        t = {}
        # ---- objective income
        income = c.w_fwd_speed * jnp.clip(vx, -c.v_ceiling, c.v_ceiling)
        if c.speed_upright_gate:
            u = jnp.clip((-grav[2] - c.speed_upright_c0) / (1.0 - c.speed_upright_c0), 0.0, 1.0)
            income = jnp.where(income > 0.0, income * u ** c.speed_upright_k, income)
        t["fwd_speed"] = jnp.where(run_phase, income, 0.0)
        t["stop"] = jnp.where(run_phase, 0.0, c.w_stop_vel * jnp.exp(-(vx / c.stop_sigma) ** 2))
        over = jnp.maximum(0.0, sprint_d - (params.sprint_dist_m + c.sprint_brake_m))
        t["overrun"] = jnp.where(run_phase, 0.0, pen(-c.w_overrun * over))
        t["time"] = -c.w_time if c.objective == "sprint" else 0.0
        t["alive"] = c.w_alive
        t["yaw_rate"] = pen(-c.w_yaw_rate * lp_yaw_true ** 2)
        y = self._y(data)
        t["lane"] = pen(-c.w_lane * jnp.maximum(jnp.abs(y) - c.lane_free_m, 0.0) ** 2)
        progress = jnp.where(run_phase, jnp.clip(vx / c.v_ceiling, 0.0, 1.0), 0.0)
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
