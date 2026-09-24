"""BalanceEnv -- DASH-01 standing still while being shoved, on MJX. One jitted step for N envs.

One env.step == one 10 ms control tick == 10 substeps of 1 kHz MJX physics:

    action (18) = the MIT frame [q 6 | kp 6 | kd 6]  -> joint range + no-load slew cap
    -> drive (drive.py: PD + back-EMF torque clamp, 6-18 ms transport delay at substep granularity)
    -> plant (per-env randomized fields incl. the whole-robot CoM shift, a jittered timestep)
    -> sensor model (RLframework's chain: noise, biases, IMU mount, accel leak, dropout)
    -> frame 42 -> history 10 x stride 2 -> actor obs 420 | + privileged tail (critic only)

THE PUSH is a force pulse on the torso: azimuth uniform, elevation +-cfg.push_elev_deg, applied at a
random point of the torso (so it torques it too), lasting 50-150 ms. Its size is the impulse,
expressed as the whole-robot velocity change dv = J / M -- `EnvParams.push_level` is the top of the
draw and is what the curriculum moves. A push is SURVIVED when the robot is still up
`push_survive_s` after it started; the step reports each verdict as an event (info) and the trainer
gates on those, not on episode length.

Actor frame (per tick, the deploy runtime builds the same thing -- controller_balance.py):
    [q - q_stance 6, qd 6, tau 6, gravity 3, gyro 3, previous action 18]  x obs_scales
The previous action is the CLIPPED action that reached the drive mapping (after the drop model).
"""
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from flax import struct
from mujoco import mjx

import drive
from plant import Plant, PlantDraw, Override, draw_plant, model_with

N_ACT = 6
ACTION_DIM = 18
FRAME_DIM = 42
ONCE_DIM = 18          # slow block: [ema g_xy 2, ema g_xy (mid) 2, ema w_xy 2, ema (q - q_nom) 6, ema tau 6]
TWO_PI = 2.0 * np.pi
# mirror: L <-> R. Joint-space quantities also negate (RLframework/gait.py mirror_frame).
MIRROR_PERM = np.array([3, 4, 5, 0, 1, 2])


class EnvParams(NamedTuple):
    """What the trainer's curricula move between rollouts (traced, not static)."""
    plant_scale: float = 1.0        # width of the plant draw, CoM shift included
    push_level: float = 1.0         # m/s: the top of the push draw
    push_on: float = 1.0            # 0 = no pushes at all (the eval's "quiet" condition)


@struct.dataclass
class EnvState:
    data: mjx.Data
    draw: PlantDraw
    key: jnp.ndarray
    step_n: jnp.ndarray
    t: jnp.ndarray
    cmd_buf: jnp.ndarray            # (3, 18) [target, kp, kd] of the last three ticks
    prev_target: jnp.ndarray
    prev_target_vel: jnp.ndarray
    prev_action: jnp.ndarray        # (18,) clipped
    prev_vel_body: jnp.ndarray
    prev_foot_xy: jnp.ndarray       # (2, 2)
    ground_ticks: jnp.ndarray       # (2,) consecutive ticks each foot has been down
    hist: jnp.ndarray               # (hist_raw_len, FRAME_DIM)
    gyro_bias: jnp.ndarray
    stale_left: jnp.ndarray
    stale_frame: jnp.ndarray
    # the slow (once-block) channels: leaky integrals of the MEASURED signals
    ema_g: jnp.ndarray              # (2,) gravity x, y      tau = cfg.ema_slow_s
    ema_gm: jnp.ndarray             # (2,) gravity x, y      tau = cfg.ema_mid_s
    ema_w: jnp.ndarray              # (2,) gyro x, y         tau = cfg.ema_fast_s
    ema_q: jnp.ndarray              # (6,) measured q - q_nom
    ema_t: jnp.ndarray              # (6,) measured tau / tau_peak
    # ---- the push
    push_countdown: jnp.ndarray     # ticks until the next push starts
    push_left: jnp.ndarray          # ticks of force remaining in the current push
    push_force: jnp.ndarray         # (3,) world N
    push_point: jnp.ndarray         # (3,) torso frame, m
    push_dv: jnp.ndarray            # the current / last push's size, m/s
    push_hard: jnp.ndarray          # drawn from the top band [0.5 L, L]
    push_pending: jnp.ndarray       # awaiting its survival verdict
    push_since: jnp.ndarray         # s since the last push started (large = none yet)
    calm_t: jnp.ndarray             # s since the last push ENDED
    push_fix: jnp.ndarray           # (3,) [dv, azimuth, elevation]; NaN = drawn (eval overrides)
    hold_s: jnp.ndarray             # seconds the base is held by the operator at the start
    hold_qpos: jnp.ndarray          # (6,) the base pose it is held at
    release_v: jnp.ndarray          # (3,) velocity imparted at the release tick
    ep_return: jnp.ndarray
    ep_len: jnp.ndarray


def _impl(d):
    return getattr(d, "_impl", d)


def mirror_action(a, xp=jnp):
    """M on the 18-dim action: positions swap sides and negate, gains swap sides."""
    q, kp, kd = a[..., 0:6], a[..., 6:12], a[..., 12:18]
    return xp.concatenate([-q[..., MIRROR_PERM], kp[..., MIRROR_PERM], kd[..., MIRROR_PERM]], axis=-1)


def mirror_frame(f, xp=jnp):
    """M on one 42-dim frame: joints swap + negate, gravity y and gyro x/z negate, the previous
    action as `mirror_action`."""
    sw = lambda x: -x[..., MIRROR_PERM]
    q, v, t = f[..., 0:6], f[..., 6:12], f[..., 12:18]
    g, w = f[..., 18:21], f[..., 21:24]
    g2 = xp.stack([g[..., 0], -g[..., 1], g[..., 2]], axis=-1)
    w2 = xp.stack([-w[..., 0], w[..., 1], -w[..., 2]], axis=-1)
    return xp.concatenate([sw(q), sw(v), sw(t), g2, w2, mirror_action(f[..., 24:42], xp)], axis=-1)


def mirror_once(o, xp=jnp):
    """M on the 18-dim slow block: [g_xy, g_xy(mid), w_xy, q-q_nom 6, tau 6]."""
    sw = lambda x: -x[..., MIRROR_PERM]
    flip_g = lambda g: xp.stack([g[..., 0], -g[..., 1]], axis=-1)
    w = o[..., 4:6]
    w2 = xp.stack([-w[..., 0], w[..., 1]], axis=-1)
    return xp.concatenate([flip_g(o[..., 0:2]), flip_g(o[..., 2:4]), w2,
                           sw(o[..., 6:12]), sw(o[..., 12:18])], axis=-1)


def mirror_actor(obs, n_hist, xp=jnp):
    """M on the whole actor observation: every history frame, then the slow block."""
    n = FRAME_DIM * n_hist
    H = obs[..., :n].reshape(obs.shape[:-1] + (n_hist, FRAME_DIM))
    return xp.concatenate([mirror_frame(H, xp).reshape(obs.shape[:-1] + (n,)),
                           mirror_once(obs[..., n:n + ONCE_DIM], xp)], axis=-1)


def gain_map(a, k0, k_lo, k_hi, xp=jnp):
    """a in [-1, 1] -> gain. a = 0 is k0 exactly; +1 is k_hi, -1 is k_lo, log-linear on each side."""
    up = k0 * (k_hi / k0) ** xp.clip(a, 0.0, 1.0)
    dn = k0 * (k0 / k_lo) ** xp.clip(a, -1.0, 0.0)
    return xp.where(a >= 0.0, up, dn)


class BalanceEnv:
    def __init__(self, cfg, n_envs=None):
        self.cfg = cfg
        self.plant = p = Plant(cfg)
        self.n_envs = int(n_envs or cfg.n_envs)
        self.control_dt = p.control_dt
        self.max_steps = int(round(cfg.episode_s / self.control_dt))
        self.hist_stride = int(cfg.history_stride)
        self.hist_raw_len = (cfg.history_len - 1) * self.hist_stride + 1
        self.hist_idx = np.array((self.hist_raw_len - 1)
                                 - (np.arange(cfg.history_len) * self.hist_stride)[::-1])
        self.action_dim = ACTION_DIM
        self.frame_dim = FRAME_DIM
        self.once_dim = ONCE_DIM
        self.actor_dim = FRAME_DIM * cfg.history_len + ONCE_DIM
        self.a_slow = float(np.exp(-self.control_dt / cfg.ema_slow_s))
        self.a_mid = float(np.exp(-self.control_dt / cfg.ema_mid_s))
        self.a_fast = float(np.exp(-self.control_dt / cfg.ema_fast_s))
        self.a_filt = float(np.exp(-self.control_dt / cfg.action_filter_tau_s))             if cfg.action_filter_tau_s > 0 else 0.0
        self.priv_dim = 25
        self.obs_dim = self.actor_dim + self.priv_dim
        # the action map's constants
        self.nominal = jnp.asarray(p.nominal_ctrl)
        self.q_scale = jnp.asarray(cfg.q_scale)
        self.kp0, self.kd0 = jnp.asarray(cfg.drive_kp), jnp.asarray(cfg.drive_kd)
        self.q_lo, self.q_hi = jnp.asarray(p.q_lo), jnp.asarray(p.q_hi)
        self.m_total = float(p.total_mass)
        self.BASE = ("x", "y", "z", "roll", "pitch", "yaw")
        self.base_qadr = np.array([p.base_q[n] for n in self.BASE])
        self.base_dadr = np.array([p.base_d[n] for n in self.BASE])
        self._reset_v = jax.jit(jax.vmap(self._reset_one, in_axes=(0, None, 0)))
        self._step_v = jax.jit(jax.vmap(self._step_one, in_axes=(0, 0, None)))

    # ------------------------------------------------------------------ the action -> MIT frame
    def reflex(self, grav, gyro):
        """The stabilising prior, in joint space, from the MEASURED gravity and gyro (config
        reflex_*). Mirrored L/R like every other joint quantity. Zero when reflex_enable is off."""
        c = self.cfg
        if not c.reflex_enable:
            return jnp.zeros(6)
        lean = jnp.clip(-(c.reflex_kp_pitch * grav[0] + c.reflex_kd_pitch * gyro[1]),
                        -c.reflex_clip_rad, c.reflex_clip_rad)
        roll = jnp.clip(-(c.reflex_kp_roll * grav[1] + c.reflex_kd_roll * gyro[0]),
                        -c.reflex_clip_rad, c.reflex_clip_rad)
        return jnp.array([roll, 0.0, lean, -roll, 0.0, -lean])

    def action_to_frame(self, a, grav, gyro):
        """(target before the slew cap, kp, kd) from a clipped action and the measured attitude.
        controller_balance.py is the numpy twin of this, of the reflex and of the slew cap after it."""
        c = self.cfg
        q = jnp.clip(self.nominal + self.q_scale * a[0:6] + self.reflex(grav, gyro),
                     self.q_lo, self.q_hi)
        kp = gain_map(a[6:12], self.kp0, c.kp_lo, c.kp_hi)
        kd = gain_map(a[12:18], self.kd0, c.kd_lo, c.kd_hi)
        return q, kp, kd

    # ------------------------------------------------------------------ helpers
    def _R(self, data):
        return data.xmat[self.plant.base_bid].reshape(3, 3)

    def _grav_body(self, data):
        return self._R(data).T @ jnp.array([0.0, 0.0, -1.0])

    def _gyro(self, data):
        a = self.plant.gyro_adr
        return data.sensordata[a:a + 3]

    def _vel_world(self, data):
        p = self.plant
        return jnp.stack([data.qvel[p.base_d["x"]], data.qvel[p.base_d["y"]], data.qvel[p.base_d["z"]]])

    def _vel_body(self, data):
        return self._R(data).T @ self._vel_world(data)

    def _contacts(self, data):
        """(toe_down[2], heel_down[2], floor_violation) from the MJX contact array."""
        con = _impl(data).contact
        p = self.plant
        g, dist = con.geom, con.dist

        def pad(gid):
            pair = ((g[:, 0] == p.floor_gid) & (g[:, 1] == gid)) | ((g[:, 1] == p.floor_gid) & (g[:, 0] == gid))
            touching = pair & (dist < 0.0)
            return touching.any(), (touching & (dist < -0.01)).any()

        toe, heel, viol = [], [], jnp.zeros((), bool)
        for i in range(2):
            t_on, t_v = pad(p.foot_gids[i])
            h_on, h_v = pad(p.heel_gids[i])
            toe.append(t_on)
            heel.append(h_on)
            viol = viol | t_v | h_v
        return jnp.stack(toe), jnp.stack(heel), viol

    def _foot_xy(self, data):
        return data.xpos[jnp.asarray(self.plant.foot_bids)][:, :2]

    def _com_xy(self, data):
        """Whole-robot centre of mass, world xy (MJX keeps it in subtree_com[0])."""
        return data.subtree_com[0][:2]

    def _support_xy(self, data, grounded):
        """Centre of the support: the mean of the feet that are down (both if none is)."""
        f = self._foot_xy(data)
        w = grounded.astype(jnp.float32)
        w = jnp.where(w.sum() > 0, w, jnp.ones(2))
        return (w[:, None] * f).sum(0) / w.sum()

    # ------------------------------------------------------------------ observation
    def _frame(self, data, tau_last, action, state, key):
        """One 42-dim frame through the measurement chain (RLframework/env.py _frame, minus the
        walker's phase / yaw / heading channels). Also returns the MEASURED signals, which the
        deploy parity test feeds to the numpy runtime."""
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
        gyro_bias = dr.gyro_bias0 + state.gyro_bias
        gyro_bias_walk = state.gyro_bias + on * c.noise_gyro_walk * jax.random.normal(k1, (3,))
        mp = q - dr.joint_zero + on * c.noise_encoder * jax.random.normal(k2, (6,))
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
        frame = jnp.concatenate([mp * s["motor_pos"], mv * s["motor_vel"], mt * s["motor_torque"],
                                 g * s["gravity"], w * s["ang_vel"], action]).astype(jnp.float32)
        meas = dict(pos=mp + jnp.asarray(p.default_motor_pos), vel=mv, tau=mt, grav=g, gyro=w)
        return frame, dict(gyro_bias=gyro_bias_walk, stale_left=stale_left,
                           stale_frame=stale_frame), v_body, meas

    def _priv(self, data, state, toe, heel, v_body):
        p, d = self.plant, state.draw
        R = self._R(data)
        base = data.xpos[p.base_bid]
        feet = jnp.stack([(R.T @ (data.xpos[b] - base))[:2] for b in p.foot_bids]).reshape(-1)
        active = (state.push_left > 0).astype(jnp.float32)
        return jnp.concatenate([
            v_body,                                                        # the estimator's target
            jnp.array([base[2] - p.height_stand]),
            toe.astype(jnp.float32), heel.astype(jnp.float32),
            d.com_shift * 10.0,
            state.push_force * active / (self.m_total * 10.0), active[None],
            jnp.array([jnp.minimum(state.push_since, 3.0) / 3.0]),
            jnp.array([d.mass_scale, d.friction, d.kp_scale.mean(), d.torque_scale, d.delay_ms / 10.0]),
            feet * 5.0,
        ]).astype(jnp.float32)

    def _once(self, state):
        """The slow block, from the leaky integrals of what the robot MEASURED."""
        return jnp.concatenate([state.ema_g, state.ema_gm, state.ema_w,
                                state.ema_q, state.ema_t]).astype(jnp.float32)

    def _ema(self, state, meas):
        """One tick of the slow channels (measured signals only)."""
        p = self.plant
        s = self.cfg.obs_scales
        q_err = meas["pos"] - jnp.asarray(p.default_motor_pos)
        return dict(
            ema_g=self.a_slow * state.ema_g + (1.0 - self.a_slow) * meas["grav"][:2],
            ema_gm=self.a_mid * state.ema_gm + (1.0 - self.a_mid) * meas["grav"][:2],
            ema_w=self.a_fast * state.ema_w + (1.0 - self.a_fast) * meas["gyro"][:2],
            ema_q=self.a_slow * state.ema_q + (1.0 - self.a_slow) * q_err,
            ema_t=self.a_slow * state.ema_t + (1.0 - self.a_slow) * meas["tau"] / jnp.asarray(p.tau_peak))

    def _obs(self, state, data, toe, heel, v_body):
        hist = state.hist[self.hist_idx].reshape(-1)
        return jnp.concatenate([hist, self._once(state),
                                self._priv(data, state, toe, heel, v_body)]).astype(jnp.float32)

    # ------------------------------------------------------------------ reset
    def _interval(self, key, rng):
        s = jax.random.uniform(key, (), minval=rng[0], maxval=rng[1])
        return jnp.maximum(1, jnp.round(s / self.control_dt)).astype(jnp.int32)

    def _reset_one(self, key, params: EnvParams, ov: Override):
        c, p = self.cfg, self.plant
        k_draw, k_pose, k_push, k_next, k_frame, k_hold = jax.random.split(key, 6)
        draw = draw_plant(k_draw, c, p, params.plant_scale, ov)
        mx_i = model_with(p, draw.fields)
        qpos = jnp.asarray(p.key_qpos)
        noise = c.reset_joint_noise * jax.random.uniform(k_pose, (p.leg_hinge_qadr.size,), minval=-1.0, maxval=1.0)
        qpos = qpos.at[jnp.asarray(p.leg_hinge_qadr)].add(noise)
        data = p.data0.replace(qpos=qpos, qvel=jnp.zeros(p.nv), ctrl=jnp.zeros(p.nu),
                               qfrc_applied=jnp.zeros(p.nv),
                               xfrc_applied=jnp.zeros_like(p.data0.xfrc_applied), time=jnp.zeros(()))
        data = mjx.forward(mx_i, data)
        # THE OPERATOR HOLD: the base is supported for hold_s, then let go with a little stray
        # velocity -- the bring-up the runner performs (crawl to the stance, hold, release).
        kh1, kh2 = jax.random.split(k_hold)
        hold_s = jax.random.uniform(kh1, (), minval=c.hold_s_range[0], maxval=c.hold_s_range[1])
        hold_qpos = jnp.stack([qpos[p.base_q[n]] for n in self.BASE])
        release_v = c.hold_release_v * jax.random.normal(kh2, (3,))
        cmd0 = jnp.concatenate([self.nominal, self.kp0, self.kd0])
        toe, heel, _ = self._contacts(data)
        state = EnvState(
            data=data, draw=draw, key=k_next, step_n=jnp.zeros((), jnp.int32), t=jnp.zeros(()),
            cmd_buf=jnp.stack([cmd0, cmd0, cmd0]), prev_target=self.nominal,
            prev_target_vel=jnp.zeros(6), prev_action=jnp.zeros(ACTION_DIM),
            prev_vel_body=jnp.zeros(3), prev_foot_xy=self._foot_xy(data),
            ground_ticks=jnp.zeros(2, jnp.int32),
            hist=jnp.zeros((self.hist_raw_len, FRAME_DIM), jnp.float32),
            gyro_bias=jnp.zeros(3), stale_left=jnp.zeros((), jnp.int32), stale_frame=jnp.zeros(6),
            ema_g=jnp.zeros(2), ema_gm=jnp.zeros(2), ema_w=jnp.zeros(2), ema_q=jnp.zeros(6),
            ema_t=jnp.zeros(6),
            hold_s=hold_s, hold_qpos=hold_qpos, release_v=release_v,
            push_countdown=self._interval(k_push, c.push_first_s),
            push_left=jnp.zeros((), jnp.int32), push_force=jnp.zeros(3), push_point=jnp.zeros(3),
            push_dv=jnp.zeros(()), push_hard=jnp.zeros((), bool), push_pending=jnp.zeros((), bool),
            push_since=jnp.full((), 99.0), calm_t=jnp.full((), 99.0),
            push_fix=jnp.stack([jnp.asarray(ov.push_dv, jnp.float32), jnp.asarray(ov.push_az, jnp.float32),
                                jnp.asarray(ov.push_el, jnp.float32)]),
            ep_return=jnp.zeros(()), ep_len=jnp.zeros((), jnp.int32))
        # the first frame carries zero torque and a zero previous action, as the runtime's start()
        frame, ns, v_body, meas = self._frame(data, jnp.zeros(6), jnp.zeros(ACTION_DIM), state, k_frame)
        # the slow channels start AT the first measurement, not at zero: an EMA seeded at zero would
        # take seconds to mean anything, and the runtime's start() seeds them the same way
        state = state.replace(hist=jnp.tile(frame[None], (self.hist_raw_len, 1)), prev_vel_body=v_body,
                              ema_g=meas["grav"][:2], ema_gm=meas["grav"][:2], ema_w=jnp.zeros(2),
                              ema_q=meas["pos"] - jnp.asarray(p.default_motor_pos),
                              ema_t=jnp.zeros(6), **ns)
        return state, self._obs(state, data, toe, heel, v_body)

    # ------------------------------------------------------------------ step
    def _push(self, state, key, params):
        """Start a push if one is due. Returns the updated push fields and the force this tick."""
        c = self.cfg
        k_dir, k_el, k_band, k_dv, k_dur, k_pt, k_next = jax.random.split(key, 7)
        due = (state.push_countdown <= 1) & (params.push_on > 0.0) & (state.t >= state.hold_s)
        L = params.push_level
        easy = jax.random.uniform(k_band) < c.push_easy_frac
        u = jax.random.uniform(k_dv)
        dv = jnp.where(easy, 0.5 * L * u, L * (0.5 + 0.5 * u))
        az = jax.random.uniform(k_dir, (), minval=0.0, maxval=TWO_PI)
        el = jnp.deg2rad(c.push_elev_deg) * jax.random.uniform(k_el, (), minval=-1.0, maxval=1.0)
        fix = state.push_fix
        dv = jnp.where(jnp.isnan(fix[0]), dv, fix[0])
        easy = jnp.where(jnp.isnan(fix[0]), easy, False)
        az = jnp.where(jnp.isnan(fix[1]), az, fix[1])
        el = jnp.where(jnp.isnan(fix[2]), el, fix[2])
        direction = jnp.array([jnp.cos(el) * jnp.cos(az), jnp.cos(el) * jnp.sin(az), jnp.sin(el)])
        dur = jax.random.uniform(k_dur, (), minval=c.push_dur_s[0], maxval=c.push_dur_s[1])
        n_ticks = jnp.maximum(1, jnp.round(dur / self.control_dt)).astype(jnp.int32)
        # the impulse is M dv, delivered over the ROUNDED duration so the level is exact
        force = direction * self.m_total * dv / (n_ticks * self.control_dt)
        box = jnp.asarray(c.push_point_box)
        point = box[:, 0] + (box[:, 1] - box[:, 0]) * jax.random.uniform(k_pt, (3,))
        new = dict(
            push_countdown=jnp.where(due, self._interval(k_next, c.push_interval_s), state.push_countdown - 1),
            push_left=jnp.where(due, n_ticks, state.push_left),
            push_force=jnp.where(due, force, state.push_force),
            push_point=jnp.where(due, point, state.push_point),
            push_dv=jnp.where(due, dv, state.push_dv),
            push_hard=jnp.where(due, ~easy, state.push_hard),
            push_pending=state.push_pending | due,
            push_since=jnp.where(due, 0.0, state.push_since),
        )
        return new, due

    def _step_one(self, state: EnvState, action, params: EnvParams):
        c, p = self.cfg, self.plant
        dt = self.control_dt
        dr = state.draw
        key, k_drop, k_push, k_jit, k_frame, k_reset = jax.random.split(state.key, 6)
        action = jnp.clip(action, -1.0, 1.0)
        drop = jax.random.uniform(k_drop) < c.ctrl_drop_prob * params.plant_scale
        action = jnp.where(drop, state.prev_action, action)
        data = state.data
        # ---- the MIT frame. The reflex reads the NEWEST observation frame -- the same measurement
        # the policy just acted on, and the only attitude the robot has at command time.
        s = c.obs_scales
        newest = state.hist[-1]
        g_meas = newest[18:21] / s["gravity"]
        w_meas = newest[21:24] / s["ang_vel"]
        target, kp, kd = self.action_to_frame(action, g_meas, w_meas)
        # one-pole filter on the target: the sampled exploration noise is mostly disturbance here
        target = self.a_filt * state.prev_target + (1.0 - self.a_filt) * target
        target, tvel = drive.slew_limit(target, state.prev_target, state.prev_target_vel,
                                        jnp.asarray(c.motor_vel_limit), c.motor_accel_limit, dt)
        cmd = jnp.concatenate([target + dr.joint_zero, kp * dr.kp_scale, kd * dr.kv_scale])
        cmd_buf = jnp.stack([cmd, state.cmd_buf[0], state.cmd_buf[1]])
        # ---- the push
        pf, started = self._push(state, k_push, params)
        active = pf["push_left"] > 0
        R = self._R(data)
        p_world = data.xpos[p.base_bid] + R @ pf["push_point"]
        f = jnp.where(active, pf["push_force"], 0.0)
        torque = jnp.cross(p_world - data.xipos[p.base_bid], f)
        xfrc = jnp.zeros_like(data.xfrc_applied).at[p.base_bid].set(jnp.concatenate([f, torque]))
        data = data.replace(xfrc_applied=xfrc, qfrc_applied=jnp.zeros(p.nv))
        # ---- physics: 10 substeps at a jittered timestep
        jit_ms = c.ctrl_jitter_ms * params.plant_scale * jax.random.uniform(k_jit, (), minval=-1.0, maxval=1.0)
        ts = p.sim_dt * (1.0 + jit_ms / (p.decimation * p.sim_dt * 1e3))
        mx_i = model_with(p, dr.fields, timestep=ts)
        peak = jnp.asarray(p.tau_peak)
        kt = jnp.asarray(c.motor_kt_joint)
        r_ohm = jnp.asarray(c.motor_r_ohm)

        held = state.t < state.hold_s
        hold_q = jnp.asarray(self.base_qadr)
        hold_d = jnp.asarray(self.base_dadr)

        def substep(carry, k):
            d, tau_sq, _ = carry
            live = drive.live_command(k, dr.delay_ms, cmd_buf)
            q = d.qpos[p.act_qadr]
            qd = d.qvel[p.act_dadr]
            lim = drive.torque_limit(qd, peak, dr.torque_scale, kt, r_ohm, c.motor_bus_volts)
            tau = drive.pd_torque(q, qd, live[:6], live[6:12], live[12:18], lim)
            d = mjx.step(mx_i, d.replace(ctrl=tau))
            # the operator's stand: the base does not move while it is held (the legs still do)
            d = d.replace(qpos=jnp.where(held, d.qpos.at[hold_q].set(state.hold_qpos), d.qpos),
                          qvel=jnp.where(held, d.qvel.at[hold_d].set(0.0), d.qvel))
            return (d, tau_sq + tau ** 2, tau), None

        (data, tau_sq, tau_last), _ = lax.scan(substep, (data, jnp.zeros(6), jnp.zeros(6)),
                                               jnp.arange(p.decimation))
        # the release tick: the hand is not a clean let-go
        released = (state.t < state.hold_s) & (state.t + dt >= state.hold_s)
        qv = data.qvel
        for i, n in enumerate(("x", "y", "z")):
            qv = qv.at[p.base_d[n]].add(jnp.where(released, state.release_v[i], 0.0))
        data = data.replace(qvel=qv)
        # ---- push bookkeeping after the tick
        push_left = jnp.maximum(pf["push_left"] - 1, 0)
        push_since = pf["push_since"] + dt
        calm_t = jnp.where(push_left > 0, 0.0, state.calm_t + dt)
        # ---- measurements for reward / termination (true state)
        grav = self._grav_body(data)
        gyro = self._gyro(data)
        base = data.xpos[p.base_bid]
        v_world = self._vel_world(data)
        toe, heel, floor_viol = self._contacts(data)
        grounded = toe | heel
        foot_xy = self._foot_xy(data)
        # a foot that has JUST landed is still moving; billing that is billing the capture step
        # itself, so slip only counts once a foot has been down cfg.slip_dwell_ticks ticks
        ground_ticks = jnp.where(grounded, state.ground_ticks + 1, 0)
        settled = grounded & (ground_ticks > c.slip_dwell_ticks)
        slip = jnp.sum(jnp.where(settled, jnp.sum(((foot_xy - state.prev_foot_xy) / dt) ** 2, axis=-1), 0.0))
        # THE THING THE TASK ACTUALLY WANTS: the CoM over the feet. A 3 cm CoM offset cannot be held
        # by the nominal stance (the sole is 8.6 cm and there is no ankle), so the robot has to move
        # its FEET under its mass -- which the old "return to the nominal joint pose" term paid it
        # not to do (bal2, 2026-09-23: 21% of episodes fell with no push at all).
        com_err = self._com_xy(data) - self._support_xy(data, grounded)
        q_now = data.qpos[p.act_qadr]
        # ---- reward
        tilt2 = grav[0] ** 2 + grav[1] ** 2
        calm = calm_t >= c.calm_after_s
        r_alive = c.w_alive
        r_up = c.w_upright * jnp.exp(-tilt2 / c.sig_upright ** 2)
        r_h = c.w_height * jnp.exp(-((base[2] - p.height_stand) / c.sig_height) ** 2)
        r_com = c.w_com_support * jnp.exp(-jnp.sum(com_err ** 2) / c.sig_com_support ** 2)
        r_cv = c.w_calm_vel * calm * jnp.exp(-(v_world[0] ** 2 + v_world[1] ** 2) / c.sig_calm_vel ** 2)
        # posture is billed ONLY as a tie-breaker, and only while the CoM is already centred: it must
        # never argue against stepping under the mass
        centred = jnp.sum(com_err ** 2) < c.sig_com_support ** 2
        r_cp = c.w_calm_pose * calm * centred * jnp.exp(
            -jnp.sum((q_now - jnp.asarray(p.default_motor_pos)) ** 2) / c.sig_calm_pose ** 2)
        c_w = c.w_angvel * (gyro[0] ** 2 + gyro[1] ** 2)
        c_tau = c.w_torque * jnp.mean((tau_last / peak) ** 2)
        c_rate = c.w_action_rate * jnp.sum((action - state.prev_action) ** 2)
        c_slip = c.w_slip * slip
        c_gain = c.w_gain * jnp.mean(kp / c.kp_hi)
        half = 0.5 * (self.q_hi - self.q_lo)
        mid = 0.5 * (self.q_hi + self.q_lo)
        c_lim = c.w_joint_limit * jnp.sum(jnp.maximum(jnp.abs(target - mid) / half - 0.95, 0.0) / 0.05)
        terms = dict(alive=r_alive, upright=r_up, height=r_h, com=r_com, calm_vel=r_cv, calm_pose=r_cp,
                     angvel=-c_w, torque=-c_tau, action_rate=-c_rate, slip=-c_slip, gain=-c_gain,
                     joint_limit=-c_lim)
        reward = sum(terms.values())
        # ---- termination
        finite = jnp.isfinite(data.qpos).all() & jnp.isfinite(data.qvel).all() & (jnp.abs(data.qvel).max() < 200.0)
        fallen = (~finite) | (base[2] < c.term_height) | (grav[2] > c.term_gravity_z) | floor_viol
        fallen = fallen & ~held        # held: the operator's stand is carrying it, not the policy
        reward = jnp.where(finite, reward - c.fall_penalty * fallen, -c.fall_penalty)
        step_n = state.step_n + 1
        truncated = step_n >= self.max_steps
        done = fallen | truncated
        # ---- push verdicts (events)
        verdict_due = pf["push_pending"] & (push_since >= c.push_survive_s)
        push_ok = verdict_due & ~fallen
        push_fail = pf["push_pending"] & fallen
        # a fall with no push awaiting a verdict: the robot fell on its own (CoM, plant, noise)
        quiet_fall = fallen & ~pf["push_pending"]
        push_pending = pf["push_pending"] & ~verdict_due & ~fallen
        # ---- observation
        frame, ns, v_body_m, meas = self._frame(data, tau_last, action, state, k_frame)
        hist = jnp.concatenate([state.hist[1:], frame[None]], axis=0)
        ns.update(self._ema(state, meas))
        new_state = state.replace(
            data=data, key=key, step_n=step_n, t=state.t + dt, cmd_buf=cmd_buf, prev_target=target,
            prev_target_vel=tvel, prev_action=action, prev_vel_body=v_body_m, prev_foot_xy=foot_xy,
            ground_ticks=ground_ticks, hist=hist, push_countdown=pf["push_countdown"], push_left=push_left,
            push_force=pf["push_force"], push_point=pf["push_point"], push_dv=pf["push_dv"],
            push_hard=pf["push_hard"], push_pending=push_pending, push_since=push_since, calm_t=calm_t,
            ep_return=state.ep_return + reward, ep_len=state.ep_len + 1, **ns)
        obs = self._obs(new_state, data, toe, heel, v_body_m)
        info = dict(
            reward_terms=terms, fallen=fallen, truncated=truncated, quiet_fall=quiet_fall,
            push_start=started, push_ok=push_ok, push_fail=push_fail,
            push_dv=pf["push_dv"], push_hard=pf["push_hard"],
            ep_return=new_state.ep_return, ep_len=new_state.ep_len,
            kp_mean=kp.mean(), kd_mean=kd.mean(), torque_util=jnp.mean(jnp.abs(tau_last) / peak),
            tilt_deg=jnp.rad2deg(jnp.arcsin(jnp.sqrt(jnp.clip(tilt2, 0.0, 1.0)))),
            com_err=jnp.linalg.norm(com_err), airborne=jnp.mean((~grounded).astype(jnp.float32)),
            held=held.astype(jnp.float32),
            com_shift=dr.com_shift, obs_final=jnp.nan_to_num(obs),
            meas=meas, target=target, kp=kp, kd=kd, action=action,
        )
        # ---- auto-reset
        rs_state, rs_obs = self._reset_one(k_reset, params, Override())
        sel = lambda a, b: jnp.where(done, a, b)
        out_state = jax.tree_util.tree_map(sel, rs_state, new_state)
        out_obs = jnp.nan_to_num(jnp.where(done, rs_obs, obs))
        return out_state, out_obs, reward.astype(jnp.float32), done, info

    # ------------------------------------------------------------------ batched API
    def reset(self, key, params: EnvParams, override: Override = None):
        ov = override if override is not None else Override()

        def _bc(x):
            x = jnp.asarray(x, jnp.float32)
            return jnp.broadcast_to(x, (self.n_envs,)) if x.ndim == 0 else x
        ov = jax.tree_util.tree_map(_bc, ov)
        return self._reset_v(jax.random.split(key, self.n_envs), params, ov)

    def step(self, state, action, params: EnvParams):
        return self._step_v(state, action, params)
