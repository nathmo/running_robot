"""The static plant description + the per-episode plant draw (domain randomization) for MJX.

`Plant` is built ONCE per process from the v2 XML: it holds the mjx.Model, every index the env
needs, and the nominal values of the model fields the randomizer rewrites. `draw_plant` is a
pure JAX function: one PRNG key -> one `PlantDraw` (batched model fields + the scalar draws the
env applies itself: torque scale, delay, thermal, resync gain, sensor calibration constants).

Randomized MODEL fields (batched per env, MJX vmaps over them): body_mass, body_inertia,
body_ipos, geom_friction, dof_damping, jnt_stiffness (the leg springs), opt.gravity (slope
proxy), site_pos (loop-closure sites: the as-built yaw bias). Everything else the artifact lists
(§07/§08) is applied outside MuJoCo: kp/kv scale and torque headroom in drive.py, the actuation
delay in the command buffer, the winding temperatures, the resync gain, and the measurement
chain (homing zero, IMU mount rotation, biases, accelerometer leak, dropout).

Every draw is a multiplier or offset on the nominal with the range narrowed by `dr_scale`
(the curriculum) and, when `override` gives a finite value, replaced by it (the envelope tool
sets one axis and leaves the rest NaN = draw).
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

PKG_DIR = Path(__file__).resolve().parent
ACT_NAMES = ("hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R")


def resolve(p):
    q = Path(p)
    return str(q) if q.exists() else str(PKG_DIR / p)


class Plant:
    def __init__(self, cfg):
        self.cfg = cfg
        self.m = mujoco.MjModel.from_xml_path(resolve(cfg.model_path))
        m = self.m
        self.mx = mjx.put_model(m)
        self.nq, self.nv, self.nu = m.nq, m.nv, m.nu
        self.sim_dt = float(m.opt.timestep)
        self.decimation = int(cfg.control_decimation)
        self.control_dt = self.sim_dt * self.decimation
        # ids
        O = mujoco.mjtObj
        self.base_bid = mujoco.mj_name2id(m, O.mjOBJ_BODY, "bodyNCS-v1")
        self.floor_gid = mujoco.mj_name2id(m, O.mjOBJ_GEOM, "floor")
        self.foot_gids = np.array([mujoco.mj_name2id(m, O.mjOBJ_GEOM, f"foot_{s}_col") for s in "LR"])
        self.heel_gids = np.array([mujoco.mj_name2id(m, O.mjOBJ_GEOM, f"heel_{s}_col") for s in "LR"])
        self.foot_bids = np.array([int(m.geom_bodyid[g]) for g in self.foot_gids])
        self.toe_r = float(m.geom_size[self.foot_gids[0], 0])
        self.col_r = {int(g): float(m.geom_size[g, 0]) for g in list(self.foot_gids) + list(self.heel_gids)}
        act_ids = [mujoco.mj_name2id(m, O.mjOBJ_ACTUATOR, n) for n in ACT_NAMES]
        assert act_ids == list(range(6)), act_ids
        self.act_qadr = np.array([int(m.jnt_qposadr[m.actuator_trnid[a, 0]]) for a in range(6)])
        self.act_dadr = np.array([int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in range(6)])
        self.hip_roll_idx = np.array([0, 3])

        def _j(name):
            j = mujoco.mj_name2id(m, O.mjOBJ_JOINT, name)
            return (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])) if j >= 0 else (-1, -1)
        self.base_q = {}
        self.base_d = {}
        for n in ("x", "y", "z", "roll", "pitch", "yaw"):
            q, d = _j(f"base_{n}")
            self.base_q[n], self.base_d[n] = q, d
        self.planar = self.base_q["y"] < 0
        self.gyro_adr = int(m.sensor_adr[mujoco.mj_name2id(m, O.mjOBJ_SENSOR, "imu_gyro")])
        # leg joints (non-base): hinge DOFs get reset noise + damping DR; the leg springs are
        # passive prismatic joints with stiffness (the series spring, §07)
        self.spring_jids = np.array([mujoco.mj_name2id(m, O.mjOBJ_JOINT, f"leg_spring_{s}") for s in "LR"])
        assert (self.spring_jids >= 0).all(), "v2 plant needs the leg_spring_L/R joints"
        self.leg_hinge_qadr = np.array([int(m.jnt_qposadr[j]) for j in range(m.njnt)
                                        if m.jnt_bodyid[j] != self.base_bid and j not in self.spring_jids])
        self.leg_dofs = np.array([int(m.jnt_dofadr[j]) for j in range(m.njnt)
                                  if m.jnt_bodyid[j] != self.base_bid])
        self.loop_sites = np.array([mujoco.mj_name2id(m, O.mjOBJ_SITE, n)
                                    for n in ("pushrod_tip_L", "leg_anchor_L", "pushrod_tip_R", "leg_anchor_R")])
        # keyframe + nominal targets
        self.key_id = mujoco.mj_name2id(m, O.mjOBJ_KEY, cfg.keyframe)
        self.key_qpos = m.key_qpos[self.key_id].copy()
        nid = mujoco.mj_name2id(m, O.mjOBJ_NUMERIC, "nominal_ctrl")
        adr = int(m.numeric_adr[nid])
        self.nominal_ctrl = m.numeric_data[adr:adr + 6].copy()
        self.default_motor_pos = self.key_qpos[self.act_qadr].copy()
        self.height_stand = float(self.key_qpos[self.base_q["z"]])
        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()
        # joint position limits for the six motor joints (the gait target is clipped to them)
        self.q_lo = np.array([m.jnt_range[m.actuator_trnid[a, 0], 0] for a in range(6)])
        self.q_hi = np.array([m.jnt_range[m.actuator_trnid[a, 0], 1] for a in range(6)])
        self.tau_peak = m.actuator_forcerange[:, 1].copy()
        # standing-torque baseline: kp (nominal - q_key) at zero velocity, the PD's holding torque
        self.stand_torque = np.asarray(cfg.drive_kp) * (self.nominal_ctrl - self.default_motor_pos)
        # workspace reference: toe positions in the base frame at the keyframe
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, self.key_id)
        mujoco.mj_forward(m, d)
        base, R = d.xpos[self.base_bid], d.xmat[self.base_bid].reshape(3, 3)
        self.ws_ref = np.array([R.T @ (d.geom_xpos[g] - base) for g in self.foot_gids])
        self.data0 = mjx.put_data(m, d)
        # nominal DR fields
        self.n_body_mass = m.body_mass.copy()
        self.n_body_inertia = m.body_inertia.copy()
        self.n_body_ipos = m.body_ipos.copy()
        self.n_geom_friction = m.geom_friction.copy()
        self.n_dof_damping = m.dof_damping.copy()
        self.n_jnt_stiffness = m.jnt_stiffness.copy()
        self.n_gravity = m.opt.gravity.copy()
        self.n_site_pos = m.site_pos.copy()
        self.total_mass = float(m.body_mass.sum())
        self.bw = self.total_mass * 9.81
        self.leg_dof_mask = np.zeros(m.nv, bool)
        self.leg_dof_mask[self.leg_dofs] = True
        self.spring_mask = np.zeros(m.njnt, bool)
        self.spring_mask[self.spring_jids] = True
        self.loop_site_mask = np.zeros(m.nsite, bool)
        self.loop_site_mask[self.loop_sites] = True
        self.ncon = int(self.data0._impl.contact.dist.shape[0]) if hasattr(self.data0, "_impl") \
            else int(self.data0.contact.dist.shape[0])

    def actuator_names(self):
        return list(ACT_NAMES)


class PlantFields(NamedTuple):
    """The batched mjx.Model fields of one env."""
    body_mass: jnp.ndarray
    body_inertia: jnp.ndarray
    body_ipos: jnp.ndarray
    geom_friction: jnp.ndarray
    dof_damping: jnp.ndarray
    jnt_stiffness: jnp.ndarray
    gravity: jnp.ndarray
    site_pos: jnp.ndarray


class PlantDraw(NamedTuple):
    fields: PlantFields
    mass_scale: jnp.ndarray          # ()
    com_off_x: jnp.ndarray           # ()   base-body CoM x offset (m), signed
    friction: jnp.ndarray            # ()
    kp_scale: jnp.ndarray            # (6,)
    kv_scale: jnp.ndarray            # (6,)
    torque_scale: jnp.ndarray        # ()
    delay_ms: jnp.ndarray            # ()
    link_scale: jnp.ndarray          # ()
    thermal_scale: jnp.ndarray       # ()
    thermal_x0: jnp.ndarray          # (6,)
    kappa: jnp.ndarray               # ()
    tilt_deg: jnp.ndarray            # (2,) roll, pitch of the gravity vector
    wind: jnp.ndarray                # (2,) N, world x/y constant force
    # measurement chain constants (per episode)
    joint_zero: jnp.ndarray          # (6,) rad, homing error (obs AND command side)
    enc_offset: jnp.ndarray          # (6,)
    vel_bias: jnp.ndarray            # (6,)
    trq_bias: jnp.ndarray            # (6,)
    grav_bias: jnp.ndarray           # (3,)
    gyro_bias0: jnp.ndarray          # (3,)
    accel_leak: jnp.ndarray          # ()
    imu_R: jnp.ndarray               # (3,3)


class Override(NamedTuple):
    """Per-env overrides for the envelope tool: finite = use, NaN = draw. Scalars broadcast."""
    mass_scale: jnp.ndarray = jnp.nan
    friction: jnp.ndarray = jnp.nan
    kp_scale: jnp.ndarray = jnp.nan
    torque_scale: jnp.ndarray = jnp.nan
    delay_ms: jnp.ndarray = jnp.nan
    link_scale: jnp.ndarray = jnp.nan
    thermal_scale: jnp.ndarray = jnp.nan
    thermal_x0: jnp.ndarray = jnp.nan
    kappa: jnp.ndarray = jnp.nan
    tilt_roll_deg: jnp.ndarray = jnp.nan
    tilt_pitch_deg: jnp.ndarray = jnp.nan
    wind_x: jnp.ndarray = jnp.nan
    wind_y: jnp.ndarray = jnp.nan
    theta: jnp.ndarray = jnp.nan     # (44,) library variant: the episode's gait, no box draw


def _ov(ov_val, drawn):
    ov_val = jnp.asarray(ov_val, dtype=drawn.dtype)
    return jnp.where(jnp.isnan(ov_val), drawn, ov_val)


def _u(key, rel, shape=()):
    """U(1-rel, 1+rel); rel=0 -> exactly 1."""
    return 1.0 + rel * jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)


def _rodrigues(axis, th):
    axis = axis / jnp.maximum(jnp.linalg.norm(axis), 1e-9)
    K = jnp.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return jnp.eye(3) + jnp.sin(th) * K + (1.0 - jnp.cos(th)) * (K @ K)


def draw_plant(key, cfg, plant: Plant, dr_scale, ov: Override = None) -> PlantDraw:
    """One episode's plant. dr_scale narrows every range (0 = nominal plant, sensor
    calibration errors included; the white noise is not scaled)."""
    ov = ov if ov is not None else Override()
    ks = jax.random.split(key, 24)
    on = 1.0 if cfg.dr_enable else 0.0
    s = dr_scale * on
    m = plant.m
    # ---- mass / inertia / CoM
    g = _u(ks[0], cfg.dr_mass_global * s)
    g = _ov(ov.mass_scale, g)
    per = _u(ks[1], cfg.dr_mass_body * s, (m.nbody,))
    m_scale = g * per
    body_mass = jnp.asarray(plant.n_body_mass) * m_scale
    i_j = _u(ks[2], cfg.dr_inertia * s, (m.nbody, 3))
    body_inertia = jnp.asarray(plant.n_body_inertia) * m_scale[:, None] * i_j
    off = cfg.dr_com_offset * s
    d_ipos = off * jax.random.uniform(ks[3], (m.nbody, 3), minval=-1.0, maxval=1.0)
    body_ipos = jnp.asarray(plant.n_body_ipos) + d_ipos
    com_off = d_ipos[plant.base_bid, 0]
    # ---- friction (absolute range contracted toward 1.0)
    lo, hi = cfg.dr_friction_range
    f = jax.random.uniform(ks[4], (), minval=1.0 + (lo - 1.0) * s, maxval=1.0 + (hi - 1.0) * s)
    f = _ov(ov.friction, f)
    nf = jnp.asarray(plant.n_geom_friction)
    geom_friction = nf.at[:, 0].set(nf[:, 0] * f / jnp.maximum(nf[:, 0].max(), 1e-9))
    # ---- drive gains, torque headroom
    kp_scale = _u(ks[5], cfg.dr_kp * s, (6,))
    kp_scale = _ov(ov.kp_scale, kp_scale)
    kv_scale = _u(ks[6], cfg.dr_kv * s, (6,))
    torque_scale = _ov(ov.torque_scale, _u(ks[7], cfg.dr_torque * s))
    # ---- joint damping (leg DOFs only)
    dmp = _u(ks[8], cfg.dr_joint_damping * s, (m.nv,))
    dof_damping = jnp.asarray(plant.n_dof_damping) * jnp.where(jnp.asarray(plant.leg_dof_mask), dmp, 1.0)
    # ---- the leg series spring +-50%
    link_scale = _ov(ov.link_scale, _u(ks[9], cfg.dr_link_spring * s))
    jnt_stiffness = jnp.asarray(plant.n_jnt_stiffness) * jnp.where(jnp.asarray(plant.spring_mask), link_scale, 1.0)
    # ---- gravity tilt (roll and pitch, deg)
    tilt = cfg.dr_gravity_tilt_deg * s * jax.random.uniform(ks[10], (2,), minval=-1.0, maxval=1.0)
    tilt = jnp.stack([_ov(ov.tilt_roll_deg, tilt[0]), _ov(ov.tilt_pitch_deg, tilt[1])])
    tr, tp = jnp.deg2rad(tilt[0]), jnp.deg2rad(tilt[1])
    gmag = float(np.linalg.norm(plant.n_gravity))
    # tilt the world: down = Rx(roll) Ry(pitch) [0,0,-1]
    down = jnp.array([-jnp.sin(tp) * jnp.cos(tr), jnp.sin(tr), -jnp.cos(tp) * jnp.cos(tr)])
    gravity = gmag * down / jnp.linalg.norm(down)
    # ---- loop-closure sites (the as-built asymmetry)
    js = cfg.dr_loop_site_m * s * jax.random.uniform(ks[11], (m.nsite, 3), minval=-1.0, maxval=1.0)
    site_pos = jnp.asarray(plant.n_site_pos) + jnp.where(jnp.asarray(plant.loop_site_mask)[:, None], js, 0.0)
    # ---- delay (ms), drawn in the full range whatever the curriculum: the RUNNER had zero
    # margin, the stabilizer must see the whole range from the start (§07)
    dlo, dhi = cfg.drive_delay_range_ms
    nom = cfg.drive_delay_ms
    delay = jax.random.uniform(ks[12], (), minval=nom + (dlo - nom) * on, maxval=nom + (dhi - nom) * on)
    delay = _ov(ov.delay_ms, delay)
    # ---- thermal: dT_max scale + hot start
    tlo, thi = cfg.thermal_scale_range
    th_scale = jax.random.uniform(ks[13], (), minval=1.0 + (tlo - 1.0) * on, maxval=1.0 + (thi - 1.0) * on)
    th_scale = _ov(ov.thermal_scale, th_scale)
    x0 = cfg.thermal_hot_start_max * jax.random.uniform(ks[14], (6,)) * (1.0 if cfg.thermal_enable else 0.0)
    x0 = _ov(ov.thermal_x0, x0)
    # ---- resync gain
    klo, khi = cfg.resync_kappa_range
    kappa = jax.random.uniform(ks[15], (), minval=cfg.resync_kappa + (klo - cfg.resync_kappa) * on,
                               maxval=cfg.resync_kappa + (khi - cfg.resync_kappa) * on)
    kappa = _ov(ov.kappa, kappa) * (1.0 if cfg.resync_enable else 0.0)
    # ---- wind: constant world-frame force, x and y (y only on the free plant)
    adv = s if cfg.adversity_curriculum else on
    wind = cfg.wind_force_max * adv * jax.random.uniform(ks[16], (2,), minval=-1.0, maxval=1.0)
    wind = jnp.stack([_ov(ov.wind_x, wind[0]), _ov(ov.wind_y, wind[1]) * (0.0 if plant.planar else 1.0)])
    # ---- measurement chain
    noise_on = 1.0 if cfg.obs_noise_enable else 0.0
    joint_zero = (s * jnp.deg2rad(cfg.dr_joint_zero_deg)
                  * jax.random.uniform(ks[17], (6,), minval=-1.0, maxval=1.0)) * noise_on
    enc_offset = jax.random.normal(ks[18], (6,)) * 0.0 * noise_on           # folded into joint_zero
    vel_bias = jax.random.normal(ks[19], (6,)) * 0.05 * noise_on
    trq_bias = jax.random.normal(ks[20], (6,)) * 1.0 * noise_on
    grav_bias = jax.random.normal(ks[21], (3,)) * cfg.noise_grav_bias * noise_on
    gyro_bias0 = jax.random.normal(ks[22], (3,)) * cfg.noise_gyro_bias * noise_on
    k_a, k_b, k_c = jax.random.split(ks[23], 3)
    accel_leak = jax.random.uniform(k_a, (), minval=0.0, maxval=cfg.noise_accel_leak) * noise_on
    ax = jax.random.normal(k_b, (3,))
    th = s * jnp.deg2rad(cfg.dr_imu_rot_deg) * jax.random.uniform(k_c, (), minval=-1.0, maxval=1.0) * noise_on
    imu_R = _rodrigues(ax, th)
    fields = PlantFields(body_mass=body_mass, body_inertia=body_inertia, body_ipos=body_ipos,
                         geom_friction=geom_friction, dof_damping=dof_damping,
                         jnt_stiffness=jnt_stiffness, gravity=gravity, site_pos=site_pos)
    return PlantDraw(fields=fields, mass_scale=g, com_off_x=com_off, friction=f,
                     kp_scale=kp_scale, kv_scale=kv_scale, torque_scale=torque_scale,
                     delay_ms=delay, link_scale=link_scale, thermal_scale=th_scale,
                     thermal_x0=x0, kappa=kappa, tilt_deg=tilt, wind=wind,
                     joint_zero=joint_zero, enc_offset=enc_offset, vel_bias=vel_bias,
                     trq_bias=trq_bias, grav_bias=grav_bias, gyro_bias0=gyro_bias0,
                     accel_leak=accel_leak, imu_R=imu_R)


def model_with(plant: Plant, fields: PlantFields, timestep=None):
    """The env's mjx.Model with this env's drawn fields (and, per tick, its jittered timestep)."""
    rep = {"body_mass": fields.body_mass, "body_inertia": fields.body_inertia,
           "body_ipos": fields.body_ipos, "geom_friction": fields.geom_friction,
           "dof_damping": fields.dof_damping, "jnt_stiffness": fields.jnt_stiffness,
           "opt.gravity": fields.gravity, "site_pos": fields.site_pos}
    if timestep is not None:
        rep["opt.timestep"] = timestep
    return plant.mx.tree_replace(rep)
