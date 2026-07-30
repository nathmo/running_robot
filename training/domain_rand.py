"""Domain randomization + sensor noise for DASH-01 sim2real.

Two independent pieces, both driven by per-episode draws:

  PlantRandomizer  — rewrites MuJoCo model parameters at every reset so the policy never sees the
                     same robot twice: masses, inertias, CoM, friction, servo gains, joint damping,
                     the passive ankle spring, torque headroom and a gravity tilt (the cheap proxy
                     for a sloped / non-flat floor). Everything is randomized around the NOMINAL
                     values captured after env.__init__ has applied its own model edits, so it
                     composes with com_lower / ankle_stiffness / ankle_damping instead of fighting
                     them.

  SensorNoise      — corrupts the observation the way the real Pi + moteus + IMU chain does:
                     per-episode calibration errors (encoder zero offsets, IMU mount misalignment,
                     gyro bias) that are CONSTANT within an episode, plus per-step white noise, plus
                     a gyro bias random walk, plus the one that actually matters for this robot —
                     accelerometer leak into the gravity vector. The pitch reflex reads grav_x
                     directly, and on hardware that number comes from an attitude filter whose
                     accelerometer is contaminated by body acceleration during running. A policy
                     trained on exact gravity has never seen the signal it will actually get.

Design notes
------------
* Every randomized quantity is a MULTIPLIER or an OFFSET on the nominal, drawn once per episode
  from a symmetric range. A range of 0 disables that axis exactly (the model is byte-identical),
  so the whole module is inert when cfg.dr_enable is False and every pre-existing preset trains
  the same as before.
* Mass and inertia are randomized TOGETHER by default (a heavier link is also a higher-inertia
  link) with an extra independent inertia jitter on top — scaling them independently by wide
  factors produces physically impossible bodies.
* The ankle spring is preload-preserving, exactly like env.__init__: the standing ankle sits well
  off the spring's rest angle, so changing k alone would change the standing POSTURE, which is a
  different perturbation than the stiffness change we mean to test.
"""
import numpy as np
import mujoco


def _u(rng, rel):
    """Scalar multiplier drawn from U(1-rel, 1+rel). rel=0 -> exactly 1.0."""
    return 1.0 if rel <= 0 else float(rng.uniform(1.0 - rel, 1.0 + rel))


def _uv(rng, rel, n):
    """n independent multipliers from U(1-rel, 1+rel)."""
    return np.ones(n) if rel <= 0 else rng.uniform(1.0 - rel, 1.0 + rel, n)


class PlantRandomizer:
    """Per-episode MuJoCo model randomization. Construct AFTER env.__init__ has finished editing
    the model, so the captured nominals include com_lower / ankle_stiffness / ankle_damping."""

    def __init__(self, model, cfg, ankle_jnt_ids, leg_dof_ids, loop_site_ids=()):
        self.cfg = cfg
        self.enabled = bool(cfg.dr_enable)
        self._ankle_j = list(ankle_jnt_ids)
        self._leg_dofs = np.asarray(leg_dof_ids, dtype=int)
        self._loop_sites = np.asarray(loop_site_ids, dtype=int)
        self.n_site_pos = model.site_pos.copy()
        # nominal snapshots — every draw is relative to these, never to the last episode's values
        self.n_mass = model.body_mass.copy()
        self.n_inertia = model.body_inertia.copy()
        self.n_ipos = model.body_ipos.copy()
        self.n_friction = model.geom_friction.copy()
        self.n_gainprm = model.actuator_gainprm.copy()
        self.n_biasprm = model.actuator_biasprm.copy()
        self.n_dof_damping = model.dof_damping.copy()
        self.n_jnt_stiffness = model.jnt_stiffness.copy()
        self.n_qpos_spring = model.qpos_spring.copy()
        self.n_gravity = model.opt.gravity.copy()
        self.n_forcerange = model.actuator_forcerange.copy()
        self._jnt_qposadr = model.jnt_qposadr.copy()
        self._jnt_dofadr = model.jnt_dofadr.copy()
        # the standing pose the ankle preload is defined at (set by the env)
        self.stand_qpos = None
        self.last = {}          # the draw actually applied, for logging / eval reporting

    def resample(self, model, rng):
        """Draw a fresh plant and write it into `model`. Returns the per-episode dict, which also
        carries the values the ENV (not the model) has to apply: torque scale and action delay."""
        c = self.cfg
        d = {}
        if not self.enabled:
            d["torque_scale"] = 1.0
            d["action_delay_steps"] = int(c.action_delay_steps)
            self.last = d
            return d

        # ---- mass / inertia / CoM ----------------------------------------------------------
        # global scale (whole-robot build tolerance) x per-body jitter (link-level CAD error).
        g = _u(rng, c.dr_mass_global)
        per = _uv(rng, c.dr_mass_body, model.nbody)
        m_scale = g * per
        model.body_mass[:] = self.n_mass * m_scale
        # inertia tracks mass, plus an independent shape/placeholder-CAD jitter. This is the axis
        # the speed oracle said dominates, and the one whose CAD values are still placeholders.
        i_scale = m_scale[:, None] * _uv(rng, c.dr_inertia, model.nbody * 3).reshape(model.nbody, 3)
        model.body_inertia[:] = self.n_inertia * i_scale
        # CoM offset, applied to every body (battery/harness placement, assembly tolerance)
        if c.dr_com_offset > 0:
            model.body_ipos[:] = self.n_ipos + rng.uniform(
                -c.dr_com_offset, c.dr_com_offset, (model.nbody, 3))
        d["mass_scale"] = float(g)
        d["total_mass"] = float(model.body_mass.sum())

        # ---- contact ------------------------------------------------------------------------
        # sliding friction only (columns 1,2 are torsional/rolling and are already ~0 here).
        if c.dr_friction > 0:
            f = float(rng.uniform(*c.dr_friction_range))
            model.geom_friction[:, 0] = self.n_friction[:, 0] * (f / max(self.n_friction[:, 0].max(), 1e-9))
            d["friction"] = f

        # ---- actuators: servo gains + torque headroom ---------------------------------------
        # MuJoCo <position>: gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kv. kp and kv must be
        # scaled consistently or the servo is not the servo you think it is.
        kp_s = _uv(rng, c.dr_kp, model.nu)
        kv_s = _uv(rng, c.dr_kv, model.nu)
        model.actuator_gainprm[:, 0] = self.n_gainprm[:, 0] * kp_s
        model.actuator_biasprm[:, 1] = self.n_biasprm[:, 1] * kp_s
        model.actuator_biasprm[:, 2] = self.n_biasprm[:, 2] * kv_s
        d["kp_scale"] = float(np.mean(kp_s))
        # torque headroom: returned, NOT written — the env combines it with the torque-budget
        # curriculum's own scale so the two never overwrite each other.
        d["torque_scale"] = _u(rng, c.dr_torque)

        # ---- joint damping / friction --------------------------------------------------------
        if c.dr_joint_damping > 0 and self._leg_dofs.size:
            s = _uv(rng, c.dr_joint_damping, self._leg_dofs.size)
            model.dof_damping[self._leg_dofs] = self.n_dof_damping[self._leg_dofs] * s

        # ---- the passive ankle spring (preload-preserving, as in env.__init__) ---------------
        # k is the parameter the whole m3 balance result hinges on, and it is a PHYSICAL SPRING
        # that will not be manufactured to spec. Randomizing it wide is the single highest-value
        # axis in this module.
        if c.dr_ankle_k > 0 and self._ankle_j and self.stand_qpos is not None:
            ks = _u(rng, c.dr_ankle_k)
            kd = _u(rng, c.dr_ankle_damping)
            for j in self._ankle_j:
                qadr = int(self._jnt_qposadr[j])
                k_old = float(self.n_jnt_stiffness[j])
                k_new = k_old * ks
                ref_old = float(self.n_qpos_spring[qadr])
                q_stand = float(self.stand_qpos[qadr])
                # keep k*(q_stand - ref) constant -> same standing preload, different gain
                model.qpos_spring[qadr] = q_stand - (k_old / k_new) * (q_stand - ref_old)
                model.jnt_stiffness[j] = k_new
                dadr = int(self._jnt_dofadr[j])
                model.dof_damping[dadr] = self.n_dof_damping[dadr] * kd
            d["ankle_k_scale"] = float(ks)

        # ---- L/R linkage asymmetry = the intrinsic yaw bias -----------------------------------
        # The as-exported model is NOT left/right symmetric: leg_anchor_L and leg_anchor_R differ by
        # ~1 mm in x AND z, coordinates that should be identical (only y mirrors). The two four-bar
        # linkages are therefore geometrically different, so an exactly mirror-symmetric joint
        # command produces asymmetric foot motion and a persistent yaw moment — measured at
        # ~0.4 rad/s, which is LARGER than the stride-steer channel's whole authority.
        #
        # The fix is not to symmetrize the model. A real robot will be asymmetric too, and a policy
        # trained on a perfectly symmetric plant would have no reason to learn the correction. So we
        # randomize the asymmetry instead: the yaw bias becomes a different value every episode, and
        # cancelling it becomes something the policy must actively do from the yaw-rate error rather
        # than a constant it can bake in. That is the version that transfers.
        if c.dr_loop_site > 0 and self._loop_sites.size:
            for sid in self._loop_sites:
                model.site_pos[sid] = self.n_site_pos[sid] + rng.uniform(
                    -c.dr_loop_site, c.dr_loop_site, 3)
            d["loop_site_jitter_mm"] = float(c.dr_loop_site * 1000)

        # ---- gravity tilt = the cheap sloped-floor proxy --------------------------------------
        # Rotating gravity by a small angle is dynamically identical to tilting the whole world,
        # and costs nothing (no terrain geometry, no contact-model changes). Also absorbs IMU
        # mount misalignment, which is indistinguishable from a slope to the policy.
        if c.dr_gravity_tilt > 0:
            tilt = np.deg2rad(rng.uniform(0.0, c.dr_gravity_tilt))
            az = rng.uniform(0.0, 2 * np.pi)
            gmag = float(np.linalg.norm(self.n_gravity))
            model.opt.gravity[:] = gmag * np.array([
                np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)])
            d["slope_deg"] = float(np.rad2deg(tilt))

        # ---- latency ---------------------------------------------------------------------------
        # The fixed 4-step delay was always a guess. Draw it per episode so the policy cannot tune
        # itself to one exact latency.
        lo, hi = c.dr_delay_steps_range
        d["action_delay_steps"] = int(rng.integers(lo, hi + 1)) if hi > lo else int(c.action_delay_steps)

        self.last = d
        return d

    def restore(self, model):
        """Put the nominal plant back (used by the robustness sweep to set an exact operating
        point without a stale randomized parameter leaking in)."""
        model.body_mass[:] = self.n_mass
        model.body_inertia[:] = self.n_inertia
        model.body_ipos[:] = self.n_ipos
        model.geom_friction[:] = self.n_friction
        model.actuator_gainprm[:] = self.n_gainprm
        model.actuator_biasprm[:] = self.n_biasprm
        model.dof_damping[:] = self.n_dof_damping
        model.jnt_stiffness[:] = self.n_jnt_stiffness
        model.qpos_spring[:] = self.n_qpos_spring
        model.opt.gravity[:] = self.n_gravity
        model.actuator_forcerange[:] = self.n_forcerange
        model.site_pos[:] = self.n_site_pos


class SensorNoise:
    """The measurement chain between the real robot and the policy.

    Per-episode constants (calibration you get wrong once and live with for the whole run):
      * encoder zero offsets per joint
      * IMU mount misalignment -> a constant tilt on the gravity vector
      * gyro bias
      * accelerometer leak coefficient (how badly the attitude filter rejects body acceleration)
    Per-step:
      * white noise on every channel
      * gyro bias random walk
      * gravity contaminated by body linear acceleration, scaled by this episode's leak
    """

    def __init__(self, cfg, nu):
        self.cfg = cfg
        self.nu = nu
        self.enabled = bool(cfg.obs_noise_enable)
        self.reset(np.random.default_rng(0))

    def reset(self, rng):
        c = self.cfg
        n = self.nu
        z = np.zeros(n)
        if not self.enabled:
            self.pos_offset, self.vel_bias, self.trq_bias = z, z.copy(), z.copy()
            self.grav_tilt = np.zeros(3)
            self.gyro_bias = np.zeros(3)
            self.accel_leak = 0.0
            return
        self.pos_offset = rng.normal(0.0, c.noise_encoder_offset, n)
        self.vel_bias = rng.normal(0.0, c.noise_motor_vel_bias, n)
        self.trq_bias = rng.normal(0.0, c.noise_torque_bias, n)
        # IMU misalignment: a small constant rotation shows up as a constant gravity offset
        self.grav_tilt = rng.normal(0.0, c.noise_grav_bias, 3)
        self.gyro_bias = rng.normal(0.0, c.noise_gyro_bias, 3)
        self.accel_leak = float(rng.uniform(0.0, c.noise_accel_leak))

    def step_bias(self, rng):
        """Gyro bias random walk — the drift a real MEMS gyro accumulates within one run."""
        if self.enabled and self.cfg.noise_gyro_walk > 0:
            self.gyro_bias += rng.normal(0.0, self.cfg.noise_gyro_walk, 3)

    def apply(self, rng, motor_pos, motor_vel, motor_trq, grav, gyro, accel_body):
        """Corrupt one raw proprioception frame. Returns the same 5 arrays, measured."""
        if not self.enabled:
            return motor_pos, motor_vel, motor_trq, grav, gyro
        c = self.cfg
        n = self.nu
        motor_pos = motor_pos + self.pos_offset + rng.normal(0.0, c.noise_encoder, n)
        motor_vel = motor_vel + self.vel_bias + rng.normal(0.0, c.noise_motor_vel, n)
        motor_trq = motor_trq * (1.0 + rng.normal(0.0, c.noise_torque_gain, n)) \
            + self.trq_bias + rng.normal(0.0, c.noise_torque, n)
        # gravity: bias + accelerometer leak + white noise, then renormalized. The leak term is
        # what makes the pitch reflex's input realistic — during a hard push-off the measured
        # "down" swings by several degrees even though the body has not rotated at all.
        g = grav + self.grav_tilt + rng.normal(0.0, c.noise_grav, 3)
        if self.accel_leak > 0.0:
            g = g - self.accel_leak * (accel_body / 9.81)
        nrm = float(np.linalg.norm(g))
        if nrm > 1e-6:
            g = g / nrm
        gyro = gyro + self.gyro_bias + rng.normal(0.0, c.noise_gyro, 3)
        return motor_pos, motor_vel, motor_trq, g, gyro
