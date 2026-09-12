"""The measured drive, in JAX, evaluated every 1 kHz substep (artifact §07).

    tau = clip( kp(phi) (q_ref - q) - kd(phi) qdot ,  +-tau_lim(w) )
    tau_lim(w) = min( tau_peak * scale ,  Kt (V_bus - Kt |w|) / R )        back-EMF clamp

plus: the pure transport delay (6-18 ms per episode, applied at substep granularity by picking
which tick's command is live at each substep), the no-load slew cap on the commanded target,
and the single-node thermal model (§07):

    tau_th dX/dt = (tau / tau_cont)^2 / s - X,     X = dT / dT_max,  s ~ U[0.8, 1.2]

with tau_th = 45 s from the two anchors (steady state at tau_cont; peak from cold hits the
limit at 5 s). Pure functions; the env owns the state.
"""
import jax.numpy as jnp
import numpy as np


def torque_limit(qd, peak, scale, kt, r_ohm, v_bus):
    """Available |torque| per actuator at joint speed qd."""
    v_avail = jnp.maximum(v_bus - kt * jnp.abs(qd), 0.0)
    return jnp.minimum(peak * scale, kt * v_avail / r_ohm)


def pd_torque(q, qd, q_ref, kp, kd, lim):
    tau = kp * (q_ref - q) - kd * qd
    return jnp.clip(tau, -lim, lim)


def slew_limit(target, prev_target, prev_vel, vel_limit, accel_limit, dt):
    """No-load command cap: the commanded target may not move faster than the motor can."""
    v_des = (target - prev_target) / dt
    if accel_limit > 0.0:
        dv = accel_limit * dt
        v_des = jnp.clip(v_des, prev_vel - dv, prev_vel + dv)
    v_des = jnp.clip(v_des, -vel_limit, vel_limit)
    return prev_target + v_des * dt, v_des


def live_command(k_substep, delay_ms, cmds3, substep_ms=1.0):
    """Which tick's command is live at substep k (0-based) of the current tick.

    cmds3: stacked [current, previous, two-back] along axis 0 (each [..., d]). A command issued
    at tick start reaches the plant delay_ms later: at substep k the live one is the newest
    whose age (k - delay) is non-negative. Delay <= 20 ms is representable."""
    age = k_substep * substep_ms - delay_ms         # >= 0 : current tick's command is live
    idx = jnp.where(age >= 0.0, 0, jnp.where(age >= -10.0, 1, 2))
    return jnp.take(cmds3, idx, axis=0)


def thermal_update(x, tau_sq_mean, dt, tau_th, tau_cont, scale):
    """One tick of the winding node. x = dT/dT_max per actuator; tau_sq_mean = mean tau^2 over
    the tick's substeps."""
    drive = tau_sq_mean / (tau_cont ** 2) / scale
    return x + (dt / tau_th) * (drive - x)


def thermal_time_to_limit(tau_frac, tau_th):
    """Seconds for a cold node to reach the limit at a constant tau/tau_cont ratio (inf if never).
    Used by the smoke test to check the anchors (peak 170/55 -> 5 s; 1.0 -> never)."""
    r2 = tau_frac ** 2
    if r2 <= 1.0:
        return np.inf
    return -tau_th * np.log(1.0 - 1.0 / r2)
