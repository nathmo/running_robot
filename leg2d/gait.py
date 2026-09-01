"""Parametric single-leg gait trajectory for the leg2d rig.

Open-loop, not learned: cadence/duty/stride/clearance/z_off ARE the trajectory, not something fit
to it (see leg2d/README.md for why this is possible -- railing y/roll/pitch/yaw removes the balance
problem). Reuses the exact stance-ramp + smoothstep-swing shape already validated in
training/scripted_walk.py's ScriptedWalker._foot_xz.
"""
import numpy as np


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def foot_xz(t, f_hz, duty, stride, clearance, z_off=0.0):
    """(dx, dz) of the toe (m), relative to the nominal stance toe, at time t (s).

    One monopod hop cycle: STANCE (fraction `duty` of the period) sweeps the toe from +stride/2 to
    -stride/2 in the BODY frame -- the foot is fixed on the ground, so the body advances `stride`
    over the stance phase. SWING (1-duty) is a zero-scuff smoothstep return from -stride/2 back to
    +stride/2, with a raised-cosine lift peaking at `clearance`.

    `z_off` is a STATIC leg-extension bias added to dz for the whole cycle (negative = longer leg).
    Commanding the settled keyframe's own toe height (z_off=0) is NOT the same as holding that
    height: the real drive's position loop is P-only, so a joint loaded by the leg's own weight
    droops short of its target by ~(gravity torque / kp) -- exactly the mechanism
    training/scripted_walk.py already found and worked around with its own tuned `z_off`
    (measured -0.07 m for double support; single support here carries ~2x the load, so the
    optimizer searches a wider, more negative range). Without it, commanding the nominal height
    while under a full-weight DYNAMIC stepping load lets the leg buckle instead of just standing --
    the earlier version of this rig pinned z away entirely to dodge that, which also (as an
    unwanted side effect) removed the ground reaction force the stance foot needs for traction; this
    is the real fix.
    """
    T = 1.0 / f_hz
    phase = (t % T) / T
    half = 0.5 * stride
    if phase < duty:
        s = phase / max(duty, 1e-9)
        return half - stride * s, z_off
    u = (phase - duty) / max(1.0 - duty, 1e-9)
    return -half + stride * smoothstep(u), z_off + clearance * (0.5 - 0.5 * np.cos(2.0 * np.pi * u))


def in_stance(t, f_hz, duty):
    T = 1.0 / f_hz
    return ((t % T) / T) < duty


def nominal_speed(f_hz, stride):
    """Average forward speed IF the trajectory tracks perfectly (m/s) -- the sweep's independent
    target, not a measurement. sweep.py reports the ACTUAL simulated speed alongside this."""
    return f_hz * stride


def flight_time(f_hz, duty):
    return (1.0 - duty) / f_hz
