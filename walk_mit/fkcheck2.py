# -*- coding: utf-8 -*-
"""Can a joint-space offset be made workspace-exact without a full IK?

Compares three ways to realise a 2 cm fore-aft foot stagger, swept through a 0.25 rad stroke:
  (a) thigh-only joint bias            what the design does now
  (b) cam-only joint bias              is the crank a more linear channel?
  (c) Jacobian pullback at nominal     differential IK: J^-1 * (dx, 0, 0), constant over the cycle
  (d) phase-dependent pullback         J^-1 evaluated at every phase (a Jacobian LUT)
and reports how much the realised stagger varies over the cycle (0% = exact).
"""
import numpy as np, mujoco

M = mujoco.MjModel.from_xml_path('model/dash01.xml'); D = mujoco.MjData(M)
K = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_KEY, 'stand')
F = [mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_GEOM, 'foot_%s_col' % s) for s in 'LR']
L = np.array([mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_EQUALITY, 'lock_' + n)
              for n in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')])
NOM = M.key_ctrl[K].copy()
HRL, CAL, THL, HRR, CAR, THR = range(6)


def foot(dc, n=2200):
    mujoco.mj_resetDataKeyframe(M, D, K); D.eq_active[L] = 1
    D.ctrl[:] = NOM + np.asarray(dc, float)
    for _ in range(n):
        mujoco.mj_step(M, D)
    return D.geom_xpos[F].copy()


def jac_left(base, h=0.015):
    """3x2 Jacobian of LEFT foot (x,z) wrt (cam, thigh), central differences about `base`."""
    J = np.zeros((2, 2))
    for c, idx in enumerate((CAL, THL)):
        p = base.copy(); p[idx] += h; a = foot(p)[0]
        m = base.copy(); m[idx] -= h; b = foot(m)[0]
        J[0, c] = (a[0] - b[0]) / (2 * h)      # dx
        J[1, c] = (a[2] - b[2]) / (2 * h)      # dz
    return J


J0 = jac_left(np.zeros(6))
print('Jacobian at nominal, left foot (x,z) vs (cam, thigh)  [m/rad]')
print('   dx: cam %+.3f  thigh %+.3f' % (J0[0, 0], J0[0, 1]))
print('   dz: cam %+.3f  thigh %+.3f' % (J0[1, 0], J0[1, 1]))
TARGET = np.array([0.02, 0.0])                 # 2 cm forward, zero lift
dq0 = np.linalg.solve(J0, TARGET)
print('   pullback for +2 cm fore-aft at constant height: cam %+.4f rad, thigh %+.4f rad' % tuple(dq0))
print()

PH = np.radians(np.arange(0, 360, 30))
STROKE = 0.25


def stroke_ctrl(ph):
    """the symmetric gait: thigh sinusoid, both legs, antiphase, mirrored sign"""
    dc = np.zeros(6)
    dc[THL] = STROKE * np.cos(ph)
    dc[THR] = -STROKE * np.cos(ph + np.pi)
    return dc


def stagger(ph, addL, addR):
    """realised fore-aft stagger (x_L - x_R change) when adding offsets on top of the stroke"""
    base = stroke_ctrl(ph)
    q = foot(base)
    p = foot(base + addL + addR)
    return (p[0][0] - q[0][0]) - (p[1][0] - q[1][0])


def report(name, fn):
    vals = np.array([fn(ph) for ph in PH])
    good = vals[np.abs(vals - np.median(vals)) < 5 * np.median(np.abs(vals - np.median(vals))) + 1e-9]
    var = 100 * (good.max() - good.min()) / abs(good.mean())
    print('%-34s mean %+.4f m   range %+.4f..%+.4f   variation %5.1f%%   (%d/%d pts)'
          % (name, good.mean(), good.min(), good.max(), var, len(good), len(vals)))


def one(idx, v):
    a = np.zeros(6); a[idx] = v; return a


# scale each joint-only bias so it gives ~2 cm at nominal, for a fair comparison
k_th = TARGET[0] / J0[0, 1]
k_ca = TARGET[0] / J0[0, 0]
report('(a) thigh-only bias',      lambda ph: stagger(ph, one(THL, k_th), one(THR, k_th)))
report('(b) cam-only bias',        lambda ph: stagger(ph, one(CAL, k_ca), one(CAR, k_ca)))
report('(c) J^-1 pullback @nominal', lambda ph: stagger(ph, one(CAL, dq0[0]) + one(THL, dq0[1]),
                                                       one(CAR, dq0[0]) + one(THR, dq0[1])))


def pullback_at(ph):
    base = stroke_ctrl(ph)
    J = jac_left(base)
    dq = np.linalg.solve(J, TARGET)
    return stagger(ph, one(CAL, dq[0]) + one(THL, dq[1]), one(CAR, dq[0]) + one(THR, dq[1]))


report('(d) J^-1(phi) pullback per phase', pullback_at)
