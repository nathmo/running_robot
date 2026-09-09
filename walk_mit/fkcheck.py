# -*- coding: utf-8 -*-
"""Does a joint-space Fourier series map cleanly onto the foot workspace?

Settles the 4-bar under position control with the base welded, so what we read is the
true joint -> foot map including loop closure.
"""
import numpy as np, mujoco

M = mujoco.MjModel.from_xml_path('model/dash01.xml')
D = mujoco.MjData(M)
K = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_KEY, 'stand')
FOOT = [mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_GEOM, 'foot_%s_col' % s) for s in 'LR']
LOCK = np.array([mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_EQUALITY, 'lock_' + n)
                 for n in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')])
NOM = M.key_ctrl[K].copy()
# ctrl order: 0 hip_roll_L 1 cam_L 2 thigh_L 3 hip_roll_R 4 cam_R 5 thigh_R
HRL, CAL, THL, HRR, CAR, THR = range(6)


def foot(dctrl, n=2500):
    mujoco.mj_resetDataKeyframe(M, D, K)
    D.eq_active[LOCK] = 1
    D.ctrl[:] = NOM + np.asarray(dctrl, float)
    for _ in range(n):
        mujoco.mj_step(M, D)
    return D.geom_xpos[FOOT].copy()          # [[Lx,Ly,Lz],[Rx,Ry,Rz]]


def d(i, v):
    a = np.zeros(6); a[i] = v; return a


ref = foot(np.zeros(6))
print('settled foot   L = (%+.4f, %+.4f, %+.4f)   R = (%+.4f, %+.4f, %+.4f)'
      % (tuple(ref[0]) + tuple(ref[1])))

# ---------------------------------------------------------------- Q1 mirror exactness
print('\nQ1  MIRROR EXACTNESS  (joint (+,-) should give mirror-image feet)')
for amp in (0.10, 0.25):
    p = foot(d(THL, amp) + d(THR, -amp) + d(CAL, 0.5 * amp) + d(CAR, -0.5 * amp))
    print('   amp %.2f : dx_L-dx_R = %+.5f m   dy_L+dy_R = %+.5f m   (0 = perfect mirror)'
          % (amp, (p[0][0] - ref[0][0]) - (p[1][0] - ref[1][0]),
                  (p[0][1] - ref[0][1]) + (p[1][1] - ref[1][1])))

# ---------------------------------------------------------------- Q2 is the map linear?
print('\nQ2  LINEARITY OF thigh -> foot x   (constant joint step, is the foot step constant?)')
prev = None
for th in np.arange(-0.30, 0.31, 0.10):
    p = foot(d(THL, th))
    x = p[0][0] - ref[0][0]
    inc = '' if prev is None else '   step %+.4f m' % (x - prev)
    print('   thigh %+0.2f rad -> foot dx %+.4f m, dz %+.4f m%s' % (th, x, p[0][2] - ref[0][2], inc))
    prev = x

# ---------------------------------------------------------------- Q3 anti DC symmetry
print('\nQ3  ANTISYMMETRIC DC  (joint (+,+) on thigh: is the foot stagger equal and opposite?)')
for c0 in (0.08, 0.16, 0.24):
    p = foot(d(THL, c0) + d(THR, c0))
    dxL, dxR = p[0][0] - ref[0][0], p[1][0] - ref[1][0]
    print('   c0 %+0.2f rad -> dx_L %+.4f   dx_R %+.4f   sum %+.4f   asym %.1f%%'
          % (c0, dxL, dxR, dxL + dxR, 100 * abs(dxL + dxR) / max(abs(dxL), abs(dxR))))

# ---------------------------------------------------------------- Q4 cross coupling
print('\nQ4  CROSS-COUPLING  (does one joint bleed into the other foot axes?)')
for nm, i, v in (('thigh', THL, 0.20), ('cam', CAL, 0.20), ('hip_roll', HRL, 0.20)):
    p = foot(d(i, v))
    dd = p[0] - ref[0]
    print('   %-9s +0.20 rad -> dx %+.4f  dy %+.4f  dz %+.4f' % (nm, dd[0], dd[1], dd[2]))

# ---------------------------------------------------------------- Q5 Jacobian over the stroke
print('\nQ5  LOCAL GAIN d(foot x)/d(thigh) ACROSS THE STROKE')
h, gains = 0.02, []
for th in np.arange(-0.30, 0.31, 0.15):
    a = foot(d(THL, th + h))[0][0]
    b = foot(d(THL, th - h))[0][0]
    g = (a - b) / (2 * h)
    gains.append(g)
    print('   at thigh %+0.2f : %+0.4f m/rad' % (th, g))
g = np.array(gains)
print('   gain varies %.0f%% across the stroke (min %.4f, max %.4f)'
      % (100 * (g.max() - g.min()) / abs(g.mean()), g.min(), g.max()))
