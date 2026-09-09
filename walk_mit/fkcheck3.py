# -*- coding: utf-8 -*-
"""Offset-knob exactness, on the env's own resettled stance, in the base frame.

Fixes two defects of fkcheck/fkcheck2: (1) they settled against the XML's raw key_ctrl instead of
the env's resettled nominal, leaving the feet 36 cm in the air; (2) they read world coordinates
while the IK LUT is body-relative. Also checks whether settling is deterministic before trusting
any number.
"""
import sys
import numpy as np, mujoco
sys.path.insert(0, '.')
from config import get_config, PRESETS
import env as E
import cpg_gait

cands = [k for k in PRESETS if 'sprint' in k and 'm6' in k] or [k for k in PRESETS if 'sprint' in k and 'mit' in k]
print('sprint/m6 presets:', cands)
name = sorted(cands, key=len)[-1]
cfg = get_config(name)
env = E.DashEnv(cfg)
M, D, K = env.model, env.data, env.key_id
NOM = env.nominal_ctrl.copy()
F = env.foot_gids_arr
LOCK = env.lock_eq_ids
BASE = env.base_id
HRL, CAL, THL, HRR, CAR, THR = range(6)
print('preset %s   resettled nominal ctrl %s' % (name, np.round(NOM, 4)))


def settle(dc, n=3000):
    mujoco.mj_resetDataKeyframe(M, D, K)
    D.eq_active[LOCK] = 1                       # weld the base
    D.ctrl[:] = NOM + np.asarray(dc, float)
    for _ in range(n):
        mujoco.mj_step(M, D)
    b = D.xpos[BASE]
    return (D.geom_xpos[F] - b).copy()          # feet in the BASE frame


def one(i, v):
    a = np.zeros(6); a[i] = v; return a


# ---------------------------------------------------------------- 0. determinism + frame
r1, r2 = settle(np.zeros(6)), settle(np.zeros(6))
print('\n0  settle twice, max |diff| = %.2e m   (must be ~0)' % np.abs(r1 - r2).max())
lut = cpg_gait.load_lut(); nt = lut['nominal_toe']
print('   base-frame toe at nominal:  L (%+.4f, %+.4f, %+.4f)   R (%+.4f, %+.4f, %+.4f)'
      % (tuple(r1[0]) + tuple(r1[1])))
print('   LUT nominal_toe:            (%+.4f, %+.4f, %+.4f)   -> frame %s'
      % (nt[0], nt[1], nt[2], 'MATCHES' if np.abs(r1[0][[0, 2]] - nt[[0, 2]]).max() < 0.02 else 'MISMATCH'))

# ---------------------------------------------------------------- 1. the map, cleanly
print('\n1  thigh -> toe x, base frame, from the real stance')
for th in (-0.30, -0.15, 0.0, 0.15, 0.30):
    p = settle(one(THL, th))[0] - r1[0]
    print('   thigh %+0.2f : dx %+.4f  dz %+.4f' % (th, p[0], p[2]))

# ---------------------------------------------------------------- 2. offset strategies through a stroke
def stroke(ph, amp=0.20):
    dc = np.zeros(6); dc[THL] = amp * np.cos(ph); dc[THR] = amp * np.cos(ph); return dc


def stagger(ph, addL, addR):
    b = stroke(ph); q = settle(b); p = settle(b + addL + addR)
    return (p[0][0] - q[0][0]) - (p[1][0] - q[1][0])


def jac(base, h=0.012):
    J = np.zeros((2, 2))
    for c, idx in enumerate((CAL, THL)):
        a = settle(base + one(idx, h))[0]; b = settle(base - one(idx, h))[0]
        J[:, c] = [(a[0] - b[0]) / (2 * h), (a[2] - b[2]) / (2 * h)]
    return J


J0 = jac(np.zeros(6)); T = np.array([0.02, 0.0]); dq0 = np.linalg.solve(J0, T)
print('\n2  J at stance: dx/dcam %+.3f dx/dthigh %+.3f | dz/dcam %+.3f dz/dthigh %+.3f'
      % (J0[0, 0], J0[0, 1], J0[1, 0], J0[1, 1]))
PH = np.radians(np.arange(0, 360, 30))


def report(label, fn):
    v = np.array([fn(ph) for ph in PH])
    ok = np.abs(v - np.median(v)) < 0.03
    g = v[ok]
    print('   %-30s mean %+.4f  min %+.4f  max %+.4f  variation %5.1f%%  outliers %d'
          % (label, g.mean(), g.min(), g.max(), 100 * (g.max() - g.min()) / abs(g.mean()), (~ok).sum()))


kth, kca = T[0] / J0[0, 1], T[0] / J0[0, 0]
report('(a) thigh-only bias', lambda ph: stagger(ph, one(THL, kth), one(THR, kth)))
report('(b) cam-only bias', lambda ph: stagger(ph, one(CAL, kca), one(CAR, kca)))
report('(c) J^-1 @stance, constant', lambda ph: stagger(ph, one(CAL, dq0[0]) + one(THL, dq0[1]),
                                                       one(CAR, dq0[0]) + one(THR, dq0[1])))


def lut_split(ph):
    b = stroke(ph); q = settle(b)
    dL, dR = q[0] - nt, q[1] - nt
    jL = cpg_gait.foot_ik(dL[0] + 0.01, dL[2], lut) - cpg_gait.foot_ik(dL[0], dL[2], lut)
    jR = cpg_gait.foot_ik(dR[0] - 0.01, dR[2], lut) - cpg_gait.foot_ik(dR[0], dR[2], lut)
    add = one(CAL, jL[0]) + one(THL, jL[1]) + one(CAR, jR[0]) + one(THR, jR[1])
    p = settle(b + add)
    return (p[0][0] - q[0][0]) - (p[1][0] - q[1][0])


report('(e) full IK via measured LUT', lut_split)
