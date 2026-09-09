# Offset/gain exactness on the env's OWN in-air rig (set_fixed_base): real plant, real loop closure,
# rigid ankle, base welded 25 cm above stance so the feet hang free. Replaces fkcheck/2/3 (base
# pinned at z=0 by an unset eq_data) and the Leg-class attempt (free-joint qpos on a slide base).
import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from config import get_config, PRESETS
import env as E, cpg_gait
name = sorted([k for k in PRESETS if 'sprint' in k and 'm6' in k], key=len)[-1]
env = E.DashEnv(get_config(name)); env.set_fixed_base(0.25); env.reset()
M, D = env.model, env.data; NOM = env.nominal_ctrl.copy(); F = env.foot_gids_arr; B = env.base_id
HRL, CAL, THL, HRR, CAR, THR = range(6)
print('preset %s | base welded at z = %.4f (stance %.4f)' % (name, D.xpos[B][2], env.default_qpos[2]))

def settle(dc, n=2500):
    env.reset(); D.ctrl[:] = NOM + np.asarray(dc, float)
    for _ in range(n): mujoco.mj_step(M, D)
    return (D.geom_xpos[F] - D.xpos[B]).copy()          # toes in the base frame
def one(i, v):
    a = np.zeros(6); a[i] = v; return a

r0 = settle(np.zeros(6)); r0b = settle(np.zeros(6))
lut = cpg_gait.load_lut(); nt = lut['nominal_toe']
print('0  determinism |diff| %.1e | toe_L rel base (%+.4f, %+.4f, %+.4f) vs LUT nominal_toe (%+.4f, %+.4f, %+.4f) -> %s'
      % (np.abs(r0-r0b).max(), r0[0][0], r0[0][1], r0[0][2], nt[0], nt[1], nt[2],
         'FRAME OK' if np.abs(r0[0][[0,2]]-nt[[0,2]]).max() < 0.02 else 'MISMATCH'))
print('   toe_R rel base (%+.4f, %+.4f, %+.4f)   mirror residual dx %+.4f dy %+.4f' % (r0[1][0], r0[1][1], r0[1][2], r0[0][0]-r0[1][0], r0[0][1]+r0[1][1]))

def J2(base, h=0.012):                                     # left toe (x,z) vs (cam, thigh)
    J = np.zeros((2,2))
    for c, idx in enumerate((CAL, THL)):
        a = settle(base+one(idx,h))[0]; b = settle(base-one(idx,h))[0]
        J[:,c] = [(a[0]-b[0])/(2*h), (a[2]-b[2])/(2*h)]
    return J
J0 = J2(np.zeros(6))
print('\n1  J at stance [m/rad]: dx/dcam %+.3f dx/dthigh %+.3f | dz/dcam %+.3f dz/dthigh %+.3f' % (J0[0,0],J0[0,1],J0[1,0],J0[1,1]))
print('   local dx/dthigh along a +-0.20 thigh stroke:', ' '.join('%+.3f' % J2(one(THL,t))[0,1] for t in (-0.2,-0.1,0,0.1,0.2)))
print('   cross-coupling: hip_roll +0.20 ->', np.round(settle(one(HRL,0.2))[0]-r0[0], 4), '(dx dy dz)')

N = 36; xs = np.array([settle(one(THL, 0.20*np.cos(2*np.pi*i/N)))[0][0] for i in range(N)])
mag = np.abs(np.fft.rfft(xs-xs.mean())/N)
print('\n2  thigh = 0.20 cos(phi): toe-x harmonics rel fundamental  k2 %.1f%%  k3 %.1f%%  k4 %.1f%%  (THD %.1f%%)'
      % (100*mag[2]/mag[1], 100*mag[3]/mag[1], 100*mag[4]/mag[1], 100*np.sqrt((mag[2:6]**2).sum())/mag[1]))

PH = np.radians(np.arange(0, 360, 30)); S = 0.20; T = 0.02
def stroke(ph):
    dc = np.zeros(6); dc[THL] = S*np.cos(ph); dc[THR] = S*np.cos(ph); return dc     # (+,+) = symmetric alternating gait
def stag(ph, dL, dR):
    b = stroke(ph); q = settle(b); p = settle(b+dL+dR)
    return (p[0][0]-q[0][0]) - (p[1][0]-q[1][0])
def report(nm, fn):
    v = np.array([fn(ph) for ph in PH])
    print('   %-30s mean %+.4f  min %+.4f  max %+.4f  variation %5.1f%%' % (nm, v.mean(), v.min(), v.max(), 100*(v.max()-v.min())/abs(v.mean())))
kth, kca = T/J0[0,1], T/J0[0,0]; dq0 = np.linalg.solve(J0, [T, 0.0])
print('\n3  +2 cm fore-aft split through the stroke  [0%% = exact]')
report('(a) thigh-only bias',        lambda ph: stag(ph, one(THL,kth), one(THR,kth)))
report('(b) cam-only bias',          lambda ph: stag(ph, one(CAL,kca), one(CAR,kca)))
report('(c) J^-1 @stance constant',  lambda ph: stag(ph, one(CAL,dq0[0])+one(THL,dq0[1]), one(CAR,dq0[0])+one(THR,dq0[1])))
def ik_add(base, leg_ix, tgt_dx):                        # Newton on the settled plant, per leg
    idx = (CAL, THL) if leg_ix == 0 else (CAR, THR); d = np.zeros(6)
    for _ in range(4):
        p = settle(base+d)[leg_ix]; q = settle(base)[leg_ix]
        err = np.array([tgt_dx - (p[0]-q[0]), 0.0 - (p[2]-q[2])])
        if np.abs(err).max() < 3e-4: break
        Jl = np.zeros((2,2)); h = 0.012
        for c, i in enumerate(idx):
            a = settle(base+d+one(i,h))[leg_ix]; bb = settle(base+d-one(i,h))[leg_ix]
            Jl[:,c] = [(a[0]-bb[0])/(2*h), (a[2]-bb[2])/(2*h)]
        st = np.linalg.solve(Jl, err); d[idx[0]] += st[0]; d[idx[1]] += st[1]
    return d
def exact(ph):
    b = stroke(ph); dL = ik_add(b, 0, +T/2); dR = ik_add(b, 1, -T/2); return stag(ph, dL, dR)
report('(e) exact IK on the plant',   exact)
print('\n4  stride gain g=1.3 on the LEFT leg in joint space: realised toe-x amplitude ratio (1.300 = exact)')
def amp(g): x = np.array([settle(one(THL, g*S*np.cos(2*np.pi*i/N)))[0][0] for i in range(N)]); return x.max()-x.min()
print('   %.3f' % (amp(1.3)/amp(1.0)))
