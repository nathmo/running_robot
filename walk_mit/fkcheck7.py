import sys, numpy as np, mujoco
sys.path.insert(0, '.')
exec(open('fkcheck5.py').read().split("print('CAM-LED STROKE")[0])       # rig, settle, one, J2, PH, T, report
def stroke(ph, ac=0.11, at=0.05):
    dc = np.zeros(6); dc[CAL] = dc[CAR] = ac*np.cos(ph); dc[THL] = dc[THR] = at*np.cos(ph); return dc
def stag(ph, dL, dR):
    b = stroke(ph); q = settle(b); p = settle(b+dL+dR); return (p[0][0]-q[0][0]) - (p[1][0]-q[1][0])
def ik_add(base, leg, tgt):
    idx = (CAL, THL) if leg == 0 else (CAR, THR); d = np.zeros(6); q = settle(base)[leg]
    for _ in range(8):
        p = settle(base+d)[leg]; err = np.array([tgt-(p[0]-q[0]), -(p[2]-q[2])])
        if np.abs(err).max() < 2e-4: break
        st = np.linalg.solve(J2(base+d, leg, h=0.006), err); n = np.linalg.norm(st)
        if n > 0.05: st *= 0.05/n
        d[idx[0]] += st[0]; d[idx[1]] += st[1]
    return d
print('REALISTIC STROKE, remaining three:')
report('(e) exact IK, damped Newton', lambda ph: stag(ph, ik_add(stroke(ph),0,+T), ik_add(stroke(ph),1,-T)), 2*T)
N = 36
def amp(g):
    x = np.array([settle(stroke(2*np.pi*i/N, ac=g*0.11, at=g*0.05))[0][0] for i in range(N)]); return x.max()-x.min()
print('   GAIN g=1.3 joint-space: realised toe-x amplitude ratio %.3f (1.300 = exact)' % (amp(1.3)/amp(1.0)))
v = np.array([[(settle(stroke(ph)+one(HRL,.1)+one(HRR,.1))[i][1]-settle(stroke(ph))[i][1]) for i in (0,1)] for ph in PH])
print('   LEAN hip_roll (+,+) 0.10: dy %+.4f..%+.4f  same direction %s  spread %.1f%%' % (v.min(), v.max(), bool((v[:,0]*v[:,1]>0).all()), 100*(v.max()-v.min())/abs(v.mean())))
