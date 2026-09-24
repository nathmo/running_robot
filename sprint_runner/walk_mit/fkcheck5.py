import sys, numpy as np, mujoco
sys.path.insert(0, '.')
exec(open('fkcheck4.py').read().split("r0 = settle")[0])      # rig, settle(), one(), J-free
r0 = settle(np.zeros(6))
def J2(base, leg, h=0.012):
    idx = (CAL, THL) if leg == 0 else (CAR, THR); J = np.zeros((2,2))
    for c, i in enumerate(idx):
        a = settle(base+one(i,h))[leg]; b = settle(base-one(i,h))[leg]; J[:,c] = [(a[0]-b[0])/(2*h), (a[2]-b[2])/(2*h)]
    return J
J0 = J2(np.zeros(6), 0)
PH = np.radians(np.arange(0, 360, 30)); T = 0.02
def stroke(ph, amp_cam=0.30, amp_th=0.10):                 # cam-led gait, thigh in support (+,+) = alternating
    dc = np.zeros(6); dc[CAL] = dc[CAR] = amp_cam*np.cos(ph); dc[THL] = dc[THR] = amp_th*np.cos(ph); return dc
def stag(ph, dL, dR):
    b = stroke(ph); q = settle(b); p = settle(b+dL+dR); return (p[0][0]-q[0][0]) - (p[1][0]-q[1][0])
def report(nm, fn, target):
    v = np.array([fn(ph) for ph in PH]); ok = np.abs(v - np.median(v)) < 0.05; g = v[ok]
    print('   %-30s mean %+.4f (target %+.4f, err %+5.1f%%)  spread %5.1f%%  bad %d' % (nm, g.mean(), target, 100*(g.mean()/target-1), 100*(g.max()-g.min())/abs(g.mean()), (~ok).sum()))
print('CAM-LED STROKE (cam 0.30, thigh 0.10): toe-x sweep of the left foot over the cycle:')
xs = np.array([settle(stroke(ph))[0][0]-r0[0][0] for ph in PH]); print('   ', np.round(xs, 3))
kca, kth = T/J0[0,0], T/J0[0,1]; dq0 = np.linalg.solve(J0, [T, 0.0])
print('\n3b  +2 cm per-foot split (stagger target 0.04) through the CAM stroke')
report('(a) thigh-only bias',       lambda ph: stag(ph, one(THL,kth), one(THR,kth)), 2*T)
report('(b) cam-only bias',         lambda ph: stag(ph, one(CAL,kca), one(CAR,kca)), 2*T)
report('(c) J^-1 @stance constant', lambda ph: stag(ph, one(CAL,dq0[0])+one(THL,dq0[1]), one(CAR,dq0[0])+one(THR,dq0[1])), 2*T)
def ik_add(base, leg, tgt):                                # damped Newton on the settled plant
    idx = (CAL, THL) if leg == 0 else (CAR, THR); d = np.zeros(6); q = settle(base)[leg]
    for _ in range(8):
        p = settle(base+d)[leg]; err = np.array([tgt-(p[0]-q[0]), -(p[2]-q[2])])
        if np.abs(err).max() < 2e-4: break
        st = np.linalg.solve(J2(base+d, leg, h=0.006), err); n = np.linalg.norm(st)
        if n > 0.05: st *= 0.05/n
        d[idx[0]] += st[0]; d[idx[1]] += st[1]
    return d
report('(e) exact IK, damped Newton', lambda ph: stag(ph, ik_add(stroke(ph),0,+T), ik_add(stroke(ph),1,-T)), 2*T)
print('\n5  LEAN knob: hip_roll (+,+) 0.10 rad through the cam stroke -> lateral shift of the two toes')
v = np.array([[(settle(stroke(ph)+one(HRL,.1)+one(HRR,.1))[i][1]-settle(stroke(ph))[i][1]) for i in (0,1)] for ph in PH])
print('   dy_L %+.4f..%+.4f   dy_R %+.4f..%+.4f   same-direction? %s   spread %.1f%%' % (v[:,0].min(), v[:,0].max(), v[:,1].min(), v[:,1].max(), bool((v[:,0]*v[:,1] > 0).all()), 100*(v.max()-v.min())/abs(v.mean())))
