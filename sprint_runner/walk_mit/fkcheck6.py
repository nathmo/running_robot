import sys, numpy as np, mujoco
sys.path.insert(0, '.')
exec(open('fkcheck5.py').read().split("print('CAM-LED STROKE")[0])       # rig + J2 + ik_add + report
def stroke(ph, ac=0.11, at=0.05):
    dc = np.zeros(6); dc[CAL] = dc[CAR] = ac*np.cos(ph); dc[THL] = dc[THR] = at*np.cos(ph); return dc
def stag(ph, dL, dR):
    b = stroke(ph); q = settle(b); p = settle(b+dL+dR); return (p[0][0]-q[0][0]) - (p[1][0]-q[1][0])
xs = np.array([settle(stroke(ph))[0][0]-r0[0][0] for ph in PH])
print('REALISTIC STROKE (cam 0.11, thigh 0.05): toe-x sweep %+.3f..%+.3f m (%.2f m stride)' % (xs.min(), xs.max(), xs.max()-xs.min()))
N = 36; x = np.array([settle(stroke(2*np.pi*i/N))[0][0] for i in range(N)]); mag = np.abs(np.fft.rfft(x-x.mean())/N)
print('   toe-x harmonics rel fundamental: k2 %.1f%%  k3 %.1f%%   THD %.1f%%' % (100*mag[2]/mag[1], 100*mag[3]/mag[1], 100*np.sqrt((mag[2:6]**2).sum())/mag[1]))
kca, kth = T/J0[0,0], T/J0[0,1]; dq0 = np.linalg.solve(J0, [T, 0.0])
print('\n+2 cm per-foot split (stagger target 0.04) through the realistic stroke')
report('(a) thigh-only bias',       lambda ph: stag(ph, one(THL,kth), one(THR,kth)), 2*T)
report('(b) cam-only bias',         lambda ph: stag(ph, one(CAL,kca), one(CAR,kca)), 2*T)
report('(c) J^-1 @stance constant', lambda ph: stag(ph, one(CAL,dq0[0])+one(THL,dq0[1]), one(CAR,dq0[0])+one(THR,dq0[1])), 2*T)
report('(e) exact IK, damped Newton', lambda ph: stag(ph, ik_add(stroke(ph),0,+T), ik_add(stroke(ph),1,-T)), 2*T)
def amp(g):
    x = np.array([settle(stroke(2*np.pi*i/N, ac=g*0.11, at=g*0.05))[0][0] for i in range(N)]); return x.max()-x.min()
print('\nGAIN knob g=1.3 in joint space on the realistic stroke: realised toe-x amplitude ratio %.3f (1.300 = exact)' % (amp(1.3)/amp(1.0)))
v = np.array([[(settle(stroke(ph)+one(HRL,.1)+one(HRR,.1))[i][1]-settle(stroke(ph))[i][1]) for i in (0,1)] for ph in PH])
print('LEAN knob hip_roll (+,+) 0.10: dy %+.4f..%+.4f, same direction %s, spread %.1f%%' % (v.min(), v.max(), bool((v[:,0]*v[:,1]>0).all()), 100*(v.max()-v.min())/abs(v.mean())))
