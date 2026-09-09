# Offset/gain knob exactness on the EXACT loop-closure FK (plot_reachability.Leg), sagittal plane.
# No dynamics, no settling, no floor: the number is the linkage geometry and nothing else.
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_reachability as reach
reach.MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dash01.xml")
leg = reach.Leg()
print("cam range %s  thigh range %s" % (np.round(leg.cam_range, 3), np.round(leg.thigh_range, 3)))

NOM = np.array([0.0, 0.12])          # (cam, thigh) resettled stance, left leg
_seed = {"s": np.array([0.0, 0.0])}
def fk(q):
    res, sd = leg.fk(float(q[0]), float(q[1]), _seed["s"])
    if res is None:
        return None
    _seed["s"] = sd
    return np.asarray(res["tip"], float)          # (x, z) rel. hip

tip0 = fk(NOM)
print("nominal tip (x,z) rel hip: %s" % np.round(tip0, 4))

def jac(q, h=0.004):
    J = np.zeros((2, 2))
    for c in range(2):
        e = np.zeros(2); e[c] = h
        a, b = fk(q + e), fk(q - e)
        J[:, c] = (a - b) / (2 * h)
    return J

def ik(q0, target, iters=25):
    """exact IK: Newton on the 2D FK from q0"""
    q = q0.copy()
    for _ in range(iters):
        p = fk(q)
        if p is None: return None
        err = target - p
        if np.abs(err).max() < 1e-7: break
        q = q + np.linalg.solve(jac(q), err)
    return q

J0 = jac(NOM)
print("J at stance [m/rad]:  dx/dcam %+.3f  dx/dthigh %+.3f | dz/dcam %+.3f  dz/dthigh %+.3f"
      % (J0[0,0], J0[0,1], J0[1,0], J0[1,1]))

# ---- 1. how nonlinear is thigh -> x along the stroke?
print("\n1  local gain dx/dthigh along a +-0.20 rad thigh stroke")
for th in (-0.20, -0.10, 0.0, 0.10, 0.20):
    print("   thigh %+0.2f : %+.4f m/rad" % (th, jac(NOM + [0, th])[0, 1]))

# ---- 2. THD of a pure joint sinusoid in tip x
N = 96; xs = np.array([fk(NOM + [0, 0.20*np.cos(2*np.pi*i/N)])[0] for i in range(N)])
mag = np.abs(np.fft.rfft(xs - xs.mean()) / N)
print("\n2  thigh = 0.20 cos(phi):  tip-x harmonics rel. fundamental  k2 %.1f%%  k3 %.1f%%  k4 %.1f%%   THD %.1f%%"
      % (100*mag[2]/mag[1], 100*mag[3]/mag[1], 100*mag[4]/mag[1], 100*np.sqrt((mag[2:6]**2).sum())/mag[1]))

# ---- 3. a +2 cm fore-aft stagger through the stroke, five strategies
PH = np.radians(np.arange(0, 360, 30)); STROKE = 0.20; T = 0.02
def legs(ph):                                   # (cam,thigh) of L at phi and R at phi+pi (mirror => same fk)
    return NOM + [0, STROKE*np.cos(ph)], NOM + [0, STROKE*np.cos(ph + np.pi)]

def stag(ph, dL, dR):
    qL, qR = legs(ph)
    return (fk(qL + dL)[0] - fk(qL)[0]) - (fk(qR + dR)[0] - fk(qR)[0])

def report(name, fn):
    v = np.array([fn(ph) for ph in PH])
    print("   %-32s mean %+.4f  min %+.4f  max %+.4f  variation %6.1f%%"
          % (name, v.mean(), v.min(), v.max(), 100*(v.max()-v.min())/abs(v.mean())))

kth = T / J0[0, 1]; kca = T / J0[0, 0]
dq0 = np.linalg.solve(J0, [T, 0.0])
print("\n3  +2 cm split (L forward, R back), swept through the stroke  [0%% = exact]")
report("(a) thigh-only bias",           lambda ph: stag(ph, [0, kth/2], [0, kth/2]))
report("(b) cam-only bias",             lambda ph: stag(ph, [kca/2, 0], [kca/2, 0]))
report("(c) J^-1 @stance, constant",    lambda ph: stag(ph, dq0/2, dq0/2))
def perphase(ph):
    qL, qR = legs(ph)
    dL = np.linalg.solve(jac(qL), [ T/2, 0.0]); dR = np.linalg.solve(jac(qR), [-T/2, 0.0])
    return stag(ph, dL, -dR)  # note: R needs -1cm in ITS frame -> apply as (+,+) convention via sign
report("(d) J^-1(phi) per phase",       perphase)
def exact(ph):
    qL, qR = legs(ph)
    pL, pR = fk(qL), fk(qR)
    qL2 = ik(qL, pL + [ T/2, 0.0]); qR2 = ik(qR, pR + [-T/2, 0.0])
    return (fk(qL2)[0] - pL[0]) - (fk(qR2)[0] - pR[0])
report("(e) exact IK (Newton on FK)",   exact)

# ---- 4. gain knob: scale the stroke amplitude by g in joint space vs in foot space
print("\n4  stride gain g = 1.3 on the LEFT leg only: realised foot-x amplitude ratio  [1.300 = exact]")
def amp_joint(g):
    x = np.array([fk(NOM + [0, g*STROKE*np.cos(2*np.pi*i/N)])[0] for i in range(N)]); return x.max()-x.min()
print("   joint-space  g*S(phi):        %.3f" % (amp_joint(1.3)/amp_joint(1.0)))
x0 = np.array([fk(NOM + [0, STROKE*np.cos(2*np.pi*i/N)]) for i in range(N)])
c = x0.mean(0); xs2 = []
for p in x0:
    q = ik(NOM, c + 1.3*(p - c)); xs2.append(fk(q)[0])
xs2 = np.array(xs2); print("   foot-space   IK(c + g(p-c)):  %.3f" % ((xs2.max()-xs2.min())/(x0[:,0].max()-x0[:,0].min())))
