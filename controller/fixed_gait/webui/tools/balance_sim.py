#!/usr/bin/env python3
"""MuJoCo test bench for the Balance mode (balance.py). Dev machine only (mujoco + scipy).

    python controller/fixed_gait/webui/tools/balance_sim.py            # the scenario table
    python controller/fixed_gait/webui/tools/balance_sim.py --plot out.png

The plant: the homing CAD (the twin's model) with a free base, the four-bar closed by a `connect`
at the verified pin (389 mm), the RL plant's measured masses (motors as point masses at their
pivots), and the real soles (the flat 66.5 x 25.4 mm face of each FootFlat mesh) on a floor.
The drives are what manual mode actually uses, SET_POS: a stiff joint servo whose REFERENCE goes
through the measured servo dynamics, 20 ms transport delay + first-order lag at 1.9 Hz (tau 84 ms;
memory: actuator Bode 2026-09-01). The IMU is the torso's attitude with a delay and noise.
balance.Balancer runs unmodified at the daemon's 100 Hz.

Each scenario runs with the controller OFF (stand pose held) and ON, and reports the peak tilt,
the final tilt, and whether it fell (a sole edge lifted by > 20 mm or tilt > 15 deg).
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import solve_stand_pose as sp          # noqa: E402
import balance                         # noqa: E402

SIGN = {"left.abd": 1, "left.cam": 1, "left.thigh": -1,        # twinmap.DEFAULT_SIGNS
        "right.abd": -1, "right.cam": -1, "right.thigh": 1}
JOINT_BODY = {"left.abd": "HipLeftNCS-v1", "left.cam": "CamLeftNCS-v1", "left.thigh": "ThighLeftNCS-v1",
              "right.abd": "HipRightNCS-v1", "right.cam": "CamRightNCS-v1", "right.thigh": "ThighRightNCS-v1"}
PHYS_DT, CTRL_HZ = 0.001, 100.0
SERVO_TAU, SERVO_DELAY = 0.084, 0.020
IMU_DELAY = 0.010


def stand_pose():
    import paths            # noqa: F401  (makes the daemon's imports resolvable)
    import daemon
    return dict(daemon.STAND_POSE_DEG)


def build_xml(rb):
    """The homing MJCF made simulatable: closure copies removed, masses swapped, soles, floor,
    motors, a free base, the loop `connect`, position servos."""
    root = ET.parse(sp.CAD).getroot()
    wb = root.find("worldbody")
    # drop the deeper duplicate of each loop body (the CAD's closure copy)
    seen = {}

    def walk(el, depth):
        for b in list(el.findall("body")):
            n = b.get("name")
            seen.setdefault(n, []).append((depth, el, b))
            walk(b, depth + 1)
    walk(wb, 0)
    for n, copies in seen.items():
        for depth, parent, b in sorted(copies, key=lambda c: c[0])[1:]:
            parent.remove(b)
    for g in wb.findall("geom"):                     # the CAD's own ground plane
        wb.remove(g)
    ET.SubElement(wb, "geom", name="floor", type="plane", size="3 3 0.1", friction="0.9 0.005 0.0001",
                  rgba="0.8 0.8 0.8 1")
    comp = root.find("compiler")
    comp.set("meshdir", os.path.dirname(sp.CAD))
    ET.SubElement(root, "option", timestep=str(PHYS_DT), integrator="implicitfast", solver="Newton",
                  iterations="50", cone="elliptic")
    d = ET.SubElement(root, "default")
    ET.SubElement(d, "geom", contype="0", conaffinity="0")
    ET.SubElement(d, "joint", damping="0.05")
    floor = wb.find("geom[@name='floor']")
    floor.set("contype", "1"); floor.set("conaffinity", "1")

    bodies = {b.get("name"): b for b in wb.iter("body")}
    torso = bodies["bodyNCS-v1"]
    torso.insert(0, ET.Element("freejoint", name="base"))
    for n, b in bodies.items():
        ine = b.find("inertial")
        m_cad = float(ine.get("mass"))
        m = rb.mass[n]
        fi = np.array([float(v) for v in ine.get("fullinertia").split()]) * (m / m_cad)
        ine.set("mass", f"{m:.6f}")
        ine.set("fullinertia", " ".join(f"{v:.9g}" for v in fi))
        if n in rb.com:
            ine.set("pos", " ".join(f"{v:.6f}" for v in rb.com[n]))
    for mm, hip, pos in rb.motors:
        mb = ET.SubElement(bodies[hip], "body", name=f"motor_{len(bodies[hip])}_{hip}",
                           pos=" ".join(f"{v:.6f}" for v in pos))
        ET.SubElement(mb, "inertial", pos="0 0 0", mass=f"{mm:.4f}", diaginertia="0.002 0.002 0.002")
    # soles: a thin box whose bottom face is the flat sole face of the mesh
    n = sp.SOLE_N / np.linalg.norm(sp.SOLE_N)
    t = np.array([-n[2], 0.0, n[0]])                  # along the sole, in the sagittal plane
    zb = -n                                           # box +z points INTO the foot
    yb = np.cross(zb, t)
    R = np.column_stack([t, yb, zb])
    import mujoco
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten())
    for S in ("Left", "Right"):
        c = sp.SOLE_C - 0.0015 * n
        ET.SubElement(bodies[f"FootFlat{S}NCS-v1"], "geom", name=f"sole_{S}", type="box",
                      size=f"{sp.SOLE_HALF:.5f} 0.0127 0.0015", pos=" ".join(f"{v:.5f}" for v in c),
                      quat=" ".join(f"{v:.6f}" for v in q), contype="1", conaffinity="1",
                      friction="0.9 0.005 0.0001", rgba="0.1 0.1 0.1 1", solref="0.004 1")
        # loop closure sites: the pin on the pushrod, and the same point on the foot at qpos 0
        ET.SubElement(bodies[f"Pushrod{S}NCS-v1"], "site", name=f"pin_rod_{S}",
                      pos=" ".join(f"{v:.6f}" for v in sp.PIN))
    W0 = rb.fk({})
    eq = ET.SubElement(root, "equality")
    for S in ("Left", "Right"):
        rod, foot = f"Pushrod{S}NCS-v1", f"FootFlat{S}NCS-v1"
        pinw = W0[rod][1] + W0[rod][0] @ sp.PIN
        pf = W0[foot][0].T @ (pinw - W0[foot][1])
        ET.SubElement(bodies[foot], "site", name=f"pin_foot_{S}", pos=" ".join(f"{v:.6f}" for v in pf))
        ET.SubElement(eq, "connect", site1=f"pin_rod_{S}", site2=f"pin_foot_{S}",
                      solref="0.002 1", solimp="0.999 0.9999 0.0001")
    act = ET.SubElement(root, "actuator")
    for name, body in JOINT_BODY.items():
        j = bodies[body].find("joint")
        j.set("armature", "0.046" if name.endswith("abd") else "0.0216")
        frc = 61.2 if name.endswith("abd") else 144.5
        # stiff: the drive's own position loop has integral action, so it does not sag under the
        # 27 N*m stance load the way a finite kp would (kp 3000 sagged 0.5 deg and put the CoM
        # 7 mm behind the sole centre); the servo's slowness is the reference lag, not this
        ET.SubElement(act, "position", name=name, joint=j.get("name"), kp="30000", kv="300",
                      forcerange=f"{-frc} {frc}", forcelimited="true")
    return ET.tostring(root, encoding="unicode")


class Sim:
    def __init__(self, zero_err=None, servo_tau=SERVO_TAU, servo_delay=SERVO_DELAY, imu_noise=0.05,
                 imu_bias=(0.0, 0.0), com_shift=0.0, seed=0):
        import mujoco
        self.mj = mujoco
        self.rb = sp.Robot()
        if com_shift:                     # model error: the real torso CoM is not where we think
            self.rb.com["bodyNCS-v1"] = self.rb.com["bodyNCS-v1"] + np.array([com_shift, 0, 0])
        self.m = mujoco.MjModel.from_xml_string(build_xml(self.rb))
        self.d = mujoco.MjData(self.m)
        self.zero_err = zero_err or {}    # deg: the robot's real joint = commanded + this
        self.tau, self.imu_noise, self.imu_bias = servo_tau, imu_noise, np.array(imu_bias, float)
        self.rng = np.random.default_rng(seed)
        self.act = {n: self.m.actuator(n).id for n in JOINT_BODY}
        self.qadr = {n: self.m.jnt_qposadr[self.m.actuator_trnid[self.act[n], 0]] for n in JOINT_BODY}
        self.delay_n = max(1, int(round(servo_delay / PHYS_DT)))
        self.imu_n = max(1, int(round(IMU_DELAY / PHYS_DT)))
        self.torso = self.m.body("bodyNCS-v1").id
        self.feet = [self.m.geom(f"sole_{S}").id for S in ("Left", "Right")]

    def reset(self, pose_norm):
        mj, m, d = self.mj, self.m, self.d
        mj.mj_resetData(m, d)
        cam, thigh = np.radians(SIGN["left.cam"] * pose_norm["left.cam"]), np.radians(SIGN["left.thigh"] * pose_norm["left.thigh"])
        q = {"CamLeftNCS-v1": cam, "ThighLeftNCS-v1": thigh, "CamRightNCS-v1": -cam, "ThighRightNCS-v1": -thigh}
        self.rb.close(q, "Left"); self.rb.close(q, "Right")
        for bname, val in q.items():
            jid = m.body_jntadr[m.body(bname).id]
            d.qpos[m.jnt_qposadr[jid]] = val
        for n in JOINT_BODY:              # apply the zero error to the REAL joint
            d.qpos[self.qadr[n]] += np.radians(SIGN[n] * self.zero_err.get(n, 0.0))
        d.qpos[3:7] = [1, 0, 0, 0]
        mj.mj_kinematics(m, d)
        # lift/lower so the lowest sole corner touches the floor
        zmin = min(self._sole_zmin(g) for g in self.feet)
        d.qpos[2] -= zmin - 0.0002
        mj.mj_forward(m, d)
        ref = self.ref_of(pose_norm)
        self.lag = dict(ref)
        self.pipe = deque([dict(ref)] * self.delay_n, maxlen=self.delay_n)
        self.imu_pipe = deque([self._imu_true()] * self.imu_n, maxlen=self.imu_n)
        for n in JOINT_BODY:
            d.ctrl[self.act[n]] = ref[n]
        self.t = 0.0
        # settle 0.5 s with the servos holding and a HAND keeping the torso upright, the way the
        # robot is actually brought up: held, started, let go (run() releases it)
        self.hand_until = 1e9
        for _ in range(int(0.5 / PHYS_DT)):
            self._phys(ref)

    def _sole_zmin(self, g):
        m, d = self.m, self.d
        R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; s = m.geom_size[g]
        return min((c + R @ (np.array([sx, sy, sz]) * s))[2]
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1))

    def ref_of(self, pose_norm):
        return {n: np.radians(SIGN[n] * (pose_norm[n] + self.zero_err.get(n, 0.0))) for n in JOINT_BODY}

    def _imu_true(self):
        R = self.d.xmat[self.torso].reshape(3, 3)
        up = R.T @ np.array([0, 0, 1.0])
        w = self.d.cvel[self.torso][:3]               # world-frame angular velocity (rot part first)
        return up, R.T @ w

    def _phys(self, ref_cmd):
        """one physics step: servo delay + first-order lag on the reference, then mj_step"""
        self.pipe.append(ref_cmd)
        r = self.pipe[0]
        if self.t < self.hand_until:      # the hand: a stiff torque holding the torso upright
            R = self.d.xmat[self.torso].reshape(3, 3)
            err = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            w = self.d.cvel[self.torso][:3]
            self.d.xfrc_applied[self.torso, 3:] = -400.0 * err - 40.0 * w
        else:
            self.d.xfrc_applied[self.torso, 3:] = 0.0
        a = PHYS_DT / (self.tau + PHYS_DT)
        for n in JOINT_BODY:
            self.lag[n] += a * (r[n] - self.lag[n])
            self.d.ctrl[self.act[n]] = self.lag[n]
        self.mj.mj_step(self.m, self.d)
        self.imu_pipe.append(self._imu_true())
        self.t += PHYS_DT

    def imu(self):
        up, gyr = self.imu_pipe[0]
        pitch, roll = balance.attitude_from_up(up)
        n = self.rng.normal(0, self.imu_noise, 2)
        return pitch + n[0] + self.imu_bias[0], roll + n[1] + self.imu_bias[1], np.degrees(gyr)

    def lift(self):
        """highest sole corner above the floor, mm (a tipping foot lifts an edge)"""
        return 1e3 * max(max((self.d.geom_xpos[g] + self.d.geom_xmat[g].reshape(3, 3) @ (np.array([sx, sy, -1]) * self.m.geom_size[g]))[2]
                             for sx in (-1, 1) for sy in (-1, 1)) for g in self.feet)

    def push(self, force_xyz, duration):
        self._push = (np.array(force_xyz, float), self.t + duration)

    def run(self, seconds, ctrl=None, pose=None, events=(), log=None, hand_s=1.0):
        """Run with Balancer `ctrl` (None = hold `pose`); the hand lets go after `hand_s`.
        events: [(t, fn)] relative to the start. Returns metrics."""
        steps_per = int(round(1.0 / (CTRL_HZ * PHYS_DT)))
        ref = self.ref_of(pose)
        ev = sorted(events, key=lambda e: e[0])
        t0 = self.t
        self.hand_until = t0 + hand_s
        peak_tilt, peak_lift, fell = 0.0, 0.0, False
        self._push = (np.zeros(3), -1)
        k = 0
        while self.t - t0 < seconds:
            while ev and self.t - t0 >= ev[0][0]:
                ev.pop(0)[1](self)
            if k % steps_per == 0:
                pitch, roll, gyr = self.imu()
                if ctrl is not None:
                    tgt = ctrl.step(1.0 / CTRL_HZ, pitch, roll, gyr[1], gyr[0])
                    ref = self.ref_of(tgt)
                if log is not None:
                    log.append((self.t - t0, pitch, roll, self.lift(), *(ctrl.out.values() if ctrl else (0, 0, 0))))
                tilt = max(abs(pitch), abs(roll))
                peak_tilt = max(peak_tilt, tilt)
                lift = self.lift()
                peak_lift = max(peak_lift, lift)
                if tilt > 15 or lift > 20:
                    fell = True
                    break
            f, tend = self._push
            self.d.xfrc_applied[self.torso, :3] = f if self.t < tend else 0.0
            self._phys(ref)
            k += 1
        pitch, roll, _ = self.imu()
        return dict(fell=fell, peak_tilt=peak_tilt, peak_lift=peak_lift, pitch=pitch, roll=roll)


SCENARIOS = [
    # name, Sim kwargs, events
    ("nominal, 5 s", {}, []),
    # the margins sit just above what the robot survives uncontrolled (see --pushes)
    ("zero error cam +0.65 / thigh -0.65 deg",
     {"zero_err": {"left.cam": .65, "right.cam": .65, "left.thigh": -.65, "right.thigh": -.65}}, []),
    ("zero error cam -0.65 / thigh +0.65 deg",
     {"zero_err": {"left.cam": -.65, "right.cam": -.65, "left.thigh": .65, "right.thigh": .65}}, []),
    ("L/R asymmetric zero error (thigh L +2, R -1; cam L -1)",
     {"zero_err": {"left.thigh": 2, "right.thigh": -1, "left.cam": -1}}, []),
    ("abd zero error L +1.5, R +0.5 deg",
     {"zero_err": {"left.abd": 1.5, "right.abd": 0.5, "left.thigh": 0.5}}, []),
    ("torso CoM 20 mm forward of the model", {"com_shift": 0.020}, []),
    ("push forward 8 N for 0.2 s (the physical limit)", {}, [(2.0, lambda s: s.push((8, 0, 0), 0.2))]),
    ("push backward 8 N for 0.2 s", {}, [(2.0, lambda s: s.push((-8, 0, 0), 0.2))]),
    ("push sideways 47 N for 0.2 s", {}, [(2.0, lambda s: s.push((0, 47, 0), 0.2))]),
    ("push forward 8 N, servo lag x2", {"servo_tau": 2 * SERVO_TAU, "servo_delay": 2 * SERVO_DELAY},
     [(2.0, lambda s: s.push((8, 0, 0), 0.2))]),
    ("IMU mount off 1 deg pitch (levels to the WRONG vertical)", {"imu_bias": (1.0, 0.0)}, []),
]


def run_table(duration=5.0):
    stand = stand_pose()
    rows = []
    for name, kw, events in SCENARIOS:
        res = {}
        for mode in ("off", "on"):
            s = Sim(**kw)
            s.reset(stand)
            ctrl = balance.Balancer(stand) if mode == "on" else None
            res[mode] = s.run(duration, ctrl=ctrl, pose=stand, events=events)
        rows.append((name, res))
        f = lambda r: ("FELL" if r["fell"] else "ok  ") + f" peak {r['peak_tilt']:4.1f} deg lift {r['peak_lift']:5.1f} mm end p{r['pitch']:+5.2f} r{r['roll']:+5.2f}"
        print(f"{name:52s} | OFF {f(res['off'])} | ON {f(res['on'])}", flush=True)
    return rows


def max_push(direction, ctrl_on, sim_kw=None, lo=0.0, hi=200.0, dur=0.2, iters=7):
    """Largest force (N, held `dur` s, 1 s after the hand lets go) the robot survives."""
    stand = stand_pose()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = Sim(**(sim_kw or {}))
        s.reset(stand)
        f = np.array(direction, float) * mid
        r = s.run(4.0, ctrl=balance.Balancer(stand) if ctrl_on else None, pose=stand,
                  events=[(2.0, lambda s, f=f: s.push(f, dur))])
        lo, hi = (mid, hi) if not r["fell"] else (lo, mid)
    return lo


def push_table(sim_kw=None, label="nominal"):
    for name, dirn in (("forward", (1, 0, 0)), ("backward", (-1, 0, 0)), ("left", (0, 1, 0))):
        off, on = max_push(dirn, False, sim_kw), max_push(dirn, True, sim_kw)
        print(f"max survivable push {name:8s} ({label}), 0.2 s: OFF {off:5.0f} N   ON {on:5.0f} N", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pushes", action="store_true", help="also bisect the largest survivable push")
    ap.add_argument("--plot", help="save a time plot of the forward push, off vs on")
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args()
    run_table(a.seconds)
    if a.pushes:
        push_table()
        push_table({"servo_tau": 2 * SERVO_TAU, "servo_delay": 2 * SERVO_DELAY}, "servo lag x2")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        stand = stand_pose()
        fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for mode in ("off", "on"):
            s = Sim(); s.reset(stand); log = []
            s.run(4.0, ctrl=balance.Balancer(stand) if mode == "on" else None, pose=stand,
                  events=[(2.0, lambda s: s.push((8, 0, 0), 0.2))], log=log)
            L = np.array(log)
            ax[0].plot(L[:, 0], L[:, 1], label=f"pitch, balance {mode}")
            ax[1].plot(L[:, 0], L[:, 3], label=f"sole lift, balance {mode}")
            if mode == "on":
                ax[2].plot(L[:, 0], L[:, 4], label="pitch posture correction (deg)")
        for x, lab in zip(ax, ("deg", "mm", "deg")):
            x.set_ylabel(lab); x.legend(); x.grid(alpha=.3)
        ax[2].set_xlabel("s (hand lets go at 1.0 s; 8 N forward push at 2.0-2.2 s)")
        fig.tight_layout(); fig.savefig(a.plot, dpi=120)
        print("wrote", a.plot)


if __name__ == "__main__":
    main()
