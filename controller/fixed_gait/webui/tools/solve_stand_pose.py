#!/usr/bin/env python3
"""Solve the standing pose that 🏠 Home drives to (daemon.STAND_POSE_DEG). Dev machine only
(needs mujoco + scipy).

    python controller/fixed_gait/webui/tools/solve_stand_pose.py

The pose: abduction 0, torso level, BOTH SOLES FLAT on the floor and the whole-robot CoM straight
above the sole centres, so the robot balances statically on its two feet. With the legs mirrored
that is two equations (sole pitch, CoM x) in two unknowns (cam, thigh); y is centred by symmetry.

Kinematics are the homing CAD's (dash-01CAD/homing, the twin's model): its qpos 0 IS normalized
0 deg, and its pushrod pin is the verified one (389 mm, see static/twin3d.js). The loop is closed
in closed form on the CAD's assembly branch, exactly like the twin.
Masses are the RL plant's (RLframework/model/dash01_free.xml: weighed torso, the six motors as
point masses at their pivots); the moving links keep their CAD COM locations. The sole is the flat
66.5 x 25.4 mm face of the FootFlat mesh: centre (-516.8, 0, -92.5) mm, outward normal
(-0.866, 0, -0.5) in the foot frame, which matches the RL model's two sole boxes.
The cam is a crank, so there are two solutions; the one nearer the homing pose is printed first.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CAD = os.path.join(ROOT, "dash-01CAD", "homing", "SpiderBotInitPos", "SpiderBotInitPos.xml")
RL = os.path.join(ROOT, "RLframework", "model", "dash01_free.xml")

PIN = np.array([0.0, 0.0, -0.389])                 # pushrod lower pin, pushrod frame
SOLE_C = np.array([-0.5168, 0.0, -0.0925])         # sole centre, foot frame
SOLE_N = np.array([-0.866, 0.0, -0.5])             # sole outward normal, foot frame
SOLE_HALF = 0.0665 / 2
SIGNS = {"cam": 1, "thigh": -1}                    # left leg, twinmap.DEFAULT_SIGNS


def rot(a, q):
    a = np.asarray(a, float) / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def vec(s):
    return np.array([float(x) for x in s.split()])


class Robot:
    def __init__(self):
        import mujoco
        root = ET.parse(CAD).getroot()
        self.B = {}
        self._walk(root.find("worldbody"), None, 0)
        self.order = sorted(self.B, key=lambda n: self.B[n]["depth"])
        m = mujoco.MjModel.from_xml_path(RL)
        rl = lambda n: float(m.body(n).mass[0])
        self.mass = {"bodyNCS-v1": rl("torso") + rl("motor_hip_roll_L") + rl("motor_hip_roll_R")}
        self.com = {"bodyNCS-v1": (rl("torso") * m.body("torso").ipos
                                   + rl("motor_hip_roll_L") * m.body("motor_hip_roll_L").pos
                                   + rl("motor_hip_roll_R") * m.body("motor_hip_roll_R").pos)
                    / self.mass["bodyNCS-v1"]}
        for S, s in (("Left", "L"), ("Right", "R")):
            self.mass[f"Hip{S}NCS-v1"] = rl(f"hip_{s}")
            self.com[f"Hip{S}NCS-v1"] = m.body(f"hip_{s}").ipos
            for cad, rlname in (("Cam", "cam"), ("Pushrod", "rod"), ("Thigh", "thigh"), ("FootFlat", "leg")):
                self.mass[f"{cad}{S}NCS-v1"] = rl(f"{rlname}_{s}")      # CAD COM location kept
        self.motors = [(rl(f"motor_{k}_{s}"), f"Hip{S}NCS-v1", m.body(f"motor_{k}_{s}").pos)
                       for S, s in (("Left", "L"), ("Right", "R")) for k in ("cam", "thigh")]
        self.total = sum(self.mass.values()) + sum(x[0] for x in self.motors)

    def _walk(self, el, parent, depth):
        for b in el.findall("body"):
            n = b.get("name")
            if n in self.B and self.B[n]["depth"] <= depth:
                continue                             # the DEEPER duplicate = the loop's closure copy
            j, ine = b.find("joint"), b.find("inertial")
            e = vec(b.get("euler", "0 0 0"))
            self.B[n] = dict(parent=parent, depth=depth, p0=vec(b.get("pos", "0 0 0")),
                             R0=rot([1, 0, 0], e[0]) @ rot([0, 1, 0], e[1]) @ rot([0, 0, 1], e[2]),
                             axis=vec(j.get("axis")) if j is not None else None,
                             jpos=vec(j.get("pos", "0 0 0")) if j is not None else None,
                             com=vec(ine.get("pos")))
            self._walk(b, n, depth + 1)

    def fk(self, q):
        W = {}
        for n in self.order:
            b = self.B[n]
            Rp, pp = (np.eye(3), np.zeros(3)) if b["parent"] is None else W[b["parent"]]
            R, p = Rp @ b["R0"], pp + Rp @ b["p0"]
            if b["axis"] is not None and q.get(n):
                Rj = rot(b["axis"], q[n])
                p, R = p + R @ (b["jpos"] - Rj @ b["jpos"]), R @ Rj
            W[n] = (R, p)
        return W

    def close(self, q, S):
        """Closed-form four-bar (same maths as twin3d.js solveLoops). Returns h^2 (<0: no assembly)."""
        rod, oth = f"Pushrod{S}NCS-v1", f"FootFlat{S}NCS-v1"
        q[rod] = q[oth] = 0.0
        W0 = self.fk({})
        pin0 = W0[rod][1] + W0[rod][0] @ PIN
        pin_oth = W0[oth][0].T @ (pin0 - W0[oth][1])
        W = self.fk(q)
        n = W[rod][0] @ self.B[rod]["axis"]
        n = n / np.linalg.norm(n)
        flat = lambda v: v - n * (v @ n)
        P = W[rod][1] + W[rod][0] @ self.B[rod]["jpos"]
        K = W[oth][1] + W[oth][0] @ self.B[oth]["jpos"]
        A0, B0 = W[rod][1] + W[rod][0] @ PIN, W[oth][1] + W[oth][0] @ pin_oth
        Lp, La = np.linalg.norm(flat(A0 - P)), np.linalg.norm(flat(B0 - K))
        branch = np.sign(np.cross(n, flat(W0[oth][1] - W0[rod][1])) @ flat(pin0 - W0[rod][1]))
        d = flat(K - P)
        D = np.linalg.norm(d)
        u = d / D
        a = (Lp ** 2 - La ** 2 + D ** 2) / (2 * D)
        h2 = Lp ** 2 - a ** 2
        X = P + a * u + branch * np.sqrt(max(h2, 0.0)) * np.cross(n, u)
        ang = lambda ax, v0, v1: np.arctan2(ax @ np.cross(v0, v1), v0 @ v1)
        q[rod] = ang(n, flat(A0 - P), flat(X - P))
        no = W[oth][0] @ self.B[oth]["axis"]
        q[oth] = ang(no / np.linalg.norm(no), flat(B0 - K), flat(X - K))
        return h2

    def pose(self, cam, thigh):
        """Left leg (cam, thigh) in MJCF qpos rad, right leg mirrored, abduction 0."""
        q = {"CamLeftNCS-v1": cam, "ThighLeftNCS-v1": thigh,
             "CamRightNCS-v1": -cam, "ThighRightNCS-v1": -thigh}
        h2 = min(self.close(q, "Left"), self.close(q, "Right"))
        W = self.fk(q)
        com = sum(self.mass[n] * (p + R @ self.com.get(n, self.B[n]["com"])) for n, (R, p) in W.items())
        com = (com + sum(mm * (W[hip][1] + W[hip][0] @ pos) for mm, hip, pos in self.motors)) / self.total
        R, p = W["FootFlatLeftNCS-v1"]
        return dict(q=q, W=W, com=com, h2=h2, sole=p + R @ SOLE_C, normal=R @ SOLE_N)

    def residual(self, x):
        s = self.pose(*x)
        return [np.arctan2(s["normal"][0], -s["normal"][2]),     # sole pitch (rad)
                (s["com"][0] - s["sole"][0]) / 0.05,              # CoM ahead of the sole centre
                min(s["h2"], 0.0) * 1e3]                          # the loop must assemble

    def solve(self):
        from scipy.optimize import least_squares
        sols = set()
        for c0 in np.radians(np.arange(-60, 61, 15)):
            for t0 in np.radians(np.arange(-60, 61, 15)):
                r = least_squares(self.residual, [c0, t0])
                if np.linalg.norm(r.fun) < 1e-8:
                    sols.add(tuple(np.round(r.x, 6)))
        return sorted(sols, key=lambda x: abs(x[0]) + abs(x[1]))


def main():
    rb = Robot()
    print(f"total mass {rb.total:.2f} kg")
    for cam, thigh in rb.solve():
        s = rb.pose(cam, thigh)
        W = s["W"]
        norm_cam, norm_thigh = SIGNS["cam"] * np.degrees(cam), SIGNS["thigh"] * np.degrees(thigh)
        print(f"STAND_POSE_DEG: cam {norm_cam:+.1f}, thigh {norm_thigh:+.1f}, abd 0.0 (both legs)   "
              f"[qpos L cam {np.degrees(cam):.3f} thigh {np.degrees(thigh):.3f} deg]")
        print(f"   hip {1e3 * (W['HipLeftNCS-v1'][1][2] - s['sole'][2]):.0f} mm above the sole, CoM "
              f"{1e3 * (s['com'][2] - s['sole'][2]):.0f} mm up, {1e3 * s['com'][0]:.1f} mm ahead of the "
              f"hip axis, sole +-{1e3 * SOLE_HALF:.1f} mm")
        # how much a zero error costs: CoM travel along the sole per degree of cam+thigh error
        for dc, dt in ((3, 3), (3, -3)):
            e = rb.pose(cam + np.radians(dc), thigh + np.radians(dt))
            print(f"   zero error cam {dc:+d} / qpos thigh {dt:+d} deg -> CoM moves "
                  f"{1e3 * (e['com'][0] - e['sole'][0]):+.1f} mm on the sole")


if __name__ == "__main__":
    main()
