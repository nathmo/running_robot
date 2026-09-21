"""Standing balance: IMU attitude -> joint posture corrections around the standing pose.

Pure numpy, no daemon state, so the SAME object runs in daemon.py (MANUAL mode, ⚖ Balance) and in
tools/balance_sim.py (MuJoCo), and what the simulation tested is what the robot runs.

WHAT THE IMU CAN AND CANNOT SEE. Both soles flat on the floor and the joints stiff make the torso a
rigid part of the mechanism: its attitude is set by the joint angles, not by where the CoM sits on
the sole. So the IMU sees two things only: a torso that is not level (zero / model error), and a
foot that has started to TIP onto its toe, heel or outer edge. It cannot see the CoM position while
the feet are flat -- that is the operator's trim below, set by eye or from the motor currents.

The posture coordinates, both legs alike, derived on the homing CAD (tools/solve_stand_pose.py,
`balance_maps`; linear to 0.5 mm / 0.1 deg over the clip ranges below; the
CoM trims shape the standing pose itself, PID or not):

  PITCH   torso pitch with the soles flat and the CoM fixed over them: how the pitch integrator
          levels the torso. (On a toe or heel it would only spin the body about its CoM.)
  COM_X   CoM forward along the sole with the soles flat and the torso level: the pitch P + D
          (to pull the CoM back from an edge) plus the operator's trim.
  COM_Y   CoM to the LEFT: both hips rolled the same way, a parallelogram. The torso stays level
          and the soles roll by the same small angle (no ankle roll joint exists to prevent it: a
          25 mm wide sole lifts an edge 0.4 mm per degree). 13.8 mm of CoM per degree of abd.

Differential leg length is NOT used to level roll: 5 mm from the four-bar's full stretch a longer
leg is mostly a foot moved 5 mm fore/aft per mm of length, which fights the other foot.

WHAT THE SIMULATION SAYS (tools/balance_sim.py, MuJoCo, measured SET_POS lag):
* Pitch: P + D -> CoM fore/aft, I -> PITCH posture (levels the torso). This raised the zero
  error the robot survives from 0.52 to 0.75 deg (cam and thigh off in opposite directions, the
  bad case). Negative gains made it worse.
* Roll: P + D -> COM_Y with the COUNTER-INTUITIVE sign: leaning right moves the CoM further RIGHT.
  To move the CoM, the centre of pressure must first move the other way, and a hip shift toward
  the fall pushes the stance foot's ground reaction outward. The intuitive sign cut the survivable
  side push from 45 to 25 N; this one gives 49 N. No integrator: roll is not observable while both
  feet are flat, so it would only wind up.
* Pushes fore/aft: ~8.4 N for 0.2 s with or without control. That is physics, not tuning: a
  14.4 kg robot with its CoM 0.79 m up on a +-33 mm sole can recover at most omega * 33 mm =
  0.12 m/s of CoM speed (1.7 N*s). Only a step, or a bigger foot, beats it.
* The dominant real-world risk is the ZERO: a 0.75 deg cam/thigh zero error already puts the CoM
  on a sole edge. The CoM trims below are how the operator centres it.

Angles are right-handed in the robot frame (x forward, y left, z up): pitch > 0 = leaning FORWARD
(nose down), roll > 0 = leaning RIGHT (left side up).
"""
import numpy as np

# normalized deg per unit, per joint role; the same on both legs unless noted
D_PITCH = {"cam": 1.2160, "thigh": 0.9991}          # per deg of torso pitch (forward +)
D_COM_X = {"cam": -0.10538, "thigh": -0.18297}      # per mm of CoM forward along the sole
D_COM_Y_ABD = -0.0726                                # left.abd deg per mm of CoM LEFT; right.abd = -that

ROLES = ("abd", "cam", "thigh")
SIDES = ("left", "right")

DEFAULTS = dict(
    kp=3.0, kd=0.8,                   # pitch P, D: mm of CoM shift per deg of tilt (*s)
    ki=0.5,                           # pitch I: deg of torso-pitch posture per deg*s of tilt
    kp_roll=-6.0, kd_roll=-0.8,       # roll: mm of lateral CoM shift per deg of tilt (*s); the
                                      # sign is deliberate, see the module docstring
    pitch_clip=6.0,                   # deg of posture correction
    com_x_clip=25.0, com_y_clip=30.0,  # mm the LOOP may add on top of the trim
    trim_clip=25.0,                   # mm, the operator's CoM trims (both axes)
    rate_tau=0.03,                    # s, low-pass on the derivative (gyro) terms
    fall_deg=15.0,                    # |tilt| beyond this = falling: the caller stops
)


# what the web UI may tune live (daemon.balance_gains), and the range each is clipped to
TUNABLE = ("kp", "kd", "ki", "kp_roll", "kd_roll")
GAIN_RANGE = {"kp": (0.0, 20.0), "kd": (0.0, 5.0), "ki": (0.0, 5.0),
              "kp_roll": (-20.0, 20.0), "kd_roll": (-5.0, 5.0)}


def attitude_from_up(up_body):
    """(pitch_fwd, roll_right) in DEGREES from world-up expressed in body axes (sensehat.fast)."""
    ux, uy, uz = (float(v) for v in up_body)
    return np.degrees(np.arctan2(-ux, uz)), np.degrees(np.arctan2(uy, uz))


class Balancer:
    def __init__(self, stand_pose, **params):
        self.stand = {n: float(v) for n, v in stand_pose.items()}
        self.p = dict(DEFAULTS)
        self.p.update({k: float(v) for k, v in params.items() if k in DEFAULTS})
        self.trim = {"pitch_deg": 0.0, "com_x_mm": 0.0, "com_y_mm": 0.0}
        self.reset()

    def reset(self):
        self.i_pitch = 0.0
        self.q_rate = np.zeros(2)            # filtered (pitch_rate, roll_rate) deg/s
        self.out = {"pitch": 0.0, "com_x": 0.0, "com_y": 0.0}
        self.last = {}

    def set_trim(self, pitch_deg=None, com_x_mm=None, com_y_mm=None):
        if pitch_deg is not None:
            self.trim["pitch_deg"] = float(np.clip(pitch_deg, -5.0, 5.0))
        if com_x_mm is not None:
            self.trim["com_x_mm"] = float(np.clip(com_x_mm, -self.p["trim_clip"], self.p["trim_clip"]))
        if com_y_mm is not None:
            self.trim["com_y_mm"] = float(np.clip(com_y_mm, -self.p["trim_clip"], self.p["trim_clip"]))

    def falling(self, pitch, roll):
        return max(abs(pitch), abs(roll)) > self.p["fall_deg"]

    def step(self, dt, pitch, roll, pitch_rate, roll_rate):
        """One control tick. Angles deg, rates deg/s (gyro y = pitch rate, gyro x = roll rate).
        Returns {motor: normalized deg} for all six joints."""
        p = self.p
        a = dt / (p["rate_tau"] + dt)
        self.q_rate += a * (np.array([pitch_rate, roll_rate], float) - self.q_rate)
        pr, rr = self.q_rate

        # pitch, split by speed. P + D -> CoM fore/aft: a tip is fast, and only moving the CoM
        # relative to the FOOT brings it back behind the toe/heel edge (the PITCH posture keeps the
        # CoM fixed on the sole, so on an edge it would just spin the body about its CoM).
        # I -> PITCH posture: levels the torso against zero / model error; while the feet are flat
        # the torso pitches by exactly the commanded amount, so the correction is minus the error.
        # At rest the P term is zero, so the CoM goes back to the trim.
        e = pitch - self.trim["pitch_deg"]
        clip = p["pitch_clip"]
        # anti-windup: integrate unless already clipped in the direction the error pushes
        if abs(self.i_pitch) < clip or np.sign(e) != np.sign(self.i_pitch):
            self.i_pitch = float(np.clip(self.i_pitch + p["ki"] * e * dt, -clip, clip))
        u_pitch = -self.i_pitch
        # the loop's authority is clipped AROUND the trim, so a large trim does not eat it
        dx = float(np.clip(-(p["kp"] * e + p["kd"] * pr), -p["com_x_clip"], p["com_x_clip"]))
        cx = self.trim["com_x_mm"] + dx

        # roll PD -> lateral CoM shift (kp_roll < 0: leaning right moves the CoM right, see top)
        dy = float(np.clip(p["kp_roll"] * roll + p["kd_roll"] * rr, -p["com_y_clip"], p["com_y_clip"]))
        cy = self.trim["com_y_mm"] + dy

        self.out = {"pitch": u_pitch, "com_x": cx, "com_y": cy}
        self.last = {"pitch": pitch, "roll": roll, "pitch_rate": pr, "roll_rate": rr,
                     "i_pitch": self.i_pitch, "sat_pitch": abs(u_pitch) >= clip - 1e-9,
                     "sat_com_x": abs(dx) >= p["com_x_clip"] - 1e-9}
        return self.targets(u_pitch, cx, cy)

    def targets(self, pitch_corr, com_x, com_y):
        t = dict(self.stand)
        for side in SIDES:
            for role in ("cam", "thigh"):
                t[f"{side}.{role}"] += D_PITCH[role] * pitch_corr + D_COM_X[role] * com_x
        t["left.abd"] += D_COM_Y_ABD * com_y
        t["right.abd"] -= D_COM_Y_ABD * com_y
        return t
