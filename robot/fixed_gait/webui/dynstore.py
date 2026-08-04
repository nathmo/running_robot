"""Persistent dynamics-identification config: weighed limb masses, the drive's per-joint PID gains,
and per-motor torque constants Kt.

None of this is control state (position-mode excitation uses the drive's own loop), so it lives in a
plain JSON side-file rather than the daemon. It is:
  * edited from the Limbs & Inertia / Dynamic-ID panels,
  * embedded into each measurement run's metadata at capture time (so a run is self-describing),
  * consumed by the offline estimator (robot/identification/): masses fix each body's total mass,
    Kt converts the logged current to torque, the PID gains let the estimator model / cross-check the
    drive's position loop.

Masses are keyed by MODEL BODY name (so they line up with the STL meshes shown in the 3D viewer);
pid/kt are keyed by webui motor name (paths.MOTOR_NAMES). All values are plain floats; a key absent
from BOTH the file and DEFAULT_MASSES means "not measured yet" and falls back to the CAD/model value
downstream.
"""
import json
import os
import threading
import time

import paths

# The 13 rigid bodies of the CAD model (mesh/body names), grouped for the UI. Torso is base-fixed
# (mass weighable, but its inertia is NOT identifiable from joint torques — see the plan).
BODY_NAMES = [
    "bodyNCS-v1",
    "HipLeftNCS-v1", "HipRightNCS-v1",
    "CamLeftNCS-v1", "CamRightNCS-v1",
    "ThighLeftNCS-v1", "ThighRightNCS-v1",
    "LegLeftNCS-v1", "LegRightNCS-v1",
    "FootLeftNCS-v1", "FootRightNCS-v1",
    "PushrodLeftNCS-v1", "PushrodRightNCS-v1",
]
# The 6 actuator "motor" point-masses welded at the joints (build_model.py add_motor_masses).
MOTOR_BODIES = ["motor_hip_roll_L", "motor_cam_L", "motor_thigh_L",
                "motor_hip_roll_R", "motor_cam_R", "motor_thigh_R"]

# webui motor name (paths.MOTOR_NAMES)  ->  model motor-body name, for Kt/PID <-> body cross-ref.
MOTOR_NAME_TO_BODY = {
    "right.abd": "motor_hip_roll_R", "right.cam": "motor_cam_R", "right.thigh": "motor_thigh_R",
    "left.abd": "motor_hip_roll_L", "left.cam": "motor_cam_L", "left.thigh": "motor_thigh_L",
}

# CubeMars servo-firmware default position-loop gains are unknown to us; these are neutral seeds the
# user overwrites with the values their controller actually uses (answer to the PID question).
DEFAULT_PID = {"kp": 0.0, "ki": 0.0, "kd": 0.0}

# The user's WEIGHED segment masses (2026-08-03), in kg. These seed every fresh install so a new Pi
# (data/ is git-ignored) starts on the real plant instead of the CAD placeholders in
# robot/model/dash01.xml, whose total is 12.83 kg against a real 15.14 kg (+18%).
#
# These are LINK-ONLY masses: set_mass writes straight into model.body_mass[body] (see
# identification/kt_calibration.set_masses) and the motors are their OWN bodies (MOTOR_BODIES,
# welded on by build_model.add_motor_masses), so the motors weighed inside an assembly must be
# subtracted here or their mass is counted twice. Same split as
# training/model/apply_measured_masses.GROUPS -- keep the two in sync:
#   base  5.764 (incl. both abduction motors + battery) - 2*0.750 = 4.264
#   hip   3.271 (incl. its cam + thigh motors)          - 2*1.400 = 0.471
#   shin  0.324 + 0.249 ankle spring                              = 0.573
# (the spring is not a body in the CAD tree; it runs parallel to and very close to the shin)
DEFAULT_MASSES = {
    "bodyNCS-v1": 4.264,
    "HipLeftNCS-v1": 0.471, "HipRightNCS-v1": 0.471,
    "CamLeftNCS-v1": 0.066, "CamRightNCS-v1": 0.066,
    "PushrodLeftNCS-v1": 0.071, "PushrodRightNCS-v1": 0.071,
    "ThighLeftNCS-v1": 0.483, "ThighRightNCS-v1": 0.483,
    "LegLeftNCS-v1": 0.573, "LegRightNCS-v1": 0.573,
    "FootLeftNCS-v1": 0.222, "FootRightNCS-v1": 0.222,
}


class DynConfig:
    def __init__(self):
        self._lock = threading.Lock()
        self.masses = dict(DEFAULT_MASSES)      # body name -> kg
        self.pid = {n: dict(DEFAULT_PID) for n in paths.MOTOR_NAMES}
        self.kt = {}                            # motor name -> Nm/A (filled by Kt calibration)
        self.updated = None

    @classmethod
    def load_or_new(cls, path=paths.DYN_CONFIG_FILE):
        c = cls()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8-sig") as f:
                    d = json.load(f)
                # the file holds the user's edits; anything it does not mention keeps its default
                c.masses.update({k: float(v) for k, v in (d.get("masses") or {}).items()})
                for n, g in (d.get("pid") or {}).items():
                    if n in c.pid:
                        c.pid[n] = {k: float(g.get(k, 0.0)) for k in ("kp", "ki", "kd")}
                c.kt = {k: float(v) for k, v in (d.get("kt") or {}).items()}
                c.updated = d.get("updated")
            except (ValueError, OSError) as e:
                print(f"(could not read {path}: {e} — starting with empty dynamics config)")
        return c

    def save(self, path=paths.DYN_CONFIG_FILE):
        with self._lock:
            d = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "masses": dict(self.masses), "pid": dict(self.pid), "kt": dict(self.kt)}
            self.updated = d["updated"]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, path)

    # ------------------------------------------------------------------ mutators
    def set_mass(self, body, kg):
        if kg is None or kg == "":
            # blanking the field reverts to the weighed default, not to "unmeasured"
            with self._lock:
                if body in DEFAULT_MASSES:
                    self.masses[body] = DEFAULT_MASSES[body]
                else:
                    self.masses.pop(body, None)
        else:
            kg = float(kg)
            if not 0.0 < kg < 100.0:
                return False, f"mass {kg} kg out of a sane range (0, 100)"
            with self._lock:
                self.masses[body] = kg
        self.save()
        return True, ""

    def set_pid(self, motor, kp=None, ki=None, kd=None):
        if motor not in self.pid:
            return False, f"unknown motor {motor}"
        with self._lock:
            g = self.pid[motor]
            for key, val in (("kp", kp), ("ki", ki), ("kd", kd)):
                if val is not None:
                    g[key] = float(val)
        self.save()
        return True, ""

    def set_kt(self, kt_by_motor):
        with self._lock:
            self.kt.update({k: float(v) for k, v in kt_by_motor.items() if k in paths.MOTOR_NAMES})
        self.save()

    # ------------------------------------------------------------------ views
    def as_dict(self):
        with self._lock:
            return {"updated": self.updated, "masses": dict(self.masses),
                    "pid": {n: dict(g) for n, g in self.pid.items()}, "kt": dict(self.kt)}

    def snapshot(self):
        """Config + the fixed body/motor catalog the frontend needs to lay out the panels."""
        d = self.as_dict()
        d["bodies"] = list(BODY_NAMES)
        d["motor_bodies"] = list(MOTOR_BODIES)
        d["motor_to_body"] = dict(MOTOR_NAME_TO_BODY)
        return d
