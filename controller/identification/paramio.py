"""Read/write `identified_params.json` — the intermediate format the estimator produces, the model
write-back consumes, and the web UI displays. PURE stdlib (usable on the Pi).

Schema (all SI: kg, m, kg*m^2, Nm, Nm/A; angles in the model radian frame):
{
  "created": iso8601,
  "kt":            {motor_name: Nm/A},
  "friction":      {motor_name: {"viscous": Nm/(rad/s), "coulomb": Nm}},
  "rotor_armature":{motor_name: kg*m^2 (reflected to the joint)},
  "bodies": {
     body_name: {"mass": kg, "com": [x,y,z], "inertia": {ixx..iyz},
                 "uncertainty": {optional per-field 1-sigma}, "cad_delta": {optional}}
  },
  "validation": {"residual_rms_nm": .., "held_out_residual_rms_nm": ..},
  "sources": [measurement file names],
  "notes": str
}
Every section is optional — a static-only run may fill just kt/friction/com, a dynamic run adds the
inertias. The model write-back falls back to CAD for anything absent.
"""
import json
import os
import time


def default():
    return {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "units": {"mass": "kg", "length": "m", "inertia": "kg*m^2", "torque": "Nm",
                      "kt": "Nm/A", "viscous": "Nm/(rad/s)", "coulomb": "Nm"},
            "kt": {}, "friction": {}, "rotor_armature": {}, "bodies": {},
            "validation": {}, "sources": [], "notes": ""}


def load(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_or_none(path):
    try:
        return load(path)
    except (OSError, ValueError):
        return None


def save(params, path):
    params = dict(params)
    params.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
    params["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    os.replace(tmp, path)
    return path
