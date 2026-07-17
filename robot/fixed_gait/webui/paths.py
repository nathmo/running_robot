"""Shared paths + motor naming for the DASH-01 web UI.

Importing this module makes the flat fixed_gait/ scripts importable (same sys.path trick as
fixed_gait/validate_gait.py) and guarantees the runtime data directories exist. Every webui module
imports paths FIRST, before any fixed_gait import.
"""
import os
import sys

WEBUI = os.path.dirname(os.path.abspath(__file__))
FIXED_GAIT = os.path.dirname(WEBUI)
REPO = os.path.dirname(FIXED_GAIT)

for p in (FIXED_GAIT, WEBUI):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA = os.path.join(WEBUI, "data")
TRAJ_DIR = os.path.join(DATA, "trajectories")
WORKSPACE_DIR = os.path.join(DATA, "workspaces")
CALIB_FILE = os.path.join(DATA, "session_calibration.json")
MODEL_MAP_FILE = os.path.join(DATA, "model_map.json")
FK_LUT_FILE = os.path.join(WEBUI, "fk_lut.npz")

for d in (DATA, TRAJ_DIR, WORKSPACE_DIR):
    os.makedirs(d, exist_ok=True)

# ---- global motor naming / ordering (matches telemetry columns everywhere) ----
# can0 = RIGHT leg, can1 = LEFT leg; per bus: 104=abduction, 105=cam, 106=hip/thigh
# (fixed_gait/run_hardware.py CALIB + fixed_gait/README.md)
SIDES = ("right", "left")
ROLES = ("abd", "cam", "thigh")
SIDE_CHANNEL = {"right": "can0", "left": "can1"}
ROLE_ID = {"abd": 104, "cam": 105, "thigh": 106}
MOTOR_NAMES = [f"{s}.{r}" for s in SIDES for r in ROLES]   # right.abd ... left.thigh
N_MOTORS = len(MOTOR_NAMES)


def motor_index(name):
    return MOTOR_NAMES.index(name)


def split_name(name):
    side, role = name.split(".")
    return side, role
