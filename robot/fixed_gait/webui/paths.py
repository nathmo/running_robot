"""Shared paths + motor naming for the DASH-01 web UI.

Importing this module makes the flat fixed_gait/ scripts importable (same sys.path trick as
fixed_gait/validate_gait.py) and guarantees the runtime data directories exist. Every webui module
imports paths FIRST, before any fixed_gait import.
"""
import os
import sys

WEBUI = os.path.dirname(os.path.abspath(__file__))
FIXED_GAIT = os.path.dirname(WEBUI)
REPO = os.path.dirname(FIXED_GAIT)                 # the robot/ dir (holds identification/, model/)

# REPO is added so the offline identification package (robot/identification/) is importable from
# the webui when its deps (mujoco/scipy) happen to be present; the pure-numpy frames helper is used
# by the inertia-comparison endpoint even on the Pi.
for p in (FIXED_GAIT, WEBUI, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA = os.path.join(WEBUI, "data")
TRAJ_DIR = os.path.join(DATA, "trajectories")
WORKSPACE_DIR = os.path.join(DATA, "workspaces")
MEASURE_DIR = os.path.join(DATA, "measurements")      # high-rate system-ID capture runs (.npz+.json)
IDENT_DIR = os.path.join(DATA, "identification")       # estimator outputs (identified_params.json)
CALIB_FILE = os.path.join(DATA, "session_calibration.json")
MODEL_MAP_FILE = os.path.join(DATA, "model_map.json")
DYN_CONFIG_FILE = os.path.join(DATA, "dynamics_config.json")   # weighed masses, drive PID gains, Kt
FK_LUT_FILE = os.path.join(WEBUI, "fk_lut.npz")

# the MuJoCo model whose inertials are the CAD "given" values compared in the Limbs & Inertia panel
MODEL_XML = os.path.join(REPO, "model", "dash01.xml")

# STL meshes for the 3D limb viewer (robot/model/dash01.xml points its meshdir at the CAD export).
_MESH_CANDIDATES = [
    os.path.join(REPO, "robotCADdescription", "MJCF_OPEN_MUJOCO_B", "dash01", "meshes"),
    os.path.join(REPO, "model", "meshes"),
    os.path.join(os.path.dirname(REPO), "training", "model", "meshes"),
]
MESH_DIR = next((d for d in _MESH_CANDIDATES if os.path.isdir(d)), _MESH_CANDIDATES[0])
MESH_SCALE = 0.001                      # STLs are in mm; the model scales them by 0.001

for d in (DATA, TRAJ_DIR, WORKSPACE_DIR, MEASURE_DIR, IDENT_DIR):
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
