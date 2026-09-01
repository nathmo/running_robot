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
# robot/deploy/ holds the pure, hardware-free control and thermal modules (thermal_excite,
# thermal, mit, jointmap). They are deliberately importable from the webui but import nothing from
# it -- the dependency runs one way, so the deployed control law stays reviewable on its own.
DEPLOY = os.path.join(REPO, "deploy")

for p in (FIXED_GAIT, WEBUI, REPO, DEPLOY):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA = os.path.join(WEBUI, "data")
TRAJ_DIR = os.path.join(DATA, "trajectories")
WORKSPACE_DIR = os.path.join(DATA, "workspaces")
MEASURE_DIR = os.path.join(DATA, "measurements")      # high-rate system-ID capture runs (.npz+.json)
THERMAL_DIR = os.path.join(DATA, "thermal")           # thermal burst logs (thermalstore keeps the index)
BLACKBOX_DIR = os.path.join(DATA, "blackbox")          # flight recorder: Tier A/B segments + events.jsonl
IDENT_DIR = os.path.join(DATA, "identification")       # estimator outputs (identified_params.json)
POLICY_DIR = os.path.join(DATA, "policies")            # exported policy bundles (.npz, see deploy/bundle.py)
POLICYRUN_DIR = os.path.join(DATA, "policyruns")       # run_policy.py logs (also its --out default)
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

for d in (DATA, TRAJ_DIR, WORKSPACE_DIR, MEASURE_DIR, IDENT_DIR, BLACKBOX_DIR,
          THERMAL_DIR, POLICY_DIR, POLICYRUN_DIR):
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


# ---- where a policy bundle may live ----
# TWO directories, because there are two ways one arrives and they used to disagree:
# export_policy.py writes to robot/deploy/bundles/ (it is part of the deploy package and knows
# nothing about the web UI), while an upload through the panel lands in data/policies/. A panel
# that reads only one of them answers "no bundles" for a bundle sitting right there, which is the
# least debuggable failure a file picker has. Uploads still WRITE to POLICY_DIR only.
BUNDLE_DIR = os.path.join(DEPLOY, "bundles")       # where export_policy.py writes


def policy_dirs():
    """Read at CALL time, not import time, so a test can redirect either directory."""
    return (POLICY_DIR, BUNDLE_DIR)


def find_policy_bundle(fname):
    """Resolve a client-supplied bundle NAME to a path, or None. basename first, so a name is a
    name and the two directories above are the whole search space."""
    fname = os.path.basename(str(fname))
    if not fname.endswith(".npz") or not fname[:-4]:
        return None
    for d in policy_dirs():
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None


def list_policy_bundles():
    """[(filename, path, where)] across both directories; POLICY_DIR wins a name collision."""
    out, seen = [], set()
    for d in policy_dirs():
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".npz") and f not in seen:
                seen.add(f)
                out.append((f, os.path.join(d, f),
                            "data/policies" if d == POLICY_DIR else "deploy/bundles"))
    return out


def motor_index(name):
    return MOTOR_NAMES.index(name)


def split_name(name):
    side, role = name.split(".")
    return side, role
