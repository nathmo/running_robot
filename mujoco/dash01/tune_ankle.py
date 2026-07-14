"""Find the ankle-spring neutral angle (springref) that makes each foot sit FLAT at the
standing pose. The ankle is a passive preloaded spring (0.5 Nm/deg, 2.27 Nm preload); its
neutral angle is a free design choice, which we pick here so the long thin foot lies flat.

Prints the best springref per side; bake into build_model.  (Run after build_model.)
Run:  .venv/Scripts/python.exe mujoco/spiderbot/tune_ankle.py
"""
import itertools
import numpy as np
import mujoco
from build_model import INIT_CTRL

model = mujoco.MjModel.from_xml_path("mujoco/spiderbot/spiderbot.xml")
np.set_printoptions(precision=3, suppress=True)
C = np.array(list(itertools.product([-1, 1], repeat=3)))
ANK = {s: model.jnt_qposadr[mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, f"Leg{'Left' if s=='L' else 'Right'}NCS-v1_Révolution-{9 if s=='L' else 10}")]
    for s in "LR"}
FOOT = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_col") for s in "LR"}


def foot_tilt(data, s):
    """Angle (deg) of the foot box long axis from horizontal: 0 = flat."""
    R = data.geom_xmat[FOOT[s]].reshape(3, 3)
    return np.degrees(np.arcsin(abs(R[:, 2][2])))


def settle(spring_L, spring_R):
    data = mujoco.MjData(model)
    model.qpos_spring[ANK["L"]] = spring_L
    model.qpos_spring[ANK["R"]] = spring_R
    mujoco.mj_resetDataKeyframe(model, data, 0)
    g = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    data.qpos[:3] = [0, 0, 1.5]
    bq = data.qpos[:7].copy()
    data.ctrl[:] = INIT_CTRL
    for _ in range(1500):
        mujoco.mj_step(model, data)
        data.qpos[:7] = bq
        data.qvel[:6] = 0
    mujoco.mj_forward(model, data)
    model.opt.gravity[:] = g
    return data


print("sweeping ankle springref (same on both sides) -> foot tilt:")
for sr in np.linspace(-0.8, 0.8, 17):
    d = settle(sr, sr)
    print(f"  springref={sr:+.2f}  tilt L={foot_tilt(d,'L'):5.1f}  R={foot_tilt(d,'R'):5.1f} deg")

# refine each side independently for minimum tilt
best = {}
for s in "LR":
    grid = np.linspace(-0.8, 0.8, 81)
    tilts = []
    for sr in grid:
        d = settle(sr if s == "L" else 0, sr if s == "R" else 0)
        tilts.append(foot_tilt(d, s))
    best[s] = grid[int(np.argmin(tilts))]
    print(f"best springref[{s}] = {best[s]:+.3f}  (tilt {min(tilts):.1f} deg)")
print(f"\n=> set ankle springref:  L={best['L']:+.3f}  R={best['R']:+.3f}")
