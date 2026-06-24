"""Re-tune the ankle spring neutral angle for a POINT-TOE stance.

With only the toe sphere touching the floor, the long foot must rest toe-DOWN (heel up) so the
rest of the foot clears the ground; otherwise tiny tilts make the heel traverse the floor. We
sweep the ankle springref, settle the standing pose resting on the toe spheres, and measure the
clearance of the 'rest of the foot' (foot-mesh vertices away from the toe). Pick the springref
that lifts the heel well clear, then bake it into build_model.

Run:  .venv/Scripts/python.exe mujoco/spiderbot/tune_foot_posture.py
"""
import numpy as np
import mujoco
from PIL import Image
from build_model import INIT_CTRL

model = mujoco.MjModel.from_xml_path("mujoco/spiderbot/spiderbot.xml")
np.set_printoptions(precision=3, suppress=True)
ANK = {s: model.jnt_qposadr[mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, f"Leg{'Left' if s=='L' else 'Right'}NCS-v1_Révolution-{9 if s=='L' else 10}")]
    for s in "LR"}
FOOT_G = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_col") for s in "LR"}
FOOT_MESH = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"Foot{'Left' if s=='L' else 'Right'}NCS-v1_geom") for s in "LR"}


def mesh_local(gid):
    mid = model.geom_dataid[gid]
    a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    return model.mesh_vert[a:a + n].reshape(-1, 3)


MV = {s: mesh_local(FOOT_MESH[s]) for s in "LR"}


def settle_stance(spring):
    data = mujoco.MjData(model)
    model.qpos_spring[ANK["L"]] = spring
    model.qpos_spring[ANK["R"]] = -spring
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
    # drop torso so the lowest toe sphere sits on z=0
    zmin = min(data.geom_xpos[FOOT_G[s]][2] - model.geom_rbound[FOOT_G[s]] for s in "LR")
    data.qpos[2] = 1.5 - zmin
    mujoco.mj_forward(model, data)
    return data


def rest_clearance(data):
    """Min height of foot-mesh points that are NOT near the toe tip (i.e., the 'rest of the foot')."""
    worst = 1e9
    for s in "LR":
        g = FOOT_MESH[s]
        R = data.geom_xmat[g].reshape(3, 3)
        w = data.geom_xpos[g] + MV[s] @ R.T
        toe = data.geom_xpos[FOOT_G[s]]
        far = np.linalg.norm(w - toe, axis=1) > 0.06   # exclude the toe-ball region
        worst = min(worst, w[far, 2].min())
    return worst


print("springref |  rest-of-foot clearance (m, want > ~0.05)")
results = []
for v in np.linspace(-1.0, 0.3, 14):
    d = settle_stance(v)
    c = rest_clearance(d)
    results.append((v, c))
    print(f"   {v:+.2f}   |   {c:+.3f}")

best = max(results, key=lambda r: r[1])
print(f"\nbest springref magnitude = {best[0]:.2f}  (clearance {best[1]:+.3f} m)  -> L=+{best[0]:.2f} R=-{best[0]:.2f}")
# render best
d = settle_stance(best[0])
r = mujoco.Renderer(model, 360, 640)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, cam)
cam.distance, cam.elevation, cam.azimuth = 1.6, -5, 90; cam.lookat[:] = [0, 0, 0.25]
r.update_scene(d, cam); Image.fromarray(r.render()).save("mujoco/spiderbot/_posture_best.png"); r.close()
print("rendered _posture_best.png")
