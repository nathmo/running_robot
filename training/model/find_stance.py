"""Search for a stable standing pose (keyframe) for DASH-01.

The knee is passively driven by the cam through the closed loop, and the L/R sagittal joint
axes are mirrored, so a symmetric stance uses ctrl = [hr, c, t, -hr, -c, -t]. For each candidate
(hip_roll, cam, thigh) target we:
  1. reach the pose in the air (gravity off, base frozen),
  2. compute the torso height that just sets the feet on the ground,
  3. drop-test under gravity and score by final height + uprightness + stillness.
Prints the best pose's qpos/ctrl (paste into build_model's keyframe) and renders it.

Run:  .venv/Scripts/python.exe mujoco/dash01/find_stance.py
"""
import itertools
import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("mujoco/dash01/dash01.xml")
data = mujoco.MjData(model)
np.set_printoptions(precision=3, suppress=True)

FOOT_G = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_col") for s in "LR"]
_CORNERS = np.array(list(itertools.product([-1, 1], repeat=3)))


def feet_min_z():
    z = []
    for g in FOOT_G:
        c = data.geom_xpos[g] + (_CORNERS * model.geom_size[g]) @ data.geom_xmat[g].reshape(3, 3).T
        z.append(c[:, 2].min())
    return min(z)


def reach_pose(ctrl, steps=1500):
    """Drive to the target pose in the air (no gravity, frozen base). Returns torso height
    that puts the lowest foot point at z=0."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    g = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    data.qpos[2] = 1.5
    base_q = data.qpos[:6].copy()          # base is 6 scalar joints (x,y,z,roll,pitch,yaw)
    data.ctrl[:] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)
        data.qpos[:6] = base_q
        data.qvel[:6] = 0
    mujoco.mj_forward(model, data)
    model.opt.gravity[:] = g
    pose = data.qpos.copy()
    pose[2] = 1.5 - feet_min_z()           # torso height so feet touch ground
    return pose


def drop_test(pose, ctrl, steps=2000):
    data.qpos[:] = pose
    data.qpos[2] += 0.01
    data.qvel[:] = 0
    data.ctrl[:] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            return dict(h=-1, up=-1, v=1e9)
    up = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bodyNCS-v1")].reshape(3, 3)[2, 2]
    return dict(h=data.qpos[2], up=up, v=np.linalg.norm(data.qvel), end=data.qpos.copy())


best = None
for hr, c, t in itertools.product([-0.3, 0.0, 0.3], [-0.4, 0.0, 0.4], [-0.4, 0.0, 0.4]):
    ctrl = np.array([hr, c, t, -hr, -c, -t])
    pose = reach_pose(ctrl)
    r = drop_test(pose, ctrl)
    score = r["h"] if (r["up"] > 0.7 and r["v"] < 5) else -1
    tag = "  <-- candidate" if score > 0 else ""
    print(f"hr={hr:+.1f} c={c:+.1f} t={t:+.1f} | reach_h={pose[2]:.2f} "
          f"end_h={r['h']:.2f} up={r['up']:+.2f} |v|={r['v']:.2f} score={score:.2f}{tag}")
    if score > 0 and (best is None or score > best["score"]):
        best = dict(score=score, hr=hr, c=c, t=t, ctrl=ctrl, pose=pose, r=r)

print("\n==================")
if best is None:
    print("No stable stance found in this grid. Need finer search / ankle tuning.")
else:
    print(f"BEST: hr={best['hr']} c={best['c']} t={best['t']}  end_h={best['r']['h']:.3f} "
          f"up={best['r']['up']:.2f}")
    print(f"ctrl = {best['ctrl']}")
    print(f"keyframe qpos (after settle):\n{best['r']['end']}")
    # render the settled best
    try:
        from PIL import Image
        data.qpos[:] = best["r"]["end"]
        data.qvel[:] = 0
        data.ctrl[:] = best["ctrl"]
        mujoco.mj_forward(model, data)
        r = mujoco.Renderer(model, 480, 640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.distance, cam.elevation, cam.azimuth = 2.2, -15, 120
        cam.lookat[:] = [0, 0, 0.4]
        r.update_scene(data, cam)
        Image.fromarray(r.render()).save("mujoco/dash01/_stance_best.png")
        r.close()
        print("rendered _stance_best.png")
    except Exception as e:
        print(f"(render skipped: {e})")
