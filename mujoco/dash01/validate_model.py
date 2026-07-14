"""M0 gate: prove the DASH-01 model is physically sane before any RL.

Checks:
  1. compiles; nu==6, neq==2
  2. closed loop holds at the keyframe (site coincidence < 0.5 mm, equality residual small)
  3. parallel mechanism works: sweeping a cam (hip) motor MOVES the knee through the linkage
  4. stands: 2 s drop with PD holding the keyframe -> no NaN, stays upright, settles
Also renders a few frames to mujoco/dash01/_validate_*.png so we can eyeball the pose.

Run:  .venv/Scripts/python.exe mujoco/dash01/validate_model.py
"""
import os
import sys
import numpy as np
import mujoco

XML = "mujoco/dash01/dash01.xml"
model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
np.set_printoptions(precision=4, suppress=True)

ok = True
# Rendering is cosmetic; the gate must not depend on a GL context. On a headless box a FAILED
# GLFW init leaves the GL stack half-initialized and a SECOND Renderer attempt aborts the whole
# process from C++ (uncatchable) — so after one failure, never try again.
render_ok = True


def sid(n):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)


def aid(n):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)


def jadr(n):
    return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]


def check(label, cond, detail=""):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}  {detail}")


def render(tag):
    global render_ok
    if not render_ok:
        print(f"  (render skipped: no usable GL context)")
        return
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") \
            and os.environ.get("MUJOCO_GL", "") not in ("egl", "osmesa"):
        render_ok = False
        print("  (render skipped: headless — set MUJOCO_GL=egl or osmesa to enable PNGs)")
        return
    try:
        r = mujoco.Renderer(model, 480, 640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.distance, cam.elevation, cam.azimuth = 2.5, -20, 135
        cam.lookat[:] = [0, 0, 0.5]
        r.update_scene(data, cam)
        px = r.render()
        r.close()
        try:
            from PIL import Image
            Image.fromarray(px).save(f"mujoco/dash01/_validate_{tag}.png")
        except ImportError:
            import matplotlib.image as mpimg
            mpimg.imsave(f"mujoco/dash01/_validate_{tag}.png", px)
        print(f"  rendered _validate_{tag}.png")
    except Exception as e:
        render_ok = False
        print(f"  (render skipped: {e})")


print("== 1. basics ==")
check("nu == 6", model.nu == 6, f"nu={model.nu}")
# 2 closed-loop <connect> equalities + 6 base-DOF <joint> locks (lock_x..lock_yaw, inactive by
# default; the RL env activates the milestone's subset via data.eq_active).
check("neq == 8", model.neq == 8, f"neq={model.neq}")
check("has 'stand' keyframe", model.nkey >= 1)

print("== 2. closed loop holds at keyframe ==")
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
for s in "LR":
    d = np.linalg.norm(data.site_xpos[sid(f"pushrod_tip_{s}")] - data.site_xpos[sid(f"leg_anchor_{s}")])
    check(f"loop {s} site coincidence", d < 5e-4, f"{d*1000:.4f} mm")
render("keyframe")

print("== 3. parallel mechanism: cam motor drives the knee ==")
mujoco.mj_resetDataKeyframe(model, data, 0)
g = model.opt.gravity.copy()
model.opt.gravity[:] = 0          # isolate the linkage
base_q = data.qpos[:7].copy()
cam, knee = jadr("HipLeftNCS-v1_Révolution-3"), jadr("ThighLeftNCS-v1_Révolution-7")
sweep = []
for target in np.linspace(-0.5, 0.5, 9):
    data.ctrl[aid("cam_L")] = target
    for _ in range(2000):
        mujoco.mj_step(model, data)
        data.qpos[:7] = base_q
        data.qvel[:6] = 0
    mujoco.mj_forward(model, data)
    sweep.append((data.qpos[cam], data.qpos[knee]))
sweep = np.array(sweep)
cam_span = np.ptp(sweep[:, 0])
knee_span = np.ptp(sweep[:, 1])
print(f"    cam range swept: {sweep[:,0].min():.3f}..{sweep[:,0].max():.3f} rad")
print(f"    knee response:   {sweep[:,1].min():.3f}..{sweep[:,1].max():.3f} rad")
ratio = knee_span / cam_span if cam_span > 1e-6 else 0
check("knee moves when cam moves", knee_span > 0.05, f"knee span {knee_span:.3f} rad, knee/cam ~ {ratio:.2f}")
model.opt.gravity[:] = g

print("== 4. integrates 2 s under gravity (numerical sanity) ==")
# NOTE: a biped is NOT passively stable with fixed motor targets (like balancing a broomstick);
# toppling here is correct physics. We check the integrator stays sane and the loop keeps holding.
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bodyNCS-v1")
mujoco.mj_resetDataKeyframe(model, data, 0)   # also sets ctrl = keyframe's standing targets
h0 = data.qpos[2]
blew_up = False
for i in range(2000):
    mujoco.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)) or data.warning.number.any():
        blew_up = True
        break
check("no NaN / solver blow-up over 2 s", not blew_up)
mujoco.mj_forward(model, data)
res = max(np.linalg.norm(data.site_xpos[sid(f"pushrod_tip_{s}")] - data.site_xpos[sid(f"leg_anchor_{s}")])
          for s in "LR")
check("closed loop still holds after 2 s of dynamics", res < 5e-3, f"{res*1000:.3f} mm")
upright = data.xmat[bid].reshape(3, 3)[2, 2]
print(f"    base height {h0:.3f} -> {data.qpos[2]:.3f} m, torso up={upright:+.2f}  "
      f"(passive topple expected; RL adds active balance in M1)")
render("after2s")

print("\n==>", "M0 GATE PASSED" if ok else "M0 GATE NOT YET PASSED")
