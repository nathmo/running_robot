"""Measure DASH-01's feasible RIDE-HEIGHT band and build a ride-height -> leg-posture LUT.

M1 of the RL curriculum rails the base (X free, everything else locked) at a height that is
RANDOMIZED per episode so the policy learns to adapt to any height. The linkage is near full leg
extension at the 1.0235 m stand and can only crouch a few cm, so the feasible vertical band is
NARROW and one-sided -- this tool measures it instead of guessing, and records, for each height, the
leg posture (12 hinge qpos + 6 PD ctrl) that seats the feet on the floor so M1's reset can start in a
valid on-floor stance (no spurious t=0 floor-violation).

Method, per candidate symmetric leg posture ctrl = [hr, cam, thigh, -hr, -cam, -thigh]:
  1. AIR-settle the closed loop (gravity off, base frozen) -> consistent hinge pose.
  2. Compute the base height H that puts the lowest toe-sphere bottom at z=0 (feet on floor).
  3. WELD-settle: lock all 6 base DOFs (base_z target = H), gravity on, PD-hold ctrl -> the true
     LOADED hinge qpos (ankle-spring + PD sag) and a stability/validity check.
Valid samples are binned by H; per bin we keep the pose whose toe is most under the base (the most
natural stance). Prints [H_lo, H_hi] and writes mujoco/dash01/ride_height_lut.npz.

Run:  .venv/Scripts/python.exe mujoco/dash01/measure_ride_band.py
"""
import numpy as np
import mujoco

MODEL = "mujoco/dash01/dash01.xml"
OUT = "mujoco/dash01/ride_height_lut.npz"

model = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(model)

BASE = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bodyNCS-v1")
FOOT_G = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_col") for s in "LR"]
LOCK_IDS = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"lock_{n}")
            for n in ("x", "y", "z", "roll", "pitch", "yaw")}
LOCK_Z = LOCK_IDS["z"]
# leg-hinge qpos live at qpos[6:] (base is 6 scalar joints); actuated joints -> qpos addresses
ACT_QADR = np.array([model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in range(model.nu)])
JNT_RANGE = model.jnt_range.copy()


def toe_bottoms():
    """z of each toe sphere's bottom (world)."""
    return np.array([data.geom_xpos[g][2] - model.geom_rbound[g] for g in FOOT_G])


def loop_residual():
    """max distance between the two connected loop sites (m); ~0 => loop assemblable/closed."""
    res = 0.0
    for s in "LR":
        p1 = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"pushrod_tip_{s}")]
        p2 = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"leg_anchor_{s}")]
        res = max(res, float(np.linalg.norm(p1 - p2)))
    return res


def hinges_in_range(q):
    """all limited leg hinges within their range (small margin)."""
    for j in range(6, model.njnt):
        if model.jnt_limited[j]:
            lo, hi = JNT_RANGE[j]
            v = q[model.jnt_qposadr[j]]
            if v < lo - 1e-3 or v > hi + 1e-3:
                return False
    return True


def air_settle(ctrl, steps=1500):
    """Settle the loop in the air (gravity off, base frozen), then return the base height that puts
    the feet on the floor and the settled qpos. Returns (H, qpos) or (None, None) if inconsistent."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    g = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    data.qpos[0:3] = [0, 0, 1.5]
    data.qpos[3:6] = 0
    base_q = data.qpos[:6].copy()
    data.ctrl[:] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)
        data.qpos[:6] = base_q
        data.qvel[:6] = 0
    mujoco.mj_forward(model, data)
    model.opt.gravity[:] = g
    if not np.all(np.isfinite(data.qpos)) or loop_residual() > 0.01:
        return None, None
    H = 1.5 - toe_bottoms().min()          # base height so lowest toe bottom sits at z=0
    q = data.qpos.copy()
    q[2] = H
    return H, q


def weld_settle(H, q_air, ctrl, steps=1200):
    """Lock all 6 base DOFs (base_z target = H), settle under gravity holding ctrl. Returns the
    loaded qpos and a validity dict, or None if it diverged / feet left the floor / tipped."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:] = q_air
    data.qpos[0:3] = [0, 0, H]
    data.qpos[3:6] = 0
    data.qvel[:] = 0
    for nm, eid in LOCK_IDS.items():
        data.eq_active[eid] = 1
    model.eq_data[LOCK_Z, 0] = H
    data.ctrl[:] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)):
        return None
    toes = toe_bottoms()
    up = data.xmat[BASE].reshape(3, 3)[2, 2]
    # toe fore/aft offset from the base origin (x), averaged -- prefer the most-under-base pose
    toe_x = np.mean([data.geom_xpos[g][0] for g in FOOT_G]) - data.qpos[0]
    ok = (loop_residual() < 0.01 and hinges_in_range(data.qpos)
          and abs(toes.min()) < 0.01 and toes.max() < 0.03 and up > 0.9)
    return dict(ok=ok, qpos=data.qpos.copy(), toe_x=abs(float(toe_x)),
                res=loop_residual(), up=float(up), toe_min=float(toes.min()),
                torque=data.actuator_force[:model.nu].copy())   # loaded holding torque at this height


def main():
    samples = []   # (H, hinges[12], ctrl[6], toe_x, torque[6])
    cams = np.linspace(-0.6, 0.6, 13)
    thighs = np.linspace(-0.2, 0.8, 21)
    hr = 0.0
    n_tried = n_valid = 0
    for c in cams:
        for t in thighs:
            n_tried += 1
            ctrl = np.array([hr, c, t, -hr, -c, -t])
            H, q_air = air_settle(ctrl)
            if H is None or not (0.80 < H < 1.05):
                continue
            r = weld_settle(H, q_air, ctrl)
            if r is None or not r["ok"]:
                continue
            n_valid += 1
            samples.append((H, r["qpos"][6:].copy(), ctrl.copy(), r["toe_x"], r["torque"]))

    if not samples:
        print("No valid ride-height stances found -- widen the ctrl sweep.")
        return
    Hs = np.array([s[0] for s in samples])
    H_lo, H_hi = float(Hs.min()), float(Hs.max())
    print(f"tried={n_tried} valid={n_valid}")
    print(f"feasible ride-height band: [{H_lo:.4f}, {H_hi:.4f}] m  (span {H_hi-H_lo:.3f} m)")

    # bin by height; keep the most-under-base pose per bin -> a clean monotone LUT
    grid = np.round(np.arange(np.ceil(H_lo*100)/100, np.floor(H_hi*100)/100 + 1e-9, 0.005), 4)
    lut_H, lut_hinges, lut_ctrl, lut_torque = [], [], [], []
    for hb in grid:
        near = [s for s in samples if abs(s[0] - hb) <= 0.0075]
        if not near:
            continue
        best = min(near, key=lambda s: s[3])       # smallest toe fore/aft offset
        lut_H.append(hb); lut_hinges.append(best[1]); lut_ctrl.append(best[2])
        lut_torque.append(best[4])
    lut_H = np.array(lut_H)
    order = np.argsort(lut_H)
    lut_H = lut_H[order]
    lut_hinges = np.array(lut_hinges)[order]
    lut_ctrl = np.array(lut_ctrl)[order]
    lut_torque = np.array(lut_torque)[order]
    np.savez(OUT, H=lut_H, hinges=lut_hinges, ctrl=lut_ctrl, torque=lut_torque)
    print(f"wrote {OUT}: {len(lut_H)} entries, H in [{lut_H.min():.4f}, {lut_H.max():.4f}]")
    print(f"holding |torque| range over band: cam/thigh max "
          f"{np.abs(lut_torque[:, [1, 2, 4, 5]]).max():.1f} Nm")
    # suggested range = interior minus ~1 cm margin each side (PD can hold statically there)
    r_lo, r_hi = round(lut_H.min() + 0.01, 3), round(lut_H.max() - 0.01, 3)
    print(f"suggested z_rail_range (interior, 1 cm margin): ({r_lo}, {r_hi})")


if __name__ == "__main__":
    main()
