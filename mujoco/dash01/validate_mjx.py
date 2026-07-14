"""MJX (GPU) physics parity gate -- Phase 0 of the MJX training migration.

This model's hardest features for MJX are the closed-loop parallel knee
(<equality><connect>, two sites per leg) and condim=6 contact with an
EXPLICIT elliptic cone (<option cone="elliptic">). Both have had rocky MJX
history on other robots. This script is the go/no-go gate: it runs the exact
same control sequence through CPU MuJoCo (source of truth) and MJX (GPU) from
the same starting state and checks MJX doesn't silently misbehave.

What it checks, in order of how badly a failure here would break training:
  1. Loads onto device at all (mjx.put_model raises loudly on truly
     unsupported fields -- a fast, clear no-go signal if it happens).
  2. Passive ankle spring (stiffness/springref) torque matches CPU at the
     stand keyframe -- isolated, no integration noise (single mj_forward /
     mjx.forward call, not a rollout).
  3. Over a driven rollout: no NaN/blow-up, the closed-loop connect
     constraint (pushrod_tip <-> leg_anchor) stays satisfied on MJX the same
     way it does on CPU, and short-horizon trajectories track before
     float32/chaos divergence takes over (expected after ~O(0.3-1s), not a
     failure by itself -- growing SLOWLY is fine, blowing up immediately is not).

Run on the training server (CUDA-enabled JAX):
    .venv/bin/python mujoco/dash01/validate_mjx.py
"""
import numpy as np
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp

MODEL_PATH = "mujoco/dash01/dash01.xml"
SEED = 0
CONTROL_DECIMATION = 20     # matches rl/config.py: sim is 1kHz, control is 50Hz
N_CONTROL_STEPS = 250       # 5 s of sim time -- long enough to fall over and
                            # stress contact/constraint solving hard, short
                            # enough that "diverged from CPU" stays meaningful
CTRL_STEP_STD = 0.03        # rad, random-walk step on the PD targets


def name2id(model, kind, name):
    return mujoco.mj_name2id(model, kind, name)


def loop_dists(site_xpos, sites):
    (s1L, s2L), (s1R, s2R) = sites
    dL = float(np.linalg.norm(site_xpos[s1L] - site_xpos[s2L]))
    dR = float(np.linalg.norm(site_xpos[s1R] - site_xpos[s2R]))
    return dL, dR


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    key_id = name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    nominal_ctrl = model.key_ctrl[key_id].copy()
    data.ctrl[:] = nominal_ctrl
    mujoco.mj_forward(model, data)

    print(f"[model] nq={model.nq} nv={model.nv} nu={model.nu} neq={model.neq}  "
          f"cone={model.opt.cone} integrator={model.opt.integrator} "
          f"solver={model.opt.solver} timestep={model.opt.timestep}")

    sites = ((name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pushrod_tip_L"),
              name2id(model, mujoco.mjtObj.mjOBJ_SITE, "leg_anchor_L")),
             (name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pushrod_tip_R"),
              name2id(model, mujoco.mjtObj.mjOBJ_SITE, "leg_anchor_R")))
    ankle_L_jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "LegLeftNCS-v1_Révolution-9")
    ankle_R_jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "LegRightNCS-v1_Révolution-10")
    ankle_dofs = [model.jnt_dofadr[ankle_L_jid], model.jnt_dofadr[ankle_R_jid]]
    foot_gids = [name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_col") for s in "LR"]
    toe_r = float(model.geom_size[foot_gids[0]][0])

    print("\n== step 1: load onto device ==")
    mjx_model = mjx.put_model(model)
    mjx_data0 = mjx.put_data(model, data)
    print("  [PASS] mjx.put_model / mjx.put_data did not raise")

    print("\n== step 2: passive ankle spring torque @ stand keyframe (no integration) ==")
    d_fwd = mjx.forward(mjx_model, mjx_data0)
    cpu_qfrc = data.qfrc_passive[ankle_dofs]
    mjx_qfrc = np.asarray(d_fwd.qfrc_passive)[ankle_dofs]
    spring_err = np.abs(cpu_qfrc - mjx_qfrc)
    print(f"  CPU qfrc_passive[ankle] = {cpu_qfrc}")
    print(f"  MJX qfrc_passive[ankle] = {mjx_qfrc}")
    same_sign = np.all(np.sign(cpu_qfrc) == np.sign(mjx_qfrc)) or np.allclose(cpu_qfrc, 0, atol=1e-6)
    print(f"  {'[PASS]' if same_sign and np.all(spring_err < 0.5) else '[FAIL]'} "
          f"max abs error = {spring_err.max():.4f} Nm, same sign = {same_sign}")

    print(f"\n== step 3: {N_CONTROL_STEPS}x{CONTROL_DECIMATION} = "
          f"{N_CONTROL_STEPS * CONTROL_DECIMATION} sim steps "
          f"({N_CONTROL_STEPS * CONTROL_DECIMATION * model.opt.timestep:.1f}s), driven rollout ==")
    rng = np.random.default_rng(SEED)
    ctrl_lo, ctrl_hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    targets = np.zeros((N_CONTROL_STEPS, model.nu), np.float32)
    t = nominal_ctrl.copy()
    for i in range(N_CONTROL_STEPS):
        t = np.clip(t + rng.normal(0, CTRL_STEP_STD, model.nu), ctrl_lo, ctrl_hi)
        targets[i] = t

    # ---- CPU rollout (source of truth) ----
    cpu_h, cpu_loopL, cpu_loopR = [], [], []
    cpu_nan_at = None
    for i in range(N_CONTROL_STEPS):
        data.ctrl[:] = targets[i]
        for _ in range(CONTROL_DECIMATION):
            mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            cpu_nan_at = i
            break
        dL, dR = loop_dists(data.site_xpos, sites)
        cpu_h.append(float(data.qpos[2])); cpu_loopL.append(dL); cpu_loopR.append(dR)

    # ---- MJX rollout (GPU, jitted single control-step of CONTROL_DECIMATION sim steps) ----
    def control_step(d, ctrl):
        def body(d, _):
            d = d.replace(ctrl=ctrl)
            return mjx.step(mjx_model, d), None
        d, _ = jax.lax.scan(body, d, None, length=CONTROL_DECIMATION)
        return d

    jit_control_step = jax.jit(control_step)
    d = mjx_data0
    mjx_h, mjx_loopL, mjx_loopR = [], [], []
    mjx_nan_at = None
    for i in range(N_CONTROL_STEPS):
        d = jit_control_step(d, jnp.asarray(targets[i]))
        qpos = np.asarray(d.qpos)
        if not np.all(np.isfinite(qpos)):
            mjx_nan_at = i
            break
        site_xpos = np.asarray(d.site_xpos)
        dL, dR = loop_dists(site_xpos, sites)
        mjx_h.append(float(qpos[2])); mjx_loopL.append(dL); mjx_loopR.append(dR)

    n = min(len(cpu_h), len(mjx_h))
    cpu_h, mjx_h = np.array(cpu_h[:n]), np.array(mjx_h[:n])
    h_err = np.abs(cpu_h - mjx_h)
    diverge_at = next((i for i, e in enumerate(h_err) if e > 0.05), None)

    print(f"  CPU: {len(cpu_h)}/{N_CONTROL_STEPS} control steps completed, "
          f"nan_at={cpu_nan_at}, loop_L max={max(cpu_loopL, default=0):.5f} m, "
          f"loop_R max={max(cpu_loopR, default=0):.5f} m")
    print(f"  MJX: {len(mjx_h)}/{N_CONTROL_STEPS} control steps completed, "
          f"nan_at={mjx_nan_at}, loop_L max={max(mjx_loopL, default=0):.5f} m, "
          f"loop_R max={max(mjx_loopR, default=0):.5f} m")
    print(f"  base-height |CPU-MJX| diverges past 5cm at control step "
          f"{diverge_at if diverge_at is not None else '(never, within ' + str(n) + ' steps)'}"
          f" (expected to diverge eventually from float32/chaos -- concerning only if immediate)")

    loop_ok = max(mjx_loopL, default=0) < 0.01 and max(mjx_loopR, default=0) < 0.01
    no_nan = mjx_nan_at is None
    late_divergence = diverge_at is None or diverge_at > 10   # >0.2s before 5cm gap

    print("\n== VERDICT ==")
    checks = {
        "loads_on_device": True,
        "ankle_spring_matches": bool(same_sign and np.all(spring_err < 0.5)),
        "no_nan_blowup": no_nan,
        "connect_constraint_holds (<1cm)": loop_ok,
        "trajectory_not_immediately_divergent": late_divergence,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("PASS" if all(checks.values()) else "FAIL")


if __name__ == "__main__":
    main()
