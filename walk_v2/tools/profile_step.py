"""Time the bare MJX physics of the v2 plant under different solver settings.

The env's step time barely scales with the batch (757 ms at 256 envs, 1.6 s at 4096 on a V100):
a fixed per-step cost dominates. This isolates the physics: one control tick = 10 substeps of
mjx.step under a joint PD that holds the keyframe pose (real stance loads), for several
`<option>` variants, and reports ms per tick, env substeps/s, and how far each variant's
trajectory drifts from the XML's own settings over --check-ticks ticks.

    python walk_v2/tools/profile_step.py --n-envs 8 --variants xml newton8 cg6
    python walk_v2/tools/profile_step.py --n-envs 256 2048 4096 --json walk_v2/results/profile_step.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from config import get_config  # noqa: E402

XML = os.path.join(HERE, "..", "model", "dash01_v2_free.xml")
CFG = get_config("v2_s2_free")
HOLD_PITCH = True
Z_Q = [2]
SUBSTEPS = 10

VARIANTS = {
    "xml": {},
    "newton16": dict(iterations=16, ls_iterations=16),
    "newton16_ls8": dict(iterations=16, ls_iterations=8),
    "newton32_ls8": dict(iterations=32, ls_iterations=8),
    "newton100_ls8": dict(iterations=100, ls_iterations=8),
    "newton1_ls4": dict(iterations=1, ls_iterations=4),
    "newton2_ls4": dict(iterations=2, ls_iterations=4),
    "newton4_ls4": dict(iterations=4, ls_iterations=4),
    "newton8": dict(iterations=8, ls_iterations=8),
    "newton4": dict(iterations=4, ls_iterations=4),
    "cg6": dict(solver=int(mujoco.mjtSolver.mjSOL_CG), iterations=6, ls_iterations=6),
    "newton8_pyr": dict(iterations=8, ls_iterations=8, cone=int(mujoco.mjtCone.mjCONE_PYRAMIDAL)),
    "newton8_euler": dict(iterations=8, ls_iterations=8,
                          integrator=int(mujoco.mjtIntegrator.mjINT_EULER)),
    "newton8_condim3": dict(iterations=8, ls_iterations=8, condim=3),
}


def load(variant: dict, xml: str) -> mujoco.MjModel:
    m = mujoco.MjModel.from_xml_path(xml)
    for k, v in variant.items():
        if k == "condim":
            m.geom_condim[:] = v
        else:
            setattr(m.opt, k, v)
    return m


def make_step(m: mujoco.MjModel, batched_timestep: bool, batched_fields: bool = False):
    mx = mjx.put_model(m)
    jid = m.actuator_trnid[:, 0]
    qadr = jnp.asarray(m.jnt_qposadr[jid])
    dadr = jnp.asarray(m.jnt_dofadr[jid])
    # the real drive holding the stance: nominal targets (<numeric nominal_ctrl>) + config gains
    nid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "nominal_ctrl")
    adr = int(m.numeric_adr[nid])
    q_key = jnp.asarray(m.numeric_data[adr:adr + 6])
    kp = jnp.asarray(CFG.drive_kp, jnp.float32)
    kd = jnp.asarray(CFG.drive_kd, jnp.float32)
    peak = jnp.asarray(m.actuator_forcerange[:, 1])
    jp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_pitch")
    jz = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_z")
    pitch_q, pitch_d = (int(m.jnt_qposadr[jp]), int(m.jnt_dofadr[jp])) if jp >= 0 else (-1, -1)
    z_q = int(m.jnt_qposadr[jz]) if jz >= 0 else 2

    fields = ("body_mass", "body_inertia", "body_ipos", "geom_friction", "dof_damping", "jnt_stiffness",
              "opt.gravity", "site_pos")

    def tick(d, ts):
        mx_i = mx.tree_replace({"opt.timestep": ts}) if batched_timestep else mx
        if batched_fields:      # per-env copies of the DR'd fields, as env.py's model_with does
            scale = 1.0 + 1e-3 * (ts / m.opt.timestep - 1.0)
            rep = {}
            for f in fields:
                v = mx.opt.gravity if f == "opt.gravity" else getattr(mx, f)
                rep[f] = v * scale
            mx_i = mx_i.tree_replace(rep)

        def body(d, _):
            tau = kp * (q_key - d.qpos[qadr]) - kd * d.qvel[dadr]
            d = d.replace(ctrl=jnp.clip(tau, -peak, peak))
            if HOLD_PITCH and pitch_d >= 0:      # the base-pitch hold: a stable stance for the drift check
                hold = -2000.0 * d.qpos[pitch_q] - 100.0 * d.qvel[pitch_d]
                d = d.replace(qfrc_applied=jnp.zeros(m.nv).at[pitch_d].set(hold))
            return mjx.step(mx_i, d), None

        d, _ = jax.lax.scan(body, d, None, length=SUBSTEPS)
        return d

    return mx, jax.jit(jax.vmap(tick, in_axes=(0, 0)))


def init_data(m: mujoco.MjModel, mx, n: int):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d0 = mjx.put_data(m, d)
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), d0)


def run(name: str, n: int, args, ref=None):
    m = load(VARIANTS[name], args.xml)
    mx, step = make_step(m, args.batched_timestep, args.batched_fields)
    Z_Q[0] = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'base_z')]) if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'base_z') >= 0 else 2
    d = init_data(m, mx, n)
    ts = jnp.full((n,), m.opt.timestep)
    t0 = time.time()
    d = step(d, ts)
    jax.block_until_ready(d)
    compile_s = time.time() - t0
    t0 = time.time()
    for _ in range(args.reps):
        d = step(d, ts)
    jax.block_until_ready(d)
    ms = (time.time() - t0) / args.reps * 1e3
    out = dict(variant=name, n_envs=n, tick_ms=ms, env_substeps_per_s=n * SUBSTEPS / ms * 1e3,
               env_ticks_per_s=n / ms * 1e3, compile_s=compile_s,
               nefc=int(d._impl.efc_J.shape[-2]),
               ncon_max=int(d._impl.contact.dist.shape[-1]),
               solver=int(m.opt.solver), iterations=int(m.opt.iterations),
               ls_iterations=int(m.opt.ls_iterations), cone=int(m.opt.cone),
               integrator=int(m.opt.integrator))
    print(f"  {name:16s} n={n:5d}: {ms:9.1f} ms/tick  {out['env_substeps_per_s']:12,.0f} env substeps/s "
          f"{out['env_ticks_per_s']:10,.0f} ticks/s  nefc {out['nefc']} ncon {out['ncon_max']} "
          f"(compile {compile_s:.0f}s)", flush=True)
    if args.check_ticks > 0:
        d = init_data(m, mx, 1)
        ts1 = jnp.full((1,), m.opt.timestep)
        traj = []
        for _ in range(args.check_ticks):
            d = step(d, ts1)
            traj.append(np.asarray(d.qpos[0]))
        traj = np.stack(traj)
        out["final_height"] = float(traj[-1, Z_Q[0]])
        if ref is not None:
            out["max_dq_vs_xml"] = float(np.abs(traj - ref).max())
            out["final_dq_vs_xml"] = float(np.abs(traj[-1] - ref[-1]).max())
            print(f"      drift vs xml over {args.check_ticks} ticks: max |dq| {out['max_dq_vs_xml']:.2e} "
                  f"final {out['final_dq_vs_xml']:.2e}, height {out['final_height']:.4f}", flush=True)
        else:
            print(f"      final height {out['final_height']:.4f}", flush=True)
        return out, traj
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=XML)
    ap.add_argument("--no-hold-pitch", action="store_true", help="no base-pitch hold in the drift check")
    ap.add_argument("--n-envs", type=int, nargs="+", default=[8])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--check-ticks", type=int, default=100)
    ap.add_argument("--batched-timestep", action="store_true",
                    help="per-env opt.timestep via tree_replace inside the vmap (what env.py does)")
    ap.add_argument("--batched-fields", action="store_true",
                    help="also per-env body_mass/inertia/ipos/friction/damping/stiffness/gravity/site_pos")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    global HOLD_PITCH
    HOLD_PITCH = not args.no_hold_pitch
    print(f"[profile] {jax.devices()[0].platform} {jax.devices()[0].device_kind}  xml {os.path.relpath(args.xml)}"
          f"  batched_timestep={args.batched_timestep} batched_fields={args.batched_fields}", flush=True)
    rows = []
    ref = None
    check_ticks = args.check_ticks
    for name in args.variants:
        if name not in VARIANTS:
            sys.exit(f"unknown variant {name}; known: {list(VARIANTS)}")
        for i, n in enumerate(args.n_envs):
            args.check_ticks = check_ticks if i == 0 else 0   # the drift check once per variant
            out, traj = run(name, n, args, ref)
            if name == "xml" and traj is not None:
                ref = traj
            rows.append(out)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(device=str(jax.devices()[0]), batched_timestep=args.batched_timestep, rows=rows),
                      f, indent=1)
        print(f"[profile] wrote {args.json}")


if __name__ == "__main__":
    main()
