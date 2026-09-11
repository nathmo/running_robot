"""Does the trained policy stay inside the MACHINE, not just inside the reward?

The sim clamps what it can (drive.torque_limit caps every substep at the back-EMF envelope,
slew_limit caps the commanded target at the no-load speed, workspace_kill ends the episode when
a foot leaves the box), so a policy can never "exceed" those in the recorded trajectory -- what
matters is how much of each budget it SPENDS, and which quantities are not clamped at all and can
therefore run away: joint speed under back-drive, winding temperature, joint end-stops, pushrod
force, ground reaction force, base speed vs the rail-sweep bound.

Two parts:

  1. static -- the configured envelope against the CubeMars datasheets (back-EMF closure
     Kt == Ke, the XML forcerange against the datasheet peak current, corner speeds, the
     continuous torques behind the thermal node).
  2. measured -- greedy rollouts of the run's policy on the eval plant (nominal, no noise, no
     pushes, no assist; --dr for the training plant), one row per limit with the margin used.

Per control tick the probe reads the plant state BEFORE the step (auto-reset would overwrite it
after), so every sample belongs to a live episode. Torque is sampled from data.ctrl (the last of
the ten 1 kHz substeps) and, exactly, as an RMS over all ten substeps inverted out of the thermal
node: x_{k+1} = x_k + (dt/tau_th)(tau_rms^2/(tau_cont^2 s) - x_k).

    python walk_v2/tools/limits_check.py --run walk_v2/runs/v2c_s2_free_dp2x_s0
    python walk_v2/tools/limits_check.py --run ... --checkpoint ...best_88473600.msgpack --dr
    python walk_v2/tools/limits_check.py --run ... --episodes 8 --seconds 45 --json out.json
"""
import argparse
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np
import jax
import jax.numpy as jnp

import drive
from env import EnvParams
from evaluate import load_run

NAMES = ["hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R"]
# CubeMars datasheets (cubemars.com, pulled 2026-09-03; same numbers as walk_mit/motor_limit_check.py)
#   AK60-39 V3.0 KV80 (hip roll): Kt 0.12 Nm/A, R_ll 0.600 ohm, peak 17 A / 72 Nm, no-load 98 rpm
#   AKE90-8  KV35    (cam+thigh): Kt 0.272 Nm/A, R_ll 0.164 ohm, peak 72 A / 170 Nm, no-load 210 rpm
GEAR = np.array([39.0, 8.0, 8.0, 39.0, 8.0, 8.0])
R_LL = np.array([0.600, 0.164, 0.164, 0.600, 0.164, 0.164])
R_PACK = 0.065
DS_NOLOAD_RPM = np.array([98 * 39, 210 * 8, 210 * 8, 98 * 39, 210 * 8, 210 * 8])   # motor side
DS_PEAK_A = np.array([17.0, 72.0, 72.0, 17.0, 72.0, 72.0])
DS_PEAK_NM = np.array([72.0, 170.0, 170.0, 72.0, 170.0, 170.0])                     # joint side
# measured upper bounds on running speed (leg2d/rail_bound.py, flight_bound.py, 2026-09-01)
V_BOUND_GROUNDED = 6.55
V_BOUND_FLIGHT = 6.2


# --------------------------------------------------------------------------- static
def static_report(cfg, plant, out):
    kt = np.asarray(cfg.motor_kt_joint)
    r = np.asarray(cfg.motor_r_ohm)
    v_bus = float(cfg.motor_bus_volts)
    peak = np.asarray(plant.tau_peak)
    w_free = v_bus / kt
    w_ds = DS_NOLOAD_RPM * 2 * np.pi / 60 / GEAR
    i_peak = peak / kt
    w_corner = (v_bus - i_peak * r) / kt
    i_cont = np.asarray(cfg.thermal_tau_cont) / kt
    vel_cap = np.asarray(cfg.motor_vel_limit)

    print("== 1. configured envelope vs datasheet")
    print(f"  bus {v_bus:.0f} V, R = R_ll + pack {R_PACK} ohm = {r[0]:.3f} / {r[1]:.3f} ohm")
    print(f"  {'joint':10s} {'tau_peak':>9s} {'/ds':>6s} {'I@peak':>7s} {'/ds':>6s} "
          f"{'no-load':>8s} {'/ds':>6s} {'corner':>7s} {'tau_cont':>9s} {'I_cont':>7s}")
    for i, n in enumerate(NAMES):
        print(f"  {n:10s} {peak[i]:7.1f} Nm {100*peak[i]/DS_PEAK_NM[i]:5.0f}% {i_peak[i]:6.1f} A "
              f"{100*i_peak[i]/DS_PEAK_A[i]:5.0f}% {w_free[i]:6.2f} r/s {100*w_free[i]/w_ds[i]:5.0f}% "
              f"{w_corner[i]:5.2f} r/s {cfg.thermal_tau_cont[i]:6.1f} Nm {i_cont[i]:6.1f} A")
    ok_emf = bool(np.allclose(w_free, w_ds, rtol=0.01))
    ok_peak = bool(np.all(i_peak <= DS_PEAK_A) and np.all(w_corner > 0))
    ok_cap = bool(np.allclose(vel_cap, w_free, rtol=0.01))
    print(f"  back-EMF closure (Kt==Ke within 1%): {'PASS' if ok_emf else 'FAIL'}"
          f"   XML peak reachable at datasheet current: {'PASS' if ok_peak else 'FAIL'}"
          f"   command cap == no-load: {'PASS' if ok_cap else 'FAIL'}")
    out["static"] = dict(tau_peak=peak.tolist(), i_at_peak=i_peak.tolist(), no_load=w_free.tolist(),
                         corner=w_corner.tolist(), i_cont=i_cont.tolist(),
                         ok_emf=ok_emf, ok_peak=ok_peak, ok_vel_cap=ok_cap)
    return dict(kt=kt, r=r, v_bus=v_bus, peak=peak, w_free=w_free, vel_cap=vel_cap)


# --------------------------------------------------------------------------- rollout
def record(env, agent, seed, n_max, dr_scale=0.0, assist=0.0):
    """Greedy rollout; per tick, per env, the plant state before the step (+ info scalars)."""
    cfg = agent.cfg
    p = env.plant
    params = EnvParams.final(cfg)._replace(dr_scale=float(dr_scale), ctrl_jitter_ms=0.0,
                                           ctrl_drop_prob=0.0, pitch_assist=float(assist),
                                           stoplight_prob=0.0)
    spring_qadr = np.array([int(p.m.jnt_qposadr[j]) for j in p.spring_jids])
    spring_dadr = np.array([int(p.m.jnt_dofadr[j]) for j in p.spring_jids])
    spring_k = np.array([float(p.m.jnt_stiffness[j]) for j in p.spring_jids])
    spring_c = np.array([float(p.m.dof_damping[d]) for d in spring_dadr])
    ws_ref = jnp.asarray(p.ws_ref)

    def probe(st):
        d = st.data
        base, R = env._base_pos(d), env._base_rot(d)
        toe = env._toe_pos(d)
        tb = jnp.stack([R.T @ (toe[i] - base) for i in range(2)])
        grounded, fn, _ = env._contacts(d)
        return dict(q=d.qpos[p.act_qadr], qd=d.qvel[p.act_dadr], tau=d.ctrl,
                    tvel=st.prev_target_vel, thermal=st.thermal_x,
                    dx=tb[:, 0] - ws_ref[:, 0], dz=tb[:, 2] - ws_ref[:, 2],
                    toe_z=env._toe_heights(d), fn=fn, grounded=grounded.astype(jnp.float32),
                    spring=d.qpos[spring_qadr], spring_v=d.qvel[spring_dadr],
                    z=base[2], grav=env._grav_body(d), gyro=env._gyro(d),
                    v_body=env._vel_body(d), v_world=env._vel_world(d), y=env._y(d),
                    th_scale=st.draw.thermal_scale, tq_scale=st.draw.torque_scale,
                    t=st.t, d_run=st.sprint_d, crossed=st.crossed.astype(jnp.float32))

    key = jax.random.PRNGKey(seed)
    state, obs = env.reset(key, params)

    def body(carry, _):
        state, obs, alive = carry
        rec = jax.vmap(probe)(state)
        a = jnp.clip(agent._act_greedy(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        state2, obs2, r, done, info = env.step(state, a, params)
        rec["alive"] = alive.astype(jnp.float32)
        rec["fallen"] = (alive & done & info["fallen"]).astype(jnp.float32)
        rec["term_ws"] = (alive & done & info["term_ws"]).astype(jnp.float32)
        rec["finished"] = (alive & done & info["finished"]).astype(jnp.float32)
        return (state2, obs2, alive & ~done), rec

    init = (state, obs, jnp.ones(env.n_envs, bool))
    _, rec = jax.lax.scan(body, init, None, length=n_max)
    return {k: np.asarray(v) for k, v in rec.items()}      # [T, N, ...]


def flat(rec, key, live):
    """Samples of rec[key] over live ticks only -> [n_samples, ...trailing]."""
    a = rec[key]
    return a[live]


def pct(x):
    return f"{100 * float(x):5.1f}%"


def _longest_run(mask):
    """Longest run of consecutive True ticks in [T, N] (per column, max over columns)."""
    best = 0
    for j in range(mask.shape[1]):
        n = 0
        for v in mask[:, j]:
            n = n + 1 if v else 0
            best = max(best, n)
    return best


def _lut_check(env, dx, dz):
    """The recorded toe positions against the MEASURED reachable map (cpg_foot_lut.npz).

    The LUT is the m7 plant's Newton-solved 4-bar map (walk_mit/model/plot_reachability.py):
    grid dx +-0.30 m / dz 0..0.10 m about ITS nominal toe, with the per-cell IK residual in
    `reach`. The v2 keyframe toe sits at a small offset from that nominal, so the recorded dx/dz
    are shifted into LUT coordinates before the lookup. Outside the tabulated grid the answer is
    'not measured', which is exactly what workspace_kill exists to prevent.
    """
    from pathlib import Path as _P
    f = _P(env.plant.cfg.model_path).parent / "cpg_foot_lut.npz"
    f = f if f.exists() else _P(__file__).resolve().parents[1] / "model" / "cpg_foot_lut.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    gx, gz, reach = d["dx_grid"], d["dz_grid"], d["reach"]
    nom = d["nominal_toe"]
    ref = np.asarray(env.plant.ws_ref)                       # [2, 3] toe in base frame at the keyframe
    off_x, off_z = float(ref[0, 0] - nom[0]), float(ref[0, 2] - nom[2])
    out_grid, unreach, worst = [], [], []
    for i in range(2):
        x, z = dx[:, i] + off_x, dz[:, i] + off_z
        og = (x < gx[0]) | (x > gx[-1]) | (z < gz[0]) | (z > gz[-1])
        ix = np.clip(np.searchsorted(gx, x), 0, len(gx) - 1)
        iz = np.clip(np.searchsorted(gz, z), 0, len(gz) - 1)
        bad = og | (reach[ix, iz] > 0.005)                   # 5 mm IK residual = not assemblable
        out_grid.append(og.mean())
        unreach.append(bad.mean())
        worst.append(x[np.argmax(np.abs(x))])
    return dict(off_x=off_x, off_z=off_z, out_grid=np.array(out_grid),
                unreach=np.array(unreach), worst_dx=np.array(worst))


def measured_report(rec, env, cfg, st, out, label):
    p = env.plant
    dt = env.control_dt
    live = rec["alive"] > 0.5                                     # [T, N]
    n_live = int(live.sum())
    kt, r, v_bus, peak, w_free = st["kt"], st["r"], st["v_bus"], st["peak"], st["w_free"]
    tau_cont = np.asarray(cfg.thermal_tau_cont)

    q, qd, tau = flat(rec, "q", live), flat(rec, "qd", live), flat(rec, "tau", live)
    tq_scale = flat(rec, "tq_scale", live)[:, None]
    lim = np.minimum(peak * tq_scale, kt * np.maximum(v_bus - kt * np.abs(qd), 0.0) / r)
    branch_emf = (kt * np.maximum(v_bus - kt * np.abs(qd), 0.0) / r) < peak * tq_scale
    sat = np.abs(tau) >= 0.98 * lim

    # exact per-tick RMS torque, inverted out of the thermal node (uses consecutive live ticks)
    x = rec["thermal"]                                             # [T, N, 6]
    pair = live[:-1] & live[1:]
    x0, x1 = x[:-1][pair], x[1:][pair]
    s_th = rec["th_scale"][:-1][pair][:, None]
    tau_rms = np.sqrt(np.maximum(((x1 - x0) * cfg.thermal_tau_s / dt + x0), 0.0)) * tau_cont * np.sqrt(s_th)

    print(f"\n== 2. measured on {label}: {n_live} live ticks "
          f"({n_live * dt:.0f} s of robot time over {rec['alive'].shape[1]} episodes)")
    print(f"  falls {int(rec['fallen'].sum())}  workspace-kills {int(rec['term_ws'].sum())}  "
          f"finishes {int(rec['finished'].sum())}")

    print("\n  -- torque (sim CLAMPS this: the number to read is the margin spent)")
    print(f"  {'joint':10s} {'|tau|max':>9s} {'/peak':>6s} {'rms/tick max':>13s} {'sat ticks':>10s} "
          f"{'emf-limited':>12s} {'I_peak':>7s} {'/ds':>5s}")
    for i, n in enumerate(NAMES):
        tmax = float(np.abs(tau[:, i]).max())
        print(f"  {n:10s} {tmax:7.1f} Nm {100*tmax/peak[i]:5.0f}% {float(tau_rms[:, i].max()):11.1f} Nm "
              f"{pct(sat[:, i].mean()):>10s} {pct(branch_emf[:, i].mean()):>12s} "
              f"{tmax/kt[i]:6.1f} A {100*tmax/kt[i]/DS_PEAK_A[i]:4.0f}%")

    print("\n  -- joint speed (NOT clamped: back-drive can exceed the no-load speed)")
    print(f"  {'joint':10s} {'|qd|max':>9s} {'p99.9':>8s} {'no-load':>8s} {'over':>7s} "
          f"{'cmd-slew at cap':>16s}")
    tvel = flat(rec, "tvel", live)
    for i, n in enumerate(NAMES):
        v = np.abs(qd[:, i])
        print(f"  {n:10s} {v.max():7.2f} r/s {np.percentile(v, 99.9):6.2f} {w_free[i]:6.2f} "
              f"{pct((v > w_free[i]).mean()):>7s} {pct((np.abs(tvel[:, i]) >= 0.99 * st['vel_cap'][i]).mean()):>16s}")

    print("\n  -- electrical (per-motor current from tau/Kt; bus current = mech + I^2R over 6 motors)")
    i_ph = np.abs(tau) / kt
    p_mech = np.sum(tau * qd, axis=1)
    p_cu = np.sum((tau / kt) ** 2 * r, axis=1)
    p_bus = np.maximum(p_mech, 0.0) + p_cu
    print(f"  peak phase current  hip_roll {i_ph[:, [0, 3]].max():.1f} A (ds 17)   "
          f"cam/thigh {i_ph[:, [1, 2, 4, 5]].max():.1f} A (ds 72)")
    print(f"  bus power  peak {p_bus.max() / 1e3:.2f} kW  mean {p_bus.mean():.0f} W  "
          f"-> pack current peak {p_bus.max() / v_bus:.0f} A  mean {p_bus.mean() / v_bus:.0f} A"
          f"   (regen peak {-min(p_mech.min(), 0.0):.0f} W)")

    print("\n  -- winding temperature (single-node, tau_th = "
          f"{cfg.thermal_tau_s:.0f} s; penalty above {cfg.thermal_penalty_frac:.2f}, limit 1.0)")
    xl = x[live]
    last = np.array([np.max(np.nonzero(live[:, j])[0]) for j in range(live.shape[1])])   # per-episode end
    x_end = np.array([x[last[j], j] for j in range(live.shape[1])]).mean(0)
    duty = (tau_rms / tau_cont) ** 2                     # steady-state X of this duty cycle
    print(f"  {'joint':10s} {'X max':>7s} {'X end':>7s} {'X_ss (duty)':>12s} {'t to 0.85':>10s} "
          f"{'t to 1.0':>9s}")
    for i, n in enumerate(NAMES):
        d_ss = float(duty[:, i].mean())
        t85 = -cfg.thermal_tau_s * np.log(1 - 0.85 / d_ss) if d_ss > 0.85 else np.inf
        t100 = -cfg.thermal_tau_s * np.log(1 - 1.0 / d_ss) if d_ss > 1.0 else np.inf
        f85 = f"{t85:8.0f}s" if np.isfinite(t85) else "   never"
        f100 = f"{t100:7.0f}s" if np.isfinite(t100) else "  never"
        print(f"  {n:10s} {xl[:, i].max():6.2f} {x_end[i]:6.2f} {d_ss:11.2f} "
              f"{f85:>10s} {f100:>9s}")

    print("\n  -- joint end-stops (XML jnt_range on the six actuated joints)")
    print(f"  {'joint':10s} {'q min':>8s} {'q max':>8s} {'range':>18s} {'margin':>8s} {'<2 deg':>8s}")
    for i, n in enumerate(NAMES):
        lo, hi = float(p.q_lo[i]), float(p.q_hi[i])
        if hi <= lo:                                    # unlimited joint
            print(f"  {n:10s} {q[:, i].min():7.3f} {q[:, i].max():7.3f}      (unlimited)")
            continue
        m = min(q[:, i].min() - lo, hi - q[:, i].max())
        near = ((q[:, i] - lo < np.deg2rad(2)) | (hi - q[:, i] < np.deg2rad(2))).mean()
        print(f"  {n:10s} {q[:, i].min():7.3f} {q[:, i].max():7.3f} "
              f"{f'[{lo:.3f}, {hi:.3f}]':>18s} {m:7.3f} {pct(near):>8s}")

    print("\n  -- foot workspace (base frame, relative to the keyframe toe; kill box "
          f"|dx|<{cfg.workspace_dx_max}, {cfg.workspace_dz_min}<dz<{cfg.workspace_dz_max}, "
          f"grace {cfg.workspace_grace_s} s)")
    dx, dz = flat(rec, "dx", live), flat(rec, "dz", live)
    out_box = ((np.abs(rec["dx"]) > cfg.workspace_dx_max) | (rec["dz"] > cfg.workspace_dz_max)
               | (rec["dz"] < cfg.workspace_dz_min)) & (rec["alive"] > 0.5)[..., None]
    for i, s in enumerate("LR"):
        run_len = _longest_run(out_box[:, :, i]) * dt
        print(f"  foot {s}: dx [{dx[:, i].min():+.3f}, {dx[:, i].max():+.3f}] m "
              f"(box +-{cfg.workspace_dx_max}; {pct(np.mean(np.abs(dx[:, i]) > cfg.workspace_dx_max))} of ticks "
              f"outside, longest run {run_len:.2f} s)   dz [{dz[:, i].min():+.3f}, {dz[:, i].max():+.3f}] m "
              f"(box {cfg.workspace_dz_min}..{cfg.workspace_dz_max})")
    lut = _lut_check(env, dx, dz)
    if lut is not None:
        print(f"  vs the MEASURED envelope (cpg_foot_lut.npz, dx +-0.30 / dz 0..0.10 about the LUT toe, "
              f"offset {lut['off_x']:+.3f} / {lut['off_z']:+.3f} m to this keyframe):")
        for i, s in enumerate("LR"):
            print(f"    foot {s}: {pct(lut['out_grid'][i])} of ticks outside the tabulated box "
                  f"({pct(lut['unreach'][i])} at (dx,dz) the linkage cannot reach), "
                  f"worst dx {lut['worst_dx'][i]:+.3f} m in LUT coordinates")
        out.setdefault("lut", {})[label] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                            for k, v in lut.items()}

    print("\n  -- structure: pushrod / leg series spring (k = "
          f"{float(np.mean([p.m.jnt_stiffness[j] for j in p.spring_jids])) / 1e3:.1f} kN/m, travel +-0.03 m)")
    sp, spv = flat(rec, "spring", live), flat(rec, "spring_v", live)
    k = np.array([float(p.m.jnt_stiffness[j]) for j in p.spring_jids])
    c_d = np.array([float(p.m.dof_damping[int(p.m.jnt_dofadr[j])]) for j in p.spring_jids])
    f_rod = sp * k + spv * c_d
    for i, s in enumerate("LR"):
        print(f"  leg {s}: deflection [{1e3*sp[:, i].min():+.2f}, {1e3*sp[:, i].max():+.2f}] mm "
              f"(stop +-30 mm)   rod force max |F| {np.abs(f_rod[:, i]).max():.0f} N")

    print("\n  -- contact / body")
    fn = flat(rec, "fn", live)
    bw = p.bw
    v_body, v_world = flat(rec, "v_body", live), flat(rec, "v_world", live)
    z, grav, gyro = flat(rec, "z", live), flat(rec, "grav", live), flat(rec, "gyro", live)
    y = flat(rec, "y", live)
    sp_h = np.linalg.norm(v_world[:, :2], axis=1)
    print(f"  GRF peak  L {fn[:, 0].max()/bw:.2f} BW   R {fn[:, 1].max()/bw:.2f} BW   "
          f"sum peak {(fn.sum(1)).max()/bw:.2f} BW   (mass {p.total_mass:.2f} kg, 1 BW = {bw:.0f} N)")
    print(f"  base speed  max |v| {sp_h.max():.2f} m/s   fwd max {v_body[:, 0].max():.2f}   "
          f"mean fwd {v_body[:, 0].mean():.2f}   bound grounded {V_BOUND_GROUNDED} / flight {V_BOUND_FLIGHT} m/s")
    print(f"  base height [{z.min():.3f}, {z.max():.3f}] m (stand {p.height_stand:.3f}, "
          f"term {cfg.term_height})   |lateral y| max {np.abs(y).max():.2f} m")
    print(f"  tilt: grav_z [{grav[:, 2].min():.2f}, {grav[:, 2].max():.2f}] (term {cfg.term_gravity_z})   "
          f"|gyro| max {np.abs(gyro).max(axis=0).round(2).tolist()} rad/s")

    out[label] = dict(
        n_live=n_live, falls=int(rec["fallen"].sum()), ws_kills=int(rec["term_ws"].sum()),
        tau_max=np.abs(tau).max(0).tolist(), tau_frac_peak=(np.abs(tau).max(0) / peak).tolist(),
        tau_rms_max=tau_rms.max(0).tolist(), sat_frac=sat.mean(0).tolist(),
        emf_frac=branch_emf.mean(0).tolist(), qd_max=np.abs(qd).max(0).tolist(),
        qd_over_noload_frac=(np.abs(qd) > w_free).mean(0).tolist(),
        i_phase_max=i_ph.max(0).tolist(), p_bus_peak=float(p_bus.max()), p_bus_mean=float(p_bus.mean()),
        thermal_max=xl.max(0).tolist(), thermal_duty_ss=duty.mean(0).tolist(),
        q_min=q.min(0).tolist(), q_max=q.max(0).tolist(),
        dx_min=dx.min(0).tolist(), dx_max=dx.max(0).tolist(),
        dz_min=dz.min(0).tolist(), dz_max=dz.max(0).tolist(),
        rod_force_max=np.abs(f_rod).max(0).tolist(),
        spring_mm=[float(1e3 * sp.min()), float(1e3 * sp.max())],
        grf_peak_bw=float(fn.sum(1).max() / bw), v_max=float(sp_h.max()),
        z_min=float(z.min()), z_max=float(z.max()), y_absmax=float(np.abs(y).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--dr", action="store_true", help="also measure on the randomized training plant")
    ap.add_argument("--assist", type=float, default=0.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump", default=None, help="npz of every per-tick record (offline analysis)")
    ap.add_argument("--dr-sweep", default=None,
                    help="comma-separated dr_scale levels: survival + envelope per randomization level")
    args = ap.parse_args()

    cfg, env, agent = load_run(args.run, args.checkpoint, n_envs=args.episodes, dr=False)
    n_max = int(round((args.seconds or cfg.episode_s) / env.control_dt))
    out = dict(run=str(args.run), checkpoint=str(args.checkpoint or ""), step=int(agent.step),
               episodes=args.episodes, seconds=n_max * env.control_dt)
    st = static_report(cfg, env.plant, out)
    rec = record(env, agent, args.seed, n_max, dr_scale=0.0, assist=args.assist)
    measured_report(rec, env, cfg, st, out, "nominal")
    if args.dump:
        np.savez_compressed(args.dump, **{f"nominal_{k}": v for k, v in rec.items()})
    if args.dr:
        cfg_d, env_d, agent_d = load_run(args.run, args.checkpoint, n_envs=args.episodes, dr=True)
        rec_d = record(env_d, agent_d, args.seed, n_max, dr_scale=1.0, assist=args.assist)
        measured_report(rec_d, env_d, cfg_d, st, out, "randomized")
        if args.dump:
            np.savez_compressed(args.dump, **{f"nominal_{k}": v for k, v in rec.items()},
                                **{f"dr_{k}": v for k, v in rec_d.items()})
    if args.dr_sweep:
        cfg_s, env_s, agent_s = load_run(args.run, args.checkpoint, n_envs=args.episodes, dr=True)
        print(f"\n== 3. randomization sweep ({args.episodes} greedy episodes per level, "
              f"cap {n_max * env_s.control_dt:.0f} s; the training plant with dr_scale dialled)")
        print(f"  {'dr_scale':>9s} {'mean ep':>8s} {'falls':>6s} {'ws-kill':>8s} {'dist':>7s} "
              f"{'|tau|max':>9s} {'|qd|max':>8s} {'X max':>6s}")
        rows = []
        for lv in [float(x) for x in args.dr_sweep.split(",")]:
            r = record(env_s, agent_s, args.seed, n_max, dr_scale=lv, assist=args.assist)
            liv = r["alive"] > 0.5
            ep = liv.sum(0) * env_s.control_dt
            ends = np.array([r["d_run"][max(int(liv[:, j].sum()) - 1, 0), j] for j in range(liv.shape[1])])
            row = dict(dr_scale=lv, ep_mean=float(ep.mean()), falls=int(r["fallen"].sum()),
                       ws=int(r["term_ws"].sum()), dist=float(ends.mean()),
                       tau_max=float(np.abs(r["tau"][liv]).max()), qd_max=float(np.abs(r["qd"][liv]).max()),
                       x_max=float(r["thermal"][liv].max()))
            rows.append(row)
            print(f"  {lv:9.3f} {row['ep_mean']:7.2f}s {row['falls']:6d} {row['ws']:8d} "
                  f"{row['dist']:6.1f}m {row['tau_max']:7.1f}Nm {row['qd_max']:6.1f}r/s {row['x_max']:6.2f}")
        out["dr_sweep"] = rows
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\n[limits] wrote {args.json}")


if __name__ == "__main__":
    main()
