"""What gait does a trained v2 policy actually settle into, and who writes it?

The v2 control law is additive (gait.assemble):

    target = feedforward(latched spec, phi)  +  reflexes(roll, pitch)  +  residual_scale * r

so the question "how much of the motion is the latched Fourier spec and how much is the per-tick
residual" is answerable exactly, by recomputing the three terms from the recorded state. This tool
runs a greedy rollout, throws away the start-up transient, and reports:

  * periodicity  -- stride period and duty from the CONTACT events (not the clock), cycle-to-cycle
                    spread of the joint trajectories on a common phase grid, and how far the
                    touchdown resync drags the latched clock
  * shape        -- duty factors, L/R phase offset, flight fraction, per-joint amplitudes, the
                    foot path in the base frame
  * authorship   -- per joint, the peak-to-peak and std of each of the three terms, their share of
                    the total target variance (with the cross terms, which are not negligible),
                    how often the residual sits on its rail, and how much the latched spec is
                    rewritten from cycle to cycle

    python walk_v2/tools/gait_shape.py --run walk_v2/runs/v2c_s2_free_dp2x_s0 \
        --checkpoint .../best_88473600.msgpack --seconds 20 --settle 4 --fig results/gait_shape.png
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

import gait
from config import config_from_dict
from env import EnvParams
from ppo import PPO
from train import make_eval_env, latest_checkpoint

JOINTS = ["hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R"]
TWO_PI = 2.0 * np.pi


def load(run, checkpoint, n_envs):
    run = Path(run)
    cfg = config_from_dict(json.loads((run / "resolved_config.json").read_text())["config"])
    ck = Path(checkpoint) if checkpoint else latest_checkpoint(run)
    env = make_eval_env(cfg, n_envs)
    agent = PPO(cfg, env, run, cfg.total_steps, seed=0, eval_env=None)
    agent.load(ck)
    print(f"[gait] {ck.name} @ {agent.step:,} steps  plant {cfg.model_path}")
    return cfg, env, agent


def record(env, agent, seed, n_max):
    """Greedy rollout; per tick, everything needed to rebuild the control law off-line."""
    act = agent._act_greedy
    params = EnvParams.final(agent.cfg)._replace(dr_scale=0.0, ctrl_jitter_ms=0.0,
                                                 ctrl_drop_prob=0.0, pitch_assist=0.0,
                                                 stoplight_prob=0.0)
    p = env.plant

    def body(carry, _):
        state, obs = carry
        a = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        # everything the target at THIS tick is built from, read before the step
        pre = dict(phi=state.phase, grav=jax.vmap(env._grav_body)(state.data),
                   gyro=jax.vmap(env._gyro)(state.data), prate=state.reflex_prate,
                   q=state.data.qpos[:, p.act_qadr], qd=state.data.qvel[:, p.act_dadr],
                   spec_prev=state.spec)
        state2, obs2, r, done, info = env.step(state, a, params)
        post = dict(spec=state2.spec, residual=state2.prev_residual, target=state2.prev_target,
                    phi_next=state2.phase, commit=info["commit"].astype(jnp.float32),
                    grounded=1.0 - info["foot_air"], alive=(~done).astype(jnp.float32),
                    freq=info["freq_hz"], toe=jax.vmap(env._toe_pos)(state2.data),
                    base=jax.vmap(env._base_pos)(state2.data),
                    rot=jax.vmap(env._base_rot)(state2.data),
                    vbody=jax.vmap(env._vel_body)(state2.data), t=state2.t,
                    qpos_full=state2.data.qpos)      # the whole pose, for gait_figures' robot
        return (state2, obs2), {**pre, **post}

    state, obs = env.reset(jax.random.PRNGKey(seed), params)
    _, rec = jax.lax.scan(jax.jit(body), (state, obs), None, length=n_max)
    return {k: np.asarray(v) for k, v in rec.items()}


def decompose(rec, cfg, plant, gp, e):
    """Rebuild feedforward / reflex / residual for env e. Returns (T,6) arrays."""
    phi, spec, res = rec["phi"][:, e], rec["spec"][:, e], rec["residual"][:, e]
    grav, gyro, prate = rec["grav"][:, e], rec["gyro"][:, e], rec["prate"][:, e]
    nominal = np.asarray(plant.nominal_ctrl)
    q_ref = np.stack([gait.feedforward(spec[i], phi[i], nominal, gp, xp=np) for i in range(len(phi))])
    add = np.zeros_like(q_ref)
    for i in range(len(phi)):
        u_roll, u_pitch = gait.reflexes(spec[i], grav[i, 1], gyro[i, 0], grav[i, 0], prate[i],
                                        gp, xp=np)
        add[i] = [u_roll, 0.0, u_pitch, u_roll, 0.0, -u_pitch]
    res_c = cfg.residual_scale * np.clip(res, -1.0, 1.0)
    return q_ref, add, res_c


def cycles_from_contact(grounded, t):
    """Touchdown ticks per foot (rising edges) -> stride periods."""
    out = []
    for f in range(2):
        g = grounded[:, f] > 0.5
        rise = np.flatnonzero(g[1:] & ~g[:-1]) + 1
        out.append(rise)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--settle", type=float, default=4.0, help="drop this much start-up transient")
    ap.add_argument("--env", type=int, default=0, help="which env to analyse in detail")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--npz", default=None, help="save the raw recording for off-line re-analysis")
    args = ap.parse_args()

    cfg, env, agent = load(args.run, args.checkpoint, args.episodes)
    gp = env.gp
    dt = env.control_dt
    n_max = int(round(args.seconds / dt))
    rec = record(env, agent, args.seed, n_max)
    if args.npz:
        np.savez_compressed(args.npz, dt=dt, **rec)
        print(f"[gait] wrote {args.npz}")
    i0 = int(round(args.settle / dt))
    alive = rec["alive"][:, args.env] > 0.5
    end = int(np.argmin(alive)) if (~alive).any() else n_max
    if end <= i0 + 200:
        print(f"[gait] env {args.env} only survived {end * dt:.2f} s -- nothing steady to analyse")
        return
    sl = slice(i0, end)
    print(f"[gait] analysing t = {i0 * dt:.1f} .. {end * dt:.1f} s of env {args.env} "
          f"({end - i0} ticks); {int(alive[:end].all())} alive throughout")

    q_ref, add, res_c = decompose(rec, cfg, env.plant, gp, args.env)
    target = rec["target"][:, args.env]
    q_lo, q_hi = np.asarray(env.plant.q_lo), np.asarray(env.plant.q_hi)
    q = rec["q"][:, args.env]
    grounded = rec["grounded"][:, args.env]
    phi = rec["phi"][:, args.env]
    out = {}

    # ---------------------------------------------------------------- periodicity
    print("\n=== PERIODICITY (from foot contact, not the clock) ===")
    rises = cycles_from_contact(grounded[sl], None)
    per = {}
    for f, r in enumerate(rises):
        if len(r) < 3:
            print(f"  foot {'LR'[f]}: {len(r)} touchdowns -- no cycle statistics")
            continue
        T = np.diff(r) * dt
        duty = np.array([grounded[sl][a:b, f].mean() for a, b in zip(r[:-1], r[1:])])
        per[f] = dict(T=T, duty=duty)
        print(f"  foot {'LR'[f]}: {len(T)} strides  period {T.mean() * 1000:6.1f} +- {T.std() * 1000:4.1f} ms "
              f"({T.std() / T.mean() * 100:4.1f} %)  [{T.min() * 1000:.0f} .. {T.max() * 1000:.0f}]  "
              f"stride freq {1 / T.mean():.2f} Hz   duty {duty.mean():.3f} +- {duty.std():.3f}")
    if 0 in per and 1 in per:
        # L/R phase offset: fraction of a stride between the left and the next right touchdown
        rl, rr = rises[0], rises[1]
        offs = []
        for a in rl:
            nxt = rr[rr > a]
            if len(nxt):
                offs.append((nxt[0] - a) * dt)
        offs = np.array(offs) / per[0]["T"].mean()
        print(f"  L->R touchdown offset {offs.mean():.3f} +- {offs.std():.3f} of a stride "
              f"(0.5 = perfectly alternating)")
        out["lr_offset"] = [float(offs.mean()), float(offs.std())]
    both = (grounded[sl][:, 0] > 0.5) & (grounded[sl][:, 1] > 0.5)
    none = (grounded[sl][:, 0] < 0.5) & (grounded[sl][:, 1] < 0.5)
    print(f"  double support {both.mean() * 100:.1f} %   flight {none.mean() * 100:.1f} %   "
          f"single support {(1 - both.mean() - none.mean()) * 100:.1f} %")
    v = rec["vbody"][sl, args.env, 0]
    print(f"  forward speed {v.mean():.2f} +- {v.std():.2f} m/s")

    # the latched clock vs the real stride
    f_latched = rec["freq"][sl, args.env]
    print(f"  latched clock {f_latched.mean():.3f} +- {f_latched.std():.3f} Hz "
          f"(range {gp.freq_lo:.1f}-{gp.freq_hi:.1f}); commits {int(rec['commit'][sl, args.env].sum())}")
    dphi = np.mod(np.diff(phi[sl]) + np.pi, TWO_PI) - np.pi
    expected = TWO_PI * f_latched[:-1] * dt
    resync = dphi - expected
    print(f"  touchdown resync moves the clock {np.abs(resync).sum() / TWO_PI:.2f} cycles over the "
          f"window ({np.abs(resync).mean() / TWO_PI * 100:.3f} % of a cycle per tick)")

    # cycle-to-cycle repeatability of the joint trajectories, on a common phase grid
    print("\n  cycle-to-cycle spread of the MEASURED joint angles (phase-aligned):")
    grid = np.linspace(0, TWO_PI, 64, endpoint=False)
    wraps = np.flatnonzero(np.diff(phi[sl]) < -np.pi) + 1
    prof = {j: [] for j in range(6)}
    for a, b in zip(wraps[:-1], wraps[1:]):
        ph = phi[sl][a:b]
        if len(ph) < 8:
            continue
        for j in range(6):
            prof[j].append(np.interp(grid, ph, q[sl][a:b, j], period=TWO_PI))
    n_cyc = len(prof[0])
    print(f"    {n_cyc} clock cycles")
    rows = []
    for j in range(6):
        P = np.array(prof[j])
        if P.size == 0:
            continue
        amp = P.mean(0).max() - P.mean(0).min()
        spread = P.std(0).mean()
        rows.append((JOINTS[j], amp, spread, spread / max(amp, 1e-9)))
        print(f"    {JOINTS[j]:<11} amplitude {np.degrees(amp):6.2f} deg   cycle-to-cycle sd "
              f"{np.degrees(spread):5.2f} deg  = {100 * spread / max(amp, 1e-9):4.1f} % of it")
    out["repeatability"] = [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in rows]

    # ---------------------------------------------------------------- authorship
    print("\n=== WHO WRITES THE MOTION (target = feedforward + reflex + residual) ===")
    raw = q_ref + add + res_c                       # what the control law asks for
    clipped = np.clip(raw, q_lo, q_hi)              # joint-limit clip
    print(f"  asked vs joint-limit clip: max {np.abs(raw - clipped).max():.3f} rad, "
          f"{100 * (np.abs(raw - clipped) > 1e-6)[sl].mean():.1f} % of joint-ticks clipped")
    d_slew = np.abs(clipped - target)[sl]
    rate = np.abs(np.diff(clipped[sl], axis=0)) / dt
    cap = np.asarray(cfg.motor_vel_limit)
    print(f"  slew limiter, in the window: median {np.degrees(np.median(d_slew)):.3f} deg, "
          f"p95 {np.degrees(np.percentile(d_slew, 95)):.3f}, max {np.degrees(d_slew.max()):.2f} deg; "
          f"commanded rate over the no-load cap on {100 * (rate > cap).mean():.1f} % of joint-ticks")
    # counterfactual: hold the residual at zero and re-clip -- what the residual actually adds
    # after the limits (the slew limiter is stateful, so this is the per-tick approximation)
    cf = np.clip(q_ref + add, q_lo, q_hi)
    d_res = np.abs(clipped - cf)[sl]
    print(f"  residual's effect AFTER the clip: mean {np.degrees(d_res.mean()):.2f} deg, "
          f"max {np.degrees(d_res.max()):.2f} deg")
    tot = clipped
    print(f"  {'joint':<11} {'p-p total':>9} | {'feedfwd':>8} {'reflex':>8} {'residual':>9} "
          f"| {'var share: ff':>13} {'reflex':>7} {'res':>6} {'cross':>7}")
    shares = {}
    for j in range(6):
        c = [q_ref[sl, j], add[sl, j], res_c[sl, j]]
        c = [x - x.mean() for x in c]
        tt = tot[sl, j] - tot[sl, j].mean()
        vt = np.var(tt)
        sv = [np.var(x) / vt if vt > 0 else 0.0 for x in c]
        cross = 1.0 - sum(sv)
        pp = lambda x: np.degrees(x.max() - x.min())
        print(f"  {JOINTS[j]:<11} {pp(tot[sl, j]):8.2f}d | {pp(q_ref[sl, j]):7.2f}d "
              f"{pp(add[sl, j]):7.2f}d {pp(res_c[sl, j]):8.2f}d | {sv[0]:12.2f} {sv[1]:7.2f} "
              f"{sv[2]:6.2f} {cross:7.2f}")
        shares[JOINTS[j]] = dict(pp_total=float(pp(tot[sl, j])), pp_ff=float(pp(q_ref[sl, j])),
                                 pp_reflex=float(pp(add[sl, j])), pp_res=float(pp(res_c[sl, j])),
                                 var_ff=float(sv[0]), var_reflex=float(sv[1]), var_res=float(sv[2]))
    out["authorship"] = shares
    print(f"  {'joint':<11} the MOTOR VELOCITY LIMIT rewrites the command:")
    for j in range(6):
        d = (target - clipped)[sl, j]
        print(f"  {JOINTS[j]:<11} slew-limited {100 * (np.abs(d) > 1e-6).mean():5.1f} % of ticks, "
              f"by up to {np.degrees(np.abs(d).max()):5.2f} deg (p-p of the correction "
              f"{np.degrees(d.max() - d.min()):5.2f} deg)")
    # the residual: a DC bias, or per-tick shaping? split it per clock cycle
    print(f"  {'joint':<11} residual = DC bias + shaping:")
    wr = np.flatnonzero(np.diff(phi[sl]) < -np.pi) + 1
    for j in range(6):
        dc, ac = [], []
        for a, b in zip(wr[:-1], wr[1:]):
            seg = res_c[sl][a:b, j]
            dc.append(seg.mean()); ac.append(seg.std())
        dc, ac = np.array(dc), np.array(ac)
        print(f"  {JOINTS[j]:<11} per-cycle DC {np.degrees(dc.mean()):+6.2f} deg "
              f"(sd across cycles {np.degrees(dc.std()):4.2f})   within-cycle shaping sd "
              f"{np.degrees(ac.mean()):4.2f} deg  -> {100 * ac.mean() ** 2 / (ac.mean() ** 2 + dc.std() ** 2 + 1e-12):3.0f} % shaping")
    # harmonic content of the achieved joint waveform (what "shape" means concretely)
    print(f"  {'joint':<11} harmonic content of the measured cycle (share of AC power):")
    for j in range(6):
        P = np.array(prof[j])
        if P.size == 0:
            continue
        F = np.fft.rfft(P.mean(0) - P.mean())
        pw = np.abs(F) ** 2
        tot_p = pw[1:].sum()
        print(f"  {JOINTS[j]:<11} 1st {100 * pw[1] / tot_p:4.1f} %   2nd {100 * pw[2] / tot_p:4.1f} %   "
              f"3rd {100 * pw[3] / tot_p:4.1f} %   >3rd {100 * pw[4:].sum() / tot_p:4.1f} %")
    r = rec["residual"][sl, args.env]
    print(f"  residual: |r| mean {np.abs(r).mean():.3f} of the rail, "
          f"{100 * (np.abs(r) >= 0.95).mean():.1f} % of joint-ticks saturated, "
          f"scale {cfg.residual_scale} rad")
    print(f"            per joint saturated %: " +
          " ".join(f"{JOINTS[j]} {100 * (np.abs(r[:, j]) >= 0.95).mean():.0f}" for j in range(6)))

    # ---------------------------------------------------------------- the spec itself
    print("\n=== THE LATCHED SPEC (mean over the window, in action units) ===")
    S = rec["spec"][sl, args.env]
    names = [("S_cam", gait.I_S_CAM, cfg.cam_amp), ("S_thigh", gait.I_S_THIGH, cfg.thigh_amp),
             ("S_hip", gait.I_S_HIP, cfg.roll_amp)]
    for nm, sl_i, amp in names:
        m = S[:, sl_i].mean(0)
        rail = 100 * (np.abs(S[:, sl_i]) >= 0.95).mean()
        print(f"  {nm:<8} a0 {m[0]:+.2f}  a1 {m[1]:+.2f} b1 {m[2]:+.2f}  a2 {m[3]:+.2f} b2 {m[4]:+.2f}"
              f"  a3 {m[5]:+.2f} b3 {m[6]:+.2f}   (x {amp} rad)  {rail:.0f} % of entries on a rail")
    kn = S[:, 39:44].mean(0)
    print(f"  knobs    Delta {kn[0]:+.2f} (x{cfg.delta_max} rad)  s {kn[1]:+.2f}  "
          f"o_cam {kn[2]:+.2f}  o_thigh {kn[3]:+.2f}  o_hip {kn[4]:+.2f}")
    print(f"  reflex   kp {S[:, 36].mean():+.2f}  kd {S[:, 37].mean():+.2f}  bias {S[:, 38].mean():+.2f}")
    print(f"  spec on a rail overall: {100 * (np.abs(S) >= 0.95).mean():.0f} % of all 44 entries")
    dS = np.abs(np.diff(S[rec['commit'][sl, args.env] > 0.5], axis=0))
    if len(dS):
        print(f"  rewritten per commit: mean |dspec| {dS.mean():.3f}, max {dS.max():.3f} "
              f"({len(dS)} commits) -- 0 = a truly fixed gait")
        out["spec_churn"] = [float(dS.mean()), float(dS.max())]

    # foot path in the base frame
    R = rec["rot"][sl, args.env]
    base = rec["base"][sl, args.env]
    toe = rec["toe"][sl, args.env]
    fb = np.einsum("tij,tfi->tfj", R, toe - base[:, None, :])
    print("\n=== FOOT PATH IN THE BASE FRAME ===")
    for f in range(2):
        print(f"  foot {'LR'[f]}: fore-aft {np.ptp(fb[:, f, 0]) * 100:5.1f} cm   "
              f"lateral {np.ptp(fb[:, f, 1]) * 100:4.1f} cm   vertical {np.ptp(fb[:, f, 2]) * 100:5.1f} cm")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    if args.fig:
        make_fig(args.fig, grid, prof, sl, q_ref, add, res_c, tot, grounded, phi, dt, n_cyc)


def make_fig(path, grid, prof, sl, q_ref, add, res_c, tot, grounded, phi, dt, n_cyc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 3, figsize=(14, 9))
    for j in range(6):
        a = ax[j // 3, j % 3]
        P = np.degrees(np.array(prof[j]))
        for k in range(0, len(P), max(1, len(P) // 40)):
            a.plot(grid / TWO_PI, P[k], color="0.75", lw=0.6)
        a.plot(grid / TWO_PI, P.mean(0), "k", lw=2)
        a.set_title(f"{JOINTS[j]}  ({n_cyc} cycles)", fontsize=9)
        a.set_xlabel("clock phase"); a.set_ylabel("deg")
    # decomposition over ~4 cycles
    n = min(len(tot[sl]), 400)
    t = np.arange(n) * dt
    a = ax[2, 0]
    a.plot(t, np.degrees(q_ref[sl][:n, 2]), label="feedforward (latched spec)")
    a.plot(t, np.degrees(add[sl][:n, 2]), label="reflex")
    a.plot(t, np.degrees(res_c[sl][:n, 2]), label="residual")
    a.plot(t, np.degrees(tot[sl][:n, 2]), "k--", lw=1, label="target")
    a.legend(fontsize=7); a.set_title("thigh_L: who writes the target", fontsize=9)
    a.set_xlabel("s"); a.set_ylabel("deg")
    a = ax[2, 1]
    a.plot(t, grounded[sl][:n, 0] * 0.9 + 1.05, lw=3)
    a.plot(t, grounded[sl][:n, 1] * 0.9, lw=3)
    a.set_yticks([0.45, 1.5]); a.set_yticklabels(["R", "L"]); a.set_title("contact", fontsize=9)
    a.set_xlabel("s")
    a = ax[2, 2]
    a.plot(t, phi[sl][:n] / TWO_PI, lw=1)
    a.set_title("latched clock phase", fontsize=9); a.set_xlabel("s")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[gait] wrote {path}")


if __name__ == "__main__":
    main()
