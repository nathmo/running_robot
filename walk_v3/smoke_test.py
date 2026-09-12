"""walk_v3 smoke test: the invariants the artifact makes checkable, run before any GPU hour.

    python walk_v3/smoke_test.py            # everything (a few minutes on CPU)
    python walk_v3/smoke_test.py --quick    # no env stepping (seconds)

Checks:
  gait     numpy == jax bit-level agreement; knobs-zero mirror identity u_L(phi) + u_R(phi+pi) = 0;
           mirror(action) reproduces the mirrored trajectory; impedance exp map ranges; sign rule
  thermal  the two anchors: peak (170/55) from cold reaches the limit at 5 s with tau_th = 45 s;
           1.0x continuous reaches 52% at 33 s; the third back-to-back dash exceeds 0.85
  drive    delay selection at substep granularity; torque-speed clamp closes to no-load speed
  plant    v2 XML loads in MJX; leg spring 5 +- 0.5 mm at 1 BW (plant_fit.json); keyframe stands
  env      obs layout 387/412 (policy) and 363/388 (library); commit flag at t0; spec latches from
           the commit tick's action and from no other; mask == commit; a random policy runs N
           ticks with finite obs; auto-reset works
  net      masked log-prob: latched dims invisible off-commit, gradient-free off-commit
"""
import argparse
import json
import sys
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import jax
import jax.numpy as jnp

import gait
import drive
import networks as nets
from config import get_config

OK = 0


def check(name, cond, detail=""):
    global OK
    print(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        OK += 1


def test_gait():
    print("gait")
    cfg = get_config("v2_s2_free")
    p = gait.GaitParams.from_cfg(cfg)
    rng = np.random.default_rng(0)
    nominal = np.array([0, 0, 0.12, 0, 0, -0.12])
    a = rng.uniform(-1, 1, 50)
    t_np = gait.assemble(a[:44], a[44:], 1.3, 0.1, -0.2, 0.05, 0.3, nominal, p, xp=np)
    t_jx = gait.assemble(jnp.asarray(a[:44]), jnp.asarray(a[44:]), jnp.float32(1.3), 0.1, -0.2, 0.05, 0.3,
                         jnp.asarray(nominal), p)
    err = max(float(np.abs(np.asarray(x) - y).max()) for x, y in zip(t_jx, t_np))
    check("numpy == jax (float32 rounding)", err < 1e-5, f"max diff {err:.1e}")
    s = a[:44].copy()
    s[39:44] = 0
    worst = 0.0
    for ph in np.linspace(0, 2 * np.pi, 9):
        qL = gait.feedforward(s, ph, nominal, p, xp=np) - nominal
        qR = gait.feedforward(s, ph + np.pi, nominal, p, xp=np) - nominal
        worst = max(worst, float(np.abs(qL[[1, 2, 0]] + qR[[4, 5, 3]]).max()))
    check("knobs 0: u_L(phi) + u_R(phi+pi) = 0", worst < 1e-12, f"{worst:.1e}")
    sm = np.asarray(gait.mirror_action(jnp.asarray(a)))[:44].astype(np.float64)
    delta = p.delta_max * np.clip(a[39], -1, 1)
    worst = 0.0
    for ph in np.linspace(0, 2 * np.pi, 7):
        q = gait.feedforward(a[:44], ph, nominal, p, xp=np) - nominal
        qm = gait.feedforward(sm, ph - np.pi - delta, nominal, p, xp=np) - nominal
        worst = max(worst, float(np.abs(qm + q[gait.MIRROR_PERM]).max()))
    check("mirror(action) = mirrored trajectory at phi - pi - Delta", worst < 1e-6, f"{worst:.1e}")
    one = np.zeros(44)
    one[21] = 1.0
    kp, _ = gait.impedance(one, 0.0, p, xp=np)
    check("kp profile +1 -> x2.5", abs(kp[1] / 200.0 - 2.5) < 1e-6, f"{kp[1]:.1f}")
    one[21] = -1.0
    kp, _ = gait.impedance(one, 0.0, p, xp=np)
    check("kp profile -1 -> /3", abs(kp[1] * 3.0 / 200.0 - 1.0) < 1e-6, f"{kp[1]:.1f}")
    one = np.zeros(44)
    one[28] = 1.0
    _, kd = gait.impedance(one, 0.0, p, xp=np)
    check("kd soften-only: +1 -> x1", abs(kd[1] - 5.0) < 1e-6, f"{kd[1]:.2f}")
    _, kd = gait.impedance(-one, 0.0, p, xp=np)
    check("kd -1 -> /4", abs(kd[1] * 4.0 - 5.0) < 1e-6, f"{kd[1]:.2f}")
    # sign rule: o_cam (+,+) is a fore-aft split -> opposite joint signs on mirrored axes
    o = np.zeros(44)
    o[41] = 1.0
    q = gait.feedforward(o, 0.0, nominal, p, xp=np) - nominal
    check("o_cam enters (+,+) in joint space", q[1] > 0 and q[4] > 0, f"cam_L {q[1]:+.3f} cam_R {q[4]:+.3f}")
    ur, up = gait.reflexes(np.zeros(44), 0.0, 0.0, 0.1, 0.0, p, xp=np)
    check("pitch reflex: nose-down -> feet forward (u_p < 0 on thigh_L, +u on thigh_R)", up < 0, f"{up:+.3f}")
    check("freq map [0.5, 5] Hz", abs(gait.frequency(-1.0, p, np) - 0.5) < 1e-9 and abs(gait.frequency(1.0, p, np) - 5.0) < 1e-9)


def test_thermal():
    print("thermal (tau_th 45 s, single node)")
    cfg = get_config("v2_s2_free")
    tau = cfg.thermal_tau_s
    t5 = drive.thermal_time_to_limit(170.0 / 55.0, tau)
    check("peak 170/55 from cold reaches the limit at 5 s", abs(t5 - 5.0) < 0.3, f"{t5:.2f} s")
    # 33 s dash at 1.0x continuous
    x = 0.0
    dt = 0.01
    for _ in range(3300):
        x = float(drive.thermal_update(jnp.asarray(x), jnp.asarray(55.0 ** 2), dt, tau, jnp.asarray(55.0), jnp.asarray(1.0)))
    check("33 s at 1.0x continuous -> 52%", abs(x - 0.52) < 0.02, f"{x:.3f}")
    x = 0.0
    peaks = []
    for _ in range(3):
        for _ in range(3300):
            x = float(drive.thermal_update(jnp.asarray(x), jnp.asarray(55.0 ** 2), dt, tau, jnp.asarray(55.0), jnp.asarray(1.0)))
        peaks.append(x)
    check("back-to-back dashes 0.52 -> 0.77 -> 0.89 (> 0.85 on the third)", peaks[2] > 0.85 and peaks[1] < 0.85,
          " ".join(f"{v:.2f}" for v in peaks))


def test_drive():
    print("drive")
    cmds = jnp.stack([jnp.full(3, 0.0), jnp.full(3, 1.0), jnp.full(3, 2.0)])   # current, prev, prev2
    picks = [int(drive.live_command(k, 12.0, cmds)[0]) for k in range(10)]
    check("12 ms delay: substeps 0-1 use two-back, 2-9 use previous", picks == [2, 2] + [1] * 8, str(picks))
    picks = [int(drive.live_command(k, 6.0, cmds)[0]) for k in range(10)]
    check("6 ms delay: substeps 0-5 previous, 6-9 current", picks == [1] * 6 + [0] * 4, str(picks))
    cfg = get_config("v2_s2_free")
    kt, r = jnp.asarray(cfg.motor_kt_joint), jnp.asarray(cfg.motor_r_ohm)
    peak = jnp.array([61.2, 144.5, 144.5, 61.2, 144.5, 144.5])
    lim0 = drive.torque_limit(jnp.zeros(6), peak, 1.0, kt, r, 48.0)
    check("torque clamp at rest = peak", bool(jnp.allclose(lim0, peak)))
    w_nl = 48.0 / kt
    lim_nl = drive.torque_limit(w_nl, peak, 1.0, kt, r, 48.0)
    check("torque clamp closes to 0 at no-load speed", float(jnp.abs(lim_nl).max()) < 1e-6,
          f"no-load {np.round(np.asarray(w_nl), 2)} rad/s")


def test_plant():
    print("plant")
    import mujoco
    from mujoco import mjx
    from plant import Plant, resolve
    fit = Path(PKG_DIR / "model" / "plant_fit.json")
    check("plant_fit.json exists (calibrate_plant.py ran)", fit.exists())
    if fit.exists():
        f = json.loads(fit.read_text())
        check("leg spring 5 +- 0.5 mm at 1 BW on one leg", abs(f["leg_defl_nominal_m"] - 0.005) < 5e-4,
              f"{1e3 * f['leg_defl_nominal_m']:.2f} mm (x0.5 {1e3 * f['leg_defl_x0p5_m']:.1f}, x1.5 {1e3 * f['leg_defl_x1p5_m']:.1f})")
        check("thigh corner within 25% of 6.3 Hz at 200/5", abs(np.log(f["thigh_corner_hz_200_5"] / 6.3)) < 0.25,
              f"{f['thigh_corner_hz_200_5']:.2f} Hz, peaking {f['thigh_peaking_db_200_5']:+.2f} dB")
    for preset in ("v2_s2_free", "v2_s1_planar"):
        cfg = get_config(preset)
        pl = Plant(cfg)
        check(f"{preset}: MJX model, nq={pl.nq} nu=6 springs present", pl.nu == 6 and (pl.spring_jids >= 0).all())
        check(f"{preset}: stance height ~1.01 m", abs(pl.height_stand - 1.009) < 0.01, f"{pl.height_stand:.4f}")


def test_heading(env, cfg):
    """The v3 heading channel: present, in the right place, and mirrored the right way.

    All three are silent failures. A frame that is the right WIDTH but writes the heading into the
    wrong column trains a policy on garbage and the widths still check out; a mirror that copies the
    heading instead of negating it teaches the symmetry loss that drifting left and drifting right
    call for the same correction, which is worse than having no symmetry loss at all."""
    import gait
    from env import FRAME_DIM
    check("frame carries the heading channel", FRAME_DIM == 34, FRAME_DIM)
    check("heading has an obs scale", "heading" in cfg.obs_scales, sorted(cfg.obs_scales))
    # the mirror must negate heading (last column) exactly as it negates the LP yaw rate (col 24)
    f = jnp.zeros((1, FRAME_DIM)).at[0, 24].set(0.7).at[0, FRAME_DIM - 1].set(0.3)
    m = gait.mirror_frame(f)
    check("mirror negates the LP yaw rate", abs(float(m[0, 24]) + 0.7) < 1e-6, f"{float(m[0, 24]):+.3f}")
    check("mirror negates the heading", abs(float(m[0, FRAME_DIM - 1]) + 0.3) < 1e-6,
          f"{float(m[0, FRAME_DIM - 1]):+.3f}")
    check("mirror is an involution on the frame",
          float(jnp.abs(gait.mirror_frame(m) - f).max()) < 1e-6,
          f"{float(jnp.abs(gait.mirror_frame(m) - f).max()):.1e}")
    # and the whole-observation mirror must agree, per history frame
    import networks as nets
    mo = nets.ObsMirror(env)
    o = jnp.zeros((1, env.obs_dim))
    for k in range(cfg.history_len):
        o = o.at[0, k * FRAME_DIM + FRAME_DIM - 1].set(0.3)
    om = mo(o[:, :env.actor_dim])
    got = [float(om[0, k * FRAME_DIM + FRAME_DIM - 1]) for k in range(cfg.history_len)]
    check("ObsMirror negates heading in every history frame",
          all(abs(g + 0.3) < 1e-6 for g in got), f"{got[0]:+.3f} x{len(got)}")


def test_env(quick):
    print("env")
    from env import DashEnvV2, EnvParams
    cfg = get_config("v2_smoke")
    env = DashEnvV2(cfg, n_envs=4)
    check("policy variant: actor 387 / obs 412 / action 50", (env.actor_dim, env.obs_dim, env.action_dim) == (387, 412, 50),
          f"{env.actor_dim}/{env.obs_dim}/{env.action_dim}")
    lib = get_config("v2_lib_s2_free")
    lib.n_envs = 4
    envl = DashEnvV2(lib, n_envs=4)
    check("library variant: actor 363 / obs 388 / action 9", (envl.actor_dim, envl.obs_dim, envl.action_dim) == (363, 388, 9),
          f"{envl.actor_dim}/{envl.obs_dim}/{envl.action_dim}")
    test_heading(env, cfg)
    if quick:
        return
    params = EnvParams.final(cfg)._replace(dr_scale=0.5, pitch_assist=1.0)
    state, obs = env.reset(jax.random.PRNGKey(0), params)
    check("commit flag = 1 at t0", bool((obs[:, env.wrap_index] == 1.0).all()))
    a0 = jax.random.uniform(jax.random.PRNGKey(1), (4, 50), minval=-1, maxval=1)
    state, obs, r, d, info = env.step(state, a0, params)
    check("spec latched from the t0 action", bool(jnp.allclose(state.spec, a0[:, :44])))
    check("mask == commit (t0)", bool(info["commit"].all()))
    a1 = -a0
    state, obs, r, d, info = env.step(state, a1, params)
    held = bool(jnp.allclose(state.spec, a0[:, :44])) or bool(info["commit"].any())
    check("spec NOT rewritten off-commit", held)
    t = time.time()
    n_done, n_commit = 0, 0
    for i in range(80):
        k = jax.random.PRNGKey(10 + i)
        a = jax.random.uniform(k, (4, 50), minval=-1, maxval=1)
        state, obs, r, d, info = env.step(state, a, params)
        n_done += int(d.sum())
        n_commit += int((info["commit"] & (state.step_n > 1)).sum())
    obs.block_until_ready()
    check("80 random ticks: finite obs, finite rewards", bool(jnp.isfinite(obs).all()) and bool(jnp.isfinite(r).all()),
          f"{80 * 4 / (time.time() - t):.0f} env-steps/s on {jax.devices()[0].platform}, {n_done} auto-resets")
    check("commit ticks appeared mid-episode (the clock wraps)", n_commit >= 1, f"{n_commit} commits after t0")


def test_net():
    print("net (masked log-prob)")
    mu = jnp.zeros((4, 50))
    ls = jnp.zeros(50)
    a = jax.random.normal(jax.random.PRNGKey(0), (4, 50))
    commit = jnp.array([False, True, False, True])
    latched = np.zeros(50, bool)
    latched[:44] = True
    m = nets.dim_mask(commit, latched)
    lp = nets.log_prob(mu, ls, a, m)
    a2 = a.at[:, :44].add(3.0)
    lp2 = nets.log_prob(mu, ls, a2, m)
    d = np.asarray(lp2 - lp)
    check("latched dims invisible off-commit", np.allclose(d[[0, 2]], 0.0) and (np.abs(d[[1, 3]]) > 1e-3).all())
    g = jax.grad(lambda ls_: nets.log_prob(mu, ls_, a, m)[jnp.array([0, 2])].sum())(ls)
    check("no gradient to latched log_std off-commit", bool(jnp.allclose(g[:44], 0.0)) and bool((jnp.abs(g[44:]) > 0).any()))
    ent = nets.entropy(ls, m)
    check("entropy masked the same way", abs(float(ent[0]) - 6 * (0.5 + 0.5 * np.log(2 * np.pi))) < 1e-5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    test_gait()
    test_thermal()
    test_drive()
    test_plant()
    test_net()
    test_env(args.quick)
    print(f"\n{'ALL OK' if OK == 0 else f'{OK} FAILURE(S)'}")
    sys.exit(1 if OK else 0)


if __name__ == "__main__":
    main()
