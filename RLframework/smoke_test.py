"""walk_v4 smoke test: the invariants the artifact makes checkable, run before any GPU hour.

    python RLframework/smoke_test.py            # everything (a few minutes on CPU)
    python RLframework/smoke_test.py --quick    # no env stepping (seconds)

Checks:
  gait     numpy == jax bit-level agreement; knobs-zero mirror identity u_L(phi) + u_R(phi+pi) = 0;
           mirror(action) reproduces the mirrored trajectory; impedance exp map ranges; sign rule
  thermal  the two anchors: peak (170/55) from cold reaches the limit at 5 s with tau_th = 45 s;
           1.0x continuous reaches 52% at 33 s; the third back-to-back dash exceeds 0.85
  drive    delay selection at substep granularity; torque-speed clamp closes to no-load speed
  plant    the XML loads in MJX; the keyframe is the solved flat-foot stance
  env      obs layout 384/409 (policy) and 363/388 (library); commit flag at t0; spec latches from
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
    cfg = get_config("dash")
    p = gait.GaitParams.from_cfg(cfg)
    rng = np.random.default_rng(0)
    nominal = np.array([0, 0, 0.12, 0, 0, -0.12])
    S = gait.SPEC_DIM
    a = rng.uniform(-1, 1, gait.ACTION_DIM)
    t_np = gait.assemble(a[:S], a[S:], 1.3, nominal, p, xp=np)
    t_jx = gait.assemble(jnp.asarray(a[:S]), jnp.asarray(a[S:]), jnp.float32(1.3),
                         jnp.asarray(nominal), p)
    # RELATIVE: the tuple is (target, kp, kd, q_ref) and the gains run to several hundred, so an
    # absolute bar on the max over all four is really a bar on kp -- at float32 epsilon it is noise
    err = max(float(np.abs(np.asarray(x) - y).max() / max(1.0, np.abs(y).max())) for x, y in zip(t_jx, t_np))
    check("numpy == jax (float32 rounding)", err < 1e-6, f"max relative diff {err:.1e}")
    s = a[:S].copy()
    s[list(gait.KNOB_IDX)] = 0
    worst = 0.0
    for ph in np.linspace(0, 2 * np.pi, 9):
        qL = gait.feedforward(s, ph, nominal, p, xp=np) - nominal
        qR = gait.feedforward(s, ph + np.pi, nominal, p, xp=np) - nominal
        worst = max(worst, float(np.abs(qL[[1, 2, 0]] + qR[[4, 5, 3]]).max()))
    check("knobs 0: u_L(phi) + u_R(phi+pi) = 0", worst < 1e-12, f"{worst:.1e}")
    sm = np.asarray(gait.mirror_action(jnp.asarray(a)))[:S].astype(np.float64)
    delta = p.delta_max * np.clip(a[gait.I_DELTA], -1, 1)
    worst = 0.0
    for ph in np.linspace(0, 2 * np.pi, 7):
        q = gait.feedforward(a[:S], ph, nominal, p, xp=np) - nominal
        qm = gait.feedforward(sm, ph - np.pi - delta, nominal, p, xp=np) - nominal
        worst = max(worst, float(np.abs(qm + q[gait.MIRROR_PERM]).max()))
    check("mirror(action) = mirrored trajectory at phi - pi - Delta", worst < 1e-6, f"{worst:.1e}")
    one = np.zeros(S)
    one[21] = 1.0
    kp, _ = gait.impedance(one, 0.0, p, xp=np)
    check("kp profile +1 -> x2.5", abs(kp[1] / 200.0 - 2.5) < 1e-6, f"{kp[1]:.1f}")
    one[21] = -1.0
    kp, _ = gait.impedance(one, 0.0, p, xp=np)
    check("kp profile -1 -> /3", abs(kp[1] * 3.0 / 200.0 - 1.0) < 1e-6, f"{kp[1]:.1f}")
    one = np.zeros(S)
    one[28] = 1.0
    _, kd = gait.impedance(one, 0.0, p, xp=np)
    check("kd soften-only: +1 -> x1", abs(kd[1] - 5.0) < 1e-6, f"{kd[1]:.2f}")
    _, kd = gait.impedance(-one, 0.0, p, xp=np)
    check("kd -1 -> /4", abs(kd[1] * 4.0 - 5.0) < 1e-6, f"{kd[1]:.2f}")
    # sign rule: o_cam (+,+) is a fore-aft split -> opposite joint signs on mirrored axes
    o = np.zeros(S)
    o[gait.I_O.start] = 1.0
    q = gait.feedforward(o, 0.0, nominal, p, xp=np) - nominal
    check("o_cam enters (+,+) in joint space", q[1] > 0 and q[4] > 0, f"cam_L {q[1]:+.3f} cam_R {q[4]:+.3f}")
    check("the action is 47 wide: spec 41 + residual 6, no reflex dims",
          (gait.SPEC_DIM, gait.ACTION_DIM) == (41, 47), f"{gait.SPEC_DIM} + 6 = {gait.ACTION_DIM}")
    lo, hi = get_config("dash").gait_freq_hz
    check(f"freq map [{lo}, {hi}] Hz",
          abs(gait.frequency(-1.0, p, np) - lo) < 1e-9 and abs(gait.frequency(1.0, p, np) - hi) < 1e-9,
          f"{gait.frequency(-1.0, p, np):.3f}..{gait.frequency(1.0, p, np):.3f}")


def test_presets():
    """One recipe, three presets. The control law carries no reflexes and no assists, the action is
    47 wide, and the per-joint residual authority is what the preset asked for."""
    print("presets: the recipe, the planar probe, the smoke config")
    from dataclasses import asdict
    rng = np.random.default_rng(1)
    nominal = np.array([0, 0, 0.12, 0, 0, -0.12])
    names = sorted(get_config.__globals__["PRESETS"])
    check("three presets: the recipe, a planar probe, a smoke config",
          names == ["dash", "dash_planar", "smoke"], str(names))
    # the averaged heading: wobble and a start transient cost little, a held offset is billed in full
    from env import heading_ema
    ah, dt_h = float(np.exp(-0.01 / 1.0)), 0.01

    def billed(yaws):
        h, acc = 0.0, 0.0
        for y in yaws:
            h = float(heading_ema(jnp.float32(h), jnp.float32(y), ah, np.pi / 2))
            acc += h * h
        return acc / len(yaws), h
    n = 1000                                             # 10 s at 100 Hz
    tt = np.arange(n) * dt_h
    wob, _ = billed(0.2 * np.sign(np.sin(2 * np.pi * 3.0 * tt)))          # +-0.2 rad at the stride rate
    check("heading avg: a +-0.2 rad stride wobble bills < 2% of the instantaneous term", wob < 0.02 * 0.2 ** 2,
          f"{wob / 0.2 ** 2:.3%}")
    ini, _ = billed(np.where(tt < 0.3, 0.3, 0.0))                          # 0.3 rad for the first 0.3 s
    check("heading avg: a 0.3 s start transient bills < 1/4 of the instantaneous term", ini < 0.25 * 0.3 ** 2 * 0.03,
          f"{ini / (0.3 ** 2 * 0.03):.2f} of it")
    _, held = billed(np.full(300, 0.1))
    check("heading avg: a held 0.1 rad offset is at 95% after 3 s", held > 0.095, f"{held:.4f}")
    # the heading-rate estimator, against a body held at fixed roll/pitch turning at r about WORLD z
    from env import heading_rate
    r, worst_e, worst_z = 0.5, 0.0, 0.0
    for roll, pitch in [(0.0, 0.0), (0.1, -0.05), (-0.2, 0.3), (0.35, 0.2)]:
        cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
        Rb = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        w_b, g_b = Rb.T @ np.array([0.0, 0.0, r]), Rb.T @ np.array([0.0, 0.0, -1.0])
        worst_e = max(worst_e, abs(float(heading_rate(jnp.asarray(w_b), jnp.asarray(1.3 * g_b), True)) - r))
        worst_z = max(worst_z, abs(float(heading_rate(jnp.asarray(w_b), jnp.asarray(g_b), False)) - r * cr * cp))
    check("heading rate (euler): = the world yaw rate at any roll/pitch, any |g|", worst_e < 1e-5, f"{worst_e:.1e}")
    check("heading rate (v3): body gyro z = r cos(roll) cos(pitch), the drift source", worst_z < 1e-6, f"{worst_z:.1e}")
    for name in names:
        cfg = get_config(name)
        p = gait.GaitParams.from_cfg(cfg)
        check(f"{name}: no lane bill, heading on the averaged estimate",
              cfg.w_lane == 0.0 and cfg.heading_avg_s > 0 and cfg.heading_euler,
              f"w_lane {cfg.w_lane}, tau {cfg.heading_avg_s} s")
        check(f"{name}: the stick and the income cap agree, so the policy plateaus at its own",
              cfg.v_max == cfg.v_ceiling == 4.0 and cfg.w_speed_income > 0,
              f"v_max {cfg.v_max} v_ceiling {cfg.v_ceiling} w_speed {cfg.w_speed_income}")
        # the scaffolding the flat foot removed: no base spring anywhere, no bring-up
        check(f"{name}: no training-wheel scaffolding left in the config",
              not any("spring" in k or "assist" in k or "reflex" in k for k in asdict(cfg))
              and not cfg.bringup_enable and not cfg.hold_enable,
              f"bringup {cfg.bringup_enable}, hold {cfg.hold_enable}")
        # a curriculum name that is never registered freezes everything queued behind it
        flat_order = [x for g in cfg.curriculum_order for x in ((g,) if isinstance(g, str) else g)]
        check(f"{name}: every queued curriculum is a real knob",
              all(hasattr(cfg, k) or k in ("cmd_lo", "cmd_hi", "shape_scale", "dr_scale",
                                           "cmd_zero_p", "eff_scale", "stance_ratio",
                                           "ctrl_jitter_ms", "ctrl_drop_prob")
                  for k in flat_order), str(flat_order))
        check(f"{name}: the control law has no reflex parameters left",
              not any(k.startswith(("reflex_", "pitch_k", "pitch_b")) for k in p._asdict()),
              str([k for k in p._asdict()]))
        check(f"{name}: residual +-0.20 on hips/thighs, +-0.10 on cams",
              p.residual_scale == (0.20, 0.10, 0.20, 0.20, 0.10, 0.20), str(p.residual_scale))
        S = gait.SPEC_DIM
        a = rng.uniform(-1, 1, gait.ACTION_DIM)
        # the residual reaches the target with per-joint authority, and only there
        r = np.zeros(6)
        r[2] = 1.0
        base = gait.assemble(a[:S], np.zeros(6), 1.3, nominal, p, xp=np)[0]
        t1 = gait.assemble(a[:S], r, 1.3, nominal, p, xp=np)[0]
        moved = t1 - base
        check(f"{name}: residual unit on thigh_L moves it by 0.20 rad and nothing else",
              abs(moved[2] - 0.20) < 1e-12 and np.abs(np.delete(moved, 2)).max() == 0.0, str(moved))
        tj = gait.assemble(jnp.asarray(a[:S]), jnp.asarray(r), jnp.float32(1.3),
                           jnp.asarray(nominal), p)[0]
        err = float(np.abs(np.asarray(tj) - t1).max())
        check(f"{name}: numpy == jax with the per-joint scale", err < 1e-5, f"{err:.1e}")
    # a scalar residual_scale still broadcasts, so a config may set one number for all six
    from dataclasses import replace as _replace
    one = _replace(get_config("dash"), residual_scale=0.15)
    p2 = gait.GaitParams.from_cfg(one)
    check("scalar residual_scale broadcasts to 6", p2.residual_scale == (0.15,) * 6,
          str(p2.residual_scale))


def test_objective():
    """The two objective-level fixes of 2026-09-16: an income that keeps paying for speed, and a
    time limit that is not a death."""
    print("objective: speed income and time limits")
    from env import joystick_income
    from ppo import Transition, gae_fn
    ji = lambda vx, cmd, w: float(joystick_income(jnp.float32(vx), jnp.float32(cmd), 0.6,
                                                  3.0, 2.0, w, 4.0))
    # below the command the monotone term adds exactly w / v_ceiling per m/s, whatever the kernel does
    d_off = ji(2.0, 4.0, 0.0) - ji(1.0, 4.0, 0.0)
    d_on = ji(2.0, 4.0, 3.0) - ji(1.0, 4.0, 3.0)
    check("speed income: adds w/v_ceiling = 0.75 per m/s under an out-of-reach command",
          abs((d_on - d_off) - 0.75) < 1e-5, f"{d_on - d_off:.3f} (kernel alone {d_off:.3f})")
    peak = [ji(v, 2.0, 3.0) for v in (1.8, 2.0, 2.2)]
    check("speed income: income still PEAKS at the command (overspeed pays less)",
          peak[1] > peak[0] and peak[1] > peak[2], " ".join(f"{x:.2f}" for x in peak))
    check("speed income: pays nothing for standing still under a zero command",
          ji(0.0, 0.0, 3.0) == ji(0.0, 0.0, 0.0), f"{ji(0.0, 0.0, 3.0):.3f}")

    one = lambda x: jnp.full((1, 1), float(x))
    def adv_of(trunc):
        tr = Transition(obs=one(0), action=one(0), log_prob=one(0), reward=one(1.0),
                        done=jnp.full((1, 1), True), value=one(0.0), mask=one(1),
                        trunc=jnp.full((1, 1), trunc), v_final=one(10.0))
        adv, _ = gae_fn(tr, jnp.zeros((1,)), 0.995, 0.95)
        return float(adv[0, 0])
    check("GAE: a TIME LIMIT bootstraps from the value of the state it cut",
          abs(adv_of(True) - (1.0 + 0.995 * 10.0)) < 1e-4, f"{adv_of(True):.3f}")
    check("GAE: a FALL does not bootstrap", abs(adv_of(False) - 1.0) < 1e-4, f"{adv_of(False):.3f}")


def test_heading_env(quick):
    """Heading bills the averaged ESTIMATE (the wiring; the averaging itself is unit-tested in
    test_presets): hold the robot's own heading at 0.3 rad and read the term back every tick."""
    if quick:
        return
    print("v4: the averaged heading in the env")
    from env import DashEnvV2
    from ppo import initial_params
    cfg = get_config("dash")
    env = DashEnvV2(cfg, n_envs=4)
    p0 = initial_params(cfg)
    prm = p0._replace(dr_scale=0.0)
    st, _ = env.reset(jax.random.PRNGKey(4), prm)
    a_h = float(np.exp(-env.control_dt / cfg.heading_avg_s))
    worst, first = 0.0, None
    for _ in range(50):
        st = st.replace(yaw_est=jnp.full_like(st.yaw_est, 0.3))
        st, _, _, done, info = env.step(st, jnp.zeros((4, env.action_dim)), prm)
        first = float(st.heading_avg[0]) if first is None else first
        want = jnp.maximum(-cfg.w_heading * st.heading_avg ** 2, -cfg.penalty_term_cap)
        worst = max(worst, float(jnp.abs(jnp.where(done, 0.0, info["reward_terms"]["heading"] - want)).max()))
    check("heading_avg takes the robot's own estimate: (1 - a) * 0.3 after one tick",
          abs(first - (1.0 - a_h) * 0.3) < 1e-6, f"{first:.5f}")
    check("heading term = -w * heading_avg^2 (not the true yaw)", worst < 1e-5, f"{worst:.1e}")


def test_thermal():
    print("thermal (tau_th 45 s, single node)")
    cfg = get_config("dash")
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
    cfg = get_config("dash")
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
    for preset in ("dash", "dash_planar"):
        cfg = get_config(preset)
        pl = Plant(cfg)
        check(f"{preset}: MJX model, nq={pl.nq} nu=6", pl.nu == 6)
        # The stance the model builder solved: sole flat and the centre of mass mid-pad.
        check(f"{preset}: stance height 0.838 m (flat-foot, solved)",
              abs(pl.height_stand - 0.838) < 0.01, f"{pl.height_stand:.4f}")
        # The gait is centred on the COMMAND that holds the stance, not on the pose it holds: a
        # position loop makes torque only from error, so the two must differ.
        off = np.degrees(np.abs(pl.nominal_ctrl - pl.nominal_pose)).max()
        check(f"{preset}: nominal is the holding COMMAND, a few deg off the pose", 2.0 < off < 10.0,
              f"{off:.2f} deg")


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
    cfg = get_config("smoke")
    env = DashEnvV2(cfg, n_envs=4)
    # The actor's width is the invariant that matters: the critic's privileged tail grew by the
    # two sole-pad contact bits per foot, and the actor must NOT have grown with it -- DASH-01 has
    # no foot contact sensing, so a contact bit in the actor's input is a sensor the robot does
    # not have.
    check("actor 384 / obs 413 / action 47",
          (env.actor_dim, env.obs_dim, env.action_dim) == (384, 413, 47),
          f"{env.actor_dim}/{env.obs_dim}/{env.action_dim}")
    check("the pad contact bits are PRIVILEGED: they are in obs, not in the actor's slice",
          env.obs_dim - env.actor_dim == 29, f"priv tail {env.obs_dim - env.actor_dim}")
    test_heading(env, cfg)
    if quick:
        return
    params = EnvParams.final(cfg)._replace(dr_scale=0.5)
    state, obs = env.reset(jax.random.PRNGKey(0), params)
    check("commit flag = 1 at t0", bool((obs[:, env.wrap_index] == 1.0).all()))
    a0 = jax.random.uniform(jax.random.PRNGKey(1), (4, gait.ACTION_DIM), minval=-1, maxval=1)
    state, obs, r, d, info = env.step(state, a0, params)
    check("spec latched from the t0 action", bool(jnp.allclose(state.spec, a0[:, :gait.SPEC_DIM])))
    check("mask == commit (t0)", bool(info["commit"].all()))
    a1 = -a0
    state, obs, r, d, info = env.step(state, a1, params)
    held = bool(jnp.allclose(state.spec, a0[:, :gait.SPEC_DIM])) or bool(info["commit"].any())
    check("spec NOT rewritten off-commit", held)
    t = time.time()
    n_done, n_commit = 0, 0
    for i in range(80):
        k = jax.random.PRNGKey(10 + i)
        a = jax.random.uniform(k, (4, gait.ACTION_DIM), minval=-1, maxval=1)
        state, obs, r, d, info = env.step(state, a, params)
        n_done += int(d.sum())
        n_commit += int((info["commit"] & (state.step_n > 1)).sum())
    obs.block_until_ready()
    check("80 random ticks: finite obs, finite rewards", bool(jnp.isfinite(obs).all()) and bool(jnp.isfinite(r).all()),
          f"{80 * 4 / (time.time() - t):.0f} env-steps/s on {jax.devices()[0].platform}, {n_done} auto-resets")
    # The clock must wrap mid-episode, or the latched spec is written once and never revised.
    # Driven with ZERO actions, not random ones: a zero action is the neutral gait at the middle
    # of the cadence band, so the wrap is due on a known tick -- with random actions the robot
    # thrashes itself over in a dozen ticks and whether a wrap lands first is luck, which made
    # this check pass on one machine and fail on another.
    f_neutral = float(gait.frequency(0.0, env.gp, np))
    ticks = int(3.0 / (f_neutral * env.control_dt))          # three full cycles
    # nominal plant: dr_scale 0 also switches off the pushes, wind and trips, which ride on it
    clean = EnvParams.final(cfg)._replace(dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0)
    state2, _ = env.reset(jax.random.PRNGKey(3), clean)
    zero, n_commit2 = jnp.zeros((4, gait.ACTION_DIM)), 0
    for _ in range(ticks):
        state2, _, _, _, info2 = env.step(state2, zero, clean)
        n_commit2 += int((info2["commit"] & (state2.step_n > 1)).sum())
    check(f"the clock wraps mid-episode ({f_neutral:.2f} Hz neutral, {ticks} ticks = 3 cycles)",
          n_commit2 >= 1, f"{n_commit2} commits after t0 over 4 envs")


def test_net():
    print("net (masked log-prob)")
    A, S = gait.ACTION_DIM, gait.SPEC_DIM
    mu = jnp.zeros((4, A))
    ls = jnp.zeros(A)
    a = jax.random.normal(jax.random.PRNGKey(0), (4, A))
    commit = jnp.array([False, True, False, True])
    latched = np.zeros(A, bool)
    latched[:S] = True
    m = nets.dim_mask(commit, latched)
    lp = nets.log_prob(mu, ls, a, m)
    a2 = a.at[:, :S].add(3.0)
    lp2 = nets.log_prob(mu, ls, a2, m)
    d = np.asarray(lp2 - lp)
    check("latched dims invisible off-commit", np.allclose(d[[0, 2]], 0.0) and (np.abs(d[[1, 3]]) > 1e-3).all())
    g = jax.grad(lambda ls_: nets.log_prob(mu, ls_, a, m)[jnp.array([0, 2])].sum())(ls)
    check("no gradient to latched log_std off-commit", bool(jnp.allclose(g[:S], 0.0)) and bool((jnp.abs(g[S:]) > 0).any()))
    ent = nets.entropy(ls, m)
    check("entropy masked the same way", abs(float(ent[0]) - 6 * (0.5 + 0.5 * np.log(2 * np.pi))) < 1e-5)


def test_queue():
    """The curriculum queue -- the mechanism the whole recipe rests on, tested without a GPU.

    Two properties, both of which have been wrong in this folder and cost runs:
      * a name NOT in the order advances freely (that is how dr_scale ramped during a stage 1 that
        was supposed to have none), and a name behind an unfinished group does not;
      * a group may not hold the queue forever. Every ramp is competence-gated and retreats, so
        without a cap one group starves the rest -- which is how five 200 M seeds finished with
        dr_scale at 0.000.
    """
    print("curriculum queue")
    from ppo import PPO
    from dataclasses import replace
    cfg = get_config("dash")

    class FakeQ:                       # just enough of PPO for _queued / _q
        _queued = PPO._queued
        _live_group = PPO._live_group
        def __init__(self, cfg):
            self.cfg, self.cur, self.step = cfg, {}, 0

    q = FakeQ(cfg)
    # The recipe's order is: widen the command band, then the gait-quality penalties, then a
    # different robot (DR), then a worse controller.  Drive it with the real group names -- a name
    # that is not in the order behaves differently, which is the first property below.
    # every member of a group must be present: a missing one reads as "not started" and holds
    # the queue, which is the mechanism, not a quirk of the fake
    def state(**progress):
        return {k: {"progress": v, "turn": 0.0} for k, v in progress.items()}

    q.cur = state(cmd_lo=0.4, cmd_hi=0.4, cmd_zero_p=0.4,
                  shape_scale=0.0, eff_scale=0.0, stance_ratio=0.0, dr_scale=0.0)
    check("the live group advances", q._queued("cmd_lo") is True)
    check("a group behind an unfinished one waits", q._queued("dr_scale") is False)
    check("a name not in the order is NOT frozen", q._queued("w_alive") is True,
          "(omitting a name does not disable it -- pin it at final instead)")

    for k in ("cmd_lo", "cmd_hi", "cmd_zero_p", "shape_scale", "eff_scale", "stance_ratio"):
        q.cur[k]["progress"] = 0.995
    check("a finished group hands over", q._queued("dr_scale") is True)

    q.cur["shape_scale"]["progress"] = 0.4
    check("and it waits again if that group retreats", q._queued("dr_scale") is False)
    for k in ("shape_scale", "eff_scale", "stance_ratio"):
        q.cur[k]["turn"] = cfg.curriculum_group_max_steps
    check("a group out of turn hands over anyway", q._queued("dr_scale") is True,
          f"(cap {cfg.curriculum_group_max_steps:,} steps)")

    # ...and the capped group itself STOPS when it hands over. Without this, the cap starts the
    # next group while the capped one keeps ramping: two curricula at once, which is what ended
    # the 2026-09-17 seeds (eff_scale 0.68 and still climbing when dr_scale was let in).
    check("a capped group stops advancing once it hands over", q._queued("eff_scale") is False)
    check("and the group that took over does advance", q._queued("dr_scale") is True)

    # ...and it gets the turn back once nothing else needs the queue, so the tail of the run
    # finishes what the cap cut short instead of leaving the budget unspent.
    for k in ("dr_scale", "ctrl_jitter_ms", "ctrl_drop_prob"):
        q.cur[k] = {"progress": 0.995, "turn": 0.0}
    check("a frozen group resumes when nothing else is live", q._queued("shape_scale") is True)

    q2 = FakeQ(replace(cfg, curriculum_group_max_steps=0))
    q2.cur = {k: {"progress": 0.995, "turn": 1e12} for k in ("cmd_lo", "cmd_hi", "cmd_zero_p")}
    q2.cur.update({k: {"progress": 0.4, "turn": 1e12}
                   for k in ("shape_scale", "eff_scale", "stance_ratio")})
    q2.cur["dr_scale"] = {"progress": 0.0}
    check("cap 0 means wait forever (the old behaviour)", q2._queued("dr_scale") is False)


def test_dr_floor():
    """DR must not be identically zero at step 0.

    Every 2026-09-17 seed trained 119 M steps on a plant that never varied, then collapsed from
    ep_len 2376 to 45 within 10 M steps of dr_scale first becoming nonzero -- at a dose of 0.088.
    A policy converged on one point in plant space has no margin to spend.
    """
    print("DR floor")
    from ppo import initial_params
    from env import EnvParams
    for name in ("dash", "dash_planar"):
        cfg = get_config(name)
        p0 = initial_params(cfg)
        check(f"{name}: the plant varies from the first rollout",
              p0.dr_scale > 0.0, f"dr_scale starts at {p0.dr_scale:.3f}")
        check(f"{name}: and the ramp still reaches full DR",
              EnvParams.final(cfg).dr_scale == 1.0)

    # NO CALIBRATION DR: the flat sole is a mechanical zero reference, so homing error is measured
    # rather than randomised. Asserted at the DRAW, not just the config, because joint_zero rides
    # two other multipliers (dr_scale and obs_noise_enable) and a config knob alone would not prove
    # the plant is clean.
    import jax
    import numpy as np
    from plant import Plant, draw_plant
    cfg = get_config("dash")
    pl = Plant(cfg)
    for s_ in (0.15, 1.0):
        dr = draw_plant(jax.random.PRNGKey(0), cfg, pl, s_)
        jz = np.abs(np.asarray(dr.joint_zero)).max()
        check(f"no homing error in the plant at dr_scale {s_:.2f}", jz == 0.0,
              f"max |joint_zero| {jz:.3e} rad")
    dr = draw_plant(jax.random.PRNGKey(0), cfg, pl, 1.0)
    check("the IMU mount angle is still randomised",
          float(np.abs(np.asarray(dr.imu_R) - np.eye(3)).max()) > 0.0,
          "(the sole pins the joints, not the IMU)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    test_gait()
    test_presets()
    test_objective()
    test_thermal()
    test_drive()
    test_plant()
    test_net()
    test_queue()
    test_dr_floor()
    test_env(args.quick)
    test_heading_env(args.quick)
    print(f"\n{'ALL OK' if OK == 0 else f'{OK} FAILURE(S)'}")
    sys.exit(1 if OK else 0)


if __name__ == "__main__":
    main()
