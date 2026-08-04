"""Fast sanity suite for the sprint training stack — run before every long training launch:

    python training/smoke_test.py

Checks (in ~30 s, no GPU needed): the self-contained model loads; every preset builds; obs/action
dims; a random-policy rollout stays finite; the sprint line latch + task-channel flip + stop-hold
termination + finish bonus; the phase-gated stance indicator's shape (continuity at the wrap,
antiphase overlap = flight window when stance_ratio < 0.5); the residual channel moves the
targets; the coef_rate phase gate is free at the cycle boundary; the fixed pitch reflex's sign /
clip / kwargs back-compat and its kick-arrest on the m3 plant; the angular-momentum reward term;
the VecNormalize warm-start rejuvenation; curriculum setters reach the env.
Exits non-zero on the first failure (safe to gate an sbatch on it).
"""
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np

import fourier_gait
from config import Config, PRESETS, get_config, config_from_dict, config_to_dict
from env import DashEnv

FAIL = 0


def check(name, ok, detail=""):
    global FAIL
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAIL += 1


def test_fourier_gait():
    print("fourier_gait:")
    N = 3
    check("dims", fourier_gait.action_dim(N) == 24 and fourier_gait.spec_dim(N) == 18)
    # steering is opt-in: 2 extra SPEC dims, and a bare call must still give the legacy 24
    check("dims (steer)", fourier_gait.action_dim(N, 2) == 26
          and fourier_gait.spec_dim(N, 2) == 20)
    # a zero steer command must reproduce the mirror-symmetric gait exactly
    cfg0 = Config()
    z6, zc = np.zeros(6), np.random.default_rng(0).uniform(-1, 1, 7)
    sym = fourier_gait.assemble(zc, zc, np.zeros(3), 0.9, 0.0, 0.0, z6, cfg0)
    zero_steer = fourier_gait.assemble(zc, zc, np.zeros(3), 0.9, 0.0, 0.0, z6, cfg0,
                                       steer=np.zeros(2))
    check("steer=0 == no steer", np.allclose(sym, zero_steer))
    # stance indicator: bounded, ~1 mid-stance, ~0 mid-swing, continuous at the phase wrap
    sr = 0.6
    I = lambda p: fourier_gait.stance_indicator(p, sr)
    mid_st = I(np.pi * sr)                    # middle of the stance window [0, 2*pi*sr)
    mid_sw = I(np.pi * (1 + sr))              # middle of the swing window [2*pi*sr, 2*pi)
    check("mid-stance ~1", mid_st > 0.99, f"{mid_st:.3f}")
    check("mid-swing ~0", mid_sw < 0.01, f"{mid_sw:.3f}")
    for b in (0.0, 2 * np.pi * sr):           # transitions centered on both boundaries
        check(f"boundary {b:.2f} = 0.5", abs(I(b) - 0.5) < 1e-6, f"{I(b):.3f}")
    wrap_gap = abs(I(2 * np.pi - 1e-6) - I(1e-6))
    check("continuous at wrap", wrap_gap < 1e-3, f"gap={wrap_gap:.4f}")
    vals = [fourier_gait.stance_indicator(p, 0.42) for p in np.linspace(0, 2 * np.pi, 500)]
    check("bounded [0,1]", min(vals) >= 0.0 and max(vals) <= 1.0)
    # flight window: with sr < 0.5 there must be phases where BOTH feet are expected in swing
    both_swing = [(1 - fourier_gait.stance_indicator(p, 0.42))
                  * (1 - fourier_gait.stance_indicator(p + np.pi, 0.42))
                  for p in np.linspace(0, 2 * np.pi, 500)]
    check("flight window exists at sr=0.42", max(both_swing) > 0.9, f"{max(both_swing):.3f}")
    both_walk = [(1 - fourier_gait.stance_indicator(p, 0.65))
                 * (1 - fourier_gait.stance_indicator(p + np.pi, 0.65))
                 for p in np.linspace(0, 2 * np.pi, 500)]
    check("no flight demanded at sr=0.65", max(both_walk) < 0.1, f"{max(both_walk):.3f}")
    # mirror/antiphase convention: with zero nominal/reflex, the right leg's deviation at phi
    # must equal MINUS the left leg's at phi+pi — the reward's stance_indicator(phi+pi) windows
    # assume exactly this; a sign regression here would train the right leg against the schedule
    cfg = Config()
    rng = np.random.default_rng(7)
    cam_c = rng.uniform(-1, 1, fourier_gait.per_joint(N))
    thigh_c = rng.uniform(-1, 1, fourier_gait.per_joint(N))
    zero6 = np.zeros(6)
    ok_mirror = True
    for phi in np.linspace(0, 2 * np.pi, 40):
        a = fourier_gait.assemble(cam_c, thigh_c, np.zeros(3), phi, 0.0, 0.0, zero6, cfg)
        b = fourier_gait.assemble(cam_c, thigh_c, np.zeros(3), phi + np.pi, 0.0, 0.0, zero6, cfg)
        if abs(a[fourier_gait.THIGH_R] + b[fourier_gait.THIGH_L]) > 1e-9 \
                or abs(a[fourier_gait.CAM_R] + b[fourier_gait.CAM_L]) > 1e-9:
            ok_mirror = False
            break
    check("right leg = mirrored antiphase left", ok_mirror)


def test_presets():
    print("presets:")
    for name in sorted(PRESETS):
        try:
            cfg = get_config(name)
            check(f"{name} builds", isinstance(cfg, Config))
        except Exception as e:
            check(f"{name} builds", False, repr(e))
    # config JSON round-trip preserves tuples
    cfg = get_config("m1_sprint")
    rt = config_from_dict(config_to_dict(cfg))
    check("config round-trip", rt.base_lock == cfg.base_lock and rt.z_rail_range == cfg.z_rail_range)


def test_env_basic():
    print("env (m2_sprint):")
    env = DashEnv(get_config("m2_sprint"))
    obs, _ = env.reset(seed=0)
    check("action dim 24", env.action_space.shape == (24,))
    check("obs dim 275", obs.shape == (55 * 5,), str(obs.shape))
    rng = np.random.default_rng(0)
    finite, floored = True, True
    for _ in range(150):
        a = rng.uniform(-1, 1, env.action_dim).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        if not (np.all(np.isfinite(obs)) and np.isfinite(r)):
            finite = False
            break
        # suicide-proofing: pre-terminal per-step total is floored (fall subtracts 100 after)
        if not term and r < -env.cfg.step_reward_floor - 1e-6:
            floored = False
        if term or trunc:
            obs, _ = env.reset()
    check("150 random steps finite", finite)
    check("per-step reward floored (non-terminal)", floored)
    # backward motion must pay NEGATIVE income (anti-shuttle): force vx < 0 and read the term
    env.reset(seed=3)
    env.data.qvel[0] = -1.0
    _, terms_bwd = env._reward(np.zeros(6, np.float32), np.array([True, True]))
    check("backward vx pays negative fwd_speed", terms_bwd["fwd_speed"] < -1.0,
          str(terms_bwd["fwd_speed"]))
    # anti-farm invariant: stop-phase income must be below the clock cost
    check("stop phase net-negative (w_stop_vel < w_time)", env.cfg.w_stop_vel < env.cfg.w_time)
    check("reward terms present", all(k in info["reward_terms"] for k in
          ("fwd_speed", "time", "phase_contact", "residual", "energy", "coef_rate")))
    # task channel: run phase = [1, dist/100]
    frame = env._proprio()
    check("task run flag", env._task[0] == 1.0 and 0.0 < env._task[1] <= 1.0,
          str(env._task))


def test_sprint_finish():
    print("sprint line latch + stop:")
    env = DashEnv(get_config("m1_sprint"))
    env.reset(seed=1)
    env.cfg.sprint_dist_m = 1.0        # takes effect on the next reset
    env.reset(seed=1)
    # teleport the base past the line, then hold still: latch -> task flip -> hold -> bonus
    env.data.qpos[0] = env._x0 + 2.0
    env.data.qvel[:] = 0.0
    a = np.zeros(env.action_dim, np.float32)
    fin_reward, finished = 0.0, False
    for _ in range(int(env.cfg.stop_hold_s / env.control_dt) + 10):
        obs, r, term, trunc, info = env.step(a)
        if env._sprint_crossed and not finished:
            check("task flips to stop", env._task[0] == 0.0 and env._task[1] == 0.0)
            finished = True
        if term:
            fin_reward = r
            break
    check("crossed latched", env._sprint_crossed)
    check("finish terminates", term and info["sprint"]["finished"])
    check("finish bonus paid", fin_reward > env.cfg.finish_bonus * 0.5, f"r={fin_reward:.1f}")


def test_residual_and_gate():
    print("residual channel + coef_rate gate:")
    cfg = get_config("m2_sprint")
    env = DashEnv(cfg)
    env.reset(seed=2)
    spec = np.zeros(env.action_dim, np.float32)
    a_res = spec.copy()
    a_res[env.spec_dim:] = 1.0
    env.step(spec)                     # flush the 1-step delay buffer
    env.step(spec)
    cmd_plain = env._prev_motor_cmd.copy()
    env.reset(seed=2)
    env.step(a_res)
    env.step(a_res)
    cmd_res = env._prev_motor_cmd.copy()
    d = float(np.max(np.abs(cmd_res - cmd_plain)))
    expect = cfg.residual_scale / cfg.action_scale
    check("residuals move targets by residual_scale", abs(d - expect) < 1e-5,
          f"d={d:.4f} expect={expect:.4f}")
    # coef_rate phase gate: a spec change applied MID-cycle must be billed; the same change
    # applied at the cycle boundary (phase 0) must be free. The 1-step delay buffer means the
    # spec sent at step k is applied at step k+1 — drive the phase by hand to test both gates.
    big = np.zeros(env.action_dim, np.float32)
    big[:env.spec_dim] = 1.0
    env.reset(seed=3)
    env.step(big)                      # buffers the big spec (applied action this step: zeros)
    env._phase = np.pi                 # next application lands mid-cycle -> sin^2(pi/2) = 1
    _, _, _, _, info = env.step(big)   # big spec applied now, prev applied was zeros
    check("coef_rate bills mid-cycle spec change", info["reward_terms"]["coef_rate"] < -0.5,
          str(info["reward_terms"]["coef_rate"]))
    env.reset(seed=3)
    env.step(big)
    env._phase = 0.0                   # next application lands exactly on the boundary -> free
    _, _, _, _, info = env.step(big)
    check("coef_rate free at cycle boundary", info["reward_terms"]["coef_rate"] == 0.0,
          str(info["reward_terms"]["coef_rate"]))
    # residual-only change is NEVER billed by coef_rate, at any phase
    env.reset(seed=3)
    res_only = np.zeros(env.action_dim, np.float32)
    res_only[env.spec_dim:] = 1.0
    env.step(res_only)
    env._phase = np.pi
    _, _, _, _, info = env.step(res_only)
    check("residual change not billed by coef_rate", info["reward_terms"]["coef_rate"] == 0.0,
          str(info["reward_terms"]["coef_rate"]))


def test_pitch_reflex():
    print("pitch reflex (fourier_gait.assemble):")
    cfg = Config()
    N = cfg.n_harmonics
    z_cam = np.zeros(fourier_gait.per_joint(N))
    z_thigh = np.zeros(fourier_gait.per_joint(N))
    z_reflex = np.zeros(3)
    nominal = np.zeros(6)
    # backward-compat: default pitch kwargs must reproduce a no-pitch call exactly
    a = fourier_gait.assemble(z_cam, z_thigh, z_reflex, 0.5, 0.0, 0.0, nominal, cfg)
    b = fourier_gait.assemble(z_cam, z_thigh, z_reflex, 0.5, 0.0, 0.0, nominal, cfg,
                              pitch=0.0, pitch_rate=0.0)
    check("default kwargs == no-pitch call", np.allclose(a, b))
    # sign: nose-down pitch (grav_x > 0) -> u_p < 0 -> thigh_L < 0 < thigh_R (feet forward),
    # symmetric magnitudes; roll/cam/hip_roll untouched by pitch
    out = fourier_gait.assemble(z_cam, z_thigh, z_reflex, 0.3, 0.0, 0.0, nominal, cfg, pitch=0.2)
    tL, tR = out[fourier_gait.THIGH_L], out[fourier_gait.THIGH_R]
    check("nose-down -> feet forward (thL<0<thR)", tL < 0 < tR)
    check("symmetric thigh offset", abs(tL + tR) < 1e-9, f"{tL:+.4f} {tR:+.4f}")
    check("cam/hip_roll unaffected by pitch",
          out[fourier_gait.CAM_L] == 0 and out[fourier_gait.HIP_ROLL_L] == 0)
    # opposite sign for nose-up
    out2 = fourier_gait.assemble(z_cam, z_thigh, z_reflex, 0.3, 0.0, 0.0, nominal, cfg, pitch=-0.2)
    check("nose-up -> feet backward (thL>0>thR)",
          out2[fourier_gait.THIGH_L] > 0 > out2[fourier_gait.THIGH_R])
    # clip saturation: huge pitch -> |offset| == pitch_clip exactly
    out3 = fourier_gait.assemble(z_cam, z_thigh, z_reflex, 0.3, 0.0, 0.0, nominal, cfg, pitch=10.0)
    check("clip saturates at pitch_clip",
          abs(abs(out3[fourier_gait.THIGH_L]) - cfg.pitch_clip) < 1e-9,
          f"{out3[fourier_gait.THIGH_L]:.4f}")
    # kd path: pure pitch_rate produces the right-sign offset
    out4 = fourier_gait.assemble(z_cam, z_thigh, z_reflex, 0.3, 0.0, 0.0, nominal, cfg,
                                 pitch=0.0, pitch_rate=1.0)
    check("kd path signed correctly", out4[fourier_gait.THIGH_L] < 0)


def test_pitch_reflex_plant():
    """Plant-level acceptance GATE. The m3 plant cannot stand passively (zero action collapses in
    height regardless of pitch — it needs an active gait), so the gate is NOT 'stands 20 s'; it is
    'the reflex ARRESTS a pitch kick markedly better than no reflex' — isolated over a short
    horizon where pitch dynamics dominate the slow height sag."""
    print("pitch reflex on the m3 plant (kick-arrest gate):")
    import mujoco
    from dataclasses import replace
    zero = np.zeros(24, np.float32)

    def end_pitch(kp, kd, clip, kick, horizon=40):
        cfg = replace(get_config("m3_speed"), pitch_kp=kp, pitch_kd=kd, pitch_clip=clip,
                      reset_joint_noise=0.0, push_interval_s=0.0)
        env = DashEnv(cfg)
        env.reset(seed=0)
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "base_pitch")
        env.data.qvel[int(env.model.jnt_dofadr[jid])] = kick   # nose-up/down pitch rate
        for _ in range(horizon):
            env.step(zero)
        return abs(float(env._gravity_body()[0]))

    d = get_config("m3_speed")
    for kick in (+1.2, -1.2):        # backward and forward kicks
        off = end_pitch(0.0, 0.0, 0.0, kick)
        on = end_pitch(d.pitch_kp, d.pitch_kd, d.pitch_clip, kick)
        check(f"reflex arrests kick {kick:+.1f} (on {on:.3f} < 0.6*off {off:.3f})",
              on < 0.6 * off, f"on={on:.3f} off={off:.3f}")


def test_pitch_assist():
    """The decaying pitch-assist training-wheel must RESTORE toward level (right sign) and be inert
    when disabled (kp=0) — a wrong-sign assist would drive the body over instead of catching it."""
    print("pitch-assist training-wheel (m2->m3 bridge):")
    from dataclasses import replace
    zero = np.zeros(24, np.float32)

    def end_pitch(scale, kick, horizon=40):
        cfg = replace(get_config("m3_speed"), pitch_assist_kp=150.0, pitch_assist_kd=15.0,
                      reset_joint_noise=0.0, push_interval_s=0.0)
        env = DashEnv(cfg)
        env.reset(seed=0)
        env.set_pitch_assist(scale)
        env.data.qvel[env._base_pitch_dadr] = kick     # nose-up/down pitch rate
        for _ in range(horizon):
            env.step(zero)
        return abs(float(env.data.qpos[env._base_pitch_qadr]))

    for kick in (+1.2, -1.2):        # both directions (on top of the fixed reflex, which is shared)
        off = end_pitch(0.0, kick)
        on = end_pitch(1.0, kick)
        check(f"assist arrests pitch kick {kick:+.1f} (on {on:.3f} < 0.6*off {off:.3f})",
              on < 0.6 * off, f"on={on:.3f} off={off:.3f}")
    # a preset with kp=0 (every non-assist preset) must NEVER write the pitch dof's qfrc_applied
    env = DashEnv(get_config("m3_speed"))
    env.reset(seed=0)
    env.set_pitch_assist(1.0)        # scale set, but kp=0 => still inert
    env.step(zero)
    check("kp=0 preset leaves pitch qfrc_applied at 0",
          env.data.qfrc_applied[env._base_pitch_dadr] == 0.0,
          str(env.data.qfrc_applied[env._base_pitch_dadr]))
    check("m3_assist preset enables assist", get_config("m3_assist").pitch_assist_kp == 150.0)
    check("m3_speed preset leaves assist off", get_config("m3_speed").pitch_assist_kp == 0.0)


def test_ankle_reflex():
    """The ankle-torque reflex (emulated actuated ankle) must RESTORE pitch (right sign) and be
    inert when disabled. Ankle joints resolved by spring stiffness; torque is stance-gated."""
    print("ankle-torque reflex (actuated-ankle emulation):")
    import mujoco
    from dataclasses import replace
    zero = np.zeros(24, np.float32)

    def end_pitch(kp, kd, clip, kick, horizon=45):
        cfg = replace(get_config("m3_reactive"), ankle_kp=kp, ankle_kd=kd, ankle_clip=clip,
                      reset_joint_noise=0.0, push_interval_s=0.0)
        env = DashEnv(cfg)
        env.reset(seed=0)
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "base_pitch")
        env.data.qvel[int(env.model.jnt_dofadr[jid])] = kick
        for _ in range(horizon):
            env.step(zero)
        return abs(float(env._gravity_body()[0]))

    a = get_config("m3_ankle")
    on_tot = off_tot = 0.0
    for kick in (+2.0, -2.0):        # nose-down and nose-up kicks
        off_tot += end_pitch(0.0, 0.0, 0.0, kick)
        on_tot += end_pitch(a.ankle_kp, a.ankle_kd, a.ankle_clip, kick)
    # a fixed reflex on this asymmetric stance-gated plant won't be perfect both ways, but must be
    # NET pitch-stabilizing (total end-|pitch| lower). Training is the real test of ankle authority.
    check(f"ankle reflex is net pitch-stabilizing (on {on_tot:.3f} < off {off_tot:.3f})",
          on_tot < off_tot, f"on={on_tot:.3f} off={off_tot:.3f}")
    env = DashEnv(get_config("m3_reactive"))      # ankle_kp=0 preset -> never writes ankle qfrc
    env.reset(seed=0)
    env.step(zero)
    check("ankle_kp=0 leaves ankle qfrc at 0",
          env.data.qfrc_applied[env._ankle_dadr[0]] == 0.0
          and env.data.qfrc_applied[env._ankle_dadr[1]] == 0.0)
    check("m3_ankle preset enables the ankle reflex", get_config("m3_ankle").ankle_kp > 0)


def test_ankle_stiffness():
    """Stiffer passive ankle spring must (a) set jnt_stiffness, (b) PRESERVE the standing preload
    torque k*(q_stand - springref) via a springref shift (so posture is unchanged, not slammed to
    the rest angle -> the robot would flip), and (c) keep a zero-action settle upright + finite."""
    print("ankle-spring stiffening (preload-preserving):")
    env = DashEnv(get_config("m3_stiff"))          # ankle_stiffness=200, ankle_damping=1.2
    aj = [j for j in range(env.model.njnt) if env.model.jnt_stiffness[j] > 0]
    check("stiffness applied (200)",
          all(abs(env.model.jnt_stiffness[j] - 200.0) < 1e-6 for j in aj))
    base = DashEnv(get_config("m3_reactive"))      # default ankle (k=28.65, ref +-0.7)
    ok_preload = True
    for j in aj:
        qadr = int(env.model.jnt_qposadr[j])
        q = float(env.default_qpos[qadr])
        pre_new = env.model.jnt_stiffness[j] * (q - env.model.qpos_spring[qadr])
        pre_old = base.model.jnt_stiffness[j] * (q - base.model.qpos_spring[qadr])
        if abs(pre_new - pre_old) > 1e-4:
            ok_preload = False
    check("standing preload preserved (springref shifted)", ok_preload)
    check("damping raised to 1.2",
          all(abs(env.model.dof_damping[int(env.model.jnt_dofadr[j])] - 1.2) < 1e-9 for j in aj))
    env.reset(seed=0)
    zero = np.zeros(env.action_dim, np.float32)
    ok = True
    for _ in range(100):
        o, r, te, tr, _ = env.step(zero)
        if not (np.all(np.isfinite(o)) and np.isfinite(r)):
            ok = False
            break
    check("stiff-ankle zero-action rollout finite", ok)
    check("stiff-ankle stays upright (no flip)", float(env._gravity_body()[2]) < -0.3,
          str(env._gravity_body()[2]))
    check("m3_stiff enables stiffness", get_config("m3_stiff").ankle_stiffness == 200.0)
    check("m3_reactive keeps default ankle", get_config("m3_reactive").ankle_stiffness == 0.0)


def test_foot_ahead():
    """foot-ahead-of-CoM reward: credited ONLY on a fresh touchdown, equal to w * sum over just-
    landed feet of clip(toe_x - com_x, 0, cap); exactly 0 when disabled or with no fresh touchdown."""
    print("foot-ahead-of-CoM reward (capture step):")
    env = DashEnv(get_config("m3_ahead"))          # w_foot_ahead=3.0
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.zeros(env.action_dim, np.float32))
    c = env.cfg
    com_x = float(env.data.subtree_com[0][0])
    toe_x = env.data.geom_xpos[env.foot_gids_arr, 0]
    env._grounded_prev = np.array([False, False])  # force a fresh double touchdown
    _, terms = env._reward(np.zeros(6, np.float32), np.array([True, True]))
    expect = c.w_foot_ahead * sum(min(max(float(toe_x[i] - com_x), 0.0), c.foot_ahead_cap_m)
                                  for i in range(2))
    check("touchdown credit matches spec", abs(terms["foot_ahead"] - expect) < 1e-6,
          f"got {terms['foot_ahead']:.4f} expect {expect:.4f}")
    check("credit within [0, 2*w*cap]",
          0.0 <= terms["foot_ahead"] <= c.w_foot_ahead * c.foot_ahead_cap_m * 2 + 1e-9)
    env._grounded_prev = np.array([True, True])     # already grounded -> no fresh touchdown
    _, terms2 = env._reward(np.zeros(6, np.float32), np.array([True, True]))
    check("no credit without a fresh touchdown", terms2["foot_ahead"] == 0.0, str(terms2["foot_ahead"]))
    env2 = DashEnv(get_config("m3_reactive"))       # disabled (w_foot_ahead=0)
    env2.reset(seed=0)
    env2.step(np.zeros(env2.action_dim, np.float32))  # init per-step state (_residual_sq etc.)
    env2._grounded_prev = np.array([False, False])
    _, terms3 = env2._reward(np.zeros(6, np.float32), np.array([True, True]))
    check("foot_ahead present + exactly 0 when disabled", terms3.get("foot_ahead") == 0.0,
          str(terms3.get("foot_ahead")))
    check("m3_ahead enables foot-ahead", get_config("m3_ahead").w_foot_ahead == 3.0)


def test_motor_limits():
    """The velocity/accel limiter caps the COMMANDED joint velocity (|d ctrl|/dt) and its rate; a
    preset without limits leaves the plant unclamped (byte-identical)."""
    print("motor velocity/acceleration limits:")
    c = get_config("m3_cad")
    check("m3_cad sets vel=22 accel=300", c.motor_vel_limit == 22.0 and c.motor_accel_limit == 300.0)
    env = DashEnv(c)
    check("limiter armed", env._vel_accel_limited)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    dt = env.control_dt
    prev, prev_v = env.data.ctrl.copy(), np.zeros(env.nu)
    vmax = amax = 0.0
    for _ in range(400):
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, env.action_dim).astype(np.float32))
        v = (env.data.ctrl - prev) / dt
        a = (v - prev_v) / dt
        vmax = max(vmax, float(np.max(np.abs(v))))
        amax = max(amax, float(np.max(np.abs(a))))
        prev, prev_v = env.data.ctrl.copy(), v
        if term or trunc:                       # reset resets the commanded state; skip the delta
            env.reset()
            prev, prev_v = env.data.ctrl.copy(), np.zeros(env.nu)
    check("commanded joint velocity <= 22 rad/s", vmax <= 22.0 + 1e-6, f"max {vmax:.2f}")
    check("commanded joint accel <= 300 rad/s^2", amax <= 300.0 + 1e-6, f"max {amax:.2f}")
    check("m3_reactive limiter off (unclamped)", not DashEnv(get_config("m3_reactive"))._vel_accel_limited)


def test_contact_switch():
    """The cadence penalty fires per foot that flips grounded<->airborne this step; 0 when disabled."""
    print("contact-switch cadence penalty:")
    env = DashEnv(get_config("m3_cad"))          # w_contact_switch=0.15
    env.reset(seed=0)
    env.step(np.zeros(env.action_dim, np.float32))
    env._grounded_prev = np.array([False, False])
    _, t = env._reward(np.zeros(6, np.float32), np.array([True, True]))     # both feet flip
    check("2 flips penalized (-0.30)", abs(t["step_rate"] - (-0.30)) < 1e-6, str(t["step_rate"]))
    env._grounded_prev = np.array([True, True])
    _, t2 = env._reward(np.zeros(6, np.float32), np.array([True, True]))    # no flip
    check("no flip -> 0", t2["step_rate"] == 0.0, str(t2["step_rate"]))
    env2 = DashEnv(get_config("m3_reactive"))     # disabled (w_contact_switch=0)
    env2.reset(seed=0)
    env2.step(np.zeros(env2.action_dim, np.float32))
    env2._grounded_prev = np.array([False, False])
    _, t3 = env2._reward(np.zeros(6, np.float32), np.array([True, True]))
    check("step_rate present + 0 when disabled", t3.get("step_rate") == 0.0, str(t3.get("step_rate")))
    check("m3_cad enables cadence penalty", get_config("m3_cad").w_contact_switch == 0.15)


def test_duty_sym():
    """The duty-symmetry penalty fires for a foot whose grounded-fraction EMA is below the floor
    (anti-one-legged); 0 when both feet share stance and when disabled."""
    print("duty-symmetry (anti-one-legged) penalty:")
    env = DashEnv(get_config("m3_sym_gait"))          # w_duty_sym=8.0, floor 0.30
    env.reset(seed=0)
    env.step(np.zeros(env.action_dim, np.float32))    # populate per-step state before _reward
    env._duty_ema[:] = [0.0, 0.5]                      # left foot never bears load
    _, t = env._reward(np.zeros(6, np.float32), np.array([True, True]))
    check("one-legged duty penalized (-2.0 cap)", abs(t["duty_sym"] - (-2.0)) < 1e-6, str(t["duty_sym"]))
    env._duty_ema[:] = [0.45, 0.45]                    # both feet share stance
    _, t2 = env._reward(np.zeros(6, np.float32), np.array([True, True]))
    check("symmetric duty -> 0", t2["duty_sym"] == 0.0, str(t2["duty_sym"]))
    env2 = DashEnv(get_config("m3_slow_gait"))         # disabled (w_duty_sym=0)
    env2.reset(seed=0)
    env2.step(np.zeros(env2.action_dim, np.float32))
    _, t3 = env2._reward(np.zeros(6, np.float32), np.array([True, True]))
    check("duty_sym present + 0 when disabled", t3.get("duty_sym") == 0.0, str(t3.get("duty_sym")))
    check("m3_sym_gait enables duty sym + 0.20 switch",
          get_config("m3_sym_gait").w_duty_sym == 8.0
          and get_config("m3_sym_gait").w_contact_switch == 0.20)


def test_workspace_kill():
    """The workspace-kill terminates when a foot's toe leaves the measured reachable box (sustained
    for the grace window); off by default and never fires at the stand pose."""
    print("workspace-kill termination:")
    env = DashEnv(get_config("m3_wskill_gait"))
    env.reset(seed=0)
    check("kill on + LUT nominal loaded", env.cfg.workspace_kill and env._ws_ref is not None)
    check("stand pose in-box -> no fire", not any(env._workspace_violation() for _ in range(60)))
    env._ws_out_t[:] = 0.0
    env._ws_ref = env._ws_ref + np.array([0.0, 0.0, 1.0])   # shift nominal 1 m -> foot reads OOB
    n = int(round(env.cfg.workspace_grace_s / env.control_dt))
    res = [env._workspace_violation() for _ in range(n + 4)]
    fire = res.index(True) if True in res else None
    check("OOB fires after ~grace", fire is not None and fire >= n - 1, str(fire))
    env2 = DashEnv(get_config("m3_sym_gait"))               # kill off
    env2.reset(seed=0)
    check("kill off -> ref None + no fire", env2._ws_ref is None and not env2._workspace_violation())


def test_torque_curriculum():
    """m7 torque-budget curriculum: freq remap, set_torque_limit scales forcerange (floor-clamped),
    torque_util reported in info; m7_freq isolates the freq fix (no curriculum)."""
    print("torque-budget curriculum + freq remap (m7):")
    c = get_config("m7")
    check("m7 gait_freq remapped to (0.5,4)", c.gait_freq_hz == (0.5, 4.0), str(c.gait_freq_hz))
    check("m7 torque_util_target 0.70", c.torque_util_target == 0.70)
    env = DashEnv(c)
    orig = env._orig_forcerange.copy()
    env.set_torque_limit(0.5)
    check("forcerange scaled to 0.5x", np.allclose(env.model.actuator_forcerange, orig * 0.5))
    check("scale stored", abs(env._torque_scale - 0.5) < 1e-9)
    env.set_torque_limit(0.01)                      # below floor
    check("floor clamps scale", env._torque_scale == c.torque_limit_floor)
    env.set_torque_limit(1.0)
    check("restore full torque", np.allclose(env.model.actuator_forcerange, orig))
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(env.action_dim, np.float32))
    check("torque_util in info, >=0", info.get("torque_util", -1) >= 0.0, str(info.get("torque_util")))
    check("m7_freq freq remapped", get_config("m7_freq").gait_freq_hz == (0.5, 4.0))
    check("m7_freq no torque curriculum", get_config("m7_freq").torque_util_target == 0.0)
    # neutral action (freq_raw=0) must now map to ~2 Hz, not ~25 Hz (the bug)
    import fourier_gait
    f_neutral = fourier_gait.frequency(0.0, c.gait_freq_hz)
    check("neutral action -> ~2 Hz cadence", 1.5 < f_neutral < 3.0, f"{f_neutral:.2f} Hz")


def test_hz200_timing():
    """200 Hz reactive stack: decimation/gamma/dt, rate-invariant reward scaling, and the sim2real
    timing randomization (jitter substeps + dropped-action hold) stay finite. 50 Hz is a no-op."""
    print("200 Hz reactive stack + sim2real timing:")
    for n in ("m2_reactive", "m3_reactive"):
        check(f"{n} builds", isinstance(get_config(n), Config))
    c = get_config("m3_reactive")
    check("control_decimation 5", c.control_decimation == 5)
    check("gamma 0.9975", abs(c.gamma - 0.9975) < 1e-9)
    check("gait_freq ceiling 50 Hz", c.gait_freq_hz == (0.5, 50.0))
    env = DashEnv(c)
    check("control_dt = 5 ms (200 Hz)", abs(env.control_dt - 0.005) < 1e-9, str(env.control_dt))
    check("reward_dt_scale = 0.25", abs(env._reward_dt_scale - 0.25) < 1e-9, str(env._reward_dt_scale))
    check("max_steps 12000 (60 s @ 200 Hz)", env.max_steps == 12000, str(env.max_steps))
    env.reset(seed=0)
    _, r, term, _, info = env.step(np.zeros(env.action_dim, np.float32))
    t = info["reward_terms"]
    expect = max(sum(t.values()) * env._reward_dt_scale,
                 -env.cfg.step_reward_floor * env._reward_dt_scale)
    if not term:
        check("reward = dt-scaled sum of terms (floored)", abs(r - expect) < 1e-5,
              f"r={r:.4f} expect={expect:.4f}")
    env.set_ctrl_jitter(4)
    env.set_ctrl_drop(0.5)
    check("jitter substeps set", env._ctrl_jitter_substeps == 4)
    check("drop prob set", env._ctrl_drop_prob == 0.5)
    rng = np.random.default_rng(0)
    finite = True
    for _ in range(200):
        o, rr, te, tr, _ = env.step(rng.uniform(-1, 1, env.action_dim).astype(np.float32))
        if not (np.all(np.isfinite(o)) and np.isfinite(rr)):
            finite = False
            break
        if te or tr:
            env.reset()
    check("jitter+drop rollout finite", finite)
    check("50 Hz preset reward_dt_scale == 1 (no-op)",
          abs(DashEnv(get_config("m2_sprint"))._reward_dt_scale - 1.0) < 1e-9)


def test_angmom_term():
    print("angular-momentum reward term:")
    from dataclasses import replace
    rng = np.random.default_rng(0)
    env = DashEnv(replace(get_config("m3_speed"), w_angmom=0.2))
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(env.action_dim, np.float32))
    t = info["reward_terms"]
    check("angmom present", "angmom" in t)
    check("angmom <= 0 and >= -cap", -env.cfg.penalty_term_cap <= t["angmom"] <= 0.0,
          str(t["angmom"]))
    check("angmom ~0 from near-rest first step", abs(t["angmom"]) < 0.2, str(t["angmom"]))
    # a spinning body pays; and w_angmom=0 => exactly 0
    for _ in range(20):
        env.step(rng.uniform(-1, 1, env.action_dim).astype(np.float32))
    env_off = DashEnv(replace(get_config("m3_speed"), w_angmom=0.0))
    env_off.reset(seed=0)
    _, _, _, _, info_off = env_off.step(np.zeros(env_off.action_dim, np.float32))
    check("angmom exactly 0 when w_angmom=0", info_off["reward_terms"]["angmom"] == 0.0)
    # m3+ presets enable it, m1/m2 leave it off
    check("m3_speed preset enables angmom", get_config("m3_speed").w_angmom == 0.2)
    check("m2_speed preset leaves angmom off", get_config("m2_speed").w_angmom == 0.0)


def test_rejuvenate_obs_rms():
    print("VecNormalize warm-start rejuvenation:")
    from train import rejuvenate_obs_rms
    from stable_baselines3.common.running_mean_std import RunningMeanStd

    class Dummy:
        pass
    d = Dummy()
    d.obs_rms = RunningMeanStd(shape=(275,))
    d.obs_rms.count = 1.8e8
    d.obs_rms.var[:] = 1.0
    d.obs_rms.var[18] = 1e-12                 # a rail-locked (pitch) dim
    d.obs_rms.mean[18] = 0.123
    d.ret_rms = None
    rejuvenate_obs_rms(d, 5e4, 1e-2)
    check("count capped", d.obs_rms.count == 5e4, str(d.obs_rms.count))
    check("tiny var floored", d.obs_rms.var[18] == 1e-2, str(d.obs_rms.var[18]))
    check("healthy var untouched", d.obs_rms.var[0] == 1.0)
    check("mean untouched", d.obs_rms.mean[18] == 0.123)


def test_curriculum_setters():
    print("curriculum setters:")
    env = DashEnv(get_config("m2_sprint"))
    env.reset(seed=0)
    env.set_stance_ratio(0.5)
    env.set_efficiency_scale(0.7)
    env.set_sprint_dist(42.0)
    env.reset(seed=0)
    check("stance ratio", env._stance_ratio == 0.5)
    check("eff scale", env._eff_scale == 0.7)
    check("sprint dist applies at reset", env._sprint_D == 42.0)


def test_m1_rail():
    print("m1 rail + ride-height LUT:")
    env = DashEnv(get_config("m1_sprint"))
    heights = set()
    for s in range(3):
        env.reset(seed=s)
        heights.add(round(float(env.data.qpos[2]), 4))
        check(f"reset {s} on-floor stance finite", np.all(np.isfinite(env.data.qpos)))
    check("ride height randomized", len(heights) > 1, str(heights))


def test_cpg_gait():
    """The CPG arm of the generator A/B (cpg_gait.py): oscillator dynamics, the measured foot-IK
    table, and the env plumbing that swaps generators."""
    print("cpg_gait:")
    import cpg_gait
    check("dims", cpg_gait.action_dim() == 14 and cpg_gait.spec_dim() == 8
          and cpg_gait.action_dim(residual=False) == 8 and cpg_gait.action_dim(2) == 16)

    cfg = get_config("ab_cpg_m2")
    # --- amplitude dynamics: r converges to mu, monotonically (critically damped, no overshoot)
    st = (np.zeros(2), np.zeros(2), np.array([0.0, np.pi]))
    mu = cpg_gait.amplitude_setpoint(np.array([0.5, 0.5]), cfg)
    f = cpg_gait.frequency(np.array([0.0, 0.0]), cfg.gait_freq_hz)
    rs = []
    for _ in range(400):
        st = cpg_gait.integrate(st, mu, f, 0.0, 0.005, cfg)
        rs.append(st[0][0])
    check("r converges to mu", abs(rs[-1] - mu[0]) < 1e-3, f"{rs[-1]:.4f} vs {mu[0]:.4f}")
    check("r does not overshoot", max(rs) <= mu[0] + 1e-6, f"max {max(rs):.4f}")
    check("r never negative", min(rs) >= 0.0)

    # --- phase coupling: legs pulled to antiphase from a BAD initial phase, and psi shifts it
    st = (np.full(2, 0.5), np.zeros(2), np.array([0.0, 0.3]))    # nearly in phase = wrong
    for _ in range(2000):
        st = cpg_gait.integrate(st, mu, f, 0.0, 0.005, cfg)
    d = float(np.mod(st[2][1] - st[2][0], 2 * np.pi))
    check("coupling restores antiphase", abs(d - np.pi) < 0.05, f"{d:.3f}")
    st = (np.full(2, 0.5), np.zeros(2), np.array([0.0, np.pi]))
    for _ in range(2000):
        st = cpg_gait.integrate(st, mu, f, 0.4, 0.005, cfg)
    d2 = float(np.mod(st[2][1] - st[2][0], 2 * np.pi))
    check("psi shifts the phase target", abs(d2 - (np.pi + 0.4)) < 0.05, f"{d2:.3f}")

    # --- swing bump agrees with the reward's stance window: no lift during expected stance
    for sr in (0.42, 0.5, 0.65):
        ph = np.linspace(0, 2 * np.pi, 400)
        lift = cpg_gait.swing_bump(ph, sr)
        stance = np.array([fourier_gait.stance_indicator(p, sr) for p in ph])
        check(f"no foot lift inside stance (sr={sr})", float(np.max(lift[stance > 0.9])) < 1e-9)
        mid = 2 * np.pi * sr + 0.5 * (2 * np.pi - 2 * np.pi * sr)      # analytic mid-swing
        check(f"lift peaks at 1 mid-swing (sr={sr})",
              abs(float(cpg_gait.swing_bump(mid, sr)) - 1.0) < 1e-9
              and float(lift.max()) <= 1.0 + 1e-9,
              f"peak {float(cpg_gait.swing_bump(mid, sr)):.6f}")

    # --- foot IK: the table is loadable, bounded, and (the bug that bit) CONTINUOUS along a stride
    lut = cpg_gait.load_lut()
    J = np.array([cpg_gait.foot_ik(cfg.cpg_stride * np.cos(t),
                                   cfg.cpg_clearance * cpg_gait.swing_bump(t, 0.5), lut)
                  for t in np.linspace(0, 2 * np.pi, 401)])
    check("IK output finite", bool(np.all(np.isfinite(J))))
    jump = float(np.abs(np.diff(J, axis=0)).max())
    # the 4-bar is redundant, so a per-cell inversion flips solution branches; the build's flood
    # fill must keep one branch. A branch flip showed up as a ~0.8 rad step in one sample.
    check("IK continuous along a stride", jump < 0.06, f"max step {jump:.4f} rad")
    check("IK stays inside the amp band", float(np.abs(J).max()) <= max(cfg.cam_amp, cfg.thigh_amp),
          f"max |joint| {float(np.abs(J).max()):.3f}")
    far = cpg_gait.foot_ik(99.0, 99.0, lut)     # saturates instead of extrapolating
    check("IK saturates out of box", bool(np.all(np.isfinite(far))))

    # every shipped IK table must be continuous and stay inside the joint clip at ITS arm's stride
    for preset in ("ab_cpg_m2", "ab_cpg_wide_m2"):
        c2 = get_config(preset)
        l2 = cpg_gait.load_lut(c2.cpg_lut)
        J2 = np.array([cpg_gait.foot_ik(c2.cpg_stride * np.cos(t),
                                        c2.cpg_clearance * cpg_gait.swing_bump(t, 0.5), l2)
                       for t in np.linspace(0, 2 * np.pi, 401)])
        j2 = float(np.abs(np.diff(J2, axis=0)).max())
        check(f"{preset} ({c2.cpg_lut}) IK continuous", j2 < 0.08, f"max step {j2:.4f} rad")
        # the point of the wide arm: saturate the SAME +-0.45 clip the fourier arm is free to use
        check(f"{preset} stride fits the joint envelope",
              float(np.abs(J2).max()) <= max(c2.cam_amp, c2.thigh_amp) + 1e-9,
              f"max |joint| {float(np.abs(J2).max()):.3f} vs {max(c2.cam_amp, c2.thigh_amp)}")
    check("wide arm uses more of the envelope than the default arm",
          float(np.abs(np.array([cpg_gait.foot_ik(0.315 * np.cos(t), 0.0,
                                                  cpg_gait.load_lut("cpg_foot_lut_wide.npz"))
                                 for t in np.linspace(0, 2 * np.pi, 200)])).max())
          > float(np.abs(J).max()))

    # --- env plumbing: widths differ per arm, obs matches, and the oscillator holds antiphase
    dims = {}
    for p in ("ab_f_m2", "ab_cpg_m2", "ab_cpg_nr_m2"):
        e = DashEnv(get_config(p))
        o, _ = e.reset(seed=0)
        dims[p] = (e.action_dim, e.observation_space.shape[0])
        check(f"{p} obs width matches space", o.shape[0] == e.observation_space.shape[0])
    check("cpg action is narrower than fourier", dims["ab_cpg_m2"][0] < dims["ab_f_m2"][0],
          str(dims))
    check("no-residual arm drops exactly 6 dims",
          dims["ab_cpg_m2"][0] - dims["ab_cpg_nr_m2"][0] == 6, str(dims))

    e = DashEnv(get_config("ab_cpg_m2"))
    e.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        _, _, term, trunc, _ = e.step(rng.uniform(-1, 1, e.action_dim))
        if term or trunc:
            break
    r_, _, th = e._cpg
    d = float(np.mod(th[1] - th[0], 2 * np.pi))
    check("oscillator stays antiphase under random actions", abs(d - np.pi) < 0.5, f"{d:.3f}")
    check("oscillator state finite", bool(np.all(np.isfinite(r_)) and np.all(np.isfinite(th))))
    # the no-residual arm must not be able to move the targets off the reconstruction
    e2 = DashEnv(get_config("ab_cpg_nr_m2"))
    e2.reset(seed=0)
    check("no-residual arm has no residual dims",
          e2.action_dim == cpg_gait.spec_dim(0), f"{e2.action_dim}")


def test_ankle_study():
    """Ankle-spring study: every arm must build, roll finite, and actually BE the arm it claims.

    The failure this guards against is not a crash — it is an arm that trains for 400 M steps
    while silently being a different arm (a 'rigid' run whose ankle still swings, an 'active' run
    with no ankle actuator, a k-sweep whose damping does not track k). That produces a clean,
    plausible, wrong curve, and nothing downstream would catch it."""
    print("ankle-spring study:")
    import os
    M = "model/dash01.xml"
    MA = "model/dash01_active.xml"
    have = all(os.path.exists(PKG_DIR / p) for p in (M, MA))
    if not have:
        check("study plants present", False, "run `python -m model.make_ankle_variants`")
        return

    # measured masses actually landed (15.14 kg, not the 12.83 kg CAD placeholder). Since the
    # correction is baked into the one plant, this now also guards every non-study preset: a CAD
    # regen that skipped model.apply_measured_masses would fail right here.
    e = DashEnv(Config(model_path=M, ankle_mode="passive"))
    check("measured-mass plant", abs(float(e.model.body_subtreemass[1]) - 15.136) < 0.01,
          f"{float(e.model.body_subtreemass[1]):.3f} kg")

    # each mode is what it says it is
    modes = {}
    for tag, kw in (("passive", dict(model_path=M, ankle_mode="passive")),
                    ("free", dict(model_path=M, ankle_mode="free")),
                    ("rigid", dict(model_path=M, ankle_mode="rigid")),
                    ("active", dict(model_path=MA, ankle_mode="active")),
                    ("active_spring", dict(model_path=MA, ankle_mode="active_spring",
                                           ankle_stiffness=350.0, ankle_zeta=0.7))):
        env = DashEnv(Config(**kw))
        o, _ = env.reset(seed=0)
        for _ in range(40):
            o, _, term, trunc, info = env.step(np.zeros(env.action_dim, np.float32))
            if term or trunc:
                break
        modes[tag] = (env, o, info)
        check(f"{tag} rolls finite", bool(np.isfinite(o).all()))

    check("free has NO spring", modes["free"][0].ankle_k == 0.0)
    # rigid: the welded ankle must not drift over a rollout (a soft equality that quietly gives way
    # would make "rigid" a mislabelled compliant arm)
    renv = modes["rigid"][0]
    renv.reset(seed=1)
    qs = float(renv.data.qpos[renv._ankle_qpos[0]])
    for _ in range(60):
        renv.step(np.zeros(renv.action_dim, np.float32))
    check("rigid ankle stays welded",
          abs(float(renv.data.qpos[renv._ankle_qpos[0]]) - qs) < 5e-3,
          f"drifted {float(renv.data.qpos[renv._ankle_qpos[0]]) - qs:+.4f} rad from {qs:+.4f}")
    check("free ankle DOES move (the null is a real null)",
          abs(modes["free"][2]["ankle_defl"]) > 0.05, f"{modes['free'][2]['ankle_defl']:.4f}")

    # active plant: 2 extra actuators, 2 extra action dims, wider obs, telemetry present
    aenv, ao, ainfo = modes["active"]
    penv, po, _ = modes["passive"]
    check("active plant has 2 ankle actuators", aenv.n_ankle_act == 2, f"{aenv.n_ankle_act}")
    check("active action is +2 dims", aenv.action_dim == penv.action_dim + 2)
    check("active obs is wider", ao.shape[0] > po.shape[0])
    check("active spec telemetry",
          all(k in ainfo for k in ("ankle_motor_trq", "ankle_motor_w", "ankle_motor_power",
                                   "ankle_motor_over_cont", "ankle_motor_util")))
    # the ankle motor is MASSLESS on purpose (upper bound: is a motor useful AT ALL, and at what
    # spec). If this ever fails, someone re-enabled the mass weld and the active arm silently
    # became "is THIS motor worth it" instead.
    check("ankle motor is massless",
          abs(float(aenv.model.body_subtreemass[1])
              - float(penv.model.body_subtreemass[1])) < 1e-6,
          f"{float(aenv.model.body_subtreemass[1]):.3f} vs {float(penv.model.body_subtreemass[1]):.3f}")
    check("ankle motor adds no rotor inertia",
          float(aenv.model.dof_armature[aenv._ankle_dof[0]])
          == float(penv.model.dof_armature[penv._ankle_dof[0]]))
    check("ankle peak torque is AKE90-class",
          abs(float(aenv._orig_forcerange[aenv.ankle_act_idx[0], 1]) - 170.0) < 1e-6,
          f"{float(aenv._orig_forcerange[aenv.ankle_act_idx[0], 1]):.1f} N*m")

    # torque-speed curve: available torque must fall to ~0 at the no-load speed, and be full at rest
    tsenv = DashEnv(Config(model_path=MA, ankle_mode="active"))
    tsenv.reset(seed=0)
    peak = float(tsenv._orig_forcerange[tsenv.ankle_act_idx[0], 1])
    got = {}
    for wfrac in (0.0, 0.5, 1.0):
        tsenv.data.qvel[tsenv._ankle_dof] = wfrac * tsenv.cfg.ankle_motor_noload_rads
        tsenv._apply_ankle_torque_speed()
        got[wfrac] = float(tsenv.model.actuator_forcerange[tsenv.ankle_act_idx[0], 1])
    check("torque-speed: full torque at rest", abs(got[0.0] - peak) < 1e-6, f"{got[0.0]:.1f}")
    check("torque-speed: half at half no-load speed",
          abs(got[0.5] - 0.5 * peak) < 1e-3, f"{got[0.5]:.1f} vs {0.5*peak:.1f}")
    check("torque-speed: zero at no-load speed", got[1.0] < 1e-6, f"{got[1.0]:.3f}")
    check("torque-speed does not touch the gait actuators",
          abs(float(tsenv.model.actuator_forcerange[1, 1])
              - float(tsenv._orig_forcerange[1, 1])) < 1e-6)

    # mode/plant mismatch must FAIL LOUDLY, not run the wrong arm
    for kw, why in ((dict(model_path=M, ankle_mode="active"), "active on the 6-actuator plant"),
                    (dict(model_path=MA, ankle_mode="passive"), "passive on the ankle-motor plant"),
                    (dict(model_path=M, ankle_mode="nonsense"), "unknown ankle_mode")):
        try:
            DashEnv(Config(**kw))
            check(f"rejects {why}", False)
        except (ValueError, KeyError):
            check(f"rejects {why}", True)

    # zeta ties damping to k: b must scale as sqrt(k), so b(4k) == 2*b(k)
    b = {}
    for k in (100.0, 400.0):
        env = DashEnv(Config(model_path=M, ankle_mode="passive",
                             ankle_stiffness=k, ankle_zeta=0.7))
        b[k] = env.ankle_b
    # Not exactly 2.0: the inertia is MEASURED by impulse response in a sim where the spring is
    # live, so a stiffer spring perturbs its own measurement slightly (~1% over a 4x k range).
    # That residual k-dependence is far smaller than the confound it replaces — the old sweep had
    # damping varying by 6x independently of k — so tolerate it rather than model it away.
    check("zeta damping scales as sqrt(k)", abs(b[400.0] / b[100.0] - 2.0) < 0.05,
          f"b(100)={b[100.0]:.3f} b(400)={b[400.0]:.3f} ratio={b[400.0]/b[100.0]:.3f}")
    # and it must be sized off the STANCE inertia, not the foot's swing inertia — the m7 bug.
    # At k=350 the stance-critical damping is ~21, so zeta=0.7 must land near 15, not near 2.
    env = DashEnv(Config(model_path=M, ankle_mode="passive",
                         ankle_stiffness=350.0, ankle_zeta=0.7))
    check("zeta uses STANCE inertia (not swing)", 10.0 < env.ankle_b < 20.0,
          f"b={env.ankle_b:.3f} (swing-inertia sizing would give ~2.0)")

    # every study preset builds
    study = [n for n in PRESETS if n.startswith("study_")]
    check("study presets exist", len(study) >= 22, f"{len(study)}")
    bad = []
    for n in study:
        try:
            PRESETS[n]()
        except Exception as exc:                       # noqa: BLE001 - report, don't abort
            bad.append(f"{n}: {exc}")
    check("all study presets build", not bad, "; ".join(bad[:3]))


if __name__ == "__main__":
    test_fourier_gait()
    test_cpg_gait()
    test_presets()
    test_env_basic()
    test_sprint_finish()
    test_residual_and_gate()
    test_pitch_reflex()
    test_pitch_reflex_plant()
    test_pitch_assist()
    test_ankle_reflex()
    test_ankle_stiffness()
    test_foot_ahead()
    test_motor_limits()
    test_contact_switch()
    test_duty_sym()
    test_workspace_kill()
    test_torque_curriculum()
    test_hz200_timing()
    test_angmom_term()
    test_rejuvenate_obs_rms()
    test_curriculum_setters()
    test_m1_rail()
    test_ankle_study()
    print(f"\n{'ALL OK' if FAIL == 0 else f'{FAIL} FAILURES'}")
    sys.exit(1 if FAIL else 0)
