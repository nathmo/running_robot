"""Is a STOP expressible by the v2 controller at all? Search per-cycle gait specs, not policy weights.

Nine reward-shaped training configurations failed to teach this robot to brake: it never slows, and it
falls while ACCELERATING (see the table in walk_v2/README.md). Two explanations remain, and they call for
opposite fixes:

  (a) an RL exploration problem -- a braking spec sequence exists, PPO never found it;
  (b) an ARCHITECTURE problem -- the latched Fourier action space does not contain a stop, in which case
      no reward term will ever produce one and the fix is a bigger residual or a brake primitive.

This decides between them. Take a trained runner, let it reach steady state, then STOP USING THE POLICY
and drive the spec directly with a smooth schedule: from the cruise spec, modulate frequency, the cam and
thigh amplitudes, and the fore-aft offset o_cam over a few seconds. The search is a cross-entropy method
run entirely inside the batched env -- one candidate schedule per env, so a whole population is a single
vmapped rollout. If any schedule brings the robot to a near standstill upright, the action space contains
a stop and (a) holds. If thousands cannot, (b) holds.

    python walk_v2/tools/brake_search.py --run walk_v2/runs/<run> --checkpoint <ckpt.msgpack>

Knobs: --pop, --iters, --elite, --cruise-s, --brake-s.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

import gait
from env import EnvParams
from evaluate import load_run

# the schedule: 4 channels x 3 knots, piecewise linear in time over the braking window
CHANNELS = ("freq_scale", "amp_scale", "o_cam", "pitch_bias")
KNOTS = 3
DIM = len(CHANNELS) * KNOTS
# per-channel (low, high) at every knot; knot 0 starts at the cruise value by construction
BOUNDS = np.array([[0.4, 1.6],      # frequency multiplier on the cruise frequency
                   [0.1, 1.4],      # cam/thigh amplitude multiplier (shorter strides = less push)
                   [-1.0, 1.0],     # o_cam in spec units (fore-aft foot placement: the capture step)
                   [-1.0, 1.0]])    # thigh offset (lean)


def schedule_at(theta, frac):
    """As `schedule`, but `frac` is PER ENV (pop,) -- each episode can be at its own point in its
    own braking window, which is what staggered brake starts need."""
    th = theta.reshape(theta.shape[0], len(CHANNELS), KNOTS)
    x = jnp.asarray(frac).reshape(-1, 1) * (KNOTS - 1)
    i0 = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, KNOTS - 2)
    w = x - i0
    a = jnp.take_along_axis(th, i0[:, :, None] * jnp.ones((1, len(CHANNELS), 1), jnp.int32), axis=2)[:, :, 0]
    b = jnp.take_along_axis(th, (i0 + 1)[:, :, None] * jnp.ones((1, len(CHANNELS), 1), jnp.int32), axis=2)[:, :, 0]
    return a + w * (b - a)


def schedule(theta, frac):
    """theta (pop, DIM), frac scalar in [0,1] -> (pop, 4) channel values, piecewise linear over knots."""
    th = theta.reshape(theta.shape[0], len(CHANNELS), KNOTS)
    x = frac * (KNOTS - 1)
    i0 = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, KNOTS - 2)
    w = x - i0
    a = jnp.take_along_axis(th, i0[None, None, None] * jnp.ones((th.shape[0], len(CHANNELS), 1), jnp.int32), axis=2)[:, :, 0]
    b = jnp.take_along_axis(th, (i0 + 1)[None, None, None] * jnp.ones((th.shape[0], len(CHANNELS), 1), jnp.int32), axis=2)[:, :, 0]
    return a + w * (b - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--elite", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=4,
                    help="episodes per candidate; the envs are independent episodes, so --reps 1 scores a candidate on ONE and the CEM then selects lucky episodes rather than robust schedules")
    ap.add_argument("--cruise-s", type=float, default=6.0, help="policy-driven run-up before braking")
    ap.add_argument("--brake-s", type=float, default=8.0, help="length of the open-loop braking window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stagger-s", type=float, default=0.0,
                    help="spread the moment braking starts over this many seconds, independently "
                         "per episode. 0 fits ONE brake point, which is right for a finish line and "
                         "wrong for a button: a red-light stop arrives whenever the operator presses, "
                         "so the schedule has to work across the spread, not at one state.")
    ap.add_argument("--free-clock", action="store_true",
                    help="fit/replay with the touchdown resync OFF, as on the robot, which has no "
                         "foot contact sensor. The dash tolerates this; braking is where phase "
                         "alignment should matter most, so a schedule meant for hardware wants it.")
    ap.add_argument("--hold-run", action="store_true",
                    help="during the brake window, keep the POLICY believing it is still running "
                         "(move the finish line out of range) so it keeps contributing its "
                         "stabilising residual instead of its learned stop response")
    ap.add_argument("--json", default=None)
    ap.add_argument("--demo", default=None, help="theta json: run the policy to the line, then brake; writes a video")
    ap.add_argument("--video", default=None)
    ap.add_argument("--brake-at", type=float, default=None,
                    help="distance (m) at which to start braking; default = the line minus the stopping "
                         "distance, so the robot comes to rest ON the line. Braking must START BEFORE the "
                         "line: past it the env is in the stop phase and the policy's own residual -- the "
                         "destabilising response this whole investigation measured -- fights the schedule.")
    args = ap.parse_args()

    cfg, env, agent = load_run(args.run, args.checkpoint, n_envs=args.pop, dr=False,
                               free_clock=args.free_clock)
    params = EnvParams.final(cfg)._replace(dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0,
                                           pitch_assist=0.0, stoplight_prob=0.0)
    dt = env.control_dt
    n_cruise, n_brake = int(args.cruise_s / dt), int(args.brake_s / dt)
    n_stag = int(args.stagger_s / dt)          # spread of the brake start, 0 = one fixed point
    n_total = n_stag + n_brake                 # the scan has to cover the latest start
    # What the policy is TOLD, during braking only. Both the policy's distance-to-go input and the
    # env's stop phase come off params.sprint_dist_m, so pushing the line out of range leaves the
    # policy in the regime it is competent in -- running -- while the schedule does the decelerating.
    # That is not a cheat for the robot: the run/stop button IS this input, so "brake without telling
    # the policy" is a controller you can actually build. The measured reason to want it: past the
    # line the policy's residual becomes the destabilising response this whole investigation chased,
    # and it fights the schedule exactly when the schedule needs it most.
    p_brake = params._replace(sprint_dist_m=1e4) if args.hold_run else params
    if args.hold_run:
        print("[brake] hold-run: the policy is not told about the line during the brake window")
    key = jax.random.PRNGKey(args.seed)

    # ---- run-up: one policy, but the envs are NOT clones -- reset_joint_noise (0.03 rad) gives
    # every env a different starting pose, and a 27 s run-up amplifies that into genuinely
    # independent episodes. So a CEM candidate is scored on its own episode (noisier, but it
    # selects for robustness), and --demo, which puts ONE schedule on every env, is a sample.
    state, obs = env.reset(key, params)
    act = agent._act_greedy

    def cruise_step(carry, _):
        state, obs = carry
        a = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        state2, obs2, _, _, info = env.step(state, a, params)
        return (state2, obs2), info["freq_hz"]

    (state, obs), fz = jax.lax.scan(cruise_step, (state, obs), None, length=n_cruise)
    v0 = float(np.asarray(state.prev_vel_body)[:, 0].mean())
    print(f"[brake] run-up {args.cruise_s:.0f} s: cruise speed {v0:.2f} m/s, clock {float(np.asarray(fz[-1]).mean()):.2f} Hz")
    if v0 < 0.8:
        print("[brake] the policy is not running; nothing to brake from")
        return

    # FIT THE SCHEDULE WHERE IT WILL BE USED. The brake point is a DISTANCE -- the line minus the
    # stopping distance -- not a time, so --cruise-s only bootstraps the speed estimate and the
    # run-up is then extended to reach that distance. Getting this wrong is quiet and expensive: a
    # run-up two seconds too long starts the robot past the brake point, the whole 8 s window then
    # straddles the finish line, the env is in its stop phase for most of it (where the policy's own
    # residual fights the schedule) and EVERY candidate falls. That reads as 0/512 upright -- a stop
    # that looks impossible when it was only mistimed.
    d_brake = args.brake_at if args.brake_at is not None else max(5.0, cfg.sprint_dist_m - v0 * args.brake_s / 2)
    d_now = float(np.asarray(state.sprint_d).mean())
    n_more = int(max(0.0, (d_brake - d_now) / max(v0, 0.5)) / dt)
    n_more = min(n_more, max(0, env.max_steps - n_cruise - n_brake - 10))
    if n_more > 0:
        (state, obs), _ = jax.lax.scan(cruise_step, (state, obs), None, length=n_more)
        d_now = float(np.asarray(state.sprint_d).mean())
        v0 = float(np.asarray(state.prev_vel_body)[:, 0].mean())
    print(f"[brake] fitting at {d_now:.1f} m ({(n_cruise + n_more) * dt:.1f} s in), brake point "
          f"{d_brake:.1f} m, line {cfg.sprint_dist_m:.0f} m, speed {v0:.2f} m/s")
    cruise_spec = state.spec      # (pop, 44); NOT identical across envs -- reset_joint_noise makes
                                  # every episode diverge, so each carries its own latched spec

    key, k_t0 = jax.random.split(key)
    t0_env = (jax.random.randint(k_t0, (args.pop,), 0, n_stag + 1) if n_stag > 0
              else jnp.zeros(args.pop, jnp.int32))
    if n_stag > 0:
        print(f"[brake] staggered start: braking begins anywhere in the next "
              f"{args.stagger_s:.1f} s, drawn per episode")
    # ---- braking window: the POLICY IS OFF for the spec, the schedule drives it
    # `t0` is the tick each env starts braking on. With --stagger-s it differs per episode, so one
    # schedule has to bring the robot down from whatever state the button happens to catch it in --
    # that is the red-light case. Before its own t0 an env just runs the policy, and the spec the
    # schedule modulates is the one latched AT t0, not a single shared cruise spec.
    def brake_rollout(theta, state, obs, t0):
        def step(carry, i):
            state, obs, alive, vmin, surv, cspec, d0 = carry
            braking = i >= t0
            cspec = jnp.where((i == t0)[:, None], state.spec, cspec)
            d0 = jnp.where(i == t0, state.sprint_d, d0)
            frac = jnp.clip((i - t0) / max(n_brake - 1, 1), 0.0, 1.0)
            ch = schedule_at(theta, frac)
            spec = cspec
            spec = spec.at[:, gait.I_FREQ].set(jnp.clip(cspec[:, gait.I_FREQ] * ch[:, 0], -1.0, 1.0))
            spec = spec.at[:, gait.I_S_CAM].multiply(ch[:, 1:2])
            spec = spec.at[:, gait.I_S_THIGH].multiply(ch[:, 1:2])
            spec = spec.at[:, gait.I_O.start].set(jnp.clip(ch[:, 2], -1.0, 1.0))
            spec = spec.at[:, gait.I_O.start + 1].set(jnp.clip(ch[:, 3], -1.0, 1.0))
            # KEEP THE POLICY'S PER-TICK RESIDUAL. It is 45-95% of joint motion on this lineage
            # (walk_mit gait_diag), so zeroing it does not test the gait spec, it just removes the
            # stabiliser and everything falls. Only the LATCHED spec is overridden by the schedule.
            pol = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
            a = jnp.where(braking[:, None],
                          jnp.concatenate([spec, pol[:, gait.SPEC_DIM:]], axis=1), pol)
            state2, obs2, _, done, info = env.step(state, a, p_brake)
            alive2 = alive & ~info["fallen"]
            vx = info["sprint_d"] - state.sprint_d
            # only score speed once this env is actually braking, else the cruise ticks of a late
            # starter would be compared against the standstill of an early one
            vmin = jnp.where(alive2 & braking, jnp.minimum(vmin, jnp.abs(vx) / dt), vmin)
            return (state2, obs2, alive2, vmin, surv + alive2.astype(jnp.float32), cspec, d0), None

        n = theta.shape[0]
        init = (state, obs, jnp.ones(n, bool), jnp.full(n, jnp.inf), jnp.zeros(n),
                jnp.zeros((n, gait.SPEC_DIM)), jnp.zeros(n))
        (fs, _, alive, vmin, surv, _, d0), _ = jax.lax.scan(step, init, jnp.arange(n_total))
        return alive, vmin, surv / float(n_total), fs.sprint_d - d0

    brake_jit = jax.jit(brake_rollout)

    # ---- demo: policy to the line, then the found schedule brakes it to a standstill
    if args.demo:
        th = np.asarray(json.loads(Path(args.demo).read_text())["best"]["theta"], np.float32)[None]
        th = np.repeat(th, args.pop, axis=0)
        # Record a handful of envs, not just env 0: at an 18% stop rate env 0 is usually NOT a stop,
        # so a video of it shows a failure while the statistics above report a success. 16 envs make
        # it ~95% likely at least one stopped, and the video then shows a typical one.
        N_VID = int(min(16, args.pop))
        state, obs = env.reset(jax.random.PRNGKey(args.seed), params)

        def to_line(carry, _):
            state, obs = carry
            a = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
            state2, obs2, _, _, _ = env.step(state, a, params)
            return (state2, obs2), state.data.qpos[:N_VID]

        # Run to the brake DISTANCE, measured -- never to an estimated time. d_brake/v0 ignores the
        # acceleration from rest, so the old estimate landed metres past the intended point, and a
        # brake schedule is sensitive enough to where it starts that this alone can decide the demo.
        # The fit and the demo have to begin at the same place or the demo is testing something else.
        d_brake = args.brake_at if args.brake_at is not None else max(5.0, cfg.sprint_dist_m - v0 * args.brake_s / 2)
        chunk = max(1, int(0.5 / dt))
        n_cap = max(chunk, env.max_steps - n_brake - 10)
        n_line, q_parts = 0, []
        while n_line + chunk <= n_cap:
            if float(np.asarray(state.sprint_d).mean()) >= d_brake:
                break
            (state, obs), q = jax.lax.scan(to_line, (state, obs), None, length=chunk)
            q_parts.append(np.asarray(q))
            n_line += chunk
        q_run = (np.concatenate(q_parts, axis=0) if q_parts
                 else np.zeros((0, N_VID) + tuple(state.data.qpos[0].shape)))
        print(f"[demo] braking begins at {float(np.asarray(state.sprint_d).mean()):.1f} m "
              f"(target {d_brake:.1f} m, line at {cfg.sprint_dist_m:.0f} m)")
        d_line = float(np.asarray(state.sprint_d)[0])
        cruise_spec = state.spec
        print(f"[demo] policy ran {d_line:.1f} m in {n_line * dt:.1f} s; braking from "
              f"{float(np.asarray(state.prev_vel_body)[0, 0]):.2f} m/s")

        # Every env is an INDEPENDENT episode (env.reset splits the key per env) and they all carry
        # the same schedule here, so this is a pop-sized sample of one brake, not one demo. Freeze
        # each env's numbers at ITS episode end: past `done` the env has auto-reset and sprint_d
        # reads ~0, which is what made an earlier demo report "came to rest at -0.0 m".
        def brake_demo(carry, i):
            state, obs, alive, d_f, v_f = carry
            ch = schedule(jnp.asarray(th), i / max(n_brake - 1, 1))
            spec = cruise_spec
            spec = spec.at[:, gait.I_FREQ].set(jnp.clip(cruise_spec[:, gait.I_FREQ] * ch[:, 0], -1.0, 1.0))
            spec = spec.at[:, gait.I_S_CAM].multiply(ch[:, 1:2])
            spec = spec.at[:, gait.I_S_THIGH].multiply(ch[:, 1:2])
            spec = spec.at[:, gait.I_O.start].set(jnp.clip(ch[:, 2], -1.0, 1.0))
            spec = spec.at[:, gait.I_O.start + 1].set(jnp.clip(ch[:, 3], -1.0, 1.0))
            pol = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
            a = jnp.concatenate([spec, pol[:, gait.SPEC_DIM:]], axis=1)
            state2, obs2, _, done, info = env.step(state, a, p_brake)
            vx = (info["sprint_d"] - state.sprint_d) / dt
            d_f = jnp.where(alive, info["sprint_d"], d_f)
            v_f = jnp.where(alive, vx, v_f)
            alive2 = alive & ~done
            return (state2, obs2, alive2, d_f, v_f), (state.data.qpos[:N_VID], info["fallen"][:N_VID])

        n_pop = th.shape[0]
        init = (state, obs, jnp.ones(n_pop, bool), state.sprint_d, jnp.zeros(n_pop))
        (state, obs, alive, d_f, v_f), (q_br, fell0) = jax.lax.scan(
            brake_demo, init, jnp.arange(n_brake))
        alive = np.asarray(alive)
        d_f, v_f = np.asarray(d_f), np.asarray(v_f)
        line = float(cfg.sprint_dist_m)
        overrun = d_f - line
        stopped = alive & (np.abs(v_f) <= 0.25)            # upright and at a standstill
        good = stopped & (overrun <= 20.0)                 # ... and inside the 20 m allowance
        fell0 = np.asarray(fell0)[:, 0]        # env 0's fall, for the env-0 line below
        i_fall = int(np.argmax(fell0)) if fell0.any() else -1
        print(f"[demo] {n_pop} episodes: upright {alive.sum()}/{n_pop}, "
              f"STOPPED (|v|<=0.25) {stopped.sum()}/{n_pop}, "
              f"stopped within 20 m of the line {good.sum()}/{n_pop}")
        if stopped.any():
            ov = overrun[stopped]
            print(f"[demo] of those that stopped: overrun past the line "
                  f"median {np.median(ov):+.1f} m, 10-90% {np.percentile(ov, 10):+.1f} .. "
                  f"{np.percentile(ov, 90):+.1f} m, worst {ov.max():+.1f} m; "
                  f"final |v| median {np.median(np.abs(v_f[stopped])):.3f} m/s")
        rolling = alive & ~stopped
        if rolling.any():
            vr = np.abs(v_f[rolling])
            print(f"[demo] the {int(rolling.sum())} upright-but-still-rolling: final |v| median "
                  f"{np.median(vr):.2f} m/s, 10-90% {np.percentile(vr, 10):.2f} .. "
                  f"{np.percentile(vr, 90):.2f} m/s, at a median {np.median(d_f[rolling]) - line:+.1f} m "
                  f"past the line -- these are NOT falls, they are stops the window ended too soon for")
        print(f"[demo] env 0 (the video): ran {d_line:.1f} m, ended at {d_f[0]:.1f} m "
              f"({overrun[0]:+.1f} m relative to the line), final speed {v_f[0]:+.3f} m/s, "
              f"fell={'no' if i_fall < 0 else f'at {i_fall * dt:.1f} s'}")
        if args.video:
            from evaluate import render_video
            # pick the most typical SUCCESS among the recorded envs: stopped, and closest to the
            # median overrun of everything that stopped. Fall back to the one that got furthest.
            cand = np.where(stopped[:N_VID])[0]
            if cand.size:
                target = np.median(overrun[stopped]) if stopped.any() else overrun[cand].mean()
                idx = int(cand[np.argmin(np.abs(overrun[cand] - target))])
                why = f"stopped {overrun[idx]:+.1f} m past the line at {abs(v_f[idx]):.3f} m/s"
            else:
                idx = int(np.argmax(np.where(alive[:N_VID], d_f[:N_VID], -1e9)))
                why = "no stop among the recorded envs; showing the one that got furthest"
            print(f"[demo] video shows env {idx}: {why}")
            qs = np.concatenate([np.asarray(q_run)[:, idx], np.asarray(q_br)[:, idx]], axis=0)
            render_video(cfg, qs, (n_line + n_brake) * dt, args.video)
        return

    # ---- cross-entropy method over the schedule
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    mu = np.repeat(((lo + hi) / 2)[:, None], KNOTS, axis=1).reshape(-1)
    sd = np.repeat(((hi - lo) / 4)[:, None], KNOTS, axis=1).reshape(-1)
    # One candidate per env scores a schedule on a SINGLE episode, and since the envs are independent
    # episodes that rewards luck: the elite are schedules whose one episode happened to be kind, and
    # they fall over on a held-out seed. Give each candidate `reps` episodes instead and score it on
    # the mean -- a single fall costs 1e3, so a candidate must survive every one of its episodes
    # before its speed is even compared.
    n_reps = max(1, args.reps)
    n_cand = max(4, args.pop // n_reps)
    n_pad = args.pop - n_cand * n_reps
    n_elite = max(4, int(n_cand * args.elite))
    best = dict(score=np.inf, v_min=np.inf, upright=False, theta=None)
    print(f"[brake] {n_cand} candidates x {n_reps} episodes = {n_cand * n_reps} of {args.pop} envs")
    for it in range(args.iters):
        key, k = jax.random.split(key)
        theta = np.asarray(jax.random.normal(k, (n_cand, DIM))) * sd + mu
        lo_t = np.repeat(lo[:, None], KNOTS, axis=1).reshape(-1)
        hi_t = np.repeat(hi[:, None], KNOTS, axis=1).reshape(-1)
        theta = np.clip(theta, lo_t, hi_t)
        if it == 0:      # control: schedule = cruise (freq x1, amp x1, offsets at their cruise values)
            ctrl = np.zeros((args.pop, DIM), np.float32)
            ctrl[:, 0:KNOTS] = 1.0
            ctrl[:, KNOTS:2 * KNOTS] = 1.0
            ctrl[:, 2 * KNOTS:3 * KNOTS] = float(np.asarray(cruise_spec)[0, gait.I_O.start])
            ctrl[:, 3 * KNOTS:4 * KNOTS] = float(np.asarray(cruise_spec)[0, gait.I_O.start + 1])
            a_c, v_c, _, _ = brake_jit(jnp.asarray(ctrl), state, obs, t0_env)
            n_c = int(np.asarray(a_c).sum())
            # The control keeps cruising for the whole window. Read it against WHERE the fit is:
            # mid-run it must be ~all upright or the harness is broken, but at the brake point it
            # runs past the finish line into the stop phase and SHOULD mostly fall -- that fall is
            # the problem being solved, not a bug. What matters there is that braking candidates,
            # which stop before the line, can stay upright where this control cannot.
            past = (d_now + v0 * args.brake_s > cfg.sprint_dist_m) and not args.hold_run
            note = ("expected to fall: the window runs past the line" if past else
                    "if this is not upright the harness is wrong")
            print(f"[brake] CONTROL (schedule = cruise): upright {n_c}/{args.pop}, "
                  f"min |v| {float(np.asarray(v_c)[0]):.2f} m/s -- {note}", flush=True)
        # env (i * n_reps + j) runs candidate i on episode j; the pad rows keep the jitted shape
        theta_env = np.repeat(theta, n_reps, axis=0)
        if n_pad:
            theta_env = np.concatenate([theta_env, np.repeat(theta[-1:], n_pad, axis=0)], axis=0)
        alive, vmin, surv, dstop = brake_jit(jnp.asarray(theta_env, jnp.float32), state, obs, t0_env)
        k = n_cand * n_reps
        alive = np.asarray(alive)[:k].reshape(n_cand, n_reps)
        vmin = np.asarray(vmin)[:k].reshape(n_cand, n_reps)
        surv = np.asarray(surv)[:k].reshape(n_cand, n_reps)
        # A fall is disqualifying but NOT flat: falls score 100..200 by how early they came, so the
        # search still has something to climb when every candidate falls. With a flat penalty the
        # scores tie, the elite are arbitrary, mu/sd drift and the run never recovers -- which is
        # exactly how a --reps 8 fit died at 0/512 while --reps 4 solved the same problem.
        score = np.where(alive, vmin, 100.0 + 100.0 * (1.0 - surv)).mean(axis=1)
        upright_all = alive.all(axis=1)
        order = np.argsort(score)
        el = order[:n_elite]
        if score[order[0]] < best["score"]:
            ds = np.asarray(dstop)[:n_cand * n_reps].reshape(n_cand, n_reps)
            best = dict(score=float(score[order[0]]), v_min=float(vmin[order[0]].max()),
                        upright=bool(upright_all[order[0]]), theta=theta[order[0]].tolist(),
                        stop_m=float(ds[order[0]].mean()))
        mu, sd = theta[el].mean(0), theta[el].std(0) + 1e-3
        n_up = int(alive.sum())
        n_all = int(upright_all.sum())
        if n_up:
            b = f"best mean |v| {score[order[0]]:.3f} m/s (elite mean {np.mean(score[el][score[el] < 1e2]) if np.any(score[el] < 1e2) else float('nan'):.3f})"
        else:
            # nothing upright: the graded fall score is all there is, so show the search climbing
            # through it rather than printing nan and looking stuck
            b = f"all fell; best survived {surv.mean(axis=1).max() * 100:.0f}% of the window (mean {surv.mean() * 100:.0f}%)"
        print(f"[brake] iter {it}: episodes upright {n_up}/{n_cand * n_reps}, candidates upright in "
              f"ALL {n_reps} {n_all}/{n_cand}, {b}", flush=True)

    print(f"\n[brake] cruise {v0:.2f} m/s -> best reachable |v| {best['v_min']:.3f} m/s while upright "
          f"(upright={best['upright']})")
    if best.get("stop_m") is not None:
        print(f"[brake] STOPPING DISTANCE of the best schedule: {best['stop_m']:.1f} m from the "
              f"brake point (mean over its episodes) -- this is the number a corridor has to fit")
    verdict = ("A STOP IS EXPRESSIBLE: the action space contains it, so the failure is RL exploration -- "
               "warm-start from this schedule." if best["v_min"] < 0.3 and best["upright"] else
               "NO STOP FOUND: the latched Fourier action space very likely does not contain one. More "
               "reward terms will not help; widen the residual or add a brake primitive.")
    print("[brake] " + verdict)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(cruise_speed=v0, best=best, verdict=verdict,
                                                   pop=args.pop, iters=args.iters), indent=1))


if __name__ == "__main__":
    main()
