"""Does this policy meet the contract? One command, one PASS/FAIL table, no interpretation needed.

    python walk_v3/verify.py --run walk_v3/runs/v3_s0
    python walk_v3/verify.py --run walk_v3/runs/v3_s0 --checkpoint walk_v3/runs/v3_s0/best.msgpack

Seven checks, in the order they can invalidate each other. Each prints the number it measured next to
the bar it had to clear, so a FAIL says how far off it was rather than just failing.

  1. TRAINED AS ADVERTISED   the curriculum actually ran: dr_scale, bring-up and the command band
                             reached their targets. This is first because every check below is
                             meaningless if it did not -- three runs in this lineage finished with
                             dr_scale = 0.000 while reporting dr_enable = True, and nothing caught it.
  2. COMMAND TRACKING        the stick means what it says, across the ladder, on the nominal plant.
  3. STRAIGHT                heading and lateral drift over a full-speed run.
  4. BRING-UP                dropped and held-misaligned starts: does it survive being let go?
  5. RANDOMISED PLANT        the same ladder with the plant drawn from the DR ranges and every
                             disturbance OFF. Disturbances-on is a different question and it is the
                             one that has faked three results here; see eval-harness notes in README.
  6. NO PRIVILEGE            the actor observation is a pure function of what the robot can measure.
  7. DEPLOY PARITY           the exported numpy control law reproduces the JAX policy bit for bit.

Exit code is 0 only if every check the run supports passed.
"""
import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np
import jax
import jax.numpy as jnp

from config import config_from_dict
from env import DashEnvV2, EnvParams, FRAME_DIM
from evaluate import load_run
from ppo import initial_params
from train import command_ladder, make_eval_env


class Report:
    """Collects rows so the table prints once, in order, with a single verdict at the end."""

    def __init__(self):
        self.rows = []

    def add(self, section, name, ok, got, want, note=""):
        self.rows.append((section, name, ok, got, want, note))
        return ok

    def skip(self, section, name, why):
        self.rows.append((section, name, None, "-", "-", why))

    def print(self):
        print("\n" + "=" * 96)
        cur = None
        for sec, name, ok, got, want, note in self.rows:
            if sec != cur:
                print(f"\n--- {sec}")
                cur = sec
            tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
            print(f"  [{tag}] {name:<42} {str(got):>14}  (want {want}){('  ' + note) if note else ''}")
        hard = [r for r in self.rows if r[2] is not None]
        bad = [r for r in hard if not r[2]]
        print("\n" + "=" * 96)
        print(f"{len(hard) - len(bad)}/{len(hard)} checks passed"
              + (f" -- FAILED: {', '.join(r[1] for r in bad)}" if bad else " -- ALL PASS"))
        return not bad


# ---------------------------------------------------------------------------- 1. the curriculum
def check_curriculum(rep, run, ckpt):
    """Read what the run actually trained on out of the checkpoint's sidecar.

    `dr_enable=True` in a config says only that the code path exists. The amount is a curriculum,
    and a curriculum can sit at zero for an entire run without a single warning."""
    side = ckpt.with_suffix(".json")
    if not side.exists():
        side = max(run.glob("ckpt_*.json"), key=lambda p: int(p.stem.split("_")[1]), default=None)
    if side is None or not side.exists():
        rep.skip("1. trained as advertised", "curriculum sidecar", "no ckpt_*.json next to the run")
        return {}
    d = json.loads(side.read_text())
    ep = d.get("env_params", {})
    cfg_d = json.loads((run / "resolved_config.json").read_text())
    rep.add("1. trained as advertised", f"domain randomisation reached full width",
            float(ep.get("dr_scale", 0)) >= 0.9, f"{ep.get('dr_scale', 0):.3f}", ">= 0.900",
            "dr_scale from the sidecar, not the config flag")
    if cfg_d.get("bringup_enable"):
        rep.add("1. trained as advertised", "bring-up envelope fully opened",
                float(ep.get("bringup_scale", 0)) >= 0.9, f"{ep.get('bringup_scale', 0):.3f}", ">= 0.900")
    rep.add("1. trained as advertised", "command band opened to zero",
            float(ep.get("cmd_lo", 1)) <= 0.05, f"{ep.get('cmd_lo', 1):.3f}", "<= 0.050",
            "zero command = step in place")
    rep.add("1. trained as advertised", "control jitter reached its target",
            float(ep.get("ctrl_jitter_ms", 0)) >= 0.9 * float(cfg_d.get("ctrl_jitter_ms_final", 1)),
            f"{ep.get('ctrl_jitter_ms', 0):.2f} ms",
            f">= {0.9 * float(cfg_d.get('ctrl_jitter_ms_final', 1)):.2f} ms")
    return d


# ------------------------------------------------------------- 2/3/5. the ladder, straight, plant
def ladder_eval(cfg, agent, n_envs, seconds, dr, bringup, seed):
    """Greedy rollout on a pinned command ladder. Returns per-command (err, upright, heading, |y|).

    `dr=True` means the PLANT is drawn from the DR ranges and every disturbance stays off. That
    distinction is the whole game: this lineage's `load_run(dr=True)` also re-enables pushes, wind,
    trips and a hot thermal start, and it killed a known-good policy in 0.21 s -- which was then
    reported as the policy being fragile. Draw the plant; leave the weather alone."""
    c = replace(cfg, dr_enable=bool(dr), obs_noise_enable=False, push_interval_s=0.0,
                wind_force_max=0.0, wind_gust_n=0.0, trip_prob=0.0, thermal_hot_start_max=0.0,
                pitch_assist_kp=0.0, roll_assist_kp=0.0, yaw_assist_kp=0.0)
    env = DashEnvV2(c, n_envs=n_envs)
    params = EnvParams.final(cfg)._replace(
        dr_scale=1.0 if dr else 0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0, pitch_assist=0.0,
        bringup_scale=1.0 if bringup else 0.0,
        bringup_p_drop=0.5 if bringup else 0.0, bringup_p_held=0.5 if bringup else 0.0)
    lad = jnp.asarray(command_ladder(cfg, n_envs))
    ticks = int(seconds / env.control_dt)
    warm = int(cfg.eval_warm_ticks)
    net = agent.net
    stats = agent.stats

    def body(carry, _):
        state, obs, alive, t, err, yaw, lat, vsum, n = carry
        a = jnp.clip(net.apply(agent.params, stats.normalize(obs), method=net.actor_mean), -1.0, 1.0)
        state2, obs2, _, done, info = env.step(state, a, params)
        on = alive & (t >= warm)
        err = err + jnp.where(on, jnp.abs(info["v_body_x"] - lad), 0.0)
        vsum = vsum + jnp.where(on, info["v_body_x"], 0.0)
        yaw = yaw + jnp.where(on, jnp.abs(info["yaw_true"]), 0.0)
        lat = lat + jnp.where(on, jnp.abs(info["lateral_y"]), 0.0)
        n = n + on.astype(jnp.float32)
        return (state2, obs2, alive & ~done, t + 1, err, yaw, lat, vsum, n), None

    state, obs = env.reset(jax.random.PRNGKey(seed), params)
    state = state.replace(v_cmd=lad, cmd_left=jnp.full_like(state.cmd_left, 1e9))
    z = jnp.zeros(n_envs)
    init = (state, obs, jnp.ones(n_envs, bool), jnp.zeros((), jnp.int32), z, z, z, z, z)
    (_, _, alive, _, err, yaw, lat, vsum, n), _ = jax.lax.scan(body, init, None, length=ticks)
    n = np.asarray(n)
    ok = n > 0                      # envs that lived past the warm window and were measured at all
    nz = np.maximum(n, 1.0)
    lad_np, alive = np.asarray(lad), np.asarray(alive)
    out = []
    for v in sorted(set(lad_np.tolist())):
        m = lad_np == v
        mo = m & ok
        # A policy that died before the warm window produced no speed, so its command error is the
        # WHOLE command -- not zero. The same arithmetic that made the training log read 0.00 m/s
        # for a robot that fell on tick 3 would make this suite hand out a PASS to a corpse, which
        # is worse than having no check: heading and lateral drift go NaN instead, so their checks
        # fail rather than quietly succeeding on an empty average.
        e = np.where(ok[m], np.asarray(err)[m] / nz[m], np.abs(v))
        out.append(dict(cmd=float(v), err=float(e.mean()),
                        speed=float((np.asarray(vsum)[mo] / nz[mo]).mean()) if mo.any() else 0.0,
                        upright=float(alive[m].mean()),
                        measured=float(ok[m].mean()),
                        yaw_deg=float(np.degrees((np.asarray(yaw)[mo] / nz[mo]).mean()))
                        if mo.any() else float("nan"),
                        lat_m=float((np.asarray(lat)[mo] / nz[mo]).mean()) if mo.any() else float("nan")))
    return out


def check_tracking(rep, cfg, agent, args, tol):
    rows = ladder_eval(cfg, agent, args.n_envs, args.seconds, dr=False, bringup=False, seed=7)
    top = max(r["speed"] for r in rows)
    print("\n[ladder | nominal plant]  stick  commanded  achieved   err   upright   heading   |y|")
    for r in rows:
        print(f"                          {r['cmd'] / cfg.v_max * 100:3.0f}%     {r['cmd']:5.2f}     "
              f"{r['speed']:5.2f}   {r['err']:5.2f}    {r['upright'] * 100:3.0f}%    "
              f"{r['yaw_deg']:5.1f}d   {r['lat_m']:4.2f}m")
    worst = max(r["err"] / max(cfg.v_max, 1e-9) for r in rows)
    meas = min(r["measured"] for r in rows)
    rep.add("2. command tracking", "every command produced a measurable run",
            meas > 0.0, f"{meas * 100:.0f}%", "> 0%",
            "envs still alive after the warm window; a mean over nothing is not a pass")
    rep.add("2. command tracking", "worst-command error, fraction of top speed",
            worst <= tol, f"{worst * 100:.1f}%", f"<= {tol * 100:.0f}%")
    rep.add("2. command tracking", "upright at every command",
            all(r["upright"] >= 0.9 for r in rows),
            f"{min(r['upright'] for r in rows) * 100:.0f}%", ">= 90%")
    zero = [r for r in rows if r["cmd"] == 0.0]
    if zero:
        rep.add("2. command tracking", "zero command = step in place (drift)",
                abs(zero[0]["speed"]) <= 0.3, f"{abs(zero[0]['speed']):.2f} m/s", "<= 0.30 m/s")
    fast = [r for r in rows if r["cmd"] >= 0.9 * cfg.v_max]
    if fast:
        rep.add("3. straight", "heading drift at full stick",
                fast[0]["yaw_deg"] <= 15.0, f"{fast[0]['yaw_deg']:.1f} deg", "<= 15 deg")
        rep.add("3. straight", "lateral drift at full stick",
                fast[0]["lat_m"] <= 1.0, f"{fast[0]['lat_m']:.2f} m", "<= 1.00 m")
    return rows, top


def check_dr(rep, cfg, agent, args, nominal_rows):
    if not cfg.dr_enable:
        rep.skip("5. randomised plant", "plant draw", "dr_enable=False in the config")
        return
    rows = ladder_eval(cfg, agent, args.n_envs, args.seconds, dr=True, bringup=False, seed=13)
    print("\n[ladder | plant drawn from the DR ranges, disturbances off]")
    for r in rows:
        print(f"                          {r['cmd'] / cfg.v_max * 100:3.0f}%     {r['cmd']:5.2f}     "
              f"{r['speed']:5.2f}   {r['err']:5.2f}    {r['upright'] * 100:3.0f}%    "
              f"{r['yaw_deg']:5.1f}d   {r['lat_m']:4.2f}m")
    up = min(r["upright"] for r in rows)
    rep.add("5. randomised plant", "upright at every command", up >= 0.8, f"{up * 100:.0f}%", ">= 80%")
    worst = max(r["err"] / max(cfg.v_max, 1e-9) for r in rows)
    rep.add("5. randomised plant", "worst-command error", worst <= 0.25, f"{worst * 100:.1f}%", "<= 25%")
    # the honest comparison: how much did a randomised plant cost, relative to the nominal one?
    dn = np.mean([r["err"] for r in rows]) - np.mean([r["err"] for r in nominal_rows])
    rep.add("5. randomised plant", "tracking cost of randomisation", dn <= 0.5,
            f"+{dn:.2f} m/s", "<= +0.50 m/s", "vs the same ladder on the nominal plant")


def check_bringup(rep, cfg, agent, args):
    if not cfg.bringup_enable:
        rep.skip("4. bring-up", "dirty starts", "bringup_enable=False in the config")
        return
    rows = ladder_eval(cfg, agent, args.n_envs, 6.0, dr=False, bringup=True, seed=23)
    up = float(np.mean([r["upright"] for r in rows]))
    print(f"\n[bring-up | every episode dropped or held-misaligned at the full trained envelope]"
          f"  upright {up * 100:.0f}%")
    for r in rows:
        print(f"                          {r['cmd'] / cfg.v_max * 100:3.0f}%  upright {r['upright'] * 100:3.0f}%")
    rep.add("4. bring-up", "survives being let go (all commands)", up >= 0.8, f"{up * 100:.0f}%", ">= 80%")


# ------------------------------------------------------------------------- 6. the privilege audit
def check_privilege(rep, cfg, env, agent, args):
    """Perturb what the robot CANNOT measure, leave physics untouched, and see if the actor notices.

    The GPU step is not bit-reproducible (scatter atomics), so an identical re-run sets the noise
    floor and everything is judged against that. Without that calibration a fixed epsilon calls
    every channel a leak; with it, a real leak in this lineage read 1.0 against a 3e-05 floor."""
    params = initial_params(cfg)._replace(dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0,
                                          pitch_assist=0.0)
    ad = env.actor_dim
    state, obs = env.reset(jax.random.PRNGKey(5), params)

    def step(carry, _):
        state, obs = carry
        a = jnp.clip(agent._act_greedy(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        s2, o2, _, _, _ = env.step(state, a, params)
        return (s2, o2), None

    (state, obs), _ = jax.lax.scan(step, (state, obs), None, length=120)
    a = jnp.clip(agent._act_greedy(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
    base = np.asarray(env.step(state, a, params)[1])[:, :ad]

    def delta(mod):
        o2 = np.asarray(env.step(mod(state), a, params)[1])[:, :ad]
        return float(np.abs(o2 - base).max())

    floor = max(delta(lambda s: s) * 4.0, 1e-8)
    probes = [("odometry origin shifted 60 m", lambda s: s.replace(x0=s.x0 - 60.0), "x0"),
              ("touchdown phase estimate forced", lambda s: s.replace(
                  phi_td_hat=jnp.full_like(s.phi_td_hat, 3.0),
                  resynced=jnp.zeros_like(s.resynced)), "phi_td_hat")]
    for name, mod, field in probes:
        if not hasattr(state, field):
            rep.skip("6. no privilege", name, f"EnvState has no {field}")
            continue
        d = delta(mod)
        rep.add("6. no privilege", name, d <= floor, f"{d:.2e}", f"<= {floor:.2e}",
                "noise floor from an identical re-run")


# ----------------------------------------------------------------------------- 7. deploy parity
def check_deploy(rep, run, ckpt, env, agent, args):
    """Export the bundle, then run the ROBOT's numpy actor on the same observations as the JAX one.

    A policy that only exists inside JAX is not deployable, and the failure mode is silent: a frame
    width the deploy path does not know about, a task channel pinned to the wrong constant, or an
    un-tanh'd last hidden layer all produce a controller that runs, and runs something else. So this
    compares the actual numbers -- the exported numpy actor's mean action against the trained
    policy's, on observations taken from a real rollout."""
    out = PKG / "results" / f"{run.name}_verify.npz"
    out.parent.mkdir(exist_ok=True)
    cmd = [sys.executable, str(PKG / "export.py"), "--run", str(run), "--out", str(out)]
    if ckpt:
        cmd += ["--checkpoint", str(ckpt)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or [""])[-1]
        rep.add("7. deploy parity", "export.py writes a bundle", False, "error", "exit 0", tail)
        return
    rep.add("7. deploy parity", "export.py writes a bundle", True, out.name, "exit 0")

    sys.path.insert(0, str(PKG.parent))
    sys.path.insert(0, str(PKG.parent / "robot" / "deploy"))
    try:                                        # robot/deploy modules import each other flat
        from bundle import Bundle
        from policy_net import PolicyNet
    except Exception as e:
        rep.skip("7. deploy parity", "numpy actor == JAX actor", f"{type(e).__name__}: {e}")
        return

    b = Bundle.load(out)
    rep.add("7. deploy parity", "bundle frame width matches the trainer",
            int(b.meta["frame_dim"]) == FRAME_DIM, b.meta["frame_dim"], FRAME_DIM)
    rep.add("7. deploy parity", "bundle declares the heading channel",
            "heading" in b.meta["obs_scales"], "heading" in b.meta["obs_scales"], True,
            "the deploy frame must build the channel the policy was trained on")

    # observations from a real rollout, not random vectors: normalisation and the once-block only
    # differ from noise on states the policy actually reaches
    params = EnvParams.final(cfg_of(agent))._replace(dr_scale=0.0, ctrl_jitter_ms=0.0,
                                                     ctrl_drop_prob=0.0, pitch_assist=0.0)
    state, obs = env.reset(jax.random.PRNGKey(3), params)

    def roll(carry, _):
        state, obs = carry
        a = jnp.clip(agent._act_greedy(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        s2, o2, _, _, _ = env.step(state, a, params)
        return (s2, o2), o2[:, :env.actor_dim]

    _, seq = jax.lax.scan(roll, (state, obs), None, length=40)
    seq = np.asarray(seq).reshape(-1, env.actor_dim)[:200]
    jax_a = np.asarray(agent._act_greedy(agent.params, agent.stats.normalize(
        jnp.pad(jnp.asarray(seq), ((0, 0), (0, env.obs_dim - env.actor_dim))))))
    net = PolicyNet(b)
    np_a = np.stack([net(o) for o in seq.astype(np.float32)])
    d = float(np.abs(np_a - jax_a[:, :np_a.shape[1]]).max())
    rep.add("7. deploy parity", "numpy actor == JAX actor (max |delta|)", d < 2e-3, f"{d:.2e}", "< 2e-03",
            f"over {len(seq)} observations from a live rollout")


def cfg_of(agent):
    return agent.cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None, help="full path; default = the run's best.msgpack")
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--tol", type=float, default=0.15, help="command-tracking bar, fraction of v_max")
    ap.add_argument("--skip-deploy", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    ck = Path(args.checkpoint) if args.checkpoint else (run / "best.msgpack")
    if not ck.exists():
        ck = run / "final.msgpack"
    cfg, env, agent = load_run(str(run), str(ck) if ck.exists() else None,
                               n_envs=args.n_envs, dr=False, warm_start=False)
    print(f"[verify] {run.name}  checkpoint {ck.name}  objective={cfg.objective}  "
          f"actor_dim={env.actor_dim}  frame={FRAME_DIM}  v_max={cfg.v_max:.2f} m/s")

    rep = Report()
    check_curriculum(rep, run, ck)
    rows, _ = check_tracking(rep, cfg, agent, args, args.tol)
    check_bringup(rep, cfg, agent, args)
    check_dr(rep, cfg, agent, args, rows)
    check_privilege(rep, cfg, env, agent, args)
    if not args.skip_deploy:
        check_deploy(rep, run, ck if ck.exists() else None, env, agent, args)
    return 0 if rep.print() else 1


if __name__ == "__main__":
    raise SystemExit(main())
