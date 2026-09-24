"""Deterministic golden trace of the v2 (latched) environment -- the CPU/GPU parity fixture.

Two implementations of the DASH-01 Walker v2 stack (this MuJoCo-CPU one and the GPU port) can
only be said to train "the same policy" if they agree on what the policy sees and is paid. This
records, for a fixed preset / seed / action sequence, everything the policy-facing contract
consists of: the observation vector (all 402 entries), the reward and its terms, the commit
flags, the phase, the live spec, the delayed command the plant received, the winding state --
per tick, for N ticks, from the clean plant (no DR, no noise, no disturbances: v2_s1_clean with
reset noise off so the trace is a function of the seed alone).

    python walk_mit/golden_v2.py --write golden/v2_s1_clean_seed0.npz     # record
    python walk_mit/golden_v2.py --check golden/v2_s1_clean_seed0.npz     # replay + compare
    python walk_mit/golden_v2.py --write ... --preset v2_lib_s1            # library variant

The GPU side should produce the same npz keys from its own env and run --check against ours
(tolerances: obs 1e-4 after the first tick's 1 kHz physics diverges at float32 level; reward
1e-3; commit flags exact; spec_live exact -- the latch is integer logic and must be bit-exact).
A checksum line is printed so a pass/fail fits in a log grep.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from config import get_config  # noqa: E402
from env import DashEnv  # noqa: E402

KEYS = ("obs", "reward", "commit", "phase", "spec_live", "ctrl_seen", "theta", "terms", "action",
        "qpos", "qvel")


def record(preset="v2_s1_clean", seed=0, n=300, act_seed=0):
    cfg = get_config(preset)
    cfg.reset_joint_noise = 0.0
    env = DashEnv(cfg)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(act_seed)
    term_names = None
    out = {k: [] for k in KEYS}
    out["obs"].append(obs.astype(np.float64))
    for k in range(n):
        # a structured action sequence: slow-varying spec, jittery residual
        a = np.zeros(env.action_dim, np.float32)
        if env.spec_source == "policy":
            a[:44] = 0.15 * np.sin(0.02 * k + np.arange(44) * 0.37)
            a[35] = -0.4                                     # ~1.85 Hz: several full cycles
            a[44:] = rng.uniform(-0.2, 0.2, 6)
        else:
            a[:6] = rng.uniform(-0.2, 0.2, 6)
            a[6:] = 0.2 * np.sin(0.02 * k + np.arange(env.action_dim - 6))
        obs, r, term, trunc, info = env.step(a)
        if term_names is None:
            term_names = sorted(info["reward_terms"])
        out["action"].append(a.astype(np.float64))
        out["obs"].append(obs.astype(np.float64))
        out["reward"].append(float(r))
        out["commit"].append(int(info["v2"]["commit"]))
        out["phase"].append(float(env._phase))
        out["spec_live"].append(env._spec_live.copy())
        out["ctrl_seen"].append(env.data.ctrl.copy())
        out["theta"].append(env._theta.copy())
        out["terms"].append([info["reward_terms"][t] for t in term_names])
        out["qpos"].append(env.data.qpos.copy())
        out["qvel"].append(env.data.qvel.copy())
        if term or trunc:
            break
    d = {k: np.asarray(v) for k, v in out.items()}
    d["term_names"] = np.asarray(term_names)
    d["meta"] = np.asarray([preset, str(seed), str(act_seed), str(n)])
    return d


def digest(d):
    h = hashlib.sha256()
    for k in ("commit", "spec_live"):
        h.update(np.ascontiguousarray(d[k]).tobytes())
    h.update(np.round(d["obs"], 4).tobytes())
    h.update(np.round(d["reward"], 3).tobytes())
    return h.hexdigest()[:16]


def compare(a, b, tol_obs=1e-4, tol_rew=1e-3):
    n = min(len(a["reward"]), len(b["reward"]))
    rep = []
    ok = True
    if len(a["reward"]) != len(b["reward"]):
        rep.append(f"episode length differs: {len(a['reward'])} vs {len(b['reward'])}")
        ok = False
    for k, tol, exact in (("commit", 0, True), ("spec_live", 0, True), ("obs", tol_obs, False),
                          ("reward", tol_rew, False), ("phase", 1e-6, False), ("theta", 1e-6, False)):
        x, y = np.asarray(a[k])[:n + (1 if k == "obs" else 0)], np.asarray(b[k])[:n + (1 if k == "obs" else 0)]
        err = np.max(np.abs(x - y)) if x.size else 0.0
        bad = (err != 0) if exact else (err > tol)
        first = int(np.argmax(np.any(np.abs(x - y) > (tol if not exact else 0), axis=tuple(range(1, x.ndim))))) if bad else -1
        rep.append(f"  {k:10s} max|diff| {err:.2e} {'EXACT' if exact else f'tol {tol:g}'} "
                   f"{'FAIL at tick %d' % first if bad else 'ok'}")
        ok &= not bad
    return ok, "\n".join(rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v2_s1_clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--act-seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--write", default=None)
    ap.add_argument("--check", default=None)
    args = ap.parse_args()
    d = record(args.preset, args.seed, args.n, args.act_seed)
    print(f"golden trace {args.preset} seed {args.seed}: {len(d['reward'])} ticks, "
          f"{int(d['commit'].sum())} commits, digest {digest(d)}")
    if args.write:
        p = Path(args.write)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **d)
        print(f"wrote {p}")
    if args.check:
        ref = dict(np.load(args.check, allow_pickle=False))
        ok, rep = compare(ref, d)
        print(f"compare against {args.check}: {'PASS' if ok else 'FAIL'} (ref digest {digest(ref)})")
        print(rep)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
