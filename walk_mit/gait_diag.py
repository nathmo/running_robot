"""Two first-run diagnostics for the latched-spec design, runnable TODAY on any existing run.

The latch (2026-09 design) moves all between-wrap reactivity into the 6-dim residual at
+-residual_scale rad. Two cheap signals tell whether that authority is enough, and whether the
policy is buying more spec updates the legal way:

  1. FREQUENCY RAILS   f = frequency(freq_raw) per step. A histogram piled at the top rail means
     the policy wants more frequent spec commits (the clock-warp instinct in a legal form); the
     bottom rail is the parked-clock half of the old exploit. Rails are measured in raw units
     (|freq_raw| >= 0.98) so the number is comparable across [0.5, 50] and [0.5, 5] Hz ranges.
  2. RESIDUAL SATURATION   per channel, fraction of steps with |r| >= 0.95 (raw units, before
     residual_scale). Saturation in long runs (tens of ms) means the policy needed MORE than the
     bound at that moment -- a capture step being clipped; saturation in 1-step bursts is chatter.
     Also reported: the residual's share of the total joint-target motion, RMS(res)/RMS(target-
     nominal) -- if that goes above ~0.5 the residual IS the controller and the CPG is a bias.

Offline (greedy, paired seeds, evaluate.build() so every curriculum-restore trap stays fixed):
    python walk_mit/gait_diag.py --run walk_mit/runs/sprint_m6_lim2_s0 --episodes 8 [--json f]

Train-time: ActionDiagCallback logs the same numbers (diag/*) from each rollout buffer -- the
greedy mean action of the current policy on the rollout's own observations, plus the sampled
residual saturation the env actually saw. Wired in train.py next to EstimatorCallback.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import fourier_gait  # noqa: E402

RAIL = 0.98          # |freq_raw| at or beyond this = on a rail
SAT = 0.95           # |residual_raw| at or beyond this = saturated
REWRITE = 0.05       # |delta freq_raw| per step beyond this = the clock was re-warped
CHANNELS = ["hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R"]


# ---------------------------------------------------------------- layout
def action_layout(cfg, spec_dim, gait_action_dim):
    """Where freq and the residual live in the raw action, for either gait generator.
    Read from the env (spec_dim / gait_action_dim) so an active-ankle or impedance tail cannot
    shift the slices. Returns dict(freq=list[int], residual=slice|None, freq_range=(lo,hi))."""
    mode = getattr(cfg, "action_mode", "fourier")
    if mode == "latched":
        import gait_v2
        lay = gait_v2.Layout(cfg.n_harmonics)
        if getattr(cfg, "spec_source", "policy") == "policy":
            freq, res = [lay.freq], lay.residual
        else:                                           # library variant: residual first, no clock
            freq, res = [], slice(0, gait_v2.N_RESIDUAL)
    elif mode == "cpg":
        freq = [2, 3]                                   # per-leg [left, right]
        res = slice(gait_action_dim - fourier_gait.N_RESIDUAL, gait_action_dim) \
            if getattr(cfg, "cpg_residual", True) else None
    else:
        freq = [2 * fourier_gait.per_joint(cfg.n_harmonics)]
        res = slice(spec_dim, spec_dim + fourier_gait.N_RESIDUAL)
    lo, hi = cfg.gait_freq_hz
    return dict(freq=freq, residual=res, freq_range=(float(lo), float(hi)))


# ---------------------------------------------------------------- stats
def freq_stats(freq_raw, freq_range, sequential=True, n_bins=10):
    """freq_raw: [T] or [T, k] in [-1, 1]. Rails in raw units; histogram in Hz (log bins)."""
    x = np.asarray(freq_raw, dtype=float).reshape(len(freq_raw), -1)
    hz = np.array([fourier_gait.frequency(v, freq_range) for v in x.ravel()]).reshape(x.shape)
    lo, hi = freq_range
    edges = np.geomspace(lo, hi, n_bins + 1)
    hist, _ = np.histogram(hz.ravel(), bins=edges)
    out = dict(
        lo_rail=float(np.mean(x <= -RAIL)),
        hi_rail=float(np.mean(x >= RAIL)),
        median_hz=float(np.median(hz)),
        mean_hz=float(np.mean(hz)),
        p10_hz=float(np.percentile(hz, 10)),
        p90_hz=float(np.percentile(hz, 90)),
        hist_edges_hz=edges.tolist(),
        hist_frac=(hist / max(hist.sum(), 1)).tolist(),
    )
    if sequential and len(x) > 1:
        out["rewrite_frac"] = float(np.mean(np.abs(np.diff(x, axis=0)) > REWRITE))
    return out


def _run_lengths(mask):
    """lengths of consecutive True runs in a 1-D bool array"""
    m = np.asarray(mask, bool)
    if not m.any():
        return np.zeros(0, int)
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def _signed_runs(col):
    """same-sign saturation runs: a +-1 chatter gives runs of 1, a clipped capture step gives
    tens of ms. Unsigned runs would read the chatter as one long saturation."""
    return np.concatenate([_run_lengths(col >= SAT), _run_lengths(col <= -SAT)])


def residual_stats(r, sequential=True, dt=None):
    """r: [T, 6] raw residual in [-1, 1]. Per-channel saturation, RMS, and (if sequential) the
    burst structure of the saturation: p90 same-sign run length, and the sign-flip rate."""
    r = np.asarray(r, dtype=float)
    sat = np.abs(r) >= SAT
    out = dict(
        sat_frac=sat.mean(axis=0).tolist(),
        sat_frac_any=float(np.mean(sat.any(axis=1))),
        rms=np.sqrt(np.mean(r ** 2, axis=0)).tolist(),
        mean_abs=np.mean(np.abs(r), axis=0).tolist(),
        p99_abs=np.percentile(np.abs(r), 99, axis=0).tolist(),
    )
    if sequential and len(r) > 1:
        runs = [_signed_runs(r[:, j]) for j in range(r.shape[1])]
        out["sat_run_p90"] = [float(np.percentile(x, 90)) if len(x) else 0.0 for x in runs]
        out["sat_run_max"] = [int(x.max()) if len(x) else 0 for x in runs]
        flips = np.sign(r[1:]) * np.sign(r[:-1]) < 0
        out["sign_flip_frac"] = flips.mean(axis=0).tolist()
        if dt is not None:
            out["sat_run_p90_ms"] = [1e3 * dt * v for v in out["sat_run_p90"]]
    return out


def authority_share(res_rad, total_rad):
    """RMS of the residual's contribution vs RMS of the whole joint-target excursion, per channel
    and overall. total = target - nominal (includes the residual); cpg = total - residual."""
    res_rad = np.asarray(res_rad, float)
    total_rad = np.asarray(total_rad, float)
    cpg = total_rad - res_rad
    rms = lambda a: np.sqrt(np.mean(a ** 2, axis=0))
    r_rms, t_rms, c_rms = rms(res_rad), rms(total_rad), rms(cpg)
    return dict(
        res_rms_rad=r_rms.tolist(),
        cpg_rms_rad=c_rms.tolist(),
        share=(r_rms / np.maximum(t_rms, 1e-9)).tolist(),
        share_overall=float(np.sqrt(np.mean(res_rad ** 2)) / max(np.sqrt(np.mean(total_rad ** 2)), 1e-9)),
    )


# ---------------------------------------------------------------- offline CLI
def rollout(run, episodes, seed0):
    from evaluate import build
    model, venv, raw = build(Path(run), None, None)
    cfg = raw.cfg
    lay = action_layout(cfg, raw.spec_dim, raw.gait_action_dim)
    nu6 = min(raw.nu, 6)
    buf_a, buf_t = [], []
    # on_control_step fires after the physics of a step: _prev_applied is THIS step's post-delay
    # action and _prev_motor_cmd this step's normalized target, so the two are aligned.
    raw.on_control_step = lambda: (buf_a.append(raw._prev_applied.copy()),
                                   buf_t.append(raw._prev_motor_cmd[:nu6] * cfg.action_scale))
    eps = []
    for e in range(episodes):
        venv.seed(seed0 + e)
        obs = venv.reset()
        buf_a.clear(); buf_t.clear()
        done, sprint, fell = False, None, False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, d, info = venv.step(a)
            sprint = info[0].get("sprint", sprint)
            if d[0]:
                # a sprint FINISH also ends the episode un-truncated; only a non-finish is a fall
                finished = sprint is not None and sprint.get("t_line") is not None
                fell = not bool(info[0].get("TimeLimit.truncated", False)) and not finished
                done = True
        A = np.asarray(buf_a)
        T = np.asarray(buf_t)
        eps.append(dict(seed=seed0 + e, fell=fell, t=len(A) * raw.control_dt,
                        x=None if sprint is None else sprint["d"],
                        t_line=None if sprint is None else sprint["t_line"],
                        actions=A, targets=T))
        print(f"  ep{e:02d}: {len(A):5d} steps  {'FELL' if fell else 'timeo'}  "
              f"x {eps[-1]['x'] if eps[-1]['x'] is not None else float('nan'):6.1f} m", flush=True)
    return cfg, lay, raw.control_dt, eps


def summarize(cfg, lay, dt, eps):
    A = np.concatenate([e["actions"] for e in eps])
    T = np.concatenate([e["targets"] for e in eps])
    out = {}
    if lay["freq"]:
        fr = A[:, lay["freq"]]
        # sequential stats (rewrite, run lengths, flips) per episode, then pooled by concatenation
        # of per-episode results -- never across an episode boundary
        f = freq_stats(fr, lay["freq_range"], sequential=False)
        f["rewrite_frac"] = float(np.mean(np.concatenate(
            [np.abs(np.diff(e["actions"][:, lay["freq"]], axis=0)) > REWRITE for e in eps])))
        out["freq"] = f
    if lay["residual"] is not None:
        R = A[:, lay["residual"]]
        r = residual_stats(R, sequential=False)
        per = [residual_stats(e["actions"][:, lay["residual"]], sequential=True, dt=dt) for e in eps]
        r["sat_run_p90_ms"] = np.max([p["sat_run_p90_ms"] for p in per], axis=0).tolist()
        r["sat_run_max_ms"] = (1e3 * dt * np.max([p["sat_run_max"] for p in per], axis=0)).tolist()
        r["sign_flip_frac"] = np.mean([p["sign_flip_frac"] for p in per], axis=0).tolist()
        out["residual"] = r
        out["authority"] = authority_share(cfg.residual_scale * R, T[:, :R.shape[1]])
    out["episodes"] = [{k: v for k, v in e.items() if k not in ("actions", "targets")} for e in eps]
    out["steps"] = int(len(A))
    out["dt"] = float(dt)
    out["residual_scale"] = float(cfg.residual_scale)
    out["freq_range"] = list(lay["freq_range"])
    return out


def report(name, s):
    lo, hi = s["freq_range"]
    eps = s["episodes"]
    finishes = sum(e.get("t_line") is not None for e in eps)
    falls = sum(e["fell"] and e.get("t_line") is None for e in eps)
    print(f"\n{name}: {len(eps)} greedy eps, {s['steps']} steps @ {1/s['dt']:.0f} Hz, "
          f"finishes {finishes}, falls {falls}")
    if "freq" in s:
        f = s["freq"]
        print(f"  FREQ  range [{lo:g}, {hi:g}] Hz   lo-rail {100*f['lo_rail']:5.1f}%   "
              f"hi-rail {100*f['hi_rail']:5.1f}%   median {f['median_hz']:.2f} Hz   "
              f"p10-p90 {f['p10_hz']:.2f}-{f['p90_hz']:.2f}   re-warped {100*f['rewrite_frac']:.0f}% of steps")
        e = f["hist_edges_hz"]
        bars = "  ".join(f"{e[i]:.3g}-{e[i+1]:.3g}:{100*h:3.0f}%" for i, h in enumerate(f["hist_frac"]))
        print(f"        hist  {bars}")
    else:
        print("  FREQ  latched by the library entry (no clock channel in this action space)")
    if "residual" not in s:
        print("  RESIDUAL  none in this action space")
        return
    r, a = s["residual"], s["authority"]
    print(f"  RESIDUAL  bound +-{s['residual_scale']:.2f} rad   saturated on any channel "
          f"{100*r['sat_frac_any']:.1f}% of steps   residual share of joint motion "
          f"{100*a['share_overall']:.0f}%")
    print(f"        {'channel':11s} {'sat%':>6s} {'rms':>6s} {'p99':>6s} {'run90':>7s} {'runmax':>7s} "
          f"{'flip%':>6s} {'share':>6s}  {'res rms':>8s} {'cpg rms':>8s}")
    for j, ch in enumerate(CHANNELS[:len(r["sat_frac"])]):
        print(f"        {ch:11s} {100*r['sat_frac'][j]:6.1f} {r['rms'][j]:6.2f} {r['p99_abs'][j]:6.2f} "
              f"{r['sat_run_p90_ms'][j]:5.0f}ms {r['sat_run_max_ms'][j]:5.0f}ms "
              f"{100*r['sign_flip_frac'][j]:6.1f} {100*a['share'][j]:5.0f}%  "
              f"{a['res_rms_rad'][j]:8.3f} {a['cpg_rms_rad'][j]:8.3f}")


# ---------------------------------------------------------------- train-time callback
try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:                                   # pragma: no cover - CLI use without SB3
    BaseCallback = object


class ActionDiagCallback(BaseCallback):
    """Logs diag/* after every rollout: frequency rails + residual saturation of the CURRENT
    policy's greedy mean on the rollout's own (already-normalized) observations, and the sampled
    residual saturation the env actually saw. One no-grad forward pass over <= max_rows rows."""

    def __init__(self, max_rows=8192):
        super().__init__()
        self.max_rows = int(max_rows)
        self._lay = None

    def _on_training_start(self) -> None:
        env = self.training_env
        self._lay = action_layout(env.get_attr("cfg")[0], env.get_attr("spec_dim")[0],
                                  env.get_attr("gait_action_dim")[0])

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        import torch
        buf = self.model.rollout_buffer
        obs = buf.observations.reshape(-1, buf.observations.shape[-1])
        acts = buf.actions.reshape(-1, buf.actions.shape[-1])
        stride = max(1, len(obs) // self.max_rows)
        obs, acts = obs[::stride], acts[::stride]
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=self.model.device)
            mean = self.model.policy.get_distribution(t).mode().cpu().numpy()
        mean = np.clip(mean, -1.0, 1.0)
        sampled = np.clip(acts, -1.0, 1.0)
        lay = self._lay
        if lay["freq"]:
            f = freq_stats(mean[:, lay["freq"]], lay["freq_range"], sequential=False)
            self.logger.record("diag/freq_hz_median", f["median_hz"])
            self.logger.record("diag/freq_lo_rail", f["lo_rail"])
            self.logger.record("diag/freq_hi_rail", f["hi_rail"])
        if lay["residual"] is not None:
            r = residual_stats(mean[:, lay["residual"]], sequential=False)
            rs = residual_stats(sampled[:, lay["residual"]], sequential=False)
            self.logger.record("diag/res_sat", float(np.mean(r["sat_frac"])))
            self.logger.record("diag/res_sat_any", r["sat_frac_any"])
            self.logger.record("diag/res_sat_sampled", float(np.mean(rs["sat_frac"])))
            self.logger.record("diag/res_rms", float(np.mean(r["rms"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--report", action="store_true",
                    help="treat --run entries as gait_diag JSON files and only print the report")
    args = ap.parse_args()
    if args.report:
        for p in args.run:
            for name, s in json.loads(Path(p).read_text()).items():
                report(name, s)
        return
    results = {}
    for run in args.run:
        print(f"== {run}", flush=True)
        cfg, lay, dt, eps = rollout(run, args.episodes, args.seed0)
        s = summarize(cfg, lay, dt, eps)
        report(Path(run).name, s)
        results[Path(run).name] = s
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
