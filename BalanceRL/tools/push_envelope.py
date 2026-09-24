"""The push-recovery envelope of a balance checkpoint (or of the zero-action stand, --baseline).

    python BalanceRL/tools/push_envelope.py --run BalanceRL/runs/bal_s0 [--ckpt best]
    python BalanceRL/tools/push_envelope.py --baseline --run BalanceRL/runs/bal_s0

Every env gets ONE horizontal push of a fixed size from a fixed direction (the first push of the
episode, at 1-2 s), on the full-width plant with the sensor noise on, under the greedy policy. It is
scored once its verdict is in: still upright push_survive_s after the push started. The grid is

    push size  dv  in DV_LADDER (m/s of whole-robot velocity change)
    direction      8 azimuths; 0 = the push comes from behind (it shoves the robot forward)
    CoM shift      centre and the four +-3 cm corners in x/y

x `--reps` random plant draws each. dv = 0 is the quiet control (no push at all can fail it).
Writes <run>/envelope_<ckpt|baseline>.json and .png.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np                                    # noqa: E402
import jax                                            # noqa: E402
import jax.numpy as jnp                               # noqa: E402

from policy_io import load_policy                     # noqa: E402
from env import EnvParams                             # noqa: E402
from plant import Override                            # noqa: E402

DV_LADDER = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)   # where this robot actually lives
DIRS = ("front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right")
# azimuth of the FORCE: a push "from behind" drives the robot forward (+x)
AZ = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
COMS = (("centre", 0.0, 0.0), ("CoM +x", 0.03, 0.0), ("CoM -x", -0.03, 0.0),
        ("CoM +y", 0.0, 0.03), ("CoM -y", 0.0, -0.03))


def run_envelope(run, ckpt=None, baseline=False, reps=2, seed=11):
    grid = [(ic, ia, iv) for ic in range(len(COMS)) for ia in range(len(AZ))
            for iv in range(len(DV_LADDER)) for _ in range(reps)]
    n = len(grid)
    cfg, env, pol = load_policy(run, ckpt, n_envs=n)
    ic, ia, iv = (np.array(x) for x in zip(*grid))
    ov = Override(com_x=jnp.asarray([COMS[i][1] for i in ic], jnp.float32),
                  com_y=jnp.asarray([COMS[i][2] for i in ic], jnp.float32),
                  com_z=jnp.zeros(n, jnp.float32),
                  push_dv=jnp.asarray(np.array(DV_LADDER)[iv], jnp.float32),
                  push_az=jnp.asarray(AZ[ia], jnp.float32), push_el=jnp.zeros(n, jnp.float32))
    prm = EnvParams(plant_scale=1.0, push_level=1.0, push_on=1.0)
    st, obs = env.reset(jax.random.PRNGKey(seed), prm, ov)
    step = jax.jit(env.step)
    verdict = np.full(n, np.nan)            # 1 survived, 0 fell (the first push only)
    n_ticks = int(round((cfg.push_first_s[1] + cfg.push_survive_s + 0.2) / env.control_dt))
    zero = jnp.zeros((n, env.action_dim))
    for _ in range(n_ticks):
        a = zero if baseline else pol.act(obs)
        st, obs, r, done, info = step(st, a, prm)
        ok = np.asarray(info["push_ok"])
        fell = np.asarray(info["fallen"])
        open_ = np.isnan(verdict)
        verdict[open_ & ok] = 1.0
        verdict[open_ & fell] = 0.0          # a fall before the push counts: it did not stand
        if not np.isnan(verdict).any():
            break
    verdict = np.nan_to_num(verdict, nan=1.0)    # verdict not in yet = still standing
    S = np.zeros((len(COMS), len(AZ), len(DV_LADDER)))
    for k in range(n):
        S[ic[k], ia[k], iv[k]] += verdict[k] / reps
    return cfg, pol, S


def summarize(S):
    by_dv = S.mean(axis=(0, 1))
    # the largest push survived >= 80% in EVERY direction and CoM condition
    ok_all = [dv for j, dv in enumerate(DV_LADDER) if (S[:, :, j] >= 0.8).all()]
    per_dir = {DIRS[i]: max([dv for j, dv in enumerate(DV_LADDER) if S[:, i, j].mean() >= 0.8], default=0.0)
               for i in range(len(DIRS))}
    return dict(survival_by_dv={f"{dv:.2f}": round(float(v), 3) for dv, v in zip(DV_LADDER, by_dv)},
                max_dv_all_dirs_and_coms_80=max(ok_all) if ok_all else 0.0,
                max_dv_per_dir_80=per_dir,
                survival_by_com={c[0]: [round(float(v), 3) for v in S[i].mean(0)] for i, c in enumerate(COMS)})


def plot(S, title, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    # sequential: one hue, light -> dark (the reference palette's blue ramp)
    seq = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    series = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")    # categorical slots 1-5
    ink, muted, grid, surface = "#1f1f1e", "#6b6a66", "#e6e5e1", "#fcfcfb"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.patch.set_facecolor(surface)
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
    M = S.mean(axis=0)                                   # direction x dv, averaged over CoM
    im = ax1.imshow(M, cmap=seq, vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks(range(len(DV_LADDER)), [f"{d:g}" for d in DV_LADDER])
    ax1.set_yticks(range(len(DIRS)), list(DIRS))
    ax1.set_xlabel("push size, m/s of whole-robot velocity change", color=muted, fontsize=10)
    ax1.set_ylabel("robot pushed toward", color=muted, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax1.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=8,
                     color="#ffffff" if v > 0.55 else ink)
    cb = fig.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)
    cb.set_label("survival (all CoM offsets)", color=muted, fontsize=9)
    cb.ax.tick_params(colors=muted, labelsize=8)
    cb.outline.set_edgecolor(grid)
    ax1.set_title("Survival by direction and push size", color=ink, fontsize=11, loc="left")
    by_com = S.mean(axis=1)                              # com x dv
    for i, (name, _, _) in enumerate(COMS):
        ax2.plot(DV_LADDER, by_com[i], color=series[i], lw=2, marker="o", ms=5, label=name)
        # direct labels are placed at the FIRST point where the curves are still apart; at the right
        # edge they all sit on zero and collide, which is worse than no label at all
        j = int(np.argmax(np.abs(by_com[i] - by_com.mean(0)) == np.max(np.abs(by_com - by_com.mean(0)), axis=1)[i]))
        if by_com[i][j] > 0.05:
            ax2.annotate(name, (DV_LADDER[j], by_com[i][j]), xytext=(4, 6), textcoords="offset points",
                         color=ink, fontsize=8)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_xlim(DV_LADDER[0] - 0.02, DV_LADDER[-1] + 0.02)
    ax2.grid(True, color=grid, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.set_xlabel("push size, m/s", color=muted, fontsize=10)
    ax2.set_ylabel("survival (all directions)", color=muted, fontsize=10)
    ax2.legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower left")
    ax2.set_title("Survival by CoM offset", color=ink, fontsize=11, loc="left")
    fig.suptitle(title, color=ink, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=surface)
    plt.close(fig)


def replot(run, tag):
    """Re-draw from a saved envelope json (no simulation)."""
    p = Path(run) / f"envelope_{tag}"
    d = json.loads(p.with_suffix(".json").read_text())
    S = np.array(d["survival"])
    title = ("zero action (nominal PD stand)" if tag == "baseline"
             else f"{Path(run).name} / {d['tag']} @ {(d.get('step') or 0) / 1e6:.0f} M")
    plot(S, title + " -- full plant, sensor noise on, one push per trial", p.with_suffix(".png"))
    return p.with_suffix(".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline", action="store_true", help="the zero action (the nominal PD stand)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--replot", action="store_true", help="re-draw from the saved json, no simulation")
    a = ap.parse_args()
    if a.replot:
        print("wrote", replot(a.run, "baseline" if a.baseline else "best"))
        return
    cfg, pol, S = run_envelope(a.run, a.ckpt, a.baseline, a.reps)
    tag = "baseline" if a.baseline else pol.path.name
    summ = summarize(S)
    base = Path(a.run) / f"envelope_{tag}"
    base.with_suffix(".json").write_text(json.dumps(dict(
        tag=tag, step=pol.side.get("step"), dv=list(DV_LADDER), dirs=list(DIRS),
        coms=[c[0] for c in COMS], survival=S.tolist(), **summ), indent=1))
    title = ("zero action (nominal PD stand)" if a.baseline else
             f"{Path(a.run).name} / {tag} @ {pol.side.get('step', 0) / 1e6:.0f} M") + \
        " -- full plant, sensor noise on, one push per trial"
    plot(S, title, base.with_suffix(".png"))
    print(json.dumps(summ, indent=1))
    print(f"wrote {base}.json / .png")


if __name__ == "__main__":
    main()
