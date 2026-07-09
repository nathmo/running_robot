#!/usr/bin/env python3
"""Map out one leg's SAFE workspace by backdriving it, then export hard limits + a safety layer.

SAFE / passive, same technique as record_trajectory.py: the leg's three motors are held LIMP
(streamed SET_CURRENT 0) so you can backdrive them by hand while their broadcast positions are
logged. Nothing ever drives the motors here.

Why this exists: the URDF/MJCF joint ranges for cam (105) and thigh (106) are CAD-derived guesses,
not validated hardstops -- and worse, cam and thigh are NOT independent. They drive a closed 4-bar
loop through the passive pushrod + knee, so only a thin, non-rectangular BAND of (cam, thigh)
combinations is mechanically assemblable (see the spiderbot-hardware notes). A per-joint min/max
box is provably wrong for that pair. So instead of asking you to type in two numbers, this script
records the actual (cam, thigh) samples you sweep out by hand and derives a safe region straight
from that scatter.

Workflow (per leg):
    python fixed_gait/calibrate_workspace.py --leg left     # backdrive LEFT leg (can1)
    python fixed_gait/calibrate_workspace.py --leg right    # backdrive RIGHT leg (can0)

Keys (in the terminal):
    SPACE  start / stop a recording segment. During a segment, move whatever you're calibrating
           through its FULL physical range -- e.g. one segment sweeping abduction stop-to-stop,
           another sweeping the knee (cam+thigh together) through its whole range: hug the
           mechanical limits AND wander the interior so the recorded scatter actually covers the
           reachable region, not just its outline. As many segments as you like.
    z      capture the CURRENT pose (all 3 raw motor angles) as this leg's ZERO reference -- pose
           the leg at the same nominal/CAD-zero stance you use as "home" elsewhere, then press z.
           This is only for the origin marker + readable relative angles in the plot; the actual
           safety limits are stored as absolute raw motor degrees (same as how record_trajectory.py's
           captured `center` is stored/replayed as an absolute value across sessions).
    u      undo the last segment
    q      finish: save raw samples, derive the safe workspace, export + plot

Output (in --dir, default fixed_gait/calibration/):
    raw_{leg}.npz          per-segment raw samples (re-processable without re-recording)
    joint_limits.npz        the reusable safety-layer file (see joint_limits.py)
    workspace_summary.png   abduction range + knee (cam,thigh) scatter/safe-region plot, per leg

Re-derive the limits/plot (different --margin-deg / --grid-deg) without re-recording:
    python fixed_gait/calibrate_workspace.py --process-only

Try the whole pipeline with fabricated data (no hardware, no CAN needed):
    python fixed_gait/calibrate_workspace.py --selftest
"""
import argparse
import os
import sys
import time

import numpy as np

try:
    import can
except ImportError:
    can = None

from record_trajectory import KeyPoller, set_current, read_positions

BITRATE = 1_000_000
LEG_CHANNEL = {"right": "can0", "left": "can1"}       # can0 = RIGHT, can1 = LEFT
MOTOR_IDS = [104, 105, 106]                           # abduction, cam, thigh
SAMPLE_HZ = 150.0
DEFAULT_DIR = "fixed_gait/calibration"


# ------------------------------------------------------------------ recording
def record(leg, interface):
    ch = LEG_CHANNEL[leg]
    print(f"Opening {interface}:{ch} @ {BITRATE} -- {leg.upper()} leg, motors {MOTOR_IDS} "
          f"(abduction, cam, thigh)")
    bus = can.Bus(interface=interface, channel=ch, bitrate=BITRATE)
    latest = {i: None for i in MOTOR_IDS}

    t_end = time.time() + 2.0
    while time.time() < t_end and any(v is None for v in latest.values()):
        for i in MOTOR_IDS:
            set_current(bus, i, 0.0)
        read_positions(bus, latest)
        time.sleep(0.005)
    missing = [i for i, v in latest.items() if v is None]
    if missing:
        print(f"!! No status from motor(s) {missing} on {ch}. Powered? servo mode? Aborting.")
        bus.shutdown()
        return None, None

    print("Motors are LIMP -- move the leg by hand.")
    print("  Sweep ABDUCTION stop-to-stop in one segment, and the KNEE (cam+thigh together --")
    print("  hug the limits AND wander the interior) in another. As many segments as you like.")
    print("  SPACE=start/stop segment   z=capture zero pose   u=undo last   q=finish+save\n")

    segments = []
    zero = None
    dt = 1.0 / SAMPLE_HZ
    recording = False
    buf = []
    next_t = time.time()
    last_print = 0.0
    try:
        with KeyPoller() as kp:
            while True:
                now = time.time()
                for i in MOTOR_IDS:                       # hold limp
                    set_current(bus, i, 0.0)
                read_positions(bus, latest)

                if recording and all(v is not None for v in latest.values()):
                    buf.append([latest[i] for i in MOTOR_IDS])

                k = kp.poll()
                if k == " ":
                    if not recording:
                        recording = True
                        buf = []
                        print(f"\n[segment {len(segments) + 1}] RECORDING...            ")
                    else:
                        recording = False
                        if len(buf) > 10:
                            segments.append(np.array(buf, float))
                            print(f"\n[segment {len(segments)}] saved: {len(buf)} samples"
                                  f"                 ")
                        else:
                            print("\n  (segment too short, discarded)      ")
                elif k == "z" and not recording and all(v is not None for v in latest.values()):
                    zero = [latest[i] for i in MOTOR_IDS]
                    print(f"\n  captured zero: abd={zero[0]:+.1f} cam={zero[1]:+.1f} "
                          f"thigh={zero[2]:+.1f} deg     ")
                elif k == "u" and not recording and segments:
                    segments.pop()
                    print(f"\n  undid last segment -> {len(segments)} left       ")
                elif k in ("q", "\x1b", "\n", "\r"):
                    break

                if (now - last_print) > 0.15:
                    last_print = now
                    pos = "  ".join(f"{n}={latest[i]:+7.1f}"
                                    for n, i in zip(("abd", "cam", "thigh"), MOTOR_IDS))
                    state = "REC " if recording else "idle"
                    ctr = "zero SET" if zero is not None else "zero: press z"
                    print(f"  [{state}] segs={len(segments)}  {ctr}  {pos} deg   ", end="\r")

                next_t += dt
                s = next_t - time.time()
                if s > 0:
                    time.sleep(s)
                else:
                    next_t = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        for i in MOTOR_IDS:
            set_current(bus, i, 0.0)
        bus.shutdown()
    zt = "no zero captured" if zero is None else f"zero={np.round(zero, 1)}"
    print(f"\nFinished {leg}: {len(segments)} segment(s), {zt}.")
    return segments, zero


# ------------------------------------------------------------------ raw save/load
def save_raw(leg, segments, zero, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"raw_{leg}.npz")
    flat = {"leg": leg, "n": len(segments), "has_zero": int(zero is not None)}
    if zero is not None:
        flat["zero"] = np.asarray(zero, float)
    for i, p in enumerate(segments):
        flat[f"p{i}"] = p
    np.savez(path, **flat)
    print(f"saved {len(segments)} raw segment(s) -> {path}")
    return path


def load_raw(path):
    z = np.load(path)
    segments = [z[f"p{i}"] for i in range(int(z["n"]))]
    zero = z["zero"] if ("has_zero" in z.files and int(z["has_zero"])) else None
    return segments, zero


# ------------------------------------------------------------------ grid morphology (no scipy --
#   the Pi runtime, requirements-rpi.txt, is deliberately numpy/onnxruntime/python-can only)
def _binary_dilate(grid, radius):
    """OR the grid with itself shifted over every offset within a `radius`-cell square."""
    if radius <= 0:
        return grid.copy()
    out = grid.copy()
    nx, ny = grid.shape
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(grid)
            sx0, sx1 = max(0, -dx), nx - max(0, dx)
            dx0, dx1 = max(0, dx), nx - max(0, -dx)
            sy0, sy1 = max(0, -dy), ny - max(0, dy)
            dy0, dy1 = max(0, dy), ny - max(0, -dy)
            shifted[dx0:dx1, dy0:dy1] = grid[sx0:sx1, sy0:sy1]
            out |= shifted
    return out


def _binary_erode(grid, radius):
    """Erosion = complement of dilating the complement."""
    if radius <= 0:
        return grid.copy()
    return ~_binary_dilate(~grid, radius)


# ------------------------------------------------------------------ processing / export
def _knee_grid(cam, thigh, grid_deg, dilate_deg, margin_deg):
    cam_lo, cam_hi = float(cam.min()) - grid_deg, float(cam.max()) + grid_deg
    th_lo, th_hi = float(thigh.min()) - grid_deg, float(thigh.max()) + grid_deg
    nc = int(np.ceil((cam_hi - cam_lo) / grid_deg)) + 1
    nt = int(np.ceil((th_hi - th_lo) / grid_deg)) + 1
    raw_grid = np.zeros((nc, nt), bool)
    ic = np.clip(((cam - cam_lo) / grid_deg).astype(int), 0, nc - 1)
    jt = np.clip(((thigh - th_lo) / grid_deg).astype(int), 0, nt - 1)
    raw_grid[ic, jt] = True

    dilate_r = max(0, int(round(dilate_deg / grid_deg)))
    margin_r = max(0, int(round(margin_deg / grid_deg)))
    safe_grid = _binary_dilate(raw_grid, dilate_r)          # fill small hand-sampling gaps
    safe_grid = _binary_erode(safe_grid, margin_r)          # then shrink inward for safety margin
    return dict(raw_grid=raw_grid, safe_grid=safe_grid,
                cam_origin=cam_lo, thigh_origin=th_lo, grid_deg=grid_deg)


def process_and_export(out_dir, margin_deg, grid_deg, dilate_deg):
    lp = os.path.join(out_dir, "raw_left.npz")
    rp = os.path.join(out_dir, "raw_right.npz")
    if not os.path.exists(lp) and not os.path.exists(rp):
        print(f"(no raw_left.npz / raw_right.npz in {out_dir} yet -- record a leg first)")
        return

    export = {}
    legdata = {}
    for leg, path in (("left", lp), ("right", rp)):
        if not os.path.exists(path):
            continue
        segments, zero = load_raw(path)
        samples = np.concatenate(segments, axis=0)          # [N,3] abd, cam, thigh
        abd = samples[:, 0]
        lo, hi = float(abd.min()), float(abd.max())
        safe_lo, safe_hi = lo + margin_deg, hi - margin_deg
        if safe_lo >= safe_hi:
            print(f"!! {leg}: --margin-deg {margin_deg:g} is too large for the observed "
                  f"abduction range [{lo:.1f},{hi:.1f}] -- safe range would be empty/inverted. "
                  f"Reduce --margin-deg or re-sweep a wider range.")

        knee = _knee_grid(samples[:, 1], samples[:, 2], grid_deg, dilate_deg, margin_deg)
        if not knee["safe_grid"].any():
            print(f"!! {leg}: the eroded knee safe-region is EMPTY -- --margin-deg/--dilate-deg "
                  f"too aggressive for --grid-deg {grid_deg:g}, or too few samples. Nothing here "
                  f"will pass validate() until this is fixed.")

        export[f"{leg}_abd_observed_min"] = lo
        export[f"{leg}_abd_observed_max"] = hi
        export[f"{leg}_abd_safe_min"] = safe_lo
        export[f"{leg}_abd_safe_max"] = safe_hi
        export[f"{leg}_abd_zero"] = float(zero[0]) if zero is not None else float("nan")
        export[f"{leg}_knee_grid"] = knee["safe_grid"]
        export[f"{leg}_knee_cam_origin"] = knee["cam_origin"]
        export[f"{leg}_knee_thigh_origin"] = knee["thigh_origin"]
        export[f"{leg}_knee_grid_deg"] = knee["grid_deg"]
        export[f"{leg}_knee_zero"] = (np.array([zero[1], zero[2]], float) if zero is not None
                                      else np.array([np.nan, np.nan]))

        legdata[leg] = dict(samples=samples, zero=zero, knee=knee,
                            abd_observed=(lo, hi), abd_safe=(safe_lo, safe_hi))
        print(f"  {leg:5s}: abduction observed [{lo:+.1f},{hi:+.1f}] deg -> safe "
              f"[{safe_lo:+.1f},{safe_hi:+.1f}] deg (margin {margin_deg:g})  |  knee samples="
              f"{len(samples)}  grid={knee['safe_grid'].shape}  occupied="
              f"{int(knee['safe_grid'].sum())}/{knee['safe_grid'].size} cells")

    out = os.path.join(out_dir, "joint_limits.npz")
    np.savez(out, **export)
    print(f"exported workspace safety limits -> {out}")
    _plot(legdata, os.path.join(out_dir, "workspace_summary.png"))


def _plot(legdata, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping the summary plot)")
        return
    legs = [l for l in ("left", "right") if l in legdata]
    if not legs:
        return
    fig, axes = plt.subplots(len(legs), 2, figsize=(11, 4.6 * len(legs)), squeeze=False)
    for row, leg in enumerate(legs):
        d = legdata[leg]
        ax_abd, ax_knee = axes[row]

        lo, hi = d["abd_observed"]
        slo, shi = d["abd_safe"]
        ax_abd.hlines(0, lo, hi, color="0.7", lw=10, label="observed")
        ax_abd.hlines(0, slo, shi, color="#2c9e3f", lw=10, label="safe (calibrated)")
        if d["zero"] is not None:
            ax_abd.axvline(d["zero"][0], color="k", ls="--", lw=1, label="zero")
        ax_abd.set_yticks([])
        ax_abd.set_xlabel("abduction (raw motor deg)")
        ax_abd.set_title(f"{leg} abduction")
        ax_abd.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
        ax_abd.grid(alpha=0.3, axis="x")

        knee = d["knee"]
        cam, thigh = d["samples"][:, 1], d["samples"][:, 2]
        ax_knee.scatter(cam, thigh, s=4, c="0.6", alpha=0.5, lw=0, label="backdriven samples")
        ci, tj = np.nonzero(knee["safe_grid"])
        cam_cells = knee["cam_origin"] + (ci + 0.5) * knee["grid_deg"]
        thigh_cells = knee["thigh_origin"] + (tj + 0.5) * knee["grid_deg"]
        ax_knee.scatter(cam_cells, thigh_cells, s=3, c="#2c9e3f", alpha=0.3, lw=0,
                       label="safe workspace (eroded)")
        if d["zero"] is not None:
            ax_knee.plot(d["zero"][1], d["zero"][2], "ks", ms=9, label="zero")
        ax_knee.set_xlabel("cam (raw motor deg)")
        ax_knee.set_ylabel("thigh (raw motor deg)")
        ax_knee.set_title(f"{leg} knee (cam, thigh)")
        ax_knee.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
        ax_knee.grid(alpha=0.3)
        ax_knee.set_aspect("equal", "box")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved summary plot -> {path}")


# ------------------------------------------------------------------ self-test (no hardware)
def _selftest(out_dir, margin_deg, grid_deg, dilate_deg):
    """Fabricate plausible backdrive recordings and run them through the exact same
    process_and_export() pipeline used on real data -- exercises grid rasterize/dilate/erode/save/
    plot end to end so it can be sanity-checked before ever touching the robot.

    Writes into a `_selftest` subfolder, NEVER into `out_dir` directly -- fabricated numbers must
    never land at the same path a real safety calibration would use."""
    out_dir = os.path.join(out_dir, "_selftest")
    rng = np.random.default_rng(0)
    print(f"Fabricating synthetic backdrive recordings (no hardware) -> {out_dir}/ ...")
    for leg, cam0, thigh0 in (("left", 20.0, -10.0), ("right", -15.0, 25.0)):
        n_abd = 800
        abd = np.concatenate([np.linspace(-44, 44, n_abd // 2),
                              np.linspace(44, -44, n_abd // 2)]) + rng.normal(0, 0.3, n_abd)
        abd_seg = np.stack([abd, np.full(n_abd, cam0), np.full(n_abd, thigh0)], axis=1)

        n_knee = 4000
        t = np.linspace(0, 4 * np.pi, n_knee)
        cam = cam0 + 35 * np.sin(t) + rng.normal(0, 1.0, n_knee)
        thigh = thigh0 + 0.4 * (cam - cam0) + 15 * np.cos(t / 2) + rng.normal(0, 1.5, n_knee)
        knee_seg = np.stack([np.full(n_knee, 0.0), cam, thigh], axis=1)

        zero = [0.0, cam0, thigh0]
        save_raw(leg, [abd_seg, knee_seg], zero, out_dir)
    process_and_export(out_dir, margin_deg, grid_deg, dilate_deg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", choices=["right", "left"], help="which leg to backdrive/record")
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="where raw + exported files go")
    ap.add_argument("--margin-deg", type=float, default=3.0,
                    help="safety margin (deg) eroded inward from the demonstrated envelope")
    ap.add_argument("--grid-deg", type=float, default=1.0, help="knee occupancy grid cell size (deg)")
    ap.add_argument("--dilate-deg", type=float, default=2.0,
                    help="knee grid: fill sampling gaps up to this many deg before eroding")
    ap.add_argument("--process-only", action="store_true",
                    help="skip recording; just re-derive limits/plot from existing raw files")
    ap.add_argument("--selftest", action="store_true",
                    help="fabricate synthetic recordings (no hardware) and run the full pipeline")
    args = ap.parse_args()

    if args.selftest:
        _selftest(args.dir, args.margin_deg, args.grid_deg, args.dilate_deg)
        return
    if args.process_only:
        process_and_export(args.dir, args.margin_deg, args.grid_deg, args.dilate_deg)
        return
    if can is None:
        print("python-can not installed. `pip install python-can` and bring up the CAN bus.")
        sys.exit(1)
    if not args.leg:
        print("choose --leg right  or  --leg left   (or --process-only / --selftest)")
        sys.exit(1)

    segments, zero = record(args.leg, args.interface)
    if segments:
        save_raw(args.leg, segments, zero, args.dir)
        process_and_export(args.dir, args.margin_deg, args.grid_deg, args.dilate_deg)


if __name__ == "__main__":
    main()
