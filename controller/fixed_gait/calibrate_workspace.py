#!/usr/bin/env python3
"""Map out one leg's SAFE workspace by backdriving it, then export hard limits + a safety layer.

SAFE / passive, same technique as record_trajectory.py: the leg's three motors are held LIMP
(streamed SET_CURRENT 0) so you can backdrive them by hand while their broadcast positions are
logged. Nothing ever drives the motors here.

Why this exists: the URDF/MJCF joint ranges for cam (105) and thigh (106) are CAD-derived guesses,
not validated hardstops -- and worse, cam and thigh are NOT independent. They drive a closed 4-bar
loop through the passive pushrod + knee, so only a thin, non-rectangular BAND of (cam, thigh)
combinations is mechanically assemblable (see the dash01-hardware notes). A per-joint min/max
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


def _fill_enclosed(grid):
    """Mark every cell the OUTSIDE cannot reach.

    A hand sweep is a PATH, and the natural way to describe a reachable region with one is to trace
    its boundary -- run the leg round the edge of where it may go and come back to the start. Doing
    that used to produce a thin ring, because the pipeline only ever knew about the cells the path
    itself crossed: the region the operator was outlining was never in the grid at all, and the
    erosion below then ate most of the ring as well.

    Flood the complement inward from the array border with 4-connectivity; whatever the flood never
    reaches is enclosed by the trace, and is exactly what was being outlined. 4-connectivity is the
    conservative choice here: it will not squeeze the outside through a diagonal pinhole in the
    trace, so a boundary that is one cell thin still closes.

    CAVEAT worth knowing: this fills ALL enclosed area, so a genuine forbidden island inside the
    reachable region would be filled in as safe if the sweep went round it rather than through it.
    The 4-bar assembly band has no such island, which is why closing is the default, but it is why
    the caller can turn it off."""
    nx, ny = grid.shape
    free = ~grid
    seen = np.zeros_like(grid)
    stack = []
    border = ([(i, 0) for i in range(nx)] + [(i, ny - 1) for i in range(nx)]
              + [(0, j) for j in range(ny)] + [(nx - 1, j) for j in range(ny)])
    for i, j in border:
        if free[i, j] and not seen[i, j]:
            seen[i, j] = True
            stack.append((i, j))
    while stack:
        i, j = stack.pop()
        for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= a < nx and 0 <= b < ny and free[a, b] and not seen[a, b]:
                seen[a, b] = True
                stack.append((a, b))
    return grid | (free & ~seen)


def _rasterize_points(cam, thigh, nc, nt, cam_lo, th_lo, grid_deg):
    """Mark only the cells that hold a sample -- the pre-2026-09-23 behaviour, for callers with no
    take structure to rasterize as paths."""
    grid = np.zeros((nc, nt), bool)
    ic = np.clip(np.floor((np.asarray(cam) - cam_lo) / grid_deg).astype(int), 0, nc - 1)
    jt = np.clip(np.floor((np.asarray(thigh) - th_lo) / grid_deg).astype(int), 0, nt - 1)
    grid[ic, jt] = True
    return grid


def _rasterize_paths(paths, nc, nt, cam_lo, th_lo, grid_deg):
    """Mark every cell the swept path CROSSES, not only the cells that happen to hold a sample.

    Two consecutive 200 Hz samples can be several cells apart when the leg is moved briskly by
    hand, and marking only the sample cells leaves a dotted line. Dilation used to paper over that,
    but a gap wider than the dilation is a hole in the traced boundary, and a hole is all it takes
    for _fill_enclosed to leak straight out through it. Joining consecutive samples closes the
    trace properly instead of hoping the dilation is big enough.

    Each path is one take: they are rasterized separately so lifting the leg between passes does
    not draw a line across the middle of the region."""
    grid = np.zeros((nc, nt), bool)
    for cam, thigh in paths:
        if not len(cam):
            continue
        ic = np.clip(np.floor((np.asarray(cam) - cam_lo) / grid_deg).astype(int), 0, nc - 1)
        jt = np.clip(np.floor((np.asarray(thigh) - th_lo) / grid_deg).astype(int), 0, nt - 1)
        grid[ic, jt] = True
        if len(ic) < 2:
            continue
        span = np.maximum(np.abs(np.diff(ic)), np.abs(np.diff(jt)))
        for k in np.flatnonzero(span > 1):
            n = int(span[k])
            grid[np.round(np.linspace(ic[k], ic[k + 1], n + 1)).astype(int),
                 np.round(np.linspace(jt[k], jt[k + 1], n + 1)).astype(int)] = True
    return grid


# ------------------------------------------------------------------ processing / export
def _knee_grid(cam, thigh, grid_deg, dilate_deg, margin_deg, paths=None, close_region=True):
    """Build the (cam, thigh) safe-region grid from a hand sweep.

    paths:         [(cam_array, thigh_array), ...], one per take, so the sweep is rasterized as the
                   PATH it was rather than as a cloud of dots. Omitted means the caller has no take
                   structure, and the points are plotted on their own exactly as before -- joining
                   an already-concatenated array would draw a bridge across the region every time
                   one take ended and the next began.
    close_region:  fill what the trace encloses (see _fill_enclosed).

    Order matters: dilate to close hand-sampling gaps, THEN close the region, THEN erode for the
    safety margin. Eroding before the fill would open the trace back up.

    The grid is padded by the full structuring-element reach before the morphology and cropped
    afterwards. Without the pad, _binary_erode -- which is the complement of dilating the
    complement, with zeros shifted in from beyond the array -- does not eat inward from the array
    edge, so a region touching that edge silently kept its margin. The pad also guarantees the
    flood in _fill_enclosed starts from genuinely outside cells.
    """
    cam, thigh = np.asarray(cam, float), np.asarray(thigh, float)
    dilate_r = max(0, int(round(dilate_deg / grid_deg)))
    margin_r = max(0, int(round(margin_deg / grid_deg)))
    pad = dilate_r + margin_r + 2

    cam_lo = float(cam.min()) - grid_deg
    th_lo = float(thigh.min()) - grid_deg
    nc = int(np.ceil((float(cam.max()) + grid_deg - cam_lo) / grid_deg)) + 1
    nt = int(np.ceil((float(thigh.max()) + grid_deg - th_lo) / grid_deg)) + 1

    p_lo_cam, p_lo_th = cam_lo - pad * grid_deg, th_lo - pad * grid_deg
    p_nc, p_nt = nc + 2 * pad, nt + 2 * pad
    if paths is None:
        raw_padded = _rasterize_points(cam, thigh, p_nc, p_nt, p_lo_cam, p_lo_th, grid_deg)
    else:
        raw_padded = _rasterize_paths(paths, p_nc, p_nt, p_lo_cam, p_lo_th, grid_deg)

    closed = _binary_dilate(raw_padded, dilate_r)           # close small hand-sampling gaps
    enclosed = 0
    if close_region:
        filled = _fill_enclosed(closed)                     # a traced outline becomes its interior
        enclosed = int(filled.sum() - closed.sum())
        closed = filled
    safe = _binary_erode(closed, margin_r)                  # then shrink inward for safety margin

    crop = (slice(pad, pad + nc), slice(pad, pad + nt))
    return dict(raw_grid=raw_padded[crop], safe_grid=safe[crop],
                cam_origin=cam_lo, thigh_origin=th_lo, grid_deg=grid_deg,
                closed=bool(close_region), enclosed_cells=enclosed,
                raw_cells=int(raw_padded[crop].sum()), safe_cells=int(safe[crop].sum()))


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

        knee = _knee_grid(samples[:, 1], samples[:, 2], grid_deg, dilate_deg, margin_deg,
                          paths=[(s[:, 1], s[:, 2]) for s in segments])
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
