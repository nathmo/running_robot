"""Recorded-trajectory processing for SpiderBot's teach-and-replay gait.

PURE NUMPY, NO MUJOCO — runs on the Raspberry Pi. Shared by record_trajectory.py (fuses + smooths
+ exports the takes you record by hand) and play_trajectory.py (reconstructs per-side targets).

Model of the data
-----------------
Each leg has three motors, columns in this fixed order:
    col 0 = abduction (id 104)  -- stays FIXED (calibrated separately, no motion in the gait)
    col 1 = cam       (id 105)  -- moves
    col 2 = hip       (id 106)  -- moves
You record the RIGHT leg a few times, then the LEFT leg a few times, moving each by hand through
one full cycle. Because the legs are mirror-mounted and each motor's 0 deg is different, the raw
right and left recordings differ by a sign and an offset.

process() turns all of that into ONE canonical, smooth, exactly-periodic shape (mean-removed, in
the right leg's convention) plus, per side, the offset (that side's mean pose), the mirror sign,
the abduction hold angle, and the safe travel range. play_trajectory then reconstructs:
    right(phase) = right_offset + canonical(phase)
    left(phase)  = left_offset  + left_sign * canonical(phase + 0.5)     # dephased 180 deg
so both legs run the *same* trajectory, 180 deg apart, each in its own motor frame and range.
"""
import numpy as np

MOTOR_IDS = np.array([104, 105, 106])       # column order: abduction, cam, hip
COL_ABDUCTION, COL_CAM, COL_HIP = 0, 1, 2
MOVING_COLS = [COL_CAM, COL_HIP]            # abduction is held fixed
NAMES = ["abduction", "cam", "hip"]


# --------------------------------------------------------------------------- resampling / smoothing
def resample_cycle(t, pos, N):
    """Resample a take (t: [T], pos: [T, 3]) onto N uniform phase points over its duration."""
    t = np.asarray(t, float)
    t = t - t[0]
    if t[-1] <= 0:
        raise ValueError("take has zero duration")
    u = t / t[-1]                                   # normalized time 0..1
    grid = np.linspace(0.0, 1.0, N, endpoint=False)
    out = np.empty((N, pos.shape[1]))
    for c in range(pos.shape[1]):
        out[:, c] = np.interp(grid, u, pos[:, c])
    return out


def fft_lowpass_periodic(x, harmonics):
    """Keep only the first `harmonics` Fourier components of a periodic signal x [N].
    Smooths AND forces exact periodicity (closes the loop), so a sloppy hand-return to the start
    is absorbed instead of becoming a seam."""
    X = np.fft.rfft(x)
    if harmonics + 1 < X.size:
        X[harmonics + 1:] = 0.0
    return np.fft.irfft(X, n=x.size)


def _circular_align(sig, ref):
    """Circular shift that best aligns sig onto ref (both [N], zero-mean). Returns (shift, corr)."""
    N = sig.size
    cc = np.fft.irfft(np.fft.rfft(sig) * np.conj(np.fft.rfft(ref)), n=N)
    shift = int(np.argmax(cc))
    denom = np.linalg.norm(sig) * np.linalg.norm(ref) + 1e-9
    return shift, float(cc[shift] / denom)


def _detect_side_sign(sample_moving, ref_hip):
    """Mirror sign of a leg relative to the reference (right) leg, from a representative take.
    Physically the two legs are mirror-mounted so the left sign is -1; we confirm it from the data
    and only override the -1 default if the +1 hypothesis fits clearly better (unusual wiring).
    Returns (sign, corr_for_that_sign)."""
    hip = sample_moving[:, 1] - sample_moving[:, 1].mean()
    _, corr_p = _circular_align(+hip, ref_hip)
    _, corr_m = _circular_align(-hip, ref_hip)
    if corr_p > corr_m + 0.15:        # +1 fits clearly better than the mirror default
        return +1.0, corr_p
    return -1.0, corr_m


# --------------------------------------------------------------------------- processing
def process(right_takes, left_takes, N=256, harmonics=8, verbose=True):
    """Fuse recorded takes into one canonical periodic shape + per-side calibration.

    right_takes / left_takes: lists of (t [T], pos [T,3]) arrays (pos cols = abduction, cam, hip).
    Returns a dict ready for save()."""
    if not right_takes:
        raise ValueError("need at least one RIGHT take")
    rs = [resample_cycle(t, p, N) for t, p in right_takes]
    ls = [resample_cycle(t, p, N) for t, p in left_takes] if left_takes else []

    # reference frame = first right take's moving shape (zero-mean). The right leg defines the
    # canonical convention (sign +1); the left leg is the physical mirror (sign detected below).
    ref_hip = rs[0][:, COL_HIP] - rs[0][:, COL_HIP].mean()
    right_sign = +1.0
    left_sign, left_corr = (-1.0, 0.0)
    if ls:
        left_sign, left_corr = _detect_side_sign(ls[0][:, MOVING_COLS], ref_hip)
        if verbose:
            print(f"  left/right mirror sign = {left_sign:+.0f} (fit corr {left_corr:.2f})")

    # bring every take into the reference convention: apply that side's sign, then shift-align on hip
    aligned = []
    for takes, sgn in ((rs, right_sign), (ls, left_sign)):
        for tk in takes:
            moving = sgn * (tk[:, MOVING_COLS] - tk[:, MOVING_COLS].mean(axis=0))
            shift, corr = _circular_align(moving[:, 1], ref_hip)
            if verbose and corr < 0.5:
                print(f"  ! a take aligns weakly (corr={corr:.2f}); its shape may differ")
            aligned.append(np.roll(moving, -shift, axis=0))

    canonical = np.mean(aligned, axis=0)                       # [N, 2] mean-removed cam,hip
    canonical -= canonical.mean(axis=0)
    for c in range(canonical.shape[1]):                        # smooth + close the loop
        canonical[:, c] = fft_lowpass_periodic(canonical[:, c], harmonics)

    def side_cal(takes, sign):
        if not takes:
            return None
        arr = np.stack(takes)                                  # [ntake, N, 3]
        offset = arr.mean(axis=(0, 1))                         # per-motor mean pose (the "0 offset")
        lo = arr.min(axis=(0, 1)); hi = arr.max(axis=(0, 1))   # recorded travel = safe range
        return dict(offset=offset, lo=lo, hi=hi, sign=float(sign),
                    abduction_hold=float(offset[COL_ABDUCTION]))

    data = dict(
        N=N, harmonics=harmonics,
        canonical=canonical,                                   # [N,2] cam,hip mean-removed
        motor_ids=MOTOR_IDS,
        right=side_cal(rs, right_sign),
        left=side_cal(ls, left_sign),
    )
    if verbose:
        amp = np.ptp(canonical, axis=0)
        print(f"  canonical amplitude: cam {amp[0]:.1f} deg, hip {amp[1]:.1f} deg  ({N} samples)")
        for s in ("right", "left"):
            c = data[s]
            if c:
                print(f"  {s:5s}: offset(abd,cam,hip)={np.round(c['offset'],1)} "
                      f"sign={c['sign']:+.0f} abduction_hold={c['abduction_hold']:.1f}")
    return data


# --------------------------------------------------------------------------- reconstruct / io
def reconstruct(data, side, phase, abduction_override=None):
    """Absolute motor targets [abduction, cam, hip] (deg) for `side` at cyclic phase in [0,1)."""
    cal = data[side]
    if cal is None:
        raise ValueError(f"no calibration for {side} leg")
    N = data["N"]
    ph = phase % 1.0
    if side == "left":
        ph = (ph + 0.5) % 1.0                                  # dephase 180 deg
    x = ph * N
    i0 = int(np.floor(x)) % N
    i1 = (i0 + 1) % N
    frac = x - np.floor(x)
    can = (1 - frac) * data["canonical"][i0] + frac * data["canonical"][i1]   # [2] cam,hip
    out = np.empty(3)
    out[COL_ABDUCTION] = cal["abduction_hold"] if abduction_override is None else abduction_override
    out[MOVING_COLS] = cal["offset"][MOVING_COLS] + cal["sign"] * can
    return np.clip(out, cal["lo"] - 2.0, cal["hi"] + 2.0)      # small margin over recorded range


def save(path, data):
    flat = {"N": data["N"], "harmonics": data["harmonics"],
            "canonical": data["canonical"], "motor_ids": data["motor_ids"]}
    for s in ("right", "left"):
        c = data[s]
        if c is not None:
            flat[f"{s}_offset"] = c["offset"]; flat[f"{s}_lo"] = c["lo"]
            flat[f"{s}_hi"] = c["hi"]; flat[f"{s}_sign"] = c["sign"]
            flat[f"{s}_abduction_hold"] = c["abduction_hold"]
    np.savez(path, **flat)


def load(path):
    z = np.load(path)
    data = dict(N=int(z["N"]), harmonics=int(z["harmonics"]),
                canonical=z["canonical"], motor_ids=z["motor_ids"])
    for s in ("right", "left"):
        if f"{s}_offset" in z:
            data[s] = dict(offset=z[f"{s}_offset"], lo=z[f"{s}_lo"], hi=z[f"{s}_hi"],
                           sign=float(z[f"{s}_sign"]), abduction_hold=float(z[f"{s}_abduction_hold"]))
        else:
            data[s] = None
    return data


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 256
    grid = np.linspace(0, 1, N, endpoint=False)
    # an ASYMMETRIC "true" world shape W (so sign-flip != half-cycle-shift -> tests sign detection)
    W_cam = 10 * np.sin(2 * np.pi * grid) + 4 * np.sin(4 * np.pi * grid + 1.1)
    W_hip = 14 * np.cos(2 * np.pi * grid) + 5 * np.cos(4 * np.pi * grid - 0.7)
    OFF = dict(right=[5.0, 20.0, -12.0], left=[-3.0, -18.0, 33.0])   # different 0 deg per motor
    MIRROR = dict(right=1.0, left=-1.0)                              # left is the mirror

    def fake_take(side, phase0, dur):
        T = int(dur * 200)
        u = (np.linspace(0, 1, T) + phase0) % 1.0
        s = MIRROR[side]; off = OFF[side]
        cam = off[1] + s * np.interp(u, grid, W_cam) + rng.normal(0, 0.4, T)
        hip = off[2] + s * np.interp(u, grid, W_hip) + rng.normal(0, 0.4, T)
        abd = off[0] + rng.normal(0, 0.3, T)
        cam += np.linspace(0, 3.0, T)               # sloppy loop closure (a few deg of drift)
        return np.linspace(0, dur, T), np.stack([abd, cam, hip], axis=1)

    right = [fake_take("right", p, d) for p, d in [(0.0, 3.1), (0.2, 2.7), (0.7, 3.5)]]
    left = [fake_take("left", p, d) for p, d in [(0.1, 2.9), (0.5, 3.2)]]
    print("processing synthetic recordings (asymmetric shape):")
    data = process(right, left)

    ph = np.linspace(0, 1, N, endpoint=False)
    R = np.array([reconstruct(data, "right", p) for p in ph])
    L = np.array([reconstruct(data, "left", p) for p in ph])
    # ground truth for playback: right = off + W ; left = off - W(phase+0.5)
    R_true = np.stack([np.full(N, OFF["right"][0]), OFF["right"][1] + W_cam, OFF["right"][2] + W_hip], 1)
    Wc2 = np.roll(W_cam, -N // 2); Wh2 = np.roll(W_hip, -N // 2)
    L_true = np.stack([np.full(N, OFF["left"][0]), OFF["left"][1] - Wc2, OFF["left"][2] - Wh2], 1)
    seam = np.abs(R[0] - R[-1]) - np.abs(R[0] - R[1])
    err_R = np.abs(R[:, MOVING_COLS] - R_true[:, MOVING_COLS]).max()
    err_L = np.abs(L[:, MOVING_COLS] - L_true[:, MOVING_COLS]).max()
    print(f"  abduction constant: right std={R[:,0].std():.3f} deg (want ~0)")
    print(f"  periodic: seam-jump (cam,hip)={np.round(seam[MOVING_COLS],3)} deg (want ~0)")
    print(f"  RIGHT reconstruction max err vs truth = {err_R:.2f} deg (want < ~2, smoothing+noise)")
    print(f"  LEFT  reconstruction max err vs truth = {err_L:.2f} deg (want < ~2 -> mirror+dephase OK)")
    ok = err_R < 2.5 and err_L < 2.5 and R[:, 0].std() < 0.1 and abs(seam[MOVING_COLS]).max() < 0.5
    print("  RESULT:", "PASS" if ok else "FAIL")
    save("fixed_gait/trajectories/_selftest.npz", data)
    print("  saved fixed_gait/trajectories/_selftest.npz")
