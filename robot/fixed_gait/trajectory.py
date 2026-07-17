"""Recorded-trajectory processing for DASH-01's teach-and-replay gait.

PURE NUMPY, NO MUJOCO — runs on the Raspberry Pi. Shared by record_trajectory.py (smooths +
re-times + exports the takes you record by hand) and play_trajectory.py (reconstructs per-side
targets).

Model of the data
-----------------
Each leg has three motors, columns in this fixed order:
    col 0 = abduction (id 104)  -- stays FIXED (held at the captured center, no motion in the gait)
    col 1 = cam       (id 105)  -- moves
    col 2 = hip       (id 106)  -- moves

Each leg is recorded and processed **completely independently** — there is no shared canonical
shape, no mirror-sign detection, and no cross-leg alignment. This is deliberate: the two motors on
opposite legs have different origins and the legs don't move as a clean global mirror, so forcing
one shape onto both (the old design) corrupted the leg with the weaker fit. Now the left leg
replays exactly the motion you taught it, in its own frame.

Two things ARE shared between the legs, on purpose:
  * a **segment timing schedule** — each leg's cycle is split at its two hip turning points into an
    A->B arc and a B->A arc, and both arcs are re-timed to fixed phase budgets (default 50/50, the
    `split`). So "swing takes as long as return", and the SAME on both legs, independent of how fast
    you happened to move your hand. This is what keeps the two legs temporally symmetric.
  * a **phase relationship** — the left leg plays `phase_shift` (default 0.5 = 180 deg) ahead.

Per side we store the leg's own mean-removed shape (`canonical`), the captured `center` pose (its
origin — you set it live while recording), the clip range, and its `phase_shift`. Playback:
    side(phase) = center + canonical((phase + phase_shift) mod 1)
with abduction held at center[abduction] (or an override).
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


def _retime_cycle(cycle, split=0.5, turn_harmonics=5):
    """Re-time one leg's cycle onto the shared A->B / B->A schedule.

    cycle: [N, 3] (abd, cam, hip) already uniform in the take's own time. We find the hip signal's
    two turning points (max = A, min = B), then rebuild the cycle so the A->B arc fills phase
    [0, split) and the B->A arc fills [split, 1). This anchors phase 0 to the hip-max pose and
    equalizes the two half-cycles to fixed phase budgets, wiping out how fast/slow your hand moved
    through each half. WITHIN each arc the taught position-vs-fraction shape is preserved.

    (Anchoring on hip also makes multiple takes of the same leg line up automatically — no circular
    cross-correlation needed.)"""
    N = cycle.shape[0]

    def smooth_turns(sig):
        return fft_lowpass_periodic(sig - sig.mean(), turn_harmonics)

    hs = smooth_turns(cycle[:, COL_HIP])
    if np.ptp(hs) < 1e-6:                       # hip barely moved — fall back to cam to find turns
        hs = smooth_turns(cycle[:, COL_CAM])
        if np.ptp(hs) < 1e-6:
            return cycle.copy()                 # no motion to anchor on; leave as-is
    iA = int(np.argmax(hs))                      # phase-0 anchor  (hip max)
    iB = int(np.argmin(hs))                      # phase-split anchor (hip min)
    len1 = (iB - iA) % N                         # samples along the A->B arc (cyclic)
    len2 = N - len1                              # samples along the B->A arc
    if len1 == 0 or len2 == 0:
        return cycle.copy()

    p = np.arange(N) / N
    orig = np.where(p < split,
                    iA + (p / split) * len1,                    # A->B arc
                    iB + ((p - split) / (1.0 - split)) * len2)  # B->A arc
    i0 = np.floor(orig).astype(int) % N
    i1 = (i0 + 1) % N
    frac = (orig - np.floor(orig))[:, None]
    return (1.0 - frac) * cycle[i0] + frac * cycle[i1]


# --------------------------------------------------------------------------- per-leg processing
def _process_leg(takes, center, N, harmonics, split, turn_harmonics, phase_shift, name, verbose):
    """Fuse one leg's takes into (canonical shape, center, clip range, phase_shift). None if empty."""
    if not takes:
        return None
    resampled = [resample_cycle(t, p, N) for t, p in takes]
    retimed = np.stack([_retime_cycle(r, split, turn_harmonics) for r in resampled])  # [ntake,N,3]
    mean_cycle = retimed.mean(axis=0)                          # [N,3]

    if center is None:
        center = mean_cycle.mean(axis=0)                       # fallback: the recorded mean pose
        if verbose:
            print(f"  {name}: no captured center -> using recorded mean "
                  f"{np.round(center, 1)} (press 'c' while recording to set it by hand)")
    center = np.asarray(center, float)

    moving = mean_cycle[:, MOVING_COLS]                        # [N,2] cam,hip
    canonical = moving - moving.mean(axis=0)                   # mean-removed oscillation
    for c in range(canonical.shape[1]):                        # smooth + close the loop
        canonical[:, c] = fft_lowpass_periodic(canonical[:, c], harmonics)

    # clip range follows the captured center: center + the recorded deviation extents
    dev = retimed[:, :, MOVING_COLS] - moving.mean(axis=0)     # [ntake,N,2] recorded travel about mean
    lo = center.copy(); hi = center.copy()
    lo[MOVING_COLS] = center[MOVING_COLS] + dev.min(axis=(0, 1))
    hi[MOVING_COLS] = center[MOVING_COLS] + dev.max(axis=(0, 1))

    cal = dict(canonical=canonical, center=center,
               abduction_hold=float(center[COL_ABDUCTION]),
               phase_shift=float(phase_shift), lo=lo, hi=hi)
    if verbose:
        amp = np.ptp(canonical, axis=0)
        print(f"  {name:5s}: amp cam {amp[0]:.1f} deg, hip {amp[1]:.1f} deg  "
              f"center(abd,cam,hip)={np.round(center, 1)}  phase_shift={phase_shift:+.2f}  "
              f"({len(takes)} take(s))")
    return cal


def process(right_takes, left_takes, right_center=None, left_center=None,
            N=256, harmonics=8, split=0.5, left_phase=0.5, turn_harmonics=5, verbose=True):
    """Process each leg independently into per-side calibration ready for save().

    right_takes / left_takes: lists of (t [T], pos [T,3]) arrays (pos cols = abduction, cam, hip).
    right_center / left_center: optional [3] captured center pose per leg (else recorded mean).
    split: fraction of the cycle the A->B arc gets (shared by both legs). left_phase: left dephase.
    """
    if not right_takes and not left_takes:
        raise ValueError("need at least one take on some leg")
    if verbose:
        print(f"  segment split={split:.2f}  left phase_shift={left_phase:+.2f}  "
              f"harmonics={harmonics}  N={N}")
    right = _process_leg(right_takes, right_center, N, harmonics, split, turn_harmonics,
                         0.0, "right", verbose)
    left = _process_leg(left_takes, left_center, N, harmonics, split, turn_harmonics,
                        left_phase, "left", verbose)
    return dict(N=N, harmonics=harmonics, split=float(split), motor_ids=MOTOR_IDS,
                right=right, left=left)


# --------------------------------------------------------------------------- reconstruct / io
def reconstruct(data, side, phase, abduction_override=None):
    """Absolute motor targets [abduction, cam, hip] (deg) for `side` at cyclic phase in [0,1)."""
    cal = data[side]
    if cal is None:
        raise ValueError(f"no calibration for {side} leg")
    N = data["N"]
    ph = (phase + cal["phase_shift"]) % 1.0
    x = ph * N
    i0 = int(np.floor(x)) % N
    i1 = (i0 + 1) % N
    frac = x - np.floor(x)
    can = (1 - frac) * cal["canonical"][i0] + frac * cal["canonical"][i1]   # [2] cam,hip
    out = np.empty(3)
    out[COL_ABDUCTION] = cal["abduction_hold"] if abduction_override is None else abduction_override
    out[MOVING_COLS] = cal["center"][MOVING_COLS] + can
    out[MOVING_COLS] = np.clip(out[MOVING_COLS],                            # margin over recorded range
                               cal["lo"][MOVING_COLS] - 2.0, cal["hi"][MOVING_COLS] + 2.0)
    return out


def save(path, data):
    flat = {"N": data["N"], "harmonics": data["harmonics"], "split": data["split"],
            "motor_ids": data["motor_ids"]}
    for s in ("right", "left"):
        c = data[s]
        if c is not None:
            flat[f"{s}_canonical"] = c["canonical"]
            flat[f"{s}_center"] = c["center"]
            flat[f"{s}_phase_shift"] = c["phase_shift"]
            flat[f"{s}_lo"] = c["lo"]
            flat[f"{s}_hi"] = c["hi"]
    np.savez(path, **flat)


def load(path):
    z = np.load(path)
    if "canonical" in z and "right_canonical" not in z:
        raise ValueError(f"{path} is an OLD-format trajectory (single shared canonical). Re-record "
                         "it with the new record_trajectory.py — the format changed to per-leg.")
    data = dict(N=int(z["N"]), harmonics=int(z["harmonics"]),
                split=float(z["split"]) if "split" in z.files else 0.5,
                motor_ids=z["motor_ids"] if "motor_ids" in z.files else MOTOR_IDS)
    for s in ("right", "left"):
        if f"{s}_canonical" in z.files:
            center = z[f"{s}_center"]
            data[s] = dict(canonical=z[f"{s}_canonical"], center=center,
                           abduction_hold=float(center[COL_ABDUCTION]),
                           phase_shift=float(z[f"{s}_phase_shift"]),
                           lo=z[f"{s}_lo"], hi=z[f"{s}_hi"])
        else:
            data[s] = None
    return data


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 256
    g = np.linspace(0, 1, N, endpoint=False)

    # Two INDEPENDENT leg shapes (NOT mirrors of each other), each with hip max at gait-phase 0 so
    # re-timing anchors phase 0 there. Different amplitudes/harmonics + different centers per leg.
    SHAPE = {
        "right": dict(center=[5.0, 20.0, -12.0],
                      cam=lambda g: 9 * np.sin(2 * np.pi * g) + 3 * np.sin(4 * np.pi * g + 0.6),
                      hip=lambda g: 14 * np.cos(2 * np.pi * g)),
        "left":  dict(center=[-3.0, -18.0, 33.0],
                      cam=lambda g: 6 * np.sin(2 * np.pi * g - 0.4),
                      hip=lambda g: 10 * np.cos(2 * np.pi * g) + 2 * np.cos(4 * np.pi * g)),
    }

    def true_pose(side, gg):
        s = SHAPE[side]; c = s["center"]
        return np.stack([np.full_like(gg, c[0]), c[1] + s["cam"](gg), c[2] + s["hip"](gg)], axis=1)

    def ramp_g(tm, dur_a, dur_b):
        """g(t) for t in [0, dur_a+dur_b): the true anchors sit at fixed real times (g=0 at
        tm=0, g=0.5 at tm=dur_a) — asymmetric hand speed (dur_a != dur_b) between them."""
        return np.where(tm < dur_a, 0.5 * tm / dur_a, 0.5 + 0.5 * (tm - dur_a) / dur_b)

    def fake_take(side, start_frac, dur_a, dur_b, dt=1 / 200):
        """One hand cycle, recording started at an arbitrary point (start_frac of the period) —
        i.e. shift WHEN we start sampling the periodic g(t) ramp, not the g values themselves."""
        T = dur_a + dur_b
        t0 = start_frac * T
        n = int(round(T / dt))
        t = np.arange(n) * dt
        gg = ramp_g((t + t0) % T, dur_a, dur_b)
        pos = true_pose(side, gg) + rng.normal(0, 0.3, (n, 3))
        return t, pos

    right = [fake_take("right", 0.0, 1.2, 2.4), fake_take("right", 0.35, 2.0, 1.0),
             fake_take("right", 0.7, 1.5, 1.5)]
    left = [fake_take("left", 0.1, 1.0, 2.2), fake_take("left", 0.6, 2.3, 1.3)]
    CENTERS = {"right": np.array(SHAPE["right"]["center"]),
               "left": np.array(SHAPE["left"]["center"])}

    print("processing synthetic recordings (two independent, non-mirror leg shapes):")
    data = process(right, left, right_center=CENTERS["right"], left_center=CENTERS["left"])

    ph = np.linspace(0, 1, N, endpoint=False)
    ok = True
    for side in ("right", "left"):
        R = np.array([reconstruct(data, side, p) for p in ph])
        shift = data[side]["phase_shift"]
        truth = true_pose(side, (ph + shift) % 1.0)                 # clean shape at the played phase
        err = np.abs(R[:, MOVING_COLS] - truth[:, MOVING_COLS]).max()
        abd_std = R[:, COL_ABDUCTION].std()
        seam = np.abs(R[0] - R[-1])[MOVING_COLS].max() - np.abs(R[0] - R[1])[MOVING_COLS].max()
        # hip should peak near the played phase 0 (anchor), i.e. where (phase+shift)%1 == 0
        hip_peak_phase = ph[np.argmax(R[:, COL_HIP])]
        want_peak = (-shift) % 1.0
        peak_err = min(abs(hip_peak_phase - want_peak), 1 - abs(hip_peak_phase - want_peak))
        print(f"  {side:5s}: recon max err={err:.2f} deg  abd std={abd_std:.3f}  "
              f"seam={seam:+.3f}  hip-peak@phase={hip_peak_phase:.3f} (want {want_peak:.3f})")
        ok = ok and err < 2.5 and abd_std < 0.1 and abs(seam) < 0.5 and peak_err < 0.03

    # independence: left is NOT the right shape mirrored/dephased
    Rr = np.array([reconstruct(data, "right", p)[MOVING_COLS] for p in ph])
    Ll = np.array([reconstruct(data, "left", p)[MOVING_COLS] for p in ph])
    print(f"  right cam amp={np.ptp(Rr[:,0]):.1f}  left cam amp={np.ptp(Ll[:,0]):.1f} "
          f"(independent shapes -> different amplitudes expected)")
    print("  RESULT:", "PASS" if ok else "FAIL")
    save("fixed_gait/trajectories/_selftest.npz", data)
    print("  saved fixed_gait/trajectories/_selftest.npz")
