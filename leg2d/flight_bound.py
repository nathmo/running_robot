"""Flight-inclusive upper bound on running speed -- closes the flight-phase loophole in the
rail-sweep bound (rail_bound.py bounds GROUNDED gaits only: with flight, stance shrinks and
ballistics stretch the stride past the workspace, so "traverse L per half-cycle" stops binding).

WHY SLIP ALONE CANNOT GIVE THE PROOF: an ideal SLIP runner is lossless (the spring conserves
energy, flight is free), so its top speed is unbounded. The finite ceiling comes from ACTUATION,
via three measured ingredients:

  1. FORCE-AT-SPEED MAP  Fz_max(x, v): the static end-effector force map (measured 2x2 toe
     Jacobian from the workspace scan) DERATED by the torque-speed envelope -- while the foot is
     planted it sweeps backward at exactly v, so joint j runs at v*|dq_j/dx| and its motoring
     torque shrinks linearly to zero at the 22 rad/s no-load speed. Fast stance => weak stance.
     Capped at STRUCT_BW = 3.5 BW (memory/leg-force-map.md's static motor capability; near the
     straight-leg singularity the Jacobian alone would allow unbounded force, which is a
     structural question, not an actuation one).
  2. MEASURED LEG CYCLING LIMIT  T_min(b): fastest measured unloaded periodic back-and-forth of
     the foot over a fore-aft stroke b (bang-bang runs on the welded-torso rig, gain-swept).
     This replaces any hand-modeled swing/reversal cost and is automatically consistent with the
     rail-sweep result (a hand model here previously implied cycle rates the rig measurably
     cannot do).
  3. IMPULSE BALANCE over a steady stride: two stances per stride, each t_st = b/v long,
     mean vertical foot force Fz_bar =>  2 * (b/v) * Fz_bar(v, b) >= m*g*T,  T >= T_min(b).
     Flight time per step falls out as t_fl = (T - 2*t_st)/2 >= 0.

  plus the touchdown-matching ceiling v <= V_TOE = max over the arc of
  (|dx/dq_cam| + |dx/dq_thigh|) * 22  (foot must run backward at ground speed at contact).

Everything optimistic (unloaded swing, unlimited traction, no balance, leg-weight torque
ignored, stance stretch placed wherever the force map is best) => a genuine upper bound for ANY
gait, flight included. NOTE the as-built passive ankle (0.42 BW transmissible,
memory/leg-force-map.md) would forbid flight running entirely; this assumes the stiff strut.

Run:  .venv/Scripts/python.exe leg2d/flight_bound.py
Outputs: results/flight_bound.json, results/flight_bound.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import rail_bound as rb  # noqa: E402

OUT_JSON = rb.RESULTS / "flight_bound.json"
OUT_PNG = rb.RESULTS / "flight_bound.png"

M, G = 15.14, 9.81
BW = M * G
STRUCT_BW = 3.5                        # static motor capability, memory/leg-force-map.md
B_GRID = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]


def local_jacobians(p1):
    """Measured 2x2 toe Jacobian d(toe x,z)/d(q cam,thigh) at each arc sample (local least
    squares over the workspace-scan cloud: quasi-static transmission, droop included)."""
    z = np.load(rb.SCAN_NPZ)
    q_meas, toe = z["q_meas"], z["toe"]
    qs = np.column_stack([p1["q_cam"], p1["q_thigh"]])
    J = np.zeros((len(qs), 2, 2))
    for i, q0 in enumerate(qs):
        sel = np.argsort(np.linalg.norm(q_meas - q0, axis=1))[:12]
        A = q_meas[sel] - q_meas[sel].mean(axis=0)
        B = toe[sel] - toe[sel].mean(axis=0)
        J[i] = np.linalg.lstsq(A, B, rcond=None)[0].T
    # a revolute joint cannot move the toe faster than its lever arm (<= leg reach ~1.15 m);
    # the local LSQ blows up where the scan cloud is near-collinear -- clip to physics
    return np.clip(J, -1.15, 1.15)


def fz_max_profile(p1, J, v, peak=None, no_load=None):
    """Max sustainable vertical foot force at each arc point while sweeping backward at v:
    tau = J^T F within the per-joint envelope (motoring side derated by the sweep-driven joint
    speed, braking side full peak), horizontal force free, capped at STRUCT_BW.
    peak / no_load default to the as-built motor (overridable for sensitivity what-ifs)."""
    peak = rb.PEAK if peak is None else peak
    no_load = rb.NO_LOAD if no_load is None else no_load
    xs = p1["x_grid"]
    dqdx = np.column_stack([np.gradient(p1["q_cam"], xs), np.gradient(p1["q_thigh"], xs)])
    w = -v * dqdx
    fx_grid = np.linspace(-1.5 * BW, 1.5 * BW, 41)
    fz_grid = np.linspace(0.0, STRUCT_BW * BW, 141)
    out = np.zeros(len(xs))
    for i in range(len(xs)):
        a = J[i].T[:, 0]                                 # d tau / d Fx per joint
        b = J[i].T[:, 1]                                 # d tau / d Fz
        tau = (a[None, :, None] * fx_grid[:, None, None]
               + b[None, :, None] * fz_grid[None, None, :])
        motoring = tau * w[i][None, :, None] > 0
        cap = np.where(motoring,
                       peak * np.clip(1.0 - np.abs(w[i]) / no_load, 0.0, 1.0)[None, :, None],
                       peak)
        feas = np.any(np.all(np.abs(tau) <= cap, axis=1), axis=0)
        out[i] = fz_grid[feas].max() if feas.any() else 0.0
    return out


def measure_t_min(p1, base_h, rig_kwargs=None):
    """Measured fastest unloaded periodic foot cycle vs fore-aft stroke b: bang-bang between
    commanded arc poses at mid-arc +- b/2 (same machinery as rail_bound's unconstrained probe),
    gain-swept, steady-state period. rig_kwargs = Rig what-if scales (sensitivity runs)."""
    xs, zs = p1["x_grid"], p1["z_of_x"]
    cc, ct = p1["c_cam"], p1["c_thigh"]
    sgn = p1["sgn"]
    x_c = xs[int(np.argmin(zs))]                         # max-extension point
    x_front = xs[-1] - 0.10                              # bang-bang overshoot past the front
    t_min = {}                                           # boundary snaps the knee fold: margin
    for b_want in B_GRID:
        hi = float(min(x_c + b_want / 2, x_front))
        lo = float(max(hi - b_want, xs[0]))
        b = hi - lo
        if b < 0.15:
            continue
        end_lo = dict(q_t=np.array([np.interp(lo, xs, cc), np.interp(lo, xs, ct)]),
                      x=float(sgn * lo))
        end_hi = dict(q_t=np.array([np.interp(hi, xs, cc), np.interp(hi, xs, ct)]),
                      x=float(sgn * hi))
        best = None
        for kp, kv in ((300.0, 3.0), (1000.0, 8.0), (2000.0, 10.0), (4000.0, 20.0)):
            r = rb.pass2_run(end_lo, end_hi, base_h, kp=kp, kv=kv, sim_s=8.0,
                             rig_kwargs=rig_kwargs)
            # a real cycle covers the commanded stroke; parking in the fold basin shows up as a
            # near-zero extent + stall-guard period and is rejected. Transiently overflying the
            # fold zone is allowed -- T_min models the SWING half too, and the airborne foot may
            # take any path (only the stance stretch must stay on the arc, enforced separately).
            good = r.get("ok") and r["period"] < 2.5 and r["extent"] >= 0.95 * b
            if good and (best is None or r["period"] < best):
                best = r["period"]
        t_min[round(b, 3)] = best
        print(f"[flight] measured T_min({b:.2f} m) = "
              + (f"{best * 1000:.0f} ms  (f = {1 / best:.1f} Hz)" if best else "no clean cycle"))
    return t_min


def compute_v_star(p1, J, base_h, mass=M, peak=None, no_load=None, rig_kwargs=None,
                   v_max=None, dv=0.25):
    """The flight-inclusive bound v* under optional what-if scalings: motor peak/no-load speed
    enter the force-at-speed map and V_TOE, rig_kwargs enters the measured cycling limit, mass
    enters the impulse requirement. Defaults reproduce the headline run bit for bit."""
    no_load_eff = rb.NO_LOAD if no_load is None else no_load
    xs = p1["x_grid"]
    v_toe = float((np.abs(J[:, 0, :]).sum(axis=1) * no_load_eff).max())
    t_min = measure_t_min(p1, base_h, rig_kwargs=rig_kwargs)

    vs = np.arange(1.0, min(v_toe, v_max if v_max is not None else 25.0), dv)
    margin = np.zeros(len(vs))
    detail = None
    for k, v in enumerate(vs):
        fz = fz_max_profile(p1, J, v, peak=peak, no_load=no_load)
        best = 0.0
        pick = None
        for b, T in t_min.items():
            if T is None:
                continue
            nb = max(2, int(round(b / (xs[1] - xs[0]))))
            if nb >= len(xs):
                continue
            fbar = np.convolve(fz, np.ones(nb) / nb, mode="valid").max()
            t_st = b / v
            T_eff = max(T, 2 * t_st)                     # both feet can't overlap in time > T
            ratio = 2.0 * t_st * fbar / (mass * G * T_eff)
            if ratio > best:
                t_fl = max(0.0, (T_eff - 2 * t_st) / 2)
                best, pick = ratio, dict(b=b, fbar=fbar / BW, t_st=t_st, T=T_eff,
                                         t_flight=t_fl, duty=2 * t_st / T_eff,
                                         stride=v * T_eff)
        margin[k] = best
        if best >= 1.0:
            detail = dict(v=float(v), **pick)
    v_star = float(vs[margin >= 1.0].max()) if np.any(margin >= 1.0) else 0.0
    if v_star and v_star >= vs[-1] - dv / 2:
        print(f"[flight] WARNING: v* = {v_star:.2f} sits at the top of the scanned grid "
              f"(v_max {vs[-1]:.2f}) -- rerun with a higher v_max, the bound is not resolved")
    return dict(v_star=v_star, v_toe=v_toe, t_min=t_min, vs=vs, margin=margin, detail=detail)


def main():
    rec, arc_p, p1, base_h = rb.prepare()
    J = local_jacobians(p1)
    fz0 = fz_max_profile(p1, J, 0.01)
    print(f"[flight] static Fz along the arc: median {np.median(fz0) / BW:.2f} BW, "
          f"max {fz0.max() / BW:.2f} BW (struct cap {STRUCT_BW} BW)")
    r = compute_v_star(p1, J, base_h)
    v_toe, t_min = r["v_toe"], r["t_min"]
    vs, margin, detail, v_star = r["vs"], r["margin"], r["detail"], r["v_star"]
    print(f"[flight] touchdown-matching ceiling V_TOE = {v_toe:.1f} m/s")
    d = detail or {}
    print(f"[flight] v* = {v_star:.1f} m/s"
          + (f"  (b={d['b']:.2f} m, Fz_bar={d['fbar']:.2f} BW, t_st={d['t_st'] * 1000:.0f} ms, "
             f"T={d['T'] * 1000:.0f} ms, flight {d['t_flight'] * 1000:.0f} ms/step, "
             f"duty {d['duty']:.2f}, stride {d['stride']:.2f} m)" if d else ""))

    out = dict(mass=M, struct_bw=STRUCT_BW, v_toe=v_toe, grounded_bound=6.55,
               t_min_ms={str(k): (v * 1000 if v else None) for k, v in t_min.items()},
               v_star=v_star, at_v_star=detail)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=130)
    ax.plot(vs, margin, color="#2a78d6", lw=2.2,
            label="best gait over stroke b and stance placement")
    ax.axhline(1.0, color="#8a8f98", lw=1.2, ls="--")
    ax.text(vs[0] + 0.2, 1.04, "feasible above this line", fontsize=9, color="#555")
    if v_star:
        ax.axvline(v_star, color="#2a78d6", lw=1.0, ls=":")
        ax.annotate(f"flight-inclusive bound {v_star:.1f}", (v_star, 0.25), color="#2a78d6",
                    fontsize=10, ha="right", rotation=90, va="bottom",
                    xytext=(-6, 0), textcoords="offset points")
    ax.axvline(6.55, color="#eb6834", lw=1.4, ls="-.")
    ax.annotate("grounded bound 6.55", (6.55, 0.25), color="#eb6834", fontsize=9.5,
                ha="right", rotation=90, va="bottom", xytext=(-6, 0), textcoords="offset points")
    ax.set_xlabel("running speed v [m/s]")
    ax.set_ylabel("available / required vertical impulse")
    ax.set_title("Flight-inclusive ceiling: impulse feasibility vs speed\n"
                 "(measured force-at-speed map x measured leg cycling limit)", fontsize=11.5)
    ax.set_ylim(0, 3.0)
    ax.grid(True, lw=0.5, alpha=0.35)
    ax.legend(fontsize=9.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"wrote {OUT_JSON}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
