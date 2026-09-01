#!/usr/bin/env python3
"""DESKTOP TOOL. Fit the two-node thermal model to blocked-rotor calibration logs.

    # prove the fitter works, and see what a good vs a bad experiment buys you
    python robot/deploy/thermal_fit.py --self-test

    # fit from the WEB UI campaign: torque-saturating bursts + hand-recorded cooldown points
    python robot/deploy/thermal_fit.py --campaign robot/fixed_gait/webui/data/thermal_runs.json \\
        --motor left.thigh --motor-type AKE90-8 --motor-mass 0.85 --r-phase 0.10 \\
        --out robot/deploy/thermal_params.json

    # or from a blocked-rotor hold log
    python robot/deploy/thermal_fit.py --logs data/thermal/thermal_left_thigh_*.npz \\
        --motor-type AKE90-8 --motor-mass 0.85 --r-phase 0.10 \\
        --out robot/deploy/thermal_params.json

TWO EXPERIMENTS FEED THIS, AND THEY MEASURE DIFFERENT THINGS
------------------------------------------------------------
  BLOCKED-ROTOR HOLD  one long constant current, the drive's own temperature logged at 50 Hz. The
                      plateau pins the steady-state gain (and therefore the continuous rating), but
                      only if the hold reaches one -- the case time constant is ~25 min.
  BURST CAMPAIGN      short torque-saturating bursts (1-30 s) with the peak read by hand, plus a
                      hand-recorded cooldown. Short bursts excite the fast WINDING mode that a long
                      hold can barely see through the case, and the cooldown pins the slow one. No
                      hour-long hold required, but it needs the operator to time the peak.
They are complementary; --campaign and --logs use the same model, the same priors and the same
derate, and differ only in the residual and the adequacy checks.

WHAT THIS EXPERIMENT CAN AND CANNOT IDENTIFY
--------------------------------------------
The logs contain ONE observable: the drive's reported temperature, which measures the CASE node.
The model has five free parameters and they are not equally visible in that signal. Pretending
otherwise is how a thermal limiter ends up confidently wrong.

  * k_cu * R_ca            steady-state case rise per amp squared. Pinned by the PLATEAU -- and
                           only if the hold is long enough to reach one. The case time constant is
                           of order 20 minutes, so a 3-minute hold does not measure a plateau, it
                           extrapolates to one, and the extrapolation is the continuous-current
                           rating. MEASURED on synthetic data: a 3-minute hold recovers the
                           continuous rating 31% low with a condition number of 2e6.
  * R_ca * C_c             the cooling time constant. Pinned by the cooldown, the cleanest part of
                           the record because the input is exactly zero.
  * R_wc * C_w             the winding time constant. Reaches the case signal only through the
                           fast transient, attenuated. Needs a sharp step.
  * the SPLIT of R_wc from C_w, and of k_cu from R_ca, are each degenerate along one direction:
    doubling R_wc while halving C_w leaves the case response IDENTICAL but doubles the predicted
    winding-to-case gradient -- the exact number the limiter exists to bound.

THREE THINGS THIS TOOL DOES ABOUT THAT
--------------------------------------
1. It MEASURES the degeneracy (SVD of the Jacobian at the optimum) and prints which directions the
   data does not constrain, rather than reporting five confident numbers.
2. It lets you break both degeneracies with physics instead of data:
     --motor-mass  copper mass x 385 J/kg/K pins C_w        (weigh the motor)
     --r-phase     k_cu = 1.5 * R_phase for a wye FOC drive (milliohm meter across two phases)
   Each is a soft prior, so a genuine disagreement with the data still shows up as a bad fit
   instead of being silently absorbed.
3. It grades the EXPERIMENT, not the residual, and refuses to mark the parameters calibrated
   unless the experiment could have pinned them (hold >= 2 x the case time constant, >= 10 degC of
   swing, both priors supplied, residual under 2 degC). MotorThermalModel will not arm a policy
   run on uncalibrated parameters, so that refusal has teeth.

   The residual is NOT the gate, because it cannot be. Measured in --self-test: a 3-minute hold
   fits the data to 0.09 degC rms with a statistical error bar of ~0% and still gets the continuous
   rating 31% LOW. That error is extrapolation BIAS, and a parameter covariance cannot see its own
   bias. So the reported percentile intervals are a diagnostic of parameter noise only, and the
   actual margin is a stated engineering derate (DERATE, 0.8) applied on top.

The residual is on the CASE temperature only. A good fit of the case does not prove the winding
node is right; it proves the model reproduces everything that was measured. That is the honest
limit of a calibration with no winding sensor, and it is why the limiter derates rather than
running to a computed edge.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

DEPLOY = Path(__file__).resolve().parent
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import thermal as TH                                                 # noqa: E402

PNAMES = ("k_cu", "c_w", "c_c", "r_wc", "r_ca", "p_idle")
CU_CP = 385.0             # J/kg/K, copper
COPPER_FRAC = 0.25        # of total motor mass; a coarse but standard rule for an outrunner
# P = 1.5 * I_q^2 * R_phase for a wye-connected machine driven by a field-oriented controller
# reporting q-axis current. If the drive reports something else this is wrong by a constant, which
# is exactly why it is a PRIOR and not a fixed value.
K_CU_FROM_R_PHASE = 1.5
# Engineering margin applied to the fitted continuous rating before it is deployed. This is NOT a
# statistical quantity and must not be confused with one: the dominant error in a case-only
# calibration is model/extrapolation BIAS (measured in the self-test: 31% low from a short hold,
# with a statistical error bar of ~0%), and a covariance cannot see its own bias. 0.8 is a
# deliberate, stated engineering choice.
DERATE = 0.8


def _params(theta, base):
    v = np.exp(theta)
    return TH.ThermalParams(base.name, k_cu=v[0], c_w=v[1], c_c=v[2], r_wc=v[3], r_ca=v[4],
                            k_fe=0.0, p_idle=v[5], t_trip=base.t_trip, t_derate=base.t_derate,
                            t_warn=base.t_warn, calibrated=False, source="fitting")


def decimate(L, dt):
    """Bin a log onto a uniform dt grid for fitting.

    The current is averaged in the SQUARE (i_eff = sqrt(mean(I^2))), not linearly: the model heats
    on I^2, so preserving the mean square preserves the heat input exactly while a linear average
    would under-count every ripple. Temperature and ambient are averaged normally -- and averaging
    the 1 degC-quantised reading over a bin recovers sub-degree resolution the fit would otherwise
    throw away.

    Fitting at 50 Hz is pointless: the fastest thing in the model is a winding time constant of
    order a minute. It is also ~100x more expensive, and the fitter evaluates the whole record once
    per Jacobian column per iteration."""
    t = np.asarray(L["t"], float)
    if len(t) < 2 or not dt:
        return L
    edges = np.arange(t[0], t[-1] + dt, dt)
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2)
    n = len(edges) - 1
    cnt = np.bincount(idx, minlength=n).astype(float)
    keep = cnt > 0

    def mean(v):
        return np.bincount(idx, weights=np.asarray(v, float), minlength=n)[keep] / cnt[keep]

    out = dict(L)
    out["t"] = (edges[:-1] + 0.5 * dt)[keep]
    out["i"] = np.sqrt(np.maximum(mean(np.asarray(L["i"], float) ** 2), 0.0))
    out["i_cmd"] = np.sqrt(np.maximum(mean(np.asarray(L["i_cmd"], float) ** 2), 0.0))
    out["temp"] = mean(L["temp"])
    out["amb"] = mean(L["amb"])
    return out


def load_log(path):
    with np.load(path, allow_pickle=False) as z:
        a = z["data"]
        meta = json.loads(str(z["meta_json"]))
    # columns: t_s, amps_cmd, amps_meas, erpm, pos_deg, temp_c, amb_c
    return {"t": a[:, 0].copy(), "i_cmd": a[:, 1].copy(), "i": a[:, 2].copy(),
            "temp": a[:, 5].copy(), "amb": a[:, 6].copy(), "meta": meta, "path": str(path)}


def residuals(theta, logs, base, prior=None):
    """Stacked (predicted - measured) case temperature over every log, plus prior residuals."""
    p = _params(theta, base)
    out = []
    for L in logs:
        _, tc = TH.simulate(p, L["t"], L["i"], t_amb=L["amb"], t0=float(L["temp"][0]))
        out.append(tc - L["temp"])
    if prior:
        for idx, mu_log, w in prior:
            out.append(np.array([w * (theta[idx] - mu_log)]))
    return np.concatenate(out)


def jacobian(theta, f0, fn, rel=1e-4):
    """Forward-difference Jacobian in log space -- the step is RELATIVE, so one tolerance is right
    for a 0.15 W/A^2 and a 900 J/K parameter alike."""
    j = np.empty((f0.size, theta.size))
    for k in range(theta.size):
        t2 = theta.copy()
        t2[k] += rel
        j[:, k] = (fn(t2) - f0) / rel
    return j


def levenberg_marquardt(fn, theta0, max_iter=200, tol=1e-10, verbose=False):
    """Compact LM. No scipy: one fewer dependency in a tool that produces a SAFETY parameter."""
    theta = np.array(theta0, float)
    f = fn(theta)
    cost = float(f @ f)
    lam = 1e-3
    for it in range(max_iter):
        j = jacobian(theta, f, fn)
        jtj = j.T @ j
        jtf = j.T @ f
        for _ in range(30):
            try:
                step = np.linalg.solve(jtj + lam * np.diag(np.maximum(np.diag(jtj), 1e-12)), -jtf)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            t2 = theta + step
            f2 = fn(t2)
            c2 = float(f2 @ f2)
            if c2 < cost:
                improve = (cost - c2) / max(cost, 1e-300)
                theta, f, cost = t2, f2, c2
                lam = max(lam * 0.3, 1e-12)
                if verbose:
                    print("  it {:3d}  cost {:.6g}  lam {:.2g}".format(it, cost, lam))
                if improve < tol:
                    return theta, f, cost, it + 1
                break
            lam *= 10
        else:
            break
    return theta, f, cost, max_iter


def covariance(j, resid, n_par):
    """Parameter covariance in LOG space, plus the SVD that shows the degenerate directions."""
    dof = max(j.shape[0] - n_par, 1)
    s2 = float(resid @ resid) / dof
    u, sv, vt = np.linalg.svd(j, full_matrices=False)
    inv = np.where(sv > sv.max() * 1e-12, 1.0 / np.maximum(sv, 1e-300) ** 2, 0.0)
    cov = (vt.T * inv) @ vt * s2
    cond = float(sv.max() / max(sv.min(), 1e-300))
    return cov, cond, sv, vt


def derived(theta, base, t_amb=25.0):
    """The numbers a limiter actually uses, from one parameter draw."""
    p = _params(theta, base)
    return {
        "i_cont": p.i_continuous(t_amb),
        "tau_w": p.tau_w,
        "tau_c": p.tau_c,
        "r_total": p.r_total,
        # winding rise ABOVE the case per amp squared -- the part no sensor on this robot can see
        "dtw_per_a2": p.k_cu * p.r_wc,
    }


def derived_uncertainty(theta, cov, base, t_amb=25.0, n=4000, seed=0):
    """Monte-Carlo the covariance through the derived quantities.

    Sampling rather than a delta method because i_continuous is a nonlinear fixed point in the
    parameters (the copper tempco makes the steady state self-referential), and because the
    5th percentile -- not a symmetric error bar -- is the number that goes into the limiter."""
    rng = np.random.default_rng(seed)
    try:
        lo = np.linalg.cholesky(cov + 1e-12 * np.eye(len(theta)))
        draws = theta + (rng.standard_normal((n, len(theta))) @ lo.T)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(cov)
        lo = v @ np.diag(np.sqrt(np.clip(w, 0, None)))
        draws = theta + (rng.standard_normal((n, len(theta))) @ lo.T)
    # a sample can wander into an unphysical corner; keep it bounded rather than letting one
    # draw dominate the percentile
    draws = np.clip(draws, theta - 3.0, theta + 3.0)
    out = {}
    vals = [derived(d, base, t_amb) for d in draws]
    for k in vals[0]:
        a = np.array([v[k] for v in vals])
        out[k] = {"median": float(np.median(a)), "p05": float(np.percentile(a, 5)),
                  "p95": float(np.percentile(a, 95))}
    return out


# ================================================================================================
# BURST + COOLDOWN campaign (the web-UI experiment)
# ================================================================================================
# A different observable from the blocked-rotor hold, and a complementary one. Each burst saturates
# the torque for 1-30 s, depositing a KNOWN energy (the daemon integrates the measured current), and
# the operator reads the temperature PEAK that follows with an external probe. Short bursts excite
# the fast winding mode, which the long hold can barely see through the case; the hand-recorded
# cooldown then pins the slow one. Between them the campaign covers both time constants without
# anyone having to hold a motor at temperature for an hour.
#
# THE ASSUMPTION THIS RESTS ON, stated because it is easy to violate: each burst starts from
# THERMAL EQUILIBRIUM, so the winding and the case are both at the recorded start temperature. Fire
# a second burst before the first has finished cooling and the winding is already hot, the model
# thinks it started cold, and the fit attributes the difference to a smaller heat capacity. The
# adequacy check below therefore looks at whether the start temperatures actually settle back.
BURST_TAIL_S = 1500.0        # how far past a burst to look for the case peak
BURST_DT = 1.0               # integration step during the burst
TAIL_DT = 5.0                # ...and after it. Both far below the fastest time constant, and the
#                              discretisation is exact, so these cost accuracy only through the
#                              copper tempco, which is a sub-0.1 degC effect over one burst.


def _burst_curve(p, i_rms, duration_s, t0_c, t_amb, tail_s=BURST_TAIL_S):
    """Simulate one burst plus its tail. Returns (t, Tw, Tc)."""
    t = np.concatenate([np.arange(0.0, duration_s, BURST_DT),
                        duration_s + np.arange(0.0, tail_s, TAIL_DT)])
    i = np.where(t < duration_s, float(i_rms), 0.0)
    tw, tc = TH.simulate(p, t, i, t_amb=t_amb, t0=t0_c)
    return t, tw, tc


def _probe_node(tw, tc, probe):
    """Which node the operator's instrument is looking at. Not a detail: the winding and the case
    differ by k_cu*I^2*R_wc, which for a saturated burst is the largest term in the experiment.
    Getting it wrong moves R_wc by exactly that factor."""
    return tw if probe == "winding" else tc


def burst_residuals(theta, bursts, cools, base, prior=None):
    p = _params(theta, base)
    out = []
    for b in bursts:
        t, tw, tc = _burst_curve(p, b["i_rms"], b["duration_s"], b["t_start_c"], b["t_amb"])
        node = _probe_node(tw, tc, b["probe"])
        out.append(np.array([float(node.max()) - b["t_peak_c"]]))
        # If the operator recorded WHEN they read the peak, that timing is a second, independent
        # constraint -- and it is the one that carries the winding time constant, because the delay
        # to the case peak is set by R_wc*C_w. A peak VALUE alone constrains only the energy.
        if b.get("t_peak_at_s"):
            k = int(np.argmin(np.abs(t - (b["duration_s"] + b["t_peak_at_s"]))))
            out.append(np.array([float(node[k]) - b["t_peak_c"]]))
    for c in cools:
        pts = np.asarray(c["points"], float)
        if len(pts) < 2:
            continue
        grid = np.arange(0.0, float(pts[-1, 0]) + TAIL_DT, TAIL_DT)
        tw, tc = TH.simulate(p, grid, np.zeros_like(grid), t_amb=c["t_amb"], t0=float(pts[0, 1]))
        node = _probe_node(tw, tc, c["probe"])
        out.append(np.interp(pts[:, 0], grid, node) - pts[:, 1])
    if prior:
        for idx, mu_log, w in prior:
            out.append(np.array([w * (theta[idx] - mu_log)]))
    return np.concatenate(out) if out else np.zeros(1)


def load_campaign(store_path, motor=None, ambient_default=25.0):
    """Read the web UI's thermal_runs.json into the shapes the fit wants.

    Only ANNOTATED, un-aborted bursts are usable -- a burst with no operator peak recorded is a
    run that happened, not a measurement."""
    with open(store_path, "r", encoding="utf-8-sig") as f:
        d = json.load(f)
    bursts, cools, skipped = [], [], []
    for r in d.get("runs", []):
        if motor and r["motor"] != motor:
            continue
        if r.get("aborted"):
            skipped.append((r["id"], "aborted: " + str(r["aborted"])))
            continue
        if r.get("t_start_c") is None or r.get("t_peak_c") is None:
            skipped.append((r["id"], "not annotated with start/peak temperatures"))
            continue
        i_rms = float(r["summary"].get("i_rms") or 0.0)
        if i_rms <= 0:
            skipped.append((r["id"], "no measured current"))
            continue
        bursts.append({"id": r["id"], "motor": r["motor"], "i_rms": i_rms,
                       "duration_s": float(r["envelope"]["duration_s"]),
                       "t_start_c": float(r["t_start_c"]), "t_peak_c": float(r["t_peak_c"]),
                       "t_peak_at_s": r.get("t_peak_at_s"),
                       "probe": r.get("probe", "case"),
                       "t_amb": float(r.get("ambient_c") or ambient_default)})
    for c in d.get("cooldowns", []):
        if motor and c["motor"] != motor:
            continue
        if len(c.get("points", [])) < 2:
            skipped.append((c["id"], "cooldown with fewer than 2 points"))
            continue
        cools.append({"id": c["id"], "motor": c["motor"], "points": c["points"],
                      "probe": c.get("probe", "case"),
                      "t_amb": float(c.get("ambient_c") or ambient_default)})
    return bursts, cools, skipped


def burst_adequacy(bursts, cools, p, rms, priors):
    """Can THIS campaign pin the parameters? Same philosophy as the blocked-rotor gate: the
    residual is not the test, the experiment is."""
    durs = sorted({round(b["duration_s"], 1) for b in bursts})
    span = max([c["points"][-1][0] for c in cools], default=0.0)
    npts = sum(len(c["points"]) for c in cools)
    rises = [b["t_peak_c"] - b["t_start_c"] for b in bursts]
    timed = sum(1 for b in bursts if b.get("t_peak_at_s"))
    # a burst fired before the previous one has cooled starts with a hot winding the model believes
    # is cold -- visible as a start temperature that never comes back down toward ambient
    not_settled = sum(1 for b in bursts if b["t_start_c"] - b["t_amb"] > 8.0)
    checks = [
        ("3+ annotated bursts", len(bursts) >= 3, "{} usable".format(len(bursts))),
        ("2+ distinct durations", len(durs) >= 2, "durations {} s".format(durs or "none")),
        ("every rise >= 3 degC", bool(rises) and min(rises) >= 3.0,
         "smallest rise {:.1f} degC".format(min(rises)) if rises else "no bursts"),
        ("1+ peak TIMED", timed >= 1,
         "{} burst(s) recorded when the peak was read -- that is what carries tau_w".format(timed)),
        ("cooldown >= 6 points over 15 min", npts >= 6 and span >= 900.0,
         "{} point(s) over {:.0f} min".format(npts, span / 60.0)),
        ("bursts started from rest", not_settled == 0,
         "{} burst(s) started >8 degC above ambient".format(not_settled)),
        ("residual < 2 degC", rms < 2.0, "rms {:.2f} degC".format(rms)),
        ("k_cu pinned (--r-phase)", "k_cu" in priors, "" if "k_cu" in priors else "not supplied"),
        ("C_w pinned (--motor-mass)", "c_w" in priors, "" if "c_w" in priors else "not supplied"),
    ]
    return {"ok": all(c[1] for c in checks), "checks": [(n, bool(v), d) for n, v, d in checks]}


def fit_campaign(bursts, cools, motor_type, motor_mass=None, copper_frac=COPPER_FRAC,
                 r_phase=None, prior_weight=30.0, t_trip=120.0, t_derate=90.0, t_warn=100.0,
                 t_amb=25.0, verbose=False):
    b0 = TH.DEFAULT_PARAMS.get(motor_type, TH.DEFAULT_PARAMS["AKE90-8"])
    base = TH.ThermalParams(motor_type, b0.k_cu, b0.c_w, b0.c_c, b0.r_wc, b0.r_ca,
                            k_fe=0.0, p_idle=b0.p_idle, t_trip=t_trip, t_derate=t_derate,
                            t_warn=t_warn)
    theta0 = np.log(np.array([base.k_cu, base.c_w, base.c_c, base.r_wc, base.r_ca,
                              max(base.p_idle, 1e-3)]))
    prior, priors_used = [], {}
    if motor_mass:
        cw = float(motor_mass) * float(copper_frac) * CU_CP
        theta0[1] = np.log(cw)
        prior.append((1, np.log(cw), float(prior_weight)))
        priors_used["c_w"] = cw
    if r_phase:
        kcu = K_CU_FROM_R_PHASE * float(r_phase)
        theta0[0] = np.log(kcu)
        prior.append((0, np.log(kcu), float(prior_weight)))
        priors_used["k_cu"] = kcu

    def fn(th):
        return burst_residuals(th, bursts, cools, base, prior)

    theta, f, cost, iters = levenberg_marquardt(fn, theta0, verbose=verbose)
    p = _params(theta, base)
    n_data = f.size - len(prior)
    p.fit_rms_c = float(np.sqrt(np.mean(f[:n_data] ** 2))) if n_data else float("nan")
    p.source = "burst campaign: {} burst(s), {} cooldown point(s), {}".format(
        len(bursts), sum(len(c["points"]) for c in cools), time.strftime("%Y-%m-%d"))
    j = jacobian(theta, f, fn)
    cov, cond, sv, vt = covariance(j, f, len(theta))
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    unc = derived_uncertainty(theta, cov, base, t_amb=t_amb)
    adeq = burst_adequacy(bursts, cools, p, p.fit_rms_c, priors_used)
    p.calibrated = bool(adeq["ok"])
    i_med = unc["i_cont"]["median"]
    return p, {"theta": theta.tolist(), "iters": iters, "cost": cost, "rms_c": p.fit_rms_c,
               "max_err_c": float(np.abs(f[:n_data]).max()) if n_data else float("nan"),
               "sigma_log": sigma.tolist(), "cov": cov.tolist(), "cond": cond,
               "sv": sv.tolist(), "vt": vt.tolist(), "priors": priors_used,
               "n_samples": n_data, "t_amb": t_amb, "adequacy": adeq,
               "temp_span_c": max((b["t_peak_c"] - b["t_start_c"] for b in bursts), default=0.0),
               "hold_s": max((b["duration_s"] for b in bursts), default=0.0),
               "record_s": max((c["points"][-1][0] for c in cools), default=0.0),
               "derived": unc, "i_cont_median": i_med, "i_cont_deploy": i_med * DERATE,
               "campaign": True, "logs": ["burst " + b["id"] for b in bursts]}


def _longest_hold(L):
    """Seconds of the longest continuous stretch with current on. The plateau -- and therefore the
    steady-state gain, and therefore the continuous rating -- lives at the END of that stretch."""
    on = np.asarray(L["i"], float) > 0.5
    t = np.asarray(L["t"], float)
    best = run = 0.0
    for k in range(1, len(t)):
        if on[k]:
            run += t[k] - t[k - 1]
            best = max(best, run)
        else:
            run = 0.0
    return best


def adequacy(logs, p, info_rms, priors, span, hold):
    """Is this EXPERIMENT capable of pinning the continuous rating -- separately from how well the
    curve was fitted?

    This is the gate that matters, and it exists because of what the self-test measured: with
    3-minute holds the fit reproduces the data to 0.09 C rms and still gets the continuous current
    31% LOW, with a statistical error bar of essentially zero. That error is BIAS from
    extrapolating a plateau that was never reached, and no amount of parameter covariance can see
    its own bias. So the fit is graded on the experiment, not on the residual."""
    checks = []
    tau_c = p.tau_c
    checks.append(("hold >= 2 x tau_case",
                   hold >= 2.0 * tau_c,
                   "held {:.0f} s against a {:.0f} s case time constant ({:.1f}x)".format(
                       hold, tau_c, hold / max(tau_c, 1e-9))))
    checks.append(("temperature swing >= 10 C", span >= 10.0,
                   "{:.0f} C of swing".format(span)))
    checks.append(("residual < 2 C", info_rms < 2.0, "rms {:.2f} C".format(info_rms)))
    checks.append(("k_cu pinned (--r-phase)", "k_cu" in priors,
                   "k_cu = 1.5 * R_phase" if "k_cu" in priors else "not supplied"))
    checks.append(("C_w pinned (--motor-mass)", "c_w" in priors,
                   "copper mass x 385 J/kg/K" if "c_w" in priors else "not supplied"))
    return {"ok": all(c[1] for c in checks), "checks": [(n, bool(v), d) for n, v, d in checks]}


def fit(logs, motor_type, motor_mass=None, copper_frac=COPPER_FRAC, r_phase=None,
        prior_weight=30.0, t_trip=120.0, t_derate=90.0, t_warn=100.0, fit_dt=0.5,
        t_amb=25.0, verbose=False):
    logs = [decimate(L, fit_dt) for L in logs]
    b0 = TH.DEFAULT_PARAMS.get(motor_type, TH.DEFAULT_PARAMS["AKE90-8"])
    base = TH.ThermalParams(motor_type, b0.k_cu, b0.c_w, b0.c_c, b0.r_wc, b0.r_ca,
                            k_fe=0.0, p_idle=b0.p_idle, t_trip=t_trip, t_derate=t_derate,
                            t_warn=t_warn)
    theta0 = np.log(np.array([base.k_cu, base.c_w, base.c_c, base.r_wc, base.r_ca,
                              max(base.p_idle, 1e-3)]))
    prior, priors_used = [], {}
    if motor_mass:
        cw = float(motor_mass) * float(copper_frac) * CU_CP
        theta0[1] = np.log(cw)
        prior.append((1, np.log(cw), float(prior_weight)))
        priors_used["c_w"] = cw
    if r_phase:
        kcu = K_CU_FROM_R_PHASE * float(r_phase)
        theta0[0] = np.log(kcu)
        prior.append((0, np.log(kcu), float(prior_weight)))
        priors_used["k_cu"] = kcu

    def fn(th):
        return residuals(th, logs, base, prior)

    theta, f, cost, iters = levenberg_marquardt(fn, theta0, verbose=verbose)
    p = _params(theta, base)
    p.calibrated = True
    p.source = "fit {} log(s), {} samples @ {} s, {}".format(
        len(logs), sum(len(L["t"]) for L in logs), fit_dt, time.strftime("%Y-%m-%d"))

    n_data = sum(len(L["t"]) for L in logs)
    data_resid = f[:n_data]
    p.fit_rms_c = float(np.sqrt(np.mean(data_resid ** 2)))
    j = jacobian(theta, f, fn)
    cov, cond, sv, vt = covariance(j, f, len(theta))
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    unc = derived_uncertainty(theta, cov, base, t_amb=t_amb)
    # what the record actually shows -- the fit's own diagnosis of the EXPERIMENT
    span = max(float(np.max(L["temp"]) - L["temp"][0]) for L in logs)
    hold = max(_longest_hold(L) for L in logs)
    adeq = adequacy(logs, p, info_rms=p.fit_rms_c, priors=priors_used, span=span, hold=hold)
    p.calibrated = bool(adeq["ok"])
    i_med = unc["i_cont"]["median"]
    return p, {"theta": theta.tolist(), "iters": iters, "cost": cost, "rms_c": p.fit_rms_c,
               "max_err_c": float(np.abs(data_resid).max()), "sigma_log": sigma.tolist(),
               "cov": cov.tolist(), "cond": cond, "sv": sv.tolist(), "vt": vt.tolist(),
               "priors": priors_used, "n_samples": n_data, "t_amb": t_amb,
               "logs": [L["path"] for L in logs], "temp_span_c": span, "hold_s": hold,
               "record_s": max(float(L["t"][-1]) for L in logs),
               "derived": unc, "adequacy": adeq,
               "i_cont_median": i_med, "i_cont_deploy": i_med * DERATE}


def report(p, info):
    t_amb = info["t_amb"]
    d = info["derived"]
    print("\nFITTED  {}".format(p.name))
    for i, n in enumerate(PNAMES):
        unit = {"k_cu": "W/A^2", "c_w": "J/K", "c_c": "J/K", "r_wc": "K/W", "r_ca": "K/W",
                "p_idle": "W"}[n]
        tag = "   <- prior" if n in info["priors"] else ""
        print("  {:<7}{:11.4f} {:<7}(+-{:.0f}%){}".format(
            n, getattr(p, n), unit, 100 * info["sigma_log"][i], tag))

    print("\nDERIVED  (median [5th - 95th percentile of PARAMETER NOISE only])")
    print("  continuous current @ {:.0f} C amb, {:.0f} C winding limit :"
          " {:5.1f} A  [{:.1f} - {:.1f}]".format(
              t_amb, p.t_trip, d["i_cont"]["median"], d["i_cont"]["p05"], d["i_cont"]["p95"]))
    print("  winding time constant                              : {:5.0f} s  [{:.0f} - {:.0f}]"
          .format(d["tau_w"]["median"], d["tau_w"]["p05"], d["tau_w"]["p95"]))
    print("  case time constant                                 : {:5.0f} s  [{:.0f} - {:.0f}]"
          .format(d["tau_c"]["median"], d["tau_c"]["p05"], d["tau_c"]["p95"]))
    print("  winding rise ABOVE case, per A^2 (no sensor can see this):"
          " {:5.2f} K/A^2 [{:.2f} - {:.2f}]".format(
              d["dtw_per_a2"]["median"], d["dtw_per_a2"]["p05"], d["dtw_per_a2"]["p95"]))
    print("  Those intervals are PARAMETER NOISE. They cannot see extrapolation bias, which in\n"
          "  this experiment is the LARGER error -- so they are a diagnostic, not a margin.")
    print("\nWHAT THE LIMITER WILL USE")
    print("  continuous current {:.1f} A x {:.2f} engineering derate = {:.1f} A".format(
        d["i_cont"]["median"], DERATE, info["i_cont_deploy"]))

    print("\nEXPERIMENT ADEQUACY  -> {}".format(
        "ADEQUATE (parameters marked calibrated)" if info["adequacy"]["ok"]
        else "NOT ADEQUATE (parameters marked UNCALIBRATED)"))
    for name, okc, detail in info["adequacy"]["checks"]:
        print("  [{}] {:<28} {}".format("ok" if okc else "XX", name, detail))
    if not info["adequacy"]["ok"]:
        print("  MotorThermalModel will REFUSE to arm a policy run on these parameters.")
        if info.get("campaign"):
            print("  Fix the failing rows above -- most often it is burst ENERGY: the case rise "
                  "goes as\n  k_cu*I^2*t / (C_w + C_c), so a readable 3 degC needs thousands of "
                  "joules, not hundreds.")
        else:
            print("  The usual fix is a longer hold: --steps '<A>:{:.0f},0:{:.0f}'".format(
                2.5 * d["tau_c"]["median"], 3.0 * d["tau_c"]["median"]))

    print("\nFIT QUALITY")
    print("  case-temperature residual: rms {:.2f} C, max {:.2f} C over {} samples"
          .format(info["rms_c"], info["max_err_c"], info["n_samples"]))
    print("  record: {:.0f} s long, {:.0f} C of temperature swing".format(
        info["record_s"], info["temp_span_c"]))
    if info["rms_c"] > 2.0:
        print("  !! rms above 2 C -- the two-node model is not explaining this data. Check that "
              "the rotor really was blocked (mechanical power is not in the model) and that "
              "ambient was steady.")

    print("\nIDENTIFIABILITY  (condition number {:.1e})".format(info["cond"]))
    sv = np.asarray(info["sv"])
    vt = np.asarray(info["vt"])
    for k in range(len(sv)):
        if sv[k] < sv[0] * 1e-3:
            direction = "  ".join("{}^{:+.2f}".format(PNAMES[i], vt[k][i])
                                  for i in np.argsort(-np.abs(vt[k]))[:3] if abs(vt[k][i]) > 0.2)
            print("  UNCONSTRAINED direction (sv {:.1e}): {}".format(sv[k], direction))
    if "c_w" not in info["priors"]:
        print("  note: no --motor-mass, so nothing breaks the R_wc / C_w degeneracy. tau_w is "
              "measured; the SPLIT is not, and the split sets the winding-to-case gradient.")
    if "k_cu" not in info["priors"]:
        print("  note: no --r-phase, so k_cu and R_ca trade off along one direction. A milliohm "
              "meter across two phases pins k_cu = 1.5 * R_phase and removes it.")


# ------------------------------------------------------------------------------------------------
def _synth(truth, amps_hold, seed, t_amb=24.0, t0=26.0, log_hz=5.0):
    """Synthetic logs at 5 Hz rather than the robot's 50 Hz: the fitter decimates to 0.5 s bins
    anyway, and generating a 2.4-hour record sample-by-sample in Python is the slow part of this
    test, not the fit."""
    rng = np.random.default_rng(seed)
    logs = []
    for k, (amps, hold, cool) in enumerate(amps_hold):
        t = np.arange(0.0, hold + cool, 1.0 / log_hz)
        i = np.where(t < hold, amps * np.clip(t / 2.0, 0, 1), 0.0)
        _, tc = TH.simulate(truth, t, i, t_amb=t_amb, t0=t0)
        # the drive reports an int8 in whole degrees; that quantisation IS the measurement noise
        temp = np.floor(tc + rng.normal(0, 0.25, tc.shape))
        logs.append({"t": t, "i_cmd": i, "i": i + rng.normal(0, 0.05, i.shape), "temp": temp,
                     "amb": np.full(t.shape, t_amb), "meta": {},
                     "path": "<synthetic {} A, {:.0f} s hold>".format(amps, hold)})
    return logs


def self_test(seed=0, full=False):
    """Fit KNOWN parameters back out of synthetic data quantised exactly as the drive quantises it.

    This is the only test of the fitter that needs no hardware, and more usefully it is a test of
    the EXPERIMENT DESIGN: the same fitter is run against a short schedule and a long one, and the
    difference is the argument for spending two hours on the calibration."""
    truth = TH.ThermalParams("SELFTEST", k_cu=0.182, c_w=95.0, c_c=1050.0, r_wc=0.72, r_ca=1.45,
                             k_fe=0.0, p_idle=0.8)
    print("SELF TEST -- truth: k_cu {:.4f}  C_w {:.0f}  C_c {:.0f}  R_wc {:.3f}  R_ca {:.3f}  "
          "P_idle {:.2f}".format(truth.k_cu, truth.c_w, truth.c_c, truth.r_wc, truth.r_ca,
                                 truth.p_idle))
    print("                    tau_w {:.0f} s   tau_c {:.0f} s   I_cont {:.2f} A   "
          "dTw/A^2 {:.3f} K".format(truth.tau_w, truth.tau_c, truth.i_continuous(24.0),
                                    truth.k_cu * truth.r_wc))
    scenarios = [
        ("SHORT, no priors (3 min holds)", [(6.0, 180.0, 900.0), (9.0, 150.0, 1200.0)], {}),
        ("LONG, no priors (2.6 x tau_c)", [(7.0, 4000.0, 4600.0)], {}),
        ("LONG + motor mass + phase resistance", [(7.0, 4000.0, 4600.0)],
         {"motor_mass": 0.98, "r_phase": truth.k_cu / K_CU_FROM_R_PHASE}),
    ]
    if not full:
        scenarios = scenarios[:1] + scenarios[-1:]
    ok = True
    bad_expected = True                    # the first scenario is the deliberately bad one
    for title, sched, kw in scenarios:
        logs = _synth(truth, sched, seed)
        p, info = fit(logs, "SELFTEST", t_amb=24.0, **kw)
        i_true = truth.i_continuous(24.0)
        e_i = p.i_continuous(24.0) / i_true - 1.0
        e_g = (p.k_cu * p.r_wc) / (truth.k_cu * truth.r_wc) - 1.0
        e_t = p.tau_w / truth.tau_w - 1.0
        print("\n=== {} ===".format(title))
        print("  I_cont {:+.0f}%   dTw/A^2 {:+.0f}%   tau_w {:+.0f}%   rms {:.2f} C   "
              "cond {:.1e}".format(100 * e_i, 100 * e_g, 100 * e_t, info["rms_c"], info["cond"]))
        safe = info["i_cont_deploy"] <= i_true
        print("  deployed rating {:.1f} A vs truth {:.1f} A -> {}   experiment {}".format(
            info["i_cont_deploy"], i_true,
            "SAFE" if safe else "OPTIMISTIC (would under-protect)",
            "ADEQUATE" if info["adequacy"]["ok"] else "flagged INADEQUATE"))
        report(p, info)
        # Two separate claims have to hold for this tool to be trustworthy:
        #   (a) whatever it deploys must not exceed the truth -- the derate has to cover the bias
        #       that a bad experiment introduces, because a covariance cannot see its own bias;
        #   (b) a bad experiment must be REFUSED outright, not merely survived by luck.
        if not safe:
            ok = False
        if bad_expected and info["adequacy"]["ok"]:
            print("  !! a 3-minute-hold experiment was accepted as adequate -- the gate is broken")
            ok = False
        if not bad_expected and not info["adequacy"]["ok"]:
            print("  !! a good experiment was rejected -- the gate is too strict to ever pass")
            ok = False
        bad_expected = False
    print("\n{}: deployed ratings stayed under the truth, and the short experiment was refused."
          .format("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def campaign_self_test(seed=0):
    """Synthesise a burst campaign from KNOWN parameters, quantise it the way a human with a
    thermometer would, and fit it back.

    The operator noise model matters here and is deliberately pessimistic: readings are rounded to
    0.5 degC (a typical IR gun), and the PEAK is read 30 s late with a 20% chance of being read
    early instead -- which is the realistic failure mode, and the one the panel's countdown exists
    to reduce. If the fit only works with perfectly-timed readings it is not a usable procedure."""
    rng = np.random.default_rng(seed)
    truth = TH.ThermalParams("SELFTEST", k_cu=0.182, c_w=95.0, c_c=1050.0, r_wc=0.72, r_ca=1.45,
                             k_fe=0.0, p_idle=0.8)
    t_amb = 24.0
    print("CAMPAIGN SELF TEST -- truth: tau_w {:.0f} s  tau_c {:.0f} s  I_cont {:.2f} A  "
          "dTw/A^2 {:.3f} K".format(truth.tau_w, truth.tau_c, truth.i_continuous(t_amb),
                                    truth.k_cu * truth.r_wc))
    bursts = []
    # Energies chosen so the CASE actually moves several degC -- see thermal_excite's note on why
    # the obvious 12 A / 10 s burst reads 0.1 degC and measures nothing.
    for k, (amps, dur) in enumerate([(25.0, 40.0), (25.0, 80.0), (25.0, 120.0),
                                     (18.0, 100.0), (25.0, 60.0)]):
        t0 = t_amb + rng.uniform(0.0, 1.5)                 # started from rest, as required
        t, tw, tc = _burst_curve(truth, amps, dur, t0, t_amb)
        peak_i = int(np.argmax(tc))
        late = t[peak_i] - dur + (rng.normal(30.0, 15.0) if rng.random() > 0.2
                                  else -rng.uniform(20.0, 60.0))
        late = float(np.clip(late, 5.0, 600.0))
        k_read = int(np.argmin(np.abs(t - (dur + late))))
        bursts.append({"id": "b{}".format(k), "motor": "left.thigh",
                       "i_rms": amps, "duration_s": dur,
                       "t_start_c": round(t0 * 2) / 2.0,
                       "t_peak_c": round(float(tc[k_read]) * 2) / 2.0,
                       "t_peak_at_s": late, "probe": "case", "t_amb": t_amb})
    # one cooldown, hand-sampled every couple of minutes for half an hour
    _, _, tc = _burst_curve(truth, 25.0, 120.0, t_amb + 1.0, t_amb, tail_s=2400.0)
    grid = np.concatenate([np.arange(0.0, 120.0, BURST_DT),
                           120.0 + np.arange(0.0, 2400.0, TAIL_DT)])
    hot = int(np.argmax(tc))
    pts = []
    for sec in (0, 120, 300, 600, 900, 1200, 1500, 1800):
        j = int(np.argmin(np.abs(grid - (grid[hot] + sec))))
        pts.append([float(sec), round(float(tc[j]) * 2) / 2.0])
    cools = [{"id": "c0", "motor": "left.thigh", "points": pts, "probe": "case", "t_amb": t_amb}]

    p, info = fit_campaign(bursts, cools, "SELFTEST", motor_mass=0.98,
                           r_phase=truth.k_cu / K_CU_FROM_R_PHASE, t_amb=t_amb)
    i_true = truth.i_continuous(t_amb)
    print("\n  I_cont {:+.0f}%   dTw/A^2 {:+.0f}%   tau_w {:+.0f}%   rms {:.2f} C".format(
        100 * (p.i_continuous(t_amb) / i_true - 1.0),
        100 * ((p.k_cu * p.r_wc) / (truth.k_cu * truth.r_wc) - 1.0),
        100 * (p.tau_w / truth.tau_w - 1.0), info["rms_c"]))
    safe = info["i_cont_deploy"] <= i_true
    print("  deployed rating {:.1f} A vs truth {:.1f} A -> {}".format(
        info["i_cont_deploy"], i_true, "SAFE" if safe else "OPTIMISTIC (would under-protect)"))
    report(p, info)
    ok = safe and info["adequacy"]["ok"]
    if not info["adequacy"]["ok"]:
        print("  !! a full, well-formed campaign was rejected -- the gate is too strict to pass")
    print("\n{}: the campaign recovers the model and deploys a conservative rating."
          .format("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="*", default=[], help="thermal_*.npz from thermal_calibrate.py")
    ap.add_argument("--campaign", default=None,
                    help="thermal_runs.json written by the web UI's thermal panel")
    ap.add_argument("--motor", default=None,
                    help="with --campaign: fit only this motor's runs (e.g. left.thigh)")
    ap.add_argument("--motor-type", default="AKE90-8")
    ap.add_argument("--motor-mass", type=float, default=None,
                    help="total motor mass, kg -- breaks the R_wc/C_w degeneracy via copper mass")
    ap.add_argument("--copper-frac", type=float, default=COPPER_FRAC)
    ap.add_argument("--r-phase", type=float, default=None,
                    help="per-phase winding resistance, ohms -- pins k_cu = 1.5 * R_phase")
    ap.add_argument("--t-trip", type=float, default=120.0, help="winding trip temperature, degC")
    ap.add_argument("--t-derate", type=float, default=90.0)
    ap.add_argument("--t-warn", type=float, default=100.0)
    ap.add_argument("--fit-dt", type=float, default=0.5,
                    help="decimate the logs to this step before fitting (0 = raw)")
    ap.add_argument("--out", default=str(DEPLOY / "thermal_params.json"))
    ap.add_argument("--ambient", type=float, default=25.0, help="ambient for the reported ratings")
    ap.add_argument("--conservative", action="store_true", default=True,
                    help="store the 5th-percentile rating (default; --no-conservative to disable)")
    ap.add_argument("--no-conservative", dest="conservative", action="store_false")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-test-full", action="store_true")
    ap.add_argument("--campaign-self-test", action="store_true",
                    help="synthesise a burst campaign from known parameters and fit it back")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.self_test or args.self_test_full:
        return self_test(full=args.self_test_full)
    if args.campaign_self_test:
        return campaign_self_test()
    if args.campaign:
        bursts, cools, skipped = load_campaign(args.campaign, args.motor, args.ambient)
        for rid, why in skipped:
            print("  skipping {}: {}".format(rid, why))
        if not bursts:
            raise SystemExit("no usable bursts in {}{}. A burst is usable once it has an operator "
                             "start AND peak temperature recorded."
                             .format(args.campaign,
                                     " for " + args.motor if args.motor else ""))
        print("fitting {} from {} burst(s) and {} cooldown point(s)".format(
            args.motor_type, len(bursts), sum(len(c["points"]) for c in cools)))
        p, info = fit_campaign(bursts, cools, args.motor_type, motor_mass=args.motor_mass,
                               copper_frac=args.copper_frac, r_phase=args.r_phase,
                               t_trip=args.t_trip, t_derate=args.t_derate, t_warn=args.t_warn,
                               t_amb=args.ambient, verbose=args.verbose)
        report(p, info)
        _store(p, info, args)
        return 0
    files = sorted({f for pat in args.logs for f in glob.glob(pat)})
    if not files:
        raise SystemExit("no logs matched. Run robot/deploy/thermal_calibrate.py on the robot "
                         "first, or --self-test to exercise the fitter.")
    logs = [load_log(f) for f in files]
    print("fitting {} from {} log(s), {} samples".format(
        args.motor_type, len(logs), sum(len(L["t"]) for L in logs)))
    p, info = fit(logs, args.motor_type, motor_mass=args.motor_mass,
                  copper_frac=args.copper_frac, r_phase=args.r_phase, t_trip=args.t_trip,
                  t_derate=args.t_derate, t_warn=args.t_warn, fit_dt=args.fit_dt,
                  t_amb=args.ambient, verbose=args.verbose)
    report(p, info)

    _store(p, info, args)
    return 0


def _store(p, info, args):
    out = Path(args.out)
    existing = TH.load_params(out) if out.exists() else {}
    existing[p.name] = p
    TH.save_params(out, existing, meta={"fitted": time.strftime("%Y-%m-%d %H:%M"),
                                        "tool": "thermal_fit.py", "info": info,
                                        "derate": DERATE,
                                        "i_cont_deploy_a": info["i_cont_deploy"]})
    print("\nwrote {}  (calibrated={})".format(out, p.calibrated))
    if not p.calibrated:
        print("The parameters are stored but marked UNCALIBRATED, so the runtime will refuse to "
              "arm a policy run on them. That is deliberate: see EXPERIMENT ADEQUACY above.")


if __name__ == "__main__":
    sys.exit(main())
