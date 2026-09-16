#!/usr/bin/env python3
"""Thesis figures: the whole control loop on the robot's Raspberry Pi 3B -- latency, jitter and
compute budget at 50, 100 and 200 Hz -- and where the 10 ms of the deployed 100 Hz tick goes.

No new measurement here. Every input is a number measured on the robot and recorded elsewhere in
the repo; this script adds them up, and marks the one that is inferred:

  transport     6.5 ms     MIT command path (2026-08-26 velocity sweep): the phase lag at 30 Hz
                           did not move with kd (-141/-151/-144 deg), i.e. a fixed 6-7 ms delay
  feedback      5.0 ms     the drives broadcast status at exactly 200.0 Hz (2026-08-06)
  policy        3.47 ms    v2 control tick in a process of its own, after GaitEval (2026-09-11)
                5.3-6.1    the same call inside the web UI daemon (arm-time probe, 2026-09-11).
                           The difference is GIL contention -- Flask, the 200 Hz IMU thread, the
                           recorder -- and is what the web interface costs a policy tick
  governor      1.9 ms     SafetyGovernor + winding observer, standalone (daemon.py:231-235);
                           it runs on every control tick of either bundle generation
  CAN send      0.4 ms     send prep + six frames, per loop tick (daemon.py:233)
  CAN receive   0.3 ms     drain + measure + log + black box, per loop tick. INFERRED: the
                           remainder of the daemon's 3.0 ms non-controller budget (daemon.py:275)
  jitter        bench      loop-period sd / p99 / max at each rate, web UI running (Pi loop bench
                           2026-08-06: drain + a 0.3 ms MLP + six writes)

The daemon's loop runs at 100 Hz (daemon.TICK_HZ); a policy at f Hz gets every (100/f)-th tick and
the last frame is re-sent on the ticks between (daemon.py _tick_policy), so one policy period holds
max(1, 100/f) loop ticks of CAN I/O. Latency is sensor to torque: a feedback frame waits half a broadcast
period for the next drain on average, the tick computes and sends, the frame crosses the transport
delay, and the zero-order hold keeps that torque for one period -- half of one on average.

Outputs results/loop_budget_table.{pdf,png}, results/loop_budget_100hz.{pdf,png} and
results/loop_timeline_100hz.{pdf,png}; prints the table as markdown.
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

# ---------------------------------------------------------------- measured inputs, ms
TRANSPORT = 6.5
FB_PERIOD = 5.0
POLICY_ALONE = 3.47
POLICY_DAEMON = (5.3, 6.1)
NETS = 1.79
GOVERNOR = 1.9
CAN_SEND = 0.4
CAN_RX = 0.3                  # inferred, see the docstring
LOOP_HZ = 100.0               # daemon.TICK_HZ
MIN_RATE_FRAC = 0.90          # daemon.POLICY_MIN_RATE_FRAC: a run is killed below 90 % of nominal

# loop period with the web UI running, Pi loop bench 2026-08-06: (sd, p99, max), ms
BENCH = {50: (2.151, 22.41, 49.29), 100: (0.251, 10.40, 12.28), 200: (0.141, 5.21, 7.46)}
RATES = (50, 100, 200)
DEPLOYED = 100

# ---------------------------------------------------------------- palette
# the table is black and grey only; the budget bar is #e30613 and its tints toward white, in bar
# order. Tints stop at 40 % white: white text is 4.9:1 on the base and ~3:1 on the lightest.
RED = "#e30613"


def tint(c, f):
    """c mixed with white by fraction f."""
    return "#" + "".join(f"{round(int(c[i:i + 2], 16) * (1 - f) + 255 * f):02x}"
                         for i in (1, 3, 5))


C_SEG = {k: tint(RED, f) for k, f in
         (("policy", 0.0), ("web", 0.1), ("gov", 0.2), ("can", 0.3), ("idle", 0.4))}
INK, INK2, MUTED, HAIR, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#ffffff"
TINT = "#f2f2f0"              # the deployed row's band
PILL = "#dcdbd5"              # the "deployed" tag
METER = INK2                  # budget used


def mid(r):
    return 0.5 * (r[0] + r[1])


def row(hz):
    period = 1000.0 / hz
    can = max(1.0, LOOP_HZ / hz) * (CAN_SEND + CAN_RX)   # loop ticks of CAN I/O per period
    work = {k: p + GOVERNOR + can for k, p in
            (("lo", POLICY_DAEMON[0]), ("mid", mid(POLICY_DAEMON)), ("hi", POLICY_DAEMON[1]),
             ("alone", POLICY_ALONE))}
    to_bus = CAN_RX + mid(POLICY_DAEMON) + GOVERNOR + CAN_SEND
    wait, hold = 0.5 * FB_PERIOD, 0.5 * period
    sd, p99, mx = BENCH[hz]
    realised = min(float(hz), 1000.0 / work["mid"])
    return {"hz": hz, "period": period, "can": can, "work": work,
            "used": work["mid"] / period, "used_lo": work["lo"] / period,
            "used_hi": work["hi"] / period, "used_alone": work["alone"] / period,
            "wait": wait, "to_bus": to_bus, "hold": hold,
            "latency": wait + to_bus + TRANSPORT + hold,
            "jit_sd": sd, "jit_p99": p99 - period, "jit_max": mx - period,
            "fits": work["mid"] <= period, "realised": realised,
            "killed": realised < MIN_RATE_FRAC * hz}


# ---------------------------------------------------------------- drawing helpers
def canvas(w, h):
    """A figure whose data coordinates are inches, so rounded patches stay round."""
    fig = plt.figure(figsize=(w, h), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out.with_suffix(".pdf"), "+ .png")


def pct(u):
    return f"{u * 100:.0f} %"


# ---------------------------------------------------------------- figure 1: the table


def table_figure(rows, out):
    W, H = 7.0, 4.1
    fig, ax = canvas(W, H)
    L = 0.25
    ax.text(L, H - 0.16, "Why the policy runs at 100 Hz", fontsize=17, weight="bold",
            color=INK, va="top")
    ax.text(L, H - 0.56, "The whole control loop on the robot's Raspberry Pi 3B, with the web "
            "interface running", fontsize=8.5, color=INK2, va="top")

    X = {"rate": 0.25, "period": 1.3, "budget": 2.15, "lat": 4.45, "jit": 5.7}
    heads = [("rate", "Policy rate", ""), ("period", "Period", ""),
             ("budget", "Compute budget used", "bar = web UI on,  ○ = web UI off"),
             ("lat", "Latency", "sensor to torque, mean"),
             ("jit", "Jitter", "loop period, p99")]
    yh = H - 1.0
    for k, name, sub in heads:
        ax.text(X[k], yh, name, fontsize=7.5, weight="bold", color=INK2, va="baseline")
        if sub:
            ax.text(X[k], yh - 0.15, sub, fontsize=5.8, color=INK2, va="baseline")
    top = yh - 0.24
    ax.plot([L - 0.08, W - L + 0.08], [top, top], color=BASE, lw=0.8)

    RH, M = 0.62, 0.95                                # row height; meter length at 100 %
    for i, r in enumerate(rows):
        yc = top - RH * (i + 0.5)
        hz = r["hz"]
        if hz == DEPLOYED:
            ax.add_patch(FancyBboxPatch((L - 0.08, yc - RH / 2 + 0.035), W - 2 * L + 0.16,
                                        RH - 0.07, boxstyle="round,pad=0,rounding_size=0.07",
                                        fc=TINT, ec="none", zorder=0))
        if i:
            ax.plot([L - 0.08, W - L + 0.08], [yc + RH / 2] * 2, color=HAIR, lw=0.6, zorder=0)
        dim = INK if r["fits"] else MUTED
        star = "" if r["fits"] else "*"

        # rate + period
        ax.text(X["rate"], yc + (0.07 if hz == DEPLOYED else 0), f"{hz} Hz", fontsize=12.5,
                weight="bold", color=INK, va="center")
        if hz == DEPLOYED:
            ax.text(X["rate"] + 0.04, yc - 0.17, "deployed", fontsize=6, color=INK, va="center",
                    bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.5", fc=PILL,
                              ec="none"))
        ax.text(X["period"], yc, f"{r['period']:g} ms", fontsize=10, color=INK, va="center")

        # budget: number + meter (track = 100 %, overflow drawn past the end tick)
        u = r["used"]
        x0 = X["budget"]
        ax.text(x0, yc + 0.1, pct(u), fontsize=12.5, weight="bold", color=INK, va="center")
        ax.text(x0 + (0.62 if u >= 1 else 0.5), yc + 0.09,
                f"{r['work']['mid']:.1f} of {r['period']:g} ms", fontsize=6.5, color=INK2,
                va="center")
        ym, mh = yc - 0.15, 0.075
        ax.add_patch(Rectangle((x0, ym - mh / 2), M, mh, fc=HAIR, ec="none"))
        ax.add_patch(Rectangle((x0, ym - mh / 2), min(u, 1.0) * M, mh,
                               fc=METER, ec="none"))
        if u > 1:
            ax.add_patch(Rectangle((x0 + M + 0.012, ym - mh / 2), (u - 1) * M - 0.012, mh,
                                   fc=METER, ec="none"))
        ax.plot([x0 + M] * 2, [ym - 0.075, ym + 0.075], color=INK, lw=0.8)
        ax.plot(x0 + r["used_alone"] * M, ym, "o", ms=4.6, mfc=SURFACE, mec=INK, mew=0.9,
                zorder=5)

        # latency + jitter
        ax.text(X["lat"], yc + 0.08, f"{r['latency']:.1f} ms{star}", fontsize=12.5,
                weight="bold", color=dim, va="center")
        ax.text(X["lat"], yc - 0.14, f"incl. {r['hold']:.1f} ms hold", fontsize=6.5,
                color=INK2, va="center")
        ax.text(X["jit"], yc + 0.08, f"+{r['jit_p99']:.2f} ms{star}", fontsize=12.5,
                weight="bold", color=dim, va="center")
        ax.text(X["jit"], yc - 0.14, f"worst +{r['jit_max']:.1f} ms", fontsize=6.5,
                color=INK2, va="center")

    ybot = top - RH * len(rows)
    ax.plot([L - 0.08, W - L + 0.08], [ybot, ybot], color=BASE, lw=0.8)
    d = rows[[r["hz"] for r in rows].index(DEPLOYED)]
    notes = [
        f"Budget = policy tick inside the web-UI daemon ({POLICY_DAEMON[0]}–{POLICY_DAEMON[1]}"
        f" ms, midpoint; {pct(d['used_lo'])}–{pct(d['used_hi'])} at {DEPLOYED} Hz) + safety "
        f"governor {GOVERNOR} ms + CAN I/O {CAN_SEND + CAN_RX:.1f} ms per {1000 / LOOP_HZ:g} ms "
        "loop tick.",
        f"○ web UI stopped: the policy tick falls to {POLICY_ALONE} ms in a process of its "
        "own.",
        f"Latency = feedback wait {0.5 * FB_PERIOD:.1f} ms + drain to bus {d['to_bus']:.1f} ms + "
        f"CAN transport {TRANSPORT} ms (measured, MIT sweep) + zero-order hold of half a period.",
        "Jitter = loop-period overrun, Pi loop bench 2026-08-06, web UI running (light 0.3 ms "
        "policy).",
        *(f"* nominal: the {r['work']['mid']:.1f} ms tick exceeds the {r['period']:g} ms period "
          f"(realised {r['realised']:.0f} Hz)." for r in rows if not r["fits"]),
    ]
    for j, n in enumerate(notes):
        ax.text(L, ybot - 0.2 - 0.15 * j, n, fontsize=5.6, color=INK2, va="baseline")
    save(fig, out)


# ---------------------------------------------------------------- figure 2: the 100 Hz budget
def budget_figure(r, out):
    W, L = 4.6, 0.3
    period = r["period"]
    web = mid(POLICY_DAEMON) - POLICY_ALONE
    idle = period - r["work"]["mid"]
    notes = textwrap.wrap(
        f"Policy tick measured on the robot: {POLICY_ALONE} ms in its own process, "
        f"{POLICY_DAEMON[0]}–{POLICY_DAEMON[1]} ms inside the web-UI daemon (2026-09-11); "
        f"web overhead = the difference at the midpoint ({POLICY_DAEMON[0] - POLICY_ALONE:.1f}"
        f"–{POLICY_DAEMON[1] - POLICY_ALONE:.1f} ms).", 98) + textwrap.wrap(
        f"Safety governor {GOVERNOR} ms and CAN send {CAN_SEND} ms per loop tick measured "
        f"(2026-09-01); CAN receive + log {CAN_RX} ms per loop tick inferred from the daemon's "
        "3.0 ms non-controller budget.", 98)
    col_h = 3.6                                        # the column, top = 0 ms, bottom = period
    H = 1.4 + col_h + 0.35 + 0.14 * len(notes)
    fig, ax = canvas(W, H)

    ax.text(L, H - 0.16, f"Where the {period:g} ms go\nat {r['hz']} Hz", fontsize=19,
            weight="bold", color=INK, va="top", linespacing=1.05)
    ax.text(L, H - 0.86, textwrap.fill(f"One v2 policy tick on the robot's Raspberry Pi 3B, web "
            f"interface running: {r['work']['mid']:.1f} ms of work, {idle:.1f} ms idle", 60),
            fontsize=8.5, color=INK2, va="top", linespacing=1.3)

    top = H - 1.4

    def y(ms):
        return top - col_h * ms / period

    segs = [("policy", "Policy inference", POLICY_ALONE,
             f"both neural nets {NETS} ms\n+ gait generator + observation"),
            ("web", "Web interface overhead", web, "GIL contention: Flask,\nIMU thread, recorder"),
            ("gov", "Safety governor", GOVERNOR, "joint limits + winding\ntemperature observer"),
            ("can", "CAN I/O", r["can"], "once per tick:\ndrain + 6 frames"),
            ("idle", "Idle", idle, "")]
    xa, xb, gap = L, L + 1.35, 0.028
    xc, xl = 0.5 * (xa + xb), xb + 0.18
    t = 0.0
    for key, name, ms, desc in segs:
        ya, yb = y(t), y(t + ms)
        ax.add_patch(FancyBboxPatch((xa, yb + gap / 2), xb - xa, ya - yb - gap,
                                    boxstyle="round,pad=0,rounding_size=0.05", fc=C_SEG[key],
                                    ec="none", zorder=2))
        ym = 0.5 * (ya + yb)
        if ya - yb < 0.4:                              # too short for two lines
            ax.text(xc, ym, f"{ms:.2f} ms · {pct(ms / period)}", fontsize=9, weight="bold",
                    color="white", ha="center", va="center", zorder=3)
        else:
            ax.text(xc, ym + 0.06, f"{ms:.2f} ms", fontsize=10, weight="bold", color="white",
                    ha="center", va="center", zorder=3)
            ax.text(xc, ym - 0.1, pct(ms / period), fontsize=7, color="white", ha="center",
                    va="center", zorder=3)
        if desc:
            ax.text(xl, ym + 0.02, name, fontsize=8.5, weight="bold", color=INK, va="bottom")
            ax.text(xl, ym - 0.02, desc, fontsize=6.6, color=INK2, va="top", linespacing=1.25)
        else:
            ax.text(xl, ym, name, fontsize=8.5, weight="bold", color=INK, va="center")
        t += ms

    yf = top - col_h - 0.3
    for j, n in enumerate(notes):
        ax.text(L, yf - 0.14 * j, n, fontsize=5.6, color=INK2, va="baseline")
    save(fig, out)


# ---------------------------------------------------------------- figure 3: the timeline
def schedule(slot, durations, n):
    """Start times of n ticks of a loop that sleeps until next_t += slot and, after an overrun,
    restarts next_t from the end of the tick -- the rule in daemon.py _loop and run_policy.py.
    Tick i lasts durations[i % len(durations)]; returns (start, kind) pairs."""
    t, next_t, out = 0.0, 0.0, []
    for i in range(n):
        k = i % len(durations)
        out.append((t, k))
        next_t += slot
        t = max(next_t, t + durations[k])
        next_t = max(next_t, t)
    return out


def fmt(d):
    s = f"{d:.2f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def timeline_figure(out):
    W, X0, T0, T1, L = 7.2, 1.45, -4.0, 37.0, 0.25
    S = (W - 0.25 - X0) / (T1 - T0)

    def x(t):
        return X0 + (t - T0) * S

    web = mid(POLICY_DAEMON) - POLICY_ALONE
    panels = [
        (f"Web-UI daemon: {1000.0 / LOOP_HZ:g} ms loop slots", 1000.0 / LOOP_HZ,
         [[("can", CAN_RX), ("policy", POLICY_ALONE), ("web", web), ("gov", GOVERNOR),
           ("can", CAN_SEND)]]),
        ("Headless run_policy.py, web UI stopped: 10 ms loop slots", 1000.0 / DEPLOYED,
         [[("can", CAN_RX), ("policy", POLICY_ALONE), ("gov", GOVERNOR), ("can", CAN_SEND)]]),
    ]
    notes = (textwrap.wrap(
        f"Tick durations: CAN drain {CAN_RX} ms (inferred), policy inference {POLICY_ALONE} ms "
        f"alone, web-UI overhead {fmt(web)} ms ({POLICY_DAEMON[0]}–{POLICY_DAEMON[1]} ms in "
        f"the daemon, midpoint, drawn as one block), safety governor {GOVERNOR} ms, CAN send "
        f"{CAN_SEND} ms. Loop rule (daemon.py _loop, run_policy.py): sleep until next_t += slot; "
        "after an overrun, next_t restarts from the end of the tick.", 165) + textwrap.wrap(
        f"Status frames: each drive every {FB_PERIOD:g} ms, not synchronised to the loop; mean "
        f"wait {0.5 * FB_PERIOD} ms. Round trip {TRANSPORT} ms: command frame out to response "
        "visible in a status frame (MIT sweep) = drive command path + motor response + status "
        "path. Hold: a command stays in effect for one policy period; mean age half a period.",
        165))
    PH = 2.2
    H = 1.3 + PH * len(panels) + 0.1 + 0.14 * len(notes)
    fig, ax = canvas(W, H)

    ax.text(L, H - 0.14, "Control-loop timeline", fontsize=17, weight="bold", color=INK,
            va="top")
    ax.text(L, H - 0.52, "Status frame in, policy, command out, response — robot's "
            f"Raspberry Pi 3B at a nominal {DEPLOYED} Hz", fontsize=8.5, color=INK2, va="top")

    # legend: swatches, then marks
    lx, ly = L, H - 0.88
    for key, name in (("policy", "Policy inference"), ("web", "Web-UI overhead"),
                      ("gov", "Safety governor"), ("can", "CAN drain / send")):
        ax.add_patch(Rectangle((lx, ly - 0.05), 0.16, 0.1, fc=C_SEG[key], ec="none"))
        ax.text(lx + 0.22, ly, name, fontsize=6.5, color=INK, va="center")
        lx += 0.22 + len(name) * 0.05 + 0.25
    lx, ly = L, H - 1.06
    for mk, kw, name in (("o", dict(color=INK), "status frame read by the policy"),
                         ("o", dict(color=BASE), "status frame not read"),
                         ("v", dict(color=INK), "command sent")):
        ax.plot(lx + 0.06, ly, mk, ms=4, **kw)
        ax.text(lx + 0.16, ly, name, fontsize=6.5, color=INK, va="center")
        lx += 0.16 + len(name) * 0.05 + 0.25

    bh = 0.24
    for pi, (ptitle, slot, kinds) in enumerate(panels):
        yt = H - 1.3 - PH * pi
        durs = [sum(d for _, d in k) for k in kinds]
        ticks = [(s, k) for s, k in schedule(slot, durs, 16) if s < T1]
        pol = [s for s, k in ticks if k == 0]
        period = pol[1] - pol[0]
        ax.text(L, yt, f"{ptitle}. Policy tick {fmt(durs[0])} ms, policy period {period:.1f} ms "
                f"({1000.0 / period:.0f} Hz)", fontsize=8, weight="bold", color=INK, va="top")
        y1, y2, y3, y4 = yt - 0.42, yt - 0.8, yt - 1.1, yt - 1.38
        for y, name in ((y1, "Status frames in"), (y2, "Raspberry Pi"),
                        (y3, "Commands out"), (y4, "Command in effect")):
            ax.text(L, y, name, fontsize=7, color=INK, va="center")
        for g in range(0, int(T1) + 1, 5):
            ax.plot([x(g)] * 2, [y4 - 0.1, y1 + 0.1], color=HAIR, lw=0.5, zorder=0)
            ax.text(x(g), y4 - 0.15, f"{g} ms" if g + 5 > T1 else f"{g}", fontsize=5.8,
                    color=INK2, ha="center", va="top", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))

        # status frames: the drives' own 5 ms broadcast, phase independent of the loop
        arr = [-0.5 * FB_PERIOD + FB_PERIOD * j for j in range(-1, int(T1 / FB_PERIOD) + 2)]
        used = {max(a for a in arr if a <= s) for s in pol}
        for a in arr:
            if T0 <= a <= T1:
                ax.plot(x(a), y1, "o", ms=3.6, color=INK if a in used else BASE, zorder=3)

        # the Pi: each tick's work in order, then the command frames it sends
        for s, k in ticks:
            t = s
            for key, d in kinds[k]:
                a, b = max(t, T0), min(t + d, T1)
                if b > a:
                    ax.add_patch(Rectangle((x(a), y2 - bh / 2), (b - a) * S, bh, fc=C_SEG[key],
                                           ec=SURFACE, lw=0.5, zorder=2))
                    if key != "can" and b == t + d and d * S > 0.2:
                        ax.text(x(t + d / 2), y2, fmt(d), fontsize=6, color="white",
                                ha="center", va="center", zorder=3)
                t += d
            if s < T1 - 1:
                ax.text(x(s), y2 + bh / 2 + 0.03, "policy" if k == 0 else "re-send",
                        fontsize=5.5, color=INK2, va="bottom")
            send = s + durs[k]
            if send < T1:
                kw = dict(color=INK) if k == 0 else dict(mfc=SURFACE, mec=INK2, mew=0.8)
                ax.plot(x(send), y3, "v", ms=4.5, zorder=3, **kw)

        # what the drive is executing: each new command, one round trip after it was sent
        edges = [T0] + [e for e in (s + durs[0] + TRANSPORT for s in pol) if e < T1] + [T1]
        for j in range(len(edges) - 1):
            a, b = edges[j], edges[j + 1]
            ax.add_patch(Rectangle((x(a), y4 - 0.07), (b - a) * S, 0.14,
                                   fc=("#d6d5ce", "#ebeae4")[j % 2], ec=SURFACE, lw=0.5,
                                   zorder=2))
            if (b - a) * S > 0.45:
                ax.text(x(0.5 * (a + b)), y4, f"command {j}", fontsize=5.5, color=INK,
                        ha="center", va="center", zorder=3)

        # the latency chain of the first policy tick, drawn on the same time axis
        yl = yt - 1.92
        t = pol[0] - 0.5 * FB_PERIOD
        bounds = [t]
        for name, d in (("wait", 0.5 * FB_PERIOD), ("compute", durs[0]),
                        ("round trip", TRANSPORT), ("hold, mean", 0.5 * period)):
            ax.plot([x(t), x(t + d)], [yl, yl], color=INK, lw=0.9)
            ax.text(x(t + d / 2), yl + 0.04, f"{fmt(d)} ms", fontsize=6.3, weight="bold",
                    color=INK, ha="center", va="bottom")
            ax.text(x(t + d / 2), yl - 0.05, name, fontsize=5.8, color=INK2, ha="center",
                    va="top")
            t += d
            bounds.append(t)
        for b in bounds:
            ax.plot([x(b)] * 2, [yl - 0.05, yl + 0.05], color=INK, lw=0.9)
        for b, ytop in zip(bounds[:4], (y1, y2 - bh / 2, y3, y4 - 0.07)):
            ax.plot([x(b)] * 2, [yl + 0.05, ytop], color=INK2, lw=0.6, alpha=0.5, zorder=1)
        ax.text(x(t) + 0.1, yl, f"= {fmt(t - bounds[0])} ms, status frame to response",
                fontsize=7, weight="bold", color=INK, va="center")
        print(f"timeline {pi}: tick {durs[0]:.2f} ms, period {period:.2f} ms "
              f"({1000.0 / period:.1f} Hz), latency {t - bounds[0]:.2f} ms")

    yf = H - 1.3 - PH * len(panels) - 0.05
    for j, n in enumerate(notes):
        ax.text(L, yf - 0.14 * j, n, fontsize=5.6, color=INK2, va="baseline")
    save(fig, out)


def markdown(rows):
    print("\n| Policy rate | Period | Compute used, web UI on | web UI off | Latency (mean) "
          "| Jitter p99 | Worst period |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        star = "" if r["fits"] else " *"
        print(f"| {r['hz']} Hz | {r['period']:g} ms | {r['work']['mid']:.1f} ms = "
              f"**{pct(r['used'])}** ({pct(r['used_lo'])}–{pct(r['used_hi'])}) | "
              f"{pct(r['used_alone'])} | {r['latency']:.1f} ms{star} | "
              f"+{r['jit_p99']:.2f} ms{star} | +{r['jit_max']:.1f} ms |")
    for r in rows:
        if not r["fits"]:
            print(f"\n* {r['hz']} Hz: the tick needs {r['work']['mid']:.1f} ms of a "
                  f"{r['period']:g} ms period -> realised {r['realised']:.0f} Hz"
                  f"{', below the daemon kill threshold' if r['killed'] else ''}")
    d = next(r for r in rows if r["hz"] == DEPLOYED)
    print(f"\nlatency at {DEPLOYED} Hz = wait {d['wait']:.1f} + drain-to-bus {d['to_bus']:.1f} + "
          f"transport {TRANSPORT} + hold {d['hold']:.1f} = {d['latency']:.1f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT), help="output directory (default %(default)s)")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial",
                                             "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    rows = [row(hz) for hz in RATES]
    markdown(rows)
    out = Path(args.out)
    table_figure(rows, out / "loop_budget_table")
    budget_figure(next(r for r in rows if r["hz"] == DEPLOYED), out / "loop_budget_100hz")
    timeline_figure(out / "loop_timeline_100hz")


if __name__ == "__main__":
    main()
