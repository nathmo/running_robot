#!/usr/bin/env python3
"""Teach-and-record one leg's walking trajectory by moving it BY HAND.

SAFE / passive: the leg's three motors are held LIMP (streamed SET_CURRENT 0, like
tools/ak_servo_sweep.py does on release) so you can backdrive them, while their broadcast
positions are logged. Nothing ever drives the motors here.

Workflow:
    python fixed_gait/record_trajectory.py --leg right     # records RIGHT (can0), a few takes
    python fixed_gait/record_trajectory.py --leg left      # records LEFT (can1), a few takes
                                                           # -> auto-smooths + exports when both exist

Keys (in the terminal):
    SPACE  start / stop a take   (move the leg through ONE full cycle: start pose -> step -> back)
    u      undo the last take
    q      finish and save

The abduction motor (id 104) does not need to move — it is held fixed and calibrated separately in
playback; only cam (105) and hip (106) trace the gait. Do a few takes; they get averaged +
smoothed. Being off by a few cm/deg on the return is fine — the loop is closed for you.
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

import trajectory as traj

BITRATE = 1_000_000
LEG_CHANNEL = {"right": "can0", "left": "can1"}       # can0 = RIGHT, can1 = LEFT
MOTOR_IDS = [104, 105, 106]                           # abduction, cam, hip
SAMPLE_HZ = 200.0
CAN_PACKET_SET_CURRENT = 1
DEFAULT_DIR = "fixed_gait/trajectories"


# ------------------------------------------------------------------ CAN helpers (servo mode)
def set_current(bus, cid, amps):
    val = int(round(amps * 1000.0))
    bus.send(can.Message(arbitration_id=cid | (CAN_PACKET_SET_CURRENT << 8),
                         data=val.to_bytes(4, "big", signed=True), is_extended_id=True))


def parse_pos(data):
    if len(data) < 2:
        return None
    return int.from_bytes(bytes(data[0:2]), "big", signed=True) * 0.1     # deg


# ------------------------------------------------------------------ non-blocking keyboard
class KeyPoller:
    """Read single keypresses without Enter. POSIX (termios) with a Windows (msvcrt) fallback."""
    def __enter__(self):
        self.win = os.name == "nt"
        if self.win:
            import msvcrt
            self._m = msvcrt
        else:
            import termios
            import tty
            self._t = termios
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *a):
        if not self.win:
            self._t.tcsetattr(self.fd, self._t.TCSADRAIN, self.old)

    def poll(self):
        if self.win:
            if self._m.kbhit():
                return self._m.getch().decode("latin-1", "ignore")
            return None
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# ------------------------------------------------------------------ recording
def read_positions(bus, latest):
    """Drain all pending frames, update latest[id] with the newest position seen."""
    while True:
        msg = bus.recv(timeout=0.0)
        if msg is None:
            break
        if msg.is_extended_id and (msg.arbitration_id & 0xFF) in latest:
            p = parse_pos(msg.data)
            if p is not None:
                latest[msg.arbitration_id & 0xFF] = p


def record(leg, interface):
    ch = LEG_CHANNEL[leg]
    print(f"Opening {interface}:{ch} @ {BITRATE} — {leg.upper()} leg, motors {MOTOR_IDS}")
    bus = can.Bus(interface=interface, channel=ch, bitrate=BITRATE)
    latest = {i: None for i in MOTOR_IDS}

    # keep limp + wait until all three motors have reported at least once
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
        return None

    print("Motors are LIMP — move the leg by hand.")
    print("  SPACE=start/stop take   u=undo last   q=finish+save\n")
    takes = []
    dt = 1.0 / SAMPLE_HZ
    recording = False
    buf_t, buf_p = [], []
    t_take0 = 0.0
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
                    buf_t.append(now - t_take0)
                    buf_p.append([latest[i] for i in MOTOR_IDS])

                k = kp.poll()
                if k == " ":
                    if not recording:
                        recording = True
                        buf_t, buf_p, t_take0 = [], [], time.time()
                        print(f"\n[take {len(takes)+1}] RECORDING...            ")
                    else:
                        recording = False
                        if len(buf_t) > 20:
                            takes.append((np.array(buf_t), np.array(buf_p)))
                            print(f"\n[take {len(takes)}] saved: {len(buf_t)} samples, "
                                  f"{buf_t[-1]:.1f}s                 ")
                        else:
                            print("\n  (take too short, discarded)      ")
                elif k == "u" and not recording and takes:
                    takes.pop()
                    print(f"\n  undid last take -> {len(takes)} left       ")
                elif k in ("q", "\x1b", "\n", "\r"):
                    break

                if (now - last_print) > 0.15:
                    last_print = now
                    pos = "  ".join(f"{n}={latest[i]:+7.1f}"
                                    for n, i in zip(("abd", "cam", "hip"), MOTOR_IDS))
                    state = "REC " if recording else "idle"
                    print(f"  [{state}] takes={len(takes)}  {pos} deg   ", end="\r")

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
    print(f"\nFinished {leg}: {len(takes)} take(s).")
    return takes


def save_raw(leg, takes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"raw_{leg}.npz")
    flat = {"leg": leg, "n": len(takes), "motor_ids": np.array(MOTOR_IDS)}
    for i, (t, p) in enumerate(takes):
        flat[f"t{i}"] = t
        flat[f"p{i}"] = p
    np.savez(path, **flat)
    print(f"saved {len(takes)} raw take(s) -> {path}")
    return path


def load_raw(path):
    z = np.load(path)
    return [(z[f"t{i}"], z[f"p{i}"]) for i in range(int(z["n"]))]


def process_and_export(out_dir, harmonics):
    rp = os.path.join(out_dir, "raw_right.npz")
    lp = os.path.join(out_dir, "raw_left.npz")
    if not os.path.exists(rp):
        print("(no raw_right.npz yet — record the right leg to build the trajectory)")
        return
    right = load_raw(rp)
    left = load_raw(lp) if os.path.exists(lp) else []
    print(f"\nSmoothing + exporting: {len(right)} right take(s), {len(left)} left take(s)")
    data = traj.process(right, left, harmonics=harmonics)
    out = os.path.join(out_dir, "gait_recorded.npz")
    traj.save(out, data)
    print(f"exported trajectory -> {out}")
    _plot(data, right, left, os.path.join(out_dir, "gait_recorded.png"))


def _plot(data, right, left, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping the preview plot)")
        return
    N = data["N"]
    ph = np.linspace(0, 1, N, endpoint=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for col, name, a in ((traj.COL_CAM, "cam", ax[0]), (traj.COL_HIP, "hip", ax[1])):
        for t, p in right:
            u = (t - t[0]) / (t[-1] - t[0])
            a.plot(u, p[:, col], color="0.8", lw=0.8)
        R = np.array([traj.reconstruct(data, "right", x)[col] for x in ph])
        a.plot(ph, R, "b", lw=2, label="right (smoothed)")
        if data["left"] is not None:
            L = np.array([traj.reconstruct(data, "left", x)[col] for x in ph])
            a.plot(ph, L, "r", lw=2, label="left (dephased 180°)")
        a.set_title(f"{name} motor"); a.set_xlabel("phase"); a.set_ylabel("deg")
        a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110)
    print(f"saved preview -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", choices=["right", "left"], help="which leg to record")
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="where raw + exported files go")
    ap.add_argument("--harmonics", type=int, default=8, help="smoothing: Fourier harmonics kept")
    ap.add_argument("--process-only", action="store_true",
                    help="skip recording; just re-smooth + export from existing raw files")
    args = ap.parse_args()

    if args.process_only:
        process_and_export(args.dir, args.harmonics)
        return
    if can is None:
        print("python-can not installed. `pip install python-can` and bring up the CAN bus.")
        sys.exit(1)
    if not args.leg:
        print("choose --leg right  or  --leg left   (or --process-only)")
        sys.exit(1)

    takes = record(args.leg, args.interface)
    if takes:
        save_raw(args.leg, takes, args.dir)
        process_and_export(args.dir, args.harmonics)      # auto-export (uses whatever legs exist)


if __name__ == "__main__":
    main()
