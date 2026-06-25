"""Drive a trained SpiderBot policy live with the keyboard (a software joystick).

Open a live viewer and steer the robot with the arrow keys (or WASD). Each key press **nudges the
target command by a fixed step** (default 0.2 m/s forward, 0.3 rad/s yaw), clamped to the policy's
max. So tapping Up: 0 -> 0.2 -> 0.4 m/s ...; tapping Down walks it back down through 0 into reverse.
To protect the robot from impossible step/stop commands, the actual command **ramps smoothly**
toward that stepped target rather than jumping to it.

Controls (focus the viewer window):
    Up    / W     +0.2 m/s forward   (one step)
    Down  / S     -0.2 m/s           (one step; e.g. 0.2 -> 0 -> -0.2)
    Left  / A     +0.3 rad/s yaw (turn left)
    Right / D     -0.3 rad/s yaw (turn right)
    X  or  0      STOP — set target to (0,0) and ramp back to standing
    R             reset the episode
    (the viewer's own keys still work: Space pauses, etc.)

Run (locally, NOT over headless SSH — needs a display):
    .venv/Scripts/python.exe -m rl.joystick --run rl/runs/m2_walk --preset m2_walk
    .venv/Scripts/python.exe -m rl.joystick --run rl/runs/m2_walk --preset m2_walk --vx-step 0.1 --accel 0.4
"""
import argparse
import time
from pathlib import Path

import numpy as np
from mujoco import viewer as mjviewer

from .evaluate import build   # reuse the exact model + VecNormalize loading

# GLFW key codes (mujoco's viewer passes raw GLFW codes as ints; no glfw import needed)
K_RIGHT, K_LEFT, K_DOWN, K_UP = 262, 263, 264, 265
K_W, K_A, K_S, K_D = 87, 65, 83, 68
K_X, K_0, K_R = 88, 48, 82


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir, e.g. rl/runs/m2_walk")
    ap.add_argument("--preset", default="m2_walk", help="config preset the policy was trained with")
    ap.add_argument("--checkpoint", default=None, help="model path w/o .zip (default: final / latest)")
    ap.add_argument("--accel", type=float, default=0.5,
                    help="command ramp rate in normalized units per second (smaller = gentler)")
    ap.add_argument("--vx-step", type=float, default=0.2,
                    help="forward speed increment per key press, in m/s")
    ap.add_argument("--yaw-step", type=float, default=0.3,
                    help="yaw rate increment per key press, in rad/s")
    ap.add_argument("--no-realtime", action="store_true",
                    help="run as fast as possible instead of pacing to wall-clock")
    args = ap.parse_args()

    model, venv, raw = build(Path(args.run), args.preset, args.checkpoint)
    raw._resample_every = 10 ** 9            # we own the command now — stop env auto-resampling
    dt = raw.control_dt
    vx_max, yaw_max = raw.cfg.vx_max, raw.cfg.yaw_max

    # target = stepped setpoint in PHYSICAL units (m/s, rad/s); live = the ramped, normalized
    # command actually fed to the policy. Keys nudge target by a fixed step; live ramps toward it.
    target = np.zeros(2, np.float32)         # [vx m/s, yaw rad/s]
    live = np.zeros(2, np.float32)           # [vx, yaw] each in [-1, 1]
    want_reset = [False]

    def on_key(key):
        if key in (K_UP, K_W):
            target[0] = min(target[0] + args.vx_step, vx_max)
        elif key in (K_DOWN, K_S):
            target[0] = max(target[0] - args.vx_step, -vx_max)
        elif key in (K_LEFT, K_A):
            target[1] = min(target[1] + args.yaw_step, yaw_max)
        elif key in (K_RIGHT, K_D):
            target[1] = max(target[1] - args.yaw_step, -yaw_max)
        elif key in (K_X, K_0):
            target[:] = 0.0
        elif key == K_R:
            want_reset[0] = True
        else:
            return
        print(f"  target  vx={target[0]:+.2f} m/s   yaw={target[1]:+.2f} rad/s")

    print(__doc__.split("Run (")[0])
    print(f"[joystick] vx_max={vx_max} m/s  yaw_max={yaw_max} rad/s   "
          f"step: {args.vx_step} m/s, {args.yaw_step} rad/s   ramp={args.accel}/s\n")

    obs = venv.reset()
    raw._command[:] = live
    step = max(args.accel * dt, 1e-6)        # per-control-step ramp increment (normalized units)

    with mjviewer.launch_passive(raw.model, raw.data, key_callback=on_key) as v:
        while v.is_running():
            t0 = time.perf_counter()

            # normalize the stepped physical target, then ramp the live command toward it
            target_norm = np.array([target[0] / vx_max, target[1] / yaw_max], np.float32)
            live += np.clip(target_norm - live, -step, step)
            raw._command[:] = live

            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = venv.step(a)
            raw._command[:] = live           # env.step pushed the obs frame; keep command pinned
            v.sync()

            if done[0] or want_reset[0]:
                want_reset[0] = False
                obs = venv.reset()
                live[:] = 0.0                # restart from standing; keep the user's target
                raw._command[:] = live

            if not args.no_realtime:
                lag = dt - (time.perf_counter() - t0)
                if lag > 0:
                    time.sleep(lag)


if __name__ == "__main__":
    main()
