"""Drive a trained command policy with a joystick, the keyboard, or a scripted demo.

  # scripted demo (no input device, works headless, records an mp4)
  python training/teleop.py --run training/runs/teleop --demo --video demo.mp4
  # keyboard in the live viewer:  W/S = speed,  A/D = turn,  SPACE = stop,  R = reset
  python training/teleop.py --run training/runs/teleop --keys
  # gamepad (needs pygame): left stick Y = speed, left stick X = turn
  python training/teleop.py --run training/runs/teleop --gamepad

The stick is mapped onto the range the policy was ACTUALLY TRAINED FOR, read from the run's
curriculum.json (cmd_v_fwd / cmd_v_back / cmd_yaw, written by CommandCurriculumCallback). This is
the deployment half of the command-scaling fix: the policy always sees a command in physical units
with a fixed normalizer, and it is the JOYSTICK MAPPING that moves as the trained envelope grows.
Retrain to a faster policy and the same stick reaches further with no change here and no change to
the policy's input semantics. --max-speed overrides it downward for a cautious first hardware run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import mujoco

from evaluate import build


def trained_box(run: Path, cfg):
    """(v_fwd, v_back, yaw) the policy was trained to accept. Falls back to the config's START box
    rather than its MAX: if the curriculum never ran, assuming the full range is the optimistic
    error, and the optimistic error here is a robot commanded past what it can do."""
    p = run / "curriculum.json"
    if p.exists():
        d = json.loads(p.read_text())
        if "cmd_v_fwd" in d:
            return float(d["cmd_v_fwd"]), float(d["cmd_v_back"]), float(d["cmd_yaw"])
    print("[teleop] WARNING: no cmd_* in curriculum.json — falling back to the START box")
    return cfg.cmd_v_fwd_start, cfg.cmd_v_back_start, cfg.cmd_yaw_start


# ---------- input sources: each returns (fwd, turn) in [-1, 1] ----------
class ScriptedStick:
    """A fixed demo sequence: everything the robot is supposed to be able to do, in order, with
    enough dwell at each command to see whether it actually settled there."""
    SEQ = [(3.0, 0.0, 0.0, "stand still"),
           (5.0, 1.0, 0.0, "walk forward"),
           (3.0, 0.0, 0.0, "stop"),
           (5.0, 0.6, 0.8, "arc left"),
           (5.0, 0.6, -0.8, "arc right"),
           (3.0, 0.0, 0.0, "stop"),
           (4.0, -0.7, 0.0, "back up"),
           (3.0, 0.0, 0.0, "stop"),
           (6.0, 1.0, 0.0, "run forward"),
           (4.0, 0.0, 1.0, "turn in place"),
           (3.0, 0.0, 0.0, "stand still")]

    def __init__(self, dt):
        self.dt, self.t, self.i, self.said = dt, 0.0, 0, False

    def read(self):
        if self.i >= len(self.SEQ):
            return 0.0, 0.0, True
        dur, f, w, label = self.SEQ[self.i]
        if not self.said:
            print(f"[teleop] {label:16s} fwd={f:+.1f} turn={w:+.1f}  ({dur:.0f}s)")
            self.said = True
        self.t += self.dt
        if self.t >= dur:
            self.t, self.i, self.said = 0.0, self.i + 1, False
        return f, w, False


# GLFW key codes for the arrow keys (the viewer hands us raw GLFW codes, and arrows are not
# characters so chr() cannot reach them)
GLFW_RIGHT, GLFW_LEFT, GLFW_DOWN, GLFW_UP = 262, 263, 264, 265


class KeyStick:
    """Viewer key_callback:  W / UP = faster,  S / DOWN = slower,  A / LEFT and D / RIGHT = turn,
    SPACE = centre the stick,  R = reset the episode.

    Each press NUDGES the virtual stick by `step` and it stays there — it does not decay back to
    centre. A held key auto-repeats through the OS, which ramps smoothly; a single tap gives one
    reproducible increment. Emulating a spring-return stick from discrete key events instead reacts
    at the mercy of the OS key-repeat delay, which feels broken at 200 Hz."""
    def __init__(self, step=0.1, dt=0.005):
        self.f = self.w = 0.0
        self.step, self.dt = step, dt
        self.reset_req = False
        self.quit_req = False

    def key(self, keycode):
        ch = chr(keycode).upper() if 0 < keycode < 0x110000 else ""
        if ch == "W" or keycode == GLFW_UP:
            self.f = min(1.0, self.f + self.step)
        elif ch == "S" or keycode == GLFW_DOWN:
            self.f = max(-1.0, self.f - self.step)
        elif ch == "A" or keycode == GLFW_LEFT:
            self.w = min(1.0, self.w + self.step)     # +yaw = left
        elif ch == "D" or keycode == GLFW_RIGHT:
            self.w = max(-1.0, self.w - self.step)
        elif ch == " ":
            self.f = self.w = 0.0
        elif ch == "R":
            self.reset_req = True
        elif ch == "Q":
            self.quit_req = True

    def read(self):
        return self.f, self.w, self.quit_req


class PadStick:
    def __init__(self, deadzone=0.12):
        import pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise SystemExit("[teleop] --gamepad: no joystick found")
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        self.pygame = pygame
        self.dz = deadzone
        print(f"[teleop] gamepad: {self.js.get_name()}")

    def _dz(self, v):
        return 0.0 if abs(v) < self.dz else float(np.clip(v, -1.0, 1.0))

    def read(self):
        self.pygame.event.pump()
        fwd = -self._dz(self.js.get_axis(1))       # stick up = forward
        turn = -self._dz(self.js.get_axis(0))      # stick left = turn left (+yaw)
        quit_ = bool(self.js.get_numbuttons() > 1 and self.js.get_button(1))
        return fwd, turn, quit_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--preset", default=None)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--demo", action="store_true", help="scripted sequence, no input device")
    src.add_argument("--keys", action="store_true", help="keyboard in the live viewer")
    src.add_argument("--gamepad", action="store_true", help="pygame joystick")
    ap.add_argument("--video", default=None, help="record an mp4 (works headless)")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--max-speed", type=float, default=None,
                    help="override the forward cap DOWNWARD (m/s) for a cautious run")
    ap.add_argument("--fixed-base", nargs="?", type=float, const=0.25, default=None,
                    metavar="CLEARANCE",
                    help="clamp all 6 base DOFs and hang the robot CLEARANCE m up (default 0.25) "
                         "— previews the bolted-to-a-stand, legs-in-the-air hardware bring-up rig")
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    model, venv, raw = build(run, args.preset, args.checkpoint)
    if args.fixed_base is not None:
        raw.set_fixed_base(args.fixed_base)
        print(f"[teleop] FIXED-BASE rig: all 6 base DOFs clamped, +{args.fixed_base:.2f} m "
              f"clearance. No ground contact — legs cycle open-loop, balance is not exercised.")
    if not raw.command_mode:
        raise SystemExit(f"[teleop] {run} is objective='{raw.cfg.objective}', not 'command' — "
                         "teleop needs a policy trained with a joystick command channel")
    v_fwd, v_back, yaw_max = trained_box(run, raw.cfg)
    if args.max_speed is not None:
        v_fwd = min(v_fwd, args.max_speed)
        v_back = min(v_back, args.max_speed)
    print(f"[teleop] trained command box: fwd {v_fwd:.2f} / back {v_back:.2f} m/s, "
          f"yaw {yaw_max:.2f} rad/s")

    dt = raw.control_dt
    if args.demo or (not args.keys and not args.gamepad):
        stick = ScriptedStick(dt)
    elif args.gamepad:
        stick = PadStick()
    else:
        stick = KeyStick(dt=dt)

    def to_command(fwd, turn):
        """[-1,1] stick -> physical command. Asymmetric forward/back, because the robot's backward
        envelope is genuinely smaller and pretending otherwise puts it outside its training box."""
        v = fwd * (v_fwd if fwd >= 0 else v_back)
        return v, turn * yaw_max

    # ---- rollout ----
    renderer = writer = None
    if args.video:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(raw.model, 480, 640)
        writer = imageio.get_writer(args.video, fps=int(round(1 / dt)))
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(raw.model, cam)
    cam.distance, cam.elevation = 3.0, -12

    obs = venv.reset()
    max_steps = int(args.seconds / dt)
    log = []
    viewer = None
    if args.keys:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(raw.model, raw.data, key_callback=stick.key)
        print("[teleop] keys:  W/UP faster   S/DOWN slower   A/LEFT & D/RIGHT turn"
              "   SPACE centre   R reset   Q quit\n"
              "         (click the viewer window first so it has keyboard focus)")

    t_next = time.perf_counter()
    for n in range(max_steps):
        fwd, turn, done = stick.read()
        if done or (viewer is not None and not viewer.is_running()):
            break
        if getattr(stick, "reset_req", False):
            obs = venv.reset()
            stick.reset_req = False
        v_cmd, yaw_cmd = to_command(fwd, turn)
        raw.set_command(v_cmd, yaw_cmd)
        a, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, r, d, info = venv.step(a)
        log.append((v_cmd, yaw_cmd, float(raw._vel_body()[0]), float(raw._ang_vel_body()[2])))
        if d[0]:
            print(f"[teleop] episode ended at t={n * dt:.1f}s — resetting")
            obs = venv.reset()
        if writer is not None:
            cam.lookat[:] = raw.data.qpos[:3]
            renderer.update_scene(raw.data, cam)
            writer.append_data(renderer.render())
        if viewer is not None:
            viewer.sync()
            t_next += dt
            lag = t_next - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
            else:
                t_next = time.perf_counter()

    if writer is not None:
        writer.close()
        print(f"[teleop] wrote {args.video}")
    if viewer is not None:
        viewer.close()

    a = np.array(log)
    if len(a):
        moving = np.abs(a[:, 0]) > 1e-6
        still = ~moving
        print(f"\n[teleop] {len(a)} steps ({len(a) * dt:.1f} s)")
        if moving.any():
            print(f"  speed  cmd {a[moving,0].mean():+.2f}  actual {a[moving,2].mean():+.2f}  "
                  f"MAE {np.abs(a[moving,2]-a[moving,0]).mean():.3f} m/s")
            print(f"  yaw    cmd {a[moving,1].mean():+.2f}  actual {a[moving,3].mean():+.2f}  "
                  f"MAE {np.abs(a[moving,3]-a[moving,1]).mean():.3f} rad/s")
        if still.any():
            print(f"  stand  |v| {np.abs(a[still,2]).mean():.3f} m/s over "
                  f"{still.sum() * dt:.1f} s of stand commands")


if __name__ == "__main__":
    main()
