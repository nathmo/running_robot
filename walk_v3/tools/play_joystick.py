"""Drive the v3 joystick policy in MuJoCo with the keyboard. W = +10% stick, S = -10%.

This runs the SHIPPING control law -- `robot/deploy/controller_v2.PolicyControllerV2`, the torch-free
numpy path the Pi executes -- against CPU MuJoCo. It needs no JAX, so it runs on the laptop, and what
you feel here is what the robot would run, not a re-implementation of it.

    python walk_v3/tools/play_joystick.py --bundle walk_v3/results/v3_s0.npz

Keys (focus the viewer window):
    W / S     stick up / down by 10% of v_max      SPACE  stick to zero
    F         stick to full                        R      reset the robot
    Q / Esc   quit                                 [ ]    slow-motion / real time

The loop mirrors the trainer: the policy runs at 100 Hz, physics at 1 kHz, and between control ticks the
joint torque is the same software PD the sim used --
`tau = clip(kp*(target - q) - kd*qd, +-min(peak, kt*(V - kt*|qd|)/R))` -- because these are torque
motors, not position servos; MuJoCo closes no loop for us. Every constant comes out of the bundle, so
this cannot silently drift from what was trained.

The readout reports FORWARD speed in the base frame, which is what the command means. World-x is
the wrong number: the policy tracks its own heading, so after a half turn an obedient robot reads as
running backwards -- and while it is pitching over, world-x flatters it (measured: 4.51 world-x
against 3.39 true forward, the difference being body pitch as it fell).

It also prints two headings. `head` is the truth from the simulator; `est` is what the POLICY thinks,
dead-reckoned from the gyro exactly as the robot would. They should agree to a fraction of a degree;
a growing gap is the heading estimate drifting, which is the one thing that would make a v3 policy
veer on hardware but not in sim.

`--lock yaw` pins the heading kinematically and `--lock rail` also pins lateral drift. Both are
DIAGNOSTICS, not features: the real machine has no such constraint, so anything that only works
locked is not deployable.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "walk_v3"))
sys.path.insert(0, str(ROOT / "robot" / "deploy"))   # controller_v2 imports gait_v2 flat

import mujoco
import mujoco.viewer

from robot.deploy.bundle import Bundle
from robot.deploy.controller_v2 import PolicyControllerV2


class Sim:
    """MuJoCo + the deploy control law, wired the way the trainer wires them."""

    def __init__(self, bundle_path, model_path=None):
        self.bundle = Bundle.load(bundle_path)
        m = self.bundle.meta
        mp = model_path or (ROOT / "walk_v3" / m["model_path"])
        self.model = mujoco.MjModel.from_xml_path(str(mp))
        self.data = mujoco.MjData(self.model)

        self.control_dt = float(m["control_dt"])
        self.substeps = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        self.v_max = float(m["v_max"])
        self.v_min = float(m.get("v_min", 0.0))

        # actuator -> joint address maps, exactly plant.py's
        self.act_q = np.array([int(self.model.jnt_qposadr[self.model.actuator_trnid[a, 0]])
                               for a in range(self.model.nu)])
        self.act_d = np.array([int(self.model.jnt_dofadr[self.model.actuator_trnid[a, 0]])
                               for a in range(self.model.nu)])
        self.key_id = 0
        self.gyro_adr = self._sensor_adr("imu_gyro")
        # base DOFs are x,y,z,roll,pitch,yaw at 0..5 (see the model's <joint name="base_*">)
        self.i_y, self.i_yaw = 1, 5
        self.lock = "none"

        # the torque-limit law's constants: from the BUNDLE, so they cannot drift from training
        self.tau_peak = np.asarray(self.bundle["forcerange"], float)
        if self.tau_peak.ndim == 2:
            self.tau_peak = self.tau_peak[:, 1]
        self.kt = np.asarray(m["motor_kt_joint"], float)
        self.r_ohm = np.asarray(m["motor_r_ohm"], float)
        self.v_bus = float(m.get("motor_bus_volts", 48.0))

        self.ctrl = PolicyControllerV2(self.bundle)
        self.v_cmd = 0.0
        self.reset()

    def _sensor_adr(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(self.model.sensor_adr[i]) if i >= 0 else None

    # ---------------------------------------------------------------- sensing
    def _grav_body(self):
        """Unit gravity in the base frame -- R.T @ [0,0,-1], as env._grav_body."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bodyNCS-v1")
        R = self.data.xmat[bid].reshape(3, 3)
        return R.T @ np.array([0.0, 0.0, -1.0])

    def _gyro(self):
        if self.gyro_adr is None:
            return np.zeros(3)
        return np.array(self.data.sensordata[self.gyro_adr:self.gyro_adr + 3], float)

    def _motor_state(self):
        q = self.data.qpos[self.act_q].copy()
        qd = self.data.qvel[self.act_d].copy()
        tau = np.array(self.data.actuator_force, float)
        return q, qd, tau

    # ---------------------------------------------------------------- control
    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mj_forward(self.model, self.data)
        self.ctrl = PolicyControllerV2(self.bundle)       # fresh history, latch and clock
        self.ctrl.set_speed(self.v_cmd, immediate=True)
        q, qd, tau = self._motor_state()
        self.ctrl.start(q, qd, tau, self._grav_body(), self._gyro())
        self.t = 0.0
        self.x0 = float(self.data.qpos[0])
        self._last_cmd = None

    def set_stick(self, frac):
        frac = float(np.clip(frac, 0.0, 1.0))
        self.v_cmd = self.v_min + frac * (self.v_max - self.v_min)
        self.ctrl.set_speed(self.v_cmd)

    @property
    def stick(self):
        return (self.v_cmd - self.v_min) / max(self.v_max - self.v_min, 1e-9)

    def control_tick(self):
        q, qd, tau = self._motor_state()
        cmd = self.ctrl.step(q, qd, tau, self._grav_body(), self._gyro())
        self._last_cmd = cmd
        target = np.asarray(cmd.target, float)
        kp, kd = np.asarray(cmd.kp, float), np.asarray(cmd.kd, float)
        for _ in range(self.substeps):
            qq = self.data.qpos[self.act_q]
            dd = self.data.qvel[self.act_d]
            # torque-speed envelope: the bus cannot push current through back-EMF forever
            v_avail = np.maximum(self.v_bus - self.kt * np.abs(dd), 0.0)
            lim = np.minimum(self.tau_peak, self.kt * v_avail / self.r_ohm)
            self.data.ctrl[:] = np.clip(kp * (target - qq) - kd * dd, -lim, lim)
            mujoco.mj_step(self.model, self.data)
            self._apply_lock()
        self.t += self.control_dt

    # ---------------------------------------------------------------- readouts
    def est_heading_deg(self):
        """What the POLICY believes its heading is -- its own integrated gyro, not the simulator's.

        Printing this next to the truth is the cheapest check that the v3 heading channel is honest:
        if the estimate drifts away from the true yaw here, it will drift on the robot too, and the
        policy will hold a heading that is not the one you pointed it at."""
        return self.ctrl.heading_deg() if hasattr(self.ctrl, "heading_deg") else float("nan")

    def speed(self):
        """FORWARD speed in the BASE frame -- what the command means.

        The world-x velocity is the wrong readout: the policy tracks its own forward axis, so after a
        half turn a perfectly obedient robot reads as running backwards. Rotate the world velocity into
        the base frame and take its x component, exactly as env._vel_body does for the reward.
        """
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bodyNCS-v1")
        R = self.data.xmat[bid].reshape(3, 3)
        return float((R.T @ np.asarray(self.data.qvel[0:3], float))[0])

    def speed_world(self):
        return float(self.data.qvel[0])

    def heading_deg(self):
        return float(np.degrees(self.data.qpos[self.i_yaw]))

    def _apply_lock(self):
        """Hold the robot on a straight line, kinematically. A sim aid, not a policy fix: the real
        machine has no such constraint, so anything that only works locked is not deployable."""
        if self.lock == "none":
            return
        self.data.qpos[self.i_yaw] = 0.0
        self.data.qvel[self.i_yaw] = 0.0
        if self.lock == "rail":                 # also pin lateral drift
            self.data.qpos[self.i_y] = 0.0
            self.data.qvel[self.i_y] = 0.0

    def fallen(self):
        return self._grav_body()[2] > -0.3 or float(self.data.qpos[2]) < 0.45


def headless(args):
    """The same control law, no window: a ladder of stick positions, each held for `--seconds`.

    This is the last independent check the deliverable has. `verify.py` proves the exported numpy
    actor agrees with the JAX one on identical observations; this runs the WHOLE shipping law --
    gait reconstruction, latch, clock, heading integrator, PD -- against a different physics
    backend (CPU MuJoCo, not MJX) and asks whether the robot walks. Nothing here imports JAX, so it
    runs on the laptop and on the Pi's own numpy.

    It prints the policy's own heading estimate next to the simulator's truth. They should agree to
    a fraction of a degree; a growing gap is the integrated-gyro estimate drifting, which is the one
    failure that would make a v3 policy veer on hardware and not in sim.
    """
    sim = Sim(args.bundle, args.model)
    sim.lock = args.lock
    ticks = int(args.seconds / sim.control_dt)
    warm = int(args.warm / sim.control_dt)
    print(f"\n[headless] {Path(args.bundle).name}   v_max {sim.v_max:.2f} m/s   "
          f"{args.seconds:.0f} s per rung, first {args.warm:.0f} s discarded   lock={sim.lock}")
    print(f"{'stick':>6} {'commanded':>10} {'achieved':>9} {'err':>7} {'upright':>8} "
          f"{'|y| m':>7} {'yaw avg':>8} {'yaw end':>8} {'est gap':>7}")
    bad = 0
    for frac in [i / 8.0 for i in range(9)]:
        sim.reset()
        sim.set_stick(frac)
        sim.ctrl.set_speed(sim.v_cmd, immediate=True)
        vs, ys, hs, ds, n, alive = [], [], [], [], 0, True
        for t in range(ticks):
            sim.control_tick()
            if sim.fallen():
                alive = False
                break
            if t >= warm:
                vs.append(sim.speed())
                ys.append(abs(float(sim.data.qpos[sim.i_y])))
                hs.append(abs(sim.heading_deg()))
                ds.append(abs(sim.heading_deg() - sim.est_heading_deg()))
                n += 1
        v = float(np.mean(vs)) if vs else 0.0
        y = float(np.mean(ys)) if ys else float("nan")
        # BOTH heading statistics, because they answer different questions and this project has
        # confused them. The MEAN |yaw| is what the trainer's eval and verify.py report -- how far
        # off straight it sits on average. The FINAL yaw is where it ended up, which is what an
        # operator watching it cross a room actually sees, and for a steady drift the final is
        # about twice the mean. A policy can look fine on the mean and still walk a diagonal.
        h_mean = float(np.mean(hs)) if hs else float("nan")
        h_end = sim.heading_deg()
        # the estimator gap while UPRIGHT only: once it is falling, integrating the gyro of a
        # tumbling body is meaningless and the number says nothing about the channel
        gap = float(np.mean(ds)) if ds else float("nan")
        err = abs(v - sim.v_cmd) if alive and vs else sim.v_cmd
        print(f"{frac * 100:>5.0f}% {sim.v_cmd:>10.2f} {v:>9.2f} {err:>7.2f} "
              f"{'yes' if alive else 'FELL':>8} {y:>7.2f} {h_mean:>7.1f}d {h_end:>7.1f}d "
              f"{gap:>6.1f}d")
        if not alive:
            bad += 1
    print(f"\n{9 - bad}/9 stick positions stayed upright for {args.seconds:.0f} s "
          f"on the CPU arm, through the shipping control law.")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True,
                    help="an .npz written by walk_v3/export.py")
    ap.add_argument("--model", default=None)
    ap.add_argument("--start-stick", type=float, default=0.0)
    ap.add_argument("--step", type=float, default=0.10, help="stick increment per key press")
    ap.add_argument("--headless", action="store_true",
                    help="no viewer: sweep a stick ladder and print achieved speed, survival and "
                         "the heading estimate against the truth")
    ap.add_argument("--seconds", type=float, default=10.0, help="headless: seconds per stick rung")
    ap.add_argument("--warm", type=float, default=2.0, help="headless: seconds discarded as the "
                                                            "start-up transient")
    ap.add_argument("--lock", choices=("none", "yaw", "rail"), default="none",
                    help="none = free; yaw = hold the heading straight; rail = yaw AND no lateral drift")
    args = ap.parse_args()
    if args.headless:
        return headless(args)

    sim = Sim(args.bundle, args.model)
    sim.lock = args.lock
    sim.set_stick(args.start_stick)
    speed_scale = [1.0]
    quit_flag = [False]

    def on_key(keycode):
        ch = chr(keycode) if 0 <= keycode < 0x110000 else ""
        if ch in ("W", "w"):
            sim.set_stick(sim.stick + args.step)
        elif ch in ("S", "s"):
            sim.set_stick(sim.stick - args.step)
        elif ch == " ":
            sim.set_stick(0.0)
        elif ch in ("F", "f"):
            sim.set_stick(1.0)
        elif ch in ("R", "r"):
            sim.reset()
        elif ch == "[":
            speed_scale[0] = max(0.1, speed_scale[0] / 2)
        elif ch == "]":
            speed_scale[0] = min(4.0, speed_scale[0] * 2)
        elif ch in ("Q", "q") or keycode == 256:
            quit_flag[0] = True

    print(f"[play] bundle {Path(args.bundle).name} | v_max {sim.v_max:.2f} m/s | "
          f"{1 / sim.control_dt:.0f} Hz control, {sim.substeps} physics substeps | lock={args.lock}")
    print("[play] W/S = stick +-10%   SPACE = 0   F = full   R = reset   [ ] = slow/fast   Q = quit")

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key,
                                      show_left_ui=False, show_right_ui=False) as v:
        last_print = 0.0
        while v.is_running() and not quit_flag[0]:
            t0 = time.perf_counter()
            sim.control_tick()
            if sim.fallen():
                print(f"[play] FELL at t={sim.t:5.1f}s, stick {sim.stick * 100:3.0f}% -- resetting")
                sim.reset()
            v.sync()
            if sim.t - last_print >= 0.5:
                last_print = sim.t
                print(f"\r[play] t {sim.t:6.1f}s  stick {sim.stick * 100:3.0f}%  "
                      f"cmd {sim.v_cmd:4.2f}  fwd {sim.speed():5.2f} m/s  "
                      f"(world-x {sim.speed_world():5.2f})  "
                      f"head {sim.heading_deg():+6.1f}d  est {sim.est_heading_deg():+6.1f}d     ",
                      end="", flush=True)
            lag = sim.control_dt / speed_scale[0] - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)
    print("\n[play] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
