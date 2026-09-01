#!/usr/bin/env python3
"""ROBOT TOOL. Run a trained policy on DASH-01 in force-control mode, under the safety governor.

    # offline dress rehearsal -- mock bus, mock IMU, no hardware at all
    python robot/deploy/run_policy.py --bundle .../imp_m2_long_204M.npz --mock --max-seconds 5

    # on the robot, with the webui daemon stopped
    sudo systemctl stop runningrobot-webui.service
    python robot/deploy/run_policy.py --bundle .../imp_m2_long_204M.npz \\
        --jointmap robot/deploy/deploy_map.json --thermal robot/deploy/thermal_params.json \\
        --v-cmd 0.0 --max-seconds 20 --deadman-file /tmp/dash_deadman

PHASES
------
  LIMP      streams a zero-gain force-control frame; nothing is commanded. Every run starts and
            ends here, and every abnormal exit lands here.
  APPROACH  slews from the measured pose to the policy's stance at low gain, slowly, watching
            tracking error. This is what makes the first policy tick legal: the controller's
            filter starts AT the stance, so if the legs are somewhere else the first command is a
            step, and a step at 200 N*m/rad is not a gait, it is an impact.
  RUN       the policy, at 200 Hz, every tick through the governor.
  STOP      soft (gains bled out over ~0.3 s, target frozen) or hard (zero gains immediately).

WHAT THIS DOES NOT DO
---------------------
It does not balance. Read the banner the runner prints: `imp_m2_long` is an m2 policy, trained
with the base's Y, roll, pitch and yaw RAILED in simulation. It has never experienced those
degrees of freedom and there is nothing in it that stabilises them. On a free-standing robot it is
open-loop in pitch and roll. Run it on a gantry, a boom, or with the torso otherwise supported,
and treat a fall as the expected outcome of removing that support, not as a surprise.

The runner refuses to start if the webui daemon is running: two processes streaming to the same
drives is a race whose loser is a motor holding whichever command arrived last.
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

# MEASURED on the robot's Pi 3B, 2026-09-01, with the imp_m3d bundle:
#
#     OPENBLAS_NUM_THREADS=1   controller.step p50 4.53 ms
#     OPENBLAS_NUM_THREADS=2                       4.75 ms
#     OPENBLAS_NUM_THREADS=4 (the default, = cores) 5.05 ms
#
# The policy's largest matrix is 593x256. That is far too small to pay for thread synchronisation,
# so OpenBLAS's default of one thread per core makes the control law SLOWER while also putting
# three extra runnable threads in front of the 200 Hz CAN loop and the 200 Hz IMU thread on a
# four-core machine. Pinned to one.
#
# This must run before numpy is imported: OpenBLAS reads the variable when the shared library is
# loaded, and by the time `import numpy` returns it is too late.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

DEPLOY = Path(__file__).resolve().parent
ROBOT = DEPLOY.parent
WEBUI = ROBOT / "fixed_gait" / "webui"
for _p in (str(DEPLOY), str(WEBUI), str(ROBOT / "fixed_gait")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                                                          # noqa: E402
import canio                                                          # noqa: E402
import calibration as calib_mod                                       # noqa: E402

import mit                                                            # noqa: E402
import thermal as TH                                                  # noqa: E402
import jointmap as JM                                                 # noqa: E402
from bundle import Bundle                                             # noqa: E402
from controller import PolicyController                               # noqa: E402
from safety import Limits, SafetyGovernor, STOP_NONE, STOP_HARD       # noqa: E402

TICK_HZ = 200.0
APPROACH_DPS = 25.0          # deg/s of joint travel during the approach -- deliberately crawling
APPROACH_KP = 40.0           # N*m/rad; enough to carry the leg, far too little to hurt anything
APPROACH_KD = 2.0
APPROACH_TRACK_ERR = np.radians(12.0)
# The AK60-39 (abduction) and AKE90-8 (cam, thigh) in MuJoCo actuator order.
MOTOR_TYPE_BY_ACTUATOR = ("AK60-39", "AKE90-8", "AKE90-8", "AK60-39", "AKE90-8", "AKE90-8")


class Runner:
    def __init__(self, args):
        self.args = args
        self.b = Bundle.load(args.bundle)
        self.dt = float(self.b.control_dt)
        if abs(1.0 / self.dt - TICK_HZ) > 1.0:
            raise SystemExit("this bundle wants {:.0f} Hz control; the runner is built for {:.0f}"
                             .format(1.0 / self.dt, TICK_HZ))
        self.jm = JM.JointMap.load(args.jointmap) if args.jointmap else JM.JointMap()
        self.ctrl = PolicyController(self.b)
        self.log = []
        self._last_rx = {}
        self.stop_requested = False
        self.exit_reason = None
        self._setup_thermal()
        self._setup_safety()

    # ------------------------------------------------------------------ setup
    def _setup_thermal(self):
        a = self.args
        params = TH.load_params(a.thermal) if a.thermal else {}
        chain = []
        for t in MOTOR_TYPE_BY_ACTUATOR:
            p = params.get(t) or TH.DEFAULT_PARAMS[t]
            chain.append(p)
        self.thermal = TH.MotorThermalModel(
            chain, dt=self.dt, t_amb=a.ambient, names=list(JM.MODEL_ACTUATORS),
            allow_uncalibrated=a.allow_uncalibrated_thermal)

    def _setup_safety(self):
        b, a = self.b, self.args
        # peak torque from the model the policy trained in; continuous from the thermal fit if we
        # have one, else a conservative fraction of peak
        peak = np.asarray(b["forcerange"], float)
        i_cont = np.array([p.i_continuous(a.ambient) for p in self.thermal.params])
        tau_cont = np.minimum(peak, self.jm.kt_joint * self.jm.kt_efficiency * i_cont)
        # the drive's own configured phase-current limit is a hard ceiling on ANY torque, thermal
        # model or not: no amount of commanded kp produces current the drive will not deliver
        if a.drive_amp_limit:
            peak = np.minimum(peak, self.jm.kt_joint * self.jm.kt_efficiency * a.drive_amp_limit)
        self.limits = Limits.from_bundle(b, tau_cont=tau_cont, deadman_s=a.deadman_s,
                                         telemetry_stale_s=a.telemetry_stale_s)
        self.limits.tau_peak = peak
        self.gov = SafetyGovernor(self.limits, self.dt, thermal=self.thermal,
                                  names=list(JM.MODEL_ACTUATORS))
        self.tau_cont, self.tau_peak = tau_cont, peak

    def banner(self):
        b, m = self.b, self.b.meta
        lock = m["base_lock"]
        railed = [n for n, l in zip("X Y Z roll pitch yaw".split(), lock) if l]
        print("=" * 78)
        print("POLICY  {} / {}   ({} action dims, {} Hz)".format(
            m["run"], m["checkpoint"], m["action_dim"], 1 / self.dt))
        print("COMMAND BOX (trained): fwd {:.2f} m/s, back {:.2f}, yaw {:.2f} rad/s".format(
            m["cmd_v_fwd_trained"], m["cmd_v_back_trained"], m["cmd_yaw_trained"]))
        if railed:
            print("!! RAILED IN TRAINING: {}".format(", ".join(railed)))
        if any(lock[3:]):
            axes = "/".join(n for n, l in zip(("roll", "pitch", "yaw"), lock[3:]) if l)
            print("!! This policy has NEVER experienced free {}. It does not balance in those".format(axes))
            print("!! axes because it has never had to. SUPPORT THE TORSO.")
        print("torque peak  {} N*m".format(np.round(self.tau_peak, 1).tolist()))
        print("torque cont  {} N*m  (thermal, {})".format(
            np.round(self.tau_cont, 1).tolist(),
            "fitted" if not self.thermal.uncalibrated else "UNCALIBRATED PLACEHOLDERS"))
        if self.thermal.uncalibrated:
            print("!! thermal parameters are placeholders -- the winding estimate is a guess")
        print("=" * 78)

    # ------------------------------------------------------------------ preflight
    def preflight(self, buses, motors):
        a = self.args
        ok, why = self.jm.check_ready()
        if not ok and not a.skip_jointmap_check:
            raise SystemExit("REFUSING TO RUN: " + why)
        cal = calib_mod.Calibration.load_or_new()
        if not cal.complete and not a.mock:
            raise SystemExit("REFUSING TO RUN: the zero/direction calibration is incomplete. "
                             "Every joint angle this policy reads is derived from it.")
        if cal.restored_from_disk and not a.mock:
            print("!! calibration was restored from disk. The drives re-randomise their raw origin "
                  "on every power cycle -- re-zero unless you are certain they have not been "
                  "power-cycled since it was captured.")
        return cal

    # ------------------------------------------------------------------ IMU
    def open_imu(self):
        if self.args.no_imu:
            print("!! --no-imu: gravity is faked upright and the fall detector CANNOT fire. "
                  "Bench use only, with the robot physically restrained.")
            return None
        import sensehat
        sh = sensehat.SenseHat(mock=self.args.mock)
        sh.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0 and sh.fast() is None:
            time.sleep(0.02)
        if sh.fast() is None:
            raise SystemExit("no IMU sample after 5 s -- refusing to run a balance-relevant "
                             "controller blind. Use --no-imu only on a restrained bench.")
        if not self.args.mock and not sh.mount.calibrated:
            raise SystemExit("the IMU mount rotation has not been calibrated, so 'up' is in CHIP "
                             "axes, not body axes. Run the mount calibration in the web UI first.")
        return sh

    def read_imu(self, sh, now):
        if sh is None:
            return np.array([0.0, 0.0, -1.0]), np.zeros(3), 0.0
        f = sh.fast()
        if f is None:
            return np.array([0.0, 0.0, -1.0]), np.zeros(3), 1e9
        t, up, gyr = f
        # MuJoCo convention: gravity is world DOWN expressed in body axes, so it is -up_body
        return -np.asarray(up, float), np.asarray(gyr, float), max(0.0, time.time() - t)

    # ------------------------------------------------------------------ CAN
    def send(self, motors, cal, target_model, kp_model, kd_model):
        """Model-frame command -> six force-control frames. Returns the set of clamped fields."""
        norm_deg = self.jm.to_norm_deg(target_model)
        kp_m = np.asarray(kp_model)[self.jm.motor_from_model]
        kd_m = np.asarray(kd_model)[self.jm.motor_from_model]
        clamped = set()
        for i, name in enumerate(paths.MOTOR_NAMES):
            m = motors[name]
            raw_deg = cal.raw(name, float(norm_deg[i]))
            payload, cl = mit.pack(np.radians(raw_deg), float(kp_m[i]), float(kd_m[i]))
            clamped.update(cl)
            canio.force_control(m.bus, m.cid, payload)
        return clamped

    def stream_limp(self, motors):
        p = mit.limp_payload()
        for m in motors.values():
            canio.force_control(m.bus, m.cid, p)
            # belt and braces: a zero-gain force frame commands nothing, and SET_CURRENT 0 is the
            # limp discipline every other tool on this robot uses. Sending both costs 6 frames.
            canio.set_current(m.bus, m.cid, 0.0)

    # ------------------------------------------------------------------ the run
    def execute(self):
        a = self.args
        import play_trajectory as pt
        buses = canio.open_buses(a.interface, sorted(set(paths.SIDE_CHANNEL.values())),
                                 mock=a.mock)
        motors, by_bus = {}, {}
        for side in paths.SIDES:
            ch = paths.SIDE_CHANNEL[side]
            by_bus.setdefault(ch, {})
            for col, role in enumerate(paths.ROLES):
                m = pt.Motor(buses[ch], paths.ROLE_ID[role], side, col)
                motors["{}.{}".format(side, role)] = m
                by_bus[ch][m.cid] = m
        cal = self.preflight(buses, motors)
        sh = None
        phase = "LIMP"
        try:
            # ---- preflight: limp until every drive reports -----------------------------------
            t_end = time.monotonic() + 3.0
            while time.monotonic() < t_end and any(m.pos is None for m in motors.values()):
                self.stream_limp(motors)
                self.drain(buses, by_bus, time.monotonic())
                time.sleep(0.005)
            silent = [n for n, m in motors.items() if m.pos is None]
            if silent:
                raise SystemExit("no status from {} -- never commanding blind".format(silent))
            sh = self.open_imu()
            self.thermal.reset(np.array([float(motors[n].temp) for n in JM.MODEL_TO_MOTOR]),
                               t_amb=a.ambient)

            # ---- APPROACH --------------------------------------------------------------------
            phase = "APPROACH"
            if not self.approach(motors, by_bus, buses, cal, sh):
                self.exit_reason = self.exit_reason or "approach aborted"
                return False

            # ---- RUN -------------------------------------------------------------------------
            phase = "RUN"
            self.run_policy(motors, by_bus, buses, cal, sh)
            return self.gov.stop == STOP_NONE
        except KeyboardInterrupt:
            self.exit_reason = "operator interrupt during {}".format(phase)
            print("\n!! interrupted")
        except SystemExit:
            raise
        except Exception as e:                              # noqa: BLE001 -- never skip the release
            self.exit_reason = "exception during {}: {!r}".format(phase, e)
            import traceback
            traceback.print_exc()
        finally:
            print("releasing: streaming zero-gain force control + 0 A for 0.5 s")
            t_rel = time.monotonic()
            while time.monotonic() - t_rel < 0.5:
                self.stream_limp(motors)
                time.sleep(0.005)
            for b in buses.values():
                try:
                    b.shutdown()
                except Exception:
                    pass
            if sh is not None:
                sh.stop()
        return False

    def drain(self, buses, by_bus, now):
        """Drain both buses and stamp PER-MOTOR arrival times.

        play_trajectory.drain returns None, so a caller cannot tell a live bus from a dead one --
        and a bus-wide 'did anything arrive' would miss the case that actually matters, which is
        ONE drive going quiet while its five neighbours keep talking. The staleness the governor
        sees is therefore the age of the OLDEST motor, not of the newest frame."""
        for ch, bus in buses.items():
            for _ in range(256):
                msg = bus.recv(timeout=0.0)
                if msg is None:
                    break
                if not getattr(msg, "is_extended_id", True):
                    continue
                m = by_bus[ch].get(msg.arbitration_id & 0xFF)
                if m is None:
                    continue
                st = canio.parse_status(msg.data)
                if st:
                    m.update_from(st, self.dt)
                    self._last_rx[id(m)] = now
        if not self._last_rx:
            return 1e9
        return now - min(self._last_rx.values())

    def measure(self, motors, cal):
        """Everything the controller and the governor need, in the MODEL frame."""
        norm = np.array([cal.norm(n, motors[n].pos) for n in paths.MOTOR_NAMES])
        pos = self.jm.to_model_rad(norm)
        amps = np.array([motors[n].cur for n in paths.MOTOR_NAMES])
        temp = np.array([float(motors[n].temp) for n in JM.MODEL_TO_MOTOR])
        err = np.array([int(motors[n].err) for n in JM.MODEL_TO_MOTOR])
        erpm = np.array([motors[n].spd for n in paths.MOTOR_NAMES])
        return pos, amps, temp, err, erpm

    def approach(self, motors, by_bus, buses, cal, sh):
        """Crawl to the policy's stance at low gain. Aborts on tracking error or a governor kill."""
        stance = np.asarray(self.b["nominal_ctrl"], float)
        pt = sys.modules["play_trajectory"]
        pos, *_ = self.measure(motors, cal)
        travel = float(np.max(np.abs(stance - pos)))
        t_total = travel / np.radians(APPROACH_DPS)
        print("APPROACH: {:.1f} deg of travel, {:.1f} s at {:.0f} deg/s".format(
            np.degrees(travel), t_total, APPROACH_DPS))
        start = pos.copy()
        t0 = time.monotonic()
        next_t = t0
        while True:
            now = time.monotonic()
            f = min(1.0, (now - t0) / max(t_total, 1e-6))
            tgt = start + (stance - start) * f
            self.drain(buses, by_bus, now)
            pos, amps, temp, err, _ = self.measure(motors, cal)
            grav, gyro, imu_age = self.read_imu(sh, now)
            # the observer tracks through the approach too: the case-temperature correction
            # should not wait for the policy to start. The governor owns the stepping (see
            # safety.SafetyGovernor.observe) so there is exactly one call site per phase.
            self.gov.observe(np.abs(amps)[self.jm.model_from_motor],
                             drive_temp=temp, t_amb=self.args.ambient)
            if np.any(err):
                print("!! drive error during approach: {}".format(err.tolist()))
                return False
            if float(np.max(np.abs(pos - tgt))) > APPROACH_TRACK_ERR:
                i = int(np.argmax(np.abs(pos - tgt)))
                print("!! {} is {:.1f} deg from its approach target -- stopping. Either the joint "
                      "map is wrong or the leg is obstructed.".format(
                          JM.MODEL_ACTUATORS[i], np.degrees(abs(pos - tgt)[i])))
                return False
            self.send(motors, cal, tgt, np.full(6, APPROACH_KP), np.full(6, APPROACH_KD))
            if f >= 1.0 and float(np.max(np.abs(pos - stance))) < np.radians(3.0):
                print("APPROACH: at stance.")
                return True
            if now - t0 > t_total + 5.0:
                print("!! approach did not converge within its budget")
                return False
            next_t += self.dt
            s = next_t - time.monotonic()
            if s > 0:
                time.sleep(s)
            else:
                next_t = time.monotonic()

    def run_policy(self, motors, by_bus, buses, cal, sh):
        a = self.args
        pt = sys.modules["play_trajectory"]
        pos, amps, temp, err, _ = self.measure(motors, cal)
        grav, gyro, _ = self.read_imu(sh, time.monotonic())
        tau = self.jm.torque_to_model(amps)
        v_cmd = float(np.clip(a.v_cmd, -self.b.cmd_v_back_trained, self.b.cmd_v_fwd_trained))
        if v_cmd != a.v_cmd:
            print("!! command {:+.2f} m/s is outside the box this checkpoint was trained to; "
                  "clamped to {:+.2f}".format(a.v_cmd, v_cmd))
        self.ctrl.start(pos, np.zeros(6), tau, grav, gyro, v_cmd=v_cmd, yaw_cmd=0.0)
        print("RUN: v_cmd {:+.2f} m/s, max {:.0f} s. Ctrl+C for a soft stop.".format(
            v_cmd, a.max_seconds))

        prev_pos = pos.copy()
        t0 = time.monotonic()
        next_t = t0
        n_late = 0
        while True:
            now = time.monotonic()
            t = now - t0
            if t >= a.max_seconds:
                self.gov.kill("max run time {:.0f} s reached".format(a.max_seconds), hard=False)
            if self.stop_requested:
                self.gov.kill("stop requested", hard=False)
            tel_age = self.drain(buses, by_bus, now)
            pos, amps, temp, err, erpm = self.measure(motors, cal)
            grav, gyro, imu_age = self.read_imu(sh, now)
            # Joint velocity from the DIFFERENTIATED position: the drive reports ERPM and nothing
            # in this repo has ever measured the ERPM-to-joint-speed scale (see jointmap). The
            # differentiated value is noisy but unambiguous, and the policy trained against a
            # velocity channel with 0.15 rad/s of noise plus a 0.05 bias, so this is inside the
            # band it has seen.
            vel = (pos - prev_pos) / self.dt
            prev_pos = pos.copy()
            tau = self.jm.torque_to_model(amps)
            cmd = self.ctrl.step(pos, vel, tau, grav, gyro)
            # the governor steps the winding observer itself, off `current` -- one owner
            v = self.gov.step(cmd.target, cmd.kp, cmd.kd, pos, vel, grav, gyro,
                              telemetry_age=tel_age,
                              deadman_age=self.deadman_age(now),
                              drive_temp=temp, drive_err=err,
                              current=np.abs(amps)[self.jm.model_from_motor],
                              t_amb=a.ambient)
            if v.limp:
                self.stream_limp(motors)
            else:
                self.send(motors, cal, v.target, v.kp, v.kd)

            if a.log:
                self.log.append(np.concatenate([
                    [t], pos, vel, tau, amps, temp, grav, gyro, v.target, v.kp, v.kd,
                    self.thermal.t_winding, [cmd.phase, cmd.freq, v.stop]]))
            if v.stop != STOP_NONE and v.limp:
                self.exit_reason = "; ".join(v.reasons)
                print("\nSTOPPED: {}".format(self.exit_reason))
                break
            next_t += self.dt
            s = next_t - time.monotonic()
            if s > 0:
                time.sleep(s)
            else:
                n_late += 1
                next_t = time.monotonic()
        print("ran {:.1f} s, {} late ticks, {}".format(
            time.monotonic() - t0, n_late, self.gov.status()["clamp_counts"] or "no clamping"))

    def deadman_age(self, now):
        f = self.args.deadman_file
        if not f:
            return 0.0
        try:
            return max(0.0, time.time() - os.path.getmtime(f))
        except OSError:
            return 1e9

    def save_log(self):
        if not self.log:
            return
        a = np.asarray(self.log, float)
        base = Path(self.args.out) / "policyrun_{}".format(time.strftime("%Y%m%d_%H%M%S"))
        base.parent.mkdir(parents=True, exist_ok=True)
        meta = {"bundle": str(self.args.bundle), "run": self.b.run,
                "checkpoint": self.b.checkpoint, "v_cmd": self.args.v_cmd,
                "exit_reason": self.exit_reason, "governor": self.gov.status(),
                "thermal_peak_winding_c": self.thermal.peak_w.tolist(),
                "columns": ("t | pos6 | vel6 | tau6 | amps6 | temp6 | grav3 | gyro3 | "
                            "target6 | kp6 | kd6 | t_winding6 | phase | freq | stop")}
        np.savez(str(base) + ".npz", data=a, meta_json=np.array(json.dumps(meta)))
        print("wrote {}.npz ({} ticks)".format(base, len(a)))
        print("peak estimated winding temperature: {} C".format(
            np.round(self.thermal.peak_w, 1).tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--jointmap", default=None, help="deploy_map.json (REQUIRED for a real run)")
    ap.add_argument("--thermal", default=None, help="thermal_params.json from thermal_fit.py")
    ap.add_argument("--v-cmd", type=float, default=0.0, help="forward speed command, m/s")
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--ambient", type=float, default=25.0)
    ap.add_argument("--deadman-file", default=None,
                    help="path whose mtime must stay fresh; stale => soft stop")
    ap.add_argument("--deadman-s", type=float, default=0.5)
    ap.add_argument("--telemetry-stale-s", type=float, default=0.05,
                    help="kill if no status frame for this long. The drives broadcast at exactly "
                         "200 Hz (measured), so the default is 10 missed frames.")
    ap.add_argument("--drive-amp-limit", type=float, default=None,
                    help="the drive's configured phase-current limit, A -- caps torque absolutely")
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--allow-uncalibrated-thermal", action="store_true")
    ap.add_argument("--skip-jointmap-check", action="store_true",
                    help="DANGEROUS: run with an unverified joint map")
    ap.add_argument("--log", action="store_true", default=True)
    ap.add_argument("--no-log", dest="log", action="store_false")
    ap.add_argument("--out", default=str(WEBUI / "data" / "policyruns"))
    args = ap.parse_args()

    if not args.mock and os.system("systemctl is-active --quiet runningrobot-webui.service") == 0:
        raise SystemExit("the webui daemon is running and streaming its own CAN commands to these "
                         "drives. sudo systemctl stop runningrobot-webui.service, then retry.")
    if args.mock and args.telemetry_stale_s == 0.05:
        # canio.MockBus broadcasts at 100 Hz and advances its physics lazily inside recv(), so its
        # frame spacing is nothing like the real drives' measured 200.0 Hz / 5.13 ms max. Relax the
        # watchdog for the dress rehearsal rather than have every mock run die on a mock artefact.
        args.telemetry_stale_s = 0.25
        print("(mock: telemetry-staleness watchdog relaxed to 250 ms -- MockBus is a 100 Hz "
              "lazily-advanced simulator, not the drives)")
    r = Runner(args)
    r.banner()
    if args.skip_jointmap_check and not args.mock:
        print("!! --skip-jointmap-check: a sign error here drives balance corrections the WRONG "
              "WAY at 200 N*m/rad. You have been told.")

    def _stop(_s, _f):
        r.stop_requested = True
    signal.signal(signal.SIGTERM, _stop)

    ok = r.execute()
    r.save_log()
    if r.exit_reason:
        print("exit: {}".format(r.exit_reason))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
