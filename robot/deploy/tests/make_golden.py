"""Record what the control law does TODAY, so a rewrite has to reproduce it bit for bit.

    python robot/deploy/tests/make_golden.py            # regenerate (only with a reason)

verify_export.py ties the numpy control law to the torch policy inside MuJoCo. That is the right
authority, and it needs torch + mujoco + a training run on disk. This is the complementary net for
REFACTORING: it ties the law after a change to the law before it, needs nothing but numpy, and runs
in a second -- so it can sit in the normal test suite and fail the moment an optimisation changes a
number it was not supposed to change.

Deterministic by construction: a fixed seed drives the measurements, and the inputs deliberately
exercise the branches that matter -- targets outside the position band, gains outside the wire
range, torque demands past the budget, and a thermal observer that actually heats.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.dirname(HERE)
sys.path.insert(0, DEPLOY)

GOLDEN = os.path.join(HERE, "golden_control_law.npz")
N_STEPS = 400


def _bundle_path():
    d = os.path.join(DEPLOY, "bundles")
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz"):
            return os.path.join(d, f)
    raise SystemExit("no bundle in robot/deploy/bundles/ to record against")


def trace(bundle_path):
    """(inputs, outputs) for N_STEPS of controller + governor + thermal, fully determined."""
    from bundle import Bundle
    from controller import PolicyController
    import safety as SAFE
    import thermal as TH
    import jointmap as JM

    b = Bundle.load(bundle_path)
    ctrl = PolicyController(b)
    chain = [TH.DEFAULT_PARAMS["AK60-39" if r == "abd" else "AKE90-8"]
             for r in ("abd", "cam", "thigh")] * 2
    th = TH.MotorThermalModel(chain, dt=0.005, t_amb=25.0, allow_uncalibrated=True,
                              names=list(JM.MODEL_ACTUATORS))
    lim = SAFE.Limits.from_bundle(b, tau_cont=0.35 * np.asarray(b["forcerange"], float))
    gov = SAFE.SafetyGovernor(lim, 0.005, thermal=th, names=list(JM.MODEL_ACTUATORS))

    rng = np.random.default_rng(20260901)
    pos = np.asarray(b["nominal_ctrl"], float).copy()
    vel = np.zeros(6)
    tau = np.zeros(6)
    ctrl.start(pos, vel, tau, np.array([0.0, 0.0, -1.0]), np.zeros(3))
    last_target = pos.copy()

    # A SECOND governor fed adversarial commands, so the clamp ladder is covered without pushing
    # the first one into a latched stop -- a latched stop never unlatches, and every step after it
    # exercises the ramp-down path instead of the path being optimised.
    th2 = TH.MotorThermalModel(chain, dt=0.005, t_amb=25.0, allow_uncalibrated=True,
                               names=list(JM.MODEL_ACTUATORS))
    gov2 = SAFE.SafetyGovernor(lim, 0.005, thermal=th2, names=list(JM.MODEL_ACTUATORS))

    out = []
    for k in range(N_STEPS):
        # a robot that moves, tilts a little and warms up -- but at a current that does not cook
        # the observer inside the trace, so the ordinary path stays the one under test
        # the joint FOLLOWS its last command with a little lag and noise. A random walk instead
        # would drift away from the target, hold the tracking limit saturated past persist_ticks
        # and latch a soft stop -- after which every remaining step exercises the ramp-down path
        # rather than the ladder this trace exists to pin down.
        pos = pos + 0.35 * (last_target - pos) + rng.normal(0.0, 0.004, 6)
        vel = rng.normal(0.0, 0.6, 6)
        tau = rng.normal(0.0, 12.0, 6)
        grav = np.array([0.05 * np.sin(k * 0.01), 0.05 * np.cos(k * 0.013), -0.99])
        gyro = rng.normal(0.0, 0.4, 3)
        temp = 30.0 + 12.0 * np.abs(np.sin(np.arange(6) + k * 0.02))
        amps = np.abs(rng.normal(2.5, 1.0, 6))
        cmd = ctrl.step(pos, vel, tau, grav, gyro)
        v = gov.step(cmd.target, cmd.kp, cmd.kd, pos, vel, grav, gyro,
                     telemetry_age=0.0, deadman_age=0.0, drive_temp=temp,
                     drive_err=np.zeros(6, int), current=amps, t_amb=25.0)
        # the adversarial half: targets past the band and gains past the wire range, so POSITION
        # and GAINS are exercised too. Only every third step, so no limit dwells long enough to
        # latch a persistence kill and shut the ladder down.
        hard = (k % 3 == 0)
        v2 = gov2.step(cmd.target + (2.0 if hard else 0.0),
                       cmd.kp * (9.0 if hard else 1.0), cmd.kd * (4.0 if hard else 1.0),
                       pos, vel, grav, gyro, telemetry_age=0.0, deadman_age=0.0,
                       drive_temp=temp, drive_err=np.zeros(6, int), current=amps, t_amb=25.0)
        last_target = np.asarray(v.target, float)
        out.append(np.concatenate([
            cmd.target, cmd.kp, cmd.kd, cmd.action, [cmd.phase, cmd.freq],
            v.target, v.kp, v.kd, th.t_winding, th.t_case, [float(v.stop), float(v.ramp)],
            v2.target, v2.kp, v2.kd, [float(v2.stop), float(v2.ramp)],
        ]))
    st = dict(gov.status()["clamp_counts"])
    for kk, vv in gov2.status()["clamp_counts"].items():
        st[kk] = st.get(kk, 0) + vv
    return np.asarray(out, float), {"clamp_counts": st, "stop": gov.status()["stop"],
                                    "stop2": gov2.status()["stop"]}

    return np.asarray(out, float), gov.status()


def main():
    arr, status = trace(_bundle_path())
    np.savez(GOLDEN, trace=arr, clamps=np.array(sorted(status["clamp_counts"].items()), dtype=object)
             if status["clamp_counts"] else np.zeros(0))
    print("wrote {}  shape {}".format(GOLDEN, arr.shape))
    print("clamps exercised: {}".format(status["clamp_counts"]))
    print("finite: {}   winding max {:.1f} C   governor: {} / {}".format(
        bool(np.all(np.isfinite(arr))), arr[:, 68:74].max(), status["stop"], status["stop2"]))


if __name__ == "__main__":
    main()
