"""What a drawn gait DEMANDS of the actuators as a function of cadence.

    python tools/analyze_gait_demand.py data/trajectories/<gait>.npz [../../../training/model/dash01.xml]

Written 2026-08-04 to justify the 2.5 Hz playback ceiling (daemon.PERIOD_MIN) after a part failed
on a high-cadence run. The run telemetry was NOT recoverable -- the webui keeps telemetry in an
in-memory ring only -- so this works from the one artefact that does survive: the commanded gait.

SPEED is exact. It is differentiated straight off the stored waveform (no sinusoid assumption) and
scales linearly with cadence, so "peak deg/s per Hz" is a property of the SHAPE alone. Comparing it
to the motor's no-load speed gives a hard kinematic ceiling: past that frequency the command cannot
be followed by any control effort, and the drive just saturates current chasing it.

TORQUE is an estimate and should be read as scaling (f^2), not as a number to design to. It is the
inertial term M[j,j] * alpha with M from the MuJoCo mass matrix at the loaded-stance keyframe, and
it EXCLUDES gravity, ground reaction and friction while OVERSTATING the swing case -- in stance the
joint reacts against the whole pinned robot, in swing only against the distal leg. Treat the ratio
between two cadences as meaningful and the absolute value as indicative.
"""
import sys

import numpy as np

TRAJ = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else None

# CubeMars output-side (post-gearbox) specs, from [[spiderbot-hardware]]
SPEC = {
    "cam":   dict(motor="AKE90-8", no_load_dps=np.degrees(22.0), peak_nm=170.0, cont_nm=55.0),
    "thigh": dict(motor="AKE90-8", no_load_dps=np.degrees(22.0), peak_nm=170.0, cont_nm=55.0),
}
COLS = ["cam", "thigh"]           # trajectory.MOVING_COLS order

z = np.load(TRAJ, allow_pickle=True)
N = int(z["N"])
print(f"trajectory: {TRAJ.split('/')[-1]}   N={N} samples/cycle, {int(z['harmonics'])} harmonics\n")

peak_w_per_hz = {}      # deg/s per Hz  (speed is linear in f)
peak_a_per_hz2 = {}     # deg/s^2 per Hz^2 (accel is quadratic in f)

for side in ("right", "left"):
    can = z[f"{side}_canonical"]                     # [N,2] mean-removed, degrees
    print(f"{side} leg")
    for j, name in enumerate(COLS):
        x = can[:, j]
        ptp = float(np.ptp(x))
        # d/dphase on a closed loop; at cadence f, d/dt = f * d/dphase
        dxdp = np.gradient(np.concatenate([x, x, x]), 1.0 / N)[N:2 * N]
        d2xdp2 = np.gradient(np.gradient(np.concatenate([x, x, x]), 1.0 / N), 1.0 / N)[N:2 * N]
        w1 = float(np.abs(dxdp).max())               # deg/s at 1 Hz
        a1 = float(np.abs(d2xdp2).max())             # deg/s^2 at 1 Hz
        peak_w_per_hz[(side, name)] = w1
        peak_a_per_hz2[(side, name)] = a1
        nl = SPEC[name]["no_load_dps"]
        print(f"  {name:6s} travel {ptp:6.1f}deg   peak speed {w1:7.1f} deg/s per Hz"
              f"   -> hits no-load ({nl:.0f} deg/s) at {nl / w1:5.2f} Hz")
    print()

print("=" * 78)
print(f"{'cadence':>8}  {'joint':12} {'peak speed':>12} {'% no-load':>10} "
      f"{'peak accel':>13} {'inertial Nm':>12}")
print("=" * 78)

inertia = {}
if MODEL:
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(MODEL)
        d = mujoco.MjData(m)
        d.qpos[:] = m.key_qpos[0]
        mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, M, d.qM)
        for nm, jn in (("cam", "cam_L"), ("thigh", "thigh_L")):
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
            dof = m.jnt_dofadr[m.actuator_trnid[a, 0]]
            inertia[nm] = float(M[dof, dof])
        print(f"(joint-space inertia at the loaded-stance keyframe: "
              + ", ".join(f"{k} {v:.4f} kg m^2" for k, v in inertia.items()) + ")")
    except Exception as e:                                   # model is optional
        print(f"(no inertial estimate: {e})")

for f in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
    for name in COLS:
        w1 = max(peak_w_per_hz[("right", name)], peak_w_per_hz[("left", name)])
        a1 = max(peak_a_per_hz2[("right", name)], peak_a_per_hz2[("left", name)])
        w, a = w1 * f, a1 * f * f
        pct = 100 * w / SPEC[name]["no_load_dps"]
        tau = inertia.get(name, float("nan")) * np.radians(a)
        flag = "  <== over no-load" if pct >= 100 else ("  <-- >70%" if pct >= 70 else "")
        print(f"{f:6.1f}Hz  {name:12} {w:9.0f}d/s {pct:9.0f}% {a:11.0f}d/s2 "
              f"{tau:11.1f}{flag}")
    print("-" * 78)
