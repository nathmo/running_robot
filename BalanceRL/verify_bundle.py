"""Prove the exported bundle + controller/deploy/controller_balance.py IS the trained policy.

    python BalanceRL/verify_bundle.py --run BalanceRL/runs/bal_s0 --bundle controller/deploy/bundles/bal_s0.npz

The JAX env runs the greedy policy with the sensor-noise model ON and pushes on. Every tick, the
MEASURED signals the env's sensor model produced (encoder, velocity, torque, gravity, gyro -- noise,
bias, IMU mount rotation and dropout included) are handed to the numpy runtime exactly as the runner
hands it the robot's. The runtime's observation, action, target, kp and kd must match the sim's.
Actuation drop is switched off (the robot has no such thing: it is a sim model of a late frame).
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "controller" / "deploy"))

import numpy as np                                    # noqa: E402
import jax                                            # noqa: E402
import jax.numpy as jnp                               # noqa: E402

from policy_io import load_policy                     # noqa: E402
from env import EnvParams                             # noqa: E402


def _meas_from_frame(f, cfg, default_pos):
    """Invert the obs scaling of the reset frame (the reset does not return its measurements)."""
    s = cfg.obs_scales
    return dict(pos=f[0:6] / s["motor_pos"] + default_pos, vel=f[6:12] / s["motor_vel"],
                tau=f[12:18] / s["motor_torque"], grav=f[18:21] / s["gravity"], gyro=f[21:24] / s["ang_vel"])


def verify(run, bundle_path, ckpt=None, n_ticks=600, seed=7, push_level=0.8, tol=1e-4, quiet=False):
    from bundle import Bundle
    from controller_balance import PolicyControllerBalance
    cfg, env, pol = load_policy(run, ckpt, n_envs=1, ctrl_drop_prob=0.0)
    b = Bundle.load(bundle_path)
    ctrl = PolicyControllerBalance(b)
    prm = EnvParams(plant_scale=1.0, push_level=push_level, push_on=1.0)
    st, obs = env.reset(jax.random.PRNGKey(seed), prm)
    step = jax.jit(env.step)
    A = env.actor_dim
    F, H = env.frame_dim, cfg.history_len
    newest = lambda o: o[(H - 1) * F:H * F]      # NOT o[-F:]: the slow block sits after the frames
    o0 = np.asarray(obs[0, :A])
    m0 = _meas_from_frame(newest(o0), cfg, np.asarray(env.plant.default_motor_pos))
    ctrl.start(m0["pos"], m0["vel"], m0["tau"], m0["grav"], m0["gyro"], exact=True)
    worst = dict(obs=0.0, action=0.0, target=0.0, kp=0.0, kd=0.0)
    n_reset = 0
    meas = None
    for t in range(n_ticks):
        # The runtime is driven with the SIM's action (override_action), so both sides carry the same
        # previous action into the next frame. cmd.action is still the runtime's OWN network output,
        # and is diffed below. Without this, a 1e-6 float32 summation-order difference re-enters
        # through the previous-action channel every tick and, through a narrow normalizer, grows
        # into a divergence that says nothing about the control law.
        a = pol.act(obs)
        m = meas if meas is not None else m0
        cmd = ctrl.step(m["pos"], m["vel"], m["tau"], m["grav"], m["gyro"], override_action=np.asarray(a[0]))
        # the observation the runtime acted on vs the sim's
        worst["obs"] = max(worst["obs"], float(np.abs(ctrl.obs() - np.asarray(obs[0, :A])).max()))
        st, obs, r, done, info = step(st, a, prm)
        worst["action"] = max(worst["action"], float(np.abs(cmd.action - np.asarray(info["action"][0])).max()))
        worst["target"] = max(worst["target"], float(np.abs(cmd.target - np.asarray(info["target"][0])).max()))
        worst["kp"] = max(worst["kp"], float(np.abs(cmd.kp - np.asarray(info["kp"][0])).max()
                                           / max(float(np.abs(np.asarray(info["kp"][0])).max()), 1.0)))
        worst["kd"] = max(worst["kd"], float(np.abs(cmd.kd - np.asarray(info["kd"][0])).max()
                                           / max(float(np.abs(np.asarray(info["kd"][0])).max()), 1.0)))
        if bool(done[0]):
            # a new episode: restart the runtime from the new episode's first frame
            n_reset += 1
            o0 = np.asarray(obs[0, :A])
            m0 = _meas_from_frame(newest(o0), cfg, np.asarray(env.plant.default_motor_pos))
            ctrl.start(m0["pos"], m0["vel"], m0["tau"], m0["grav"], m0["gyro"], exact=True)
            meas = None
            continue
        meas = {k: np.asarray(v[0], np.float64) for k, v in info["meas"].items()}
    ok = all(v <= tol for v in worst.values())
    if not quiet:
        print(f"[verify] {n_ticks} ticks, {n_reset} episode resets, noise ON, pushes to {push_level} m/s")
        for k, v in worst.items():
            print(f"[verify]   max |deploy - sim| {k:7s} = {v:.2e}  {'ok' if v <= tol else 'FAIL'}")
        print(f"[verify] {'PASS' if ok else 'FAIL'} (tol {tol:g}; kp/kd relative)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ticks", type=int, default=600)
    a = ap.parse_args()
    sys.exit(0 if verify(a.run, a.bundle, a.ckpt, a.ticks) else 1)


if __name__ == "__main__":
    main()
