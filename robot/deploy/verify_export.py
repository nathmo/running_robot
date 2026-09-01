#!/usr/bin/env python3
"""DESKTOP TOOL. Prove the deployed numpy control law IS the trained policy.

    python robot/deploy/verify_export.py --bundle robot/deploy/bundles/imp_m2_long_204M.npz \
        --run walk_mit/runs/imp_m2_long --steps 3000

WHY THIS EXISTS
---------------
Porting a policy out of a training stack means re-implementing an observation builder, a gait
reconstruction, an action-delay buffer, a filter and a gain map. Every one of those is a place to
be silently off by one frame or one sign, and none of them raises an exception when wrong -- the
robot just walks worse, or falls, and you cannot tell that from a policy that is simply not good
enough. So the port is not "reviewed", it is MEASURED, against the torch policy driving the same
MuJoCo plant.

Three tests, increasing in strength:

  1. NET     random observations through torch vs numpy. Isolates the MLP arithmetic.
  2. OBS     the controller, fed the simulator's own measurements, must rebuild the simulator's
             own observation vector. This is the one that catches frame-offset and scaling bugs.
  3. LOOP    closed loop: the torch policy drives the plant while the numpy controller shadows it
             from measurements only. Actions, joint targets and impedance gains must all match.

The verification plant has domain randomization and sensor noise turned OFF. That is not making
the test easy -- it is making it DECISIVE. With noise on, the controller sees a different
measurement than the observation the env built, so any diff is uninterpretable. The deployed
control law must be exact on the clean plant; whether the POLICY tolerates noise is what the
training curriculum answered, and is a different question.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

DEPLOY = Path(__file__).resolve().parent
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from bundle import Bundle                                            # noqa: E402
from controller import PolicyController                              # noqa: E402
from policy_net import PolicyNet                                     # noqa: E402


def _measure(env):
    """Exactly the quantities the robot can measure, read out of the simulator.

    Read BEFORE env.step(), i.e. from the state left by the previous step's physics -- which is
    the same instant DashEnv._proprio sampled when it built the newest observation frame."""
    return dict(motor_pos=env.data.qpos[env.act_qadr].copy(),
                motor_vel=env.data.qvel[env.act_dadr].copy(),
                motor_tau=env.data.actuator_force[:env.nu].copy(),
                grav=env._gravity_body(),
                gyro=env._ang_vel_body())


def build_clean_env(run, bundle, preset=None, checkpoint=None):
    walk = Path(run).resolve().parent.parent
    if str(walk) not in sys.path:
        sys.path.insert(0, str(walk))
    import evaluate
    from env import DashEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    run = Path(run)
    cfg = evaluate.load_run_config(run, preset)
    # Everything below turns off a SIMULATOR-SIDE corruption of the measurement or the plant.
    # None of it changes the policy or the control law; it makes the comparison well posed.
    cfg.dr_enable = False               # keeps the impedance base gains at the model values,
    #                                     which is what the bundle exported
    cfg.obs_noise_enable = False        # the controller has no noise model to match
    cfg.push_interval_s = 0.0
    cfg.trip_prob = 0.0
    cfg.ctrl_jitter_ms_final = 0.0
    cfg.ctrl_drop_prob_final = 0.0
    cfg.reset_joint_noise = 0.0
    env = DashEnv(cfg)
    env.set_ctrl_jitter(0.0)
    env.set_ctrl_drop(0.0)
    venv = VecNormalize.load(str(evaluate.pick_vecnormalize(
        run, evaluate.pick_model(run, checkpoint))), DummyVecEnv([lambda: env]))
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(evaluate.pick_model(run, checkpoint),
                     custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0,
                                     "clip_range": lambda _: 0.2})
    # the exported constants must describe THIS env, or the whole comparison is against a
    # different plant. Cheap, decisive, and it catches "exported one run, verifying another".
    for name, got, want in (("nominal_ctrl", env.nominal_ctrl, bundle["nominal_ctrl"]),
                            ("default_motor_pos", env.default_motor_pos,
                             bundle["default_motor_pos"]),
                            ("imp_kp_base", env._imp_base[0], bundle["imp_kp_base"])):
        if not np.allclose(got, want, atol=1e-12):
            raise SystemExit("bundle {} does not match the env it is being verified against:\n"
                             "  env    {}\n  bundle {}".format(name, got, want))
    return env, venv, model


def test_net(bundle, model, n=512, seed=0):
    """Random observations through both implementations."""
    import torch as th
    rng = np.random.default_rng(seed)
    net = PolicyNet(bundle)
    n_actor, n_priv = bundle.n_actor, 6
    # observations here are already VecNormalize-scaled, so a standard normal is representative
    obs = rng.standard_normal((n, n_actor + n_priv)).astype(np.float32)
    with th.no_grad():
        lat = model.policy.mlp_extractor.forward_actor(th.as_tensor(obs))
        a_th = model.policy.action_net(lat).numpy()
    a_np = np.stack([net.act_from_normalized(o[:n_actor])[0] for o in obs])
    err = np.abs(a_th - a_np)
    return {"n": n, "max_abs": float(err.max()), "mean_abs": float(err.mean()),
            "rel": float(err.max() / max(np.abs(a_th).max(), 1e-9))}


def test_loop(bundle, env, venv, model, steps, seed=0):
    """Closed loop. The torch policy drives; the numpy controller shadows from measurements."""
    ctrl = PolicyController(bundle)
    obs = venv.reset(seed=seed) if "seed" in venv.reset.__code__.co_varnames else venv.reset()
    m0 = _measure(env)
    ctrl.start(v_cmd=env._v_cmd, yaw_cmd=env._yaw_cmd, **m0)
    # the env sampled its command through its own deadband already; mirror the exact state so a
    # deadband applied twice cannot show up as a task-channel diff
    ctrl._v_cmd, ctrl._yaw_cmd, ctrl._standing = env._v_cmd, env._yaw_cmd, env._standing

    d_act, d_obs, d_tgt, d_kp, d_kd = [], [], [], [], []
    n_done = 0
    for t in range(steps):
        ctrl._v_cmd, ctrl._yaw_cmd, ctrl._standing = env._v_cmd, env._yaw_cmd, env._standing
        meas = _measure(env)
        raw_obs = venv.get_original_obs()[0]                 # what DashEnv._obs() returned
        a_th, _ = model.predict(obs, deterministic=True)
        # Shadow the TORCH action through the controller's own state. Without this the two
        # implementations share a feedback path -- prev_action is part of the observation -- so
        # the ~1e-6 float32 disagreement compounds around the loop (measured: 1e-6 -> 2.4e-2 over
        # 1500 steps) and the test stops measuring the port and starts measuring Lyapunov growth.
        # Driving both from the same action makes every reported diff a per-step port error.
        cmd = ctrl.step(override_action=a_th[0], **meas)
        d_obs.append(np.abs(np.asarray(ctrl._obs()) - raw_obs[:bundle.n_actor]).max())
        d_act.append(np.abs(np.asarray(a_th[0], np.float32) - cmd.action).max())
        obs, _, done, _ = venv.step(a_th)
        if done[0]:
            # DummyVecEnv AUTO-RESETS on done, so env.data / actuator_gainprm now describe the
            # NEXT episode's reset pose, not the step we just took. Comparing against them here
            # reports a ~0.6 rad target diff that is entirely an artefact of the vec-env contract
            # (measured: excluding this one sample the diff is exactly 0.0). Restart the shadow
            # controller on the new state instead -- which also exercises start() mid-run.
            n_done += 1
            ctrl.start(v_cmd=env._v_cmd, yaw_cmd=env._yaw_cmd, **_measure(env))
            continue
        d_tgt.append(np.abs(env.data.ctrl[:env.nu] - cmd.target).max())
        d_kp.append(np.abs(env.model.actuator_gainprm[:env.nu, 0] - cmd.kp).max())
        d_kd.append(np.abs(-env.model.actuator_biasprm[:env.nu, 2] - cmd.kd).max())
    return {"steps": len(d_act), "episodes_ended": n_done,
            "action_max": float(np.max(d_act)), "obs_max": float(np.max(d_obs)),
            "target_max": float(np.max(d_tgt)), "kp_max": float(np.max(d_kp)),
            "kd_max": float(np.max(d_kd))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--net-samples", type=int, default=512)
    # float32 matmuls do not associate, so torch and numpy differ in the last bits and the error
    # compounds through 5 layers. 2e-5 on an action in [-1, 1] is ~1e-5 rad of joint target: four
    # orders of magnitude below the encoder resolution, and six below anything mechanical.
    ap.add_argument("--tol", type=float, default=2e-5)
    args = ap.parse_args()

    b = Bundle.load(args.bundle)
    env, venv, model = build_clean_env(args.run, b, checkpoint=args.checkpoint)

    print("1. NET  (random observations, torch vs numpy)")
    r = test_net(b, model, args.net_samples)
    print("   n={n}  max|d action| = {max_abs:.3e}  mean {mean_abs:.3e}".format(**r))
    ok_net = r["max_abs"] < args.tol

    print("2/3. OBS + LOOP  (torch drives the plant, numpy shadows it)")
    L = test_loop(b, env, venv, model, args.steps)
    print("   steps={steps}  episodes_ended={episodes_ended}".format(**L))
    print("   max|d observation|   = {obs_max:.3e}".format(**L))
    print("   max|d action|        = {action_max:.3e}".format(**L))
    print("   max|d joint target|  = {target_max:.3e} rad".format(**L))
    print("   max|d kp|            = {kp_max:.3e} N*m/rad".format(**L))
    print("   max|d kd|            = {kd_max:.3e} N*m*s/rad".format(**L))
    ok_loop = all(L[k] < args.tol for k in ("obs_max", "action_max", "kp_max", "kd_max"))
    # the joint target is in radians and carries the same float32 error scaled by kp-sized
    # numbers through the filter; give it the same absolute tolerance, it is still ~1e-5 rad
    ok_loop = ok_loop and L["target_max"] < max(args.tol, 1e-5)

    if ok_net and ok_loop:
        print("\nPASS -- the numpy control law reproduces the trained policy to float32 noise.")
        return 0
    print("\nFAIL -- the deployed control law is NOT the trained policy. Do not run this on the "
          "robot.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
