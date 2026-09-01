#!/usr/bin/env python3
"""DESKTOP TOOL. Turn a trained walk_mit run into one self-contained deployment bundle.

    python robot/deploy/export_policy.py --run walk_mit/runs/imp_m2_long \
        --out robot/deploy/bundles/imp_m2_long_204M.npz

WHAT IT DOES, AND WHY IT BUILDS THE ENV
---------------------------------------
It constructs the SAME DashEnv the evaluator does -- via walk_mit.evaluate.build(), so the run's
curriculum.json restore (cmd_scale, stance_ratio, efficiency, drive bandwidth, torque limit) is
applied by the one function that has already had four separate restore bugs fixed in it -- and
then reads the deployment constants OFF THE LIVE OBJECT.

That matters most for the standing pose. `DashEnv._resettle_keyframe` re-solves the stance against
this arm's ankle law and REWRITES `nominal_ctrl` and `default_motor_pos`. Those two vectors are the
origin of the entire control law: every Fourier target is `nominal_ctrl + delta`, and every motor
position the policy observes is `q - default_motor_pos`. Neither is in resolved_config.json. Copy
them by hand and the robot runs a policy whose zero is somewhere else.

WHAT IS DELIBERATELY *NOT* EXPORTED
-----------------------------------
* the critic (`value_net`) -- it reads privileged simulator state and never runs on the robot
* `log_std` -- the robot runs the mean action, always. Sampling on hardware is not exploration,
  it is noise injected into a machine with 144 N*m joints. (It IS reported, because a checkpoint
  whose std is still pinned at the clamp is a mid-training checkpoint.)
* anything from the reward, the curriculum callbacks or the domain randomizer
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

DEPLOY = Path(__file__).resolve().parent
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from bundle import Bundle                                            # noqa: E402


def _torch_state_dict(model_zip):
    """policy.pth out of an SB3 checkpoint, as plain numpy. Needs torch, desktop-only."""
    import torch as th
    with zipfile.ZipFile(model_zip) as z:
        sd = th.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=False)
        data = json.loads(z.read("data").decode())
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}, data


def _gait_cfg(cfg):
    """Exactly the fields gait.decode/frequency/assemble read -- no more.

    Kept explicit rather than dumping the whole Config so that adding a training-only knob can
    never silently change the deployed control law, and so a reviewer can see the entire set of
    numbers that turn an action into a joint target on one screen."""
    return {
        "n_harmonics": int(cfg.n_harmonics),
        "gait_freq_hz": [float(cfg.gait_freq_hz[0]), float(cfg.gait_freq_hz[1])],
        "cam_amp": float(cfg.cam_amp),
        "thigh_amp": float(cfg.thigh_amp),
        "reflex_kp_scale": float(cfg.reflex_kp_scale),
        "reflex_kd_scale": float(cfg.reflex_kd_scale),
        "reflex_bias_scale": float(cfg.reflex_bias_scale),
        "pitch_kp": float(cfg.pitch_kp),
        "pitch_kd": float(cfg.pitch_kd),
        "pitch_bias": float(cfg.pitch_bias),
        "pitch_clip": float(cfg.pitch_clip),
        "pitch_reflex_rate_lp": float(cfg.pitch_reflex_rate_lp),
        "steer_stride_scale": float(cfg.steer_stride_scale),
        "steer_width_scale": float(cfg.steer_width_scale),
    }


def export(run_dir, out_path, checkpoint=None, preset=None):
    run = Path(run_dir)
    walk = run.resolve().parent.parent           # walk_mit/runs/<name> -> walk_mit/
    if not (walk / "evaluate.py").exists():
        raise SystemExit(f"{walk} does not look like the training package (no evaluate.py)")
    if str(walk) not in sys.path:
        sys.path.insert(0, str(walk))
    import evaluate                                                  # walk_mit/evaluate.py
    import mujoco

    model, venv, env = evaluate.build(run, preset, checkpoint)
    cfg = env.cfg
    vn = venv                       # VecNormalize wrapper (evaluate.build already loaded stats)

    if env.cpg_mode:
        raise SystemExit("this exporter deploys the FOURIER gait only; the run is action_mode=cpg")
    if env.n_ankle_act:
        raise SystemExit("this exporter deploys the 6-actuator plant; the run has actuated ankles")
    if not env.command_mode:
        raise SystemExit("this exporter deploys the joystick 'command' objective only; the run's "
                         "objective is {!r} (a sprint policy has a distance-to-go task channel "
                         "with no meaning on a robot in a room)".format(cfg.objective))

    sd, data = _torch_state_dict(evaluate.pick_model(run, checkpoint) + ".zip")
    std = np.exp(sd["log_std"])

    # --- the deterministic actor, as plain matrices --------------------------------------------
    # asym_policy.AsymExtractor: estimator(actor_obs) -> 3, then policy_net([actor_obs, est]),
    # then SB3's action_net. Activation is Tanh (ActorCriticPolicy's default, not overridden by
    # this run) with out_act=True on policy_net -- so the LAST hidden layer is tanh'd before
    # action_net, which a naive port gets wrong.
    act_fn = str(data.get("policy_kwargs", {}).get("activation_fn", "Tanh"))
    if "Tanh" not in act_fn:
        raise SystemExit("unexpected activation {!r}: the numpy runtime hard-codes tanh"
                         .format(act_fn))

    n_actor = env.frame_dim * cfg.history_len
    f64 = np.float64
    f32 = np.float32
    arrays = {
        "est_w0": np.asarray(sd["mlp_extractor.estimator.0.weight"], f32),
        "est_b0": np.asarray(sd["mlp_extractor.estimator.0.bias"], f32),
        "est_w1": np.asarray(sd["mlp_extractor.estimator.2.weight"], f32),
        "est_b1": np.asarray(sd["mlp_extractor.estimator.2.bias"], f32),
        "est_w2": np.asarray(sd["mlp_extractor.estimator.4.weight"], f32),
        "est_b2": np.asarray(sd["mlp_extractor.estimator.4.bias"], f32),
        "pi_w0": np.asarray(sd["mlp_extractor.policy_net.0.weight"], f32),
        "pi_b0": np.asarray(sd["mlp_extractor.policy_net.0.bias"], f32),
        "pi_w1": np.asarray(sd["mlp_extractor.policy_net.2.weight"], f32),
        "pi_b1": np.asarray(sd["mlp_extractor.policy_net.2.bias"], f32),
        "act_w": np.asarray(sd["action_net.weight"], f32),
        "act_b": np.asarray(sd["action_net.bias"], f32),
        # VecNormalize: only the ACTOR slice. The privileged tail is normalized during training
        # too, but the robot never forms it, and exporting it would only invite someone to build a
        # 596-vector and hand the actor the wrong 590 of it.
        "obs_mean": np.asarray(vn.obs_rms.mean[:n_actor], f64),
        "obs_var": np.asarray(vn.obs_rms.var[:n_actor], f64),
        # --- plant / control-law constants, read off the LIVE env ---
        "nominal_ctrl": np.asarray(env.nominal_ctrl, f64),
        "default_motor_pos": np.asarray(env.default_motor_pos, f64),
        "ctrl_lo": np.asarray(env.ctrl_lo, f64),
        "ctrl_hi": np.asarray(env.ctrl_hi, f64),
        "motor_vel_limit": np.asarray(cfg.motor_vel_limit, f64),
        "forcerange": np.asarray(env.model.actuator_forcerange[:env.nu, 1], f64),
        # impedance base gains: a MuJoCo position actuator has gainprm[0]=kp, biasprm[2]=-kv
        "imp_kp_base": np.asarray(env._imp_base[0], f64),
        "imp_kd_base": np.asarray(-env._imp_base[2], f64),
        "imp_leg_ix": np.asarray(env._imp_leg_ix, np.int32),
        "hist_idx": np.asarray(env._hist_idx, np.int32),
        "stand_torque": np.asarray(env._stand_torque, f64),
    }

    cur = {}
    cj = run / "curriculum.json"
    if cj.exists():
        cur = json.loads(cj.read_text())

    meta = {
        "run": run.name,
        "checkpoint": Path(evaluate.pick_model(run, checkpoint)).name,
        "preset": json.loads((run / "resolved_config.json").read_text()).get("preset"),
        "nu": int(env.nu),
        "action_dim": int(env.action_dim),
        "gait_action_dim": int(env.gait_action_dim),
        "spec_dim": int(env.spec_dim),
        "n_steer": int(env.n_steer),
        "imp_dim": int(env.imp_dim),
        "imp_action_start": int(env.imp_action_start),
        "frame_dim": int(env.frame_dim),
        "history_len": int(cfg.history_len),
        "history_stride": int(env._hist_stride),
        "hist_raw_len": int(env._hist_raw_len),
        "task_dim": int(env.task_dim),
        "obs_base_vel": bool(env.obs_base_vel),
        "phase_obs_dim": int(env.phase_obs_dim),
        "est_hidden": [128, 64],
        "policy_hidden": [int(x) for x in cfg.policy_hidden],
        "control_dt": float(env.control_dt),
        "obs_scales": {k: float(v) for k, v in cfg.obs_scales.items()},
        "action_scale": float(cfg.action_scale),
        "action_filter": float(cfg.action_filter),
        "action_delay_steps": int(cfg.action_delay_steps),
        "residual_scale": float(cfg.residual_scale),
        "motor_accel_limit": float(cfg.motor_accel_limit),
        "vel_accel_limited": bool(env._vel_accel_limited),
        "clip_obs": float(vn.clip_obs),
        "vn_epsilon": float(vn.epsilon),
        # command channel
        "cmd_v_norm": float(cfg.cmd_v_norm),
        "cmd_yaw_norm": float(cfg.cmd_yaw_norm),
        "cmd_deadband": float(cfg.cmd_deadband),
        "cmd_yaw_deadband": float(cfg.cmd_yaw_deadband),
        # the box the policy was actually TRAINED to at this checkpoint, from curriculum.json.
        # Commanding outside it is out-of-distribution -- the runner refuses to.
        "cmd_v_fwd_trained": float(cur.get("cmd_v_fwd", cfg.cmd_v_fwd_start)),
        "cmd_v_back_trained": float(cur.get("cmd_v_back", cfg.cmd_v_back_start)),
        "cmd_yaw_trained": float(cur.get("cmd_yaw", cfg.cmd_yaw_start)),
        "cmd_scale": float(cur.get("cmd_scale", 0.0)),
        # impedance channel
        "imp_kp_up": float(cfg.imp_kp_up), "imp_kp_dn": float(cfg.imp_kp_dn),
        "imp_kd_up": float(cfg.imp_kd_up), "imp_kd_dn": float(cfg.imp_kd_dn),
        # fall / termination thresholds the sim used -- the robot reuses them as kill conditions
        "term_gravity_z": float(cfg.term_gravity_z),
        "term_height": float(cfg.term_height),
        "height_target": float(env.height_target),
        # base DOFs the policy trained with. 1 = RAILED IN SIM = never experienced on hardware.
        "base_lock": [int(x) for x in cfg.base_lock],
        "gait_cfg": _gait_cfg(cfg),
        "policy_std_mean": float(std.mean()),
        "policy_std_max": float(std.max()),
        "max_log_std": float(cfg.max_log_std),
        "motor_kt_joint": [float(x) for x in cfg.motor_kt_joint],
        "actuator_names": [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                           for a in range(env.nu)],
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Bundle.save(out, arrays, meta)
    b = Bundle.load(out)
    _report(b, out)
    return b


def _report(b, out):
    m, a = b.meta, b.a
    print("wrote {}  ({:.2f} MB)".format(out, Path(out).stat().st_size / 1e6))
    print("  run/checkpoint : {} / {}  preset={}".format(m["run"], m["checkpoint"], m["preset"]))
    print("  actor obs      : {} = frame {} x history {} (stride {})".format(
        b.n_actor, m["frame_dim"], m["history_len"], m["history_stride"]))
    print("  action         : {} (gait {} + impedance {})".format(
        m["action_dim"], m["gait_action_dim"], m["imp_dim"]))
    print("  control rate   : {:.0f} Hz   action delay {} steps ({:.0f} ms), filter {}".format(
        1 / m["control_dt"], m["action_delay_steps"],
        m["action_delay_steps"] * m["control_dt"] * 1e3, m["action_filter"]))
    print("  stance ctrl    : {} rad".format(np.round(a["nominal_ctrl"], 4).tolist()))
    print("  obs zero pose  : {} rad".format(np.round(a["default_motor_pos"], 4).tolist()))
    print("  joint kp/kd    : {} / {}".format(np.round(a["imp_kp_base"], 1).tolist(),
                                              np.round(a["imp_kd_base"], 2).tolist()))
    kp_lo, kp_hi = a["imp_kp_base"] / m["imp_kp_dn"], a["imp_kp_base"] * m["imp_kp_up"]
    kd_lo, kd_hi = a["imp_kd_base"] / m["imp_kd_dn"], a["imp_kd_base"] * m["imp_kd_up"]
    print("  impedance span : kp [{:.0f}, {:.0f}] N*m/rad   kd [{:.2f}, {:.2f}] N*m*s/rad".format(
        kp_lo.min(), kp_hi.max(), kd_lo.min(), kd_hi.max()))
    print("  command box    : fwd {:.2f} / back {:.2f} / yaw {:.2f}  (cmd_scale {:.2f})".format(
        m["cmd_v_fwd_trained"], m["cmd_v_back_trained"], m["cmd_yaw_trained"], m["cmd_scale"]))
    lock = m["base_lock"]
    railed = [n for n, l in zip("X Y Z roll pitch yaw".split(), lock) if l]
    print("  base_lock      : {}  -> RAILED IN TRAINING: {}".format(
        lock, ", ".join(railed) if railed else "nothing (free base)"))
    if any(lock[3:]):
        axes = "/".join(n for n, l in zip(("roll", "pitch", "yaw"), lock[3:]) if l)
        print("  !! this policy has NEVER experienced free {}. On a robot that is not physically "
              "held in those axes it is running open-loop on them.".format(axes))
    if m["policy_std_mean"] > 0.5:
        print("  !! policy std is {:.3f} (clamp {:.2f}) -- the entropy anneal has not finished, so "
              "this is a mid-training checkpoint. Still deployable (the robot runs the MEAN "
              "action), but a later checkpoint from the same run should be better."
              .format(m["policy_std_mean"], np.exp(m["max_log_std"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None, help="model path w/o .zip (default: latest)")
    ap.add_argument("--preset", default=None)
    args = ap.parse_args()
    export(args.run, args.out, args.checkpoint, args.preset)


if __name__ == "__main__":
    main()
