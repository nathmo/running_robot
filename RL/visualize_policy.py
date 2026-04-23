"""
Visualize a trained policy in 3D using MuJoCo's viewer.

Runs an interactive scan of all training runs (under running_robot/models/)
showing performance metrics per checkpoint so you can pick what to visualize.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import mujoco
import mujoco.viewer
import time
import sys
import os
import csv

# Add RL directory to path for imports
script_dir = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(script_dir))

import config as cfg
from environment import InvertedPendulumEnv
from environment.mujoco_env import LeggedRobotEnv
from controllers.pid_controller import PendulumPIDController, PendulumPIDGains
from utils import get_models_dir


# ---------------------------------------------------------------------------
# Model scanning
# ---------------------------------------------------------------------------

def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _epoch_from_name(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return -1


def scan_checkpoints(variant_dir: Path):
    """
    Return list of dicts for every checkpoint that has a matching .zip file.

    Each entry:
        {
            "epoch": int,
            "checkpoint_path": Path,
            "metrics": dict | None,   # contents of metrics_epoch_*.json
        }
    """
    checkpoints_dir = variant_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return []

    entries = []
    for zip_path in sorted(checkpoints_dir.glob("model_epoch_*.zip")):
        epoch = _epoch_from_name(zip_path)
        metrics_path = checkpoints_dir / f"metrics_epoch_{epoch:06d}.json"
        entries.append(
            {
                "epoch": epoch,
                "checkpoint_path": zip_path,
                "metrics": _load_json(metrics_path) if metrics_path.exists() else None,
            }
        )
    return entries


def summarize_run(variant_dir: Path):
    """
    Return an info dict for a single run, or None if the run has no usable data.
    """
    metadata = _load_json(variant_dir / "metadata.json") or {}
    checkpoints = scan_checkpoints(variant_dir)

    # Also pull aggregated metrics.json from logs (training history)
    logs_metrics_file = (
        variant_dir.parent.parent / "logs" / variant_dir.name / "metrics.json"
    )
    logs_metrics = _load_json(logs_metrics_file) if logs_metrics_file.exists() else None

    # Determine best checkpoint by mean reward
    best = None
    latest = None
    if checkpoints:
        latest = checkpoints[-1]
        scored = [c for c in checkpoints if c["metrics"] is not None]
        if scored:
            best = max(
                scored,
                key=lambda c: c["metrics"].get("eval_mean_reward", float("-inf")),
            )

    return {
        "name": variant_dir.name,
        "path": variant_dir,
        "preset": metadata.get("config_preset", "?"),
        "algorithm": metadata.get("algorithm", "?"),
        "n_envs": metadata.get("n_envs", "?"),
        "n_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
        "latest": latest,
        "best": best,
        "logs_metrics": logs_metrics,
    }


def scan_all_runs(models_dir: Path, only_with_checkpoints: bool = True):
    """Return list of summaries for every run under models_dir."""
    if not models_dir.exists():
        return []

    runs = []
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        info = summarize_run(d)
        if only_with_checkpoints and info["n_checkpoints"] == 0:
            continue
        runs.append(info)
    return runs


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _fmt(val, fmt="{:7.2f}"):
    if val is None:
        return "   --  "
    try:
        return fmt.format(val)
    except Exception:
        return str(val)


def print_runs_table(runs, sort_by="reward"):
    """Print a sortable table of runs."""
    if not runs:
        print("No runs with checkpoints found.")
        return []

    # Sort
    def key_fn(r):
        if r["best"] is None or r["best"]["metrics"] is None:
            return float("-inf")
        return r["best"]["metrics"].get("eval_mean_reward", float("-inf"))

    if sort_by == "reward":
        runs = sorted(runs, key=key_fn, reverse=True)
    elif sort_by == "name":
        runs = sorted(runs, key=lambda r: r["name"])

    header = (
        f"{'#':>3}  {'Run':<36} {'Preset':<15} {'Ckpts':>5} "
        f"{'Best Ep':>8} {'Reward':>9} {'Length':>8} {'Distance':>9} "
        f"{'Last Ep':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for i, r in enumerate(runs):
        best_ep = r["best"]["epoch"] if r["best"] else None
        last_ep = r["latest"]["epoch"] if r["latest"] else None
        m = r["best"]["metrics"] if (r["best"] and r["best"]["metrics"]) else {}
        print(
            f"{i:>3}  {r['name']:<36} {r['preset']:<15} {r['n_checkpoints']:>5} "
            f"{_fmt(best_ep, '{:>8d}') if best_ep is not None else '     --  '} "
            f"{_fmt(m.get('eval_mean_reward'), '{:>9.3f}')} "
            f"{_fmt(m.get('eval_mean_length'), '{:>8.1f}')} "
            f"{_fmt(m.get('eval_mean_speed'), '{:>9.3f}')} "
            f"{_fmt(last_ep, '{:>8d}') if last_ep is not None else '     --  '}"
        )
    print("=" * len(header))
    print("Distance = eval_mean_speed (cumulative |dx| during eval episode)")
    return runs


def print_checkpoints_table(run):
    """Print all checkpoints for a single run."""
    checkpoints = run["checkpoints"]
    if not checkpoints:
        print("No checkpoints for this run.")
        return

    header = (
        f"{'#':>3}  {'Epoch':>7}  {'Reward':>10}  "
        f"{'Length':>8}  {'Distance':>9}  {'Std':>8}"
    )
    print("\n" + "=" * len(header))
    print(f"Run: {run['name']}  (preset={run['preset']}, algo={run['algorithm']})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    best_epoch = run["best"]["epoch"] if run["best"] else None
    latest_epoch = run["latest"]["epoch"] if run["latest"] else None

    for i, c in enumerate(checkpoints):
        m = c["metrics"] or {}
        tag = ""
        if c["epoch"] == best_epoch:
            tag += " [BEST]"
        if c["epoch"] == latest_epoch:
            tag += " [LATEST]"
        print(
            f"{i:>3}  {c['epoch']:>7d}  "
            f"{_fmt(m.get('eval_mean_reward'), '{:>10.3f}')}  "
            f"{_fmt(m.get('eval_mean_length'), '{:>8.1f}')}  "
            f"{_fmt(m.get('eval_mean_speed'), '{:>9.3f}')}  "
            f"{_fmt(m.get('eval_std_reward'), '{:>8.3f}')}"
            f"{tag}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Interactive selection
# ---------------------------------------------------------------------------

def _prompt(msg, default=None):
    suffix = f" [{default}]" if default is not None else ""
    try:
        s = input(f"{msg}{suffix}: ").strip()
    except EOFError:
        return default
    if not s:
        return default
    return s


def interactive_select(models_dir: Path):
    """
    Walk the user through picking a run and then an epoch.

    Returns (variant_dir, epoch) or (None, None) if cancelled.
    """
    runs = scan_all_runs(models_dir, only_with_checkpoints=True)
    if not runs:
        print(f"[ERROR] No runs with checkpoints found in {models_dir}")
        return None, None

    runs = print_runs_table(runs, sort_by="name")
    print(
        "\nChoose a run by number, or 'a' to list alphabetically, "
        "'q' to quit."
    )
    while True:
        choice = _prompt("Run #", default="0")
        if choice is None or choice.lower() == "q":
            return None, None
        if choice.lower() == "a":
            runs = print_runs_table(runs, sort_by="name")
            continue
        try:
            idx = int(choice)
            if 0 <= idx < len(runs):
                break
        except ValueError:
            pass
        print("Invalid selection.")

    run = runs[idx]
    print_checkpoints_table(run)

    # Decide default epoch: best if available, else latest
    default_tag = "best" if run["best"] else "latest"
    print(
        "\nPick a checkpoint: row # from the table above, "
        "'best', 'latest', a raw epoch number, or 'q' to cancel."
    )
    while True:
        choice = _prompt("Checkpoint", default=default_tag)
        if choice is None or choice.lower() == "q":
            return None, None
        c = choice.lower()
        if c == "best" and run["best"]:
            return run["path"], run["best"]["epoch"]
        if c == "latest" and run["latest"]:
            return run["path"], run["latest"]["epoch"]
        try:
            n = int(choice)
        except ValueError:
            print("Invalid selection.")
            continue
        # Row # in table?
        if 0 <= n < len(run["checkpoints"]):
            return run["path"], run["checkpoints"][n]["epoch"]
        # Raw epoch number?
        for c_entry in run["checkpoints"]:
            if c_entry["epoch"] == n:
                return run["path"], n
        print(f"Epoch {n} not found in this run.")


# ---------------------------------------------------------------------------
# Variant resolution (for non-interactive CLI use)
# ---------------------------------------------------------------------------

def resolve_variant(models_dir: Path, variant: str):
    """Resolve a variant string to a specific variant directory."""
    if not models_dir.exists():
        return None
    if (models_dir / variant).exists():
        return models_dir / variant
    matches = sorted(models_dir.glob(f"{variant}_*"))
    if matches:
        return matches[-1]
    return None


# ---------------------------------------------------------------------------
# Policy rollout
# ---------------------------------------------------------------------------

def visualize_policy(
    variant_dir: Path,
    epoch: int,
    episodes: int = 3,
    max_steps: int = 1000,
    speed: float = 1.0,
    render: bool = True,
    controller: str = "policy",
    pid_kp: float = 0.06,
    pid_ki: float = 0.001,
    pid_kd: float = 0.02,
):
    """Load a trained policy from variant_dir/checkpoints and roll it out."""
    checkpoints_dir = variant_dir / "checkpoints"
    checkpoint_path = checkpoints_dir / f"model_epoch_{epoch:06d}.zip"
    if controller == "policy" and not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    print(f"\n{'='*60}")
    print(f"POLICY VISUALIZATION")
    print(f"{'='*60}")
    print(f"Model: {variant_dir.name} (epoch {epoch})")
    if controller == "policy":
        print(f"Path:  {checkpoint_path}")
    print(f"Controller: {controller}")

    # Load the config that was saved with the run, falling back to default only
    # if the run predates config snapshots.
    config_path = variant_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Config loaded: {config_path.name}")
    else:
        config = cfg.get_config("default")
        print("[WARN] Run config.json not found; falling back to default config.")

    # Make relative model paths work no matter where the run was created from.
    if not Path(config["ROBOT"]["urdf_path"]).is_absolute():
        config["ROBOT"]["urdf_path"] = str(script_dir / config["ROBOT"]["urdf_path"])

    # Choose the matching environment for the checkpoint/task.
    env_class = LeggedRobotEnv if "PATHS" in config or "simple_biped" in config["ROBOT"]["urdf_path"] else InvertedPendulumEnv
    print(f"Environment: {env_class.__name__}")

    # Keep a handle to the raw InvertedPendulumEnv so we can still read mj_data / joint_id
    # after wrapping it in DummyVecEnv + VecNormalize.
    raw_env_ref = {"env": None}

    def _make_env():
        e = env_class(config)
        raw_env_ref["env"] = e
        return e

    vec_env = DummyVecEnv([_make_env])
    raw_env = raw_env_ref["env"]
    is_pendulum_env = hasattr(raw_env, "joint_id")

    # Restore the VecNormalize running stats that the policy was trained against.
    # Without this the policy sees an obs distribution nothing like training and
    # performs dramatically worse than the eval metrics suggest.
    vecnorm_path = checkpoints_dir / f"vecnormalize_epoch_{epoch:06d}.pkl"
    if vecnorm_path.exists():
        try:
            vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            print(f"VecNormalize stats loaded: {vecnorm_path.name}")
        except AssertionError as exc:
            print(
                f"[WARN] VecNormalize stats are incompatible with the current env: {exc}. "
                f"Proceeding without normalization."
            )
    else:
        print(
            f"[WARN] No VecNormalize stats at {vecnorm_path.name} — "
            f"policy will see unnormalized obs and likely misbehave."
        )

    model = None
    pid_controller = None
    if controller == "policy":
        print(f"\nLoading policy...")
        model = PPO.load(str(checkpoint_path))
        print(f"Policy loaded: {type(model.policy).__name__}")
    elif controller == "pid":
        if not is_pendulum_env:
            print("[ERROR] PID controller is only available for InvertedPendulumEnv.")
            vec_env.close()
            return
        pid_controller = PendulumPIDController(
            setpoint_deg=90.0,
            gains=PendulumPIDGains(kp=pid_kp, ki=pid_ki, kd=pid_kd),
            action_limit=1.0,
        )
        print("\nUsing PID controller in place of policy output")
        print(f"PID gains: kp={pid_kp}, ki={pid_ki}, kd={pid_kd}")
    else:
        print(f"[ERROR] Unknown controller type: {controller}")
        vec_env.close()
        return

    mj_model = raw_env.model
    mj_data = raw_env.data

    print(f"\n{'='*60}")
    print(f"Episodes: {episodes}, Max steps: {max_steps}, Speed: {speed}x")
    print(f"{'='*60}\n")

    episode_rewards = []
    episode_lengths = []
    episode_reasons = []
    trajectories = []  # Store trajectory data for all episodes

    def run_episode(ep_idx, viewer=None):
        obs = vec_env.reset()  # shape (1, obs_dim), normalized
        if pid_controller is not None:
            pid_controller.reset()

        ep_reward = 0.0
        ep_length = 0
        term_reason = None
        trajectory = []  # Collect trajectory for this episode

        for step in range(max_steps):
            if controller == "policy":
                action, _ = model.predict(obs, deterministic=True)
            else:
                joint_idx = raw_env.model.jnt_dofadr[raw_env.joint_id]
                angle_deg = float(raw_env.data.qpos[joint_idx])
                ang_vel_deg_s = float(raw_env.data.qvel[joint_idx])
                action_value = pid_controller.compute(
                    angle_deg=angle_deg,
                    angular_velocity_deg_s=ang_vel_deg_s,
                    dt=config["ROBOT"]["control_dt"],
                )
                action = np.array([[action_value]], dtype=np.float32)
            obs, reward, done, info = vec_env.step(action)

            ep_reward += float(reward[0])
            ep_length += 1

            # Extract trajectory data from raw environment.
            if is_pendulum_env:
                joint_idx = raw_env.model.jnt_dofadr[raw_env.joint_id]
                angle_deg = float(raw_env.data.qpos[joint_idx])
                ang_vel_deg_s = float(raw_env.data.qvel[joint_idx])
                torque = float(raw_env.data.ctrl[0])
                angle_norm = angle_deg
                while angle_norm > 180.0:
                    angle_norm -= 360.0
                while angle_norm < -180.0:
                    angle_norm += 360.0
                angle_norm = angle_norm / 180.0
            else:
                base = raw_env.data.body(raw_env.base_body_id)
                xmat = base.xmat.reshape(3, 3)
                body_z_world_z = float(np.clip(xmat[2, 2], -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(body_z_world_z)))
                ang_vel_deg_s = float(np.linalg.norm(base.cvel[3:]) * (180.0 / np.pi))
                torque = float(np.linalg.norm(raw_env.data.ctrl))
                angle_norm = angle_deg / 180.0

            # Get raw (unnormalized) reward from environment's reward breakdown
            reward_breakdown = info[0].get("reward_breakdown", {}) if isinstance(info, list) else info.get("reward_breakdown", {})
            raw_reward = (
                reward_breakdown.get("proximity", 0.0) +
                reward_breakdown.get("upright_bonus", 0.0) -
                reward_breakdown.get("effort_penalty", 0.0)
            )

            trajectory.append({
                "step": step,
                "angle": angle_deg,
                "velocity": ang_vel_deg_s,
                "torque": torque,
                "raw_reward": raw_reward,
                "normalized_reward": float(reward[0]),
                "angle_normalized": angle_norm,  # For debugging
            })

            if viewer is not None:
                viewer.sync()
                time.sleep(mj_model.opt.timestep / speed)

            if done[0]:
                term_reason = info[0].get("termination_reason", "done")
                break
        else:
            term_reason = "max_steps"

        print(
            f"[Ep {ep_idx + 1}/{episodes}] "
            f"reward={ep_reward:8.3f}  length={ep_length:4d}  "
            f"upright_time={info[0].get('upright_time', 0):6.2f}s  reason={term_reason}"
        )
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_reasons.append(term_reason)
        trajectories.append(trajectory)

    if render:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            for ep in range(episodes):
                run_episode(ep, viewer=viewer)
    else:
        for ep in range(episodes):
            run_episode(ep)

    print(f"\n{'='*60}")
    print(f"SUMMARY  ({variant_dir.name}, epoch {epoch})")
    print(f"{'='*60}")
    print(
        f"Mean Reward:   {np.mean(episode_rewards):.3f} "
        f"± {np.std(episode_rewards):.3f}"
    )
    print(f"Mean Length:   {np.mean(episode_lengths):.1f} steps")
    # Termination breakdown: fall / timeout / off_path / max_steps
    from collections import Counter
    counts = Counter(episode_reasons)
    reasons_str = ", ".join(f"{k}: {v}" for k, v in counts.most_common())
    print(f"Terminations:  {reasons_str}")

    # Export trajectories to CSV files
    logs_dir = checkpoints_dir.parent.parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Open debug log file for this visualization
    debug_log_file = logs_dir / f"debug_visualization_epoch_{epoch:06d}.txt"
    debug_log = open(debug_log_file, "w", encoding="utf-8")
    debug_log.write(f"\n{'='*70}\n")
    debug_log.write(f"VISUALIZATION DEBUG LOG - Epoch {epoch}\n")
    debug_log.write(f"{'='*70}\n")
    debug_log.write(f"Episodes: {episodes}, Max steps: {max_steps}\n\n")
    
    # Log angle normalization reference
    debug_log.write("ANGLE NORMALIZATION REFERENCE:\n")
    debug_log.write("  0 deg -> 0.000 (horizontal)\n")
    debug_log.write("  90 deg -> 0.500 (upright)\n")
    debug_log.write("  180 deg -> 1.000 (opposite side)\n")
    debug_log.write("  270 deg -> -0.500 (hanging down)\n")
    debug_log.write("  -90 deg -> -0.500 (same as 270)\n")
    debug_log.write("  -180 deg -> -1.000 (opposite of 180)\n")
    debug_log.write("\n")
    debug_log.write(f"Run: {variant_dir.name}\n")
    debug_log.write(f"Environment: {env_class.__name__}\n")
    debug_log.write(f"Controller: {controller}\n")
    if pid_controller is not None:
        debug_log.write(f"PID gains: kp={pid_kp}, ki={pid_ki}, kd={pid_kd}\n")
    debug_log.write(f"Obs space: {raw_env.observation_space.shape if raw_env is not None else 'n/a'}\n")
    debug_log.write(f"Angle field: {'joint angle' if is_pendulum_env else 'body tilt'}\n")
    
    for ep_idx, trajectory in enumerate(trajectories):
        traj_file = logs_dir / f"trajectory_epoch_{epoch:06d}_ep_{ep_idx:03d}.csv"
        with open(traj_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "angle", "angle_normalized", "velocity", "torque", "raw_reward", "normalized_reward"])
            writer.writeheader()
            writer.writerows(trajectory)
        print(f"Trajectory saved: {traj_file.name}")
        debug_log.write(f"\n--- Episode {ep_idx + 1} ---\n")
        for row in trajectory:
            debug_log.write(
                f"step={row['step']:04d} "
                f"angle={row['angle']:8.3f} "
                f"angle_norm={row['angle_normalized']:7.4f} "
                f"vel={row['velocity']:8.3f} "
                f"torque={row['torque']:7.3f} "
                f"raw_reward={row['raw_reward']:8.4f} "
                f"norm_reward={row['normalized_reward']:8.4f}\n"
            )
        debug_log.write(
            f"EPISODE {ep_idx + 1}: steps={len(trajectory)} final_angle={trajectory[-1]['angle']:7.2f} "
            f"final_norm={trajectory[-1]['angle_normalized']:7.4f}\n"
        )
    
    debug_log.write(f"\n{'='*70}\n")
    debug_log.close()
    print(f"\nDebug log saved: {debug_log_file.name}")

    vec_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained RL policy. "
        "Runs interactively when --variant is not given."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant name (exact or prefix). If omitted, launches interactive picker.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to visualize. If omitted, uses best (then latest).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the runs table and exit (no viewer).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed (1.0 = real-time).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Roll out the policy without launching the MuJoCo viewer.",
    )
    parser.add_argument(
        "--sort",
        choices=["reward", "name"],
        default="name",
        help="Sort order for --list output (default: chronological by name).",
    )
    parser.add_argument(
        "--controller",
        choices=["policy", "pid"],
        default="policy",
        help="Use trained policy output or PID output as the action source.",
    )
    parser.add_argument("--pid-kp", type=float, default=0.06, help="PID Kp gain.")
    parser.add_argument("--pid-ki", type=float, default=0.001, help="PID Ki gain.")
    parser.add_argument("--pid-kd", type=float, default=0.02, help="PID Kd gain.")

    args = parser.parse_args()
    models_dir = get_models_dir(__file__)

    if args.list:
        runs = scan_all_runs(models_dir, only_with_checkpoints=True)
        print_runs_table(runs, sort_by=args.sort)
        return

    # Resolve variant + epoch
    if args.variant is None:
        variant_dir, epoch = interactive_select(models_dir)
        if variant_dir is None:
            print("Cancelled.")
            return
    else:
        variant_dir = resolve_variant(models_dir, args.variant)
        if variant_dir is None:
            print(f"[ERROR] No model found for variant '{args.variant}' in {models_dir}")
            runs = scan_all_runs(models_dir, only_with_checkpoints=True)
            print_runs_table(runs)
            return

        run = summarize_run(variant_dir)
        if args.epoch is not None:
            epoch = args.epoch
        elif run["best"] is not None:
            epoch = run["best"]["epoch"]
            print(f"[INFO] Using best epoch {epoch} (by eval_mean_reward).")
        elif run["latest"] is not None:
            epoch = run["latest"]["epoch"]
            print(f"[INFO] Using latest epoch {epoch}.")
        else:
            print(f"[ERROR] No checkpoints in {variant_dir}")
            return

    visualize_policy(
        variant_dir=variant_dir,
        epoch=epoch,
        episodes=args.episodes,
        max_steps=args.max_steps,
        speed=args.speed,
        render=not args.no_render,
        controller=args.controller,
        pid_kp=args.pid_kp,
        pid_ki=args.pid_ki,
        pid_kd=args.pid_kd,
    )


if __name__ == "__main__":
    main()
