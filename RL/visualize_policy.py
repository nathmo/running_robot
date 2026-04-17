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
import mujoco
import mujoco.viewer
import time
import sys
import os

# Add RL directory to path for imports
script_dir = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(script_dir))

import config as cfg
from environment import LeggedRobotEnv
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

    runs = print_runs_table(runs, sort_by="reward")
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
):
    """Load a trained policy from variant_dir/checkpoints and roll it out."""
    checkpoints_dir = variant_dir / "checkpoints"
    checkpoint_path = checkpoints_dir / f"model_epoch_{epoch:06d}.zip"
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    print(f"\n{'='*60}")
    print(f"POLICY VISUALIZATION")
    print(f"{'='*60}")
    print(f"Model: {variant_dir.name} (epoch {epoch})")
    print(f"Path:  {checkpoint_path}")

    # Load config and create single environment
    config = cfg.get_config("default")
    if not Path(config["ROBOT"]["urdf_path"]).is_absolute():
        config["ROBOT"]["urdf_path"] = str(script_dir / config["ROBOT"]["urdf_path"])

    env = LeggedRobotEnv(config)

    print(f"\nLoading policy...")
    model = PPO.load(str(checkpoint_path))
    print(f"Policy loaded: {type(model.policy).__name__}")

    mj_model = env.model
    mj_data = env.data

    print(f"\n{'='*60}")
    print(f"Episodes: {episodes}, Max steps: {max_steps}, Speed: {speed}x")
    print(f"{'='*60}\n")

    episode_rewards = []
    episode_lengths = []
    episode_distances = []

    def run_episode(ep_idx, viewer=None):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        ep_reward = 0.0
        ep_length = 0
        ep_distance = 0.0
        last_x = env.data.body(env.base_body_id).xpos[0]

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            ep_reward += reward
            ep_length += 1
            cur_x = env.data.body(env.base_body_id).xpos[0]
            ep_distance += abs(cur_x - last_x)
            last_x = cur_x

            if viewer is not None:
                viewer.sync()
                time.sleep(mj_model.opt.timestep / speed)

            if terminated or truncated:
                break

        print(
            f"[Ep {ep_idx + 1}/{episodes}] "
            f"reward={ep_reward:8.3f}  length={ep_length:4d}  "
            f"distance={ep_distance:6.2f} m"
        )
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_distances.append(ep_distance)

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
    print(f"Mean Distance: {np.mean(episode_distances):.2f} m")

    env.close()


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
        default="reward",
        help="Sort order for --list output.",
    )

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
    )


if __name__ == "__main__":
    main()
