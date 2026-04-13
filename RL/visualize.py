"""
Visualization utilities for the environment and training
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json


def visualize_environment(config, save_path=None):
    """
    Visualize the environment setup (terrain, paths, etc.)

    Args:
        config: Configuration dict
        save_path: Save figure to this path
    """
    from environment import TerrainGenerator
    from environment import CircularPath, SineWavePath, SpiralPath

    # Generate terrain
    terrain_gen = TerrainGenerator(
        seed=config["TERRAIN"]["seed"],
        grid_size=config["TERRAIN"]["grid_size"],
        grid_spacing=config["TERRAIN"]["grid_spacing"],
    )

    if config["TERRAIN"]["type"] == "perlin":
        heightfield = terrain_gen.generate_perlin(
            scale=config["TERRAIN"]["perlin_scale"],
            octaves=config["TERRAIN"]["perlin_octaves"],
            persistence=config["TERRAIN"]["perlin_persistence"],
            lacunarity=config["TERRAIN"]["perlin_lacunarity"],
            height_scale=config["TERRAIN"]["height_scale"],
            height_offset=config["TERRAIN"]["height_offset"],
        )
    else:
        heightfield = terrain_gen.generate_flat()

    x, y = terrain_gen.get_mesh_coordinates()

    # Create figure
    fig = plt.figure(figsize=(14, 10))

    # 3D terrain + paths
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot_surface(x, y, heightfield, cmap="terrain", alpha=0.7, rcount=50, ccount=50)

    # Plot paths
    path_radii = config["PATHS"]["radii"]
    colors = ["red", "blue", "green", "orange"]

    for radius, color in zip(path_radii[:4], colors):
        path = CircularPath(radius=radius)
        t_samples = np.linspace(0, 1, 100)
        xs, ys = zip(*[path.get_position(t) for t in t_samples])
        zs = [heightfield[int(y / 0.1), int(x / 0.1)] if 0 <= int(y / 0.1) < heightfield.shape[0] and 0 <= int(x / 0.1) < heightfield.shape[1] else 0.5 for x, y in zip(xs, ys)]
        ax1.plot(xs, ys, zs, color=color, linewidth=2, label=f"r={radius}m")

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Height (m)")
    ax1.set_title("Terrain + Paths (3D)")
    ax1.legend()

    # 2D terrain heatmap
    ax2 = fig.add_subplot(222)
    im = ax2.imshow(
        heightfield,
        cmap="terrain",
        origin="lower",
        extent=[0, x.max(), 0, y.max()],
    )
    plt.colorbar(im, ax=ax2, label="Height (m)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Terrain (Top View)")
    ax2.grid(True, alpha=0.3)

    # Add paths to 2D view
    for radius, color in zip(path_radii[:4], colors):
        path = CircularPath(radius=radius)
        t_samples = np.linspace(0, 1, 100)
        xs, ys = zip(*[path.get_position(t) for t in t_samples])
        ax2.plot(xs, ys, color=color, linewidth=2, label=f"r={radius}m")
    ax2.legend()

    # Terrain profile
    ax3 = fig.add_subplot(223)
    center_row = heightfield.shape[0] // 2
    profile = heightfield[center_row, :]
    distances = np.arange(len(profile)) * config["TERRAIN"]["grid_spacing"]
    ax3.plot(distances, profile, "b-", linewidth=2)
    ax3.fill_between(distances, profile, alpha=0.3)
    ax3.set_xlabel("Distance (m)")
    ax3.set_ylabel("Height (m)")
    ax3.set_title("Terrain Profile")
    ax3.grid(True, alpha=0.3)

    # Config summary
    ax4 = fig.add_subplot(224)
    ax4.axis("off")

    config_text = f"""
    ENVIRONMENT CONFIGURATION

    Terrain:
    - Type: {config["TERRAIN"]["type"]}
    - Height scale: {config["TERRAIN"]["height_scale"]} m
    - Grid size: {config["TERRAIN"]["grid_size"]}
    - Grid spacing: {config["TERRAIN"]["grid_spacing"]} m

    Paths:
    - Radii: {config["PATHS"]["radii"]}
    - Types: {config["PATHS"]["track_types"]}

    Robot:
    - URDF: {config["ROBOT"]["urdf_path"]}
    - Control: {config["ROBOT"]["control_mode"]}
    - Action repeat: {config["ROBOT"]["action_repeat"]}

    RL Training:
    - Algorithm: {config["RL"]["algorithm"]}
    - n_envs: {config["RL"]["n_envs"]}
    - Learning rate: {config["RL"]["learning_rate"]}
    - Total epochs: {config["RL"]["n_epochs"]}
    """

    ax4.text(0.1, 0.9, config_text, transform=ax4.transAxes, fontsize=9, verticalalignment="top", family="monospace")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Environment visualization saved: {save_path}")

    plt.show()


def plot_training_comparison(logs_dirs, variant_names, metric="eval_mean_reward", save_path=None):
    """
    Compare training curves across multiple variants

    Args:
        logs_dirs: List of log directories
        variant_names: List of variant names
        metric: Metric to plot
        save_path: Save figure to path
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(logs_dirs)))

    for log_dir, variant, color in zip(logs_dirs, variant_names, colors):
        metrics_file = Path(log_dir) / "metrics.json"

        if not metrics_file.exists():
            print(f"Warning: metrics file not found for {variant}")
            continue

        with open(metrics_file, "r") as f:
            data = json.load(f)

        epochs = data.get("epochs", [])
        values = data.get(metric, [])

        if epochs and values:
            ax.plot(epochs, values, "o-", label=variant, color=color, markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"Training Comparison: {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved: {save_path}")

    plt.show()


def plot_path_tracking(config, max_steps=100, save_path=None):
    """
    Visualize different paths

    Args:
        config: Configuration dict
        max_steps: Number of samples per path
        save_path: Save figure to path
    """
    from environment import CircularPath, SineWavePath, SpiralPath

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Different paths
    paths_to_plot = [
        ("Circular r=3m", CircularPath(radius=3.0)),
        ("Circular r=5m", CircularPath(radius=5.0)),
        ("Sine Wave", SineWavePath(amplitude=1.0, frequency=2.0, length=20.0)),
        ("Spiral", SpiralPath(start_radius=1.0, end_radius=8.0)),
    ]

    for ax, (name, path) in zip(axes, paths_to_plot):
        t_samples = np.linspace(0, 1, max_steps)
        xs, ys = zip(*[path.get_position(t) for t in t_samples])

        ax.plot(xs, ys, "b-", linewidth=2, label="Path")
        ax.scatter([xs[0]], [ys[0]], c="g", s=100, label="Start", zorder=5)
        ax.scatter([xs[-1]], [ys[-1]], c="r", s=100, label="End", zorder=5)

        # Add heading vectors
        for t in np.linspace(0, 1, 8):
            x, y = path.get_position(t)
            heading = path.get_heading(t)
            dx = 0.5 * np.cos(heading)
            dy = 0.5 * np.sin(heading)
            ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.1, fc="red", ec="red", alpha=0.5)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Path visualization saved: {save_path}")

    plt.show()


# Example usage
if __name__ == "__main__":
    import config as cfg

    config = cfg.get_config("default")

    # Visualize environment
    visualize_environment(config, save_path="RL/logs/environment_setup.png")

    # Visualize paths
    plot_path_tracking(config, save_path="RL/logs/paths.png")
