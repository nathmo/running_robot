"""
Quick test script to verify RL framework setup
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from environment import TerrainGenerator, CircularPath, PathTracker
from visualize import visualize_environment, plot_path_tracking


def test_terrain_generation():
    """Test Perlin noise terrain generation"""
    print("\n" + "="*60)
    print("Testing Terrain Generation...")
    print("="*60)

    config = cfg.get_config("default")
    terrain_gen = TerrainGenerator(
        seed=config["TERRAIN"]["seed"],
        grid_size=64,  # Small for testing
        grid_spacing=config["TERRAIN"]["grid_spacing"],
    )

    # Generate different terrains
    perlin = terrain_gen.generate_perlin(
        scale=config["TERRAIN"]["perlin_scale"],
        octaves=config["TERRAIN"]["perlin_octaves"],
        height_scale=config["TERRAIN"]["height_scale"],
    )

    flat = terrain_gen.generate_flat()
    stairs = terrain_gen.generate_stairs()

    print(f"[OK] Generated Perlin terrain: shape={perlin.shape}, range=[{perlin.min():.3f}, {perlin.max():.3f}]")
    print(f"[OK] Generated flat terrain: shape={flat.shape}")
    print(f"[OK] Generated stairs terrain: shape={stairs.shape}")

    # Test height interpolation
    height = terrain_gen.get_height_at_position(perlin, 2.5, 3.7)
    print(f"[OK] Height interpolation at (2.5, 3.7): {height:.3f} m")

    return terrain_gen, perlin, flat, stairs


def test_paths():
    """Test path generation and tracking"""
    print("\n" + "="*60)
    print("Testing Path Generation...")
    print("="*60)

    config = cfg.get_config("default")

    # Create paths
    paths = {
        "circle": CircularPath(radius=3.0),
        "circle_large": CircularPath(radius=10.0),
    }

    for name, path in paths.items():
        # Sample path
        t_samples = [0, 0.25, 0.5, 0.75, 1.0]
        positions = [path.get_position(t) for t in t_samples]
        print(f"\n[OK] {name}:")
        for t, (x, y) in zip(t_samples, positions):
            print(f"  t={t}: ({x:.2f}, {y:.2f})")

    # Test path tracking
    print(f"\n[OK] Path Tracking:")
    circle = CircularPath(radius=5.0)
    tracker = PathTracker(circle)

    # Test deviation at off-path position
    deviation = tracker.get_deviation(6.0, 0.0)
    print(f"  Deviation from (6.0, 0.0): {deviation:.3f} m")

    # Test progress tracking
    progress = tracker.get_progress(5.0, 0.0)
    print(f"  Progress at (5.0, 0.0): {progress:.3f}")

    return paths, tracker


def test_config():
    """Test configuration system"""
    print("\n" + "="*60)
    print("Testing Configuration System...")
    print("="*60)

    presets = ["default", "easy_terrain", "hard_terrain", "fast_training"]

    for preset in presets:
        config = cfg.get_config(preset)
        print(f"[OK] {preset}:")
        print(f"  - Terrain height scale: {config['TERRAIN']['height_scale']}")
        print(f"  - RL n_steps: {config['RL']['n_steps']}")
        print(f"  - Learning rate: {config['RL']['learning_rate']}")


def test_environment_creation():
    """Test environment creation (if URDF exists)"""
    print("\n" + "="*60)
    print("Testing Environment Creation...")
    print("="*60)

    config = cfg.get_config("default")

    # Check if URDF exists
    urdf_path = Path(__file__).parent / config["ROBOT"]["urdf_path"].replace("RL/", "")

    if not urdf_path.exists():
        print(f"[WARNING] URDF not found at: {urdf_path}")
        print("  To test environment, download Unitree A1 URDF and place in RL/assets/")
        return False

    try:
        import mujoco

        model = mujoco.MjModel.from_path(str(urdf_path))
        print(f"[OK] Loaded URDF: {urdf_path.name}")
        print(f"  - Actuators: {len(model.actuator_names)}")
        print(f"  - Bodies: {model.nbody}")
        print(f"  - DOFs: {model.nv}")
        return True
    except Exception as e:
        print(f"[FAIL] Error loading URDF: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("RL FRAMEWORK SETUP VERIFICATION")
    print("="*70)

    # Test configuration
    test_config()

    # Test terrain
    terrain_gen, perlin, flat, stairs = test_terrain_generation()

    # Test paths
    paths, tracker = test_paths()

    # Test environment (if URDF available)
    has_urdf = test_environment_creation()

    # Summary
    print("\n" + "="*70)
    print("SETUP VERIFICATION COMPLETE")
    print("="*70)

    print("\n[OK] All basic tests passed!")

    if not has_urdf:
        print("\n[WARNING] WARNING: URDF not found. To run training:")
        print("  1. Download Unitree A1 URDF")
        print("  2. Place in RL/assets/unitree_a1.urdf")
        print("  3. Ensure mesh files are in correct location")

    print("\nNext steps:")
    print("  1. cd RL/")
    print("  2. python visualize.py  # To visualize terrain and paths")
    print("  3. python train.py --preset fast_training --n-epochs 10  # Quick training test")

    return 0


if __name__ == "__main__":
    sys.exit(main())
