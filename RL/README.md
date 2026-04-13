# RL Framework for Legged Robot Training

Complete RL training framework designed for morphology optimization of legged robots with terrain variation and path tracking.

## Project Structure

```
RL/
├── environment/                 # MuJoCo environment and terrain
│   ├── mujoco_env.py           # Main Gymnasium environment wrapper
│   ├── terrain.py              # Perlin noise terrain generator
│   └── paths.py                # Path tracking (circular, sine, spiral)
├── models/                      # Saved models and checkpoints
├── logs/                        # Training logs and metrics
├── config.py                    # Configuration management
├── train.py                     # Main training script
├── visualize.py                 # Visualization utilities
├── utils.py                     # Logging, checkpointing, metrics
├── requirements_rl.txt          # Dependencies
└── assets/                      # Robot URDF files
    └── unitree_a1.urdf         # (Download required)
```

## Installation

1. Install dependencies:
```bash
cd RL
pip install -r requirements_rl.txt
```

2. Download robot URDF:
   - **Unitree A1**: Download from [Unitree GitHub](https://github.com/unitreerobotics/unitree_legged_sdk) or use the A1 SDK
   - Place in `RL/assets/` and name it `unitree_a1.urdf`
   - Ensure collision/visual meshes are correctly referenced

## Quick Start

### Train with default configuration

```bash
cd RL
python train.py --variant "my_experiment" --preset "default"
```

### Resume training from checkpoint

```bash
python train.py --variant "my_experiment" --resume 100
```

### Custom terrain and training parameters

```bash
python train.py \
    --variant "hard_terrain_test" \
    --preset "hard_terrain" \
    --terrain-type "perlin" \
    --n-envs 8 \
    --n-epochs 1000
```

## Configuration

Edit `config.py` to customize:

### Terrain
- **type**: "flat", "perlin", "stairs"
- **perlin_scale**: Noise scale (smaller = more details)
- **height_scale**: Maximum terrain variation
- **seed**: Fixed seed for reproducibility

### Path Tracking
- **track_types**: ["circle", "sine_wave", "spiral"]
- **radii**: Different circular path radii
- **spawn_distribution**: Robot starting position randomization

### RL Training
- **algorithm**: "PPO"
- **n_envs**: Number of parallel environments
- **learning_rate**: PPO learning rate
- **n_epochs**: Total training epochs
- **checkpoint_interval**: Save checkpoint every N epochs

### Reward Function
Customize `REWARD` section:
- `forward_speed_weight`: Reward for moving forward
- `stability_weight`: Penalty for rotation
- `energy_weight`: Penalty for power consumption
- `track_deviation_weight`: Penalty for leaving path

## Key Features

### 1. Terrain Generation
- **Perlin noise** for procedural terrain (Minecraft-style)
- **Fixed seeds** for reproducible environments
- **Parameterizable** difficulty (scale, octaves, height)
- Support for flat, stairs, and custom terrains

### 2. Path Tracking
- Curved paths: circles, sine waves, spirals
- Variable radii for different difficulty levels
- Random spawn positions along paths
- Speed measured along path direction

### 3. RL Training
- **PPO** algorithm via Stable-Baselines3
- **Vectorized environments** for parallel training
- **Checkpoint management**: Save/resume at any epoch
- **Metrics logging**: Appended to text file + JSON after each epoch
- **Convergence plotting**: Automatic training curve visualization

### 4. Checkpoint & Resumption
```python
# Automatically managed by CheckpointManager
models/
└── my_experiment_20240413_143022/
    ├── config.json              # Configuration used
    ├── metadata.json
    ├── checkpoints/
    │   ├── model_epoch_000050.zip
    │   ├── model_epoch_000100.zip
    │   └── metrics_epoch_000100.json
    └── running_avg.pkl          # Normalization stats
```

### 5. Logging & Analysis
```
logs/
└── my_experiment_20240413_143022/
    ├── training_log.txt         # Human-readable log (appended)
    ├── metrics.json             # Machine-readable metrics
    └── convergence.png          # Training curves
```

## Visualization

### Environment setup visualization
```python
from visualize import visualize_environment
import config as cfg

config = cfg.get_config("default")
visualize_environment(config, save_path="environment.png")
```

### Path tracking visualization
```python
from visualize import plot_path_tracking

plot_path_tracking(config, save_path="paths.png")
```

### Training comparison
```python
from visualize import plot_training_comparison

plot_training_comparison(
    logs_dirs=["logs/variant1", "logs/variant2"],
    variant_names=["Variant 1", "Variant 2"],
    metric="eval_mean_reward",
    save_path="comparison.png"
)
```

## Advanced Configuration Presets

### Easy terrain (for quick testing)
```bash
python train.py --preset "easy_terrain" --variant "quick_test"
```

### Hard terrain (for robustness)
```bash
python train.py --preset "hard_terrain" --variant "robust_test"
```

### Fast training (fewer steps, quick iteration)
```bash
python train.py --preset "fast_training" --variant "debug"
```

### Long training (thorough optimization)
```bash
python train.py --preset "long_training" --variant "final_model"
```

## Creating Your Own Configuration

In `config.py`:

```python
"fast_terrain": {
    "TERRAIN": {**TERRAIN, "height_scale": 0.05, "perlin_scale": 2.0},
    "RL": {**RL, "learning_rate": 1e-3},
    # ... override any settings
}
```

Then run:
```bash
python train.py --preset "fast_terrain"
```

## Plan: Morphology-Policy Co-Optimization

Current setup handles **Option A** of the architecture:

1. **Phase 1** (current): Train RL policy for fixed robot configuration
   - Test with Unitree A1 default morphology
   - Verify training loop works
   - Get baseline performance

2. **Phase 2** (next): Design space exploration (outer loop CMA-ES)
   - Parameterize robot: segment lengths, masses, motor specs
   - For each morphology sample: train RL policy (inner loop)
   - Evaluate fitness = max speed achieved
   - CMA-ES searches design space

3. **Phase 3** (optional): Universal policy experiment (Option B)
   - Domain randomization over morphologies
   - Single policy trained on all designs
   - Compare convergence vs specialized policies

## Debugging

### Visualize terrain with different seeds
```python
from environment import TerrainGenerator
gen = TerrainGenerator(seed=42)
terrain = gen.generate_perlin(scale=1.0, octaves=4)
gen.visualize(terrain)
```

### Check robot loading
```python
import mujoco
model = mujoco.MjModel.from_path("assets/unitree_a1.urdf")
print(f"Actuators: {len(model.actuator_names)}")
print(f"Bodies: {model.nbody}")
```

### Monitor environment step
```python
from environment import LeggedRobotEnv
import config as cfg

config = cfg.get_config("default")
env = LeggedRobotEnv(config)
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
```

## Next Steps

1. **Get Unitree A1 URDF**: Find or create valid URDF in `assets/`
2. **Test environment**: Run visualization, check reward calculation
3. **Quick training run**: `python train.py --preset "fast_training" --n-epochs 10`
4. **Verify logging**: Check that metrics are being saved and plotted
5. **Extend path types**: Add more complex curved paths
6. **Implement outer loop**: Add morphology parameter search (CMA-ES)

## Notes

- **Arena size**: Terrain grid is 256x256 at 0.1m spacing = 25.6m x 25.6m
- **Simulation**: 1kHz MuJoCo physics, 10x action repeat = 100Hz control
- **Parallel training**: 4 environments by default (change `n_envs` in config)
- **Friction**: Randomized between episodes for robustness
