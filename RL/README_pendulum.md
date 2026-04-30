Pendulum sim-to-real workflow

Files:
- `environment/pendulum_env.py`: MuJoCo inverted pendulum environment with raw hardware-style observations
- `train.py`: PPO training loop with curriculum, evaluation, and checkpointing
- `export_pendulum_onnx.py`: export a trained PPO checkpoint to ONNX with VecNormalize baked in
- `evaluate_pendulum.py`: evaluate a checkpoint or ONNX policy in simulation, with optional MuJoCo viewer
- `run_pendulum_pi.py`: run the exported ONNX policy on Raspberry Pi through moteus at 50 Hz

Quick start (training server):

1. Create venv and install deps:

```bash
python3 -m venv ~/pendulum_env
source ~/pendulum_env/bin/activate
pip install --upgrade pip
pip install -r RL/requirements_rl.txt
```

2. Train:

```bash
python RL/train.py --variant pendulum_sim2real --preset default --n-envs 4 --n-epochs 1000 --n-steps 1024
```

This creates a timestamped run folder under `RL/models/`, for example `RL/models/pendulum_sim2real_20260430_123456/`.

3. Evaluate the checkpoint in simulation:

```bash
python RL/evaluate_pendulum.py --checkpoint RL/models/pendulum_sim2real_<timestamp>/checkpoints/model_epoch_001000.zip --stats RL/models/pendulum_sim2real_<timestamp>/checkpoints/vecnormalize_epoch_001000.pkl --render
```

4. Export ONNX:

```bash
python RL/export_pendulum_onnx.py --checkpoint RL/models/pendulum_sim2real_<timestamp>/checkpoints/model_epoch_001000.zip --stats RL/models/pendulum_sim2real_<timestamp>/checkpoints/vecnormalize_epoch_001000.pkl --output RL/models/pendulum.onnx
```

5. Run on Raspberry Pi:

```bash
python RL/run_pendulum_pi.py --model RL/models/pendulum.onnx --device /dev/ttyACM0 --rate 50 --max-torque 1.0 --debug
```

Notes:
- The policy observes `[position_turns, velocity_turns_per_s, torque_nm]` and outputs a torque command in Nm.
- Success means the pendulum reaches the upright band within 5 seconds and stays within `±0.1` turn and `±0.1` turn/s until the end of the 20 second episode.
- On Linux, `config.py` defaults `n_envs` to the available CPU cores.
