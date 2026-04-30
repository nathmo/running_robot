Pendulum sim-to-real training

Files:
- `pendulum_env.py`: lightweight Gym environment for inverted pendulum (50 Hz default)
- `train_pendulum.py`: training script using Stable-Baselines3 PPO and ONNX export

Quick start (on training server):

1. Create venv and install deps:

```bash
python3 -m venv ~/pendulum_env
source ~/pendulum_env/bin/activate
pip install --upgrade pip
pip install stable-baselines3[extra] torch onnx onnxruntime gym
```

2. Train (example):

```bash
python RL/train_pendulum.py --timesteps 200000 --n-envs 4 --start-range 0.1 --export models/pendulum.onnx
```

3. The script saves the final SB3 model under `models/pendulum_final.zip` and writes an ONNX actor to the given export path.

Running on Raspberry Pi:
- Install `onnxruntime` for the Pi (use the wheel matching your Python).
- Load `models/pendulum.onnx` using `onnxruntime.InferenceSession` and run inference at 50 Hz.
- Use `moteus` to send torque commands (clipped to ±1 Nm).

Notes:
- The ONNX export wrapper attempts to use the SB3 policy forward path; if export fails, you can export a small PyTorch MLP by loading `models/pendulum_final.zip` and reconstructing the network.
- This repo's training script is intentionally minimal to be portable across CPU-only servers.
