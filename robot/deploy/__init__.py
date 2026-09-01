"""DASH-01 policy deployment: everything needed to run a trained walk_mit policy on the robot.

Split deliberately into two halves:

  DESKTOP (needs torch / stable-baselines3 / mujoco)
      export_policy.py    a trained run -> one self-contained .npz bundle
      verify_export.py    proves the numpy runtime reproduces the torch policy, in the sim
      thermal_fit.py      fits the lumped thermal model to a calibration log

  ROBOT (pure numpy + python-can, runs on the Pi 3B's 3.13 venv)
      policy_net.py       the MLP forward pass, no torch
      fourier_gait.py     the gait reconstruction, vendored byte-for-byte from walk_mit
      controller.py       the 200 Hz control law: obs -> action -> joint targets + impedance
      thermal.py          per-motor winding-temperature observer + torque budget
      safety.py           the clamp ladder and the kill conditions
      mit.py              CubeMars force-control CAN frames
      jointmap.py         MuJoCo joint frame <-> hardware motor frame

Nothing in the ROBOT half imports torch, mujoco or walk_mit. That is the point: the Pi has no
internet and no torch, and the deployed control law must be reviewable without a training stack.
"""
