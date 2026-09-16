"""DASH-01 policy deployment: everything needed to run a trained policy on the robot.

TWO GENERATIONS, one package. `bundle.version` picks the runtime, and it is a lookup, not a guess:

  v1  `walk_mit`  200 Hz, per-step Fourier gait, a velocity + yaw command fixed at arm time
  v2  `walk_v2`   100 Hz, a 44-dim gait spec LATCHED at each clock wrap + a 6-dim per-tick
                  residual, and a command channel that is a RUN / STOP flag driven live

Split deliberately into two halves:

  DESKTOP (needs the training stack)
      export_policy.py    a trained v1 run -> one self-contained .npz bundle   (torch, mujoco)
      verify_export.py    proves the numpy runtime reproduces the torch policy, in the sim
      thermal_fit.py      fits the lumped thermal model to a calibration log
      ../../walk_v2/export.py  the same job for a v2 run                       (jax, mjx)

  ROBOT (pure numpy + python-can, runs on the Pi 3B's 3.13 venv)
      bundle.py           the .npz format, both generations
      policy_net.py       the MLP forward pass, no torch (shared)
      fourier_gait.py     v1 gait reconstruction, vendored byte-for-byte from walk_mit
      gait_v2.py          v2 gait generator, vendored from walk_v2/gait.py with jax removed
      controller.py       the v1 200 Hz control law: obs -> action -> joint targets + impedance
      controller_v2.py    the v2 100 Hz control law: the latch, the clock, the run/stop task
      thermal.py          per-motor winding-temperature observer + torque budget
      safety.py           the clamp ladder and the kill conditions (shared)
      mit.py              CubeMars force-control CAN frames
      jointmap.py         MuJoCo joint frame <-> hardware motor frame

Nothing in the ROBOT half imports torch, mujoco, jax, walk_mit or walk_v2. That is the point: the
Pi has no internet and no torch, and the deployed control law must be reviewable without a
training stack. See README.md for the bring-up order and for what v2 changes.
"""
