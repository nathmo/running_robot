"""DASH-01 dynamic-parameter identification.

Layered by dependency so the Raspberry-Pi web UI can import the light parts:
  * frames.py, model_inertials.py, paramio.py  — PURE numpy + stdlib (safe to import anywhere,
    used by the web server's inertia-comparison endpoint).
  * dataset.py                                  — needs scipy (filtering/derivatives).
  * kt_calibration.py, mujoco_id.py, validate.py, run.py, apply_identified.py — need mujoco (+scipy).

Nothing at package import time pulls in mujoco/scipy, so `import identification.frames` works on the
Pi even though the estimator itself runs on the dev/training machine. See the plan.
"""
