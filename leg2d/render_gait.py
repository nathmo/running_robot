"""Render one leg2d gait point to mp4 -- a visual sanity check that an optimizer/sweep result is a
real periodic gait and not a sim artifact the numeric checks happened to miss.

Usage
  .venv/Scripts/python.exe leg2d/render_gait.py --f 8 --duty 0.85 --stride 0.2 --out leg2d/results/gait.mp4
  .venv/Scripts/python.exe leg2d/render_gait.py --from-optimize   # render results/optimize.json's best point
"""
import argparse
import json
from pathlib import Path

import mujoco

import gait
import motor
import sim

VIDEO_FPS = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=float, default=8.0, help="cadence Hz")
    ap.add_argument("--duty", type=float, default=0.85)
    ap.add_argument("--stride", type=float, default=0.20)
    ap.add_argument("--clearance", type=float, default=0.05)
    ap.add_argument("--z-off", type=float, default=0.0)
    ap.add_argument("--cycles", type=float, default=6.0)
    ap.add_argument("--out", default=str(sim.PKG / "results" / "gait.mp4"))
    ap.add_argument("--from-optimize", action="store_true",
                    help="use the best point from results/optimize.json instead of --f/--duty/...")
    args = ap.parse_args()

    if args.from_optimize:
        opt = json.loads((sim.PKG / "results" / "optimize.json").read_text())
        args.f, args.duty, args.stride, args.clearance, args.z_off = opt["x_best"]

    model, data, ids = sim.build_sim()
    lut, ik0 = sim.load_lut()
    nominal_cam, nominal_thigh, _stand_z = sim.nominal_pose(model, data, ids)

    sub_steps = max(1, int(round(1.0 / sim.CONTROL_HZ / model.opt.timestep)))
    control_dt = sub_steps * model.opt.timestep
    t_end = args.cycles / args.f

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance, cam.elevation, cam.azimuth = 1.8, -10, 90

    import imageio.v2 as imageio
    vid_every = max(1, int(round(1.0 / (control_dt * VIDEO_FPS))))
    writer = imageio.get_writer(args.out, fps=int(round(1.0 / (control_dt * vid_every))))

    t, i = 0.0, 0
    while t < t_end:
        dx, dz = gait.foot_xz(t, args.f, args.duty, args.stride, args.clearance, args.z_off)
        djk = cpg_gait_delta(lut, ik0, dx, dz)
        q_cam = data.qpos[ids["qadr_cam"]]
        dq_cam = data.qvel[ids["dadr_cam"]]
        q_thigh = data.qpos[ids["qadr_thigh"]]
        dq_thigh = data.qvel[ids["dadr_thigh"]]
        tau_cam = motor.clamp_torque(sim.KP * ((nominal_cam + djk[0]) - q_cam) - sim.KV * dq_cam, dq_cam)
        tau_thigh = motor.clamp_torque(sim.KP * ((nominal_thigh + djk[1]) - q_thigh) - sim.KV * dq_thigh, dq_thigh)
        data.ctrl[ids["a_cam"]] = float(tau_cam)
        data.ctrl[ids["a_thigh"]] = float(tau_thigh)
        data.ctrl[ids["a_hr"]] = 0.0
        for _ in range(sub_steps):
            mujoco.mj_step(model, data)
        if i % vid_every == 0:
            cam.lookat[:] = [data.qpos[ids["qadr_x"]], 0.2, 0.5]
            renderer.update_scene(data, cam)
            writer.append_data(renderer.render())
        t += control_dt
        i += 1
    writer.close()
    print(f"wrote {args.out}  (f={args.f}Hz duty={args.duty} L={args.stride}m z_off={args.z_off}m, "
          f"final x={float(data.qpos[ids['qadr_x']]):.3f} m, final z={float(data.qpos[ids['qadr_z']]):.3f} m)")


def cpg_gait_delta(lut, ik0, dx, dz):
    import cpg_gait
    import numpy as np
    return np.asarray(cpg_gait.foot_ik(dx, dz, lut), float) - ik0


if __name__ == "__main__":
    main()
