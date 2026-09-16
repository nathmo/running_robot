#!/usr/bin/env python3
"""Visualize the hard-coded walking gait on DASH-01 with the base FIXED IN SPACE (in the air).

This is the desktop counterpart to run_hardware.py: it drives the SAME gait.py trajectory into a
MuJoCo model whose floating base has been welded to the world, so the robot hangs in the air and
you watch all six motors cycle through the walking pattern. No balance, no ground — just the legs.

The model is derived from mujoco/dash01/dash01.xml at load time (via MjSpec): the root
freejoint is removed (base fixed), the standing keyframe and the floor are dropped, and gravity is
turned off by default (the gait choice you picked). Nothing about the real model is duplicated on
disk, so this can never drift from the robot description.

Run:
    .venv/Scripts/python.exe fixed_gait/sim_fixed_base.py                 # interactive viewer
    .venv/Scripts/python.exe fixed_gait/sim_fixed_base.py --gravity       # legs hang under gravity
    .venv/Scripts/python.exe fixed_gait/sim_fixed_base.py --period 2.0 --thigh-amp 0.3
    .venv/Scripts/python.exe fixed_gait/sim_fixed_base.py --video out.mp4 --duration 6

Controls in the viewer: mouse to orbit/zoom; space pauses; the gait keeps streaming.
"""
import argparse
import time

import numpy as np
import mujoco

from gait import GaitParams, GaitGenerator

MODEL = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "model" / "dash01.xml")
CONTROL_HZ = 100.0            # rate the position targets are refreshed (matches the CAN streamer)


def build_fixed_base_model(gravity=False, floor=False, base_height=None, collidable=False):
    """Load dash01.xml and weld the base to the world: drop the root freejoint (so the torso is
    fixed in space), the standing keyframe (its qpos carries the now-removed free dofs), and — for
    an in-air demo — the floor. With collidable=True the leg meshes are made collidable so the
    validator can detect link self-collision (the normal model only collides the foot spheres).
    Returns a compiled MjModel."""
    spec = mujoco.MjSpec.from_file(MODEL)
    # weld the base: delete the 6 composite base joints (base_x/y/z, base_roll/pitch/yaw) AND their
    # <equality><joint> locks (which reference those joints and would otherwise dangle). The two
    # closed-loop <connect> equalities are kept.
    base_joints = {"base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw"}
    for eq in list(spec.equalities):
        if eq.name.startswith("lock_"):
            spec.delete(eq)
    for j in list(spec.joints):
        if j.name in base_joints:
            spec.delete(j)
    for k in list(spec.keys):
        spec.delete(k)
    for g in list(spec.geoms):
        if g.name == "floor" and not floor:
            spec.delete(g)
        elif collidable and g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.contype = 1
            g.conaffinity = 1
    if base_height is not None:
        for b in spec.bodies:
            if b.name == "bodyNCS-v1":
                b.pos = [b.pos[0], b.pos[1], base_height]
    m = spec.compile()
    if not gravity:
        m.opt.gravity[:] = 0.0
    return m


def home_hinges():
    """The 12 hinge angles of the standing keyframe (drop the 6 base-joint qpos entries), used to
    start the sim on the physical assembly branch of the closed pushrod loop."""
    m0 = mujoco.MjModel.from_xml_path(MODEL)
    kid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_KEY, "stand")
    return m0.key_qpos[kid][6:].copy()


def make_params(args):
    p = GaitParams()
    if args.period is not None:
        p.period_s = args.period
    if args.thigh_amp is not None:
        p.thigh_amp = args.thigh_amp
    if args.cam_amp is not None:
        p.cam_amp = args.cam_amp
    if args.thigh_center is not None:
        p.thigh_center = args.thigh_center
    if args.cam_center is not None:
        p.cam_center = args.cam_center
    if args.ramp is not None:
        p.ramp_s = args.ramp
    return p


def reset_to_home(m, d, gen):
    d.qpos[:] = home_hinges()
    d.qvel[:] = 0.0
    # settle onto the center pose so the passive knee/ankle sit where the gait expects
    d.ctrl[:] = gen.center_pose()
    for _ in range(2000):
        mujoco.mj_step(m, d)
    d.qvel[:] = 0.0


def run_video(m, d, gen, path, duration, speed):
    import imageio
    ren = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.4, 0.0, -0.35]
    cam.distance, cam.azimuth, cam.elevation = 1.9, 90, -10
    reset_to_home(m, d, gen)
    sim_dt = m.opt.timestep
    fps = 50
    frames = []
    t = 0.0
    next_frame = 0.0
    ctrl_dt = 1.0 / CONTROL_HZ
    next_ctrl = 0.0
    while t < duration:
        if t >= next_ctrl:
            d.ctrl[:] = gen.targets(t)
            next_ctrl += ctrl_dt
        mujoco.mj_step(m, d)
        t += sim_dt
        if t >= next_frame:
            ren.update_scene(d, cam)
            frames.append(ren.render())
            next_frame += 1.0 / fps
    imageio.mimsave(path, frames, fps=fps)
    print(f"wrote {len(frames)} frames -> {path}")


def run_viewer(m, d, gen, speed):
    import mujoco.viewer
    reset_to_home(m, d, gen)
    sim_dt = m.opt.timestep
    ctrl_dt = 1.0 / CONTROL_HZ
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.lookat[:] = [0.4, 0.0, -0.35]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.9, 90, -10
        t0 = time.time()
        sim_t = 0.0
        next_ctrl = 0.0
        while viewer.is_running():
            wall = (time.time() - t0) * speed
            # step physics until sim time catches up to (scaled) wall time
            while sim_t < wall:
                if sim_t >= next_ctrl:
                    d.ctrl[:] = gen.targets(sim_t)
                    next_ctrl += ctrl_dt
                mujoco.mj_step(m, d)
                sim_t += sim_dt
            viewer.sync()
            time.sleep(0.001)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gravity", action="store_true", help="enable gravity (legs hang/load)")
    ap.add_argument("--floor", action="store_true", help="keep the ground plane (default: removed)")
    ap.add_argument("--no-collide", action="store_true",
                    help="disable leg-mesh self-collision (faster/cleaner; less faithful to the "
                         "rigid hardware, where the pushrod cannot pass through the hip)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--period", type=float, default=None)
    ap.add_argument("--thigh-amp", type=float, default=None)
    ap.add_argument("--cam-amp", type=float, default=None)
    ap.add_argument("--thigh-center", type=float, default=None)
    ap.add_argument("--cam-center", type=float, default=None)
    ap.add_argument("--ramp", type=float, default=None)
    ap.add_argument("--video", default=None, help="render to this mp4/gif instead of the live viewer")
    ap.add_argument("--duration", type=float, default=6.0, help="seconds (video mode)")
    args = ap.parse_args()

    m = build_fixed_base_model(gravity=args.gravity, floor=args.floor,
                               collidable=not args.no_collide)
    d = mujoco.MjData(m)
    gen = GaitGenerator(make_params(args))
    print(f"Model: {m.nq} dof, {m.nu} actuators. gravity={'on' if args.gravity else 'OFF'}, "
          f"floor={'on' if args.floor else 'off'}, "
          f"self-collision={'off' if args.no_collide else 'on'}. period={gen.p.period_s}s")
    print("center pose (rad):", np.round(gen.center_pose(), 3))

    if args.video:
        run_video(m, d, gen, args.video, args.duration, args.speed)
    else:
        run_viewer(m, d, gen, args.speed)


if __name__ == "__main__":
    main()
