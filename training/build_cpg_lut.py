"""Build the foot-IK lookup table the CPG gait generator needs (training/model/cpg_foot_lut.npz).

WHY a LUT and not a formula. The CPG-RL mapping (Bellegarda & Ijspeert 2022) is written in FOOT
space: the oscillator's phase drives a foot position (fore-aft travel + a swing-phase lift), and an
inverse kinematics step turns that into joint targets. DASH-01's sagittal leg is a cam crank driving
a parallel 4-bar, so there is no closed-form IK and, worse, the naive joint-space reading is simply
wrong: measured from the stand keyframe, BOTH cam and thigh move the toe mostly FORE-AFT, and either
sign of cam SHORTENS the leg (the standing leg is near-straight, so leg length is second-order in
every joint). A hand-written "thigh = swing, cam = lift" mapping would therefore drive the foot
nowhere near the intended trajectory.

So we measure the forward map once, numerically, and invert it on a grid:

  forward : (dcam, dthigh) ctrl deltas from the stand pose  ->  (dx, dz) toe delta in the BODY frame
            obtained by actually settling the PD-controlled leg in MuJoCo with the base pinned in
            the air (no ground contact, which would otherwise clamp dz).
  inverse : regular grid over (dx, dz) -> the (dcam, dthigh) that reaches it with the smallest joint
            excursion. Cells outside the reachable set are filled with their nearest reachable
            neighbour, so the runtime lookup is total (never NaN) and saturates gracefully.

The runtime side (cpg_gait.foot_ik) bilinearly interpolates the inverse grid — pure numpy, so it
ships to the Pi with everything else.

Run:  python training/build_cpg_lut.py [--preset m7_freq] [--out training/model/cpg_foot_lut.npz]
"""
import argparse
import os
import numpy as np
import mujoco

from config import get_config
from env import DashEnv

L_CAM, L_THIGH = 1, 2          # actuator indices of the LEFT sagittal pair (see fourier_gait)


def measure_forward(cfg, cam_grid, thigh_grid, settle=900, base_h=2.0):
    """Toe (x, z) in the body frame for every (dcam, dthigh), leg hanging free."""
    env = DashEnv(cfg)
    m, d = env.model, env.data
    gL = env.foot_gids[0]
    quat0 = env.default_qpos[3:7].copy()

    def settle_at(ctrl, n):
        d.ctrl[:] = ctrl
        for _ in range(n):
            # pin the base: hold it high (no contact) and level, and kill its velocity, so what we
            # measure is leg kinematics alone
            d.qpos[0:3] = (0.0, 0.0, base_h)
            d.qpos[3:7] = quat0
            d.qvel[0:6] = 0.0
            mujoco.mj_step(m, d)
        base = d.xpos[env.base_id].copy()
        R = d.xmat[env.base_id].reshape(3, 3)
        return R.T @ (d.geom_xpos[gL] - base)

    mujoco.mj_resetDataKeyframe(m, d, env.key_id)
    ref = settle_at(env.nominal_ctrl.copy(), 4000)      # nominal toe, well settled
    out = np.zeros((len(cam_grid), len(thigh_grid), 2))
    for i, dc in enumerate(cam_grid):
        # walk each cam column from the settled previous state (warm start -> fewer substeps)
        mujoco.mj_resetDataKeyframe(m, d, env.key_id)
        settle_at(env.nominal_ctrl + np.array([0, dc, 0, 0, 0, 0]), settle)
        for j, dt in enumerate(thigh_grid):
            ctrl = env.nominal_ctrl.copy()
            ctrl[L_CAM] += dc
            ctrl[L_THIGH] += dt
            p = settle_at(ctrl, settle)
            out[i, j] = (p[0] - ref[0], p[2] - ref[2])
        print(f"  cam {dc:+.3f}  dx[{out[i, :, 0].min():+.3f},{out[i, :, 0].max():+.3f}] "
              f"dz[{out[i, :, 1].min():+.3f},{out[i, :, 1].max():+.3f}]", flush=True)
    return ref, out


def invert(cam_grid, thigh_grid, fwd, dx_grid, dz_grid, effort=0.005, fold_dz=0.40, cont=0.02):
    """Invert the measured forward map onto a regular (dx, dz) grid — CONTINUOUSLY.

    Samples with dz > fold_dz are DROPPED first: past ~0.4 m of apparent "lift" on a 1 m leg the
    4-bar has snapped onto its folded branch (the cam-fold the fixed_gait work already hit). Those
    poses are reachable in sim but are not a gait — leaving them in lets the IK answer a modest lift
    request with a fold, which is exactly the failure the physical robot must never learn.

    The inversion is a flood fill from the (0, 0) cell rather than an independent argmin per cell,
    because the 4-bar is REDUNDANT: several (dcam, dthigh) reach the same toe point, and choosing
    each cell on its own makes the table jump between solution branches. Measured on the first
    build, that put a 0.8 rad thigh discontinuity in the middle of the swing phase — a step the
    motors would have to take in one 5 ms control tick. So each cell is chosen as

        argmin_k  ||fwd_k - target||  +  cont * ||joints_k - anchor||   +  effort * ||joints_k||

    where `anchor` is the mean of the neighbours already assigned. Propagating outward from a seed
    keeps the whole table on one branch, and `effort` still breaks ties toward the smaller joint
    excursion. `reach` records the position residual so the caller can see what is really reachable.
    """
    CC, TT = np.meshgrid(cam_grid, thigh_grid, indexing="ij")
    joints = np.stack([CC.ravel(), TT.ravel()], 1)          # (K, 2)
    pts = fwd.reshape(-1, 2)                                # (K, 2)
    ok = pts[:, 1] <= fold_dz
    joints, pts = joints[ok], pts[ok]
    print(f"[cpg-lut] dropped {int((~ok).sum())}/{ok.size} folded-branch samples (dz > {fold_dz} m)")
    jcost = effort * np.linalg.norm(joints, axis=1)

    A, B = len(dx_grid), len(dz_grid)
    inv = np.zeros((A, B, 2))
    reach = np.full((A, B), np.nan)
    done = np.zeros((A, B), bool)

    def pick(a, b, anchor):
        err = np.linalg.norm(pts - np.array([dx_grid[a], dz_grid[b]]), axis=1)
        c = err + jcost
        if anchor is not None:
            c = c + cont * np.linalg.norm(joints - anchor, axis=1)
        k = int(np.argmin(c))
        inv[a, b] = joints[k]
        reach[a, b] = err[k]
        done[a, b] = True

    seed = (int(np.argmin(np.abs(dx_grid))), int(np.argmin(np.abs(dz_grid))))
    pick(seed[0], seed[1], None)
    # breadth-first so every cell is decided from neighbours that are already on the chosen branch
    from collections import deque
    q = deque([seed])
    while q:
        a, b = q.popleft()
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            na, nb = a + da, b + db
            if not (0 <= na < A and 0 <= nb < B) or done[na, nb]:
                continue
            nb_vals = [inv[na + p, nb + q_] for p, q_ in ((1, 0), (-1, 0), (0, 1), (0, -1))
                       if 0 <= na + p < A and 0 <= nb + q_ < B and done[na + p, nb + q_]]
            pick(na, nb, np.mean(nb_vals, axis=0))
            q.append((na, nb))
    return inv, reach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="m7_freq", help="preset whose plant/keyframe defines the LUT")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cam", type=float, default=0.9, help="half-range of the cam ctrl delta (rad)")
    ap.add_argument("--thigh", type=float, default=0.6, help="half-range of the thigh ctrl delta")
    ap.add_argument("--n-cam", type=int, default=37)
    ap.add_argument("--n-thigh", type=int, default=25)
    # Operating envelope of the FOOT trajectory the CPG is allowed to ask for. Deliberately far
    # inside what the joints can reach: on a ~1 m leg a running stride is ~+-0.25 m of fore-aft
    # travel and <=0.12 m of swing lift. Asking for more only buys folded-branch nonsense.
    ap.add_argument("--dx-max", type=float, default=0.30, help="fore-aft half-span of the IK box (m)")
    ap.add_argument("--dz-max", type=float, default=0.10, help="max swing lift of the IK box (m)")
    ap.add_argument("--smooth", type=int, default=8, help="box-blur passes over the inverse table")
    ap.add_argument("--reinvert", action="store_true",
                    help="reuse the forward map already in --out and only redo the inversion")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "model", "cpg_foot_lut.npz")

    if args.reinvert:
        z = np.load(out, allow_pickle=True)
        cam_grid, thigh_grid, fwd, ref = z["cam_grid"], z["thigh_grid"], z["fwd"], z["ref"]
        print(f"[cpg-lut] reusing forward map from {out}")
    else:
        cam_grid = np.linspace(-args.cam, args.cam, args.n_cam)
        thigh_grid = np.linspace(-args.thigh, args.thigh, args.n_thigh)
        cfg = get_config(args.preset)
        print(f"[cpg-lut] measuring forward map ({args.n_cam}x{args.n_thigh}) "
              f"on preset {args.preset}")
        ref, fwd = measure_forward(cfg, cam_grid, thigh_grid)
    print(f"[cpg-lut] nominal toe (body frame) = {np.round(ref, 4)}")

    # inverse box: the running envelope, not the joint envelope (dz > 0 = shorter leg = foot up).
    dx_grid = np.linspace(-args.dx_max, args.dx_max, 81)
    dz_grid = np.linspace(0.0, args.dz_max, 25)
    print(f"[cpg-lut] inverse box: dx +-{args.dx_max:.3f} m, dz 0..{args.dz_max:.3f} m")
    inv, reach = invert(cam_grid, thigh_grid, fwd, dx_grid, dz_grid)
    print(f"[cpg-lut] residual: median {np.median(reach)*1000:.1f} mm, "
          f"p90 {np.percentile(reach, 90)*1000:.1f} mm, max {reach.max()*1000:.1f} mm")
    # Nearest-neighbour inversion snaps to the forward grid, which puts STAIRCASE steps into the
    # reconstructed joint waveform — the one thing this whole experiment is trying not to feed the
    # motors. A couple of 3x3 box passes (edges replicated) remove the steps; the map stays within
    # ~one forward-grid cell of the measured inverse, which is inside the measurement error anyway.
    for _ in range(args.smooth):
        p = np.pad(inv, ((1, 1), (1, 1), (0, 0)), mode="edge")
        inv = sum(p[a:a + inv.shape[0], b:b + inv.shape[1]]
                  for a in range(3) for b in range(3)) / 9.0
    print(f"[cpg-lut] smoothed the inverse table with {args.smooth} box passes")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, cam_grid=cam_grid, thigh_grid=thigh_grid, fwd=fwd, ref=ref,
             dx_grid=dx_grid, dz_grid=dz_grid, inv=inv, reach=reach,
             preset=args.preset, nominal_toe=ref)
    print(f"[cpg-lut] wrote {out}")


if __name__ == "__main__":
    main()
