"""Calibrate the v2 pushrod series spring to the measured loaded-stance deflection (artifact §07).

Hardware: 5 mm loaded-stance deflection, taken as the stance leg under the robot's own weight
(~148 N) => foot-referred k_link ~ 30 kN/m. The sim's compliance lives in a slide joint along
each pushrod (make_v2_plant.py); its stiffness is NOT 30 kN/m because the rod sees the foot load
through the 4-bar's lever ratio, so it is found numerically: bisection on log k until the
one-leg-stance sink (relative to a rigid rod, so contact and PD compliance cancel) hits the
target. Also reports the sink at the +-50 % DR extremes (the 15-45 kN/m band = 3-10 mm) and at
3.5 BW (must stay <= 2 cm), then writes dash01_v2.xml.

    python model/calibrate_loop_spring.py [--target-mm 5] [--armature HIP CT] [--dry-run]
    python model/calibrate_loop_spring.py --literal 30000      # bake k = 30 kN/m at the rod

FINDING (2026-09-09, first run): the one-leg stance CANNOT reach 5 mm through the rod. At the
settled stance the leg is a near-straight strut (99.5 % reach) and the pushrod carries only
10-45 N of the 148 N (the knee moment is tiny), so the sink is <= 0.5 mm for ANY rod stiffness
(0.54 mm at 5 kN/m, 0.12 mm at 100 kN/m; softer rods carry even less). The 5 mm measured on the
robot is therefore not rod compliance -- it is drive/structure compliance in the main load path.
RESOLUTION: the spring moved into the main load path -- a prismatic joint along the vertical
foot strut (ankle -> toe), `--spring shin` (the default), where the bisection lands at
13.06 kN/m axial = 5.0 mm at 1 BW (DR +-50 %: 9.7 / 3.4 mm). This is also where the GPU port
(walk_v2/) put its spring, so both implementations meet the artifact's verification criterion.
`--spring rod` keeps the literal pushrod-tip variant. NOTE the "3.5 BW <= 2 cm" figure printed
below scales gravity on the WHOLE robot in the one-leg rig, so it is dominated by PD sag
amplification through the bent leg (76 mm), not by the spring (17 mm axial); the artifact's
2 cm refers to the scripted-walk base drop during stepping, a different protocol.
"""
import argparse
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_v2_plant as mk  # noqa: E402


SPRING = "shin"
_MASS = None


def _mass():
    global _MASS
    if _MASS is None:
        import mujoco
        _MASS = mk.spring_mass(mujoco.MjModel.from_xml_path(mk.SRC), SPRING)
    return _MASS


def damping(k):
    return 2.0 * 0.7 * np.sqrt(k * _mass())          # zeta 0.7 on the moved mass


def sink_mm(rod_k, armature=None, side="L"):
    m = mk.build_model(rod_k, damping(rod_k), armature, spring=SPRING)
    return 1e3 * mk.one_leg_sink(m, side=side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mm", type=float, default=5.0)
    ap.add_argument("--armature", type=float, nargs=2, default=None, metavar=("HIP", "CT"))
    ap.add_argument("--dry-run", action="store_true", help="calibrate, do not write the XML")
    ap.add_argument("--out", default=mk.OUT)
    ap.add_argument("--literal", type=float, default=None,
                    help="skip the bisection and bake this rod stiffness (N/m), reporting the sink")
    ap.add_argument("--spring", default="shin", choices=("shin", "rod"),
                    help="where the series spring sits: shin axis (fits 5 mm) or pushrod tip")
    args = ap.parse_args()
    global SPRING
    SPRING = args.spring

    if args.literal is not None:
        k = float(args.literal)
        b = damping(k)
        m = mk.build_model(k, b, args.armature, spring=SPRING)
        print(f"rod k = {k:.0f} N/m (b {b:.1f}): one-leg sink L {1e3 * mk.one_leg_sink(m, side='L'):.2f} mm, "
              f"R {1e3 * mk.one_leg_sink(m, side='R'):.2f} mm;  DR band 0.5k {sink_mm(0.5 * k, args.armature):.2f} mm, "
              f"1.5k {sink_mm(1.5 * k, args.armature):.2f} mm")
        if not args.dry_run:
            out, m = mk.write(k, b, args.armature, out=args.out, spring=SPRING)
            print(f"wrote {out}  (nq {m.nq}, nv {m.nv})")
        return

    lo, hi = (3e3, 3e5) if SPRING == "shin" else (2e4, 4e6)    # N/m bracket on the spring
    s_lo, s_hi = sink_mm(lo, args.armature), sink_mm(hi, args.armature)
    print(f"bracket: k {lo:.0f} -> {s_lo:.2f} mm, k {hi:.0f} -> {s_hi:.2f} mm (target {args.target_mm} mm)")
    if not (s_hi < args.target_mm < s_lo):
        raise SystemExit("target outside the bracket -- widen lo/hi")
    for it in range(18):
        mid = float(np.sqrt(lo * hi))
        s = sink_mm(mid, args.armature)
        if s > args.target_mm:
            lo, s_lo = mid, s
        else:
            hi, s_hi = mid, s
        print(f"  it {it:2d}: k {mid:9.0f} N/m -> {s:.3f} mm")
        if abs(s - args.target_mm) < 0.02:
            break
    k = float(np.sqrt(lo * hi))
    b = damping(k)
    print(f"\ncalibrated {SPRING} spring k = {k:.0f} N/m (b = {b:.1f} N s/m, zeta 0.7 on the moved mass)")
    m = mk.build_model(k, b, args.armature, spring=SPRING)
    sL, sR = 1e3 * mk.one_leg_sink(m, side="L"), 1e3 * mk.one_leg_sink(m, side="R")
    print(f"  one-leg sink  L {sL:.2f} mm   R {sR:.2f} mm   (spec: 5 +- 1 mm)")
    for name, sc in (("DR low  (0.5 k)", 0.5), ("DR high (1.5 k)", 1.5)):
        print(f"  {name}: {sink_mm(k * sc, args.armature):.2f} mm")
    # 3.5 BW check by scaling gravity (the leg is linear in this range if the number is small)
    m35 = mk.build_model(k, b, args.armature, spring=SPRING)
    m35.opt.gravity[2] *= 3.5
    print(f"  3.5 BW sink: {1e3 * mk.one_leg_sink(m35):.1f} mm   (must be <= 20 mm)")
    if args.dry_run:
        return
    out, m = mk.write(k, b, args.armature, out=args.out, spring=SPRING)
    print(f"wrote {out}  (nq {m.nq}, nv {m.nv})")


if __name__ == "__main__":
    main()
