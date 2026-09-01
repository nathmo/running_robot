"""Build deploy_map.json — the normalized-degrees -> model-radians map run_policy.py requires.

    python robot/deploy/make_deploy_map.py --model-map robot/fixed_gait/webui/data/model_map.json \
        --out robot/deploy/deploy_map.json

WHY THIS EXISTS
---------------
Two files already hold most of the answer and nothing ever joined them up:

  * `webui/data/model_map.json` — fklut's per-side cam/thigh sign+offset, FITTED against the
    recorded workspace by maximising the fraction of hand-recorded cells landing inside the
    simulated assembly band. On 2026-08-29 that fit was decisive on both legs at 100% coverage.
  * `deploy_map.json` — what jointmap.JointMap reads, in MODEL_TO_MOTOR order, which the policy
    runner refuses to start without.

Both use the same convention, `model = sign * norm + offset`, so cam and thigh transcribe
directly. Abduction is the part fklut cannot supply: its LUT is the 2-DOF planar leg, and
abduction is out of that plane entirely.

ABDUCTION, AND WHY THE TWO SIDES DIFFER
---------------------------------------
MEASURED on the model (walk_mit/model/dash01.xml), +0.25 rad of each hip_roll actuator:

    hip_roll_L  left  foot Y +0.1875 -> +0.3764   OUTWARD (away from the centreline)
    hip_roll_R  right foot Y -0.2054 -> -0.0126   INWARD  (toward the centreline)

Both hip_roll joints share the axis (+1, 0, 0) — unlike cam and thigh, which ARE mirrored
(0, +-1, 0). So positive model abduction swings both legs the same way in world terms, not both
outward.

MEASURED on the robot (operator, 2026-08-29): moving a leg away from the body drives normalized
abduction POSITIVE on both legs.

Left therefore agrees with the model and right is inverted:  sign +1 / -1.

The offset is 0 for both because the operator zeroes abduction with the legs aligned with the
base, which is also the model's qpos-0 pose, and hip_roll's range is symmetric (+-0.785 rad).
Unlike cam there is no 4-bar dead centre here to hide a large offset.

RE-RUN THIS AFTER EVERY RE-ZERO. Every drive re-randomises its raw origin on a power cycle and the
webui calibration is re-captured with it, which makes the offsets below stale — the same rule
JointMap.invalidate() states. Re-fit fklut first (POST /api/fk/verify), then re-run this.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jointmap                                                        # noqa: E402

# side -> (sign, offset_deg). See the module docstring for how each was established.
ABDUCTION = {"left": (+1.0, 0.0), "right": (-1.0, 0.0)}
ABD_NOTE = ("model hip_roll_{} moves the foot {} at +0.25 rad while the robot reads + outward; "
            "offset 0 because abduction is zeroed aligned with the base, which is model qpos-0")


def build(model_map, when=None):
    """fklut's per-side map + the measured abduction convention -> a JointMap."""
    entries, notes = {}, []
    for side in ("left", "right"):
        src = model_map.get(side)
        if not src:
            raise SystemExit("model_map.json has no {!r} side".format(side))
        for role in ("cam", "thigh"):
            entries["{}.{}".format(side, role)] = {
                "sign": float(src[role]),
                "offset_deg": float(src["{}_off_deg".format(role)]),
                "verified": bool(model_map.get("verified", {}).get(side)),
                "verified_when": when,
                "note": "fklut fit against the recorded workspace band",
            }
        sign, off = ABDUCTION[side]
        entries["{}.abd".format(side)] = {
            "sign": sign, "offset_deg": off, "verified": True, "verified_when": when,
            "note": ABD_NOTE.format("L" if side == "left" else "R",
                                    "outward" if sign > 0 else "inward"),
        }
        if not model_map.get("verified", {}).get(side):
            notes.append("{}: fklut has NOT verified this side -- re-run POST /api/fk/verify"
                         .format(side))
    return jointmap.JointMap(entries), notes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-map", default="robot/fixed_gait/webui/data/model_map.json")
    ap.add_argument("--out", default="robot/deploy/deploy_map.json")
    ap.add_argument("--when", default=None, help="stamp written into verified_when")
    a = ap.parse_args()

    with open(a.model_map, "r", encoding="utf-8-sig") as f:
        mm = json.load(f)
    jm, notes = build(mm, when=a.when)

    ok, why = jm.check_ready()
    print("model_map : {}".format(a.model_map))
    print("%-12s %6s %12s  %s" % ("joint", "sign", "offset_deg", "provenance"))
    for n in jointmap.MODEL_TO_MOTOR:
        e = jm.e[n]
        print("%-12s %+6.0f %+12.2f  %s" % (n, e["sign"], e["offset_deg"], e["note"][:58]))
    for w in notes:
        print("!! " + w)
    if not ok:
        print("!! " + why)
        return 1
    jm.save(a.out)
    print("\nwrote {}".format(a.out))
    print("run_policy.py will now start without --skip-jointmap-check.")
    print("RE-RUN after any re-zero: the offsets describe one calibration frame only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
