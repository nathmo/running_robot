"""Patch a compiled dash01.xml from identified_params.json — PURE ElementTree (no mujoco).

Writes the measured values back into the model to "close the loop": per-body inertials (mass, CoM,
full inertia tensor), per-joint rotor armature + viscous damping, and the actuator/motor masses.
Anything absent in the JSON is left at its CAD value. This operates on the ALREADY-COMPILED model
file (robust — it does not depend on re-running the CAD-export build), and is also called by
build_model.py so a fresh regen picks up the same measured values.

    python -m identification.apply_identified --model dash01.xml --params identified_params.json
"""
import argparse
import xml.etree.ElementTree as ET

# webui motor name -> model joint name (mirror of dataset.MOTOR_TO_JOINT; kept dep-free here)
MOTOR_TO_JOINT = {
    "right.abd": "bodyNCS-v1_Révolution-2", "left.abd": "bodyNCS-v1_Révolution-1",
    "right.cam": "HipRightNCS-v1_Révolution-4", "left.cam": "HipLeftNCS-v1_Révolution-3",
    "right.thigh": "HipRightNCS-v1_Révolution-6", "left.thigh": "HipLeftNCS-v1_Révolution-5",
}


def _fullinertia(inertia):
    return " ".join(f"{inertia[k]:.9g}" for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))


def apply(model_path, params, out_path=None):
    """params: dict (identified_params.json). Returns a summary of what was patched."""
    out_path = out_path or model_path
    tree = ET.parse(model_path)
    root = tree.getroot()
    changed = {"bodies": [], "armature": [], "damping": []}

    bodies = {b.get("name"): b for b in root.iter("body")}
    for name, bp in (params.get("bodies") or {}).items():
        body = bodies.get(name)
        if body is None:
            continue
        inertial = body.find("inertial")
        if inertial is None:
            continue
        if bp.get("mass") is not None:
            inertial.set("mass", f"{float(bp['mass']):.9g}")
        if bp.get("com") is not None:
            inertial.set("pos", " ".join(f"{float(v):.9g}" for v in bp["com"]))
        if bp.get("inertia"):
            inertial.attrib.pop("diaginertia", None)
            inertial.attrib.pop("quat", None)
            inertial.set("fullinertia", _fullinertia(bp["inertia"]))
        changed["bodies"].append(name)

    arm = params.get("rotor_armature") or {}
    fr = params.get("friction") or {}
    joints = {j.get("name"): j for j in root.iter("joint")}
    for motor, jname in MOTOR_TO_JOINT.items():
        j = joints.get(jname)
        if j is None:
            continue
        if motor in arm:
            j.set("armature", f"{float(arm[motor]):.6g}")
            changed["armature"].append(motor)
        if motor in fr and fr[motor].get("viscous") is not None:
            j.set("damping", f"{float(fr[motor]['viscous']):.6g}")
            changed["damping"].append(motor)

    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="unicode", xml_declaration=False)
    return changed


def main():
    import json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", default=None, help="output path (default: overwrite --model)")
    args = ap.parse_args()
    with open(args.params, "r", encoding="utf-8-sig") as f:
        params = json.load(f)
    changed = apply(args.model, params, args.out)
    print(f"patched {args.out or args.model}: "
          f"{len(changed['bodies'])} bodies, {len(changed['armature'])} armatures, "
          f"{len(changed['damping'])} dampings")


if __name__ == "__main__":
    main()
