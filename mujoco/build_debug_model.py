from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "robotURDF" / "urdf" / "000_Assy_Full.SLDASM.csv"
XML_PATH = ROOT / "mujoco" / "running_robot_debug.xml"
MESH_DIR = ROOT / "robotURDF" / "meshes"
DEFAULT_HINGE_RANGE = (-1.5707963267949, 1.5707963267949)


def parse_float(value: str | None) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    return float(text.replace(",", "."))


def parse_vec(row: dict[str, str], prefix: str) -> tuple[float, float, float]:
    return (
        parse_float(row.get(f"{prefix} X")),
        parse_float(row.get(f"{prefix} Y")),
        parse_float(row.get(f"{prefix} Z")),
    )


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


def fmt(values: Any) -> str:
    if isinstance(values, (tuple, list)):
        return " ".join(f"{value:.15g}" for value in values)
    return f"{values:.15g}"


def slugify(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "link"


def basename_from_package_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_tree(rows: list[dict[str, str]]) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, str]]]:
    by_name: dict[str, dict[str, str]] = {}
    children: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        link_name = row["Link Name"].strip()
        by_name[link_name] = row

    for row in rows:
        link_name = row["Link Name"].strip()
        parent_name = row.get("Parent", "").strip()
        if parent_name:
            children[parent_name].append(row)

    return children, by_name


def unique_slugs(rows: list[dict[str, str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    counts: dict[str, int] = defaultdict(int)

    for row in rows:
        link_name = row["Link Name"].strip()
        base = slugify(link_name)
        counts[base] += 1
        slug = base if counts[base] == 1 else f"{base}_{counts[base]}"
        result[link_name] = slug

    return result


def mesh_file_from_row(row: dict[str, str]) -> str:
    for key in ("Mesh Filename", "Collision Mesh Filename"):
        value = row.get(key, "").strip()
        if value:
            return basename_from_package_uri(value)
    raise ValueError(f"No mesh filename found for {row['Link Name']}")


def should_include_row(row: dict[str, str]) -> bool:
    link_name = row["Link Name"].lower()
    if "four bar" not in link_name:
        return True
    return "right" in link_name


def emitted_joint_name(row: dict[str, str], slugs: dict[str, str]) -> str:
    link_name = row["Link Name"].strip()
    slug = slugs[link_name]
    joint_name = row.get("Joint Name", "").strip() or f"{slug}_joint"
    return f"{slug}_{slugify(joint_name)}"


def parse_axis(row: dict[str, str]) -> tuple[float, float, float]:
    axis = (
        parse_float(row.get("Joint Axis X")),
        parse_float(row.get("Joint Axis Y")),
        parse_float(row.get("Joint Axis Z")),
    )
    if axis == (0.0, 0.0, 0.0):
        return (0.0, 0.0, 1.0)
    return axis


def write_body(
    lines: list[str],
    row: dict[str, str],
    slugs: dict[str, str],
    children: dict[str, list[dict[str, str]]],
    indent: int,
    is_root: bool = False,
) -> None:
    pad = "  " * indent
    link_name = row["Link Name"].strip()
    slug = slugs[link_name]

    visual_pos = parse_vec(row, "Visual")
    visual_quat = quat_from_rpy(
        parse_float(row.get("Visual Roll")),
        parse_float(row.get("Visual Pitch")),
        parse_float(row.get("Visual Yaw")),
    )

    mass = parse_float(row.get("Mass"))
    inertia = (
        parse_float(row.get("Moment Ixx")),
        parse_float(row.get("Moment Iyy")),
        parse_float(row.get("Moment Izz")),
        parse_float(row.get("Moment Ixy")),
        parse_float(row.get("Moment Ixz")),
        parse_float(row.get("Moment Iyz")),
    )
    com_pos = parse_vec(row, "Center of Mass")
    com_quat = quat_from_rpy(
        parse_float(row.get("Center of Mass Roll")),
        parse_float(row.get("Center of Mass Pitch")),
        parse_float(row.get("Center of Mass Yaw")),
    )

    mesh_name = f"{slug}_mesh"
    mesh_file = mesh_file_from_row(row)
    rgba = (
        parse_float(row.get("Color Red")) or 0.75,
        parse_float(row.get("Color Green")) or 0.75,
        parse_float(row.get("Color Blue")) or 0.75,
        parse_float(row.get("Color Alpha")) or 1.0,
    )

    if is_root:
        lines.append(f'{pad}<body name="{slug}">')
        lines.append(f'{pad}  <freejoint name="{slug}_root_freejoint"/>')
        lines.append(f'{pad}  <site name="{slug}_frame" pos="0 0 0" size="0.007" rgba="0.95 0.75 0.1 1"/>')
    else:
        joint_origin_pos = parse_vec(row, "Joint Origin")
        joint_origin_quat = quat_from_rpy(
            parse_float(row.get("Joint Origin Roll")),
            parse_float(row.get("Joint Origin Pitch")),
            parse_float(row.get("Joint Origin Yaw")),
        )
        joint_axis = parse_axis(row)
        lines.append(
            f'{pad}<body name="{slug}" pos="{fmt(joint_origin_pos)}" quat="{fmt(joint_origin_quat)}">'
        )
        joint_name = row.get("Joint Name", "").strip() or f"{slug}_joint"
        lines.append(
            f'{pad}  <joint name="{slug}_{slugify(joint_name)}" type="hinge" axis="{fmt(joint_axis)}" limited="true" range="{DEFAULT_HINGE_RANGE[0]:.15g} {DEFAULT_HINGE_RANGE[1]:.15g}"/>'
        )
        lines.append(
            f'{pad}  <site name="{slug}_pivot_child" pos="0 0 0" size="0.006" rgba="0.15 0.65 0.95 1"/>'
        )

    lines.append(
        f'{pad}  <inertial pos="{fmt(com_pos)}" mass="{mass:.15g}" fullinertia="{fmt(inertia)}"/>'
    )
    lines.append(
        f'{pad}  <geom name="{slug}_geom" type="mesh" mesh="{mesh_name}" pos="{fmt(visual_pos)}" quat="{fmt(visual_quat)}" rgba="{fmt(rgba)}" contype="0" conaffinity="0" group="1"/>'
    )

    for child in children.get(link_name, []):
        child_link_name = child["Link Name"].strip()
        child_slug = slugs[child_link_name]
        child_joint_pos = parse_vec(child, "Joint Origin")
        child_joint_quat = quat_from_rpy(
            parse_float(child.get("Joint Origin Roll")),
            parse_float(child.get("Joint Origin Pitch")),
            parse_float(child.get("Joint Origin Yaw")),
        )
        lines.append(
            f'{pad}  <site name="{child_slug}_parent_anchor" pos="{fmt(child_joint_pos)}" quat="{fmt(child_joint_quat)}" size="0.006" rgba="0.95 0.45 0.1 1"/>'
        )

    for child in children.get(link_name, []):
        write_body(lines, child, slugs, children, indent + 1)

    lines.append(f"{pad}</body>")


def build_xml(rows: list[dict[str, str]]) -> str:
    included_rows = [row for row in rows if should_include_row(row)]
    slugs = unique_slugs(included_rows)
    children, by_name = build_tree(included_rows)

    roots = [row for row in included_rows if not row.get("Parent", "").strip() or row.get("Parent", "").strip() not in by_name]
    if not roots:
        raise ValueError("No root body found in CSV")

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append('<mujoco model="running_robot_debug">')
    lines.append('  <compiler angle="radian" coordinate="local" autolimits="true" inertiafromgeom="false" fusestatic="false" meshdir="../robotURDF/meshes"/>')
    lines.append('  <option gravity="0 0 0" integrator="Euler" solver="Newton" iterations="50" tolerance="1e-10"/>')
    lines.append('  <statistic meansize="0.25" extent="1.25" center="0 0 0"/>')
    lines.append('  <visual>')
    lines.append('    <global azimuth="135" elevation="-25" fovy="45" offwidth="1600" offheight="1200"/>')
    lines.append('    <quality shadowsize="2048" offsamples="4"/>')
    lines.append('    <headlight ambient="0.25 0.25 0.25" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>')
    lines.append('    <rgba joint="0.2 0.6 0.9 1" com="0.9 0.9 0.9 1" inertia="0.8 0.2 0.2 0.45" constraint="0.9 0 0 1"/>')
    lines.append('  </visual>')
    lines.append('  <default>')
    lines.append('    <geom rgba="0.7 0.7 0.7 1" contype="0" conaffinity="0"/>')
    lines.append('  </default>')
    lines.append('  <asset>')
    for row in included_rows:
        link_name = row["Link Name"].strip()
        slug = slugs[link_name]
        mesh_file = mesh_file_from_row(row)
        lines.append(f'    <mesh name="{slug}_mesh" file="{mesh_file}"/>')
    lines.append('  </asset>')
    lines.append('  <equality>')
    available_joints = {emitted_joint_name(row, slugs) for row in included_rows if row.get("Joint Name", "").strip()}
    for row in included_rows:
        link_name = row["Link Name"].strip()
        parent_name = row.get("Parent", "").strip()
        joint_name = row.get("Joint Name", "").strip()
        if not parent_name or parent_name not in by_name:
            continue
        if not joint_name:
            continue
        child_slug = slugs[link_name]
        lines.append(
            f'    <connect name="{child_slug}_{slugify(joint_name)}_placeholder" active="false" site1="{child_slug}_parent_anchor" site2="{child_slug}_pivot_child" solref="0.02 1" solimp="0.9 0.95 0.001"/>'
        )
    lines.append('  </equality>')
    lines.append('  <actuator>')
    actuator_targets = [
        ("femur_right_hip", "femur_right_hip_joint_right"),
        ("femur_left_hip", "femur_left_hip_joint_left"),
        ("hip_right_adduction", "hip_right_hip_joint_right_adduction"),
        ("hip_left_adduction", "hip_left_hip_joint_left_adduction"),
        ("right_hip_abduction", "four_bar_beam_right_four_bar_hip_joint"),
    ]
    for actuator_name, joint_name in actuator_targets:
        if joint_name not in available_joints:
            continue
        lines.append(
            f'    <motor name="{actuator_name}" joint="{joint_name}" gear="1" ctrllimited="true" ctrlrange="-1 1"/>'
        )
    lines.append('  </actuator>')
    lines.append('  <worldbody>')
    root = next((row for row in included_rows if row["Link Name"].strip() == "Main Body"), roots[0])
    write_body(lines, root, slugs, children, 2, is_root=True)
    lines.append('  </worldbody>')
    lines.append('</mujoco>')

    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows()
    xml = build_xml(rows)
    XML_PATH.write_text(xml, encoding="utf-8")
    print(f"Wrote {XML_PATH}")


if __name__ == "__main__":
    main()