#!/usr/bin/env python3
"""
Patch URDF joint <axis> entries using axes from MuJoCo XML.
Mapping is done by normalizing child link names to MuJoCo body names.
"""
from pathlib import Path
import xml.etree.ElementTree as ET


def normalize(name: str) -> str:
    return ''.join(ch for ch in name.lower() if ch.isalnum())


def parse_mujoco_axes(mj_path: Path):
    tree = ET.parse(mj_path)
    root = tree.getroot()
    axes = {}
    # traverse bodies and collect joints
    for body in root.findall('.//body'):
        bname = body.get('name', '')
        norm = normalize(bname)
        for joint in body.findall('joint'):
            jname = joint.get('name', '')
            axis = joint.get('axis')
            if axis:
                axes[norm] = axis
                # if multiple, keep first
                break
    return axes


def patch_urdf(urdf_path: Path, mujoco_axes: dict):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    updated = 0
    for joint in root.findall('joint'):
        child = joint.find('child')
        if child is None:
            continue
        child_link = child.get('link', '')
        norm = normalize(child_link)
        axis_val = mujoco_axes.get(norm)
        axis_elem = joint.find('axis')
        print(f'Joint {joint.get("name")} child={child_link} norm={norm} -> axis={axis_val}')
        if axis_val:
            if axis_elem is None:
                axis_elem = ET.SubElement(joint, 'axis')
            axis_elem.set('xyz', axis_val)
            updated += 1
    if updated:
        bak = urdf_path.with_suffix(urdf_path.suffix + '.axes.bak')
        if not bak.exists():
            bak.write_bytes(urdf_path.read_bytes())
        tree.write(urdf_path, encoding='utf-8', xml_declaration=True)
    return updated


def main():
    root = Path(__file__).resolve().parents[1]
    mj = root / 'mujoco' / 'running_robot_debug.xml'
    urdf = root / 'robotURDF' / 'urdf' / 'Assy_Full_Aligned_URDF.urdf'
    if not mj.exists() or not urdf.exists():
        print('Missing files')
        return
    mujoco_axes = parse_mujoco_axes(mj)
    updated = patch_urdf(urdf, mujoco_axes)
    print(f'Updated {updated} joint axis entries in {urdf}')


if __name__ == '__main__':
    main()
