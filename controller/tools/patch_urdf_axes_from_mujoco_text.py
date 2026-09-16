#!/usr/bin/env python3
"""
Text-based URDF joint axis patcher using MuJoCo XML axes.
This avoids XML parsing issues by operating on the URDF text and replacing
or inserting <axis xyz="..."/> lines inside each <joint> block.
"""
import re
from pathlib import Path
import xml.etree.ElementTree as ET


def normalize(name: str) -> str:
    return ''.join(ch for ch in name.lower() if ch.isalnum())


def get_mj_axes(mj_path: Path):
    tree = ET.parse(mj_path)
    axes = {}
    for body in tree.findall('.//body'):
        name = body.get('name','')
        norm = normalize(name)
        for joint in body.findall('joint'):
            axis = joint.get('axis')
            if axis:
                axes[norm] = axis
                break
    return axes


def patch_text(urdf_path: Path, axes_map: dict):
    text = urdf_path.read_text(encoding='utf-8')
    joint_re = re.compile(r'(<joint\b.*?>)(.*?)(</joint>)', re.DOTALL)
    changed = 0

    def repl(m):
        nonlocal changed
        head, body, tail = m.group(1), m.group(2), m.group(3)
        child_m = re.search(r'<child\s+link="([^"]+)"\s*/?>', body)
        if not child_m:
            return m.group(0)
        child = child_m.group(1)
        norm = normalize(child)
        axis_val = axes_map.get(norm)
        if not axis_val:
            return m.group(0)
        if re.search(r'<axis\b', body):
            body2 = re.sub(r'(<axis[^>]*xyz=")([^"]+)("[^>]*/?>)', r"\1" + axis_val + r"\3", body, count=1)
        else:
            # insert axis before limit or at end of body
            ins = f'    <axis xyz="{axis_val}" />\n'
            if '<limit' in body:
                body2 = re.sub(r'(\s*<limit\b)', ins + r'\1', body, count=1)
            else:
                body2 = body + '\n' + ins
        if body2 != body:
            changed += 1
        return head + body2 + tail

    new_text = joint_re.sub(repl, text)
    if changed:
        bak = urdf_path.with_suffix(urdf_path.suffix + '.axes_text.bak')
        if not bak.exists():
            bak.write_text(text, encoding='utf-8')
        urdf_path.write_text(new_text, encoding='utf-8')
    return changed


def main():
    root = Path(__file__).resolve().parents[1]
    mj = root / 'mujoco' / 'running_robot_debug.xml'
    urdf = root / 'robotURDF' / 'urdf' / 'Assy_Full_Aligned_URDF.urdf'
    if not mj.exists() or not urdf.exists():
        print('Missing files')
        return
    axes_map = get_mj_axes(mj)
    print('MuJoCo axes map:', axes_map)
    changed = patch_text(urdf, axes_map)
    print(f'Patched {changed} joint axis entries in {urdf}')


if __name__ == '__main__':
    main()
