#!/usr/bin/env python3
"""
Patch URDF joint <origin> (xyz + rpy) and <axis> using MuJoCo worldbody data.
Reads MuJoCo running_robot_debug.xml and the original URDF backup
Assy_Full_Aligned_URDF.urdf.patched.bak (or .urdf.bak) to compute correct
parent->child joint transforms and writes the patched URDF, saving a backup.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import math
import numpy as np


def normalize(name: str) -> str:
    return ''.join(ch for ch in name.lower() if ch.isalnum())


def quat_to_rot(q):
    # q expected as (w,x,y,z)
    w, x, y, z = q
    # rotation matrix
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])
    return R


def rot_to_rpy(R):
    # returns roll, pitch, yaw (rpy) in radians
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return (x, y, z)


def parse_mujoco(mj_path: Path):
    tree = ET.parse(mj_path)
    root = tree.getroot()
    bodies = {}
    # collect worldbody direct children
    for body in root.findall('.//worldbody//body'):
        name = body.get('name','')
        norm = normalize(name)
        pos = body.get('pos')
        quat = body.get('quat')
        if pos:
            posv = tuple(float(x) for x in pos.split())
        else:
            posv = (0.0, 0.0, 0.0)
        if quat:
            q = tuple(float(x) for x in quat.split())
        else:
            q = (1.0, 0.0, 0.0, 0.0)
        # find joint axis if exists on this body
        axis = None
        for j in body.findall('joint'):
            a = j.get('axis')
            if a:
                axis = a
                break
        bodies[norm] = {
            'name': name,
            'pos': np.array(posv),
            'quat': q,
            'axis': axis
        }
    # also include top-level main body (if named 'main_body')
    # find first worldbody/body with name main_body or main body
    return bodies


def patch_urdf(urdf_src: Path, urdf_dst: Path, bodies: dict):
    tree = ET.parse(urdf_src)
    root = tree.getroot()
    updated = 0
    for joint in root.findall('.//joint'):
        parent = joint.find('parent')
        child = joint.find('child')
        if parent is None or child is None:
            continue
        parent_link = parent.get('link','')
        child_link = child.get('link','')
        pnorm = normalize(parent_link)
        cnorm = normalize(child_link)
        parent_body = bodies.get(pnorm)
        child_body = bodies.get(cnorm)
        if child_body is None:
            # try alternative: child_link as lowercase with underscores
            # skip if missing
            continue
        parent_pos = parent_body['pos'] if parent_body else np.zeros(3)
        parent_quat = parent_body['quat'] if parent_body else (1.0,0,0,0)
        child_pos = child_body['pos']
        child_quat = child_body['quat']
        # compute origin xyz = child_pos - parent_pos
        origin_xyz = child_pos - parent_pos
        # compute relative rotation R = R_parent_inv * R_child
        Rpar = quat_to_rot(parent_quat)
        Rchild = quat_to_rot(child_quat)
        Rrel = Rpar.T.dot(Rchild)
        rpy = rot_to_rpy(Rrel)
        # find or create origin element under joint
        origin_elem = joint.find('origin')
        if origin_elem is None:
            origin_elem = ET.SubElement(joint, 'origin')
        origin_elem.set('xyz', f'{origin_xyz[0]:.6f} {origin_xyz[1]:.6f} {origin_xyz[2]:.6f}')
        origin_elem.set('rpy', f'{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}')
        # axis
        axis_val = child_body.get('axis') or child_body.get('axis')
        # if axis defined on the child body in MuJoCo, use it; else, try child body's joint element
        if axis_val is None:
            # find in bodies mapping: maybe child's own 'axis' already set
            axis_val = child_body.get('axis')
        if axis_val:
            axis_elem = joint.find('axis')
            if axis_elem is None:
                axis_elem = ET.SubElement(joint, 'axis')
            axis_elem.set('xyz', axis_val)
        updated += 1
    # write dst
    if updated:
        # backup dst
        bak = urdf_dst.with_suffix(urdf_dst.suffix + '.joints.bak')
        if not bak.exists():
            bak.write_bytes(urdf_dst.read_bytes())
        tree.write(urdf_dst, encoding='utf-8', xml_declaration=True)
    return updated


def main():
    root = Path(__file__).resolve().parents[1]
    mj = root / 'mujoco' / 'running_robot_debug.xml'
    bak1 = root / 'robotURDF' / 'urdf' / 'Assy_Full_Aligned_URDF.urdf.patched.bak'
    bak2 = root / 'robotURDF' / 'urdf' / 'Assy_Full_Aligned_URDF.urdf.bak'
    if bak1.exists():
        urdf_src = bak1
    elif bak2.exists():
        urdf_src = bak2
    else:
        print('No URDF backup found to use as source')
        return
    urdf_dst = root / 'robotURDF' / 'urdf' / 'Assy_Full_Aligned_URDF.urdf'
    bodies = parse_mujoco(mj)
    updated = patch_urdf(urdf_src, urdf_dst, bodies)
    print(f'Patched {updated} joints in {urdf_dst} using source {urdf_src}')


if __name__ == '__main__':
    main()
