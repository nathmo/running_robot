#!/usr/bin/env python3
"""
Patch URDF mesh filenames and inertial values using `inertia_results.json`.
Matches mesh basenames case-insensitively and ignores spaces/underscores.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET


def normalize(name: str) -> str:
    return ''.join(ch for ch in name.lower() if ch.isalnum())


def main():
    root = Path(__file__).resolve().parents[1]
    results_path = root / 'robotURDF' / 'meshes' / 'processed' / 'inertia_results.json'
    urdf_path = root / 'robotURDF' / 'urdf' / 'Assy_Full_Aligned_URDF.urdf'
    aligned_base = 'package://Assy_Full_Aligned_URDF/meshes/processed/aligned/'
    collision_base = 'package://Assy_Full_Aligned_URDF/meshes/processed/collision/'

    if not results_path.exists():
        print('Results file not found:', results_path)
        return
    if not urdf_path.exists():
        print('URDF not found:', urdf_path)
        return

    with results_path.open('r', encoding='utf-8') as f:
        results = json.load(f)

    norm_map = {normalize(k): k for k in results.keys()}

    tree = ET.parse(urdf_path)
    root_el = tree.getroot()
    updated = 0
    for link in root_el.findall('link'):
        for geom_parent in link.findall('visual') + link.findall('collision'):
            mesh_elem = geom_parent.find('geometry/mesh')
            if mesh_elem is None:
                continue
            filename = mesh_elem.get('filename', '')
            basename = Path(filename).name
            norm = normalize(basename)
            matched = norm_map.get(norm)
            if not matched:
                continue
            aligned_name = Path(results[matched]['aligned_file']).name
            # set new filename for visual; collision gets collision if exists
            if geom_parent.tag == 'visual':
                mesh_elem.set('filename', aligned_base + aligned_name)
            else:
                # collision
                col_name = aligned_name.replace('_aligned.stl', '_collision.stl')
                mesh_elem.set('filename', collision_base + col_name)
            updated += 1

        # update inertial if match found for any of the link's meshes
        # check visual first
        mesh_names = [ (geom.find('geometry/mesh').get('filename','')) for geom in link.findall('visual') if geom.find('geometry/mesh') is not None ]
        if not mesh_names:
            continue
        # normalize first visual mesh
        basename = Path(mesh_names[0]).name
        norm = normalize(basename)
        matched = norm_map.get(norm)
        if not matched:
            continue
        data = results[matched]
        com = data['center_of_mass']
        inertia = data['inertia_matrix']
        mass = data.get('mass')

        inertial = link.find('inertial')
        if inertial is None:
            inertial = ET.SubElement(link, 'inertial')
        origin = inertial.find('origin')
        if origin is None:
            origin = ET.SubElement(inertial, 'origin')
        origin.set('xyz', f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}")
        origin.set('rpy', '0 0 0')
        mass_el = inertial.find('mass')
        if mass_el is None:
            mass_el = ET.SubElement(inertial, 'mass')
        if mass is not None:
            mass_el.set('value', f"{mass:.6f}")
        inertia_el = inertial.find('inertia')
        if inertia_el is None:
            inertia_el = ET.SubElement(inertial, 'inertia')
        inertia_el.set('ixx', f"{inertia[0][0]:.6f}")
        inertia_el.set('ixy', f"{inertia[0][1]:.6f}")
        inertia_el.set('ixz', f"{inertia[0][2]:.6f}")
        inertia_el.set('iyy', f"{inertia[1][1]:.6f}")
        inertia_el.set('iyz', f"{inertia[1][2]:.6f}")
        inertia_el.set('izz', f"{inertia[2][2]:.6f}")
        updated += 1

    if updated:
        bak = urdf_path.with_suffix(urdf_path.suffix + '.patched.bak')
        if not bak.exists():
            bak.write_bytes(urdf_path.read_bytes())
        tree.write(urdf_path, encoding='utf-8', xml_declaration=True)
        print(f'Patched URDF ({updated} changes), backup at {bak}')
    else:
        print('No changes applied to URDF.')


if __name__ == '__main__':
    main()
