#!/usr/bin/env python3
"""
Interactive mesh alignment and collision generator.

Features:
- For each STL in a meshes directory: compute PCA of vertex coordinates,
  report eigenvalues/eigenvectors and percent error vs world axes.
- Optionally translate mesh centroid to origin and rotate so principal axes
  align with X (largest variance), Y, Z.
- Compute mass properties (inertia tensor) in the aligned mesh frame and
  save per-mesh inertial data to a JSON file.
- Generate a simplified collision mesh (convex hull) per mesh.

Dependencies: trimesh, numpy
Install: pip install trimesh numpy

Usage: python tools/recenter_align_meshes.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import sys
import math
import numpy as np
import trimesh


def find_stl_files(folder: Path):
    return sorted(folder.rglob('*.stl'))


def pca_axes(vertices: np.ndarray):
    # vertices: (N,3)
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    cov = np.cov(centered, rowvar=False)
    # eigh returns ascending eigenvalues
    vals, vecs = np.linalg.eigh(cov)
    # sort descending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    # ensure right-handed
    if np.linalg.det(vecs) < 0:
        vecs[:, 2] *= -1
    return centroid, vals, vecs


def alignment_errors(vecs: np.ndarray):
    # compute 1 - abs(dot) with world axes
    axes = np.eye(3)
    errors = []
    for i in range(3):
        dot = np.dot(vecs[:, i], axes[:, i])
        errors.append(1.0 - abs(dot))
    return np.array(errors)


def align_mesh(mesh: trimesh.Trimesh, centroid: np.ndarray, pca_vecs: np.ndarray):
    # translate to origin then rotate so pca_vecs become identity axes
    centered_vertices = mesh.vertices - centroid
    # rotation matrix to new coords: R = V^T where V columns are pca_vecs
    R = pca_vecs.T
    # apply rotation
    new_vertices = centered_vertices.dot(R.T)
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces, process=False)
    return new_mesh, R


def format_vec(v):
    return '[' + ', '.join(f'{x: .6f}' for x in v) + ']' 


def print_summary(path: Path, centroid, vals, vecs, errors):
    print('\n' + '=' * 60)
    print(f'Mesh: {path}')
    print(f'  centroid: {centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}')
    print('  eigenvalues (variance):', ', '.join(f'{v:.6f}' for v in vals))
    print('  principal axes (columns):')
    for i in range(3):
        print(f'    axis {i} (target {"XYZ"[i]}): {format_vec(vecs[:, i])}  error={errors[i]:.6%}')


def compute_inertia(mesh: trimesh.Trimesh, density: float = 1.0):
    # trimesh API can expose mass_properties as a callable or a cached object.
    mass_props = None
    try:
        if callable(getattr(mesh, 'mass_properties', None)):
            mass_props = mesh.mass_properties(density=density)
        else:
            # some trimesh versions expose a MassProperties object on the attribute
            mass_props = mesh.mass_properties
    except Exception:
        # fallback to trimesh function
        try:
            mass_props = trimesh.mass.mass_properties(mesh, density=density)
        except Exception as e:
            raise RuntimeError('Failed to compute mass properties: ' + str(e))

    # mass_props may be a dict-like or an object with attributes
    mass = None
    if isinstance(mass_props, dict):
        com = mass_props.get('center_mass')
        inertia = mass_props.get('inertia')
        mass = mass_props.get('mass')
    else:
        com = getattr(mass_props, 'center_mass', None)
        inertia = getattr(mass_props, 'inertia', None)
        mass = getattr(mass_props, 'mass', None)

    if com is None:
        com = [0.0, 0.0, 0.0]
    if inertia is None:
        inertia = np.zeros((3, 3)).tolist()

    return com, np.array(inertia), float(mass) if mass is not None else None


def make_collision_mesh(mesh: trimesh.Trimesh, try_simplify=True):
    # first try convex hull (fast, conservative)
    hull = mesh.convex_hull
    if try_simplify:
        try:
            # attempt quadratic decimation if available
            simplified = hull.simplify_quadratic_decimation(max(len(hull.faces) // 2, 100))
            if simplified and len(simplified.faces) > 0:
                return simplified
        except Exception:
            pass
    return hull


def prompt_choice(prompt: str, choices: str = 'y/n'):
    choices = choices.lower()
    while True:
        resp = input(f"{prompt} [{choices}]: ").strip().lower()
        if resp == '' and 'y' in choices:
            return 'y'
        if resp in choices.split('/'):
            return resp
        if resp in list(choices):
            return resp


def process_all(mesh_folder: Path, out_folder: Path, density: float, update_urdf: Path | None):
    out_folder.mkdir(parents=True, exist_ok=True)
    aligned_folder = out_folder / 'aligned'
    collision_folder = out_folder / 'collision'
    aligned_folder.mkdir(parents=True, exist_ok=True)
    collision_folder.mkdir(parents=True, exist_ok=True)
    stls = find_stl_files(mesh_folder)
    if not stls:
        print('No STL files found in', mesh_folder)
        return

    inertia_results = {}

    for stl in stls:
        print('\nProcessing:', stl)
        mesh = trimesh.load(stl, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            print('  skipping (not a mesh)')
            continue

        centroid, vals, vecs = pca_axes(mesh.vertices)
        errors = alignment_errors(vecs)
        print_summary(stl, centroid, vals, vecs, errors)

        # if all errors very small, offer to skip quickly
        if np.all(errors < 0.001):
            print('  All alignment errors < 0.1% — likely already aligned.')
            resp = prompt_choice('Keep as-is (skip transforms)?', 'y/n')
            if resp == 'y':
                # still compute inertia and optionally collision
                aligned_mesh = mesh.copy()
            else:
                aligned_mesh, R = align_mesh(mesh, centroid, vecs)
        else:
            print('\nProposed action: translate centroid to origin and rotate principal axes -> XYZ')
            resp = prompt_choice('Apply alignment transform?', 'y/n')
            if resp == 'y':
                aligned_mesh, R = align_mesh(mesh, centroid, vecs)
                out_path = aligned_folder / (stl.stem + '_aligned.stl')
                aligned_mesh.export(out_path)
                print('  Saved aligned mesh to', out_path)
            else:
                aligned_mesh = mesh.copy()

        # compute inertia in aligned frame
        com, inertia, mass = compute_inertia(aligned_mesh, density=density)
        key = stl.name
        inertia_results[key] = {
            'aligned_file': str((aligned_folder / (stl.stem + '_aligned.stl')).relative_to(mesh_folder.parent)) if (aligned_folder / (stl.stem + '_aligned.stl')).exists() else None,
            'center_of_mass': [float(x) for x in (com.tolist() if com is not None else [0,0,0])],
            'inertia_matrix': inertia.tolist(),
            'mass': float(mass) if mass is not None else None,
        }

        print('\nMass properties (density={})'.format(density))
        print('  center_of_mass:', inertia_results[key]['center_of_mass'])
        print('  inertia matrix:')
        for row in inertia:
            print('   ', ['{:.6f}'.format(x) for x in row])

        # collision mesh
        resp = prompt_choice('Generate convex-hull collision mesh?', 'y/n')
        if resp == 'y':
            col = make_collision_mesh(aligned_mesh)
            col_path = collision_folder / (stl.stem + '_collision.stl')
            col.export(col_path)
            print('  Saved collision mesh to', col_path)

    # write inertia results
    results_path = out_folder / 'inertia_results.json'
    with results_path.open('w', encoding='utf-8') as f:
        json.dump(inertia_results, f, indent=2)
    print('\nWrote inertia results to', results_path)
    # optionally update URDF inertial entries
    if update_urdf:
        update_urdf_file(update_urdf, inertia_results, mesh_folder, aligned_folder, collision_folder)


def update_urdf_file(urdf_path: Path, inertia_results: dict, mesh_folder: Path, aligned_folder: Path, collision_folder: Path):
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        print('URDF file not found:', urdf_path)
        return

    backup = urdf_path.with_suffix(urdf_path.suffix + '.bak')
    if not backup.exists():
        backup.write_bytes(urdf_path.read_bytes())

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # For each link, rewrite visual/collision mesh filenames to the processed meshes.
    mesh_filenames = set(inertia_results.keys())
    updated = 0
    for link in root.findall('link'):
        link_name = link.get('name', '')
        # look for mesh filename under visual or collision
        found_mesh = None
        for geom_parent in link.findall('visual') + link.findall('collision'):
            mesh_elem = geom_parent.find('geometry/mesh')
            if mesh_elem is None:
                continue
            filename = mesh_elem.get('filename', '')
            # compare by basename
            basename = Path(filename).name
            if basename in mesh_filenames:
                found_mesh = basename
                break

        if not found_mesh:
            continue

        processed_name = Path(found_mesh).stem + '_aligned.stl'
        collision_name = Path(found_mesh).stem + '_collision.stl'
        updated_here = False
        for geom_parent in link.findall('visual'):
            mesh_elem = geom_parent.find('geometry/mesh')
            if mesh_elem is None:
                continue
            basename = Path(mesh_elem.get('filename', '')).name
            if basename != found_mesh:
                continue
            new_path = f'package://Assy_Full_Aligned_URDF/meshes/processed/aligned/{processed_name}'
            mesh_elem.set('filename', new_path)
            updated_here = True

        for geom_parent in link.findall('collision'):
            mesh_elem = geom_parent.find('geometry/mesh')
            if mesh_elem is None:
                continue
            basename = Path(mesh_elem.get('filename', '')).name
            if basename != found_mesh:
                continue
            collision_path = collision_folder / collision_name
            if collision_path.exists():
                new_path = f'package://Assy_Full_Aligned_URDF/meshes/processed/collision/{collision_name}'
            else:
                new_path = f'package://Assy_Full_Aligned_URDF/meshes/processed/aligned/{processed_name}'
            mesh_elem.set('filename', new_path)
            updated_here = True

        data = inertia_results.get(found_mesh)
        if data:
            com = data['center_of_mass']
            inertia = np.array(data['inertia_matrix'])
            mass = data.get('mass', None)

            # find or create inertial element
            inertial = link.find('inertial')
            if inertial is None:
                inertial = ET.SubElement(link, 'inertial')

            origin = inertial.find('origin')
            if origin is None:
                origin = ET.SubElement(inertial, 'origin')
            origin.set('xyz', f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}")
            origin.set('rpy', '0 0 0')

            mass_elem = inertial.find('mass')
            if mass_elem is None:
                mass_elem = ET.SubElement(inertial, 'mass')
            if mass is not None:
                mass_elem.set('value', f"{mass:.6f}")

            inertia_elem = inertial.find('inertia')
            if inertia_elem is None:
                inertia_elem = ET.SubElement(inertial, 'inertia')

            inertia_elem.set('ixx', f"{inertia[0, 0]:.6f}")
            inertia_elem.set('ixy', f"{inertia[0, 1]:.6f}")
            inertia_elem.set('ixz', f"{inertia[0, 2]:.6f}")
            inertia_elem.set('iyy', f"{inertia[1, 1]:.6f}")
            inertia_elem.set('iyz', f"{inertia[1, 2]:.6f}")
            inertia_elem.set('izz', f"{inertia[2, 2]:.6f}")
            updated_here = True

        if updated_here:
            updated += 1

    if updated > 0:
        # write the updated urdf in place, keeping the .bak backup untouched
        tree.write(urdf_path, encoding='utf-8', xml_declaration=True)
        print(f'Updated {updated} link(s) inertial in {urdf_path} (backup written).')
    else:
        print('No matching links found to update in URDF.')


def main():
    parser = argparse.ArgumentParser(description='Recenter and align STL meshes to PCA axes')
    parser.add_argument('--meshes', type=Path, default=Path('robotURDF/meshes'), help='meshes folder (defaults to robotURDF/meshes)')
    parser.add_argument('--out', type=Path, default=Path('robotURDF/meshes/processed'), help='output folder for transformed meshes')
    parser.add_argument('--density', type=float, default=1.0, help='assumed density for inertia computation')
    parser.add_argument('--update-urdf', type=Path, default=None, help='optional URDF file to update inertial entries (not implemented automatically)')
    args = parser.parse_args()

    mesh_folder = args.meshes
    if not mesh_folder.exists():
        print('Meshes folder not found:', mesh_folder)
        sys.exit(1)

    process_all(mesh_folder, args.out, args.density, args.update_urdf)


if __name__ == '__main__':
    main()
