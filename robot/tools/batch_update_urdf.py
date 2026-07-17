#!/usr/bin/env python3
"""
Batch process all STLs: align (PCA), compute inertia, build collision hulls,
and patch URDF with computed inertial values. Non-interactive.
"""
from pathlib import Path
import json
import importlib.util
import sys


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    root = Path(__file__).resolve().parents[1]
    meshes = root / 'robotURDF' / 'meshes'
    out = root / 'robotURDF' / 'meshes' / 'processed'
    urdf = root / 'robotURDF' / 'urdf' / 'Assy_Full_Aligned_URDF.urdf'

    tool = load_module(root / 'tools' / 'recenter_align_meshes.py', 'recenter')

    aligned_folder = out / 'aligned'
    collision_folder = out / 'collision'
    aligned_folder.mkdir(parents=True, exist_ok=True)
    collision_folder.mkdir(parents=True, exist_ok=True)

    stls = list(meshes.rglob('*.stl'))
    results = {}
    for stl in stls:
        print('Processing', stl.name)
        mesh = tool.trimesh.load(stl, force='mesh')
        centroid, vals, vecs = tool.pca_axes(mesh.vertices)
        aligned_mesh, R = tool.align_mesh(mesh, centroid, vecs)
        aligned_path = aligned_folder / (stl.stem + '_aligned.stl')
        aligned_mesh.export(aligned_path)
        com, inertia, mass = tool.compute_inertia(aligned_mesh, density=1.0)
        col = tool.make_collision_mesh(aligned_mesh)
        col_path = collision_folder / (stl.stem + '_collision.stl')
        col.export(col_path)

        results[stl.name] = {
            'aligned_file': str(aligned_path.relative_to(root)),
            'center_of_mass': [float(x) for x in (com.tolist() if com is not None else [0,0,0])],
            'inertia_matrix': inertia.tolist(),
            'mass': float(mass) if mass is not None else None,
        }

    results_path = out / 'inertia_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # patch URDF
    tool.update_urdf_file(urdf, results, meshes, aligned_folder, collision_folder)
    print('Done. Wrote', results_path)


if __name__ == '__main__':
    main()
