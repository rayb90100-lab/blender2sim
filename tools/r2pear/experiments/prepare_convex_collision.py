"""Create convex collision assets from a closed source mesh with explicit estimated density."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--threshold-m', type=float, default=.001)
    ap.add_argument('--density', type=float, default=600, help='Estimated kg/m^3, not measured mass')
    args = ap.parse_args()
    import coacd
    import trimesh
    z = np.load(args.assets/'mesh.npz')
    mesh = trimesh.Trimesh(z['vertices'], z['faces'], process=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError('Source must be a closed, consistently oriented mesh before mass/decomposition comparisons')
    mesh.export(args.assets/'visual.obj')
    mass = mesh.volume*args.density
    # Both collision methods share mass properties of the same source mesh.
    inertia = mesh.moment_inertia*args.density
    settings = dict(threshold=args.threshold_m, real_metric=True, preprocess_mode='off', resolution=2000,
                    mcts_nodes=20, mcts_iterations=60, mcts_max_depth=3, merge=True, seed=0)
    coacd.set_log_level('info')
    start = time.perf_counter()
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), **settings)
    elapsed = time.perf_counter()-start
    rows = []
    for i,(v,f) in enumerate(parts):
        part = trimesh.Trimesh(v,f,process=False)
        name = f'convex-{i:03d}.obj'; part.export(args.assets/name)
        rows.append({'file': name, 'vertices': len(v), 'faces': len(f), 'volume_m3': float(abs(part.volume))})
    report = {'source_sha256': hashlib.sha256((args.assets/'mesh.npz').read_bytes()).hexdigest(),
              'coacd_version': importlib.metadata.version('coacd'), 'trimesh_version': trimesh.__version__,
              'settings': settings, 'preprocess_seconds': elapsed, 'parts': rows,
              'mass_kg': mass, 'center_of_mass_m': mesh.center_mass.tolist(), 'inertia_kg_m2': inertia.tolist(),
              'source_volume_m3': float(mesh.volume), 'watertight': bool(mesh.is_watertight),
              'density_kg_m3': args.density, 'mass_grade': 'estimated_from_source_volume_and_assumed_density',
              'source_height_m': float(np.ptp(mesh.vertices[:,2])),
              'geometry_note': 'Full source visual mesh; CoACD threshold is an algorithm setting, not a certified error bound.'}
    (args.assets/'collision-assets.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
