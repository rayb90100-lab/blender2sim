"""Export a full evaluated Blender mesh, centered in metric local coordinates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--blend', type=Path, required=True)
    ap.add_argument('--object', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    obj = bpy.data.objects[args.object]
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    world = np.array(obj.matrix_world)
    scale = np.linalg.norm(world[:3, :3], axis=0)
    vertices = np.array([v.co[:] for v in mesh.vertices])*scale
    faces = np.array([f.vertices[:] for f in mesh.loop_triangles], dtype=np.int32)
    center = (vertices.min(0)+vertices.max(0))/2
    vertices -= center
    pose = np.eye(4)
    pose[:3, :3] = world[:3, :3]/scale
    pose[:3, 3] = world[:3, 3] + pose[:3, :3] @ center
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out/'mesh.npz', vertices=vertices, faces=faces, world_pose=pose)
    metadata = {'source_blend': str(args.blend.resolve()), 'blend_sha256': hashlib.sha256(args.blend.read_bytes()).hexdigest(),
                'object': args.object, 'vertices': len(vertices), 'triangles': len(faces),
                'bounds_m': [vertices.min(0).tolist(), vertices.max(0).tolist()], 'world_pose': pose.tolist()}
    (args.out/'mesh-source.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(metadata, ensure_ascii=False))
    evaluated.to_mesh_clear()


if __name__ == '__main__':
    main()
