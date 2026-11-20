"""Export authored surroundings with rigid roots driven by sampled MuJoCo poses."""
import json
from pathlib import Path

import numpy as np


def approximate_procedural_colors(materials):
    """glTF cannot evaluate Blender noise ramps; retain an explicit color approximation.

    Only the disposable export process is changed. Image-based materials and
    the original .blend / Cycles views are preserved.
    """
    changes = []
    for material in materials:
        if not material.use_nodes or not material.node_tree:
            continue
        tree = material.node_tree
        for shader in tree.nodes:
            if shader.type != 'BSDF_PRINCIPLED':
                continue
            base = shader.inputs['Base Color']
            if not base.is_linked:
                continue
            direct = base.links[0].from_node
            if direct.type != 'VALTORGB':
                continue
            pending, seen = [direct], set()
            while pending:
                node = pending.pop()
                if node in seen:
                    continue
                seen.add(node)
                pending.extend(link.from_node for socket in node.inputs for link in socket.links)
            kinds = {node.type for node in seen}
            if 'TEX_IMAGE' in kinds or not kinds.intersection({'TEX_NOISE', 'TEX_VORONOI', 'TEX_MUSGRAVE'}):
                continue
            color = np.mean([element.color[:] for element in direct.color_ramp.elements], axis=0)
            for link in list(base.links):
                tree.links.remove(link)
            base.default_value = color.tolist()
            changes.append({'material': material.name, 'base_color_linear': color.tolist(),
                            'method': 'mean of procedural color-ramp stops; texture variation omitted'})
    return changes


def export_glb(path, spec, objects, robot_geoms, pose):
    import bpy
    from mathutils import Matrix

    scene = bpy.context.scene
    base = np.asarray(spec["robot"]["T_world_base"])
    bindings = []

    def bind(name, meshes, matrix, source):
        node = bpy.data.objects.new(name, None)
        scene.collection.objects.link(node)
        node.matrix_world = Matrix(matrix.tolist())
        for ob in meshes:
            world = ob.matrix_world.copy()
            ob.parent = node
            ob.parent_type = "OBJECT"
            ob.matrix_parent_inverse = Matrix.Identity(4)
            ob.matrix_world = world
        bindings.append({"node": node.name, **source})

    # Both embodiments use the same rigid robot meshes in the portable viewer.
    # Interactive views can also opt into the same native geometry via the binding.
    for name, ob in robot_geoms.items():
        bind("b2s_" + name, [ob], base @ np.asarray(pose["bodies"][name]).reshape(4, 4),
             {"kind": "geom", "id": int(name[4:])})
    for name, row in objects.items():
        bind("b2s_object_" + name, [row["ob"], *(ob for ob, _ in row["companions"])],
             base @ np.asarray(pose["state"]["objects"][name]).reshape(4, 4),
             {"kind": "body", "name": name})
    bpy.context.view_layer.update()
    colors = approximate_procedural_colors(bpy.data.materials)
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB", use_renderable=True,
        export_animations=False, export_skins=False, export_apply=True,
        export_cameras=False, export_lights=False, export_yup=True,
    )
    Path(str(path) + ".bindings.json").write_text(json.dumps(bindings, indent=2))
    Path(str(path) + ".materials.json").write_text(json.dumps(colors, ensure_ascii=False, indent=2))
