"""GLB -> MuJoCo/Blender 资产提取（在 Blender 内运行）。

调用方：tools/r2pear/glb_scene.py

设计依据（都是实测踩出来的，见 DEVELOPMENT_LOG 第十二轮）：
  1. **MuJoCo 的 OBJ 加载器只读第一个 object 组**。Blender 导出的 room.obj 有 65 组，
     MuJoCo 只吃第一组（地毯），131734 面 -> 586 面，且不报错。所以每个 MuJoCo mesh
     必须对应一个**只含单个 object 组**的 OBJ 文件。
  2. **Blender 的 OBJ 导出用全局顶点索引**，按 "o " 行切文件会让索引失效
     （报 vertex index does not exist）。拆完必须重新编号。
  3. **零厚度平面无法生成凸包**。GLB 里墙/玻璃/地板是单面片，只有 6 个，qhull 会
     报 "mesh volume is too small"。给它们补 1e-3 m 厚度。
  4. **inertia 是 <mesh> 的属性，不是 <geom> 的**。写错位置会 schema 报错。
     纯视觉 geom（contype=0 + 不指定 mass）不会被算体积/凸包，这是避免编译爆炸的关键：
     对照实测 22788 顶点的部件，contype=0 -> hull=False，contype 有效 -> hull=True。
  5. glTF 的 mesh 顶点**已经是世界坐标**（节点平移在 Empty 父级上）。刚体导出前要把
     原点搬到自身包围盒中心，并把这个中心记下来，MuJoCo 的 body 就放在那里。

产物：
  <outdir>/parts/*.obj          场景视觉，一片材一个文件（单 object 组）
  <outdir>/bodies/<tag>.obj     动态刚体，原点在几何中心
  <outdir>/textures/*.jpg       内嵌贴图
  <outdir>/glb_export.json      部件表 + 刚体表（中心/尺寸/材质/贴图）
"""
import bpy
import json
import os
import sys
import mathutils
import numpy as np


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    a = {}
    for i in range(0, len(argv), 2):
        a[argv[i].lstrip("-")] = argv[i + 1]
    return a


ARGS = parse_args()
GLB = ARGS["glb"]
OUT = ARGS["out"]
BODIES = json.loads(ARGS.get("bodies", "[]"))
STATIC = json.loads(ARGS.get("static", "{}"))
PARTS_DIR = os.path.join(OUT, "parts")
BODIES_DIR = os.path.join(OUT, "bodies")
TEX_DIR = os.path.join(OUT, "textures")
for d in (PARTS_DIR, BODIES_DIR, TEX_DIR):
    os.makedirs(d, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)

report = {"glb": GLB, "parts": [], "bodies": {}, "textures": [], "warnings": []}

# ---------------------------------------------------------------- 贴图
# glTF 内嵌图在 Blender 里是 packed、filepath 为空。
# **has_data 是惰性属性**：不先访问 img.size，它会是 False，导致整批贴图被跳过。
# 实测：直接查 has_data -> 全部 False、保存 0 张；先读 img.size -> 20/20 成功。
tex_map = {}
for img in bpy.data.images:
    if img.type != 'IMAGE':
        continue
    _ = img.size                      # 强制惰性加载，别删
    if img.size[0] == 0:
        continue
    name = (img.name or "tex").replace("/", "_").replace(" ", "_")
    if not name.lower().endswith((".png", ".jpg", ".jpeg")):
        name += ".png"
    path = os.path.join(TEX_DIR, name)
    try:
        img.filepath_raw = path
        img.file_format = 'PNG'
        img.save()
        tex_map[img.name] = name
        report["textures"].append({"name": img.name, "file": name,
                                   "size": list(img.size)})
    except Exception as e:
        report["warnings"].append(f"texture {img.name}: {e}")


# ---------------------------------------------------------------- 展平层级
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
bpy.ops.object.select_all(action='DESELECT')
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def material_info(o):
    """取对象的首个材质的基色与贴图文件名（sRGB，MuJoCo 也按 sRGB 解释）。"""
    if not o.material_slots:
        return None, [0.6, 0.6, 0.6, 1.0]
    m = o.material_slots[0].material
    if m is None or not m.use_nodes:
        return None, [0.6, 0.6, 0.6, 1.0]
    bsdf = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    rgba = [0.6, 0.6, 0.6, 1.0]
    tex = None
    if bsdf:
        bc = bsdf.inputs.get("Base Color")
        if bc.is_linked:
            node = bc.links[0].from_node
            img = getattr(node, "image", None)
            if img is not None:
                tex = tex_map.get(img.name)
        else:
            rgba = [float(v) for v in bc.default_value]
    return (m.name, tex), rgba


def bounds(o):
    cs = [o.matrix_world @ mathutils.Vector(v) for v in o.bound_box]
    lo = [min(c[i] for c in cs) for i in range(3)]
    hi = [max(c[i] for c in cs) for i in range(3)]
    return lo, hi


def tri_count(o):
    ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    me.calc_loop_triangles()
    n = len(me.loop_triangles)
    ev.to_mesh_clear()
    return n


def export_obj(o, path):
    bpy.ops.object.select_all(action='DESELECT')
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True,
                          forward_axis='Y', up_axis='Z',
                          export_materials=False, export_uv=True,
                          export_normals=True)


# ---------------------------------------------------------------- 动态刚体
# 原点搬到包围盒中心，OBJ 顶点即相对 body 原点；body 放在该中心。
body_objs = {}
for tag in BODIES:
    o = bpy.data.objects.get(tag)
    if o is None:
        report["warnings"].append(f"body not found: {tag}")
        continue
    lo, hi = bounds(o)
    center = [(lo[i] + hi[i]) / 2 for i in range(3)]
    dims = [hi[i] - lo[i] for i in range(3)]
    o.data.transform(mathutils.Matrix.Translation(-mathutils.Vector(center)))
    p = os.path.join(BODIES_DIR, f"{tag.replace(' ', '_').replace('/', '_')}.obj")
    export_obj(o, p)
    mat, rgba = material_info(o)
    body_objs[tag] = o
    report["bodies"][tag] = {
        "file": os.path.basename(p),
        "center_world": [round(x, 5) for x in center],
        "dims": [round(x, 5) for x in dims],
        "tris": tri_count(o),
        "obj_mb": round(os.path.getsize(p) / 1e6, 3),
        "material": mat[0] if mat else None,
        "texture": mat[1] if mat else None,
        "rgba": [round(v, 4) for v in rgba],
        "static": False,
    }
    # 从场景视觉里移除（避免重复渲染）
    bpy.data.objects.remove(o, do_unlink=True)

# ---------------------------------------------------------------- 静态碰撞体
# --static "tag=节点名" 形式：场景里保留视觉，另外记录一条 box 碰撞规格。
static_specs = {}
for tag, node in STATIC.items():
    o = bpy.data.objects.get(node)
    if o is None:
        report["warnings"].append(f"static not found: {node}")
        continue
    lo, hi = bounds(o)
    static_specs[tag] = {
        "node": node,
        "min": [round(x, 5) for x in lo],
        "max": [round(x, 5) for x in hi],
        "center": [round((lo[i] + hi[i]) / 2, 5) for i in range(3)],
        "half": [round((hi[i] - lo[i]) / 2, 5) for i in range(3)],
    }
report["static"] = static_specs

# ---------------------------------------------------------------- 场景视觉
# 按材质切分：同一材质的几何合并成一个 MuJoCo mesh，正好一片材一个文件。
remaining = [o for o in bpy.data.objects if o.type == 'MESH']
bpy.ops.object.select_all(action='DESELECT')
for o in remaining:
    o.select_set(True)
bpy.context.view_layer.objects.active = remaining[0]
if len(remaining) > 1:
    bpy.ops.object.join()
joined = bpy.context.view_layer.objects.active
joined.name = "ROOM_ALL"

mat_list = []
for slot in joined.material_slots:
    if slot.material is not None and slot.material not in mat_list:
        mat_list.append(slot.material)

part_index = 0
for m in mat_list:
    sel_idx = {i for i, s in enumerate(joined.material_slots) if s.material == m}
    bpy.ops.object.select_all(action='DESELECT')
    joined.select_set(True)
    bpy.context.view_layer.objects.active = joined
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    for poly in joined.data.polygons:
        poly.select = poly.material_index in sel_idx
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.separate(type='SELECTED')
    bpy.ops.object.mode_set(mode='OBJECT')
    new = [o for o in bpy.context.selected_objects if o is not joined]
    if not new:
        continue
    new = new[0]
    if len(new.data.polygons) == 0:
        bpy.data.objects.remove(new, do_unlink=True)
        continue
    lo, hi = bounds(new)
    _, rgba = material_info(new)
    bsdf = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None) if m.use_nodes else None
    tex = None
    if bsdf and bsdf.inputs["Base Color"].is_linked:
        img = getattr(bsdf.inputs["Base Color"].links[0].from_node, "image", None)
        if img is not None:
            tex = tex_map.get(img.name)
    fn = f"{part_index:03d}.obj"
    export_obj(new, os.path.join(PARTS_DIR, fn))
    report["parts"].append({
        "file": fn, "material": m.name, "texture": tex,
        "rgba": [round(v, 4) for v in rgba],
        "min": [round(x, 5) for x in lo], "max": [round(x, 5) for x in hi],
        "tris": len(new.data.polygons),
    })
    part_index += 1
    bpy.data.objects.remove(new, do_unlink=True)

report["n_parts"] = len(report["parts"])
report["bounds"] = {
    "min": [round(min(p["min"][i] for p in report["parts"]), 4) for i in range(3)],
    "max": [round(max(p["max"][i] for p in report["parts"]), 4) for i in range(3)],
}
report["bounds"]["extent"] = [round(report["bounds"]["max"][i] - report["bounds"]["min"][i], 4)
                              for i in range(3)]

json.dump(report, open(os.path.join(OUT, "glb_export.json"), "w"),
          ensure_ascii=False, indent=1)
print("GLB_EXPORT_OK", json.dumps({"parts": report["n_parts"],
                                   "bodies": len(report["bodies"]),
                                   "textures": len(report["textures"])},
                                  ensure_ascii=False))
