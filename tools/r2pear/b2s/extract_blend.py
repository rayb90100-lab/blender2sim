"""b2s 第 0 步：从任意 .blend 提取原始事实（在 bpy 里运行，只提取，不做任何判断）。

所有「这是机器人 / 这是桌子 / 这是任务物体」的决策都在 parse_scene.py 里做，并写进证据。
这样换一个 blend 只需要重跑解析，不需要重新打开 Blender。

提取内容：
  scene     帧范围、fps、分辨率、单位、自定义属性
  objects   每个对象：类型、父子、所在集合、自定义属性、首帧世界矩阵、局部包围盒
  meshes    网格对象：求值后（含修改器）的局部顶点（超过上限时均匀抽样）、面数、体积、材质颜色
  dynamic   在 5 个探测帧里世界矩阵有变化的对象 → 全帧世界矩阵（覆盖动画、约束、驱动、骨骼父子）
  armatures 骨骼层级 + 每帧 matrix_basis（局部旋转）+ 每帧骨骼世界矩阵
  props_anim 每个 action 中的自定义属性曲线（如 aperture_mm）逐帧求值
  cameras   镜头、传感器、shift、分辨率
  lights    类型、能量、颜色

用法（bpy 作为 Python 模块）:
  python tools/r2pear/b2s/extract_blend.py --blend X.blend --out extract.json
注意：本文件名不能叫 inspect.py 等标准库名（会遮蔽 stdlib，bpy 导入时崩溃，实测）。
"""
import argparse
import json
import math
import sys

MAX_VERTS = 6000
PROBE_N = 5


def _jsonable(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    try:
        return [_jsonable(x) for x in v]
    except TypeError:
        return str(v)


def props(idb):
    out = {}
    for k in idb.keys():
        if k.startswith("_") or k in ("cycles", "cycles_visibility"):
            continue
        try:
            out[k] = _jsonable(idb[k].to_dict() if hasattr(idb[k], "to_dict") else idb[k])
        except Exception:
            out[k] = str(idb[k])
    return out


def mat(m):
    return [[round(float(x), 7) for x in row] for row in m]


def material_info(mt):
    info = {"name": mt.name, "base_color_linear": None, "textured": False, "image_mean_srgb": None,
            "roughness": None, "metallic": None}
    if not mt.use_nodes or not mt.node_tree:
        c = mt.diffuse_color
        info["base_color_linear"] = [float(c[0]), float(c[1]), float(c[2])]
        return info
    bsdf = next((n for n in mt.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        c = mt.diffuse_color
        info["base_color_linear"] = [float(c[0]), float(c[1]), float(c[2])]
        return info
    bc = bsdf.inputs["Base Color"]
    info["base_color_linear"] = [float(x) for x in bc.default_value[:3]]
    for k, key in (("Roughness", "roughness"), ("Metallic", "metallic")):
        if k in bsdf.inputs and not bsdf.inputs[k].is_linked:
            info[key] = float(bsdf.inputs[k].default_value)
    if bc.is_linked:
        info["textured"] = True
        # 沿链接向上找图片或色带，取平均色作为 MuJoCo 的单色近似
        stack, seen = [bc.links[0].from_node], set()
        while stack:
            n = stack.pop()
            if n.name in seen:
                continue
            seen.add(n.name)
            if n.type == "TEX_IMAGE" and n.image is not None and info["image_mean_srgb"] is None:
                try:
                    im = n.image
                    w, h = im.size
                    if w * h > 0:
                        px = im.pixels[:]
                        step = max(1, (w * h) // 4096) * 4
                        r = px[0::step]
                        g = px[1::step]
                        b = px[2::step]
                        info["image_mean_srgb"] = [sum(r) / len(r), sum(g) / len(g), sum(b) / len(b)]
                        info["image"] = im.filepath or im.name
                except Exception as e:  # 打包图片读不到时只记原因
                    info["image_error"] = str(e)
            if n.type == "VALTORGB":
                els = n.color_ramp.elements
                info["ramp_mean_linear"] = [sum(e.color[i] for e in els) / len(els) for i in range(3)]
            for inp in n.inputs:
                for lk in inp.links:
                    stack.append(lk.from_node)
    return info


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import bpy
    import bmesh
    bpy.ops.wm.open_mainfile(filepath=a.blend)
    sc = bpy.context.scene
    f0, f1 = sc.frame_start, sc.frame_end
    frames = list(range(f0, f1 + 1))
    out = {"extract_schema_version": 2, "blend": a.blend, "blender_version": bpy.app.version_string,
           "scene": {"name": sc.name, "frame_start": f0, "frame_end": f1,
                     "fps": sc.render.fps / sc.render.fps_base,
                     "resolution": [sc.render.resolution_x, sc.render.resolution_y],
                     "unit_system": sc.unit_settings.system, "scale_length": sc.unit_settings.scale_length,
                     "props": props(sc), "camera": sc.camera.name if sc.camera else None},
           "collections": {c.name: {"objects": [o.name for o in c.objects],
                                    "children": [ch.name for ch in c.children], "props": props(c)}
                           for c in bpy.data.collections},
           "objects": {}, "meshes": {}, "armatures": {}, "cameras": {}, "lights": {}, "props_anim": {},
           "dynamic": {}, "object_basis": {}}
    sc.frame_set(f0)
    dg = bpy.context.evaluated_depsgraph_get()
    for o in bpy.data.objects:
        if o.name not in sc.objects:
            continue
        out["objects"][o.name] = {
            "type": o.type, "parent": o.parent.name if o.parent else None, "parent_type": o.parent_type,
            "parent_bone": o.parent_bone or None, "collections": [c.name for c in o.users_collection],
            "props": props(o), "matrix_world": mat(o.matrix_world),
            "bbox_local": [list(v) for v in o.bound_box], "hide_render": bool(o.hide_render),
            "constraints": [c.type for c in o.constraints],
            "action": o.animation_data.action.name if o.animation_data and o.animation_data.action else None,
            "matrix_basis": mat(o.matrix_basis), "matrix_parent_inverse": mat(o.matrix_parent_inverse),
            "rotation_mode": o.rotation_mode,
            "data_props": props(o.data) if o.data is not None and hasattr(o.data, "keys") else {},
        }
        if o.type == "MESH":
            ev = o.evaluated_get(dg)
            me = ev.to_mesh()
            bm = bmesh.new()
            bm.from_mesh(me)
            try:
                vol = float(bm.calc_volume(signed=False))
            except Exception:
                vol = None
            bm.free()
            vs = [tuple(round(c, 6) for c in v.co) for v in me.vertices]
            nv = len(vs)
            if nv > MAX_VERTS:
                step = math.ceil(nv / MAX_VERTS)
                vs = vs[::step]
            out["meshes"][o.name] = {"n_verts": nv, "n_faces": len(me.polygons), "volume_local": vol,
                                     "verts_local": vs,
                                     "materials": [material_info(s.material) for s in o.material_slots if s.material]}
            ev.to_mesh_clear()
        elif o.type == "CAMERA":
            cd = o.data
            out["cameras"][o.name] = {"lens": cd.lens, "sensor_width": cd.sensor_width,
                                      "sensor_height": cd.sensor_height, "sensor_fit": cd.sensor_fit,
                                      "shift": [cd.shift_x, cd.shift_y], "clip": [cd.clip_start, cd.clip_end],
                                      "type": cd.type, "props": props(o)}
        elif o.type == "LIGHT":
            ld = o.data
            out["lights"][o.name] = {"type": ld.type, "energy": ld.energy, "color": list(ld.color),
                                     "size": getattr(ld, "size", None)}
        elif o.type == "ARMATURE":
            out["armatures"][o.name] = {"bones": {b.name: {"parent": b.parent.name if b.parent else None,
                                                           "head": list(b.head_local), "tail": list(b.tail_local),
                                                           "matrix_local": mat(b.matrix_local)}
                                                  for b in o.data.bones},
                                        "basis": {b.name: [] for b in o.pose.bones},
                                        "world": {b.name: [] for b in o.pose.bones},
                                        "rotation_mode": {b.name: b.rotation_mode for b in o.pose.bones}}

    # 动态对象探测：等距 PROBE_N 帧比较世界矩阵
    probe = sorted({frames[int(round(i * (len(frames) - 1) / (PROBE_N - 1)))] for i in range(PROBE_N)})
    ref = {}
    dyn = set()
    for f in probe:
        sc.frame_set(f)
        for n in out["objects"]:
            m = bpy.data.objects[n].matrix_world
            if n not in ref:
                ref[n] = [list(r) for r in m]
            elif max(abs(m[i][j] - ref[n][i][j]) for i in range(4) for j in range(4)) > 1e-6:
                dyn.add(n)
    out["dynamic_probe_frames"] = probe
    # 自定义属性动画：找每个 action 的使用者
    owners = {}
    for o in bpy.data.objects:
        if o.animation_data and o.animation_data.action:
            owners.setdefault(o.animation_data.action.name, []).append(o.name)
    for act in bpy.data.actions:
        for fc in act.fcurves:
            if fc.data_path.startswith('["'):
                key = fc.data_path[2:-2]
                for own in owners.get(act.name, []):
                    out["props_anim"].setdefault(own, {})[key] = [float(fc.evaluate(f)) for f in frames]
        out.setdefault("actions", {})[act.name] = {
            "users": owners.get(act.name, []),
            "channels": sorted({f"{fc.data_path}[{fc.array_index}]" for fc in act.fcurves}),
            "n_keys": max((len(fc.keyframe_points) for fc in act.fcurves), default=0),
            "props": props(act)}
    for n in dyn:
        out["dynamic"][n] = []
    # Local transforms retain joint rotation independently of upstream motion.
    # Include authored channels even when the five world probes miss a motion.
    for n, o in out["objects"].items():
        if o["action"] or o["props"].get("source_field") == "observation.joint_position":
            out["object_basis"][n] = []
    for f in frames:
        sc.frame_set(f)
        for n in out["object_basis"]:
            out["object_basis"][n].append(mat(bpy.data.objects[n].matrix_basis))
        for n in dyn:
            out["dynamic"][n].append(mat(bpy.data.objects[n].matrix_world))
        for an, ad in out["armatures"].items():
            ob = bpy.data.objects[an]
            for b in ob.pose.bones:
                ad["basis"][b.name].append(mat(b.matrix_basis))
                ad["world"][b.name].append(mat(ob.matrix_world @ b.matrix))   # 人手 retarget 用
    out["frames"] = frames
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print(f"extract: {len(out['objects'])} objects, {len(out['meshes'])} meshes, {len(dyn)} dynamic, "
          f"{len(out['armatures'])} armatures, {len(out['cameras'])} cameras, frames {f0}-{f1} -> {a.out}")


if __name__ == "__main__":
    main()
