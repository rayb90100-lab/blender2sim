#!/usr/bin/env python3
"""Web UI 的 Blender 视图：直接打开**原 blend**，用 MuJoCo 状态驱动（b2s 产出的环境用）。

和 tools/r2pear/blender_worker.py 协议相同（NDJSON over stdin/stdout），web_ui 按 spec 里有没有
blend_binding 选择用哪个 worker。区别：
  blender_worker  按 spec 用代码重搭一个简化场景（test5/6，重建场景里本来就没有 blend）
  blend_worker    原 blend 的全部资产/材质/灯光/背景保留；只改「会动的东西」：
                    物体     matrix_world ← MuJoCo body 位姿（换回 blend 世界系、补回原点偏移与缩放）
                    机械臂   骨骼 matrix_basis ← 关节角（q_bone = (q_mj − offset) / sign，按提取时的轴）
                    夹爪     驱动器读取的自定义属性 ← MuJoCo 2F-85 driver 关节角（运行时扫描驱动器找属性）
                  blend 里没有机械臂（UMI/Ego retarget）→ 隐藏原演示者，导入 MuJoCo 机器人网格

请求额外字段 "state": {"q": [7], "grip": driver_rad, "objects": {name: [16 行主序, 基座系]}}
渲染引擎缺省 EEVEE（R2P_BLEND_ENGINE=CYCLES 可换），灯光沿用 blend 原有设置。
"""
import argparse
import base64
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

GRIP_DRIVER_MAX = 0.8          # menagerie 2F-85 driver 行程（rad）
GRIP_APERTURE_MAX_MM = 85.0


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _axis_rot(ax, a):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_rotvec(np.asarray(ax, float) * a).as_matrix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--selftest", default=None, help="渲染一张 agentview 到该路径后退出")
    ap.add_argument("--export-glb", help="Offline export of the bound scene for trajectory playback")
    ap.add_argument("--state-json", help="Initial MuJoCo pose for offline export")
    ap.add_argument("--render-sequence", help="Offline JSONL pose requests, one PNG per recorded state")
    ap.add_argument("--frames-dir", help="Output directory for offline PNG frames")
    a = ap.parse_args()
    import spec as SP
    spec = SP.load(os.path.join(a.testdir, "stage0_spec", "task_spec.json"))
    B = spec["blend_binding"]
    T_wb = np.array(spec["robot"]["T_world_base"])

    # Reserve the protocol pipe before bpy imports or opens the scene: even
    # open_mainfile prints C-level startup messages that are not NDJSON.
    if not (a.selftest or a.export_glb or a.render_sequence):
        real = os.dup(1)
        dn = os.open(os.devnull, os.O_WRONLY)
        os.dup2(dn, 1)
        os.close(dn)
        out = os.fdopen(real, "w", buffering=1)

    import bpy
    from mathutils import Matrix, Vector
    bpy.ops.wm.open_mainfile(filepath=spec["source"]["blend"])
    sc = bpy.context.scene
    sc.frame_set(sc.frame_start)
    # Explicit diagnostic scene registration for static visual assets. Dynamic
    # objects receive their full physical poses below; never offset them twice.
    for name, delta in B.get('static_translations_world_m', {}).items():
        ob = bpy.data.objects[name]
        world = ob.matrix_world.copy()
        world.translation += Vector(delta)
        ob.matrix_world = world
    eng = os.environ.get("R2P_BLEND_ENGINE", "BLENDER_EEVEE_NEXT")
    sc.render.engine = eng
    if eng.startswith("BLENDER_EEVEE"):
        sc.eevee.taa_render_samples = int(os.environ.get("R2P_BLEND_SAMPLES", 16))
    else:
        sc.cycles.samples = int(os.environ.get("R2P_BLEND_SAMPLES", 16))
    sc.render.resolution_percentage = 100

    def bake_runtime_modifiers():
        """Bake display-only modifiers in this disposable runtime worker.

        The source .blend is never saved. Baking Bevel/Weighted Normal/Solidify/
        Subdivision once removes their repeated dependency-graph evaluation while
        preserving the evaluated high-resolution mesh for the interactive view.
        Deform and simulation modifiers are left live because they can depend on
        the incoming pose.
        """
        allowed = {"BEVEL", "WEIGHTED_NORMAL", "SOLIDIFY", "SUBSURF"}
        depsgraph = bpy.context.evaluated_depsgraph_get()
        baked = 0
        for ob in list(bpy.data.objects):
            if ob.type != "MESH" or not ob.modifiers:
                continue
            if any(m.type not in allowed for m in ob.modifiers):
                continue
            evaluated = ob.evaluated_get(depsgraph)
            try:
                mesh = bpy.data.meshes.new_from_object(
                    evaluated, depsgraph=depsgraph, preserve_all_data_layers=True)
            except TypeError:
                mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
            if mesh is None:
                continue
            old = ob.data
            ob.data = mesh
            ob.modifiers.clear()
            if old.users == 0:
                bpy.data.meshes.remove(old)
            baked += 1
        if baked:
            bpy.context.view_layer.update()
        print(f"[blend_worker] runtime modifier bake: {baked} meshes", file=sys.stderr)

    if os.environ.get("R2P_BLEND_RUNTIME_BAKE", "0") == "1":
        bake_runtime_modifiers()
    # ---- 物体：去掉 blend 里的近似动画，记下原点偏移 / 缩放 / 朝向差
    objs = {}
    for s, b in B["objects"].items():
        ob = bpy.data.objects[b["blend_name"]]
        if ob.animation_data:
            ob.animation_data.action = None
        M0 = np.array(ob.matrix_world)
        S = np.linalg.norm(M0[:3, :3], axis=0)
        R_bl0 = M0[:3, :3] / S
        # Explicit initial body orientation supports experimental full-pose meshes.
        R_base0 = np.array(b["initial_body_rotation_base"]) if "initial_body_rotation_base" in b else _rz(spec["objects"][s]["yaw_rad"])
        R_body0 = T_wb[:3, :3] @ R_base0
        companions = []
        for name in b.get("visual_companions", []):
            companion = bpy.data.objects[name]
            relative = ob.matrix_world.inverted() @ companion.matrix_world
            if companion.animation_data:
                companion.animation_data.action = None
            companions.append((companion, relative))
        objs[s] = {"ob": ob, "S": S, "Q": R_body0.T @ R_bl0, "cloc": np.array(b["cloc"]),
                   "companions": companions}

    # ---- 机械臂
    rb = B.get("robot")
    arm = None
    object_robot = None
    if rb:
        if rb.get("representation") == "object_joints":
            from object_motion import ObjectRobotBinding
            object_robot = ObjectRobotBinding(bpy, rb)
        else:
            arm = bpy.data.objects[rb["armature"]]
            if arm.animation_data:
                arm.animation_data.action = None
    # ---- 夹爪：找驱动器读取的自定义属性
    grip_props = []
    for o in bpy.data.objects:
        ad = o.animation_data
        if not (ad and ad.drivers):
            continue
        for fc in ad.drivers:
            for v in fc.driver.variables:
                for t in v.targets:
                    if t.id is not None and t.data_path.startswith('["'):
                        key = (t.id.name, t.data_path[2:-2])
                        if key not in grip_props:
                            grip_props.append(key)
    for on, _ in grip_props:
        ob = bpy.data.objects.get(on)
        if ob is not None and ob.animation_data:
            ob.animation_data.action = None
    print(f"[blend_worker] 物体 {len(objs)}，机械臂 {'有' if rb else '无'}，夹爪属性 {grip_props}", file=sys.stderr)

    # ---- 没有机械臂：隐藏原演示者，导入 MuJoCo 机器人网格
    robot_geoms = {}
    native_robot = rb is None or a.export_glb or B.get('robot_visual_source') == 'mujoco'
    if native_robot:
        hidden = list(B.get("hide", []))
        if arm is not None:
            hidden += [arm.name, *(ob.name for ob in arm.children_recursive)]
        if object_robot is not None:
            root = bpy.data.objects[rb["root"]]
            hidden += [root.name, *(ob.name for ob in root.children_recursive)]
        for n in hidden:
            ob = bpy.data.objects.get(n)
            if ob is not None:
                ob.hide_render = True
        import blender_live as BL
        geoms, _, _ = BL.collect(a.testdir, spec, 0.0)
        mat_w = bpy.data.materials.new("mj_robot_white")
        mat_w.diffuse_color = (0.9, 0.9, 0.88, 1)
        mat_d = bpy.data.materials.new("mj_robot_dark")
        mat_d.diffuse_color = (0.08, 0.08, 0.09, 1)
        for gi in geoms:
            if not (gi["body"].startswith(("link", "g_")) or gi["body"] == "attachment"):
                continue
            me = bpy.data.meshes.new(gi["name"])
            me.from_pydata(gi["verts"].tolist(), [], [[int(x) for x in f] for f in gi["faces"]])
            me.update()
            ob = bpy.data.objects.new(gi["name"], me)
            sc.collection.objects.link(ob)
            ob.data.materials.append(mat_w if float(np.mean(gi["rgba"][:3])) > 0.5 else mat_d)
            ob.matrix_world = Matrix((T_wb @ gi["matrix"]).tolist())
            robot_geoms[gi["name"]] = ob

    # ---- 相机
    cams = {}
    for cn, bn in B.get("cameras", {}).items():
        if bn in bpy.data.objects:
            cams[cn] = bpy.data.objects[bn]
        elif spec["cameras"][cn].get("virtual"):
            camera = spec["cameras"][cn]
            cd = bpy.data.cameras.new(bn)
            # Blender defaults to a 10 cm near plane, which cuts through the
            # gripper housing when this is a wrist camera. Keep nearby links.
            cd.clip_start = float(camera.get("near_clip_m", 0.001))
            cd.sensor_fit = "VERTICAL"
            cd.lens = cd.sensor_height / (2 * math.tan(math.radians(camera["fovy_deg"]) / 2))
            ob = bpy.data.objects.new(bn, cd)
            sc.collection.objects.link(ob)
            cams[cn] = ob
    cd = bpy.data.cameras.new("orbit")
    cd.sensor_fit = "VERTICAL"
    cd.angle_y = math.radians(45.0)
    orbit = bpy.data.objects.new("orbit", cd)
    sc.collection.objects.link(orbit)
    cams["orbit"] = orbit
    if native_robot and 'wrist' in cams:
        # A physical world pose must not be transformed again by the authored
        # robot's animated parent when Blender updates its dependency graph.
        wrist = cams['wrist']
        world = wrist.matrix_world.copy()
        wrist.parent = None
        wrist.matrix_world = world
        wrist.animation_data_clear()
        for constraint in list(wrist.constraints):
            wrist.constraints.remove(constraint)
    look = np.mean([np.array(o["ob"].matrix_world)[:3, 3] for o in objs.values()], axis=0)

    last_dynamic_key = None

    def apply_state(req):
        nonlocal last_dynamic_key
        st = req.get("state") or {}
        dynamic = {
            "state": st,
            "bodies": req.get("bodies") or {},
            "wrist_matrix": req.get("wrist_matrix"),
        }
        has_dynamic = bool(dynamic["state"] or dynamic["bodies"] or dynamic["wrist_matrix"])
        if not has_dynamic:
            return
        dynamic_key = json.dumps(dynamic, sort_keys=True, separators=(",", ":"))
        if dynamic_key == last_dynamic_key:
            return
        if object_robot is not None and not native_robot:
            object_robot.apply(st.get("q"), float(st["grip"])/GRIP_DRIVER_MAX if st.get("grip") is not None else None)
        for s, M16 in (st.get("objects") or {}).items():
            if s not in objs:
                continue
            o = objs[s]
            Tb = np.array(M16, float).reshape(4, 4)
            Tw = T_wb @ Tb
            R = Tw[:3, :3] @ o["Q"]
            M = np.eye(4)
            M[:3, :3] = R * o["S"]
            M[:3, 3] = Tw[:3, 3] - R @ o["cloc"]
            o["ob"].matrix_world = Matrix(M.tolist())
            for companion, relative in o["companions"]:
                companion.matrix_world = o["ob"].matrix_world @ relative
        if arm is not None and st.get("q") is not None:
            q = np.array(st["q"], float)
            for i, bn in enumerate(rb["bones"]):
                qb = (q[i] - rb["offset"][i]) / rb["sign"][i]
                Mb = np.eye(4)
                Mb[:3, :3] = _axis_rot(rb["axes"][i], qb)
                arm.pose.bones[bn].matrix_basis = Matrix(Mb.tolist())
        if st.get("grip") is not None:
            ang = float(st["grip"])
            for on, key in grip_props:
                ob = bpy.data.objects.get(on)
                if ob is None:
                    continue
                k = key.lower()
                if any(w in k for w in ("aperture", "width", "opening")):
                    ob[key] = GRIP_APERTURE_MAX_MM * (1 - min(max(ang / GRIP_DRIVER_MAX, 0), 1))
                else:
                    ob[key] = ang
                ob.update_tag()
        for name, M16 in (req.get("bodies") or {}).items():
            ob = robot_geoms.get(name)
            if ob is not None:
                ob.matrix_world = Matrix((T_wb @ np.array(M16).reshape(4, 4)).tolist())
        if native_robot and req.get("wrist_matrix") and "wrist" in cams:
            cams["wrist"].matrix_world = Matrix((T_wb @ np.array(req["wrist_matrix"]).reshape(4, 4)).tolist())
        bpy.context.view_layer.update()
        last_dynamic_key = dynamic_key

    def render(req, path):
        cam = str(req.get("cam", "orbit"))
        sc.render.resolution_x, sc.render.resolution_y = int(req.get("w", 512)), int(req.get("h", 288))
        if cam == "orbit":
            az, el = math.radians(float(req.get("az", 36))), math.radians(float(req.get("el", -30)))
            dist = float(req.get("dist", 0.6))
            p = look + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el)])
            orbit.location = tuple(p)
            orbit.rotation_euler = (Vector(tuple(look)) - Vector(tuple(p))).to_track_quat("-Z", "Y").to_euler()
        if cam not in cams:
            raise KeyError(f"未知相机 {cam}，可选 {list(cams)}")
        sc.camera = cams[cam]
        sc.render.image_settings.file_format = "JPEG"
        sc.render.image_settings.quality = int(req.get("quality", 80))
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)

    if a.render_sequence:
        from render_sequence import render_sequence
        render_sequence(a.render_sequence, a.frames_dir, cams, apply_state)
        return

    if a.export_glb:
        from export_display import export_glb
        with open(a.state_json) as fh:
            req = json.load(fh)
        apply_state(req)
        export_glb(a.export_glb, spec, objs, robot_geoms, req)
        return

    if a.selftest:
        req = json.loads(os.environ.get("R2P_SELFTEST_REQ", "{}"))
        apply_state(req)
        render({"cam": req.get("cam", "agentview"), "w": req.get("w", 640), "h": req.get("h", 360)}, a.selftest)
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            apply_state(req)
            tmp = f"/tmp/r2p_blend_{req.get('cam', 'orbit')}_{os.getpid()}.jpg"
            render(req, tmp)
            with open(tmp, "rb") as fh:
                out.write(json.dumps({"ok": True, "cam": req.get("cam"),
                                      "jpeg_b64": base64.b64encode(fh.read()).decode()}) + "\n")
        except Exception as e:
            import traceback
            out.write(json.dumps({"ok": False, "err": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}"})
                      + "\n")
        out.flush()


if __name__ == "__main__":
    main()
