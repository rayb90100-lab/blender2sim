#!/usr/bin/env python3
"""环节 8：MuJoCo 原生轨迹 → Blender，输出 Real / Blender / MuJoCo 三路同帧对照。

论文 Sec 3.1："the native physics trajectory is reimported into Blender, enabling frame-matched
visual comparisons across Real RGB, Blender RGB, and MuJoCo RGB renderings."

做法：
  1. 复用 droid_stage2_physics.run（同模型、同关节真值、确定性）重跑，逐源帧记录可见几何体世界位姿
  2. 导出 MuJoCo 网格（Panda / Robotiq 官方 mesh，桌面与方块箱体）进 Blender，逐帧写 matrix_world 关键帧
     —— Blender 只「按轨迹拍照」，不做任何物理/动画插值以外的修改
  3. Blender 相机 = DROID 外参（欧拉 xyz, OpenCV→Blender 轴变换）+ fx=130（与 MuJoCo 同投影）
  4. Cycles 渲染，三路拼接视频 + 关键帧对照图

可见几何体：geom_group ≤ 2（Panda/Robotiq 的 visual 类为 group 2，collision 为 group 3 不渲染），
物体与桌面为 group 0。

用法:
  python tools/droid_stage8_blender.py --testdir <testdir> [--samples 32] [--scale 2]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import droid_common as C  # noqa: E402

FFMPEG = "/home/ubuntu/mambaforge/envs/E2E/bin/ffmpeg"
CAMS = ("exterior_1_left", "exterior_2_left", "wrist_left")
EXT_CAMS = ("exterior_1_left", "exterior_2_left")
WRIST_FX = 150.0
W, H = 320, 180


def log(m): print(m, flush=True)


# ---------------- 1. 重跑物理并记录 ----------------
def record_rollout(testdir):
    import mujoco
    import droid_stage2_physics as P

    poses, mj_frames, meta = {}, {c: {} for c in CAMS}, {}
    meta["wrist_pose"] = {}
    meta["qpos"] = {}

    def on_frame(m, d, src):
        if "geoms" not in meta:
            meta["geoms"] = [g for g in range(m.ngeom) if m.geom_group[g] <= 2]
            meta["model"] = m
            meta["renderers"] = {c: mujoco.Renderer(m, H * 2, W * 2) for c in CAMS}
        gs = meta["geoms"]
        poses[src] = (d.geom_xpos[gs].copy(), d.geom_xmat[gs].copy())
        meta["qpos"][src] = d.qpos.copy()
        wid = m.camera("wrist_left").id
        meta["wrist_pose"][src] = (d.cam_xpos[wid].copy(), d.cam_xmat[wid].copy())
        for c in CAMS:
            r = meta["renderers"][c]
            r.update_scene(d, camera=c)
            mj_frames[c][src] = r.render().copy()

    rep = P.run(testdir, render=False, tag=os.environ.get("DROID_ROLLOUT_TAG", "stage8_rollout"), on_frame=on_frame,
                qfile=os.environ.get("DROID_QFILE") or None)
    for r in meta["renderers"].values():
        r.close()
    record_rollout.qpos = meta["qpos"]
    return meta["model"], meta["geoms"], poses, mj_frames, rep, meta["wrist_pose"]


def geom_mesh(m, g):
    """返回 (verts, faces)，几何体局部系。"""
    import mujoco
    t = m.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        return m.mesh_vert[va:va + vn].copy(), m.mesh_face[fa:fa + fn].copy()
    s = m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * s
        f = np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])
        return v, f
    if t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE):
        r = s[0]
        h = s[1] if t != mujoco.mjtGeom.mjGEOM_SPHERE else s[0]
        n = 16
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
        v = np.vstack([np.c_[ring, np.full(n, -h)], np.c_[ring, np.full(n, h)]])
        f = []
        for i in range(n):
            j = (i + 1) % n
            f += [[i, j, n + j], [i, n + j, n + i]]
        for i in range(1, n - 1):
            f += [[0, i + 1, i], [n, n + i, n + i + 1]]
        return v, np.array(f)
    return None, None


def geom_rgba(m, g):
    mid = m.geom_matid[g]
    return (m.mat_rgba[mid] if mid >= 0 else m.geom_rgba[g]).copy()


# ---------------- 2. Blender 场景 ----------------
APPEARANCE_DEFAULT = {"table_material": "wood_medium", "light_tint": [1.0, 1.0, 1.0],
                      "light_azimuth_deg": 0.0, "background_rgb": [0.78, 0.80, 0.82]}
TABLE_LOOKS = {   # 只改外观；程序材质，不依赖外部贴图
    "wood_medium": dict(wood=True, ramp=((0.30, 0.13, 0.06), (0.52, 0.26, 0.13)), rough=0.45),
    "wood_dark": dict(wood=True, ramp=((0.10, 0.05, 0.03), (0.22, 0.11, 0.06)), rough=0.35),
    "white_laminate": dict(wood=False, rgba=(0.85, 0.85, 0.82, 1), rough=0.25),
}


def build_blender(m, geoms, ep, fx, scale, samples, appearance=None):
    ap = dict(APPEARANCE_DEFAULT, **(appearance or {}))
    import bpy
    from mathutils import Matrix

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x = W * scale
    sc.render.resolution_y = H * scale
    sc.view_settings.view_transform = "AgX"

    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = tuple(ap["background_rgb"]) + (1,)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
    sc.world = world

    def material(name, rgba, rough=0.5, wood=False, ramp=None):
        mt = bpy.data.materials.new(name)
        mt.use_nodes = True
        nt = mt.node_tree
        b = nt.nodes["Principled BSDF"]
        ramp_cols = ramp
        b.inputs["Roughness"].default_value = rough
        if wood:
            # 程序木纹：拉长的噪声 → 颜色渐变（只改外观，不影响物理）
            tc = nt.nodes.new("ShaderNodeTexCoord")
            mp = nt.nodes.new("ShaderNodeMapping")
            mp.inputs["Scale"].default_value = (2.0, 40.0, 2.0)
            nz = nt.nodes.new("ShaderNodeTexNoise")
            nz.inputs["Scale"].default_value = 3.0
            nz.inputs["Detail"].default_value = 8.0
            ramp = nt.nodes.new("ShaderNodeValToRGB")
            c0, c1 = ramp_cols if ramp_cols else ((0.30, 0.13, 0.06), (0.52, 0.26, 0.13))
            ramp.color_ramp.elements[0].color = tuple(c0) + (1,)
            ramp.color_ramp.elements[1].color = tuple(c1) + (1,)
            nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
            nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
            nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
            nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
        else:
            b.inputs["Base Color"].default_value = tuple(rgba[:3]) + (1.0,)
        return mt

    # 外观表（只影响 Blender 画面）：对照真实视频取色
    tl = TABLE_LOOKS[ap["table_material"]]
    LOOK = {"table_top": dict(wood=tl["wood"], ramp=tl.get("ramp"), rgba=tl.get("rgba", (0.5, 0.5, 0.5, 1)), rough=tl["rough"]),
            "green_block_geom": dict(rgba=(0.02, 0.32, 0.12, 1), rough=0.4),
            "block_stack_geom": dict(rgba=(0.80, 0.22, 0.03, 1), rough=0.4)}

    objs = []
    mats = {}
    for g in geoms:
        v, f = geom_mesh(m, g)
        if v is None:
            objs.append(None)
            continue
        name = m.geom(g).name or f"geom{g}"
        me = bpy.data.meshes.new(name)
        me.from_pydata(v.tolist(), [], f.tolist())
        me.update()
        ob = bpy.data.objects.new(name, me)
        sc.collection.objects.link(ob)
        if name in LOOK:
            lk = LOOK[name]
            key = name
            if key not in mats:
                mats[key] = material(key, lk.get("rgba", (0.5, 0.5, 0.5, 1)), lk["rough"], lk.get("wood", False),
                                     lk.get("ramp"))
        else:
            rgba = geom_rgba(m, g)
            key = tuple(np.round(rgba, 3))
            if key not in mats:
                mats[key] = material(f"mat{len(mats)}", rgba, 0.35)
        ob.data.materials.append(mats[key])
        objs.append(ob)

    # 灯光
    az = np.radians(ap["light_azimuth_deg"])
    for loc, en, sz in [((0.6, -0.6, 1.4), 400, 1.5), ((0.2, 0.8, 1.2), 200, 2.0)]:
        ld = bpy.data.lights.new("area", "AREA")
        ld.energy = en
        ld.size = sz
        ld.color = tuple(ap["light_tint"])
        lo = bpy.data.objects.new("area", ld)
        cx_, cy_ = 0.55, 0.0
        rx, ry = loc[0] - cx_, loc[1] - cy_
        lo.location = (cx_ + np.cos(az) * rx - np.sin(az) * ry, cy_ + np.sin(az) * rx + np.cos(az) * ry, loc[2])
        lo.rotation_euler = (0, 0, 0)
        sc.collection.objects.link(lo)
        tr = lo.constraints.new("TRACK_TO")
        tgt = bpy.data.objects.new("tgt", None)
        tgt.location = (0.55, 0.0, 0.0)
        sc.collection.objects.link(tgt)
        tr.target = tgt
        tr.track_axis = "TRACK_NEGATIVE_Z"
        tr.up_axis = "UP_Y"

    # 相机：OpenCV (x右 y下 z前) -> Blender (本地 -Z 前, +Y 上)
    cams = {}
    for c in CAMS:
        if c == "wrist_left":
            cd = bpy.data.cameras.new(c)
            cd.sensor_fit = "HORIZONTAL"
            cd.sensor_width = 36.0
            cd.lens = WRIST_FX / W * 36.0
            cd.clip_start = 0.005
            co = bpy.data.objects.new(c, cd)
            sc.collection.objects.link(co)
            cams[c] = co          # 位姿逐帧由 MuJoCo cam_xpos/cam_xmat 写入（与 Blender 相机轴约定相同）
            continue
        ext = ep[f"camera_extrinsics.{c}"][0]
        Rb = C.euler_to_R(ext[3:6], "xyz") @ np.diag([1, -1, -1])
        cd = bpy.data.cameras.new(c)
        cd.sensor_fit = "HORIZONTAL"
        cd.sensor_width = 36.0
        cd.lens = fx / W * 36.0
        cd.clip_start = 0.01
        co = bpy.data.objects.new(c, cd)
        sc.collection.objects.link(co)
        M = np.eye(4)
        M[:3, :3] = Rb
        M[:3, 3] = ext[:3]
        co.matrix_world = Matrix(M.tolist())
        cams[c] = co
    if os.environ.get("DROID_REFINED"):
        refine_scene(sc, m, geoms, objs, world)
    return sc, objs, cams


def refine_scene(sc, m, geoms, objs, world):
    """环节 8 外观精修：只改显示，不改几何位姿与物理轨迹（DEVELOPMENT_LOG §57）。"""
    import bpy
    import bmesh

    def principled(name, base, rough, coat=0.0, metal=0.0, noise=0.0, noise_scale=60.0):
        mt = bpy.data.materials.new(name)
        mt.use_nodes = True
        nt = mt.node_tree
        b = nt.nodes["Principled BSDF"]
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metal
        b.inputs["Coat Weight"].default_value = coat
        if noise > 0:   # 细微色彩起伏（漆面木料 / 混凝土），亮度 ±noise
            tc = nt.nodes.new("ShaderNodeTexCoord")
            nz = nt.nodes.new("ShaderNodeTexNoise")
            nz.inputs["Scale"].default_value = noise_scale
            nz.inputs["Detail"].default_value = 6.0
            ramp = nt.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].color = tuple(np.clip(np.array(base) * (1 - noise), 0, 1)) + (1,)
            ramp.color_ramp.elements[1].color = tuple(np.clip(np.array(base) * (1 + noise), 0, 1)) + (1,)
            nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
            nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
            nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
        else:
            b.inputs["Base Color"].default_value = tuple(base) + (1,)
        return mt

    def bevel(ob, width, seg=3):
        me = ob.data
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        md = ob.modifiers.new("bevel", "BEVEL")
        md.width = width
        md.segments = seg
        md.limit_method = "ANGLE"
        md.harden_normals = False
        for poly in me.polygons:
            poly.use_smooth = True

    robot_mats = {}
    for g, ob in zip(geoms, objs):
        if ob is None:
            continue
        name = ob.name
        bn = m.body(m.geom_bodyid[g]).name
        if bn.startswith("stack_") or bn == "green_block":
            # 积木：倒圆角 1.5 mm + 漆面木料
            base = tuple(geom_rgba(m, g)[:3])
            ob.data.materials.clear()
            ob.data.materials.append(principled(f"paint_{name}", base, 0.5, coat=0.25, noise=0.08, noise_scale=80))
            bevel(ob, 0.0015)
        elif bn.startswith(("link", "g_", "hand", "attachment")) or "link" in bn:
            rgba = geom_rgba(m, g)
            bright = float(np.mean(rgba[:3])) > 0.6
            key = "white" if bright else "dark"
            if key not in robot_mats:
                robot_mats[key] = (principled("robot_white", (0.92, 0.92, 0.90), 0.28, coat=0.4) if bright
                                   else principled("robot_dark", (0.12, 0.12, 0.13), 0.35, metal=0.5))
            ob.data.materials.clear()
            ob.data.materials.append(robot_mats[key])
            for poly in ob.data.polygons:
                poly.use_smooth = True

    # 桌子：按真实外形拟合（stage1_scene/table_display.json，SAM2 双视角 + 围板箱体 IoU），仅显示。
    # 物理里的支撑仍是固定大板（积木都在桌面中部，桌子外形不影响物理结果）。
    tb = m.body("table").pos
    ts = m.geom("table_top").size
    top_z = tb[2] + ts[2]
    floor_z = top_z - 0.74
    tdp = os.path.join(os.environ.get("DROID_TESTDIR", "."), "stage1_scene", "table_display.json")
    tbl_ob = next((o for o in objs if o is not None and o.name == "table_top"), None)
    if os.path.exists(tdp) and tbl_ob is not None:
        td = json.load(open(tdp))
        cxy, sxy, yaw = td["center_xy"], td["size_xy"], td["yaw_rad"]
        tbl_mat = tbl_ob.data.materials[0]
        tbl_ob.hide_render = True                      # 隐藏物理大板
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        tt = bpy.context.object
        tt.name = "table_display_top"
        tt.scale = (sxy[0], sxy[1], 0.025)
        tt.location = (cxy[0], cxy[1], top_z - 0.0125)
        tt.rotation_euler = (0, 0, yaw)
        tt.data.materials.append(tbl_mat)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bev = tt.modifiers.new("bevel", "BEVEL")
        bev.width, bev.segments = 0.006, 4
        # 围板（真实桌子前方可见的抽屉面板）
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        ap_ = bpy.context.object
        ap_.scale = (sxy[0] - 0.06, sxy[1] - 0.06, td.get("apron_m", 0.12) - 0.025)
        ap_.location = (cxy[0], cxy[1], top_z - 0.025 - ap_.scale[2] / 2)
        ap_.rotation_euler = (0, 0, yaw)
        ap_.data.materials.append(tbl_mat)
        legs_c, legs_s, legs_yaw = cxy, sxy, yaw
    else:
        legs_c, legs_s, legs_yaw = (tb[0], tb[1]), (2 * ts[0], 2 * ts[1]), 0.0
        bev = tbl_ob.modifiers.new("bevel", "BEVEL") if tbl_ob is not None else None
    leg_mat = principled("leg", (0.08, 0.08, 0.09), 0.4, metal=0.6)
    c_, s_ = np.cos(legs_yaw), np.sin(legs_yaw)
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = sx * (legs_s[0] / 2 - 0.05), sy * (legs_s[1] / 2 - 0.05)
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            lg = bpy.context.object
            h = top_z - 0.025 - floor_z
            lg.scale = (0.04, 0.04, h)
            lg.location = (legs_c[0] + c_ * lx - s_ * ly, legs_c[1] + s_ * lx + c_ * ly, floor_z + h / 2)
            lg.rotation_euler = (0, 0, legs_yaw)
            lg.data.materials.append(leg_mat)
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(tb[0], tb[1], floor_z))
    fl = bpy.context.object
    fl.data.materials.append(principled("floor", (0.62, 0.62, 0.60), 0.8, noise=0.12, noise_scale=6))

    # 柔光平行光：接触阴影
    sd = bpy.data.lights.new("sun", "SUN")
    sd.energy = 4.0
    sd.angle = np.radians(10)
    so = bpy.data.objects.new("sun", sd)
    so.rotation_euler = (np.radians(35), np.radians(-20), np.radians(30))
    sc.collection.objects.link(so)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--only", type=int, nargs="*", default=None, help="只渲染这些源帧（预检用）")
    a = ap.parse_args()
    import cv2
    import bpy  # noqa: F401  必须先于 mathutils 导入（bpy 注册 mathutils 模块）
    from mathutils import Matrix

    T = a.testdir
    os.environ["DROID_TESTDIR"] = T
    out = os.path.join(T, os.environ.get("DROID_STAGE8_OUT", "stage8_render"))
    os.makedirs(out, exist_ok=True)
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    fx = json.load(open(os.path.join(T, "stage0_replay", "intrinsics_adopted.json")))["fx"]

    log("1) 重跑原生物理并记录几何位姿 ...")
    m, geoms, poses, mj_frames, rep, wrist_pose = record_rollout(T)
    log(f"   task_success={rep['task_success']}  可见几何体 {len(geoms)} 个  帧 {len(poses)}")

    log("2) 建 Blender 场景 ...")
    appearance = json.load(open(os.environ["DROID_APPEARANCE"])) if os.environ.get("DROID_APPEARANCE") else None
    sc, objs, cams = build_blender(m, geoms, ep, fx, a.scale, a.samples, appearance)
    frames = sorted(poses)
    sc.frame_start, sc.frame_end = frames[0], frames[-1]
    for fr in frames:
        xp, xm = poses[fr]
        for k, ob in enumerate(objs):
            if ob is None:
                continue
            M = np.eye(4)
            M[:3, :3] = xm[k].reshape(3, 3)
            M[:3, 3] = xp[k]
            ob.matrix_world = Matrix(M.tolist())
            ob.keyframe_insert("location", frame=fr)
            ob.keyframe_insert("rotation_euler", frame=fr)
        wp, wm = wrist_pose[fr]
        M = np.eye(4)
        M[:3, :3] = wm.reshape(3, 3)
        M[:3, 3] = wp
        cams["wrist_left"].matrix_world = Matrix(M.tolist())
        cams["wrist_left"].keyframe_insert("location", frame=fr)
        cams["wrist_left"].keyframe_insert("rotation_euler", frame=fr)
    # 关键帧插值设为常数：每个源帧严格对应一次物理状态，不让 Blender 自己插值
    for ob in objs + [cams["wrist_left"]]:
        if ob is not None and ob.animation_data and ob.animation_data.action:
            for fc in ob.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"
    import bpy
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "scene_stage8.blend"))

    if a.only:
        frames = [f for f in frames if f in a.only]

    # ---- 曝光标定（同 test3 §26.3）：桌面区域亮度对齐真实画面 ----
    # 初版灯光 400W/200W 直接把桌面冲成粉白（与 test3 同样的坑）。
    # 取 ext1 首帧桌面中央块（真实与渲染里都是桌面）做单标量搜索。
    cap = cv2.VideoCapture(os.path.join(T, "input", "videos", "exterior_1_left.mp4"))
    cap.read()
    _, real0 = cap.read()
    cap.release()
    patch = (slice(int(0.55 * H), int(0.75 * H)), slice(int(0.30 * W), int(0.60 * W)))
    target = float(cv2.cvtColor(real0, cv2.COLOR_BGR2GRAY)[patch].mean())
    lights = [o for o in bpy.data.objects if o.type == "LIGHT"]
    base_e = [o.data.energy for o in lights]
    bg = sc.world.node_tree.nodes["Background"].inputs[1]
    base_bg = bg.default_value
    sc.camera = cams["exterior_1_left"]
    sc.frame_set(frames[0])
    keep = (sc.render.resolution_percentage, sc.cycles.samples)
    sc.render.resolution_percentage, sc.cycles.samples = 50, 8
    tmp = os.path.join(out, "_tmp.png")
    best = (1e9, 1.0)
    # 外观增强：固定曝光（否则自动标定会把深/浅桌面拉回真实亮度，抵消外观改动）
    fixed = os.environ.get("DROID_EXPOSURE")
    for mul in ((float(fixed),) if fixed else (1.0, 0.5, 0.25, 0.12, 0.06, 0.03)):
        for o, e in zip(lights, base_e):
            o.data.energy = e * mul
        bg.default_value = base_bg * mul ** 0.5
        sc.render.filepath = tmp
        bpy.ops.render.render(write_still=True)
        im = cv2.resize(cv2.imread(tmp, cv2.IMREAD_GRAYSCALE), (W, H))
        got = float(im[patch].mean())
        log(f"   曝光 ×{mul:<5} 桌面亮度 {got:6.1f}（目标 {target:.1f}）")
        if abs(got - target) < best[0]:
            best = (abs(got - target), mul)
    mul = best[1]
    for o, e in zip(lights, base_e):
        o.data.energy = e * mul
    bg.default_value = base_bg * mul ** 0.5
    sc.render.resolution_percentage, sc.cycles.samples = keep
    log(f"   采用曝光 ×{mul}")
    exposure_mul = mul
    log("3) Cycles 渲染 ...")
    bl_frames = {c: {} for c in CAMS}
    tmp = os.path.join(out, "_tmp.png")
    for c in CAMS:
        sc.camera = cams[c]
        for fr in frames:
            sc.frame_set(fr)
            sc.render.filepath = tmp
            bpy.ops.render.render(write_still=True)
            bl_frames[c][fr] = cv2.imread(tmp)
            if a.only:   # 关键帧模式：每个渲染器单独存图，供增强验收
                os.makedirs(os.path.join(out, "media"), exist_ok=True)
                cv2.imwrite(os.path.join(out, "media", f"blender_{c}_f{fr:03d}.png"), bl_frames[c][fr])
                cv2.imwrite(os.path.join(out, "media", f"mujoco_{c}_f{fr:03d}.png"),
                            cv2.cvtColor(mj_frames[c][fr], cv2.COLOR_RGB2BGR))
        log(f"   {c}: {len(frames)} 帧")
    if a.only:
        import hashlib
        cams_meta = {}
        for c in CAMS:
            cd = cams[c].data
            cams_meta[c] = {"lens_mm": cd.lens, "sensor_width_mm": cd.sensor_width,
                            "resolution": [sc.render.resolution_x, sc.render.resolution_y],
                            "mujoco_resolution": [W * 2, H * 2],
                            "convention": "Blender 相机 -Z 前/+Y 上；MuJoCo 同；OpenCV 外参经 diag(1,-1,-1)",
                            "frame_mapping": "媒体帧号 = 源视频帧号；物理第 i 记录 = 源帧 i（每源帧 33 物理步）"}
            if c != "wrist_left":
                cams_meta[c]["extrinsics_xyzrpy"] = ep[f"camera_extrinsics.{c}"][0].tolist()
        rm = {"frames": frames, "cameras": cams_meta,
              "qfile": os.environ.get("DROID_QFILE"), "appearance": appearance,
              "qpos_by_frame": {str(f): record_rollout.qpos[f].tolist() for f in frames}}
        json.dump(rm, open(os.path.join(out, "render_manifest.json"), "w"), indent=2, ensure_ascii=False)
    os.remove(tmp)

    log("4) 三路对照 ...")
    tw, th = 480, 270
    caps = {c: cv2.VideoCapture(os.path.join(T, "input", "videos", f"{c}.mp4")) for c in CAMS}
    real = {c: [] for c in CAMS}
    for c in CAMS:
        while True:
            ok, f = caps[c].read()
            if not ok:
                break
            real[c].append(f)
    vp = os.path.join(out, "compare_real_blender_mujoco.mp4")
    vw = None
    key = {84: "grasp", 110: "transport", 142: "release", 156: "end"}
    G = ep["observation.state.gripper_position"].ravel()
    for fr in frames:
        rows = []
        for c in CAMS:
            r_ = cv2.resize(real[c][fr], (tw, th), interpolation=cv2.INTER_CUBIC)
            b_ = cv2.resize(bl_frames[c][fr], (tw, th), interpolation=cv2.INTER_AREA)
            m_ = cv2.resize(cv2.cvtColor(mj_frames[c][fr], cv2.COLOR_RGB2BGR), (tw, th),
                            interpolation=cv2.INTER_AREA)
            sep = np.full((th, 4, 3), 40, np.uint8)
            row = np.hstack([r_, sep, b_, sep, m_])
            for x, t, col in [(6, "REAL", (0, 0, 255)), (tw + 10, "BLENDER", (255, 160, 0)),
                              (2 * tw + 14, "MUJOCO", (0, 160, 255))]:
                cv2.putText(row, t, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(row, f"{c}  src f{fr:03d}  grip={G[fr]:.2f}", (6, th - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            rows.append(row)
        frame = np.vstack(rows)
        if vw is None:
            vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 15, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
        if fr in key:
            cv2.imwrite(os.path.join(out, f"compare_f{fr:03d}_{key[fr]}.png"), frame)
    vw.release()
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", vp, "-c:v", "libx264", "-crf", "20",
                    "-pix_fmt", "yuv420p", vp.replace(".mp4", "_h264.mp4")], check=True)
    json.dump({"source": "native MuJoCo rollout（关节真值驱动，与 stage2 报告一致）",
               "frames": [frames[0], frames[-1]], "fps": 15, "cameras": list(CAMS), "fx": fx,
               "render": {"engine": "Cycles", "samples": a.samples, "resolution": [W * a.scale, H * a.scale]},
               "appearance": "程序木纹桌面 + 对照真实视频取色的方块；Panda/Robotiq 用官方网格与材质色；"
                             "背景为中性灰（真实环境未重建）",
               "interpolation": "CONSTANT：每源帧严格对应一次物理状态",
               "exposure_multiplier": exposure_mul,
               "task_success": rep["task_success"]},
              open(os.path.join(out, "stage8_manifest.json"), "w"), indent=2, ensure_ascii=False)
    log(f"   {vp.replace('.mp4', '_h264.mp4')}")


if __name__ == "__main__":
    main()
