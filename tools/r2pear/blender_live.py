#!/usr/bin/env python3
"""test6 级 Blender 可交互渲染（EEVEE 实时预览）。

目标（用户需求）：把 test6 的仿真场景放进 Blender，**用鼠标转视角、实时看到材质与光照**。
不是离线精修渲染 —— 那是 droid_stage8_blender.py 的用途（Cycles，0.44 s/帧，
2.2 fps，只适合出对照视频）。这里要的是帧率够交互。

引擎选择（本机实测，640x360）：
  Workbench  44 ms/frame  22.6 fps   无材质
  EEVEE     120 ms/frame   8.3 fps   ← 用这个：有材质/光照/阴影
  Cycles    462 ms/frame   2.2 fps   光追，太慢

与 test4 的 droid_stage8_blender.py 的区别：
  - 全部从 task_spec.json + scene.json 读取（物体名/形状/尺寸/颜色/相机），无硬编码
  - 物理用 r2pear 的 sim.build（test4 用的是它自己的 droid_stage2_physics）
  - 引擎 EEVEE，面向交互而非出片
  - 相机从 spec 的 pos/quat_wxyz/fovy 直接转换为 Blender 相机

用法:
  # 静态图（验证通道）
  python tools/r2pear/blender_live.py --testdir <T> --still out.png
  # 交互（需在 Blender GUI 里打开生成的 .blend）
  python tools/r2pear/blender_live.py --testdir <T> --blend out.blend
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import predicates as PR  # noqa: E402
import sim as S  # noqa: E402
import spec as SP  # noqa: E402

SENSOR_W = 36.0          # Blender 默认传感器宽度（mm）
SAMPLE_STEPS = 1.0       # 记录轨迹的仿真时长（秒）


def srgb_to_linear(c):
    """sRGB 编码值 → 线性光值。**这一步不能省。**

    spec 里的 rgba 来自源视频像素（sRGB 编码），MuJoCo 也按 sRGB 解释它；
    而 Blender 的 Base Color 输入是**线性**的。直接把 sRGB 值塞进线性槽位，
    等于把画面整体提亮一档 gamma —— 表现就是"颜色发白、饱和度丢失"。

    实测：绿柱 sRGB (0.141, 0.416, 0.236) 的线性值是 (0.018, 0.144, 0.045)，
    暗通道差近 8 倍。test4 当年没踩到这个坑，是因为它用了手工调暗的常量
    （(0.02, 0.32, 0.12)），恰好把误差抵消了；spec-driven 管线不能靠手调。
    """
    a = np.asarray(c, float)
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    a = np.asarray(c, float)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(np.clip(a, 0, None), 1 / 2.4) - 0.055)


# ---------------------------------------------------------------- MuJoCo 侧
def collect(testdir, spec, seconds=0.0):
    """建模型，可选推进 seconds 秒，返回每个 geom 的世界位姿与尺寸。

    返回 list[dict]：name/shape/verts/faces/matrix(4x4)/rgba/body
    """
    import mujoco
    m, d = S.build(spec)
    S.reset(m, d, spec)
    if seconds > 0:
        for _ in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)

    out = []
    for g in range(m.ngeom):
        # 只导出可见 geom：group 0 = 物体/桌面，2 = 机器人 visual；
        # 3 是 collision（test6 的 cylinder/prism 碰撞体就在这里，必须跳过，
        # 三层过滤，缺一不可：
        #   1) group > 2 → 机器人 collision geom（不渲染）
        #   2) rgba 的 alpha == 0 → 本管线的"不可见碰撞体"约定（sim.py 把 test6 的
        #      cylinder_cap / prism_lo/hi 设成 rgba=[0,0,0,0]，它们和视觉 geom 重叠，
        #      不滤掉会把颜色盖成黑色 —— 实测绿柱渲染全黑就是这个原因）
        #   3) 无网格数据 → 跳过
        if m.geom_group[g] > 2:
            continue
        if float(geom_rgba(m, g)[3]) <= 0.01:
            continue
        v, f = geom_mesh(m, g)
        if v is None:
            continue
        M = np.eye(4)
        M[:3, :3] = d.geom_xmat[g].reshape(3, 3)
        M[:3, 3] = d.geom_xpos[g]
        # 名字：**机械臂的 63 个 geom 在 MuJoCo 里名字是空的**（全是 body 上的匿名 geom），
        # 只有桌面和物体有名字。早期用 `m.geom(g).name or f"geom{g}"` 生成 Blender 对象名，
        # 而 web_ui 的位姿同步是按 MuJoCo geom 名发 key、空名字直接跳过 ——
        # 结果机械臂的位姿永远同步不到，Blender 里机械臂停在初始姿态不动
        # （表现为「Blender 里没有机械臂 / 机械臂不动」，而 MuJoCo 里能动）。
        # 统一用 **geom id** 做稳定标识，不依赖名字是否为空。
        out.append({"gid": g,
                    "name": f"geom{g}",
                    "body": m.body(m.geom_bodyid[g]).name,
                    "verts": v, "faces": f, "matrix": M,
                    "rgba": geom_rgba(m, g)})
    return out, m, d


def geom_rgba(m, g):
    mid = m.geom_matid[g]
    return (m.mat_rgba[mid] if mid >= 0 else m.geom_rgba[g]).copy()


def geom_mesh(m, g):
    """geom 局部系顶点与面。与 droid_stage8_blender.geom_mesh 同源，但补了 capsule。"""
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
        f = np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
                      [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])
        return v, f
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return _cyl(s[0], s[1], caps=True)
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return _cyl(s[0], s[1] + s[0], caps=True)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return _cyl(s[0], 0.0, caps=True, sphere=True)
    if t == mujoco.mjtGeom.mjGEOM_PLANE:
        hs = s[:2]
        v = np.array([[-hs[0], -hs[1], 0], [hs[0], -hs[1], 0], [hs[0], hs[1], 0], [-hs[0], hs[1], 0]])
        return v, np.array([[0, 1, 2, 3]])
    return None, None


def _cyl(r, half_h, caps=True, n=24, sphere=False):
    """圆柱/球：返回 (verts, faces)。

    注意：face 列表里侧面是四边形（4 点）、端盖是 n 边形（n 点），
    **每个面的点数不同**，不能直接 np.array(f)（会报 inhomogeneous shape）。
    统一按 list[list[int]] 返回，由调用方处理。
    """
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(ang) * r, np.sin(ang) * r], 1)
    if sphere:
        v = np.vstack([np.array([[0, 0, r]]), np.c_[ring, np.zeros(n)], np.array([[0, 0, -r]])])
        f = []
        for i in range(n):
            j = (i + 1) % n
            f.append([0, 1 + j, 1 + i])
            f.append([n + 1, 1 + i, 1 + j])
        return v, f
    top = np.c_[ring, np.full(n, half_h)]
    bot = np.c_[ring, np.full(n, -half_h)]
    v = np.vstack([bot, top])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f.append([i, j, n + j, n + i])
    if caps:
        f.append(list(range(n - 1, -1, -1)))            # 底盖
        f.append(list(range(n, 2 * n)))                  # 顶盖
    return v, f


# ---------------------------------------------------------------- Blender 侧
def build_blender(geoms, spec, engine="BLENDER_EEVEE_NEXT", res_scale=2, refine=True):
    """按 test4 的 §57 外观精修手法搭场景（用户要求"要 test4 那种效果"）。

    与早期简化版的差别（那些正是效果差的原因）：
      - 木纹桌面用程序噪声（不是纯色平面）
      - 桌面按 table 的真实外形 + 倒角，而不是 MuJoCo 的物理大板
      - 四条桌腿 + 围板 + 地面（早期只有一块板悬在空中）
      - 柔和平行光（Sun 4W, angle 10°）+ 环境光，而不是两盏过曝面光
      - 积木倒角 + 漆面（coat），机械臂分亮/暗两套材质
      - AgX 色调映射（test4 用的就是 AgX；早期改成 Standard 是为了贴近 MuJoCo
        的纯色，但那样就丢掉了"精修感"——这里以观感为准，回到 AgX，
        并在世界背景上补偿亮度）
    """
    import bpy
    from mathutils import Matrix

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = engine
    if engine == "CYCLES":
        sc.cycles.device = "CPU"
        sc.cycles.samples = 32
    sc.render.resolution_x = spec["render"]["width"] * res_scale
    sc.render.resolution_y = spec["render"]["height"] * res_scale
    # AgX：与 test4 精修一致。早期用 Standard 是为了对齐 MuJoCo 的纯色，
    # 但那会让画面失去层次（用户反馈"出不来 test4 那种效果"）。
    sc.view_settings.view_transform = "AgX" if refine else "Standard"
    sc.view_settings.look = "None"

    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.78, 0.80, 0.82, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.2
    sc.world = w

    def principled(name, base, rough, coat=0.0, metal=0.0, noise=0.0, noise_scale=60.0):
        mt = bpy.data.materials.new(name)
        mt.use_nodes = True
        nt = mt.node_tree
        b = nt.nodes["Principled BSDF"]
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metal
        b.inputs["Coat Weight"].default_value = coat
        if noise > 0:
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

    def wood(name, rough=0.45, ramp=((0.105, 0.046, 0.021), (0.182, 0.091, 0.046))):
        mt = bpy.data.materials.new(name)
        mt.use_nodes = True
        nt = mt.node_tree
        b = nt.nodes["Principled BSDF"]
        b.inputs["Roughness"].default_value = rough
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (2.0, 40.0, 2.0)      # 拉长 → 木纹
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 3.0
        nz.inputs["Detail"].default_value = 8.0
        rmp = nt.nodes.new("ShaderNodeValToRGB")
        rmp.color_ramp.elements[0].color = tuple(ramp[0]) + (1,)
        rmp.color_ramp.elements[1].color = tuple(ramp[1]) + (1,)
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
        nt.links.new(nz.outputs["Fac"], rmp.inputs["Fac"])
        nt.links.new(rmp.outputs["Color"], b.inputs["Base Color"])
        return mt

    def bevel(ob, width, seg=3):
        import bmesh
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
        for poly in me.polygons:
            poly.use_smooth = True

    # ---- 建 geom ----
    objs = []
    robot_mats = {}
    for gi in geoms:
        name = gi["name"]
        body = gi["body"]
        is_robot = body.startswith(("link", "g_", "hand", "attachment")) or "link" in body
        is_obj = body in spec["objects"]
        is_table = body == "table"

        me = bpy.data.meshes.new(name)
        me.from_pydata(gi["verts"].tolist(), [], [[int(x) for x in f] for f in gi["faces"]])
        me.update()
        ob = bpy.data.objects.new(name, me)
        sc.collection.objects.link(ob)
        ob.matrix_world = Matrix(gi["matrix"].tolist())

        rgba = gi["rgba"]
        base = tuple(float(x) for x in rgba[:3])
        if refine and is_obj:
            # 两步还原：
            #  1) scene.json 的 rgba 是「实测像素 / 曝光增益 0.78」的反照率补偿值
            #     （merge_scene.measure_rgba 的做法，对 MuJoCo 的纯色渲染合适），
            #     在 Blender 里要乘回 0.78 才是源视频的实测 sRGB 像素。
            #  2) 再转线性：Blender 的 Base Color 是线性空间。
            base = tuple(srgb_to_linear(np.clip(np.array(base) * 0.78, 0, 1)))

        if is_robot:
            bright = float(np.mean(base)) > 0.6
            key = "white" if bright else "dark"
            if key not in robot_mats:
                robot_mats[key] = (principled("robot_white", (0.92, 0.92, 0.90), 0.28, coat=0.4)
                                   if bright else
                                   principled("robot_dark", (0.12, 0.12, 0.13), 0.35, metal=0.5))
            ob.data.materials.append(robot_mats[key])
            for poly in me.polygons:
                poly.use_smooth = True
        elif is_table and refine:
            # 物理大板隐藏，稍后按 table 的真实外形重建（§57 做法）
            ob.hide_render = True
            ob.data.materials.append(wood("wood_tmp"))
        elif is_obj:
            ob.data.materials.append(principled(f"paint_{name}", base, 0.5,
                                                coat=0.25, noise=0.08, noise_scale=80))
            if refine:
                bevel(ob, 0.0015)
        else:
            ob.data.materials.append(principled(f"m_{name}", base, 0.45))
        objs.append(ob)

    if refine:
        _add_table_and_floor(sc, spec, geoms, objs)

    # ---- 相机 ----
    cams = {}
    for cname, cam in spec["cameras"].items():
        cd = bpy.data.cameras.new(cname)
        cd.sensor_fit = "VERTICAL"
        cd.sensor_height = SENSOR_W
        cd.angle_y = math.radians(cam["fovy_deg"])
        cd.clip_start = 0.005
        co = bpy.data.objects.new(cname, cd)
        sc.collection.objects.link(co)
        if cam["mount"] == "world":
            # 必须写 matrix_world：分别设 location / rotation_quaternion 后，
            # matrix_world 要等依赖图更新才会反映 —— 而我们随后就要用它，
            # 读到的是单位阵（实测相机停在原点、朝向 +Y，画面里看不到机械臂）。
            from mathutils import Matrix, Quaternion
            q = Quaternion((cam["quat_wxyz"][0], cam["quat_wxyz"][1],
                            cam["quat_wxyz"][2], cam["quat_wxyz"][3]))
            Mw = Matrix.Translation(tuple(cam["pos"])) @ q.to_matrix().to_4x4()
            co.matrix_world = Mw
        cams[cname] = co
    return sc, objs, cams


def _add_table_and_floor(sc, spec, geoms, objs):
    """按 spec.table 的真实外形重建可见桌面 + 桌腿 + 围板 + 地面 + 灯光（§57 手法）。

    灯光必须搬全：test4 精修场景是 **两个 AREA 面光（400 W / 200 W，追着桌面打）
    + 一盏 Sun(4 W, angle 10°)**。早期这里只放了 Sun，结果画面全靠环境光平铺，
    没有方向性、没有接触阴影、颜色被 AgX 推白 —— 这就是"出不来 test4 效果"的主因之一。
    """
    import bpy
    tb = spec["table"]
    top_z = tb["top_z"]
    floor_z = top_z - tb.get("height_m", 0.75)
    cxy = tb["center_xy"]
    sxy = tb["size_xy"]
    yaw = tb.get("yaw_rad", 0.0)
    apron = tb.get("apron_m", 0.12)

    def wood_material(name, ramp=((0.105, 0.046, 0.021), (0.182, 0.091, 0.046)), rough=0.45):
        """程序木纹：**拉长的**噪声（Mapping scale=(2,40,2)）→ 颜色渐变。

        注意不能用各向同性噪声（noise_scale=12 那种）：那是色块斑点，不是木纹。
        test4 的 table_look 用的就是这个拉伸比例。
        """
        mt = bpy.data.materials.new(name)
        mt.use_nodes = True
        nt = mt.node_tree
        b = nt.nodes["Principled BSDF"]
        b.inputs["Roughness"].default_value = rough
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (2.0, 40.0, 2.0)
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 3.0
        nz.inputs["Detail"].default_value = 8.0
        rmp = nt.nodes.new("ShaderNodeValToRGB")
        rmp.color_ramp.elements[0].color = tuple(ramp[0]) + (1,)
        rmp.color_ramp.elements[1].color = tuple(ramp[1]) + (1,)
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
        nt.links.new(nz.outputs["Fac"], rmp.inputs["Fac"])
        nt.links.new(rmp.outputs["Color"], b.inputs["Base Color"])
        return mt

    def principled_local(name, base, rough, coat=0.0, metal=0.0, noise=0.0, noise_scale=60.0):
        mt = bpy.data.materials.new(name)
        mt.use_nodes = True
        b = mt.node_tree.nodes["Principled BSDF"]
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metal
        b.inputs["Coat Weight"].default_value = coat
        if noise > 0:
            tc = mt.node_tree.nodes.new("ShaderNodeTexCoord")
            nz = mt.node_tree.nodes.new("ShaderNodeTexNoise")
            nz.inputs["Scale"].default_value = noise_scale
            ramp = mt.node_tree.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].color = tuple(np.clip(np.array(base) * (1 - noise), 0, 1)) + (1,)
            ramp.color_ramp.elements[1].color = tuple(np.clip(np.array(base) * (1 + noise), 0, 1)) + (1,)
            mt.node_tree.links.new(tc.outputs["Object"], nz.inputs["Vector"])
            mt.node_tree.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
            mt.node_tree.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
        else:
            b.inputs["Base Color"].default_value = tuple(base) + (1,)
        return mt

    wood_mat = wood_material("table_wood")
    leg_mat = principled_local("leg", (0.08, 0.08, 0.09), 0.4, metal=0.6)

    # 桌面
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    tt = bpy.context.object
    tt.name = "table_display_top"
    tt.scale = (sxy[0], sxy[1], 0.025)
    tt.location = (cxy[0], cxy[1], top_z - 0.0125)
    tt.rotation_euler = (0, 0, yaw)
    tt.data.materials.append(wood_mat)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bev = tt.modifiers.new("bevel", "BEVEL")
    bev.width, bev.segments = 0.006, 4

    # 围板
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    ap_ = bpy.context.object
    ap_.scale = (sxy[0] - 0.06, sxy[1] - 0.06, apron - 0.025)
    ap_.location = (cxy[0], cxy[1], top_z - 0.025 - ap_.scale[2] / 2)
    ap_.rotation_euler = (0, 0, yaw)
    ap_.data.materials.append(wood_mat)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # 四条腿
    c_, s_ = np.cos(yaw), np.sin(yaw)
    h = top_z - 0.025 - floor_z
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = sx * (sxy[0] / 2 - 0.05), sy * (sxy[1] / 2 - 0.05)
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            lg = bpy.context.object
            lg.scale = (0.04, 0.04, h)
            lg.location = (cxy[0] + c_ * lx - s_ * ly, cxy[1] + s_ * lx + c_ * ly, floor_z + h / 2)
            lg.rotation_euler = (0, 0, yaw)
            lg.data.materials.append(leg_mat)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # 地面
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(cxy[0], cxy[1], floor_z))
    fl = bpy.context.object
    fl.data.materials.append(principled_local("floor", (0.62, 0.62, 0.60), 0.8,
                                              noise=0.12, noise_scale=6))

    # ---- 灯光 ----
    # 主光：两个 AREA 面光，TRACK_TO 指向桌面中心。
    # 数值不是照抄 test4 的 400/200 W —— 那是 Cycles 的值，直接搬到 EEVEE 会过曝
    # （实测整帧白到只剩轮廓）。这里按「渲染出的物体/桌面颜色 vs 源视频实测色」扫描得到：
    #   20 W / 10 W + Sun 1.0 + 世界光 0.2 → 物体色误差 0.195、桌面色误差 0.187
    # 扫描网格见 DEVELOPMENT_LOG §64。
    look_at = (cxy[0], cxy[1], top_z)
    for loc, en, sz in [((0.6, -0.6, 1.4), 20, 1.5), ((0.2, 0.8, 1.2), 10, 2.0)]:
        ld = bpy.data.lights.new("area", "AREA")
        ld.energy = en
        ld.size = sz
        lo = bpy.data.objects.new("area", ld)
        # 位置按桌面中心平移（test4 硬编码在 (0.55, 0.0)，这里跟 spec 走）
        lo.location = (look_at[0] + loc[0] - 0.55, look_at[1] + loc[1], look_at[2] + loc[2])
        sc.collection.objects.link(lo)
        tgt = bpy.data.objects.new("light_target", None)
        tgt.location = look_at
        sc.collection.objects.link(tgt)
        tr = lo.constraints.new("TRACK_TO")
        tr.target = tgt
        tr.track_axis = "TRACK_NEGATIVE_Z"
        tr.up_axis = "UP_Y"

    # 补光：柔和平行光，给底部和桌面接触处一点方向性阴影
    sd = bpy.data.lights.new("sun", "SUN")
    sd.energy = 1.0
    sd.angle = np.radians(10)
    so = bpy.data.objects.new("sun", sd)
    so.rotation_euler = (np.radians(35), np.radians(-20), np.radians(30))
    sc.collection.objects.link(so)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--engine", default="BLENDER_EEVEE_NEXT",
                    choices=["BLENDER_EEVEE_NEXT", "CYCLES", "BLENDER_WORKBENCH"])
    ap.add_argument("--seconds", type=float, default=0.0, help="先推进仿真多少秒再导出")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--still", default=None, help="渲染一张静态图到该路径")
    ap.add_argument("--blend", default=None, help="保存 .blend（可在 Blender GUI 里交互）")
    ap.add_argument("--camera", default=None, help="渲染用哪个相机（默认第一个）")
    a = ap.parse_args()

    T = a.testdir
    spec_path = os.path.join(T, "stage0_spec", "task_spec.json")
    spec = SP.load(spec_path)
    geoms, m, d = collect(T, spec, a.seconds)
    print(f"导出 {len(geoms)} 个可见 geom（跳过 collision group 3）")
    by_body = {}
    for g in geoms:
        by_body.setdefault(g["body"], 0)
        by_body[g["body"]] += 1
    for k, v in by_body.items():
        print(f"  {k:20s} {v} geom")

    sc, objs, cams = build_blender(geoms, spec, a.engine, a.scale)
    print(f"引擎 {sc.render.engine}  {sc.render.resolution_x}x{sc.render.resolution_y}")
    print(f"相机 {list(cams)}")

    if a.blend:
        sc.camera = cams[a.camera or list(cams)[0]]
        bpy_save(a.blend)
        print(f"已保存 {a.blend}（用 Blender GUI 打开即可交互）")
    if a.still:
        import bpy
        sc.camera = cams[a.camera or list(cams)[0]]
        sc.render.filepath = os.path.abspath(a.still)
        import time
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"渲染 {a.still}  用时 {time.time()-t:.2f}s")


def bpy_save(path):
    import bpy
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path))


if __name__ == "__main__":
    main()
