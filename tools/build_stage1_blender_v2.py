#!/usr/bin/env python3
"""Stage 1 Blender 建 mesh —— Real2Gym test2 (真人取放耳机盒)

依据论文 Sec 3.1:
  "Each instance is represented as a complete mesh within a unified scene frame.
   While initial point clouds provide depth, orientation, and scale cues, occluded
   surfaces are completed using RGB silhouettes, visible structures..."

流程：
  1. 载入 MoGe 点云，转为 Blender 坐标（OpenCV -> Z-up）
  2. RANSAC 拟合桌面平面，建桌面 mesh
  3. 每个实例建完整实体 mesh（可见面来自点云，被遮挡面按轮廓补全并标注 inferred）
  4. 相机按 MoGe 内参配置
  5. 渲染并与真实首帧对照（重投影检查）
"""
import json
import math
import os

import bpy
import numpy as np
from mathutils import Vector

ROOT = "/home/ubuntu/workstation/RoboLab/Real2Gym/tests/test2_real_human_pickplace"
MG = f"{ROOT}/stage1_scene/moge3"
OUT = f"{ROOT}/stage1_scene/blender"
os.makedirs(OUT, exist_ok=True)

# ---------- 1) 读点云 ----------
g = np.load(f"{MG}/moge_out.npz")
P_cam, D, K_norm = g["points"], g["depth"], g["intrinsics"]
H, W = D.shape

fx = K_norm[0, 0] * W
fy = K_norm[1, 1] * H
cx = K_norm[0, 2] * W
cy = K_norm[1, 2] * H
print(f"intrinsics(pixel): fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")


def cam2blender(p):
    """OpenCV (x右,y下,z前) -> Blender (x右,y前,z上)"""
    p = np.atleast_2d(p)
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=1)


# ---------- 2) 桌面平面 ----------
table_pts = np.load(f"{MG}/table_points.npy")
# 只用画面下半部（确定是桌面，避开背景墙）
sel = table_pts[:, 1] > np.percentile(table_pts[:, 1], 45)
cand = table_pts[sel]

rng = np.random.default_rng(0)
best_n, best_inl = None, None
for _ in range(600):
    i = rng.choice(len(cand), 3, replace=False)
    a, b, c = cand[i]
    n = np.cross(b - a, c - a)
    if np.linalg.norm(n) < 1e-9:
        continue
    n /= np.linalg.norm(n)
    inl = np.abs((cand - a) @ n) < 0.008
    if best_inl is None or inl.sum() > best_inl.sum():
        best_n, best_inl = n, inl

n_cam = best_n
d_cam = float(np.mean(cand[best_inl] @ n_cam))
if n_cam[1] > 0:                    # 让法线朝上（-y 方向）
    n_cam, d_cam = -n_cam, -d_cam
inlier_ratio = float(best_inl.mean())
tilt_deg = math.degrees(math.acos(min(1.0, abs(n_cam[1]))))
print(f"桌面平面: n={np.round(n_cam,4)} d={d_cam:.4f} 内点率={inlier_ratio:.3f} 倾角={tilt_deg:.1f}deg")
if tilt_deg > 8.0:
    print(f"  !! 点云拟合出的桌面倾斜 {tilt_deg:.1f}deg，超过 8deg 阈值")
    print("  !! 按 skill 的 physical support 约束，改用水平桌面 + 点云估计高度")

# ---------- 3) 清空场景 ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def make_material(name, rgba, rough=0.6, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    return m


def add_box(name, center_bl, size_bl, mat, rot_euler=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center_bl)
    ob = bpy.context.object
    ob.name = name
    ob.scale = (size_bl[0], size_bl[1], size_bl[2])
    ob.rotation_euler = rot_euler
    ob.data.materials.append(mat)
    bpy.ops.object.shade_flat()
    return ob


# ---------- 桌面 mesh ----------
# skill 要求 enforce physical support relations。
# MoGe (refine_steps=0) 的点云有噪声，RANSAC 拟合出的法线倾角偏大，
# 因此这里按"水平桌面 + 点云估计高度"处理，并在 manifest 中记录该约束。
USE_LEVEL_TABLE = tilt_deg > 8.0

p0_cam = n_cam * d_cam
p0_bl = cam2blender(p0_cam)[0]
n_bl = cam2blender(n_cam)[0]
n_bl /= np.linalg.norm(n_bl)

tb_bl = cam2blender(cand[best_inl])

if USE_LEVEL_TABLE:
    # 水平桌面：法线强制 +Z，高度取桌面点云在 Blender Z 的中位数
    n_bl = np.array([0.0, 0.0, 1.0])
    z_level = float(np.median(tb_bl[:, 2]))
    p0_bl = np.array([0.0, 0.0, z_level])
    table_note = f"level_constraint_applied (fitted tilt {tilt_deg:.1f}deg > 8deg); z from table point median"
else:
    z_level = float(p0_bl[2])
    table_note = "fitted plane used"

print(f"桌面(Blender): 点={np.round(p0_bl,4)} 法线={np.round(n_bl,4)}")
print(f"  {table_note}")

# 桌面尺寸：按点云在水平面的展开范围
u = np.array([1.0, 0.0, 0.0])
v = np.array([0.0, 1.0, 0.0])
local = tb_bl @ np.stack([u, v], axis=1)
u_span = float(np.ptp(local[:, 0]))
v_span = float(np.ptp(local[:, 1]))
center_uv = (local.min(0) + local.max(0)) / 2
center_bl = p0_bl + center_uv[0] * u + center_uv[1] * v

table_mat = make_material("table_wood", (0.55, 0.40, 0.26, 1), rough=0.55)
# 桌面用立方体（有厚度，符合"完整有厚度结构"要求）
thick = 0.03
tbl_center = center_bl - n_bl * (thick / 2)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=tuple(tbl_center))
table_ob = bpy.context.object
table_ob.name = "table_top"
# 立方体本地 z 轴已对齐世界 +Z（水平桌面）
table_ob.scale = (u_span * 1.15, v_span * 1.15, thick)
table_ob.data.materials.append(table_mat)
bpy.ops.object.shade_flat()

# ---------- 4) 实例 mesh ----------
# 观测到的 z(深度)跨度偏小（背面不可见），按类别补全到合理厚度
INSTANCES = {
    "earbud_case": {
        "label": "earbud_case",
        "mat": ((0.045, 0.045, 0.05, 1), 0.35, 0.0),   # 黑色磨砂
        "depth_completion": 1.55,   # 观测 1.7cm -> 补到约 2.6cm（TWS 盒常见厚度）
        "note": "背面与底面 inferred，按 TWS 耳机盒典型厚度补全",
    },
    "tissue_box": {
        "label": "tissue_box",
        "mat": ((0.86, 0.84, 0.90, 1), 0.75, 0.0),     # 白紫包装
        "depth_completion": 1.30,   # 观测 7cm -> 补到约 9cm
        "note": "背面 inferred，按抽纸盒典型深度补全",
    },
}

manifest = {
    "source_frame": "input/frames/f000.png",
    "camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "W": W, "H": H,
               "note": "MoGe-3 估计内参，未标定"},
    "table_plane": {"normal_blender": n_bl.tolist(), "point_blender": p0_bl.tolist(),
                    "inlier_ratio": round(inlier_ratio, 4),
                    "fitted_tilt_deg": round(tilt_deg, 2),
                    "constraint": table_note},
    "instances": {},
}

for name, cfg in INSTANCES.items():
    pts_cam = np.load(f"{MG}/{name}_points.npy")
    pts_bl = cam2blender(pts_cam)

    lo = pts_bl.min(0)
    hi = pts_bl.max(0)
    center = (lo + hi) / 2
    size = hi - lo

    # 深度方向（Blender 的 Y = MoGe 的 z）补全
    depth_axis = 1   # Blender Y
    orig_depth = size[depth_axis]
    size[depth_axis] = orig_depth * cfg["depth_completion"]
    # 补全时保持靠相机一侧不动（Blender Y 较小侧是离相机近的一面）
    center[depth_axis] = lo[depth_axis] + size[depth_axis] / 2

    rgba, rough, metal = cfg["mat"]
    mat = make_material(name, rgba, rough, metal)

    # 物理支撑约束：底面贴合桌面高度（skill: enforce physical support relations）
    bottom = center[2] - size[2] / 2
    support_gap = bottom - z_level
    if abs(support_gap) > 0.004:
        center[2] = z_level + size[2] / 2
        print(f"  {name}: 底面与桌面差 {support_gap*1000:+.1f}mm -> 吸附到桌面")
    ob = add_box(name, tuple(center), tuple(size), mat)

    manifest["instances"][name] = {
        "center_blender": [round(float(x), 5) for x in center],
        "size_blender": [round(float(x), 5) for x in size],
        "observed_depth_m": round(float(orig_depth), 5),
        "completion_factor": cfg["depth_completion"],
        "points": int(len(pts_bl)),
        "status": "visible_surfaces_observed; occluded_faces=inferred",
        "note": cfg["note"],
    }
    print(f"{name:14s} center={np.round(center,4)} size={np.round(size,4)} "
          f"(观测深度 {orig_depth:.3f}m x {cfg['depth_completion']})")


# ---------- 5) 相机 ----------
cam_data = bpy.data.cameras.new("moGe_cam")
cam_data.sensor_fit = 'HORIZONTAL'
cam_data.sensor_width = 36.0
cam_data.lens = fx / W * 36.0          # 48.5mm
cam_data.clip_start = 0.01
cam_data.clip_end = 50.0
cam_ob = bpy.data.objects.new("moGe_cam", cam_data)
scene.collection.objects.link(cam_ob)
cam_ob.location = (0, 0, 0)
cam_ob.rotation_mode = 'XYZ'
cam_ob.rotation_euler = (math.radians(90), 0, 0)   # 相机朝 +Y，上方向 +Z
scene.camera = cam_ob
print(f"camera lens={cam_data.lens:.2f}mm  sensor=36mm  res={W}x{H}")

# ---------- 灯光 ----------
bpy.ops.object.light_add(type='AREA', location=(0.4, -0.6, 0.9))
key = bpy.context.object
key.data.energy = 60
key.data.size = 0.6
if n_bl[2] < 0:
    n_bl = -n_bl
bpy.ops.object.light_add(type='AREA', location=(-0.5, 0.3, 0.7))
fill = bpy.context.object
fill.data.energy = 20
fill.data.size = 0.8

# ---------- 渲染设置 ----------
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 96
scene.cycles.use_denoising = True
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = 'AgX'
scene.render.film_transparent = False
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.62, 0.66, 0.70, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
scene.world = world

# ---------- 渲染：相机视角 ----------
scene.render.filepath = f"{OUT}/render_f000.png"
bpy.ops.render.render(write_still=True)
print("rendered:", scene.render.filepath)

# ---------- 保存工程 ----------
bpy.ops.wm.save_as_mainfile(filepath=f"{OUT}/scene_stage1.blend")
print("saved:", f"{OUT}/scene_stage1.blend")

json.dump(manifest, open(f"{OUT}/instance_manifest.json", "w"),
          indent=2, ensure_ascii=False)
print("saved: instance_manifest.json")
