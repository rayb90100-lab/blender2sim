#!/usr/bin/env python3
"""Stage 1: 真实视频 -> Blender 场景（重力对齐版）

修正 v2 的核心错误：cam2blender 直接用 (x,z,-y) 把相机系当世界系，
但相机是俯拍（俯角 21°），导致"上"方向错误、桌面倾斜、物体悬空。

本版按论文 Sec 3.1 "enforcing physical support relations"：
  1. RANSAC 拟合桌面平面（相机系），其法线即真实重力"上"方向
  2. 构造重力对齐的世界坐标系（Z=重力上）
  3. 所有点投影到该世界系
  4. 桌面 z=0，物体按点云水平 PCA 估 yaw，底面贴合桌面
  5. 相机沿 +Y 放置，up=+Z
"""
import json, math, os
import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = "/home/ubuntu/workstation/RoboLab/Real2Gym/tests/test2_real_human_pickplace"
MG = f"{ROOT}/stage1_scene/moge3"
OUT = f"{ROOT}/stage1_scene/blender"
os.makedirs(OUT, exist_ok=True)

g = np.load(f"{MG}/moge_out.npz")
P, D, K = g["points"], g["depth"], g["intrinsics"]
H, W = D.shape
fx, fy = K[0,0]*W, K[1,1]*H
cx, cy = K[0,2]*W, K[1,2]*H
print(f"intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

# ---- 1) 桌面平面 ----
tbl = np.load(f"{MG}/table_points.npy")
cand = tbl[tbl[:,1] > np.percentile(tbl[:,1], 45)]
rng = np.random.default_rng(0)
best = None
for _ in range(800):
    i = rng.choice(len(cand), 3, replace=False)
    a,b,c = cand[i]
    n = np.cross(b-a, c-a)
    if np.linalg.norm(n) < 1e-9: continue
    n = n/np.linalg.norm(n)
    inl = np.abs((cand-a) @ n) < 0.008
    if best is None or inl.sum() > best[1].sum(): best = (n, inl)
n_cam, inl = best
d_cam = float(np.mean(cand[inl] @ n_cam))
# 让法线朝"上"（相机系 y 向下，所以朝上 = n_y < 0）
if n_cam[1] > 0: n_cam, d_cam = -n_cam, -d_cam
tilt = math.degrees(math.acos(min(1.0, abs(n_cam[1]))))
print(f"桌面平面: n={np.round(n_cam,4)} d={d_cam:.5f} 内点率={inl.mean():.3f}")

# ---- 2) 重力对齐世界基 ----
up_w  = n_cam / np.linalg.norm(n_cam)          # 世界 +Z（重力上）
fwd_c = np.array([0.0, 0.0, 1.0])               # 相机视线方向
fwd_w = fwd_c - (fwd_c @ up_w) * up_w
fwd_w /= np.linalg.norm(fwd_w)                  # 世界 +Y（水平前）
right_w = np.cross(fwd_w, up_w)                 # 世界 +X
right_w /= np.linalg.norm(right_w)
print(f"world +Z(up) = {np.round(up_w,4)}")
print(f"world +Y(fwd)= {np.round(fwd_w,4)}")
print(f"world +X     = {np.round(right_w,4)}")

def cam2world(p):
    p = np.atleast_2d(np.asarray(p, dtype=float))
    return np.stack([p @ right_w, p @ fwd_w, p @ up_w], axis=1)

# 验证：桌面点应接近同一 z
tz = cam2world(cand[inl])[:,2]
print(f">>> 桌面点世界 z: median={np.median(tz):.4f} IQR={np.percentile(tz,75)-np.percentile(tz,25):.4f}")
z_table = float(np.median(tz))

# ---- 3) 场景 ----
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

def mat_make(name, rgba, rough=0.6, metal=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bs = m.node_tree.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = rgba
    bs.inputs["Roughness"].default_value = rough
    bs.inputs["Metallic"].default_value = metal
    return m

# 桌面：水平 slab，顶面在 z_table
tb_w = cam2world(cand[inl])
u_span = float(np.ptp(tb_w[:,0])) * 1.05
v_span = float(np.ptp(tb_w[:,1])) * 1.05
ctr_x = float((tb_w[:,0].min()+tb_w[:,0].max())/2)
ctr_y = float((tb_w[:,1].min()+tb_w[:,1].max())/2)
thick = 0.03
bpy.ops.mesh.primitive_cube_add(size=1.0,
    location=(ctr_x, ctr_y, z_table - thick/2))
t = bpy.context.object; t.name = "table_top"
t.scale = (u_span, v_span, thick)
t.data.materials.append(mat_make("wood", (0.55,0.40,0.26,1), 0.55))
bpy.ops.object.shade_flat()
print(f"桌面: span={u_span:.3f}x{v_span:.3f} at z={z_table:.4f}")

# ---- 4) 实例 ----
INST = {
  "earbud_case": dict(rgba=(0.045,0.045,0.05,1), rough=0.35, grow=1.15,
                      note="被遮挡面 inferred；按 TWS 盒典型厚度补全"),
  "tissue_box":  dict(rgba=(0.86,0.84,0.90,1), rough=0.75, grow=1.10,
                      note="背面 inferred；按抽纸盒典型深度补全"),
}
manifest = {
  "source_frame": "input/frames/f000.png",
  "camera": dict(fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H,
                 note="MoGe-3 估计，未标定"),
  "world_basis": {"right": right_w.tolist(), "fwd": fwd_w.tolist(), "up": up_w.tolist()},
  "table": {"z": round(z_table,5), "span": [round(u_span,4), round(v_span,4)],
            "plane_normal_cam": n_cam.tolist(),
            "note": "gravity-aligned world; Z=table normal"},
  "instances": {},
}

for name, cfg in INST.items():
    pts_cam = np.load(f"{MG}/{name}_points.npy")
    pts_w = cam2world(pts_cam)

    # 水平 PCA 估 yaw
    xy = pts_w[:,:2] - pts_w[:,:2].mean(0)
    w_, V_ = np.linalg.eigh(xy.T @ xy / len(xy))
    yaw_axis = V_[:, np.argmax(w_)]
    yaw = math.atan2(yaw_axis[1], yaw_axis[0])

    # 转到 yaw 局部系求尺寸
    ca, sa = math.cos(-yaw), math.sin(-yaw)
    Rz = np.array([[ca,-sa],[sa,ca]])
    local_xy = xy @ Rz.T
    ext_x = float(np.ptp(local_xy[:,0])) * cfg["grow"]
    ext_y = float(np.ptp(local_xy[:,1])) * cfg["grow"]
    z_lo, z_hi = float(pts_w[:,2].min()), float(pts_w[:,2].max())
    ext_z = (z_hi - z_lo) * cfg["grow"]

    ctr_local = np.array([(local_xy[:,0].min()+local_xy[:,0].max())/2,
                          (local_xy[:,1].min()+local_xy[:,1].max())/2])
    ctr_world_xy = Rz.T @ ctr_local + pts_w[:,:2].mean(0)
    # 底面贴合桌面（物理支撑）
    z_center = z_lo + ext_z/2
    gap = z_lo - z_table
    if abs(gap) > 0.003:
        z_center = z_table + ext_z/2
        print(f"  {name}: 底面距桌 {gap*1000:+.1f}mm -> 贴合桌面")

    bpy.ops.mesh.primitive_cube_add(size=1.0,
        location=(float(ctr_world_xy[0]), float(ctr_world_xy[1]), z_center))
    ob = bpy.context.object; ob.name = name
    ob.scale = (ext_x, ext_y, ext_z)
    ob.rotation_euler = (0, 0, yaw)
    ob.data.materials.append(mat_make(name, cfg["rgba"], cfg["rough"]))
    bpy.ops.object.shade_flat()

    manifest["instances"][name] = {
      "center_world": [round(float(ctr_world_xy[0]),5), round(float(ctr_world_xy[1]),5), round(z_center,5)],
      "size_world": [round(ext_x,5), round(ext_y,5), round(ext_z,5)],
      "yaw_deg": round(math.degrees(yaw),2),
      "points": int(len(pts_w)),
      "status": "visible=observed; occluded=inferred",
      "note": cfg["note"],
    }
    print(f"{name:14s} ctr={np.round(ctr_world_xy,3)} z={z_center:.3f} "
          f"size=({ext_x:.3f},{ext_y:.3f},{ext_z:.3f}) yaw={math.degrees(yaw):.1f}deg")

# ---- 5) 相机 ----
cd = bpy.data.cameras.new("cam_moge")
cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36.0
cd.lens = fx/W*36.0; cd.clip_start = 0.005; cd.clip_end = 100.0
cob = bpy.data.objects.new("cam_moge", cd)
scene.collection.objects.link(cob)
cob.location = (0,0,0)
# 相机本地：X->world X, Y->world Z, Z->-world Y
Rm = Matrix(((1,0,0), (0,0,-1), (0,1,0)))   # 列 = 本地轴在世界中的方向
cob.rotation_mode = 'QUATERNION'
cob.rotation_quaternion = Rm.to_quaternion()
scene.camera = cob
print(f"camera lens={cd.lens:.2f}mm")

# ---- 灯光 ----
def add_light(loc, energy, size):
    bpy.ops.object.light_add(type='AREA', location=loc)
    o = bpy.context.object; o.data.energy = energy; o.data.size = size
    o.rotation_euler = (0,0,0)
    return o
add_light((0.6, 0.2, 1.2), 120, 1.0)
add_light((-0.6, -0.3, 0.9), 45, 1.2)

# ---- 渲染 ----
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 96
scene.cycles.use_denoising = True
scene.render.resolution_x = W; scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = 'AgX'
world = bpy.data.worlds.new("w"); world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.60,0.64,0.68,1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
scene.world = world

scene.render.filepath = f"{OUT}/render_f000.png"
bpy.ops.render.render(write_still=True)

# 透明背景版本，供 mask 比对
scene.render.film_transparent = True
scene.render.filepath = f"{OUT}/render_f000_alpha.png"
bpy.ops.render.render(write_still=True)
scene.render.film_transparent = False

bpy.ops.wm.save_as_mainfile(filepath=f"{OUT}/scene_stage1.blend")
json.dump(manifest, open(f"{OUT}/instance_manifest.json","w"), indent=2, ensure_ascii=False)
print("saved:", OUT)
