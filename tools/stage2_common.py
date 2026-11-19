#!/usr/bin/env python3
"""Stage 2 公共几何：把 Stage 1 的相机系几何转到 MuJoCo 世界系。

Stage 1 的 manifest 全部在 **OpenCV 相机坐标系**（x右, y下, z前）里，
这正是 Blender 场景的坐标系（Blender = (x, z, -y)，相机在原点）。

MuJoCo 需要 **Z 向上**、桌面上表面在 z=0 的重力系。所以必须重新定义世界系：

    Z_w = 支撑面法线 n_cam       （相机系下的"上"）
    X_w = cam_x 投影到桌面        （≈ (1,0,0)，因为 n 的 x 分量≈0）
    Y_w = Z_w × X_w
    原点 = 相机在桌面上的垂足

于是相机在世界系里位于 (0, 0, -d)，高度 = -d = 0.193 m。

注意：不要把这个变换理解为"把桌子掰平"。桌子本来就是平的，是**相机**俯冲了
21–24°。在相机系里做水平化会歪掉整个场景（DEVELOPMENT_LOG §22 的教训）；
在世界系里把桌子放平、把相机的倾角如实保留，才是物理上正确的。
"""
import json
import os

import numpy as np


def plane_frame(n_cam):
    """由支撑面法线构造世界系基向量（世界系 → 相机系）。"""
    e_z = np.asarray(n_cam, float)
    e_z = e_z / np.linalg.norm(e_z)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(e_z @ tmp) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e_x = tmp - (tmp @ e_z) * e_z
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(e_z, e_x)
    return e_x, e_y, e_z


def load_geometry(testdir):
    """读取 manifest 与相机系轨迹，返回世界系几何。"""
    man = json.load(open(os.path.join(testdir, "stage1_blender", "manifest.json")))
    n_cam = np.array(man["support_plane"]["normal_cam"], float)
    d_cam = float(man["support_plane"]["d_cam"])

    e_x, e_y, e_z = plane_frame(n_cam)
    # R: 世界系 → 相机系（列为世界基向量在相机系中的表示）
    R_wc = np.stack([e_x, e_y, e_z], axis=1)      # cam = R_wc @ world

    def cam_to_world(p):
        """相机系点 -> 世界系。原点取相机在桌面上的垂足。"""
        p = np.asarray(p, float)
        return (p - d_cam * e_z) @ R_wc           # = R_wc^T (p - d*n)

    def dir_to_world(v):
        return np.asarray(v, float) @ R_wc

    return man, n_cam, d_cam, R_wc, cam_to_world, dir_to_world


def report(testdir):
    man, n_cam, d_cam, R_wc, c2w, d2w = load_geometry(testdir)
    print(f"支撑面: n_cam={np.round(n_cam,4)} d={d_cam:.4f}")
    print(f"世界基(相机系): X={np.round(R_wc[:,0],4)} Y={np.round(R_wc[:,1],4)} Z={np.round(R_wc[:,2],4)}")
    cam_w = c2w(np.zeros(3))
    print(f"相机在世界系: {np.round(cam_w,4)}  高度={cam_w[2]:.4f} m")
    # 相机光轴在世界系
    axis = d2w(np.array([0.0, 0.0, 1.0]))
    print(f"相机光轴(世界系): {np.round(axis,4)}  俯角={np.degrees(np.arcsin(-axis[2])):.1f}°")
    for name, v in man["instances"].items():
        c_w = c2w(v["center_cam"])
        print(f"{name:14s} 世界系中心={np.round(c_w,4)}  离桌面高度={c_w[2]*1000:+.1f} mm  "
              f"尺寸={np.round(v['size'],4)}")
    return man, c2w, d2w


if __name__ == "__main__":
    import sys
    report(sys.argv[1])
