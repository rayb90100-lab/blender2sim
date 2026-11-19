#!/usr/bin/env python3
"""修正 Stage 1 的物体朝向：用轮廓 IoU 搜索最优 yaw。

Stage 1 建的是**轴对齐箱体**（yaw=0），但真实物体是斜放的。
最小面积矩形给出的初值不够可靠（耳机盒的点云散布特征值比只有 1.56），
所以这里用「投影轮廓 vs SAM2 mask 的 IoU」做判据：

  对每个 yaw，在 (cu, cv) 与 (su, sv) 上做局部优化，
  把箱体投影成图像轮廓，与 mask 求 IoU，取 IoU 最大者。

这比最小面积矩形稳健，因为它直接对齐的是「看到的那一片像素」。

用法:
  python tools/stage2_fix_orientation.py --testdir <testdir> [--apply]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_common import load_geometry  # noqa: E402


def log(m): print(m, flush=True)


def project_box(center, axes, size, K, W, H):
    """把有向箱体的 8 个顶点投影到图像，返回凸包 mask。"""
    import cv2
    c = np.asarray(center, float)
    a1, a2, a3 = [np.asarray(a, float) for a in axes]
    s1, s2, s3 = np.asarray(size, float) / 2
    pts = []
    for i in (-1, 1):
        for j in (-1, 1):
            for k in (-1, 1):
                pts.append(c + i * s1 * a1 + j * s2 * a2 + k * s3 * a3)
    pts = np.array(pts)
    uv = np.stack([K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2],
                   K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]], 1)
    hull = cv2.convexHull(uv.astype(np.int32))
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def refine(testdir, name, mask, P, D, K, ex, ey, ez, d_plane, yaw0, size0, ctr0,
           yaw_range=(-np.pi / 2, np.pi / 2), coarse=72, iters=3):
    """在 yaw × (中心, 尺寸) 上做坐标下降，最大化 IoU。"""
    from scipy.optimize import minimize

    H, W = mask.shape
    best = {"iou": -1, "yaw": yaw0, "size": list(size0), "center": list(ctr0)}

    def axes_for(yaw):
        a1 = np.cos(yaw) * ex + np.sin(yaw) * ey     # 箱体本地 x
        a2 = -np.sin(yaw) * ex + np.cos(yaw) * ey    # 箱体本地 y
        a3 = ez
        return (a1, a2, a3)

    def score(yaw, ctr_local, size):
        a1, a2, a3 = axes_for(yaw)
        c = ctr_local[0] * ex + ctr_local[1] * ey + ctr_local[2] * ez
        pm = project_box(c, (a1, a2, a3), size, K, W, H)
        return iou(pm, mask)

    # 初值：把中心、尺寸转到 (u,v,h) 基
    cu0 = float(np.dot(ctr0, ex)); cv0 = float(np.dot(ctr0, ey))
    ch0 = float(np.dot(ctr0, ez) - d_plane)

    yaws = np.linspace(yaw_range[0], yaw_range[1], coarse)
    for y in yaws:
        s = score(y, (cu0, cv0, ch0), size0)
        if s > best["iou"]:
            best = {"iou": s, "yaw": float(y),
                    "size": list(size0), "center": [cu0, cv0, ch0]}

    # 坐标下降：交替优化 (size, center) 与 yaw
    for _ in range(iters):
        x0 = np.array([cu0, cv0, np.log(best["size"][0]), np.log(best["size"][1])])

        def neg(x):
            cu, cv, lsx, lsy = x
            sz = [np.exp(lsx), np.exp(lsy), best["size"][2]]
            return -score(best["yaw"], (cu, cv, ch0), sz)

        res = minimize(neg, x0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-5, "fatol": 1e-5})
        cu0, cv0, lsx, lsy = res.x
        best["size"] = [float(np.exp(lsx)), float(np.exp(lsy)), float(best["size"][2])]
        best["center"] = [float(cu0), float(cv0), ch0]

        # 精搜 yaw
        cand = np.linspace(best["yaw"] - 0.25, best["yaw"] + 0.25, 51)
        for y in cand:
            s = score(y, (cu0, cv0, ch0), best["size"])
            if s > best["iou"]:
                best = {"iou": s, "yaw": float(y),
                        "size": list(best["size"]), "center": [cu0, cv0, ch0]}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    T = a.testdir

    man, n_cam, d_cam, R_wc, c2w, d2w = load_geometry(T)
    g = np.load(os.path.join(T, "stage1_moge3", "moge_out.npz"))
    P, D = g["points"], g["depth"]
    K = g["intrinsics_px"]
    H, W = D.shape

    e_z = np.array(n_cam) / np.linalg.norm(n_cam)
    e_x = R_wc[:, 0]
    e_y = R_wc[:, 1]

    out = {}
    for name in ["earbud_case", "tissue_box"]:
        mask = np.load(os.path.join(T, "stage1_sam2", f"{name}_mask.npy")).astype(bool)
        v = man["instances"][name]
        size0 = np.array(v["size"], float)
        ctr0 = np.array(v["center_cam"], float)

        # 基线：yaw=0，Stage 1 的轴对齐盒
        base_iou = iou(project_box(ctr0, (e_x, e_y, e_z), size0, K, W, H), mask)

        best = refine(T, name, mask, P, D, K, e_x, e_y, e_z, d_cam,
                      yaw0=0.0, size0=size0, ctr0=ctr0)
        log(f"{name:14s} yaw=0 基线 IoU {base_iou:.4f}  ->  最优 yaw={np.degrees(best['yaw']):+7.2f}° "
            f"IoU {best['iou']:.4f}  (+{(best['iou']-base_iou)*100:.1f} 点)")
        log(f"{'':14s} 尺寸 {np.round(size0*1000,1)} -> {np.round(np.array(best['size'])*1000,1)} mm")

        out[name] = {
            "yaw_rad": best["yaw"], "yaw_deg": float(np.degrees(best["yaw"])),
            "size": best["size"], "center_local": best["center"],
            "iou_baseline": float(base_iou), "iou_refined": float(best["iou"]),
            "size_baseline": list(size0), "center_cam_baseline": list(ctr0),
        }

    p = os.path.join(T, "stage2_physics", "mujoco", "orientation.json")
    json.dump({"note": "yaw 由轮廓 IoU 搜索得出（对比轴对齐基线）", "objects": out},
              open(p, "w"), indent=2, ensure_ascii=False)
    log(f"\n写出 {p}")
    if a.apply:
        log("场景会由 stage2_build_scene.py 读取 orientation.json")


if __name__ == "__main__":
    main()
