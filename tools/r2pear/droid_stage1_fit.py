#!/usr/bin/env python3
"""Stage 1 几何拟合（通用，形状：box / cylinder / prism / arch）。

参数：共享桌面高度 table_z；每物体 (x, y, yaw, 尺寸…)；z 由堆叠链决定（底面 = 支撑物顶面）。
目标 = −Σ_视角 Σ_物体 IoU(遮挡感知投影剪影, SAM2 mask) + 机器人约束罚项 + 尺寸弱先验
  遮挡：按物体中心到相机距离由远及近绘制标签图，物体可见区域 = 标签图中属于它的像素
机器人约束（task.target / task.support，事件帧来自 objects.json）：
  R1 抓取帧 pinch 水平距目标轴 ≤ 0.5·半宽；竖直在目标上下沿内（留 3 mm）
  R2 释放帧：目标底面（= pinch − 抓取时 pinch 相对目标底面的高度）落在支撑顶面上方 0–12 mm；pinch 水平在支撑足迹内
优化：Powell，多起点 yaw。输出 scene.json（与 r2pear 导出兼容）+ fit_overlay.png

用法: python tools/r2pear/droid_stage1_fit.py --testdir <T>
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
import droid_cams as DC  # noqa: E402

CAMS = ("exterior_1_left", "exterior_2_left", "wrist_left")
NSIZE = {"box": 3, "arch": 3, "prism": 3, "cylinder": 2}


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def parts(shape, size):
    """返回凸部件列表，每个部件为局部坐标点集（底面中心为原点，z 向上）。"""
    if shape == "cylinder":
        r, h = size
        t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        ring = np.stack([r * np.cos(t), r * np.sin(t)], 1)
        return [np.vstack([np.c_[ring, np.zeros(20)], np.c_[ring, np.full(20, h)]])]
    sx, sy, h = size
    def box(x0, x1, y0, y1, z0, z1):
        return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    if shape == "box":
        return [box(-sx / 2, sx / 2, -sy / 2, sy / 2, 0, h)]
    if shape == "prism":                      # 屋顶形：屋脊沿局部 x
        return [np.array([[x, y, 0] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2)] +
                         [[-sx / 2, 0, h], [sx / 2, 0, h]])]
    if shape == "arch":                       # 与 sim.py 一致：顶板 0.35h，腿高 0.65h、各占长 30%
        ts, lw = 0.35 * h, 0.30 * sx
        return [box(-sx / 2, sx / 2, -sy / 2, sy / 2, h - ts, h),
                box(-sx / 2, -sx / 2 + lw, -sy / 2, sy / 2, 0, h - ts),
                box(sx / 2 - lw, sx / 2, -sy / 2, sy / 2, 0, h - ts)]
    raise ValueError(shape)


def height(shape, size):
    return size[-1]


class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.names = list(cfg["objects"])
        self.shape = {n: cfg["objects"][n]["shape"] for n in self.names}
        self.stack = cfg["stacking"]
        # 参数布局：[table_z] + 每物体 [x, y, yaw?, size...]（圆柱无 yaw）
        self.layout, i = {}, 1
        for n in self.names:
            k = 2 + (0 if self.shape[n] == "cylinder" else 1) + NSIZE[self.shape[n]]
            self.layout[n] = (i, i + k)
            i += k
        # 每相机：相机系小旋转 3（度）+ 世界系平移 3（m）
        self.cam_layout = {c: (i + 6 * j, i + 6 * j + 6) for j, c in enumerate(CAMS)}
        self.dim = i + 6 * len(CAMS)

    def scales(self):
        sc = np.zeros(self.dim)
        sc[0] = 0.01
        for n in self.names:
            a, b = self.layout[n]
            k = b - a
            if self.shape[n] == "cylinder":
                sc[a:b] = [0.01, 0.01] + [0.005] * (k - 2)
            else:
                sc[a:b] = [0.01, 0.01, 0.2] + [0.005] * (k - 3)
        for c, (a, b) in self.cam_layout.items():
            free = c in getattr(self, "free_cams", ())
            sc[a:a + 3] = 1.0 if free else 0.0
            sc[a + 3:b] = 0.01 if free else 0.0
        return sc

    def cam_delta(self, v, c):
        a, b = self.cam_layout[c]
        return v[a:a + 3]

    def cam_trans(self, v, c):
        a, b = self.cam_layout[c]
        return v[a + 3:b]

    def unpack(self, v):
        tz = v[0]
        objs = {}
        for n in self.names:
            a, b = self.layout[n]
            p = v[a:b]
            if self.shape[n] == "cylinder":
                objs[n] = {"xy": p[:2], "yaw": 0.0, "size": np.abs(p[2:])}
            else:
                objs[n] = {"xy": p[:2], "yaw": p[2], "size": np.abs(p[3:])}
        # 堆叠链求 z（底面）
        def bottom(n, depth=0):
            s = self.stack[n]
            if s == "table":
                return tz
            return bottom(s, depth + 1) + height(self.shape[s], objs[s]["size"])
        for n in self.names:
            objs[n]["z0"] = bottom(n)
        return tz, objs

    def world_parts(self, n, o):
        R = rotz(o["yaw"])
        off = np.array([o["xy"][0], o["xy"][1], o["z0"]])
        return [P @ R.T + off for P in parts(self.shape[n], o["size"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--cams", nargs="+", default=list(CAMS), help="参与 IoU 的视角")
    ap.add_argument("--free_cams", nargs="*", default=[], help="允许联合修正位姿的相机（已标定的相机不应放开）")
    ap.add_argument("--tag", default="", help="输出文件后缀")
    a = ap.parse_args()
    globals()["CAMS"] = tuple(a.cams)            # 模块级视角列表（Model 与各闭包按全局名读取）
    import cv2
    from scipy.optimize import minimize
    T = a.testdir
    sd = os.path.join(T, "stage1_scene")
    cfg = json.load(open(os.path.join(sd, "objects.json")))
    masks = dict(np.load(os.path.join(sd, "masks.npz")))
    cams = DC.Cams(T)
    M = Model(cfg)
    M.free_cams = tuple(a.free_cams)
    fr = cfg["frame"]
    tgt, sup = cfg["task"]["target"], cfg["task"]["support"]
    pg, pr = cams.pinch(cfg["events"]["grasp"]), cams.pinch(cfg["events"]["release"])
    camP0 = {c: cams.pose_cv(c, fr) for c in CAMS}
    camP = dict(camP0)
    from scipy.spatial.transform import Rotation as Rot

    def set_cams(v):
        for c in CAMS:
            R0, t = camP0[c]
            camP[c] = (R0 @ Rot.from_euler("xyz", M.cam_delta(v, c), degrees=True).as_matrix(), t + M.cam_trans(v, c))

    def proj(c, P):
        Rcv, t = camP[c]
        pc = (P - t) @ Rcv
        f = cams.fx(c)
        return np.stack([160 + f * pc[:, 0] / pc[:, 2], 90 + f * pc[:, 1] / pc[:, 2]], 1), pc[:, 2]

    def silhouettes(objs):
        out = {}
        for c in CAMS:
            order = []
            for k, n in enumerate(M.names):
                wp = M.world_parts(n, objs[n])
                _, z = proj(c, np.vstack(wp))
                order.append((float(np.mean(z)), k, n, wp))
            lab = np.full((180, 320), -1, np.int16)
            for _, k, n, wp in sorted(order, reverse=True):          # 远 → 近
                for P in wp:
                    uv, z = proj(c, P)
                    if (z <= 0.02).any():
                        continue
                    hull = cv2.convexHull(np.round(uv * 4).astype(np.int32))
                    cv2.fillConvexPoly(lab, hull, k, shift=2)
            out[c] = lab
        return out

    SC = M.scales()

    def loss(u, detail=False):
        v = u * SC
        set_cams(v)
        tz, objs = M.unpack(v)
        sil = silhouettes(objs)
        iou_sum, n_terms, per = 0.0, 0, {}
        for c in CAMS:
            for k, n in enumerate(M.names):
                key = f"{n}|{c}"
                if key not in masks:
                    continue
                pm, mk = sil[c] == k, masks[key]
                u = (pm | mk).sum()
                iou = (pm & mk).sum() / u if u else 0.0
                per[key] = round(float(iou), 3)
                iou_sum += iou
                n_terms += 1
        L = -iou_sum / n_terms
        # 机器人约束（单位 m，5 mm 违反 ≈ 0.05 IoU）
        o, s = objs[tgt], objs[sup]
        half = (o["size"][0] if M.shape[tgt] == "cylinder" else min(o["size"][:2]) / 2)
        hz = height(M.shape[tgt], o["size"])
        pen = 0.0
        dxy = np.linalg.norm(pg[:2] - o["xy"])
        pen += max(0.0, dxy - 0.5 * half) ** 2
        pen += max(0.0, (o["z0"] + 0.003) - pg[2]) ** 2 + max(0.0, pg[2] - (o["z0"] + hz - 0.003)) ** 2
        off = pg[2] - o["z0"]
        gap = (pr[2] - off) - (s["z0"] + height(M.shape[sup], s["size"]))
        pen += max(0.0, -gap) ** 2 + max(0.0, gap - 0.012) ** 2
        loc = rotz(-s["yaw"])[:2, :2] @ (pr[:2] - s["xy"])
        if M.shape[sup] != "cylinder":
            pen += max(0.0, abs(loc[0]) - s["size"][0] / 2) ** 2 + max(0.0, abs(loc[1]) - s["size"][1] / 2) ** 2
        L += pen * (0.05 / 0.005 ** 2)
        # 尺寸弱先验（相对 ±30% 内不罚）
        pri = cfg.get("size_prior_m", {})
        for n in M.names:
            if n in pri:
                rel = np.asarray(objs[n]["size"]) / np.asarray(pri[n]) - 1
                L += 0.05 * float(np.sum(np.maximum(0, np.abs(rel) - 0.3) ** 2))
        for c in CAMS:                                            # 相机修正：±8° 内不罚
            L += 0.02 * float(np.sum(np.maximum(0, np.abs(M.cam_delta(v, c)) - 8.0) ** 2))
            L += 20.0 * float(np.sum(np.maximum(0, np.abs(M.cam_trans(v, c)) - 0.05) ** 2))   # ±5 cm 内不罚
        if detail:
            return L, per, {"grasp_dxy_mm": dxy * 1000, "grasp_height_in_target_mm": off * 1000,
                            "release_gap_mm": gap * 1000, "release_pinch_in_support_local_mm": (loc * 1000).tolist()}
        return L

    # 初值：堆叠物体 xy = 抓取 pinch；支撑 = 释放 pinch；其余由 ext1 mask 质心反投影到桌面
    tz0 = pg[2] - (sum(cfg["size_prior_m"][n][-1] for n in ("green_arch", "yellow_cube")) if False else 0.0)
    def chain_h(n):
        s = cfg["stacking"][n]
        return 0.0 if s == "table" else chain_h(s) + cfg["size_prior_m"][s][-1]
    tz0 = pg[2] - chain_h(tgt) - cfg["size_prior_m"][tgt][-1] / 2

    def backproject(n):
        c0 = next(c for c in CAMS if f"{n}|{c}" in masks)      # 用第一个参与拟合、且有该物体 mask 的视角
        mk = masks[f"{n}|{c0}"]
        ys, xs = np.nonzero(mk)
        u, vv = xs.mean(), ys.mean()
        Rcv, t = camP[c0]
        f = cams.fx(c0)
        ray = Rcv @ np.array([(u - 160) / f, (vv - 90) / f, 1.0])
        lam = (tz0 + 0.01 - t[2]) / ray[2]
        return (t + lam * ray)[:2]

    best = None
    rng = np.random.default_rng(0)
    for r in range(a.restarts):
        v = [tz0]
        for n in M.names:
            role = cfg["objects"][n]["role"]
            xy = pg[:2] if cfg["stacking"][n] != "table" or role == "target" or n == "green_arch" else (
                pr[:2] if role == "support" else backproject(n))
            if n == "green_arch":
                xy = pg[:2]
            xy = np.asarray(xy) + (rng.normal(0, 0.004, 2) if r else 0)
            yaw = [] if M.shape[n] == "cylinder" else [rng.uniform(-np.pi / 2, np.pi / 2) if r else 0.0]
            v += list(xy) + yaw + list(cfg["size_prior_m"][n])
        v += [0.0] * (6 * len(CAMS))
        v = np.array(v, float)
        act = SC > 0                                       # 固定相机的参数不参与优化
        def loss_act(u_act):
            u = np.zeros(M.dim)
            u[act] = u_act
            return loss(u)
        res = minimize(loss_act, (v / np.where(act, SC, 1))[act], method="Powell",
                       options={"maxiter": 60000, "xtol": 1e-3, "ftol": 1e-6})
        full = np.zeros(M.dim)
        full[act] = res.x
        res.x = full
        print(f"restart {r}: loss {res.fun:.4f}  nfev {res.nfev}")
        if best is None or res.fun < best.fun:
            best = res
    L, per, rob = loss(best.x, detail=True)
    vb = best.x * SC
    set_cams(vb)
    tz, objs = M.unpack(vb)
    cam_delta = {c: {"rot_deg": np.round(M.cam_delta(vb, c), 2).tolist(),
                     "trans_mm": np.round(M.cam_trans(vb, c) * 1000, 1).tolist()} for c in CAMS}
    print("camera deltas (deg, on top of stage0):", cam_delta)
    print("table_z", round(tz, 4), "loss", round(L, 4))
    for n in M.names:
        o = objs[n]
        print(f"  {n:15s} xy=({o['xy'][0]:.4f},{o['xy'][1]:.4f}) yaw={math.degrees(o['yaw']):6.1f} size_mm={np.round(np.array(o['size'])*1000,1).tolist()} z0={o['z0']*1000:.1f}mm")
    print("IoU", per)
    print("robot", {k: (np.round(v, 2).tolist() if isinstance(v, list) else round(v, 2)) for k, v in rob.items()})

    # 输出 scene.json（中心坐标 = 底面 + h/2；圆柱 size = [r, h]）
    scene = {"frame": fr, "table_z": float(tz), "objects": {}, "camera_delta_deg_on_stage0": cam_delta,
             "fit": {"loss": float(L), "iou": per, "robot_constraints": rob},
             "method": "遮挡感知多视角剪影 IoU + 机器人抓放约束 + 尺寸弱先验 + 相机小旋转联合精修（论文 Sec 3.1），Powell"}
    for n in M.names:
        o = objs[n]
        h = height(M.shape[n], o["size"])
        scene["objects"][n] = {"shape": M.shape[n], "center": [float(o["xy"][0]), float(o["xy"][1]), float(o["z0"] + h / 2)],
                               "yaw_rad": float(o["yaw"]), "size": [float(x) for x in o["size"]],
                               "role": cfg["objects"][n]["role"]}
    scene["fit"]["views"], scene["fit"]["free_cams"] = list(CAMS), list(a.free_cams)
    scene["fit"]["camera_sources"] = {c: cams.source(c) for c in CAMS}
    json.dump(scene, open(os.path.join(sd, f"scene{a.tag}.json"), "w"), indent=1, ensure_ascii=False)
    sil = silhouettes(objs)
    rows = []
    for c in CAMS:
        cap = cv2.VideoCapture(os.path.join(T, "input", "videos", f"{c}.mp4"))
        cap.set(1, fr)
        _, bgr = cap.read()
        v = bgr.copy()
        for k, n in enumerate(M.names):
            cnt, _ = cv2.findContours((sil[c] == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cnt, -1, (255, 0, 255), 1)
            if f"{n}|{c}" in masks:
                cnt, _ = cv2.findContours(masks[f"{n}|{c}"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(v, cnt, -1, (0, 255, 0), 1)
        v = cv2.resize(v, (640, 360), interpolation=cv2.INTER_NEAREST)
        cv2.putText(v, f"{c}  magenta=fit  green=SAM2", (5, 18), 0, 0.5, (0, 255, 255), 2)
        rows.append(v)
    cv2.imwrite(os.path.join(sd, f"fit_overlay{a.tag}.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
