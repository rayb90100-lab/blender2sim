#!/usr/bin/env python3
"""test4 Stage 1：DROID 双外部相机下的场景重建（桌面 + 方块）。

和 test3 的区别：
  - 世界系 = Panda 基座（数据集给定），不用桌面法线反推
  - 两路外部相机外参已知、内参已估（fx=130），可多视角联合拟合，不依赖单目深度
  - SAM2 提示点来自机器人 FK：抓取时刻 pinch 点在被抓物体上，释放时刻 pinch 点在放置目标正上方

拟合：每个物体参数化为直立箱体 (cx, cy, yaw, sx, sy, h)，底面贴共享桌面 z_table；
目标函数 = 两路相机「箱体投影 vs SAM2 mask」IoU 之和，Nelder-Mead 多起点。

用法:
  python tools/droid_stage1_scene.py --testdir <testdir>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import droid_common as C  # noqa: E402

SAM2_CFG = "configs/sam2/sam2_hiera_l.yaml"
SAM2_W = "/media/data1/RoboLab_data/weights/sam2/sam2_hiera_large.pt"
CAMS = ("exterior_1_left", "exterior_2_left")
# 尺寸先验（m，宽×高投影量级），只用于挑 mask 和判遮挡，不进入最终尺寸
PRIOR_SIZE = {"green_block": (0.04, 0.04), "block_stack": (0.06, 0.09)}
W, H = 320, 180


def log(m): print(m, flush=True)


class Cam:
    def __init__(self, ext, fx, fy, cx, cy):
        self.t = np.asarray(ext[:3], float)
        self.R = C.euler_to_R(ext[3:6], "xyz")       # 相机 -> 世界
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def project(self, P):
        P = np.atleast_2d(P)
        pc = (P - self.t) @ self.R                    # 世界 -> 相机
        z = pc[:, 2]
        uv = np.stack([self.fx * pc[:, 0] / z + self.cx, self.fy * pc[:, 1] / z + self.cy], 1)
        return uv, z

    def ray(self, u, v):
        d = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        d = self.R @ d
        return self.t, d / np.linalg.norm(d)


def box_corners(cx, cy, yaw, sx, sy, z0, h):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = []
    for i in (-1, 1):
        for j in (-1, 1):
            for z in (z0, z0 + h):
                x = cx + c * i * sx / 2 - s * j * sy / 2
                y = cy + s * i * sx / 2 + c * j * sy / 2
                pts.append([x, y, z])
    return np.array(pts)


def box_mask(cam, corners):
    import cv2
    uv, z = cam.project(corners)
    m = np.zeros((H, W), np.uint8)
    if (z <= 0.01).any():
        return m.astype(bool)
    hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def pinch_world(ep, frame):
    import mujoco
    m, d, _ = C.build_model()
    d.qpos[C.arm_qadr(m)] = ep["observation.state.joint_position"][frame]
    mujoco.mj_kinematics(m, d)
    return d.site_xpos[m.site("g_pinch").id].copy()


def triangulate(cams, uvs):
    """多相机射线最小二乘交点。"""
    A, b = [], []
    for cam, (u, v) in zip(cams, uvs):
        o, d = cam.ray(u, v)
        P = np.eye(3) - np.outer(d, d)
        A.append(P)
        b.append(P @ o)
    return np.linalg.solve(np.sum(A, 0), np.sum(b, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--grasp_frame", type=int, default=84)
    ap.add_argument("--release_frame", type=int, default=142)
    a = ap.parse_args()
    import cv2
    import torch
    from scipy.optimize import minimize
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    T = a.testdir
    out = os.path.join(T, "stage1_scene")
    os.makedirs(out, exist_ok=True)
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    K = json.load(open(os.path.join(T, "stage0_replay", "intrinsics_adopted.json")))
    cams = {c: Cam(ep[f"camera_extrinsics.{c}"][0], K["fx"], K["fy"], K["cx"], K["cy"]) for c in CAMS}

    # 首帧图像
    imgs = {}
    for c in CAMS:
        cap = cv2.VideoCapture(os.path.join(T, "input", "videos", f"{c}.mp4"))
        ok, f = cap.read()
        cap.release()
        imgs[c] = f

    # ---- FK 给出的物体锚点 ----
    p_grasp = pinch_world(ep, a.grasp_frame)        # 绿方块中心附近
    p_release = pinch_world(ep, a.release_frame)    # 放置后绿方块中心（在方块堆顶上）
    log(f"抓取时 pinch（世界系）: {np.round(p_grasp, 4)}")
    log(f"释放时 pinch（世界系）: {np.round(p_release, 4)}")

    anchors = {
        "green_block": [p_grasp],
        # 方块堆：释放点正下方几个高度
        "block_stack": [p_release - np.array([0, 0, dz]) for dz in (0.05, 0.08)],
    }

    pred = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_W, device="cuda"))
    masks = {}
    for c in CAMS:
        pred.set_image(cv2.cvtColor(imgs[c], cv2.COLOR_BGR2RGB))
        for name, pts in anchors.items():
            uv, z = cams[c].project(np.array(pts))
            keep = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0)
            uv = uv[keep]
            if not len(uv):
                log(f"  {c} {name}: 锚点不在画面内")
                continue
            with torch.inference_mode():
                ms, sc, _ = pred.predict(point_coords=uv.astype(np.float32),
                                         point_labels=np.ones(len(uv), np.int32),
                                         multimask_output=True)
            # 按「预期像素面积」选 mask：尺寸先验 / 深度 → 像素面积。
            # 曾用「分数>0.6 的最小 mask」，在 ext2 选中了整张桌子（4971 px）。
            prior = PRIOR_SIZE[name]
            exp_area = (K["fx"] * prior[0] / z[keep].mean()) * (K["fy"] * prior[1] / z[keep].mean())
            ratios = np.array([mm.sum() / exp_area for mm in ms])
            cost = np.abs(np.log(np.maximum(ratios, 1e-6))) - 0.5 * sc
            best = int(np.argmin(cost))
            r = ratios[best]
            if r > 5 or r < 0.2 or sc[best] < 0.3:
                log(f"  {c:16s} {name:12s} 预期 {exp_area:.0f} px，最接近 {int(ms[best].sum())} px "
                    f"(×{r:.1f}, score {sc[best]:.2f}) → 判定该视角被遮挡/分割失败，丢弃")
                continue
            masks[(c, name)] = ms[best].astype(bool)
            log(f"  {c:16s} {name:12s} 提示 {np.round(uv, 1).tolist()}  score {sc[best]:.3f}  "
                f"像素 {int(ms[best].sum())}（预期 {exp_area:.0f}，×{r:.2f}）")

    # ---- 初值：mask 质心三角化 ----
    init = {}
    for name in anchors:
        uvs, cs = [], []
        for c in CAMS:
            mk = masks.get((c, name))
            if mk is None or mk.sum() < 10:
                continue
            ys, xs = np.nonzero(mk)
            uvs.append((xs.mean(), ys.mean()))
            cs.append(cams[c])
        if len(cs) == 2:
            init[name] = triangulate(cs, uvs)
            log(f"  {name}: 质心三角化 {np.round(init[name], 4)}")
        elif len(cs) == 1:
            init[name] = np.array(anchors[name][0], float)
            log(f"  {name}: 仅 1 个视角可见，初值取 FK 锚点 {np.round(init[name], 4)}")

    # ---- 联合拟合 ----
    # 桌面高度初值：绿方块质心 - 2 cm
    z_tab0 = float(p_grasp[2] - 0.02)
    names = [n for n in anchors if n in init]

    def unpack(x):
        z_tab = x[0]
        objs = {}
        for i, n in enumerate(names):
            cx, cy, yaw, lsx, lsy, lh = x[1 + 6 * i: 7 + 6 * i]
            objs[n] = (cx, cy, yaw, np.exp(lsx), np.exp(lsy), np.exp(lh))
        return z_tab, objs

    # 机器人测量：f84 夹爪闭合夹住绿方块 → pinch 点在方块内部。
    # 用软约束：水平偏离超出半尺寸、或竖直偏离超出半高时加罚。
    def grasp_penalty(objs, z_tab):
        if "green_block" not in objs:
            return 0.0
        cx, cy, yaw, sx, sy, h = objs["green_block"]
        c, s_ = np.cos(yaw), np.sin(yaw)
        dx, dy = p_grasp[0] - cx, p_grasp[1] - cy
        lx, ly = c * dx + s_ * dy, -s_ * dx + c * dy
        lz = p_grasp[2] - (z_tab + h / 2)
        ex = max(0, abs(lx) - sx / 2) + max(0, abs(ly) - sy / 2) + max(0, abs(lz) - h / 2)
        return 50.0 * ex

    def score(x):
        z_tab, objs = unpack(x)
        s = -grasp_penalty(objs, z_tab)
        for n, (cx, cy, yaw, sx, sy, h) in objs.items():
            cor = box_corners(cx, cy, yaw, sx, sy, z_tab, h)
            for c in CAMS:
                mk = masks.get((c, n))
                if mk is not None:
                    s += iou(box_mask(cams[c], cor), mk)
        return -s

    best = None
    rng = np.random.default_rng(0)
    for trial in range(12):
        x0 = [z_tab0 + rng.normal(0, 0.01)]
        for n in names:
            p = init[n]
            sz = 0.04 if n == "green_block" else 0.07
            hh = 0.04 if n == "green_block" else 0.09
            x0 += [p[0], p[1], rng.uniform(-np.pi / 4, np.pi / 4),
                   np.log(sz), np.log(sz), np.log(hh)]
        r = minimize(score, np.array(x0), method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-6, "adaptive": True})
        if best is None or r.fun < best.fun:
            best = r
        log(f"  起点 {trial:2d}: IoU 和 {-r.fun:.3f}")
    z_tab, objs = unpack(best.x)

    # ---- 第二轮：用第一轮箱体投影框做 box 提示，回填被丢弃的视角 ----
    # 只回填“投影框没有被其他物体完全挡住”的视角：以 box 面积比 + 分数双重判定
    added = 0
    for c in CAMS:
        pred.set_image(cv2.cvtColor(imgs[c], cv2.COLOR_BGR2RGB))
        for n in names:
            if (c, n) in masks:
                continue
            pm = box_mask(cams[c], box_corners(*objs[n][:5], z_tab, objs[n][5]))
            if pm.sum() < 10:
                continue
            # 几何遮挡判定：更近的物体投影盖住本物体投影 >30% → 不回填。
            # 曾只用「SAM2 mask 与投影 IoU>0.3」判定，ext2 里绿方块被堆挡住，
            # SAM2 却在投影框里分到了背后的纸箱（IoU 0.58），被误采纳。
            ctr_n = np.array([objs[n][0], objs[n][1], z_tab + objs[n][5] / 2])
            dist_n = np.linalg.norm(ctr_n - cams[c].t)
            occ = np.zeros_like(pm)
            for o in names:
                if o == n:
                    continue
                ctr_o = np.array([objs[o][0], objs[o][1], z_tab + objs[o][5] / 2])
                if np.linalg.norm(ctr_o - cams[c].t) < dist_n:
                    occ |= box_mask(cams[c], box_corners(*objs[o][:5], z_tab, objs[o][5]))
            occ_frac = (pm & occ).sum() / pm.sum()
            if occ_frac > 0.3:
                log(f"  回填 {c:16s} {n:12s} 被更近物体遮挡 {occ_frac*100:.0f}% → 跳过")
                continue
            ys, xs = np.nonzero(pm)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
            # 多 mask 输出：方块堆由多块不同颜色组成，单 mask 常只分出其中一块（面积比 0.09）
            with torch.inference_mode():
                ms, scs, _ = pred.predict(box=box, multimask_output=True)
            k = int(np.argmax([iou(mm.astype(bool), pm) for mm in ms]))
            mk, sc = ms[k].astype(bool), scs[k:k + 1]
            r = mk.sum() / pm.sum()
            ok = 0.4 < r < 2.5 and sc[0] > 0.3 and iou(mk, pm) > 0.3
            log(f"  回填 {c:16s} {n:12s} box {box.astype(int).tolist()} score {sc[0]:.2f} "
                f"面积比 {r:.2f} IoU(投影) {iou(mk, pm):.2f} → {'采用' if ok else '不采用（遮挡）'}")
            if ok:
                masks[(c, n)] = mk
                added += 1
    if added:
        r2 = minimize(score, best.x, method="Nelder-Mead",
                      options={"maxiter": 6000, "xatol": 1e-6, "fatol": 1e-7, "adaptive": True})
        log(f"  双视角精修：IoU 和 {-best.fun:.3f} → {-r2.fun:.3f}")
        best = r2
        z_tab, objs = unpack(best.x)
    log(f"\n桌面高度 z_table = {z_tab:.4f} m（基座系）")

    result = {"frame": 0, "table_z": float(z_tab), "objects": {},
              "cameras": {c: {"extrinsics": ep[f"camera_extrinsics.{c}"][0].tolist()} for c in CAMS},
              "intrinsics": K, "method": "双视角箱体投影 IoU 联合拟合；SAM2 提示来自 FK pinch 点"}
    vis = []
    for c in CAMS:
        v = imgs[c].copy()
        for n, (cx, cy, yaw, sx, sy, h) in objs.items():
            mk = masks.get((c, n))
            if mk is not None:
                v[mk] = (0.5 * v[mk] + [0, 100, 0]).astype(np.uint8)
            pm = box_mask(cams[c], box_corners(cx, cy, yaw, sx, sy, z_tab, h))
            cnt, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cnt, -1, (0, 0, 255), 1)
        vis.append(cv2.resize(v, (640, 360), interpolation=cv2.INTER_CUBIC))
    for n, (cx, cy, yaw, sx, sy, h) in objs.items():
        ious = {c: float(iou(box_mask(cams[c], box_corners(cx, cy, yaw, sx, sy, z_tab, h)),
                             masks[(c, n)])) for c in CAMS if (c, n) in masks}
        result["objects"][n] = {"center": [float(cx), float(cy), float(z_tab + h / 2)],
                                "yaw_rad": float(yaw), "size": [float(sx), float(sy), float(h)],
                                "iou": ious}
        log(f"{n:12s} 中心 ({cx:.4f}, {cy:.4f}, {z_tab + h / 2:.4f})  yaw {np.degrees(yaw):+6.1f}°  "
            f"尺寸 {sx*1000:.1f}×{sy*1000:.1f}×{h*1000:.1f} mm  IoU {ious}")

    # 物理一致性：抓取时 pinch 应在绿方块内部
    g = result["objects"]["green_block"]
    d = np.array(p_grasp) - np.array(g["center"])
    log(f"\n校验：抓取时 pinch 相对绿方块中心偏差 {np.round(d * 1000, 1)} mm "
        f"（半尺寸 {np.round(np.array(g['size']) / 2 * 1000, 1)} mm）")
    result["check_grasp_pinch_offset_mm"] = (d * 1000).tolist()

    cv2.imwrite(os.path.join(out, "scene_fit.png"), np.hstack(vis))
    for (c, n), mk in masks.items():
        np.save(os.path.join(out, f"mask_{c}_{n}.npy"), mk)
    json.dump(result, open(os.path.join(out, "scene.json"), "w"), indent=2, ensure_ascii=False)
    log(f"写出 {out}/scene.json, scene_fit.png")


if __name__ == "__main__":
    main()
