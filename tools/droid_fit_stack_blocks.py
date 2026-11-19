#!/usr/bin/env python3
"""把方块堆拆成独立积木（拱 + 长板 + 小方块），多视角联合拟合。

背景（用户指出，DEVELOPMENT_LOG §49）：Stage 1 把整摞积木建成一个 163 g 刚体箱，
它不可能被碰倒 → 「放积木不碰倒塔」这一任务核心的物理无法被验证。

真实结构（腕部/外部放大图核实）：
  arch  橙红色 ∩ 形拱（两腿中空）  —— 桌面
  plank 黄色长条板（两端伸出拱外） —— 架在拱上
  cube  红色小方块                —— 压在板上；绿方块最终叠在它上面

做法：
  1. SAM2 提示点：旧整箱在 z = z_t + {0.2, 0.55, 0.85}·h 处的投影（自动，不手点）；多 mask 输出按颜色校验
  2. 三视角 ext1 f0 / ext2 f0 / wrist f0 联合 IoU，链式支撑：plank 底 = arch 顶，cube 底 = plank 顶
  3. 物理中 arch 建为复合体：顶板 + 两腿（腿宽、中空为近似，图像分辨率不足以测量）

用法: python tools/droid_fit_stack_blocks.py --testdir <testdir> [--apply]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import droid_common as C  # noqa: E402
from droid_stage1_scene import Cam, box_corners, box_mask, iou  # noqa: E402
from droid_refine_block_wrist import wrist_cam  # noqa: E402

SAM2_CFG = "configs/sam2/sam2_hiera_l.yaml"
SAM2_W = "/media/data1/RoboLab_data/weights/sam2/sam2_hiera_large.pt"
BLOCKS = [("arch", 0.20, "red"), ("plank", 0.55, "yellow"), ("cube", 0.85, "red")]
W, H = 320, 180


def log(m): print(m, flush=True)


def color_class(bgr_mean):
    b, g, r = bgr_mean
    if r > 120 and g > 120 and b < 110:
        return "yellow"
    if r > 110 and g < 110:
        return "red"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    import cv2
    import torch
    from scipy.optimize import minimize
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    T = a.testdir
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    sp = os.path.join(T, "stage1_scene", "scene.json")
    scene = json.load(open(sp))
    K = scene["intrinsics"]
    zt = scene["table_z"]
    st = scene["objects"].get("block_stack") or next(h["block_stack"] for h in scene["history"] if "block_stack" in h)
    scx, scy = st["center"][:2]
    sh = st["size"][2]

    cams = {"exterior_1_left": Cam(ep["camera_extrinsics.exterior_1_left"][0], K["fx"], K["fy"], K["cx"], K["cy"]),
            "exterior_2_left": Cam(ep["camera_extrinsics.exterior_2_left"][0], K["fx"], K["fy"], K["cx"], K["cy"]),
            "wrist_left": wrist_cam(T, ep, 0, 150.0)}
    imgs = {}
    for c in cams:
        cap = cv2.VideoCapture(os.path.join(T, "input", "videos", f"{c}.mp4"))
        imgs[c] = cap.read()[1]
        cap.release()

    pred = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_W, device="cuda"))
    masks = {}
    for c in cams:
        pred.set_image(cv2.cvtColor(imgs[c], cv2.COLOR_BGR2RGB))
        for name, frac, col in BLOCKS:
            if (c, name) in masks:
                continue
            uv, z = cams[c].project(np.array([[scx, scy, zt + frac * sh]]))
            u, v = uv[0]
            if not (0 <= u < W and 0 <= v < H and z[0] > 0):
                log(f"  {c:16s} {name:6s} 提示点出画，跳过")
                continue
            with torch.inference_mode():
                ms, sc, _ = pred.predict(point_coords=uv.astype(np.float32), point_labels=np.ones(1, np.int32),
                                         multimask_output=True)
            # 选颜色对、面积最小的（大 mask 常把整摞吞进来）
            cand = []
            for k in range(len(ms)):
                mk = ms[k].astype(bool)
                if mk.sum() < 8:
                    continue
                cc = color_class(imgs[c][mk].mean(0))
                cand.append((cc == col, -mk.sum(), sc[k], k, cc))
            cand.sort(reverse=True)
            # 俯视的腕部视角里，按高度投影的提示点会错位（arch 点落到黄板上）：
            # 若颜色不符但得到的是黄色 mask，且 plank 尚未分到，则归给 plank。
            if cand and not cand[0][0] and name == "arch":
                yel = [x for x in cand if x[4] == "yellow"]
                if yel and (c, "plank") not in masks and sc[yel[0][3]] >= 0.3:
                    masks[(c, "plank")] = ms[yel[0][3]].astype(bool)
                    log(f"  {c:16s} arch   提示落在黄板上 → 该 mask 归 plank（像素 {int(masks[(c, 'plank')].sum())}）")
                    continue
            if cand and cand[0][0] and sc[cand[0][3]] < 0.3:
                log(f"  {c:16s} {name:6s} 颜色匹配但 SAM2 分数 {sc[cand[0][3]]:.2f} < 0.3，丢弃")
                continue
            if not cand or not cand[0][0]:
                log(f"  {c:16s} {name:6s} 提示 ({u:.0f},{v:.0f}) 无颜色匹配的 mask（{[x[4] for x in cand]}），跳过")
                continue
            k = cand[0][3]
            masks[(c, name)] = ms[k].astype(bool)
            log(f"  {c:16s} {name:6s} 提示 ({u:.0f},{v:.0f})  颜色 {col}  像素 {int(ms[k].sum())}  score {sc[k]:.2f}")

    names = [b[0] for b in BLOCKS]
    # 机器人测量：f142 释放时夹爪中心 xy —— 真实画面（腕部 f146）绿方块正对红方块，
    # 故红方块中心应在释放点附近（±5 mm）。机器人运动学比单视角腕部深度可靠（§49）。
    import mujoco
    mm_, dd_, _ = C.build_model()
    dd_.qpos[C.arm_qadr(mm_)] = ep["observation.state.joint_position"][142]
    mujoco.mj_kinematics(mm_, dd_)
    p_rel = dd_.site_xpos[mm_.site("g_pinch").id].copy()
    log(f"  释放点 pinch xy = ({p_rel[0]:.4f}, {p_rel[1]:.4f})")

    def unpack(x):
        out, z0 = {}, zt
        for i, n in enumerate(names):
            cx, cy, yaw, lsx, lsy, lh = x[6 * i: 6 * i + 6]
            sx, sy, h = np.exp(lsx), np.exp(lsy), np.exp(lh)
            out[n] = (cx, cy, yaw, sx, sy, z0, h)
            z0 = z0 + h                                # 链式支撑
        return out

    def score(x):
        bl = unpack(x)
        s = 0.0
        for (c, n), mk in masks.items():
            cx, cy, yaw, sx, sy, z0, h = bl[n]
            s += iou(box_mask(cams[c], box_corners(cx, cy, yaw, sx, sy, z0, h)), mk)
        # 软先验：总高 ≈ 旧整箱高度（已由释放时 pinch 高度独立校验，§42），各块中心在旧整箱足迹内
        top = bl["cube"][5] + bl["cube"][6]
        s -= 20.0 * abs(top - (zt + sh))
        for n in names:
            s -= 20.0 * max(0.0, np.hypot(bl[n][0] - scx, bl[n][1] - scy) - 0.04)
        s -= 200.0 * max(0.0, np.hypot(bl["cube"][0] - p_rel[0], bl["cube"][1] - p_rel[1]) - 0.005)
        return -s

    # 初值：整箱切三段
    h0 = {"arch": 0.40 * sh, "plank": 0.22 * sh, "cube": 0.38 * sh}
    s0 = {"arch": (0.060, 0.035), "plank": (0.075, 0.025), "cube": (0.032, 0.032)}
    best = None
    for yaw0 in np.radians([-60, -30, 0, 30, 60, 90]):
        x0 = []
        for n in names:
            x0 += [scx, scy, yaw0, np.log(s0[n][0]), np.log(s0[n][1]), np.log(h0[n])]
        r = minimize(score, np.array(x0), method="Nelder-Mead",
                     options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-7, "adaptive": True})
        log(f"  yaw0 {np.degrees(yaw0):+4.0f}°: 目标 {-r.fun:.3f}")
        if best is None or r.fun < best.fun:
            best = r
    bl = unpack(best.x)
    res = {}
    for n in names:
        cx, cy, yaw, sx, sy, z0, h = bl[n]
        ious = {c: round(float(iou(box_mask(cams[c], box_corners(cx, cy, yaw, sx, sy, z0, h)), mk)), 3)
                for (c, nn), mk in masks.items() if nn == n}
        res[n] = {"center": [float(cx), float(cy), float(z0 + h / 2)], "yaw_rad": float(yaw),
                  "size": [float(sx), float(sy), float(h)], "iou": ious}
        log(f"  {n:6s} 中心 ({cx:.4f},{cy:.4f},{z0 + h/2:.4f})  yaw {np.degrees(yaw):+6.1f}°  "
            f"尺寸 {sx*1000:.1f}×{sy*1000:.1f}×{h*1000:.1f} mm  底 z {z0*1000:.1f}  IoU {ious}")

    # 可视化
    colors = {"arch": (0, 0, 255), "plank": (0, 255, 255), "cube": (255, 0, 255)}
    vis = []
    for c in cams:
        v = imgs[c].copy()
        for n in names:
            cx, cy, yaw, sx, sy, z0, h = bl[n]
            pm = box_mask(cams[c], box_corners(cx, cy, yaw, sx, sy, z0, h))
            cnt, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cnt, -1, colors[n], 1)
        vis.append(cv2.resize(v, (640, 360), interpolation=cv2.INTER_NEAREST))
    out_png = os.path.join(T, "stage1_scene", "stack_blocks_fit.png")
    cv2.imwrite(out_png, np.hstack(vis))
    log(f"  可视化: {out_png}（红=arch 黄=plank 紫=cube）")

    if a.apply:
        objs = scene["objects"]
        if "block_stack" in objs:
            objs.pop("block_stack")
            scene.setdefault("history", []).append({"block_stack": st, "note": "单箱体，已拆为 arch/plank/cube（§49）"})
        shapes = {"arch": "arch", "plank": "box", "cube": "box"}
        for n in names:
            objs[f"stack_{n}"] = dict(res[n], shape=shapes[n],
                                      source="三视角联合 IoU，链式支撑（droid_fit_stack_blocks.py）")
        json.dump(scene, open(sp, "w"), indent=2, ensure_ascii=False)
        log("  已写回 scene.json")


if __name__ == "__main__":
    main()
