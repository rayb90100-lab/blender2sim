#!/usr/bin/env python3
"""桌面矩形拟合（通用）：SAM2 桌面 mask（f0，各视角一个提示点）vs 桌面平面 z=table_z 上矩形 (cx, cy, yaw, sx, sy) 的投影四边形 IoU。
table_z 取自 scene.json（机器人约束 + 物体拟合）。物体 mask 从桌面 mask 中挖去前已并入（SAM2 桌面 mask 通常不含物体，投影四边形含 → 并入物体 mask 后再比）。
输出 stage1_scene/table.json、table_fit.png

用法: python tools/r2pear/droid_stage1_table.py --testdir <T> --scene scene_ext2_wrist.json --prompt "exterior_2_left:90,130;40,40" --prompt "wrist_left:250,70;160,170"
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

SAM2_CFG = "configs/sam2/sam2_hiera_l.yaml"
SAM2_W = "/media/data1/RoboLab_data/weights/sam2/sam2_hiera_large.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--prompt", action="append", required=True, help="cam:u,v[;u,v 负点 ...]，第一个为正点")
    ap.add_argument("--fix_sx", type=float, default=None,
                    help="x 向（前后）尺寸不可观测时固定（如近机器人桌沿出画/被夹爪挡），记录来源")
    ap.add_argument("--fix_sx_source", default="")
    a = ap.parse_args()
    import cv2
    import torch
    from scipy.optimize import minimize
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    T = a.testdir
    sd = os.path.join(T, "stage1_scene")
    scene = json.load(open(os.path.join(sd, a.scene)))
    objm = dict(np.load(os.path.join(sd, "masks.npz")))
    cams = DC.Cams(T)
    tz = scene["table_z"]
    pred = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_W, device="cuda"))
    masks, imgs = {}, {}
    for pr in a.prompt:
        cam, pts = pr.split(":")
        pts = [np.array([float(x) for x in q.split(",")], np.float32) for q in pts.split(";")]
        uv = np.array(pts, np.float32)
        lab = np.array([1] + [0] * (len(pts) - 1), np.int32)
        cap = cv2.VideoCapture(os.path.join(T, "input", "videos", f"{cam}.mp4"))
        _, bgr = cap.read()
        imgs[cam] = bgr
        pred.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            ms, ss, _ = pred.predict(point_coords=uv, point_labels=lab, multimask_output=True)
        mk = ms[int(np.argmax([m.sum() for m in ms]))].astype(bool)          # 负点排除纸板/夹爪后取最大者（整张桌面）
        for k, v in objm.items():
            if k.endswith("|" + cam):
                mk |= v
        masks[cam] = mk

    def quad(p):
        cx, cy, yaw, sx, sy = p
        c, s = math.cos(yaw), math.sin(yaw)
        loc = np.array([[sx / 2, sy / 2], [-sx / 2, sy / 2], [-sx / 2, -sy / 2], [sx / 2, -sy / 2]])
        xy = loc @ np.array([[c, s], [-s, c]]) + [cx, cy]
        return np.c_[xy, np.full(4, tz)]

    def poly_mask(cam, P):
        Rcv, t = cams.pose_cv(cam, 0)
        # 细分边以处理部分在相机后方的情况
        pts = []
        for i in range(4):
            for u in np.linspace(0, 1, 20, endpoint=False):
                pts.append(P[i] * (1 - u) + P[(i + 1) % 4] * u)
        pts = np.array(pts)
        pc = (pts - t) @ Rcv
        pc = pc[pc[:, 2] > 0.05]
        f = cams.fx(cam)
        uv = np.stack([160 + f * pc[:, 0] / pc[:, 2], 90 + f * pc[:, 1] / pc[:, 2]], 1)
        mk = np.zeros((180, 320), np.uint8)
        if len(uv) >= 3:
            cv2.fillPoly(mk, [np.round(uv * 4).astype(np.int32)], 1, shift=2)
        return mk.astype(bool)

    def full(p):
        return p if a.fix_sx is None else np.array([p[0], p[1], p[2], a.fix_sx, p[3]])

    def loss(p):
        p = full(p)
        tot = 0.0
        for cam, mk in masks.items():
            pm = poly_mask(cam, quad(p))
            tot += (pm & mk).sum() / max((pm | mk).sum(), 1)
        return -tot / len(masks)

    objs = scene["objects"]
    c0 = np.mean([o["center"][:2] for o in objs.values()], 0)
    best = None
    for yaw0 in (0.0, 0.3, -0.3):
        for s0 in (0.5, 0.7):
            x0 = [c0[0] + 0.1, c0[1], yaw0, s0, s0] if a.fix_sx is None else [c0[0] + 0.1, c0[1], yaw0, s0]
            r = minimize(loss, x0, method="Powell", options={"maxiter": 8000, "xtol": 1e-4})
            if best is None or r.fun < best.fun:
                best = r
    best.x = full(best.x)
    cx, cy, yaw, sx, sy = best.x
    per = {cam: round(float((poly_mask(cam, quad(best.x)) & mk).sum() / max((poly_mask(cam, quad(best.x)) | mk).sum(), 1)), 3)
           for cam, mk in masks.items()}
    print(f"table center ({cx:.3f},{cy:.3f}) yaw {math.degrees(yaw):.1f}° size {abs(sx):.3f}×{abs(sy):.3f} m  IoU {per}")
    json.dump({"center_xy": [float(cx), float(cy)], "size_xy": [abs(float(sx)), abs(float(sy))], "yaw_rad": float(yaw),
               "table_z": tz, "apron_m": 0.12, "iou": per,
               "fixed_sx": a.fix_sx, "fixed_sx_source": a.fix_sx_source,
               "method": "SAM2 桌面 mask（并入物体 mask，负点排除纸板/夹爪）vs 桌面平面矩形投影 IoU，Powell；table_z 来自 " + a.scene},
              open(os.path.join(sd, "table.json"), "w"), indent=1, ensure_ascii=False)
    rows = []
    for cam, mk in masks.items():
        v = imgs[cam].copy()
        v[mk] = (0.6 * v[mk] + [0, 80, 0]).astype(np.uint8)
        cnt, _ = cv2.findContours(poly_mask(cam, quad(best.x)).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(v, cnt, -1, (255, 0, 255), 1)
        rows.append(cv2.resize(v, (640, 360)))
    cv2.imwrite(os.path.join(sd, "table_fit.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
