#!/usr/bin/env python3
"""Stage 1 分割（通用）：按 objects.json 给每个物体、每个视角生成 SAM2 mask（f0）。
提示：本物体为正点；其他物体的提示点为负点（叠放物体互相区分）。多 mask 输出中取「面积 ≤ max_area 且得分最高」者。
输出 stage1_scene/masks.npz（key = <obj>|<cam>）、masks_overlay.png、masks.json（得分/面积/提示）

用法: python tools/r2pear/droid_stage1_masks.py --testdir <T>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import droid_cams as DC  # noqa: E402

SAM2_CFG = "configs/sam2/sam2_hiera_l.yaml"
SAM2_W = "/media/data1/RoboLab_data/weights/sam2/sam2_hiera_large.pt"
CAMS = ("exterior_1_left", "exterior_2_left", "wrist_left")
MAX_AREA = {"exterior_1_left": 1500, "exterior_2_left": 1500, "wrist_left": 6000}
COL = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 0), (0, 0, 255), (255, 255, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    a = ap.parse_args()
    import cv2
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    T = a.testdir
    cfg = json.load(open(os.path.join(T, "stage1_scene", "objects.json")))
    cams = DC.Cams(T)
    fr = cfg["frame"]
    pin = {k: cams.pinch(v) for k, v in cfg["events"].items()}

    def prompt(o, cam):
        p = o["prompt"]
        if "manual" in p:
            return np.array(p["manual"][cam], float) if cam in p["manual"] else None
        P = pin[p["robot"]] + np.array([0, 0, p["dz"]])
        uv = cams.project(cam, P, fr)[0]
        return None if np.isnan(uv).any() else uv

    pred = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_W, device="cuda"))
    masks, info, rows = {}, {}, []
    for cam in CAMS:
        cap = cv2.VideoCapture(os.path.join(T, "input", "videos", f"{cam}.mp4"))
        cap.set(1, fr)
        ok, bgr = cap.read()
        pred.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pts = {n: prompt(o, cam) for n, o in cfg["objects"].items()}
        vis = bgr.copy()
        for i, (n, o) in enumerate(cfg["objects"].items()):
            if pts[n] is None:
                continue
            neg = [v for k, v in pts.items() if k != n and v is not None and np.linalg.norm(v - pts[n]) > 4]
            coords = np.array([pts[n]] + neg, np.float32)
            labels = np.array([1] + [0] * len(neg), np.int32)
            with torch.inference_mode():
                ms, ss, _ = pred.predict(point_coords=coords, point_labels=labels, multimask_output=True)
            cand = [(float(s), m.astype(bool)) for s, m in zip(ss, ms) if 15 <= m.sum() <= MAX_AREA[cam]]
            if not cand:
                info[f"{n}|{cam}"] = {"status": "rejected_no_mask_within_area", "prompt": pts[n].tolist()}
                continue
            s, mk = max(cand, key=lambda t: t[0])
            masks[f"{n}|{cam}"] = mk
            info[f"{n}|{cam}"] = {"status": "ok", "score": round(s, 3), "area": int(mk.sum()), "prompt": pts[n].round(1).tolist()}
            vis[mk] = (0.45 * vis[mk] + 0.55 * np.array(COL[i % len(COL)])).astype(np.uint8)
            cv2.circle(vis, tuple(int(x) for x in pts[n]), 2, (255, 255, 255), -1)
        v = cv2.resize(vis, (640, 360), interpolation=cv2.INTER_NEAREST)
        cv2.putText(v, cam, (5, 18), 0, 0.55, (0, 255, 255), 2)
        rows.append(v)
    od = os.path.join(T, "stage1_scene")
    np.savez_compressed(os.path.join(od, "masks.npz"), **masks)
    json.dump(info, open(os.path.join(od, "masks.json"), "w"), indent=1)
    cv2.imwrite(os.path.join(od, "masks_overlay.png"), np.vstack(rows))
    for k, v in info.items():
        print(k, v)


if __name__ == "__main__":
    main()
