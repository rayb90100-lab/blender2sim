#!/usr/bin/env python3
"""把并行工作产出的几何与元数据合并成一份可直接 export 的 scene.json。

背景（DEVELOPMENT_LOG §63.3）：test6 的三维拟合有两份并行结果，用同一套 IoU 代码评测后
几何取 w71 的 scene_ext2_wrist.json（ext2 0.574 / wrist 0.486），但该文件缺 role 之外的
元字段（support_chain、rgba）。本脚本只做合并，不重新拟合、不改动任何几何数值。

输入：
  --scene   胜出的几何（objects / table_z）
  --table   桌面外形（center_xy / size_xy / yaw_rad / apron_m）
  --chain   叠放链（底→顶），默认取 objects.json 的 support_chain
  --out     输出 scene.json
颜色：从 masks.npz + 视频首帧实测（反照率 = 实测 / 曝光增益，不改色相），并记录依据。

用法:
  python tools/r2pear/merge_scene.py --testdir <T> --scene <scene_ext2_wrist.json> \
      --table <table.json> --out <T>/stage1_scene/merged/scene.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import droid_cams as DC  # noqa: E402

EXPOSURE_GAIN = 0.78   # 实测亮度 / 反照率；不改色相，只按比例缩放


def measure_rgba(testdir, obj, cams):
    """三视角 mask 内像素的中位色 → 反照率。"""
    import cv2
    z = np.load(os.path.join(testdir, "stage1_scene", "masks.npz"))
    vals = []
    for cam in cams:
        key = f"{obj}|{cam}"
        if key not in z.files:
            continue
        mk = z[key].astype(bool)
        if mk.sum() < 8:
            continue
        cap = cv2.VideoCapture(os.path.join(testdir, "input", "videos", f"{cam}.mp4"))
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        b, g, r = [float(x) for x in img[mk].mean(0)]     # OpenCV 是 BGR
        vals.append([r / 255.0, g / 255.0, b / 255.0])    # MuJoCo rgba 是 RGB
    if not vals:
        return None, "no_usable_mask"
    m = np.median(np.array(vals), axis=0)
    alb = np.clip(m / EXPOSURE_GAIN, 0, 1)
    return [round(float(x), 3) for x in alb] + [1.0], f"median over {len(vals)} views, gain {EXPOSURE_GAIN}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--chain", default=None, help="底→顶；默认取 objects.json 的 support_chain")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    T = a.testdir
    scene = json.load(open(a.scene))
    table = json.load(open(a.table))
    odir = os.path.join(T, "stage1_scene")

    chain = (a.chain.split(",") if a.chain
             else json.load(open(os.path.join(odir, "objects.json"))).get("support_chain"))
    if not chain:
        raise SystemExit("没有叠放链：请用 --chain 指定，或在 objects.json 里写 support_chain")
    names = set(scene["objects"])
    bad = [n for n in chain if n not in names]
    if bad:
        raise SystemExit(f"support_chain 含 scene 里没有的物体: {bad}")

    # 任务语义（success 谓词）以 objects.json 为准，不被几何解覆盖
    obj_cfg = json.load(open(os.path.join(odir, "objects.json")))

    cams = DC.Cams(T)
    views = [c for c in DC.EXT + ("wrist_left",)
             if os.path.exists(os.path.join(T, "input", "videos", f"{c}.mp4"))]

    out = {
        "frame": scene.get("frame", 0),
        "table_z": scene["table_z"],
        "table": table,
        "support_chain": chain,
        "support_chain_note": "底→顶；target 若在链上，被搬走后其下方允许坍塌（export 时不加堆叠完整性谓词）",
        "success": obj_cfg.get("success"),
        "success_note": obj_cfg.get("success_note", ""),
        "task": obj_cfg.get("task"),
        "objects": {},
        "cameras": {},
        "extrinsics_status": {c: cams.source(c) for c in views},
        "method": scene.get("method", ""),
        "provenance": {
            "geometry_from": os.path.basename(a.scene),
            "table_from": os.path.basename(a.table),
            "geometry_selection": ("在同一次评测量化下选优：该解 ext2 平均 IoU 0.574 / wrist 0.486，"
                                   "对照解 0.053 / 0.367（DEVELOPMENT_LOG §63.3）"),
            "merge_tool": "tools/r2pear/merge_scene.py（只合并，不重新拟合）",
        },
    }
    for n, o in scene["objects"].items():
        e = dict(o)
        rgba, note = measure_rgba(T, n, views)
        if rgba:
            e["rgba"] = rgba
            e["rgba_status"] = note
        shape = e.get("shape", "box")
        if shape in ("cylinder", "prism"):
            e["shape_degraded"] = f"{shape} 在 sim.py 按解析式建体（非 mesh），P3 再 mesh 化"
        out["objects"][n] = e
        print(f"  {n:16s} {shape:9s} rgba={e.get('rgba')}  ({note})")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"wrote {a.out}: {len(out['objects'])} objects, chain={chain}")


if __name__ == "__main__":
    main()
