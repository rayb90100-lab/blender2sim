#!/usr/bin/env python3
"""生成 Real / Blender 并排对比视频与抽帧对照图。

用法:
  python tools/make_compare_video.py --testdir <testdir> [--every 1]

前提: stage1_animation/f%04d.png 已渲染，source.mp4 可解码。
输出:
  stage1_animation/compare_real_vs_blender.mp4       并排视频
  stage1_animation/compare_grid.png                  抽帧对照图
"""
import argparse
import glob
import json
import os
import subprocess

import cv2
import numpy as np

FFMPEG = "/home/ubuntu/mambaforge/envs/E2E/bin/ffmpeg"


def log(m): print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--grid-frames", type=int, default=7)
    a = ap.parse_args()
    T = a.testdir
    ad = os.path.join(T, "stage1_animation")

    man_path = os.path.join(ad, "animation_manifest.json")
    stride = 1
    if os.path.exists(man_path):
        stride = int(json.load(open(man_path)).get("stride", 1))

    sim = sorted(glob.glob(os.path.join(ad, "f*.png")))
    if not sim:
        raise FileNotFoundError(f"{ad} 下没有渲染帧")
    log(f"渲染帧: {len(sim)}  (stride={stride})")

    cand = [os.path.join(T, "source.mp4"),
            os.path.join(T, "input", "source.mp4"),
            os.path.join(T, "input", "source_hevc.mp4")]
    vid = next((c for c in cand if os.path.exists(c)), None)
    if vid is None:
        raise FileNotFoundError(f"未找到视频: {cand}")
    cap = cv2.VideoCapture(vid)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = src_fps / stride
    log(f"视频: {n_src} 帧 @ {src_fps:.1f}fps -> 对比 {fps:.1f}fps")

    # ---- 并排视频 ----
    vw = None
    grid = []
    picks = np.linspace(0, len(sim) - 1, a.grid_frames).astype(int).tolist()
    for i, s in enumerate(sim):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * stride, n_src - 1))
        ok, real = cap.read()
        im = cv2.imread(s)
        if not ok or im is None:
            continue
        if real.shape[:2] != im.shape[:2]:
            real = cv2.resize(real, (im.shape[1], im.shape[0]))
        row = np.hstack([real, np.full((im.shape[0], 8, 3), 40, np.uint8), im])
        cv2.putText(row, "REAL", (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(row, "BLENDER (motion-fit preview)",
                    (im.shape[1] + 24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 255), 3)
        cv2.putText(row, f"src frame {i * stride:03d}", (14, im.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if vw is None:
            p = os.path.join(ad, "compare_real_vs_blender.mp4")
            vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (row.shape[1], row.shape[0]))
            log(f"写出 {p}")
        vw.write(row)
        if i in picks:
            grid.append(row)
    cap.release()
    if vw:
        vw.release()
    log(f"并排视频: {len(sim)} 帧")

    # ---- H.264 重编码（浏览器/播放器兼容）----
    raw = os.path.join(ad, "compare_real_vs_blender.mp4")
    h264 = os.path.join(ad, "compare_real_vs_blender_h264.mp4")
    if os.path.exists(raw):
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", raw,
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", h264],
                       check=True)
        log(f"H.264: {h264} ({os.path.getsize(h264)/1024/1024:.1f} MB)")

    # ---- 抽帧对照图 ----
    if grid:
        h = 560
        sc = [cv2.resize(r, (int(r.shape[1] * h / r.shape[0]), h)) for r in grid]
        gp = os.path.join(ad, "compare_grid.png")
        cv2.imwrite(gp, np.vstack(sc))
        log(f"对照图: {gp}")


if __name__ == "__main__":
    main()
