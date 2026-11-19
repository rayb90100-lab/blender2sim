#!/usr/bin/env python3
"""渲染 Stage 2 物理搬运，并按源视频相机做 Real / MuJoCo 并排对比。

角度用 `source_view`：由 Stage 1 的 manifest 相机系位姿转到世界系，
与真实视频同一视角、同一内参（fx=974.07, fovy=66.6°, 相机高 193 mm，俯角 24.2°）。

用法:
  python tools/stage2_render_compare.py --testdir <testdir>
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_common import load_geometry  # noqa: E402

import cv2  # noqa: E402
import mujoco  # noqa: E402

FFMPEG = "/home/ubuntu/mambaforge/envs/E2E/bin/ffmpeg"
FPS = 25   # 仅在无物理采样时作回退


def log(m): print(m, flush=True)


def map_source_frame(t_sim, t_total, f_start, f_end):
    """物理时间 -> 源视频帧。同一事件对齐，不伪造源子帧。"""
    a = min(max(t_sim / t_total, 0.0), 1.0)
    return int(round(f_start + a * (f_end - f_start)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    a = ap.parse_args()
    T = a.testdir
    outdir = os.path.join(T, "stage2_physics", "validation")
    os.makedirs(outdir, exist_ok=True)

    # 懒导入，避免循环
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import stage2_validate_physics as V

    man, n_cam, d_cam, R_wc, c2w, d2w = load_geometry(T)
    m, d = V._build_transport_scene(T)
    oa = int(m.joint("obj_free").qposadr[0])
    p0 = d.qpos[oa:oa + 3].copy()

    box_half = m.geom("obj_geom").size[0]
    gap = 2 * box_half + 0.001
    bx = {s: float(m.body(f"plate_{s}").pos[0]) for s in "lr"}
    lx_tgt = (p0[0] - gap / 2) - bx["l"]
    rx_tgt = (p0[0] + gap / 2) - bx["r"]
    lx0 = float(d.qpos[m.joint("plate_l_x").qposadr[0]])
    rx0 = float(d.qpos[m.joint("plate_r_x").qposadr[0]])

    r = mujoco.Renderer(m, 1280, 720)
    frames = []
    meta = []

    def shot(tag):
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="source_view")
        frames.append(r.render().copy())
        meta.append({"tag": tag, "src_frame": map_source_frame(
            d.time, TOTAL_T, SRC_START, SRC_END), "t": float(d.time)})

    def run(n, tag, fn):
        for k in range(n):
            fn(k)
            mujoco.mj_step(m, d)
            if k % per_step == 0:
                shot(tag)

    # 源视频事件区间：抓取 f88 ~ 释放 f174
    SRC_START, SRC_END = 88, 174

    # 阶段时长：按真实视频的节奏定，不是随手给的整数。
    # 实测源视频（30 fps）核心动作 f84-174 = 3.00 s：
    #   抓取 0.27 s / 抬升 0.47 s / 搬运 1.13 s / 下降 0.87 s / 释放 ~0.2 s
    # 上一版总共 5.04 s，比真人慢 1.68 倍（用户指出"动作相比人手慢"）。
    LIFT, XFER = 0.10, 0.14
    # 按源视频实测帧数换算成物理步（timestep=0.002 s，30 fps 下 1 帧 = 0.0333 s = 16.7 步）：
    #   抓取 f84-92  8 帧 = 133 步   抬升 f92-106 14 帧 = 233 步
    #   搬运 f106-140 34 帧 = 567 步  下降 f140-166 26 帧 = 433 步
    #   释放 f166-174 8 帧 = 133 步   核心合计 90 帧 = 3.00 s = 1500 步
    stages = [("rest", 20), ("close", 100), ("squeeze", 33),
              ("lift", 233), ("transfer", 567), ("lower", 433),
              ("release", 133), ("settle", 20)]
    # 采样率 = 1/(per_step*timestep) = 1/0.004 = 250 fps 采，输出按 FPS 播放会太慢；
    # 所以输出帧率取 1/(per_step*timestep)，保证播放时长 = 物理时长
    per_step = 4          # 每 4 个物理步（0.008 s）采一帧 -> 125 fps 采样

    TOTAL_T = sum(n for _, n in stages) * m.opt.timestep

    N = dict(stages)
    log(f"物理总时长 {TOTAL_T:.2f} s（源视频核心动作 3.00 s），"
        f"采样后约 {sum(n // per_step + 1 for _, n in stages)} 帧")

    shot("rest")
    run(N["rest"], "rest", lambda k: None)
    run(N["close"], "close", lambda k: (d.ctrl.__setitem__(0, lx0 + (k + 1) / N["close"] * (lx_tgt - lx0)),
                                        d.ctrl.__setitem__(1, rx0 + (k + 1) / N["close"] * (rx_tgt - rx0))))
    run(N["squeeze"], "squeeze", lambda k: None)
    run(N["lift"], "lift", lambda k: (d.ctrl.__setitem__(2, LIFT * (k + 1) / N["lift"]),
                                      d.ctrl.__setitem__(3, LIFT * (k + 1) / N["lift"])))
    run(N["transfer"], "transfer", lambda k: (d.ctrl.__setitem__(0, lx_tgt + XFER * (k + 1) / N["transfer"]),
                                              d.ctrl.__setitem__(1, rx_tgt + XFER * (k + 1) / N["transfer"])))
    run(N["lower"], "lower", lambda k: (d.ctrl.__setitem__(2, LIFT * (1 - (k + 1) / N["lower"])),
                                        d.ctrl.__setitem__(3, LIFT * (1 - (k + 1) / N["lower"]))))
    run(N["release"], "release", lambda k: (d.ctrl.__setitem__(0, lx_tgt + XFER - 0.06 * (k + 1) / N["release"]),
                                            d.ctrl.__setitem__(1, rx_tgt + XFER + 0.06 * (k + 1) / N["release"])))
    run(N["settle"], "settle", lambda k: None)
    r.close()
    log(f"渲染 {len(frames)} 帧")

    # 真实视频
    cand = [os.path.join(T, "source.mp4"), os.path.join(T, "input", "source.mp4")]
    vid = next((c for c in cand if os.path.exists(c)), None)
    cap = cv2.VideoCapture(vid)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = 1.0 / (per_step * m.opt.timestep)
    log(f"输出帧率 {out_fps:.0f} fps，播放时长应为 {len(frames)/out_fps:.2f} s")
    vw = None
    strip = []
    for i, (img, mt) in enumerate(zip(frames, meta)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(mt["src_frame"], n_src - 1))
        ok, real = cap.read()
        if not ok:
            continue
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        row = np.hstack([real, np.full((real.shape[0], 8, 3), 40, np.uint8), bgr])
        cv2.putText(row, "REAL", (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(row, "MUJOCO (physics)", (real.shape[1] + 24, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 255), 3)
        cv2.putText(row, f"src f{mt['src_frame']:03d}  sim t={mt['t']:.2f}s",
                    (14, real.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(row, mt["tag"], (real.shape[1] + 24, real.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if vw is None:
            p = os.path.join(outdir, "compare_real_vs_mujoco.mp4")
            vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), out_fps,
                                 (row.shape[1], row.shape[0]))
            log(f"写出 {p}")
        vw.write(row)
        if i % max(1, len(frames) // 6) == 0:
            strip.append(row)
    cap.release()
    vw.release()

    raw = os.path.join(outdir, "compare_real_vs_mujoco.mp4")
    h264 = os.path.join(outdir, "compare_real_vs_mujoco_h264.mp4")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", raw, "-c:v", "libx264",
                    "-crf", "20", "-pix_fmt", "yuv420p", h264], check=True)
    log(f"H.264: {h264} ({os.path.getsize(h264)/1024/1024:.1f} MB)")

    if strip:
        h = 520
        sc = [cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h)) for x in strip]
        gp = os.path.join(outdir, "compare_real_vs_mujoco_grid.png")
        cv2.imwrite(gp, np.vstack(sc))
        log(f"对照图: {gp}")

    json.dump({"camera": "source_view (from Stage 1 manifest)",
               "frames": len(frames), "fps": out_fps,
               "src_range": [SRC_START, SRC_END],
               "time_mapping": "物理时间线性映射到源帧区间 [88,174]，同事件对齐，不伪造源子帧",
               "note": "等效接触对物理执行；机器人本体未建模（见 HANDOFF §4.2）"},
              open(os.path.join(outdir, "compare_manifest.json"), "w"), indent=2,
              ensure_ascii=False)


if __name__ == "__main__":
    main()
