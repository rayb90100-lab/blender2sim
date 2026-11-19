#!/usr/bin/env python3
"""Stage 4 观测检查（通用）：MuJoCo 渲染 vs 源视频，出对照图与静置视频。

为什么需要（DEVELOPMENT_LOG §63.5）：拟合叠加图只能验证「投影对不对」，
渲染图才能验证「模型建出来是不是那个东西」。test4/test5 有这一环，test6 一度跳过。

输出（默认 <testdir>/stage4_review/）：
  render_vs_source_<cam>.png   左=源视频首帧，右=MuJoCo 渲染，下=差异概览
  render_three_views.png       两路渲染并排 + 源帧
  settle_front.mp4             静置 3 s 视频（agentview）
  settle_metrics.json          每物体位姿漂移、速度衰减曲线

用法: python tools/r2pear/review_render.py --testdir <T> [--seconds 3] [--no_video]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import spec as SP  # noqa: E402
import sim as S  # noqa: E402
import predicates as PR  # noqa: E402

W_SRC, H_SRC = 320, 180


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--seconds", type=float, default=3.0, help="静置时长")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--no_video", action="store_true")
    a = ap.parse_args()
    import cv2
    import mujoco

    T = a.testdir
    spec_path = os.path.join(T, "stage0_spec", "task_spec.json")
    out = os.path.join(T, "stage4_review")
    os.makedirs(out, exist_ok=True)
    sp = SP.load(spec_path)

    m, d = S.build(sp)
    S.reset(m, d, sp)
    # 静置到稳定（与 env.open 的 settle 一致）
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)

    rw, rh = sp["render"]["width"], sp["render"]["height"]
    renderer = mujoco.Renderer(m, rh, rw)

    def shot(cam):
        renderer.update_scene(d, camera=cam)
        # Renderer.render() 返回 **RGB**；本脚本其余部分（读视频、imwrite）都用 BGR 约定，
        # 不转换会让整张图红蓝互换（曾把红棱柱渲染成蓝色）。
        return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

    # ---- 1. 逐相机 渲染 vs 源视频 ----
    rows = []
    for cam in sp["cameras"]:
        src_cam = sp["cameras"][cam].get("source_camera", cam)
        vp = os.path.join(T, "input", "videos", f"{src_cam}.mp4")
        if not os.path.exists(vp):
            print(f"  跳过 {cam}: 没有源视频 {vp}")
            continue
        cap = cv2.VideoCapture(vp)
        ok, src = cap.read()
        cap.release()
        if not ok:
            print(f"  跳过 {cam}: 读帧失败")
            continue
        ren = shot(cam)
        # 源帧放大到渲染尺寸，便于对比
        src_big = cv2.resize(src, (rw, rh), interpolation=cv2.INTER_NEAREST)
        diff = cv2.absdiff(cv2.cvtColor(src_big, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(ren, cv2.COLOR_BGR2GRAY))
        panel = np.hstack([src_big, ren])
        cv2.putText(panel, f"source {src_cam}", (8, 22), 0, 0.6, (0, 255, 255), 2)
        cv2.putText(panel, f"mujoco {cam}", (rw + 8, 22), 0, 0.6, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(out, f"render_vs_source_{cam}.png"), panel)
        rows.append(panel)
        print(f"  写出 render_vs_source_{cam}.png  （平均灰度差 {diff.mean():.1f}）")

    if rows:
        cv2.imwrite(os.path.join(out, "render_three_views.png"), np.vstack(rows))

    # ---- 2. 静置漂移与稳定性 ----
    names = list(sp["objects"])
    start = {n: d.qpos[m.joint(f"{n}_free").qposadr[0]:m.joint(f"{n}_free").qposadr[0] + 3].copy()
             for n in names}
    traj = {n: [] for n in names}
    speeds = {n: [] for n in names}
    steps = int(a.seconds / m.opt.timestep)
    per_frame = max(1, int(round(1.0 / (a.fps * m.opt.timestep))))
    frames = []
    for i in range(steps):
        mujoco.mj_step(m, d)
        if i % per_frame == 0:
            frames.append(shot("agentview"))
        for n in names:
            j = m.joint(f"{n}_free")
            qa, da = j.qposadr[0], j.dofadr[0]
            traj[n].append(float(np.linalg.norm(d.qpos[qa:qa + 3] - start[n]) * 1000))
            speeds[n].append(float(np.linalg.norm(d.qvel[da:da + 6])))

    metrics = {"seconds": a.seconds, "timestep": float(m.opt.timestep),
               "noslip_iterations": int(m.opt.noslip_iterations), "objects": {}}
    print(f"\n静置 {a.seconds:.1f} s：")
    for n in names:
        tr, spd = np.array(traj[n]), np.array(speeds[n])
        tail = spd[len(spd) // 2:]
        metrics["objects"][n] = {
            "max_drift_mm": round(float(tr.max()), 4),
            "final_drift_mm": round(float(tr[-1]), 4),
            "speed_first_half_max": round(float(spd[:len(spd) // 2].max()), 5),
            "speed_tail_max": round(float(tail.max()), 5),
            "decaying": bool(tail.max() < spd[:len(spd) // 2].max() * 0.9),
        }
        e = metrics["objects"][n]
        print(f"  {n:15s} 漂移 max {e['max_drift_mm']:6.3f} mm 末 {e['final_drift_mm']:6.3f} mm"
              f"  速度 前半max {e['speed_first_half_max']:.5f} 后半max {e['speed_tail_max']:.5f}"
              f"  {'衰减✓' if e['decaying'] else '不衰减 ✗'}")

    ok, detail = PR.evaluate(m, d, sp)
    metrics["success_at_end"] = bool(ok)
    metrics["predicates"] = detail
    mn = min(d.contact[i].dist for i in range(d.ncon)) if d.ncon else 0.0
    metrics["max_penetration_mm"] = round(float(mn) * 1000, 4)
    print(f"\n  最深穿透 {metrics['max_penetration_mm']:+.2f} mm   终态 success={ok}")

    # ---- 3. 静置视频 ----
    if not a.no_video and frames:
        vp = os.path.join(out, "settle_front.mp4")
        vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (rw, rh))
        for f in frames:
            vw.write(f)
        vw.release()
        metrics["settle_video"] = os.path.basename(vp)
        print(f"  写出 {vp}（{len(frames)} 帧 @ {a.fps} fps）")

    renderer.close()
    json.dump(metrics, open(os.path.join(out, "settle_metrics.json"), "w"), indent=1, ensure_ascii=False)
    print(f"写出 {out}/settle_metrics.json")


if __name__ == "__main__":
    main()
