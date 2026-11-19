#!/usr/bin/env python3
"""渲染两段视频（通用，只读 task_spec.json + 源 episode）：

  1. demo_replay_vs_source.mp4   DROID 关节真值在 r2pear 仿真里回放，与源视频逐帧对齐（15 fps）
                                 2×2：左=源视频（ext1 / wrist），右=MuJoCo（agentview / wrist）
                                 —— Real2Sim 对照：同一条机器人轨迹，仿真里的物体是否被同样地搬动
  2. oracle_solve.mp4            特权 oracle 经 PEAR 协议动作解题（可解性证明），agentview | wrist 并排

两段都用 H.264 / yuv420p 编码（浏览器、VS Code 预览都能播）。

用法: python tools/r2pear/render_episode.py --testdir <T> [--obj <target> --support <support>]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import controller as CT  # noqa: E402
import droid_common as C  # noqa: E402
import env as E  # noqa: E402
import oracle as O  # noqa: E402
import predicates as PR  # noqa: E402
import sim as S  # noqa: E402
import spec as SP  # noqa: E402

FFMPEG = os.environ.get("FFMPEG", "/home/ubuntu/mambaforge/envs/E2E/bin/ffmpeg")
FPS = 15


def encode(frames, path):
    """BGR 帧 → H.264 mp4。先写无损中间文件再转码，避免 OpenCV mp4v 在浏览器里不能播。"""
    import cv2
    h, w = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "raw.avi")
        vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", "-movflags", "+faststart", path], check=True)


def label(img, text, color=(0, 255, 255)):
    import cv2
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return img


class Cam:
    def __init__(self, m, w, h):
        import mujoco
        self.mj = mujoco
        self.r = mujoco.Renderer(m, h, w)

    def shot(self, d, cam):
        import cv2
        self.r.update_scene(d, camera=cam)
        return cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)      # Renderer 返回 RGB

    def close(self):
        self.r.close()


def read_video(path):
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def demo_replay(T, spec, out_path):
    import cv2
    import mujoco
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    J = ep["observation.state.joint_position"]
    G = ep["observation.state.gripper_position"].ravel()
    w, h = spec["render"]["width"], spec["render"]["height"]
    src = {k: read_video(os.path.join(T, "input", "videos", f"{spec['cameras'][k]['source_camera']}.mp4"))
           for k in ("agentview", "wrist")}
    m, d = S.build(spec)
    S.reset(m, d, spec)
    arm, g = S.arm_actuators(m), S.grip_actuator(m)
    sub = int(round(1 / FPS / m.opt.timestep))
    cam = Cam(m, w, h)
    frames = []
    n = min(len(J), *(len(v) for v in src.values()))
    for f in range(n):
        if f > 0:
            for k in range(sub):
                al = k / sub
                d.ctrl[arm] = (1 - al) * J[f - 1] + al * J[f]
                d.ctrl[g] = 255 * ((1 - al) * G[f - 1] + al * G[f])
                mujoco.mj_step(m, d)
        rows = []
        for k, name in (("agentview", "ext1"), ("wrist", "wrist")):
            s = cv2.resize(src[k][f], (w, h), interpolation=cv2.INTER_LINEAR)
            label(s, f"source {name}  f{f}")
            r = label(cam.shot(d, k), f"mujoco {k}  (DROID joints)")
            rows.append(np.hstack([s, r]))
        frames.append(np.vstack(rows))
    # 末尾 1 s 静置后判定，并把判定结果烧进最后几帧
    for _ in range(int(round(1.0 / m.opt.timestep))):
        mujoco.mj_step(m, d)
    ok, det = PR.evaluate(m, d, spec)
    tail = frames[-1].copy()
    txt = "SUCCESS" if ok else "FAIL: " + ",".join(f"{p['pred']}({p.get('obj', '')})" for p in det if not p["ok"])
    cv2.putText(tail, txt, (w + 8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(tail, txt, (w + 8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA)
    frames += [tail] * FPS
    cam.close()
    encode(frames, out_path)
    return {"frames": len(frames), "success_after_replay": ok,
            "failed": [f"{p['pred']}({p.get('obj', '')})" for p in det if not p["ok"]]}


def oracle_video(T, spec_path, obj, support, out_path):
    import cv2
    env = E.R2PEnv(spec_path, render=False)
    env.open()
    w, h = env.spec["render"]["width"], env.spec["render"]["height"]
    cam = Cam(env.m, w, h)
    frames, phase = [], {"name": "start", "i": 0}
    names = ["open_above", "descend", "close", "lift", "transfer", "lower", "release", "retreat"]

    def grab(_fmax=None):
        row = [label(cam.shot(env.d, c), f"{c}  [{phase['name']}]  step {env.used_steps}") for c in ("agentview", "wrist")]
        frames.append(np.hstack(row))

    grab()
    orig_move = CT.PoseController.move

    def move(self, budget, on_control_step=None):
        phase["name"] = names[min(phase["i"], len(names) - 1)]
        phase["i"] += 1
        return orig_move(self, budget, on_control_step=lambda f: grab(f))

    CT.PoseController.move = move
    try:
        O.solve(env, obj, support)
    finally:
        CT.PoseController.move = orig_move
    ok, det = PR.evaluate(env.m, env.d, env.spec)
    phase["name"] = "done: SUCCESS" if ok else "done: FAIL"
    for _ in range(FPS):
        grab()
    cam.close()
    env.close()
    encode(frames, out_path)
    return {"frames": len(frames), "control_steps": len(frames) - 1 - FPS, "success": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--obj", default=None)
    ap.add_argument("--support", default=None)
    a = ap.parse_args()
    T = a.testdir
    spec_path = os.path.join(T, "stage0_spec", "task_spec.json")
    spec = SP.load(spec_path)
    on = next(p for p in spec["success"] if p["pred"] == "on")
    obj, support = a.obj or on["obj"], a.support or on["support"]
    out = os.path.join(T, "stage4_review")
    os.makedirs(out, exist_ok=True)

    r1 = demo_replay(T, spec, os.path.join(out, "demo_replay_vs_source.mp4"))
    print("demo_replay_vs_source.mp4:", r1)
    r2 = oracle_video(T, spec_path, obj, support, os.path.join(out, "oracle_solve.mp4"))
    print("oracle_solve.mp4:", r2)
    json.dump({"demo_replay": r1, "oracle": r2}, open(os.path.join(out, "videos.json"), "w"), indent=1,
              ensure_ascii=False)


if __name__ == "__main__":
    main()
