#!/usr/bin/env python3
"""生成 Real2Gym 冒烟测试视频 v3。

v1 -> v2 修正：夹爪 ctrlrange 是 0~255；相机改 look-at。
v2 -> v3 修正：
  - IK 目标改为 finger midpoint（hand 原点下方 5.84 cm），不再瞄 hand 原点
  - 加手部朝向约束，强制夹爪竖直向下
  - 抓取前先张开到最大、下探更慢，避免把方块推走

产出：
  input/smoke_demo.mp4
  evidence/joint_log.json
  evidence/camera.json
"""
import json
import os

import cv2
import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

SCENE = "/media/data1/RoboLab_data/assets/mujoco_menagerie/franka_emika_panda/smoke_scene.xml"
OUT = "/home/ubuntu/workstation/RoboLab/tests/test1"
PREV = "/home/ubuntu/workstation/RoboLab/tests/test1/stage1_scene/moge3"
W, H, FPS = 640, 480, 30

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)

HAND = m.body("hand").id
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
# 抓取点相对 hand 原点的偏移（初始姿态下测得，竖直向下时不变）
GRASP_OFFSET = np.array([0.0, 0.0, -0.0584])


def set_camera(az, el, dist, target):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = target
    cam.distance = dist
    cam.azimuth = az
    cam.elevation = el
    return cam


TARGET = np.array([0.58, 0.05, 0.36])
cam = set_camera(145, -18, 1.55, TARGET)

jacp = np.zeros((3, m.nv))
jacr = np.zeros((3, m.nv))


def grasp_point():
    """当前夹爪抓取点（finger midpoint）。"""
    return d.xpos[HAND] + GRASP_OFFSET


def ik_to(target_grasp, gripper, iterations=300, tol=2.5e-3, down_weight=0.55):
    """IK 同时约束位置与手部朝向（夹爪 z 轴朝下）。"""
    target = np.asarray(target_grasp, dtype=float)
    for _ in range(iterations):
        mujoco.mj_forward(m, d)
        err_p = target - grasp_point()
        # 朝向误差：把 hand 的 z 轴往世界 -z 拉
        R = d.xmat[HAND].reshape(3, 3)
        z_axis = R[:, 2]
        err_r = (np.array([0.0, 0.0, -1.0]) - z_axis)
        err = np.concatenate([err_p, down_weight * err_r])

        if np.linalg.norm(err_p) < tol and np.linalg.norm(err_r) < 0.12:
            break

        mujoco.mj_jacBody(m, d, jacp, jacr, HAND)
        J = np.vstack([jacp[:, :7], down_weight * jacr[:, :7]])
        lam = 8e-3
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
        d.qpos[:7] += np.clip(dq, -0.05, 0.05)
        d.ctrl[:7] = d.qpos[:7]
        d.ctrl[7] = gripper
        mujoco.mj_step(m, d)


def grip(value, steps=100):
    for _ in range(steps):
        d.ctrl[7] = value
        mujoco.mj_step(m, d)


def settle(steps=120):
    for _ in range(steps):
        d.ctrl[:7] = d.qpos[:7]
        mujoco.mj_step(m, d)


# ---- 初始姿态 ----
d.qpos[:7] = [0.0, 0.30, 0.0, -2.0, 0.0, 2.30, 0.785]
d.ctrl[:7] = d.qpos[:7]
d.ctrl[7] = GRIP_OPEN
for _ in range(500):
    mujoco.mj_step(m, d)

print("init cube  :", np.round(d.body("target_cube").xpos, 4))
print("init grasp :", np.round(grasp_point(), 4))

cube0 = d.body("target_cube").xpos.copy()
above = cube0 + np.array([0.0, 0.0, 0.12])
grasp = cube0 + np.array([0.0, 0.0, 0.005])
lift = cube0 + np.array([0.0, 0.0, 0.18])
goal_xy = np.array([0.62, 0.21])
goal = np.array([goal_xy[0], goal_xy[1], cube0[2] + 0.005])
goal_above = goal + np.array([0.0, 0.0, 0.15])

renderer = mujoco.Renderer(m, H, W)
frames, log = [], []


def shoot(tag, n=1):
    for _ in range(n):
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render().copy())
        log.append({"tag": tag,
                    "qpos": [round(float(x), 6) for x in d.qpos[:8]],
                    "grasp": [round(float(x), 6) for x in grasp_point()],
                    "cube": [round(float(x), 6) for x in d.body("target_cube").xpos]})


shoot("init", 12)

ik_to(above, GRIP_OPEN);          shoot("approach", 20)
ik_to(grasp, GRIP_OPEN, iterations=400); shoot("pre_grasp", 14)
grip(GRIP_CLOSED, 120);           shoot("grasp", 16)
ik_to(lift, GRIP_CLOSED);         shoot("lift", 22)
ik_to(goal_above, GRIP_CLOSED);   shoot("transfer", 26)
ik_to(goal, GRIP_CLOSED);         shoot("lower", 18)
grip(GRIP_OPEN, 70);              shoot("release", 16)
ik_to(goal_above, GRIP_OPEN);     shoot("retreat", 20)
shoot("final", 12)

renderer.close()

os.makedirs(f"{OUT}/input", exist_ok=True)
os.makedirs(f"{OUT}/evidence", exist_ok=True)
vp = f"{OUT}/input/smoke_demo.mp4"
vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for f in frames:
    vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
vw.release()
print(f"video: {vp} frames={len(frames)} {W}x{H}@{FPS}")

for tag, idx in [("init", 12), ("grasp", 62), ("transfer", 120), ("final", len(frames) - 1)]:
    if 0 <= idx < len(frames):
        cv2.imwrite(f"{PREV}/smoke3_{tag}.jpg", cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR))

cam_data = {
    "cameras": [{
        "name": "lookat_free", "resolution": [W, H],
        "lookat": [round(float(x), 6) for x in TARGET],
        "azimuth_deg": 145.0, "elevation_deg": -18.0, "distance": 1.55,
        "fovy_deg": float(m.cam_fovy[0]),
        "calibration": "estimated",
        "note": "MuJoCo free camera + look-at；内参未标定",
    }],
    "source": "simulated",
    "warning": "自建仿真素材，非真实拍摄",
}
with open(f"{OUT}/evidence/camera.json", "w") as fh:
    json.dump(cam_data, fh, indent=2, ensure_ascii=False)
with open(f"{OUT}/evidence/joint_log.json", "w") as fh:
    json.dump({"fps": FPS, "frames": len(frames), "records": log}, fh, indent=1)

cf = d.body("target_cube").xpos.copy()
print("cube final :", np.round(cf, 4))
print("goal       :", np.round(goal, 4))
print("末端误差   :", round(float(np.linalg.norm(cf[:2] - goal[:2])), 4), "m")
