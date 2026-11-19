#!/usr/bin/env python3
"""DROID 相机内参拟合：数据集给了外参但没给内参。

做法：
  1. 关节真值 + 外参（欧拉 xyz，OpenCV 相机系）→ MuJoCo 渲染机器人分割 mask
  2. 用 FK 投影出的 pinch / link7 / link6 像素作为 SAM2 正提示，分割真实画面里的机器人
  3. 在 fx 网格上取「渲染 mask vs SAM2 mask」IoU 最大者（cx, cy 取画面中心，fx = fy）

已验证（2026-11-17, ep17）：
  - FK(joint_position) 的法兰位置 与 cartesian_position 差 0.0 mm
  - 欧拉约定 xyz（外旋）：姿态差恒为 135°（Panda attachment 安装角），std 0

用法:
  python tools/droid_fit_intrinsics.py --testdir <testdir> [--frames 75 90 120 140]
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
W, H = 320, 180


def cam_quat_mj(rpy):
    from scipy.spatial.transform import Rotation as R
    Rmj = C.euler_to_R(rpy, "xyz") @ np.diag([1, -1, -1])   # OpenCV -> MuJoCo 相机轴
    q = R.from_matrix(Rmj).as_quat()
    return [q[3], q[0], q[1], q[2]]


def render_mask(ep, cam, frame, fx, ss=3):
    import mujoco
    import cv2
    ext = ep[f"camera_extrinsics.{cam}"][frame]
    spec = C.build_spec()
    c = spec.worldbody.add_camera(name="cam", pos=ext[:3], quat=cam_quat_mj(ext[3:]))
    c.fovy = float(np.degrees(2 * np.arctan(H / 2 / fx)))
    spec.visual.global_.offwidth = W * ss
    spec.visual.global_.offheight = H * ss
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[C.arm_qadr(m)] = ep["observation.state.joint_position"][frame]
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, H * ss, W * ss)
    r.enable_segmentation_rendering()
    r.update_scene(d, camera="cam")
    seg = r.render()[..., 0]
    r.close()
    return cv2.resize((seg >= 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def project_points(ep, cam, frame, fx, names=("g_pinch", "attachment_site")):
    import mujoco
    m, d, _ = C.build_model()
    d.qpos[C.arm_qadr(m)] = ep["observation.state.joint_position"][frame]
    mujoco.mj_kinematics(m, d)
    ext = ep[f"camera_extrinsics.{cam}"][frame]
    Rc = C.euler_to_R(ext[3:], "xyz")
    pts = [d.site_xpos[m.site(n).id] for n in names]
    # 外部相机里通常只有夹爪入画（机械臂从画面上方伸入），所以提示点取夹爪自身部件
    pts += [d.xpos[m.body(b).id] for b in ("g_left_pad", "g_right_pad", "g_base", "link7")]
    uv = []
    for p in pts:
        pc = Rc.T @ (p - ext[:3])
        if pc[2] <= 0:
            continue
        u, v = fx * pc[0] / pc[2] + W / 2, fx * pc[1] / pc[2] + H / 2
        if 2 <= u < W - 2 and 2 <= v < H - 2:
            uv.append([u, v])
    return np.array(uv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[75, 90, 120, 140])
    ap.add_argument("--fx0", type=float, default=145.0)
    a = ap.parse_args()
    import cv2
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    T = a.testdir
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    pred = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_W, device="cuda"))
    out_dir = os.path.join(T, "stage0_replay")
    os.makedirs(out_dir, exist_ok=True)

    grid = np.arange(100, 201, 2.5)
    result = {}
    vis_rows = []
    for cam in ("exterior_1_left", "exterior_2_left"):
        cap = cv2.VideoCapture(os.path.join(T, "input", "videos", f"{cam}.mp4"))
        scores = np.zeros(len(grid))
        used = 0
        for fr in a.frames:
            cap.set(1, fr)
            ok, bgr = cap.read()
            if not ok:
                continue
            uv = project_points(ep, cam, fr, a.fx0)
            if len(uv) < 2:
                print(f"  {cam} f{fr}: 可见提示点 {len(uv)} 个，跳过")
                continue
            pred.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            with torch.inference_mode():
                ms, ss_, _ = pred.predict(point_coords=uv.astype(np.float32),
                                          point_labels=np.ones(len(uv), np.int32),
                                          multimask_output=False)
            real = ms[0].astype(bool)
            ious = []
            for fx in grid:
                rm = render_mask(ep, cam, fr, fx)
                inter = (rm & real).sum()
                uni = (rm | real).sum()
                ious.append(inter / uni if uni else 0)
            ious = np.array(ious)
            scores += ious
            used += 1
            k = int(np.argmax(ious))
            print(f"  {cam} f{fr}: 提示点 {len(uv)}  SAM2 score {float(ss_[0]):.3f}  "
                  f"最优 fx={grid[k]:.1f} IoU={ious[k]:.3f}")
            # 可视化：绿=SAM2 真实，红=渲染（最优 fx）
            rm = render_mask(ep, cam, fr, grid[k])
            v = bgr.copy()
            v[real] = (0.5 * v[real] + [0, 120, 0]).astype(np.uint8)
            cnt, _ = cv2.findContours(rm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cnt, -1, (0, 0, 255), 1)
            for p in uv:
                cv2.circle(v, (int(p[0]), int(p[1])), 2, (255, 255, 0), -1)
            v = cv2.resize(v, (480, 270), interpolation=cv2.INTER_NEAREST)
            cv2.putText(v, f"{cam[:10]} f{fr} fx={grid[k]:.0f} IoU={ious[k]:.2f}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            vis_rows.append(v)
        cap.release()
        if used:
            mean = scores / used
            k = int(np.argmax(mean))
            result[cam] = {"fx": float(grid[k]), "fy": float(grid[k]), "cx": W / 2, "cy": H / 2,
                           "mean_iou": float(mean[k]), "frames_used": used}
            print(f"==> {cam}: fx=fy={grid[k]:.1f}  平均 IoU {mean[k]:.3f}  ({used} 帧)")

    if vis_rows:
        while len(vis_rows) % 4:
            vis_rows.append(np.zeros_like(vis_rows[0]))
        rows = [np.hstack(vis_rows[i:i + 4]) for i in range(0, len(vis_rows), 4)]
        cv2.imwrite(os.path.join(out_dir, "intrinsics_fit.png"), np.vstack(rows))
    json.dump({"method": "FK 渲染 mask vs SAM2 机器人 mask 的 IoU 网格搜索；cx,cy 取中心；fx=fy",
               "extrinsics_convention": "camera_extrinsics = [x,y,z,roll,pitch,yaw]，scipy euler 'xyz'，"
                                        "OpenCV 相机系（x右 y下 z前），基座=Panda link0",
               "resolution": [W, H], "cameras": result},
              open(os.path.join(out_dir, "intrinsics.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
