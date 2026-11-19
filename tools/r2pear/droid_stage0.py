#!/usr/bin/env python3
"""DROID Stage 0 收尾（通用）：
  1. 腕部相机相对 link7 的位姿：LeRobot 移植版 wrist 外参只存首帧基座系位姿（§47.2）→ rel = link7(q0)⁻¹ · T_cam(0)
     （test4 时此步在对话中手算，未脚本化；test6 补齐）
  2. 采用的内参：外部相机取 camera_refined.json 的 fx；腕部取 --wrist_fx（估计）
输出 stage0_replay/wrist_cam_rel_link7.json、intrinsics_adopted.json

用法: python tools/r2pear/droid_stage0.py --testdir <T> [--wrist_fx 150]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import droid_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--wrist_fx", type=float, default=150.0)
    a = ap.parse_args()
    import mujoco
    from scipy.spatial.transform import Rotation as R
    T = a.testdir
    od = os.path.join(T, "stage0_replay")
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    m, d, _ = C.build_model()
    d.qpos[C.arm_qadr(m)] = ep["observation.state.joint_position"][0]
    mujoco.mj_kinematics(m, d)
    b = m.body("link7").id
    T7 = np.eye(4)
    T7[:3, :3], T7[:3, 3] = d.xmat[b].reshape(3, 3), d.xpos[b]
    ext = ep["camera_extrinsics.wrist_left"][0]
    Tc = np.eye(4)
    Tc[:3, :3] = C.euler_to_R(ext[3:], "xyz") @ np.diag([1, -1, -1])      # OpenCV → MuJoCo 相机轴
    Tc[:3, 3] = ext[:3]
    rel = np.linalg.inv(T7) @ Tc
    q = R.from_matrix(rel[:3, :3]).as_quat()
    json.dump({"pos": rel[:3, 3].tolist(), "quat_wxyz": [q[3], q[0], q[1], q[2]],
               "method": "link7(q0)^-1 · wrist_extrinsics(f0)，MuJoCo 相机轴", "handeye_translation_cm":
               (rel[:3, 3] * 100).round(1).tolist()}, open(os.path.join(od, "wrist_cam_rel_link7.json"), "w"), indent=1)
    cr = json.load(open(os.path.join(od, "camera_refined.json")))
    fxs = [cr[c]["fx"] for c in cr]
    json.dump({"fx": float(np.median(fxs)), "fy": float(np.median(fxs)), "cx": 160.0, "cy": 90.0,
               "resolution": [320, 180], "per_camera": {c: cr[c]["fx"] for c in cr}, "wrist_fx": a.wrist_fx,
               "status": "estimated", "evidence": ["camera_refined.json（外参小旋转 + fx 联合精修）",
                                                  "ZED 2 HD720 标称 fx≈527 → 320 宽 ≈132"]},
              open(os.path.join(od, "intrinsics_adopted.json"), "w"), indent=1, ensure_ascii=False)
    print("handeye (cm):", (rel[:3, 3] * 100).round(1), " fx adopted:", np.median(fxs), fxs)


if __name__ == "__main__":
    main()
