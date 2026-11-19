#!/usr/bin/env python3
"""用腕部视角精修绿方块的 yaw 与水平尺寸（五问 Q3 触发的几何修正）。

问题（review_gate I3）：f84 腕部三路对照中，真实方块的面与指垫平行，仿真方块斜约 26°，且偏大约 20%。
原因：Stage 1 只用 ext1 一个 320×180 视角拟合（ext2 被遮挡），yaw 与水平尺寸约束弱。
腕部相机距方块十几厘米，是三路里分辨率最高的证据。

做法：接触前 f80（方块仍在桌上原位、两指张开、方块完整可见），
  - 腕部相机位姿取自 MuJoCo 模型（挂 link7，随 FK 运动），fx=150
  - SAM2 在腕部 f80 点选方块（固定像素点，等价人工点击）
  - 优化 (yaw, sx, sy)，目标 = 腕部 IoU + ext1 IoU；中心 xy、桌面 z、高度 h 保持 Stage 1 值
    （中心与高度已由夹爪 pinch 测量约束，§42）

用法: python tools/droid_refine_block_wrist.py --testdir <testdir> [--frame 80] [--apply]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import droid_common as C  # noqa: E402
from droid_stage1_scene import Cam, box_corners, box_mask, iou  # noqa: E402

SAM2_CFG = "configs/sam2/sam2_hiera_l.yaml"
SAM2_W = "/media/data1/RoboLab_data/weights/sam2/sam2_hiera_large.pt"
W, H = 320, 180


def log(m): print(m, flush=True)


def wrist_cam(testdir, ep, frame, fx):
    import mujoco
    rel = json.load(open(os.path.join(testdir, "stage0_replay", "wrist_cam_rel_link7.json")))
    spec = C.build_spec()
    spec.body("link7").add_camera(name="w", pos=rel["pos"], quat=rel["quat_wxyz"])
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[C.arm_qadr(m)] = ep["observation.state.joint_position"][frame]
    mujoco.mj_forward(m, d)          # 相机位姿在 mj_camlight 里算，mj_kinematics 不会填 cam_xmat
    cid = m.camera("w").id
    Rcv = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])       # MuJoCo 相机 → OpenCV
    from scipy.spatial.transform import Rotation as R
    rpy = R.from_matrix(Rcv).as_euler("xyz")
    return Cam(np.r_[d.cam_xpos[cid], rpy], fx, fx, W / 2, H / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--frame", type=int, default=80)
    ap.add_argument("--wrist_fx", type=float, default=150.0)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    import cv2
    import torch
    from scipy.optimize import minimize
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    T = a.testdir
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    sp = os.path.join(T, "stage1_scene", "scene.json")
    scene = json.load(open(sp))
    K = scene["intrinsics"]
    g = scene["objects"]["green_block"]
    zt = scene["table_z"]
    cx, cy = g["center"][:2]
    h = g["size"][2]

    cams = {"wrist_left": wrist_cam(T, ep, a.frame, a.wrist_fx),
            "exterior_1_left": Cam(ep["camera_extrinsics.exterior_1_left"][0], K["fx"], K["fy"], K["cx"], K["cy"])}
    pred = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_W, device="cuda"))
    masks, imgs = {}, {}
    for c in cams:
        cap = cv2.VideoCapture(os.path.join(T, "input", "videos", f"{c}.mp4"))
        cap.set(1, a.frame if c == "wrist_left" else 0)
        _, img = cap.read()
        cap.release()
        imgs[c] = img
        if c == "exterior_1_left":
            masks[c] = np.load(os.path.join(T, "stage1_scene", "mask_exterior_1_left_green_block.npy"))
            continue
        # 方块投影中心作为点提示（等价人工在方块上点一下）
        uv, _ = cams[c].project(np.array([[cx, cy, zt + h / 2]]))
        pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            ms, sc, _ = pred.predict(point_coords=uv.astype(np.float32), point_labels=np.ones(1, np.int32),
                                     multimask_output=True)
        # 方块颜色单一，取分数最高的
        k = int(np.argmax(sc))
        masks[c] = ms[k].astype(bool)
        log(f"  {c} f{a.frame}: 提示 {np.round(uv[0], 1)}  SAM2 score {sc[k]:.3f}  像素 {int(masks[c].sum())}")

    # 中心 xy 也放开：腕部视角显示 Stage 1 中心偏约 2 cm（ext1 单视角 + 夹爪约束的精度量级）。
    # 保留物理约束：f84 夹爪闭合时 pinch 点必须在方块内部（§42 同一软约束）。
    import mujoco
    mm_, dd_, _ = C.build_model()
    # f88：腕部真实画面两侧指垫均已贴合、方块居中于两指间（§47/§49），用它作为中心测量
    dd_.qpos[C.arm_qadr(mm_)] = ep["observation.state.joint_position"][88]
    mujoco.mj_kinematics(mm_, dd_)
    p_grasp = dd_.site_xpos[mm_.site("g_pinch").id].copy()

    def pinch_pen(px, py, yaw, sx, sy):
        # 强约束：中心 xy 距夹持中心 ≤ 5 mm（机器人运动学精确；腕部单视角沿视线深度约束弱，
        # 曾在只要求「pinch 在方块内」时把中心推到 17 mm 外，导致叠放时滑落，§49）
        return 200.0 * max(0.0, np.hypot(p_grasp[0] - px, p_grasp[1] - py) - 0.005)

    def score(x):
        px, py, yaw, lsx, lsy = x
        sx_, sy_ = np.exp(lsx), np.exp(lsy)
        cor = box_corners(px, py, yaw, sx_, sy_, zt, h)
        return -sum(iou(box_mask(cams[c], cor), masks[c]) for c in cams) + pinch_pen(px, py, yaw, sx_, sy_)

    def ious(px, py, yaw, sx, sy):
        cor = box_corners(px, py, yaw, sx, sy, zt, h)
        return {c: round(float(iou(box_mask(cams[c], cor), masks[c])), 3) for c in cams}

    base = ious(cx, cy, g["yaw_rad"], g["size"][0], g["size"][1])
    best = None
    for y0 in np.radians(np.arange(-45, 46, 15)):
        for dxy in ((0, 0), (0.015, 0), (-0.015, 0), (0, 0.015), (0, -0.015)):
            r = minimize(score, [cx + dxy[0], cy + dxy[1], y0, np.log(0.032), np.log(0.032)],
                         method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-6})
            if best is None or r.fun < best.fun:
                best = r
    ncx, ncy, yaw = best.x[0], best.x[1], best.x[2]
    sx, sy = float(np.exp(best.x[3])), float(np.exp(best.x[4]))
    new = ious(ncx, ncy, yaw, sx, sy)
    log(f"  原值: 中心 ({cx:.4f},{cy:.4f}) yaw {np.degrees(g['yaw_rad']):+.1f}°  尺寸 {g['size'][0]*1000:.1f}×{g['size'][1]*1000:.1f} mm  IoU {base}")
    log(f"  精修: 中心 ({ncx:.4f},{ncy:.4f}) yaw {np.degrees(yaw):+.1f}°  尺寸 {sx*1000:.1f}×{sy*1000:.1f} mm  IoU {new}")
    log(f"  中心移动 {np.hypot(ncx-cx, ncy-cy)*1000:.1f} mm；pinch 罚项 {pinch_pen(ncx, ncy, yaw, sx, sy):.4f}")

    vis = []
    for c in cams:
        v = imgs[c].copy()
        v[masks[c]] = (0.5 * v[masks[c]] + [0, 100, 0]).astype(np.uint8)
        for (qx, qy, yy, a1, a2, col) in [(cx, cy, g["yaw_rad"], g["size"][0], g["size"][1], (0, 0, 255)),
                                          (ncx, ncy, yaw, sx, sy, (255, 255, 0))]:
            pm = box_mask(cams[c], box_corners(qx, qy, yy, a1, a2, zt, h))
            cnt, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cnt, -1, col, 1)
        vis.append(cv2.resize(v, (640, 360), interpolation=cv2.INTER_NEAREST))
    out = os.path.join(T, "stage1_scene", "refine_green_wrist.png")
    cv2.imwrite(out, np.hstack(vis))
    log(f"  可视化（红=原值，青=精修，绿=SAM2）: {out}")

    if a.apply:
        g.setdefault("history", []).append({"yaw_rad": g["yaw_rad"], "size": g["size"], "iou": g.get("iou"),
                                            "center": g["center"], "source": "stage1 双视角拟合（ext1 单视角有效）"})
        g["center"] = [float(ncx), float(ncy), g["center"][2]]
        g["yaw_rad"] = float(yaw)
        g["size"] = [sx, sy, h]
        g["iou"] = new
        g["refined_by"] = f"腕部 f{a.frame} + ext1 联合 IoU 精修中心xy/yaw/sx/sy，保留 f84 pinch 在方块内约束（review_gate I3）"
        json.dump(scene, open(sp, "w"), indent=2, ensure_ascii=False)
        log("  已写回 scene.json")


if __name__ == "__main__":
    main()
