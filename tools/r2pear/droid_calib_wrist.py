#!/usr/bin/env python3
"""腕部相机相对 link7 的位姿标定：用画面中始终可见的 Robotiq 手指作标定物（通用）。

问题（test6）：rel = link7(q0)⁻¹ · wrist_extrinsics(f0) 推出的位姿使渲染手指与真实手指偏 15–20 px，
物体三角化高度方向偏 2–4 cm。手指与相机刚性相连、占画面面积大 → 旋转与平移均可观测。
真实手指 mask：画面下部 (y > 0.45H) 暗像素（灰度 < thr）中与底边相连的连通块，形态学清理。
目标：Σ_帧 IoU(FK 渲染夹爪 g_* 剪影, 真实手指 mask)；参数 rel 旋转修正 3（度）+ 平移修正 3（m）+ fx。
只用夹爪张开的帧（渲染不建模开合）。输出 stage0_replay/wrist_cam_rel_link7.json（原文件备份为 *_from_extrinsics.json）。

用法: python tools/r2pear/droid_calib_wrist.py --testdir <T> --frames 0 20 40 60 80
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import droid_common as C  # noqa: E402

W, H = 320, 180


def real_finger_mask(bgr, thr):
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = (g < thr).astype(np.uint8)
    dark[: int(0.45 * H)] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectivityStats if False else cv2.connectedComponentsWithStats(dark)
    keep = np.zeros_like(dark, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if y + h >= H - 1 and area >= 80:
            keep |= lab == i
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--thr", type=int, default=55)
    a = ap.parse_args()
    import cv2
    import mujoco
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation as R
    T = a.testdir
    od = os.path.join(T, "stage0_replay")
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    G = ep["observation.state.gripper_position"].ravel()
    assert all(G[f] < 0.05 for f in a.frames), "只用夹爪张开的帧"
    relp = os.path.join(od, "wrist_cam_rel_link7.json")
    bak = os.path.join(od, "wrist_cam_rel_link7_from_extrinsics.json")
    if not os.path.exists(bak):
        shutil.copy(relp, bak)
    rel0 = json.load(open(bak))
    K = json.load(open(os.path.join(od, "intrinsics_adopted.json")))
    fx0 = float(K.get("wrist_fx", 150.0))
    p0 = np.array(rel0["pos"])
    q = rel0["quat_wxyz"]
    Rr0 = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    spec = C.build_spec()
    cam = spec.body("link7").add_camera(name="w", pos=list(p0), quat=list(q))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    m = spec.compile()
    d = mujoco.MjData(m)
    rd = mujoco.Renderer(m, H, W)
    rd.enable_segmentation_rendering()
    cid = m.camera("w").id
    grip = np.array([m.body(m.geom_bodyid[g]).name.startswith("g_") for g in range(m.ngeom)])
    qa = C.arm_qadr(m)

    cap = cv2.VideoCapture(os.path.join(T, "input", "videos", "wrist_left.mp4"))
    imgs = []
    while True:
        ok, b = cap.read()
        if not ok:
            break
        imgs.append(b)
    real = {f: real_finger_mask(imgs[f], a.thr) for f in a.frames}

    def set_params(x):
        Rr = Rr0 @ R.from_rotvec(np.radians(x[:3])).as_matrix()
        qq = R.from_matrix(Rr).as_quat()
        m.cam_pos[cid] = p0 + x[3:6]
        m.cam_quat[cid] = [qq[3], qq[0], qq[1], qq[2]]
        m.cam_fovy[cid] = float(np.degrees(2 * np.arctan(H / 2 / (fx0 + x[6]))))
        return Rr

    def render(f):
        d.qpos[qa] = ep["observation.state.joint_position"][f]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        rd.update_scene(d, camera="w")
        seg = rd.render()[..., 0]
        return (seg >= 0) & grip[np.clip(seg, 0, m.ngeom - 1)]

    def score(x, per=False):
        set_params(x)
        ious = {}
        for f in a.frames:
            mk, rm = render(f), real[f]
            ious[f] = float((mk & rm).sum() / max((mk | rm).sum(), 1))
        return ious if per else -float(np.mean(list(ious.values())))

    SC = np.array([1, 1, 1, 0.005, 0.005, 0.005, 5.0])
    s0 = score(np.zeros(7), per=True)
    best = None
    for start in [np.zeros(7)] + [np.r_[np.random.default_rng(i).normal(0, 2, 3), np.zeros(4)] for i in range(3)]:
        res = minimize(lambda u: score(u * SC), start / SC, method="Powell", options={"maxiter": 6000, "xtol": 1e-3})
        if best is None or res.fun < best.fun:
            best = res
    x = best.x * SC
    s1 = score(x, per=True)
    Rr = set_params(x)
    qq = R.from_matrix(Rr).as_quat()
    print(f"IoU 前 {np.mean(list(s0.values())):.3f} {s0}")
    print(f"IoU 后 {np.mean(list(s1.values())):.3f} {s1}")
    print(f"修正：旋转 {np.round(x[:3], 2)}°  平移 {np.round(x[3:6] * 1000, 1)} mm  fx {fx0} → {fx0 + x[6]:.1f}")

    vis = []
    for f in a.frames[:4]:
        v = imgs[f].copy()
        v[real[f]] = (0.5 * v[real[f]] + [0, 100, 0]).astype(np.uint8)
        for col, xx in (((0, 0, 255), np.zeros(7)), ((255, 0, 255), x)):
            set_params(xx)
            cnt, _ = cv2.findContours(render(f).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cnt, -1, col, 1)
        v = cv2.resize(v, (480, 270))
        cv2.putText(v, f"f{f} red=before magenta=after green=dark-mask", (4, 16), 0, 0.42, (0, 255, 255), 1)
        vis.append(v)
    while len(vis) % 2:
        vis.append(np.zeros_like(vis[0]))
    cv2.imwrite(os.path.join(od, "calib_wrist_fingers.png"),
                np.vstack([np.hstack(vis[i:i + 2]) for i in range(0, len(vis), 2)]))
    json.dump({"pos": (p0 + x[3:6]).tolist(), "quat_wxyz": [qq[3], qq[0], qq[1], qq[2]], "fx": float(fx0 + x[6]),
               "method": "Robotiq 手指剪影（暗像素、连底边）vs FK 渲染夹爪 IoU，rel 6DoF + fx，Powell",
               "iou_before": s0, "iou_after": s1, "frames": a.frames,
               "delta_vs_extrinsics": {"rot_deg": x[:3].round(2).tolist(), "trans_mm": (x[3:6] * 1000).round(1).tolist()},
               "handeye_translation_cm": ((p0 + x[3:6]) * 100).round(1).tolist()},
              open(relp, "w"), indent=1, ensure_ascii=False)
    K["wrist_fx"] = float(fx0 + x[6])
    K["wrist_fx_source"] = "droid_calib_wrist.py（手指剪影）"
    json.dump(K, open(os.path.join(od, "intrinsics_adopted.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
