#!/usr/bin/env python3
"""DROID 外部相机外参精修（论文 Sec 3.1：重投影检查联合精修相机参数，并以机器人运动学为约束）。

问题（test6 ep71 发现）：数据集外参可有数十像素系统偏移 → FK 投影的 SAM2 提示点落到背景 → 内参拟合失效。
做法：相机系下小旋转修正 R_cam' = R_cam · Rot(θx, θy, θz)，加 fx；
  目标 = Σ_帧 IoU( FK 渲染机器人 mask(θ, fx), SAM2 mask(提示点按同一 θ, fx 投影) )
  1) θx, θy 粗网格（消除画面平移），fx 固定为先验
  2) 局部坐标下降细化 θx, θy, θz, fx
平移不修（与小旋转在远处近似等价，单相机不可分）。结果写 stage0_replay/camera_refined.json，并输出前后对照图。

用法: python tools/r2pear/droid_refine_extrinsics.py --testdir <T> --cam exterior_1_left --frames 80 95 125 155 [--fx0 130]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import droid_common as C  # noqa: E402

SAM2_CFG = "configs/sam2/sam2_hiera_l.yaml"
SAM2_W = "/media/data1/RoboLab_data/weights/sam2/sam2_hiera_large.pt"
W, H = 320, 180
PROMPT_SITES = ("g_pinch",)
PROMPT_BODIES = ("link7", "link6", "g_base", "link5")


def rot(ax, ay, az):
    from scipy.spatial.transform import Rotation as R
    return R.from_euler("xyz", [ax, ay, az], degrees=True).as_matrix()


class Scene:
    def __init__(self, ep):
        import mujoco
        self.mj = mujoco
        self.ep = ep
        spec = C.build_spec()
        self.cam = spec.worldbody.add_camera(name="cam", pos=[0, 0, 0])
        spec.visual.global_.offwidth, spec.visual.global_.offheight = W * 3, H * 3
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.qa = C.arm_qadr(self.m)
        self.r = mujoco.Renderer(self.m, H * 3, W * 3)
        self.cid = self.m.camera("cam").id

    def set(self, frame, cam_name, dth, fx):
        ext = self.ep[f"camera_extrinsics.{cam_name}"][0]
        Rcv = C.euler_to_R(ext[3:], "xyz") @ rot(*dth)          # OpenCV 相机系下的修正
        Rmj = Rcv @ np.diag([1, -1, -1])
        from scipy.spatial.transform import Rotation as R
        q = R.from_matrix(Rmj).as_quat()
        self.m.cam_pos[self.cid] = ext[:3]
        self.m.cam_quat[self.cid] = [q[3], q[0], q[1], q[2]]
        self.m.cam_fovy[self.cid] = float(np.degrees(2 * np.arctan(H / 2 / fx)))
        self.d.qpos[self.qa] = self.ep["observation.state.joint_position"][frame]
        self.mj.mj_forward(self.m, self.d)
        self.Rcv, self.pos, self.fx = Rcv, ext[:3], fx

    def project(self, P):
        pc = self.Rcv.T @ (np.asarray(P) - self.pos)
        if pc[2] <= 0.05:
            return None
        return np.array([W / 2 + self.fx * pc[0] / pc[2], H / 2 + self.fx * pc[1] / pc[2]])

    def prompts(self):
        pts = [self.d.site_xpos[self.m.site(s).id] for s in PROMPT_SITES]
        pts += [self.d.xpos[self.m.body(b).id] for b in PROMPT_BODIES]
        uv = [self.project(p) for p in pts]
        return np.array([u for u in uv if u is not None and 2 <= u[0] < W - 2 and 2 <= u[1] < H - 2])

    def robot_mask(self):
        import cv2
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, camera="cam")
        seg = self.r.render()[..., 0]
        self.r.disable_segmentation_rendering()
        geom_body = self.m.geom_bodyid
        robot = np.zeros(self.m.ngeom + 1, bool)
        for g in range(self.m.ngeom):
            robot[g] = self.m.body(geom_body[g]).name not in ("world",)
        mk = (seg >= 0) & robot[np.clip(seg, 0, self.m.ngeom)]
        return cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--fx0", type=float, default=130.0)
    ap.add_argument("--grid_deg", type=float, default=12.0)
    a = ap.parse_args()
    import cv2
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    T = a.testdir
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    sc = Scene(ep)
    pred = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_W, device="cuda"))
    cap = cv2.VideoCapture(os.path.join(T, "input", "videos", f"{a.cam}.mp4"))
    imgs = {}
    for f in a.frames:
        cap.set(1, f)
        ok, bgr = cap.read()
        imgs[f] = bgr
    emb = {}
    for f in a.frames:                                       # 每帧只编码一次图像
        pred.set_image(cv2.cvtColor(imgs[f], cv2.COLOR_BGR2RGB))
        emb[f] = (pred._features, pred._orig_hw)

    def score(th, fx, detail=False):
        tot, per = 0.0, {}
        for f in a.frames:
            sc.set(f, a.cam, th, fx)
            uv = sc.prompts()
            if len(uv) < 2:
                per[f] = 0.0
                continue
            pred._features, pred._orig_hw = emb[f]
            pred._is_image_set = True
            with torch.inference_mode():
                ms, _, _ = pred.predict(point_coords=uv.astype(np.float32),
                                        point_labels=np.ones(len(uv), np.int32), multimask_output=False)
            real = ms[0].astype(bool)
            rm = sc.robot_mask()
            u = (rm | real).sum()
            iou = (rm & real).sum() / u if u else 0.0
            per[f] = float(iou)
            tot += iou
        return (tot / len(a.frames), per) if detail else tot / len(a.frames)

    base, base_per = score((0, 0, 0), a.fx0, True)
    print(f"原外参 fx={a.fx0}: 平均 IoU {base:.3f} {base_per}")
    g = np.arange(-a.grid_deg, a.grid_deg + 1e-9, 1.0)
    best = (base, (0.0, 0.0, 0.0), a.fx0)
    for ax in g:
        for ay in g:
            s = score((ax, ay, 0.0), a.fx0)
            if s > best[0]:
                best = (s, (float(ax), float(ay), 0.0), a.fx0)
    print(f"粗网格: θ={best[1]} IoU {best[0]:.3f}")
    s0, th, fx = best
    th = list(th)
    steps = [(0.5, 2.5), (0.25, 1.0)]
    for dth, dfx in steps:
        improved = True
        while improved:
            improved = False
            for i in range(3):
                for sgn in (-1, 1):
                    t2 = list(th)
                    t2[i] += sgn * dth
                    s = score(tuple(t2), fx)
                    if s > s0 + 1e-4:
                        s0, th, improved = s, t2, True
            for sgn in (-1, 1):
                s = score(tuple(th), fx + sgn * dfx)
                if s > s0 + 1e-4:
                    s0, fx, improved = s, fx + sgn * dfx, True
    fin, fin_per = score(tuple(th), fx, True)
    print(f"精修: θ={np.round(th, 2).tolist()} fx={fx} IoU {fin:.3f} {fin_per}")

    vis = []
    for f in a.frames:
        row = []
        for tag, t_, fx_ in (("orig", (0, 0, 0), a.fx0), ("refined", tuple(th), fx)):
            sc.set(f, a.cam, t_, fx_)
            v = imgs[f].copy()
            cnt, _ = cv2.findContours(sc.robot_mask().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cnt, -1, (0, 0, 255), 1)
            for p in sc.prompts():
                cv2.circle(v, (int(p[0]), int(p[1])), 2, (255, 255, 0), -1)
            v = cv2.resize(v, (480, 270))
            cv2.putText(v, f"{a.cam[:10]} f{f} {tag}", (5, 18), 0, 0.5, (0, 255, 255), 2)
            row.append(v)
        vis.append(np.hstack(row))
    od = os.path.join(T, "stage0_replay")
    os.makedirs(od, exist_ok=True)
    cv2.imwrite(os.path.join(od, f"extrinsics_refine_{a.cam}.png"), np.vstack(vis))
    p = os.path.join(od, "camera_refined.json")
    allc = json.load(open(p)) if os.path.exists(p) else {}
    allc[a.cam] = {"delta_rot_deg_xyz_cv": [float(x) for x in th], "fx": float(fx), "fy": float(fx), "cx": W / 2,
                   "cy": H / 2, "iou_before": float(base), "iou_after": float(fin), "per_frame_after": fin_per,
                   "frames": a.frames, "fx_prior": a.fx0,
                   "method": "相机系小旋转 + fx，最大化 FK 渲染机器人 mask 与 SAM2（同参数投影提示点）IoU；平移不修"}
    json.dump(allc, open(p, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
