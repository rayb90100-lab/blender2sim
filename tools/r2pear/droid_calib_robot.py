#!/usr/bin/env python3
"""用机器人做标定物：外部相机 6-DoF 位姿 + fx 联合标定（通用）。

原理：机器人 FK 是可信基准（FK 与数据集 cartesian_position 差 0.0 mm，§41.2）。
  真实：SAM2 视频跟踪夹爪（单帧 box 提示，box 来自 FK 渲染）→ 每帧剪影质心
  渲染：同一相机参数下 FK 渲染夹爪（g_* 体）剪影 → 每帧质心
  最小化 Σ_帧 ‖质心_真实 − 质心_渲染‖（robust soft_l1），参数 = 相机系旋转修正 3 + 世界平移修正 3 + fx
  两侧质心定义相同，剪影形状偏差大部分抵消（§45 的做法）。剔除：任一侧贴画面边缘、面积比偏离 [0.5, 2]。
初值：stage0_replay/camera_refined.json（若有）。输出 stage0_replay/camera_calib_robot.json + 对照图。

用法: python tools/r2pear/droid_calib_robot.py --testdir <T> --cam exterior_1_left --prompt_frames 91 137
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import droid_common as C  # noqa: E402
from droid_vision_probe import SAM2_VCFG, export_frames  # noqa: E402

SAM2_W = "/media/data1/RoboLab_data/weights/sam2/sam2_hiera_large.pt"
W, H = 320, 180
FLIP = np.diag([1.0, -1.0, -1.0])


def rotdeg(v):
    from scipy.spatial.transform import Rotation as R
    return R.from_rotvec(np.radians(v)).as_matrix()


class Renderer:
    def __init__(self, ep):
        import mujoco
        self.mj = mujoco
        spec = C.build_spec()
        spec.worldbody.add_camera(name="cam", pos=[0, 0, 0])
        spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, H, W)
        self.r.enable_segmentation_rendering()
        self.cid = self.m.camera("cam").id
        self.qa = C.arm_qadr(self.m)
        self.J = ep["observation.state.joint_position"]
        self.G = ep["observation.state.gripper_position"].ravel()
        self.grip_geom = np.array([self.m.body(self.m.geom_bodyid[g]).name.startswith("g_") for g in range(self.m.ngeom)])
        ga = int(self.m.joint("g_right_driver_joint").qposadr[0]) if self._has("g_right_driver_joint") else None
        self.ga = ga

    def _has(self, j):
        try:
            self.m.joint(j)
            return True
        except KeyError:
            return False

    def set_cam(self, Rcv, t, fx):
        from scipy.spatial.transform import Rotation as R
        q = R.from_matrix(Rcv @ FLIP).as_quat()
        self.m.cam_pos[self.cid] = t
        self.m.cam_quat[self.cid] = [q[3], q[0], q[1], q[2]]
        self.m.cam_fovy[self.cid] = float(np.degrees(2 * np.arctan(H / 2 / fx)))

    def mask(self, f):
        self.d.qpos[self.qa] = self.J[f]
        self.mj.mj_kinematics(self.m, self.d)
        self.mj.mj_camlight(self.m, self.d)       # 相机位姿不在 mj_kinematics 里更新（第一版漏了这步，相机停在原点）
        self.r.update_scene(self.d, camera="cam")
        seg = self.r.render()[..., 0]
        return (seg >= 0) & self.grip_geom[np.clip(seg, 0, self.m.ngeom - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--prompt_frames", type=int, nargs="+", required=True)
    ap.add_argument("--pad", type=int, default=10)
    a = ap.parse_args()
    import cv2
    import torch
    from scipy.optimize import least_squares
    from sam2.build_sam import build_sam2_video_predictor
    T = a.testdir
    od = os.path.join(T, "stage0_replay")
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    ext = ep[f"camera_extrinsics.{a.cam}"][0]
    R_data, t_data = C.euler_to_R(ext[3:], "xyz"), np.asarray(ext[:3], float)
    ref = json.load(open(os.path.join(od, "camera_refined.json"))).get(a.cam) if os.path.exists(
        os.path.join(od, "camera_refined.json")) else None
    from scipy.spatial.transform import Rotation as Rot
    R0 = R_data @ (Rot.from_euler("xyz", ref["delta_rot_deg_xyz_cv"], degrees=True).as_matrix() if ref else np.eye(3))
    fx0 = ref["fx"] if ref else 130.0
    rd = Renderer(ep)
    n = len(rd.J)

    # ---- SAM2 视频跟踪 ----
    rd.set_cam(R0, t_data, fx0)
    prompts = {}
    for pf in a.prompt_frames:                     # 多帧提示：box（FK 渲染夹爪外扩 pad）+ pinch 投影正点
        pm = rd.mask(pf)
        ys, xs = np.nonzero(pm)
        box = np.array([xs.min() - a.pad, ys.min() - a.pad, xs.max() + a.pad, ys.max() + a.pad],
                       np.float32).clip(0, [W - 1, H - 1, W - 1, H - 1])
        rd.d.qpos[rd.qa] = rd.J[pf]
        rd.mj.mj_kinematics(rd.m, rd.d)
        P = rd.d.site_xpos[rd.m.site("g_pinch").id]
        pc = (P - t_data) @ R0
        uv = np.array([160 + fx0 * pc[0] / pc[2], 90 + fx0 * pc[1] / pc[2]], np.float32)
        prompts[pf] = (box, uv)
    fdir = os.path.join(od, f"_frames_{a.cam}")
    nf = export_frames(os.path.join(T, "input", "videos", f"{a.cam}.mp4"), fdir)
    pred = build_sam2_video_predictor(SAM2_VCFG, SAM2_W, device="cuda")
    real = np.zeros((nf, H, W), bool)
    with torch.inference_mode():
        st = pred.init_state(video_path=fdir)
        for pf, (box, uv) in prompts.items():
            pred.add_new_points_or_box(st, frame_idx=pf, obj_id=1, box=box * 3,
                                       points=(uv * 3)[None], labels=np.array([1], np.int32))
        for rev in (False, True):
            for fi, ids, ml in pred.propagate_in_video(st, reverse=rev):
                mm = (ml[0, 0] > 0).cpu().numpy().astype(np.uint8)
                real[fi] = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    shutil.rmtree(fdir)
    n = min(n, nf)

    def border(mk):
        return mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any()
    # 真实与渲染被同一画面边沿裁切，质心仍可比：贴边帧保留，但要求两侧贴边状态一致（见 resid）

    rc = {}
    for f in range(n):
        mk = real[f]
        if mk.sum() >= 30:
            yy, xx = np.nonzero(mk)
            rc[f] = (np.array([xx.mean(), yy.mean()]), int(mk.sum()), border(mk))
    frames = sorted(rc)

    def unpack(p):
        return R0 @ rotdeg(p[:3]), t_data + p[3:6], fx0 + p[6]

    used = {"n": 0}

    def resid(p):
        Rc, tc, fx = unpack(p)
        rd.set_cam(Rc, tc, fx)
        out = []
        cnt = 0
        for f in frames:
            mk = rd.mask(f)
            if mk.sum() < 30 or border(mk) != rc[f][2] or not (0.5 <= mk.sum() / rc[f][1] <= 2.0):
                out += [0.0, 0.0]
                continue
            yy, xx = np.nonzero(mk)
            out += list(np.array([xx.mean(), yy.mean()]) - rc[f][0])
            cnt += 1
        used["n"] = cnt
        return np.array(out)

    STEP = np.array([0.5, 0.5, 0.5, 0.005, 0.005, 0.005, 1.0])   # 剪影按像素离散：差分步长须使质心可变

    def jac(p, *args):
        f0 = resid(p)
        Jm = np.zeros((len(f0), 7))
        for i in range(7):
            dp = np.zeros(7)
            dp[i] = STEP[i]
            Jm[:, i] = (resid(p + dp) - resid(p - dp)) / (2 * STEP[i])
        return Jm

    r0 = resid(np.zeros(7))
    n0 = used["n"]
    e0 = np.linalg.norm(r0.reshape(-1, 2), axis=1)
    sol = least_squares(resid, np.zeros(7), loss="soft_l1", f_scale=2.0, method="trf",
                        diff_step=None, x_scale=[1, 1, 1, 0.01, 0.01, 0.01, 2], jac=jac)
    r1 = resid(sol.x)
    e1 = np.linalg.norm(r1.reshape(-1, 2), axis=1)
    nz0, nz1 = e0[e0 > 0], e1[e1 > 0]
    Rc, tc, fx = unpack(sol.x)
    print(f"{a.cam}: 跟踪有效帧 {len(frames)}/{n}；残差帧 {n0} → {used['n']}")
    print(f"  质心误差 px：前 中位 {np.median(nz0):.2f} / P90 {np.percentile(nz0, 90):.2f}  →  后 中位 {np.median(nz1):.2f} / P90 {np.percentile(nz1, 90):.2f}")
    print(f"  修正（相对 stage0 精修）：旋转 {np.round(sol.x[:3], 2)}° 平移 {np.round(sol.x[3:6] * 1000, 1)} mm  fx {fx0} → {fx:.1f}")

    # 对照图
    vis = []
    cap = cv2.VideoCapture(os.path.join(T, "input", "videos", f"{a.cam}.mp4"))
    imgs = []
    while True:
        ok, b = cap.read()
        if not ok:
            break
        imgs.append(b)
    for f in [frames[len(frames) * k // 5] for k in range(5)]:
        v = imgs[f].copy()
        v[real[f]] = (0.5 * v[real[f]] + [0, 100, 0]).astype(np.uint8)
        for col, (Rq, tq, fq) in (((0, 0, 255), (R0, t_data, fx0)), ((255, 0, 255), (Rc, tc, fx))):
            rd.set_cam(Rq, tq, fq)
            cnt, _ = cv2.findContours(rd.mask(f).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cnt, -1, col, 1)
        v = cv2.resize(v, (480, 270), interpolation=cv2.INTER_NEAREST)
        cv2.putText(v, f"f{f} red=before magenta=after green=SAM2", (4, 16), 0, 0.42, (0, 255, 255), 1)
        vis.append(v)
    cv2.imwrite(os.path.join(od, f"calib_robot_{a.cam}.png"), np.vstack([np.hstack(vis[:3]), np.hstack(vis[3:] + [np.zeros_like(vis[0])])]))
    p = os.path.join(od, "camera_calib_robot.json")
    allc = json.load(open(p)) if os.path.exists(p) else {}
    q = Rot.from_matrix(Rc).as_euler("xyz")
    allc[a.cam] = {"R_cv": Rc.tolist(), "t": tc.tolist(), "euler_xyz": q.tolist(), "fx": float(fx), "fy": float(fx),
                   "cx": W / 2, "cy": H / 2, "delta_vs_dataset": {
                       "rot_total_deg": float(np.degrees(Rot.from_matrix(R_data.T @ Rc).magnitude())),
                       "trans_mm": ((tc - t_data) * 1000).round(1).tolist()},
                   "centroid_err_px": {"before_median": float(np.median(nz0)), "before_p90": float(np.percentile(nz0, 90)),
                                       "after_median": float(np.median(nz1)), "after_p90": float(np.percentile(nz1, 90)),
                                       "frames": int(used["n"])},
                   "prompt_frames": a.prompt_frames,
                   "method": "SAM2 视频跟踪夹爪剪影质心 vs FK 渲染夹爪剪影质心，相机 6DoF+fx 鲁棒最小二乘"}
    json.dump(allc, open(p, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
