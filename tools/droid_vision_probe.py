#!/usr/bin/env python3
"""视觉测量误差探测：在 DROID 320×180 下，从两路外部视频定位机械臂末端能有多准。

公平性设计：
  - SAM2 视频跟踪只在一帧上给一次提示（等价于人工点一下），其余帧全自动传播；
    不逐帧用 FK 投影当提示，否则等于把真值喂给视觉。
  - 误差按同一定义比较：「真实可见剪影质心」vs「FK 渲染可见剪影质心」（两者都被画面边界同样裁剪），
    两路相机各自得到 2D 质心后三角化，比较 3D 点。
  - 同时给出「去掉每路相机恒定 2D 偏置」后的误差：代表内参/外参标定做好后的水平。

注意：本探测测的是「剪影质心」的定位误差。实际要的是 TCP/pinch 位姿，
从剪影到 TCP 还要再加一步（模型配准），误差只会更大 —— 所以这是乐观下界。

用法:
  python tools/droid_vision_probe.py --testdir <testdir> [--prompt_frame 60]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import droid_common as C  # noqa: E402
import droid_fit_intrinsics as F  # noqa: E402
from droid_stage1_scene import Cam, triangulate  # noqa: E402

SAM2_VCFG = "configs/sam2/sam2_hiera_l.yaml"
SAM2_W = "/media/data1/RoboLab_data/weights/sam2/sam2_hiera_large.pt"
CAMS = ("exterior_1_left", "exterior_2_left")
W, H = 320, 180


def log(m): print(m, flush=True)


def export_frames(video, outdir):
    import cv2
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    cap = cv2.VideoCapture(video)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        # SAM2 对 320×180 太小的图效果差，上采样 3 倍再跟踪，mask 再缩回
        cv2.imwrite(os.path.join(outdir, f"{i:05d}.jpg"),
                    cv2.resize(f, (W * 3, H * 3), interpolation=cv2.INTER_CUBIC),
                    [cv2.IMWRITE_JPEG_QUALITY, 98])
        i += 1
    return i


def render_masks(ep, fx, n):
    """FK 渲染：每帧每路相机的机器人可见 mask。"""
    import mujoco
    import cv2
    spec = C.build_spec()
    for c in CAMS:
        ext = ep[f"camera_extrinsics.{c}"][0]
        cam = spec.worldbody.add_camera(name=c, pos=ext[:3], quat=F.cam_quat_mj(ext[3:]))
        cam.fovy = float(np.degrees(2 * np.arctan(H / 2 / fx)))
    spec.visual.global_.offwidth = W * 3
    spec.visual.global_.offheight = H * 3
    m = spec.compile()
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H * 3, W * 3)
    r.enable_segmentation_rendering()
    J = ep["observation.state.joint_position"]
    G = ep["observation.state.gripper_position"].ravel()
    out = {c: np.zeros((n, H, W), bool) for c in CAMS}
    for i in range(n):
        d.qpos[C.arm_qadr(m)] = J[i]
        for jn in ("g_right_driver_joint", "g_left_driver_joint"):
            try:
                d.qpos[int(m.joint(jn).qposadr[0])] = 0.8 * G[i]
            except Exception:
                pass
        mujoco.mj_forward(m, d)
        for c in CAMS:
            r.update_scene(d, camera=c)
            seg = r.render()[..., 0]
            out[c][i] = cv2.resize((seg >= 0).astype(np.uint8), (W, H),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
    r.close()
    return out


def prompt_points(ep, cam, frame, fx):
    """单帧提示点：夹爪部件的 FK 投影（模拟人工在这一帧上点几下）。"""
    uv = F.project_points(ep, cam, frame, fx)
    return uv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--prompt_frame", type=int, default=60)
    a = ap.parse_args()
    import cv2
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    T = a.testdir
    out = os.path.join(T, "stage2_physics", "vision_probe")
    os.makedirs(out, exist_ok=True)
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    K = json.load(open(os.path.join(T, "stage0_replay", "intrinsics_adopted.json")))
    fx = K["fx"]
    n = len(ep["observation.state.joint_position"])

    log("FK 渲染真值剪影 ...")
    rmask = render_masks(ep, fx, n)

    pred = build_sam2_video_predictor(SAM2_VCFG, SAM2_W, device="cuda")
    tmask = {}
    for c in CAMS:
        fdir = os.path.join(out, f"frames_{c}")
        nf = export_frames(os.path.join(T, "input", "videos", f"{c}.mp4"), fdir)
        # 单帧 box 提示（模拟人工在这一帧框一下夹爪）。
        # 曾用 3 个点提示：夹爪又细又黑挡在绿纸板前，点落到纸板上，SAM2 全程跟踪了纸板（IoU 0.014）。
        ys, xs = np.nonzero(rmask[c][a.prompt_frame])
        pad = 3
        box = np.array([max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                        min(xs.max() + pad, W - 1), min(ys.max() + pad, H - 1)], np.float32) * 3
        log(f"{c}: {nf} 帧，仅在 f{a.prompt_frame} 给 1 个框 {np.round(box / 3).astype(int).tolist()}")
        masks = np.zeros((nf, H, W), bool)
        with torch.inference_mode():
            st = pred.init_state(video_path=fdir)
            pred.add_new_points_or_box(st, frame_idx=a.prompt_frame, obj_id=1, box=box)
            for rev in (False, True):
                for fi, ids, ml in pred.propagate_in_video(st, reverse=rev):
                    mm = (ml[0, 0] > 0).cpu().numpy().astype(np.uint8)
                    masks[fi] = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            pred.reset_state(st)
        tmask[c] = masks[:n]
        shutil.rmtree(fdir)

    # ---- 逐帧比较 ----
    cams = {c: Cam(ep[f"camera_extrinsics.{c}"][0], fx, fx, K["cx"], K["cy"]) for c in CAMS}
    rows = []
    for i in range(n):
        rec = {"frame": i}
        ok = True
        for c in CAMS:
            tm, rm = tmask[c][i], rmask[c][i]
            if tm.sum() < 15 or rm.sum() < 15:
                ok = False
                break
            ty, tx = np.nonzero(tm)
            ry, rx = np.nonzero(rm)
            rec[c] = {"real": [tx.mean(), ty.mean()], "fk": [rx.mean(), ry.mean()],
                      "iou": float((tm & rm).sum() / (tm | rm).sum())}
        if ok:
            rows.append(rec)
    log(f"两路都可见的帧: {len(rows)}/{n}")

    def tri(key, bias=None):
        pts = []
        for r in rows:
            uvs = []
            for c in CAMS:
                u, v = r[c][key]
                if bias is not None:
                    u, v = u - bias[c][0], v - bias[c][1]
                uvs.append((u, v))
            pts.append(triangulate([cams[c] for c in CAMS], uvs))
        return np.array(pts)

    P_fk = tri("fk")
    P_real = tri("real")
    e_raw = np.linalg.norm(P_real - P_fk, axis=1) * 1000
    bias = {c: np.median([np.subtract(r[c]["real"], r[c]["fk"]) for r in rows], 0) for c in CAMS}
    P_cor = tri("real", bias)
    e_cor = np.linalg.norm(P_cor - P_fk, axis=1) * 1000
    px = {c: np.array([np.linalg.norm(np.subtract(r[c]["real"], r[c]["fk"])) for r in rows]) for c in CAMS}
    ious = {c: np.array([r[c]["iou"] for r in rows]) for c in CAMS}
    fr = np.array([r["frame"] for r in rows])

    # 抓取窗口（f70–100，夹爪接近和闭合）单独统计：这是决定成败的时段
    win = (fr >= 70) & (fr <= 100)
    rep = {
        "frames_compared": int(len(rows)), "frames_total": int(n),
        "prompt": f"仅 f{a.prompt_frame} 一帧给 1 个框（框取自该帧 FK 渲染范围，等价于人工框选），其余 SAM2 视频传播",
        "px_err_median": {c: float(np.median(px[c])) for c in CAMS},
        "px_bias": {c: np.round(bias[c], 2).tolist() for c in CAMS},
        "mask_iou_median": {c: float(np.median(ious[c])) for c in CAMS},
        "err3d_mm_raw": {"median": float(np.median(e_raw)), "p90": float(np.percentile(e_raw, 90)),
                         "grasp_window_median": float(np.median(e_raw[win])) if win.any() else None},
        "err3d_mm_bias_corrected": {"median": float(np.median(e_cor)), "p90": float(np.percentile(e_cor, 90)),
                                    "grasp_window_median": float(np.median(e_cor[win])) if win.any() else None},
        "threshold_from_noise_sweep_mm": 9.0,
        "note": "剪影质心定位误差，是 TCP 位姿估计误差的乐观下界；未含姿态",
    }
    json.dump(rep, open(os.path.join(out, "vision_probe_report.json"), "w"), indent=2, ensure_ascii=False)
    np.savez(os.path.join(out, "probe.npz"), frames=fr, P_fk=P_fk, P_real=P_real, P_cor=P_cor)
    for c in CAMS:
        log(f"  {c}: 2D 质心误差中位 {np.median(px[c]):.2f} px  恒定偏置 {np.round(bias[c], 1)} px  "
            f"mask IoU 中位 {np.median(ious[c]):.3f}")
    log(f"  3D 误差（原始）       中位 {np.median(e_raw):.1f} mm  P90 {np.percentile(e_raw, 90):.1f} mm  "
        f"抓取窗口中位 {rep['err3d_mm_raw']['grasp_window_median']}")
    log(f"  3D 误差（去恒定偏置） 中位 {np.median(e_cor):.1f} mm  P90 {np.percentile(e_cor, 90):.1f} mm  "
        f"抓取窗口中位 {rep['err3d_mm_bias_corrected']['grasp_window_median']}")

    # 可视化：抽 6 帧，绿=SAM2 跟踪，红=FK 渲染轮廓
    vis = []
    for i in np.linspace(0, n - 1, 6).astype(int):
        row = []
        for c in CAMS:
            cap = cv2.VideoCapture(os.path.join(T, "input", "videos", f"{c}.mp4"))
            cap.set(1, i)
            _, img = cap.read()
            cap.release()
            tm = tmask[c][i]
            img[tm] = (0.5 * img[tm] + [0, 110, 0]).astype(np.uint8)
            cnt, _ = cv2.findContours(rmask[c][i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnt, -1, (0, 0, 255), 1)
            img = cv2.resize(img, (480, 270), interpolation=cv2.INTER_NEAREST)
            cv2.putText(img, f"{c[:10]} f{i}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            row.append(img)
        vis.append(np.hstack(row))
    cv2.imwrite(os.path.join(out, "probe_vis.png"), np.vstack(vis))
    log(f"写出 {out}/")


if __name__ == "__main__":
    main()
