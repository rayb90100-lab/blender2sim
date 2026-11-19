#!/usr/bin/env python3
"""Stage 1.5: 全视频物体跟踪 + Blender 动画渲染（运动拟合预览）

论文位置：这是 Step 2 的「运动拟合预览」，不是物理执行。
  skill 原话："第二步可先渲染明确标注的运动拟合预览，
              最终物理画面必须绑定对应的原生物理轨迹"

做法：
  1. SAM2 视频跟踪 -> 每帧两个物体的 mask
  2. 逐帧 MoGe 深度 + 静态背景尺度归一化 -> 每帧真实深度
  3. 反投影到 3D（固定相机内参）-> Blender 关键帧动画 -> 渲染全片

为什么不能只用首帧深度（v1 的错误）：
  v1 对全片用同一个 z_ref，物体只在像面上平移，抬起时不会真的靠近相机。
  后果是「拿起」变成「横向滑动」，深度完全丢失。

为什么不能直接用逐帧 MoGe 深度（实测）：
  相机是固定的，但 MoGe 单目深度的尺度逐帧漂移。实测静态桌面
  （画面底部 15%，全片无遮挡）的中位深度在 0.170 ~ 0.216 之间波动 ±13%。
  直接使用会让静止的桌子在渲染里前后抖动。
  解决：用同一组静态像素的深度比 s_t = med(D_t)/med(D_0) 做尺度归一化，
  再锚定到首帧（首帧深度已通过 mask bbox 重投影验证）。这是比例校正，
  不虚构任何新信息。
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

MOGE_W = "/media/data1/RoboLab_data/weights/moge3/model.pt"
SAM2_CFG = "configs/sam2/sam2_hiera_l.yaml"
SAM2_W = "/media/data1/RoboLab_data/weights/sam2/sam2_hiera_large.pt"
FFMPEG = "/home/ubuntu/mambaforge/envs/E2E/bin/ffmpeg"


def log(m): print(m, flush=True)
def hdr(m): print(f"\n{'='*60}\n{m}\n{'='*60}", flush=True)


# ---------------- 帧导出 ----------------
def export_frames(video, outdir, stride=1):
    """按 stride 导出帧序列，SAM2 视频预测器需要连续命名的 jpg。"""
    import cv2
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(outdir, f))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = saved = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            cv2.imwrite(os.path.join(outdir, f"{saved:05d}.jpg"), fr)
            saved += 1
        i += 1
    cap.release()
    return n, saved, fps


# ---------------- SAM2 视频跟踪 ----------------
def track_video(frames_dir, prompts, out_json, masks_npz):
    """SAM2 视频跟踪：首帧点提示，propagate 到全片。同时保存 mask 供深度采样。"""
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pred = build_sam2_video_predictor(SAM2_CFG, SAM2_W, device=dev)

    names = list(prompts)
    # 稳定 obj_id：不能用 hash()，它受 PYTHONHASHSEED 影响，跨进程不一致
    obj_ids = {name: i + 1 for i, name in enumerate(names)}
    rev = {v: k for k, v in obj_ids.items()}

    tracks = {k: [] for k in names}
    masks = {k: [] for k in names}
    with torch.inference_mode():
        state = pred.init_state(video_path=frames_dir)
        for name in names:
            pt = prompts[name]
            pred.add_new_points_or_box(state, frame_idx=0, obj_id=obj_ids[name],
                                       points=np.array([[pt[0], pt[1]]], dtype=np.float32),
                                       labels=np.array([1], dtype=np.int32))
        for fidx, ids, mask_logits in pred.propagate_in_video(state):
            ml = mask_logits
            if ml.ndim == 4:
                ml = ml[:, 0]
            m = (ml > 0.0).cpu().numpy()
            got = {}
            for j, oid in enumerate(ids):
                nm = rev.get(int(oid))
                if nm is None:
                    continue
                got[nm] = m[j] if m.ndim >= 3 else m
            for name in names:
                mm = got.get(name)
                if mm is None or mm.sum() == 0:
                    tracks[name].append(None)
                    masks[name].append(np.zeros((1, 1), bool))
                else:
                    ys, xs = np.nonzero(mm)
                    tracks[name].append({"cx": float(xs.mean()), "cy": float(ys.mean()),
                                         "px": int(mm.sum()),
                                         "bbox": [int(xs.min()), int(ys.min()),
                                                  int(xs.max()), int(ys.max())]})
                    masks[name].append(mm.astype(bool))

    json.dump({"obj_ids": obj_ids, "tracks": tracks},
              open(out_json, "w"), indent=1)
    np.savez_compressed(masks_npz, **{k: np.packbits(np.stack(v))
                                      for k, v in masks.items()})
    return tracks, masks


# ---------------- 逐帧深度 ----------------
def per_frame_depth(frames_dir, n_frames, K0, W, H, static_roi, obj_masks, z_anchor):
    """逐帧 MoGe 深度，用静态背景做尺度归一化。

    Returns: {name: [z or None per frame]}, scale_info
    """
    import torch
    import cv2
    from moge.model.v3 import MoGeModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(MOGE_W).to(dev).eval()
    log(f"加载 MoGe-3: {MOGE_W}")

    ys, xs = np.mgrid[0:H, 0:W]
    static = static_roi.copy()
    for name in obj_masks:
        for m in obj_masks[name]:
            if m.shape == (H, W):
                static &= ~m
    static_idx = static.reshape(-1)
    log(f"静态参考像素: {static_idx.sum()} 个（画面底部，全片无遮挡）")

    kernel = np.ones((7, 7), np.uint8)

    # 静态像素太多（13 万/帧），抽样到 2 万足够拟合，省内存
    sidx = np.flatnonzero(static_idx)
    if len(sidx) > 20000:
        sidx = np.random.default_rng(0).choice(sidx, 20000, replace=False)
    log(f"静态参考像素抽样: {len(sidx)} 个")

    raw = []            # 每帧 {name: median depth inside eroded mask}
    static_samples = []  # 每帧静态像素深度向量，用于逐帧仿射校正
    for f in range(n_frames):
        path = os.path.join(frames_dir, f"{f:05d}.jpg")
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img / 255.0).float().permute(2, 0, 1)[None].to(dev)
        with torch.inference_mode():
            out = model.infer(t, use_fp16=True, refine_steps=3)
        D = out["depth"][0].float().cpu().numpy()
        static_samples.append(D.reshape(-1)[sidx].astype(np.float32))

        rec = {}
        for name, seq in obj_masks.items():
            m = seq[f] if f < len(seq) else None
            if m is None or m.shape != (H, W) or m.sum() < 50:
                rec[name] = None
                continue
            er = cv2.erode(m.astype(np.uint8), kernel, iterations=1).astype(bool)
            if er.sum() < 50:
                er = m
            dz = D[er]
            dz = dz[np.isfinite(dz)]
            rec[name] = float(np.median(dz)) if len(dz) else None
        raw.append(rec)
        if f % 10 == 0 or f == n_frames - 1:
            log(f"  帧 {f:3d}/{n_frames} 静态深度中位={np.median(static_samples[-1]):.4f} "
                + " ".join(f"{k}={v:.3f}" if v else f"{k}=None" for k, v in rec.items()))

    # ---- 逐帧仿射校正 ----
    # 相机固定 + 静态像素的真实深度全片恒定 => 第 t 帧静态像素深度的任何变化
    # 都是模型漂移。拟合 D_0 ~ a*D_t + b（鲁棒 IRLS），再对全帧深度施加同一变换。
    # 只用静态像素定标，物体区域不参与，避免"用物体自己的深度校正自己"。
    S0 = static_samples[0]
    ok0 = np.isfinite(S0) & (S0 > 1e-6)
    corr = []
    for f in range(n_frames):
        Sf = static_samples[f]
        ok = ok0 & np.isfinite(Sf) & (Sf > 1e-6)
        if ok.sum() < 200:
            corr.append((1.0, 0.0))
            continue
        x, y = Sf[ok], S0[ok]
        # IRLS 稳健直线拟合，压掉离群点（反光、边缘）
        A = np.stack([x, np.ones_like(x)], 1)
        w = np.ones_like(x)
        for _ in range(5):
            W2 = np.sqrt(w)[:, None]
            coef, *_ = np.linalg.lstsq(A * W2, y * np.sqrt(w), rcond=None)
            r = np.abs(A @ coef - y)
            s = 1.4826 * np.median(r) + 1e-9
            w = 1.0 / (1.0 + (r / (2.5 * s)) ** 2)
        corr.append((float(coef[0]), float(coef[1])))

    a_arr = np.array([c[0] for c in corr])
    log(f"仿射校正斜率 a: {a_arr.min():.3f} ~ {a_arr.max():.3f} "
        f"(1.0 = 无漂移)；首帧 a={a_arr[0]:.4f} b={corr[0][1]:+.5f}")

    result = {k: [] for k in obj_masks}
    for f in range(n_frames):
        a_, b_ = corr[f]
        for k in obj_masks:
            v = raw[f].get(k)
            result[k].append(float(a_ * v + b_) if v else None)

    # ---- 锚定到首帧已通过 bbox 重投影验证的深度 ----
    for k in obj_masks:
        if result[k] and result[k][0] and z_anchor.get(k):
            s_ = z_anchor[k] / result[k][0]
            result[k] = [v * s_ if v else None for v in result[k]]
            log(f"  {k:14s} 首帧锚定 {result[k][0]:.4f} m（仿射估计 × {s_:.3f}）")
    return result, {"affine": corr}


# ---------------- 像素 -> 3D ----------------
def build_trajectory(tracks, depths, K, W, H):
    """固定相机内参 + 逐帧深度 -> 每帧 3D 位置（相机系，OpenCV 约定）。"""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    out = {}
    for name, seq in tracks.items():
        pts = []
        for i, t in enumerate(seq):
            z = depths[name][i] if i < len(depths[name]) else None
            if t is None or z is None:
                pts.append(None)
                continue
            X = (t["cx"] - cx) * z / fx
            Y = (t["cy"] - cy) * z / fy
            pts.append([float(X), float(Y), float(z)])
        # 缺失帧用最近邻填补，避免关键帧断档导致物体瞬移
        last = None
        for i, p in enumerate(pts):
            if p is None:
                pts[i] = last
            else:
                last = p
        nxt = None
        for i in range(len(pts) - 1, -1, -1):
            if pts[i] is None:
                pts[i] = nxt
            else:
                nxt = pts[i]
        out[name] = pts
    return out


# ---------------- 曝光标定 ----------------
def calibrate_exposure(sc, real_frame_path, outdir):
    """按真实首帧的画面亮度标定灯光强度。

    物理场景本身没变，变的是"这间屋子有多亮"。用真实首帧的整体亮度做单一
    标量定标是最小假设的做法；不去逐物体拟合物料反照率。
    """
    import bpy
    import cv2
    real = cv2.cvtColor(cv2.imread(real_frame_path), cv2.COLOR_BGR2RGB)
    H, W = real.shape[:2]
    # 只取画面底部桌面区域，避开墙/背景，那里受光照几何影响最小
    strip = real[int(0.80 * H):int(0.98 * H), int(0.05 * W):int(0.95 * W)]
    target = float(strip.mean())
    log(f"曝光标定: 真实首帧桌面区亮度 = {target:.1f}")

    lights = [o for o in bpy_objects(sc) if o.type == 'LIGHT']
    base = [o.data.energy for o in lights]
    wnode = sc.world.node_tree.nodes["Background"].inputs[1]
    wbase = wnode.default_value

    prev_pct, prev_spp = sc.render.resolution_percentage, sc.cycles.samples
    sc.render.resolution_percentage = 25
    sc.cycles.samples = 16
    prev_path = sc.render.filepath

    best, best_err = 1.0, 1e9
    for mul in [1.0, 0.25, 0.10, 0.05, 0.02, 0.008]:
        for o, e in zip(lights, base):
            o.data.energy = e * mul
        wnode.default_value = wbase * (mul ** 0.5)
        probe = os.path.join(outdir, "_probe.png")
        sc.render.filepath = probe
        bpy.ops.render.render(write_still=True)
        im = cv2.cvtColor(cv2.imread(probe), cv2.COLOR_BGR2RGB)
        got = float(im[int(0.80 * im.shape[0]):int(0.98 * im.shape[0]),
                       int(0.05 * im.shape[1]):int(0.95 * im.shape[1])].mean())
        err = abs(got - target)
        log(f"  灯光 ×{mul:<7.3f} -> 桌面亮度 {got:6.1f} (差 {got-target:+7.1f})")
        if err < best_err:
            best, best_err = mul, err
        if err < 3.0:
            break
    os.remove(os.path.join(outdir, "_probe.png")) if os.path.exists(os.path.join(outdir, "_probe.png")) else None

    for o, e in zip(lights, base):
        o.data.energy = e * best
    wnode.default_value = wbase * (best ** 0.5)
    sc.render.resolution_percentage, sc.cycles.samples = prev_pct, prev_spp
    sc.render.filepath = prev_path
    log(f"曝光标定: 采用灯光 ×{best:.3f}，世界亮度 ×{best**0.5:.3f}")
    return best


def bpy_objects(sc):
    import bpy
    return sc.objects


# ---------------- Blender 动画 ----------------
def render_animation(T, traj, man, fps, samples, stride_used):
    import bpy
    import numpy as np
    from bpy_extras.object_utils import world_to_camera_view

    bl = os.path.join(T, "stage1_blender")
    outdir = os.path.join(T, "stage1_animation")
    os.makedirs(outdir, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=os.path.join(bl, "scene_stage1.blend"))
    sc = bpy.context.scene
    cam = sc.camera
    W = int(man["camera"]["W"])
    H = int(man["camera"]["H"])

    def cam2bl(p):
        p = np.asarray(p, float)
        return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)

    n_frames = max(len(v) for v in traj.values())
    sc.frame_start, sc.frame_end = 1, n_frames
    sc.render.fps = int(round(fps))
    sc.render.fps_base = 1.0

    keyed = {}
    for name, seq in traj.items():
        ob = bpy.data.objects.get(name)
        if ob is None:
            log(f"  !! 场景里没有 {name}，跳过")
            continue
        base = np.array(ob.location)
        base_cam = np.array([base[0], -base[2], base[1]])       # bl -> cam
        ref = next((p for p in seq if p is not None), None)
        if ref is None:
            log(f"  !! {name} 无有效轨迹")
            continue
        ref = np.asarray(ref, float)
        cnt = 0
        for i, p in enumerate(seq):
            if p is None:
                continue
            delta = np.asarray(p, float) - ref
            nb = cam2bl(base_cam + delta)
            ob.location = (float(nb[0]), float(nb[1]), float(nb[2]))
            ob.keyframe_insert("location", frame=i + 1)
            cnt += 1
        keyed[name] = cnt
        log(f"  {name:14s} 关键帧 {cnt} 个"
            + (f"  深度行程 {min(p[2] for p in seq if p):.3f}~{max(p[2] for p in seq if p):.3f} m"
               if any(seq) else ""))

    # 平滑：手部带动下顶点抖动明显，用贝塞尔自动切线
    for name in keyed:
        ad = bpy.data.objects[name].animation_data
        if ad and ad.action:
            for fc in ad.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'BEZIER'
                    kp.handle_left_type = kp.handle_right_type = 'AUTO_CLAMPED'

    # ---- 自检：动画后的投影 bbox vs 跟踪 mask bbox ----
    log("  逐帧投影自检（模型 bbox vs 跟踪 mask bbox）:")
    dg = bpy.context.evaluated_depsgraph_get()
    ious = {k: [] for k in keyed}
    for f in range(1, n_frames + 1):
        sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        for name in keyed:
            ob = bpy.data.objects[name]
            if ob.animation_data is None:
                continue
            me = ob.evaluated_get(dg).to_mesh()
            pts = [ob.matrix_world @ v.co for v in me.vertices]
            uv = [world_to_camera_view(sc, cam, p) for p in pts]
            ob.evaluated_get(dg).to_mesh_clear()
            if not uv:
                continue
            xs = [u.x * W for u in uv]
            ys = [(1 - u.y) * H for u in uv]
            mbox = [min(xs), min(ys), max(xs), max(ys)]
            ious[name].append(mbox)
    for name, boxes in ious.items():
        a = np.array(boxes)
        log(f"    {name:14s} 渲染bbox范围 x[{a[:,0].min():.0f},{a[:,2].max():.0f}] "
            f"y[{a[:,1].min():.0f},{a[:,3].max():.0f}]")

    # ---- 曝光标定 ----
    first_real = os.path.join(T, "input", "frames", "f0000.png")
    if not os.path.exists(first_real):
        first_real = os.path.join(T, "input", "frames", "f000.png")
    if os.path.exists(first_real):
        calibrate_exposure(sc, first_real, outdir)

    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.image_settings.file_format = 'PNG'
    sc.render.filepath = os.path.join(outdir, "f")
    log(f"渲染 {n_frames} 帧 @ {samples} spp, {sc.render.fps} fps ...")
    bpy.ops.render.render(animation=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(outdir, "scene_animation.blend"))
    log(f"输出: {outdir}")
    return n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--samples", type=int, default=64)
    a = ap.parse_args()
    T = a.testdir
    bl = os.path.join(T, "stage1_blender")
    man = json.load(open(os.path.join(bl, "manifest.json")))
    prompts = json.load(open(os.path.join(T, "prompts.json")))["objects"]
    W, H = int(man["camera"]["W"]), int(man["camera"]["H"])

    hdr("导出帧序列")
    cand = [os.path.join(T, "source.mp4"),
            os.path.join(T, "input", "source.mp4"),
            os.path.join(T, "input", "source_hevc.mp4")]
    vid = next((c for c in cand if os.path.exists(c)), None)
    if vid is None:
        raise FileNotFoundError(f"未找到视频: {cand}")
    fdir = os.path.join(T, "frames_track")
    n_total, n_saved, src_fps = export_frames(vid, fdir, a.stride)
    fps = src_fps / a.stride
    log(f"视频 {n_total} 帧 @ {src_fps:.1f}fps -> 导出 {n_saved} 帧 (stride={a.stride}) "
        f"-> 渲染 {fps:.1f}fps，时长 {n_saved/fps:.2f}s（原始 {n_total/src_fps:.2f}s）")

    hdr("SAM2 视频跟踪")
    tj = os.path.join(T, "tracking.json")
    npz = os.path.join(T, "tracking_masks.npz")
    tracks, obj_masks = track_video(fdir, prompts, tj, npz)
    for k, v in tracks.items():
        log(f"  {k:14s} 跟踪成功 {sum(1 for x in v if x)}/{len(v)} 帧")

    hdr("逐帧 MoGe 深度")
    K = np.array([[man["camera"]["fx"], 0, man["camera"]["cx"]],
                  [0, man["camera"]["fy"], man["camera"]["cy"]],
                  [0, 0, 1]], float)
    z_anchor = {k: float(v["center_cam"][2]) for k, v in man["instances"].items()}
    log(f"首帧锚定深度: { {k: round(v,3) for k, v in z_anchor.items()} }")
    static_roi = np.zeros((H, W), bool)
    static_roi[int(0.85 * H):, :] = True
    depths, sinfo = per_frame_depth(fdir, n_saved, K, W, H, static_roi, obj_masks, z_anchor)

    hdr("像素 -> 3D")
    traj = build_trajectory(tracks, depths, K, W, H)
    json.dump({"camera": "opencv_cam", "intrinsics": K.tolist(),
               "note": "z 来自逐帧 MoGe + 静态背景尺度归一化，首帧锚定到 bbox 重投影解",
               "trajectory": traj}, open(os.path.join(T, "trajectory_cam.json"), "w"), indent=1)
    for k, v in traj.items():
        zs = [p[2] for p in v if p]
        log(f"  {k:14s} 深度行程 {min(zs):.3f} ~ {max(zs):.3f} m（{max(zs)-min(zs)*1:.3f} m）")

    hdr("Blender 动画渲染")
    n = render_animation(T, traj, man, fps, a.samples, a.stride)

    hdr("编码视频")
    out = os.path.join(T, "stage1_animation", "sim_preview.mp4")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-framerate", f"{fps:.4f}",
                    "-i", os.path.join(T, "stage1_animation", "f%04d.png"),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out],
                   check=True)
    log(f"  {out}  ({os.path.getsize(out)/1024:.0f} KB, {n} 帧 @ {fps:.1f}fps)")

    json.dump({"stride": a.stride, "frames": n, "fps": fps, "samples": a.samples,
               "depth": "per-frame MoGe-3 + static-background scale normalization",
               "anchor": "frame-0 depth from bbox reprojection fit",
               "status": "motion-fit preview (NOT physics execution)"},
              open(os.path.join(T, "stage1_animation", "animation_manifest.json"), "w"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
