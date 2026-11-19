#!/usr/bin/env python3
"""Real2Gym Stage 1 完整管线：真实视频 -> Blender 场景

对应论文 Sec 3.1 "Scene Reconstruction and Alignment"：
  1. MoGe-3 几何初始化（单视角）
  2. SAM2 实例分割（交互物体 + 支撑面）
  3. Blender 完整 mesh + 迭代重投影精修

用法:
  python stage1_pipeline.py --input <video> --outdir <dir> --prompts <json>

prompts.json 格式:
  {"objects": {"earbud_case": [140, 660], "tissue_box": [530, 540]}}
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

MOGE_W = "/media/data1/RoboLab_data/weights/moge3/model.pt"
SAM2_CFG = "configs/sam2/sam2_hiera_l.yaml"
SAM2_W = "/media/data1/RoboLab_data/weights/sam2/sam2_hiera_large.pt"
FFMPEG = "/home/ubuntu/mambaforge/envs/E2E/bin/ffmpeg"

def log(msg): print(msg, flush=True)
def hdr(msg): print(f"\n{'='*60}\n{msg}\n{'='*60}", flush=True)


# ---------------- Step 0: 视频 -> 帧 ----------------
def step0_frames(video, outdir):
    hdr("Step 0: 视频解码与抽帧")
    os.makedirs(outdir, exist_ok=True)
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        # HEVC 回退：用 ffmpeg 转码
        log("OpenCV 无法解码，尝试 ffmpeg 转 H.264")
        fixed = os.path.join(outdir, "source.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-i", video,
                        "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                        "-pix_fmt", "yuv420p", "-an", fixed, "-y"], check=True)
        video = fixed
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"无法解码: {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"视频: {W}x{H} @ {fps:.1f}fps, {n} 帧")

    frames_dir = os.path.join(outdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    idxs = sorted({int(i) for i in np.linspace(0, n-1, min(9, n))})
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok: cv2.imwrite(os.path.join(frames_dir, f"f{i:04d}.png"), fr)
    cap.release()
    log(f"抽出 {len(idxs)} 个稀疏帧: {idxs}")
    first = os.path.join(frames_dir, f"f{idxs[0]:04d}.png")
    return first, {"frames": n, "fps": fps, "size": [W, H], "sampled": idxs}


# ---------------- Step 1: MoGe-3 ----------------
def step1_moge(frame_path, outdir, refine_steps=3):
    hdr("Step 1: MoGe-3 几何初始化")
    import torch
    from moge.model.v3 import MoGeModel
    import cv2
    os.makedirs(outdir, exist_ok=True)

    img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"加载 MoGe-3 权重: {MOGE_W}")
    model = MoGeModel.from_pretrained(MOGE_W).to(dev).eval()

    t = torch.from_numpy(img/255.0).float().permute(2,0,1).unsqueeze(0).to(dev)
    import time; t0 = time.time()
    with torch.inference_mode():
        out = model.infer(t, force_projection=False, refine_steps=refine_steps)
    log(f"推理完成 {time.time()-t0:.2f}s (refine_steps={refine_steps})")

    K = out["intrinsics"][0].cpu().numpy()
    fx, fy = K[0,0]*W, K[1,1]*H
    cx, cy = K[0,2]*W, K[1,2]*H
    res = {
        "points": out["points"][0].cpu().numpy(),
        "depth": out["depth"][0].cpu().numpy(),
        "normal": out["normal"][0].cpu().numpy(),
        "mask": out["mask"][0].cpu().numpy(),
        "intrinsics_px": np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]),
        "size": [W, H],
    }
    np.savez(os.path.join(outdir, "moge_out.npz"),
             points=res["points"], depth=res["depth"], normal=res["normal"],
             mask=res["mask"], intrinsics_px=res["intrinsics_px"])
    log(f"内参(像素): fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    log(f"有效像素: {res['mask'].mean():.3f}")
    if dev == "cuda":
        log(f"显存峰值: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    return res


# ---------------- Step 2: SAM2 ----------------
def step2_sam2(frame_path, outdir, prompts):
    hdr("Step 2: SAM2 实例分割")
    import torch, cv2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    os.makedirs(outdir, exist_ok=True)

    img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"加载 SAM2: {SAM2_W}")
    pred = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_W, device=dev))

    masks, report = {}, {}
    with torch.inference_mode():
        pred.set_image(img)
        for name, pt in prompts.items():
            px, py = int(pt[0]), int(pt[1])
            m, sc, _ = pred.predict(point_coords=np.array([[px,py]]),
                                    point_labels=np.array([1]), multimask_output=True)
            best = np.asarray(m[int(np.argmax(sc))]).astype(bool)
            ys, xs = np.nonzero(best)
            masks[name] = best
            report[name] = {"prompt_xy": [px,py], "score": round(float(max(sc)),4),
                            "px": int(best.sum()),
                            "bbox": [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]}
            np.save(os.path.join(outdir, f"{name}_mask.npy"), best)
            vis = img.copy()
            vis[best] = (vis[best]*0.35 + np.array([255,60,60])*0.65).astype(np.uint8)
            cv2.circle(vis, (px,py), 6, (0,255,255), -1)
            cv2.imwrite(os.path.join(outdir, f"{name}_overlay.png"),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            log(f"  {name:14s} score={report[name]['score']:.3f} px={best.sum():6d} bbox={report[name]['bbox']}")

    json.dump(report, open(os.path.join(outdir, "sam2_report.json"),"w"), indent=2, ensure_ascii=False)
    return masks, report


# ---------------- Step 3: 平面 + 世界基 ----------------
def estimate_plane(P, D, H, W):
    """RANSAC 拟合支撑面。只用画面下部区域，避免背景墙。"""
    ys, xs = np.mgrid[0:H, 0:W]
    roi = (ys > int(H*0.55)) & np.isfinite(D)
    cand = P[roi]
    if len(cand) < 100:
        cand = P[np.isfinite(D)]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(800):
        i = rng.choice(len(cand), 3, replace=False)
        a,b,c = cand[i]
        n = np.cross(b-a, c-a)
        nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n = n/nn
        inl = np.abs((cand-a) @ n) < 0.008
        if best is None or inl.sum() > best[1].sum(): best = (n, inl)
    n_cam, inl = best
    d_cam = float(np.mean(cand[inl] @ n_cam))
    # 法线朝上（相机系 y 向下）
    if n_cam[1] > 0: n_cam, d_cam = -n_cam, -d_cam
    return n_cam, d_cam, float(inl.mean())


def cam2bl(p):
    """OpenCV 相机系 (x右, y下, z前) -> Blender (x右, y前, z上)。线性变换，点与方向通用。"""
    p = np.asarray(p, dtype=float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def estimate_plane(P, D, H, W):
    """RANSAC 拟合支撑面（相机系）。只用画面下部避开背景墙。"""
    ys, xs = np.mgrid[0:H, 0:W]
    roi = (ys > int(H * 0.55)) & np.isfinite(D)
    cand = P[roi]
    if len(cand) < 100:
        cand = P[np.isfinite(D)]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(800):
        i = rng.choice(len(cand), 3, replace=False)
        a, b, c = cand[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        inl = np.abs((cand - a) @ n) < 0.008
        if best is None or inl.sum() > best[1].sum():
            best = (n, inl)
    n_cam, inl = best
    d_cam = float(np.mean(cand[inl] @ n_cam))
    if n_cam[1] > 0:                      # 法线朝上（相机系 y 向下）
        n_cam, d_cam = -n_cam, -d_cam

    # 平面用 ROI 拟合（避开背景墙），但"桌面范围"必须用全图共面点，
    # 否则远端桌面被 ROI 裁掉，桌面 mesh 只剩近处一小块。
    finite = np.isfinite(D).reshape(-1)
    dist = np.abs(P.reshape(-1, 3) @ n_cam - d_cam)
    on_plane = (dist < 0.012) & finite
    full_pts = P.reshape(-1, 3)[on_plane]
    return n_cam, d_cam, float(inl.mean()), full_pts


def plane_frame(n_cam):
    """由平面法线构造面内正交基 (e_x, e_y, e_z=法线)。"""
    e_z = n_cam / np.linalg.norm(n_cam)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(e_z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e_x = np.cross(tmp, e_z)
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(e_z, e_x)
    return e_x, e_y, e_z


def fit_object(pts_cam, mask, K, e_x, e_y, e_z, plane_d):
    """相机系内拟合有向盒：盒轴 = 支撑面基 (e_x,e_y,e_z)。

    用 scipy.least_squares 直接解 4 个未知量 (cu, cv, log su, log sv)
    去匹配 mask bbox 的 4 条边。高度取点云相对支撑面高度的 99 分位。

    注意：e_x 沿桌面纵深、e_y 沿水平，两者对像面投影的贡献完全不同
    （e_x 主要影响纵向、e_y 主要影响横向），不能用同一个幂律同时校正。
    """
    from scipy.optimize import least_squares

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.nonzero(mask)
    tx0, tx1, ty0, ty1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())

    u = pts_cam @ e_x
    v = pts_cam @ e_y
    h = pts_cam @ e_z - plane_d
    sh = max(float(np.percentile(h, 99)), 1e-3)

    # 初始：点云在面内的包围范围
    x0 = np.array([(u.min() + u.max()) / 2,
                   (v.min() + v.max()) / 2,
                   np.log(max(np.ptp(u), 1e-3)),
                   np.log(max(np.ptp(v), 1e-3))])

    def bbox_of(params):
        cu, cv, lsu, lsv = params
        su, sv = np.exp(lsu), np.exp(lsv)
        c = cu * e_x + cv * e_y + (plane_d + sh / 2) * e_z
        corners = []
        for i in (-1, 1):
            for j in (-1, 1):
                for k in (-1, 1):
                    corners.append(c + i * su / 2 * e_x + j * sv / 2 * e_y
                                   + k * sh / 2 * e_z)
        corners = np.array(corners)
        if (corners[:, 2] <= 1e-6).any():
            return None, c, su, sv
        uv = np.stack([fx * corners[:, 0] / corners[:, 2] + cx,
                       fy * corners[:, 1] / corners[:, 2] + cy], 1)
        return (uv[:, 0].min(), uv[:, 0].max(),
                uv[:, 1].min(), uv[:, 1].max()), c, su, sv

    def resid(params):
        bb, _, _, _ = bbox_of(params)
        if bb is None:
            return np.full(4, 1e4)
        return np.array([bb[0] - tx0, bb[1] - tx1, bb[2] - ty0, bb[3] - ty1])

    sol = least_squares(resid, x0, method="lm", max_nfev=400)
    bb, c, su, sv = bbox_of(sol.x)

    err = {
        "bbox_model": [round(float(bb[0]), 1), round(float(bb[2]), 1),
                       round(float(bb[1]), 1), round(float(bb[3]), 1)],
        "bbox_target": [int(tx0), int(ty0), int(tx1), int(ty1)],
        "center_err_px": [round(float((bb[0] + bb[1]) / 2 - (tx0 + tx1) / 2), 1),
                          round(float((bb[2] + bb[3]) / 2 - (ty0 + ty1) / 2), 1)],
        "size_ratio": [round(float((bb[1] - bb[0]) / max(tx1 - tx0, 1)), 3),
                       round(float((bb[3] - bb[2]) / max(ty1 - ty0, 1)), 3)],
        "rms_px": round(float(np.sqrt(np.mean(resid(sol.x) ** 2))), 2),
    }
    return {"center": c, "size": np.array([su, sv, sh]),
            "axes": (e_x, e_y, e_z)}, err


def step3_blender(frame_path, moge, masks, outdir):
    hdr("Step 3: Blender 场景构建")
    import bpy
    from mathutils import Matrix
    os.makedirs(outdir, exist_ok=True)

    P, D, valid = moge["points"], moge["depth"], moge["mask"]
    K = moge["intrinsics_px"]
    W, H = moge["size"]

    n_cam, d_cam, inl, plane_pts = estimate_plane(P, D, H, W)
    log(f"共面点 {len(plane_pts)} 个（含 ROI 之外的远端桌面）")
    e_x, e_y, e_z = plane_frame(n_cam)
    log(f"支撑面: n={np.round(n_cam, 4)} 内点率={inl:.3f}")

    fits, errs = {}, {}
    for name, m in masks.items():
        sel = m & valid & np.isfinite(D)
        pts = P[sel]
        if len(pts) < 50:
            log(f"  {name}: 点太少，跳过")
            continue
        fit, e = fit_object(pts, m, K, e_x, e_y, e_z, d_cam)
        fits[name], errs[name] = fit, e
        log(f"  {name:14s} size={np.round(fit['size'], 3)}")
        log(f"                 重投影 中心误差={e['center_err_px']} px 尺寸比={e['size_ratio']}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    def mat(name, rgba, rough=0.6):
        mt = bpy.data.materials.new(name)
        mt.use_nodes = True
        b = mt.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = rgba
        b.inputs["Roughness"].default_value = rough
        return mt

    def set_box(ob, center_cam, axes_cam, size, material):
        """一次性构造含旋转+缩放的 matrix_world。

        注意：先赋 matrix_world 再设 ob.scale 会互相覆盖（scale 会重写矩阵的缩放部分），
        必须把缩放直接乘进旋转矩阵的列，一次赋值。
        """
        au, av, ah = [cam2bl(a) for a in axes_cam]
        su, sv, sh = (float(size[0]), float(size[1]), float(size[2]))
        c = cam2bl(center_cam)
        M = Matrix((
            (au[0] * su, av[0] * sv, ah[0] * sh, c[0]),
            (au[1] * su, av[1] * sv, ah[1] * sh, c[1]),
            (au[2] * su, av[2] * sv, ah[2] * sh, c[2]),
            (0.0, 0.0, 0.0, 1.0),
        ))
        ob.matrix_world = M
        ob.data.materials.append(material)
        bpy.ops.object.shade_flat()

    # 桌面：沿支撑面的板
    loc = np.stack([plane_pts @ e_x, plane_pts @ e_y, np.zeros(len(plane_pts))], 1)
    su = float(np.ptp(loc[:, 0])) * 1.08
    sv = float(np.ptp(loc[:, 1])) * 1.08
    cu = float((loc[:, 0].min() + loc[:, 0].max()) / 2)
    cv = float((loc[:, 1].min() + loc[:, 1].max()) / 2)
    th = 0.03
    tbl_center = cu * e_x + cv * e_y + (d_cam - th / 2) * e_z
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    t = bpy.context.object
    t.name = "table"
    set_box(t, tbl_center, (e_x, e_y, e_z), (su, sv, th), mat("wood", (0.55, 0.40, 0.26, 1), 0.55))
    log(f"桌面: {su:.3f} x {sv:.3f} m")

    # ---- 背景：显著背景实体（论文要求 salient background entities）----
    ys_g, xs_g = np.mgrid[0:H, 0:W]
    bg_roi = (ys_g < int(H * 0.30)) & np.isfinite(D)
    bg_pts = P[bg_roi]
    if len(bg_pts) > 2000:
        rngb = np.random.default_rng(1)
        bestb = None
        for _ in range(500):
            i = rngb.choice(len(bg_pts), 3, replace=False)
            a, b, c = bg_pts[i]
            nn = np.cross(b - a, c - a)
            nb = np.linalg.norm(nn)
            if nb < 1e-9:
                continue
            nn = nn / nb
            inl = np.abs((bg_pts - a) @ nn) < 0.012
            if bestb is None or inl.sum() > bestb[1].sum():
                bestb = (nn, inl)
        nb_cam, bgl = bestb
        db = float(np.mean(bg_pts[bgl] @ nb_cam))
        if nb_cam @ e_z > 0:                 # 法线朝相机
            nb_cam, db = -nb_cam, -db
        ang = float(np.degrees(np.arccos(min(1.0, abs(nb_cam @ e_z)))))
        log(f"背景面: n={np.round(nb_cam,3)} d={db:.3f} 内点率={bgl.mean():.3f} 与支撑面夹角={ang:.1f}deg")
        if bgl.mean() > 0.5 and ang > 60:
            # 墙面范围用「墙面自身基」计算，不要用桌面面内基（会把方向拧歪）
            zc = nb_cam / np.linalg.norm(nb_cam)
            tmpx = np.array([1.0, 0.0, 0.0]) if abs(zc[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            xc = np.cross(tmpx, zc); xc /= np.linalg.norm(xc)
            yc = np.cross(zc, xc)

            onb = bg_pts[bgl]
            loc = np.stack([onb @ xc, onb @ yc], 1)
            wu = float(np.ptp(loc[:, 0])) * 1.35 + 0.5
            wv = float(np.ptp(loc[:, 1])) * 1.35 + 0.5
            cu = float((loc[:, 0].min() + loc[:, 0].max()) / 2)
            cv = float((loc[:, 1].min() + loc[:, 1].max()) / 2)

            # 墙中心直接取背景点云的质心，避免人为抬高/偏移
            bctr = cu * xc + cv * yc + db * zc
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            bg_ob = bpy.context.object
            bg_ob.name = "background_wall"
            set_box(bg_ob, bctr, (xc, yc, zc), (wu, wv, 0.02),
                    mat("wall", (0.58, 0.62, 0.66, 1), 0.9))
            log(f"背景墙: {wu:.2f} x {wv:.2f} m  中心(相机系)={np.round(bctr,3)}")

    COLORS = {"earbud_case": (0.045, 0.045, 0.05, 1), "tissue_box": (0.86, 0.84, 0.90, 1)}
    for name, f in fits.items():
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        ob = bpy.context.object
        ob.name = name
        set_box(ob, f["center"], f["axes"], f["size"],
                mat(name, COLORS.get(name, (0.7, 0.7, 0.7, 1)), 0.5))

    # 相机：本地 -Z 指向相机系 +z(前)，本地 +Y 指向 -y(上)
    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36.0
    cd.lens = K[0, 0] / W * 36.0
    cd.clip_start = 0.005
    cd.clip_end = 100.0
    cob = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cob)
    cob.matrix_world = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    sc.camera = cob
    log(f"相机 lens={cd.lens:.2f}mm")

    for loc_, en, sz in [((0.4, 0.3, 0.5), 90, 1.0), ((-0.4, 0.2, 0.3), 35, 1.2)]:
        bpy.ops.object.light_add(type="AREA", location=loc_)
        o = bpy.context.object
        o.data.energy = en
        o.data.size = sz

    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 96
    sc.cycles.use_denoising = True
    sc.render.resolution_x = W
    sc.render.resolution_y = H
    sc.view_settings.view_transform = "AgX"
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.60, 0.64, 0.68, 1)
    sc.world = w

    # 渲染前用 Blender 自己的投影自检
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    for name, f in fits.items():
        v = world_to_camera_view(sc, cob, Vector(cam2bl(f["center"]).tolist()))
        flag = "OK" if (0 <= v.x <= 1 and 0 <= v.y <= 1 and v.z > 0) else "!! 不在画面内"
        log(f"  自检 {name:14s} NDC=({v.x:.3f},{v.y:.3f}) depth={v.z:.3f}  {flag}")

    sc.render.filepath = os.path.join(outdir, "render_f000.png")
    bpy.ops.render.render(write_still=True)
    sc.render.film_transparent = True
    sc.render.filepath = os.path.join(outdir, "render_f000_alpha.png")
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(outdir, "scene_stage1.blend"))

    manifest = {
        "frame": frame_path,
        "frame_convention": "Blender = (x_cam, z_cam, -y_cam); camera at origin looking +Ybl",
        "camera": {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]),
                   "cy": float(K[1, 2]), "W": W, "H": H},
        "support_plane": {"normal_cam": n_cam.tolist(), "d_cam": float(d_cam),
                          "inlier": round(float(np.mean(inl)), 4),
                          "note": "相机系下的倾角是正确的（俯拍），不做水平化"},
        "instances": {k: {"center_cam": v["center"].tolist(),
                          "size": v["size"].tolist()} for k, v in fits.items()},
        "reprojection": errs,
        "note": "visible surfaces observed; occluded faces inferred via mask bbox constraint",
    }
    json.dump(manifest, open(os.path.join(outdir, "manifest.json"), "w"), indent=2, ensure_ascii=False)
    return fits, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--refine-steps", type=int, default=3)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    prompts = json.load(open(a.prompts))["objects"]

    frame, vinfo = step0_frames(a.input, os.path.join(a.outdir,"input"))
    json.dump(vinfo, open(os.path.join(a.outdir,"input","video_meta.json"),"w"), indent=2)

    moge = step1_moge(frame, os.path.join(a.outdir,"stage1_moge3"), a.refine_steps)
    masks, rep = step2_sam2(frame, os.path.join(a.outdir,"stage1_sam2"), prompts)
    fits, errs = step3_blender(frame, moge, masks, os.path.join(a.outdir,"stage1_blender"))

    hdr("完成")
    print(f"输出目录: {a.outdir}")
    for k,v in errs.items():
        print(f"  {k:14s} 中心误差 {v['center_err_px']} px  尺寸比 {v['size_ratio']}")


if __name__ == "__main__":
    main()
