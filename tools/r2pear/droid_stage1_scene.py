#!/usr/bin/env python3
"""Stage 1 三维场景拟合（通用，test6 起）。

与 tools/droid_stage1_scene.py（test4 专用）的区别：
  - 物体名、形状、角色、事件帧全部来自 stage1_scene/objects.json，无硬编码
  - mask 直接读 stage1_scene/masks.npz（由 droid_stage1_masks.py 生成），本脚本不再重新分割
  - 相机统一走 droid_cams.Cams：外参精修（camera_refined.json）存在则用，缺失回退原始外参并显式标注
  - 支撑关系由 objects.json 的 role + stack_order 推导，不再「除目标外全是塔」

旧脚本 droid_stage1_scene.py 保持原样，随时可退回。

拟合：每个物体参数化为直立箱体 (cx, cy, yaw, sx, sy, h)，底面贴共享桌面 z_table；
叠放物按 stack_order 链式支撑（上层底 = 下层顶）。arch 按复合体投影（顶板 + 两腿）。
目标函数 = 各视角「投影 vs SAM2 mask」IoU 之和 − 机器人软约束罚项；Nelder-Mead 多起点。

用法:
  python tools/r2pear/droid_stage1_scene.py --testdir <T>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import droid_common as C  # noqa: E402
import droid_cams as DC  # noqa: E402

W, H = DC.W, DC.H
# arch 复合体比例，与 r2pear/sim.py 的构造保持一致
ARCH_TOP_FRAC, ARCH_LEG_FRAC = 0.35, 0.30
# 角色 → 底面约束
ROLES = ("target", "stack", "support", "distractor")
# 圆柱绕自身轴旋转对称，yaw 无定义，固定为 0 减少自由度
YAW_FREE = {"box": True, "arch": True, "prism": True, "cylinder": False}
# 每物体保底项（见 score_for 内注释）：平均得分低于 tau 时按 w 惩罚。
# tau 取 0.35 ≈ 单视角合格线下限（quality_gate.fit_iou_min=0.30）留一点余量。
BOTTOM_TAU = 0.35
BOTTOM_W = 4.0


def log(m):
    print(m, flush=True)


def corners_box(cx, cy, yaw, sx, sy, z0, z1):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = []
    for i in (-1, 1):
        for j in (-1, 1):
            for z in (z0, z1):
                pts.append([cx + c * i * sx / 2 - s * j * sy / 2,
                            cy + s * i * sx / 2 + c * j * sy / 2, z])
    return np.array(pts)


def shape_corners(shape, cx, cy, yaw, sx, sy, z0, h):
    """返回组成该物体的若干角点集（arch 为顶板 + 两腿，其余为单箱体）。"""
    if shape != "arch":
        return [corners_box(cx, cy, yaw, sx, sy, z0, z0 + h)]
    ts, lh, lw = ARCH_TOP_FRAC * h, (1 - ARCH_TOP_FRAC) * h, ARCH_LEG_FRAC * sx
    parts = [corners_box(cx, cy, yaw, sx, sy, z0 + h - ts, z0 + h)]
    c, s = np.cos(yaw), np.sin(yaw)
    for sgn in (-1, 1):
        ox = sgn * (sx / 2 - lw / 2)
        parts.append(corners_box(cx + c * ox, cy + s * ox, yaw, lw, sy, z0, z0 + lh))
    return parts


def proj_mask(cams, cam, shape, cx, cy, yaw, sx, sy, z0, h):
    import cv2
    m = np.zeros((H, W), np.uint8)
    for part in shape_corners(shape, cx, cy, yaw, sx, sy, z0, h):
        uv = cams.project(cam, part)
        if np.isnan(uv).any():
            continue
        hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--views", default=None,
                    help="参与拟合的视角，逗号分隔；默认全部有 mask 的视角")
    ap.add_argument("--starts", type=int, default=10, help="Nelder-Mead 起点数")
    ap.add_argument("--order_starts", type=int, default=4, help="推断叠放顺序时的短跑起点数")
    ap.add_argument("--no_table_display", action="store_true", help="不写 table_display.json")
    ap.add_argument("--out_subdir", default="mine",
                    help="输出子目录（相对 stage1_scene/）。并行工作时用来隔离写入；传空串则写到根目录")
    a = ap.parse_args()
    import cv2
    from scipy.optimize import minimize

    T = a.testdir
    # 输入目录（objects.json / masks.npz）与输出目录分离：
    # 与 workstation-71 并行工作时输出写到 mine/，共享的根目录保持只读。
    indir = os.path.join(T, "stage1_scene")
    out = os.path.join(indir, a.out_subdir) if a.out_subdir else indir
    os.makedirs(out, exist_ok=True)
    cfg_path = os.path.join(indir, "objects.json")
    cfg = json.load(open(cfg_path))
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    cams = DC.Cams(T, ep)

    mk_path = os.path.join(indir, "masks.npz")
    if not os.path.exists(mk_path):
        raise SystemExit(f"缺少 {mk_path}，先跑 tools/r2pear/droid_stage1_masks.py")
    z = np.load(mk_path)
    # masks.npz 的 key 是 "<obj>|<cam>"（droid_stage1_masks.py 的约定），这里转成 (cam, obj)
    masks = {}
    for k in z.files:
        o, c = k.split("|", 1)
        masks[(c, o)] = z[k].astype(bool)

    all_cams = sorted({c for (c, _n) in masks})
    views = [v.strip() for v in a.views.split(",")] if a.views else all_cams

    imgs = {}
    for c in views:
        cap = cv2.VideoCapture(os.path.join(T, "input", "videos", f"{c}.mp4"))
        cap.set(1, int(cfg["frame"]))
        imgs[c] = cap.read()[1]
        cap.release()

    # ---- 角色与叠放顺序 ----
    # 叠放链（底→顶）含所有「站在别的物体上」的物体，不只是 role=stack：
    # ep71 的 target 绿柱在 frame 0 时也站在塔顶，必须链式支撑，否则会贴到桌面上。
    objs = cfg["objects"]
    role = {n: o.get("role", "distractor") for n, o in objs.items()}
    stack = [n for n in objs if role[n] == "stack"]
    target = [n for n in objs if role[n] == "target"]
    if len(target) != 1:
        raise SystemExit(f"objects.json 需要恰好一个 role=target，实际 {target}")
    target = target[0]
    chain = cfg.get("support_chain")
    if chain is None:
        chain = cfg.get("stack_order") or stack
        log(f"未给 support_chain，叠放链按 stack_order/名称排序（底→顶）: {chain}")
    if set(chain) - set(objs):
        raise SystemExit(f"support_chain 含未知物体: {set(chain) - set(objs)}")
    if len(set(chain)) != len(chain):
        raise SystemExit(f"support_chain 有重复物体: {chain}")
    for n in objs:
        if n in chain and role[n] == "support":
            raise SystemExit(f"{n} 同时是 support 和叠放链成员，语义冲突")
    on_stack = {n for n in chain}
    log(f"叠放链（底→顶）: {chain}")

    # ---- 机器人锚点（来自 objects.json 的事件帧）----
    def pinch_at(frame):
        import mujoco
        m, d, _ = C.build_model()
        d.qpos[C.arm_qadr(m)] = ep["observation.state.joint_position"][frame]
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[m.site("g_pinch").id].copy()

    g_fr, r_fr = int(cfg["events"]["grasp"]), int(cfg["events"]["release"])
    p_grasp, p_release = pinch_at(g_fr), pinch_at(r_fr)
    log(f"抓取 f{g_fr} pinch {np.round(p_grasp, 4)}；释放 f{r_fr} pinch {np.round(p_release, 4)}")

    # ---- 可用 mask 统计 ----
    fit_names = [n for n in objs if any((c, n) in masks and masks[(c, n)].sum() >= 8 for c in views)]
    for n in objs:
        if n not in fit_names:
            log(f"  {n}: 无可用 mask，跳过拟合（将不在 scene.json 中）")
    if not fit_names:
        raise SystemExit("没有任何物体有可用 mask")

    # ---- 参数化 ----
    fixed_yaw = {n: (not YAW_FREE.get(objs[n].get("shape", "box"), True)) for n in fit_names}
    npar = 1 + sum(5 if fixed_yaw[n] else 6 for n in fit_names)

    def unpack(x):
        zt = x[0]
        p, i = {}, 1
        for n in fit_names:
            if fixed_yaw[n]:
                cx, cy, lsx, lsy, lh = x[i:i + 5]
                yaw = 0.0
                i += 5
            else:
                cx, cy, yaw, lsx, lsy, lh = x[i:i + 6]
                i += 6
            p[n] = (cx, cy, yaw, np.exp(lsx), np.exp(lsy), np.exp(lh))
        return zt, p

    def bottoms(zt, p, ordr):
        """返回每个物体的底面 z：stack 链式支撑，其余贴桌面。"""
        z0 = {n: zt for n in p}
        prev = None
        for n in ordr:
            if n not in p:
                continue
            if prev is not None:
                z0[n] = z0[prev] + p[prev][5]
            prev = n
        return z0

    def render_scene(cam, p, z0, shape_of):
        """遮挡感知投影：按距离由远及近绘制标签图，每个物体只保留自己可见的像素。

        不做遮挡会导致远处物体把近处物体的 IoU 强行拉低，优化器只能靠单视角硬拟合。
        返回 {obj: visible_mask}。
        """
        import cv2
        order_c = sorted(fit_names,
                         key=lambda n: -np.linalg.norm(
                             np.array([p[n][0], p[n][1], z0[n] + p[n][5] / 2]) - cams.pose_cv(cam)[1]))
        label = np.full((H, W), -1, np.int32)
        for idx, n in enumerate(order_c):
            cx, cy, yaw, sx, sy, h = p[n]
            for part in shape_corners(shape_of(n), cx, cy, yaw, sx, sy, z0[n], h):
                uv = cams.project(cam, part)
                if np.isnan(uv).any():
                    continue
                hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
                cv2.fillConvexPoly(label, hull, idx)
        return {n: (label == i) for i, n in enumerate(order_c)}

    def score_for(ordr):
        def score(x):
            zt, p = unpack(x)
            z0 = bottoms(zt, p, ordr)
            shape_of = lambda n: objs[n].get("shape", "box")
            s = 0.0
            # ★ 每物体保底项（2026-11-18，ep88 实测教训）
            #   原版是所有 (物体, 视角) 的 IoU 直接求和，**没有任何下限保护**。
            #   ep88 的 6 个物体里有两个同色小黄块被优化器「牺牲」：它们被推到错误位置，
            #   自报 IoU 精确为 0.0、质心误差 19-55 px，只因把资源让给 4 个大块能让总和更高。
            #   IoU 是 [0,1] 有界量，总和最优解会把最难的物体按到 0（max-sum ≠ max-min）。
            #   做法：对每个物体取「它在各视角的平均表现」，低于 BOTTOM_TAU 的部分按
            #   BOTTOM_W 加权惩罚。这样牺牲任何一个物体都要付固定代价，不再是免费的。
            per_obj = {n: [0.0, 0] for n in fit_names}
            for c in views:
                vis = render_scene(c, p, z0, shape_of)
                for n in fit_names:
                    mk = masks.get((c, n))
                    if mk is None or mk.sum() < 8:
                        continue
                    pm = vis[n]
                    # IoU 主项
                    u = (pm | mk).sum()
                    g = (pm & mk).sum() / u if u else 0.0
                    s += g
                    per_obj[n][0] += g
                    per_obj[n][1] += 1
                    # 质心软项：IoU 为 0 时没有梯度，优化器会放弃该视角（实测 ext2/wrist 全 0）。
                    # 用高斯核把「质心距离」变成有界、可下降的量，尺度 = 5 px。
                    uvp, zz = cams.project(c, np.array([[p[n][0], p[n][1], z0[n] + p[n][5] / 2]]))[0][0], None
                    if np.isnan(uvp).any():
                        s -= 0.05
                        continue
                    ys_, xs_ = np.nonzero(mk)
                    dpx = np.linalg.norm(uvp - np.array([xs_.mean(), ys_.mean()]))
                    s += 0.35 * np.exp(-(dpx / 5.0) ** 2) - 0.35 * min(dpx / 80.0, 1.0)
                    per_obj[n][0] += 0.35 * np.exp(-(dpx / 5.0) ** 2) - 0.35 * min(dpx / 80.0, 1.0)
            # 保底：逐物体平均得分低于 tau 的部分重罚
            for n in fit_names:
                tot, cnt = per_obj[n]
                if cnt == 0:
                    continue
                avg = tot / cnt
                if avg < BOTTOM_TAU:
                    s -= BOTTOM_W * (BOTTOM_TAU - avg)
            # 软约束 0：桌面高度先验。**必须放在所有视角循环之外**，否则每物体每视角罚一次，放大 15 倍。
            # 这一项很弱（2.0 × 误差/2cm ≈ 0.1），只防漂移，不主导拟合 —— 主导项必须是 IoU + 质心。
            s -= 0.1 * abs(zt - zt0) / 0.02
            # 软约束 1：抓取时刻 pinch 点应落在目标物体内部
            if target in p:
                cx, cy, yaw, sx, sy, h = p[target]
                c_, s_ = np.cos(yaw), np.sin(yaw)
                dx, dy = p_grasp[0] - cx, p_grasp[1] - cy
                lx, ly = c_ * dx + s_ * dy, -s_ * dx + c_ * dy
                lz = p_grasp[2] - (z0[target] + h / 2)
                ex = (max(0, abs(lx) - sx / 2) + max(0, abs(ly) - sy / 2)
                      + max(0, abs(lz) - h / 2))
                s -= 50.0 * ex
            # 软约束 2：释放点（目标最终落点）应在支撑物水平范围内
            for n in fit_names:
                if role[n] != "support":
                    continue
                cx, cy, yaw, sx, sy, h = p[n]
                c_, s_ = np.cos(yaw), np.sin(yaw)
                dx, dy = p_release[0] - cx, p_release[1] - cy
                lx, ly = c_ * dx + s_ * dy, -s_ * dx + c_ * dy
                ex = max(0, abs(lx) - sx / 2) + max(0, abs(ly) - sy / 2)
                s -= 100.0 * ex
            return -s
        return score

    def make_x0(rng, zt0):
        x = [zt0 + rng.normal(0, 0.01)]
        for n in fit_names:
            prior = objs[n].get("size_prior")
            s = prior[0] if prior else rng.choice([0.030, 0.045, 0.065])
            h = prior[1] if prior else rng.choice([0.030, 0.045, 0.065])
            x.append(rng.uniform(0.55, 0.75))       # cx：桌面中心附近
            x.append(rng.uniform(-0.15, 0.15))      # cy
            if not fixed_yaw[n]:
                x.append(rng.uniform(-np.pi / 2, np.pi / 2))
            x += [np.log(s), np.log(s * rng.uniform(0.8, 1.25)), np.log(h)]
        return np.array(x)

    # 用 mask 质心三角化给出更靠谱的 xy 初值
    def triangulate(cs, uvs):
        A, b = [], []
        for cam, (u, v) in zip(cs, uvs):
            Rcv, t = cams.pose_cv(cam)
            d = Rcv @ np.array([(u - W / 2) / cams.fx(cam), (v - H / 2) / cams.fx(cam), 1.0])
            d /= np.linalg.norm(d)
            P = np.eye(3) - np.outer(d, d)
            A.append(P)
            b.append(P @ t)
        return np.linalg.solve(np.sum(A, 0), np.sum(b, 0))

    anchor_xy = {}
    for n in fit_names:
        uvs, cs = [], []
        for c in views:
            mk = masks.get((c, n))
            if mk is None or mk.sum() < 8:
                continue
            ys, xs = np.nonzero(mk)
            uvs.append((xs.mean(), ys.mean()))
            cs.append(c)
        if len(cs) >= 2:
            try:
                anchor_xy[n] = triangulate(cs, uvs)[:2]
                log(f"  {n}: 质心三角化 xy {np.round(anchor_xy[n], 4)}")
                continue
            except np.linalg.LinAlgError:
                pass
        if n == target:
            anchor_xy[n] = p_grasp[:2]
        elif role[n] == "support":
            anchor_xy[n] = p_release[:2]
        else:
            anchor_xy[n] = np.array([0.65, 0.0])
        log(f"  {n}: 初值 xy 取锚点 {np.round(anchor_xy[n], 4)}")

    # 桌面高度初值：pinch 在目标物体中部，减固定偏移只是粗猜。
    # 实测这个粗猜本身不可靠（曾给出 +36.6 mm，正确值约 -28 mm，差 6.5 cm），
    # 因此**不让它主导**：seed 时对 z_table 做网格多起点扫描，让 IoU 决定。
    zt0 = float(p_grasp[2] - 0.045)
    zt_grid = [zt0 + dz for dz in (-0.08, -0.055, -0.03, 0.0, 0.03)]

    def seed(rng, ordr, zt_seed=None):
        zt_s = zt0 if zt_seed is None else zt_seed
        x0 = make_x0(rng, zt_s)
        x0[0] = zt_s                      # 显式覆盖，make_x0 里加了噪声
        i = 1
        for n in fit_names:
            k = 5 if fixed_yaw[n] else 6
            x0[i] = anchor_xy[n][0]
            x0[i + 1] = anchor_xy[n][1]
            i += k
        return x0

    def run(ordr, starts, maxiter, tag=""):
        sc = score_for(ordr)
        best = None
        rng = np.random.default_rng(0)
        for t in range(starts):
            # z_table 按网格轮转，让每个高度值都有机会作为起点 —— 否则全部起点共用同一个
            # 错误先验，Nelder-Mead 在 12 维里几乎不可能自己爬出来（实测差 6.5 cm）。
            zt_s = zt_grid[t % len(zt_grid)]
            r = minimize(sc, seed(rng, ordr, zt_s), method="Nelder-Mead",
                         options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-6, "adaptive": True})
            if best is None or r.fun < best.fun:
                best = r
            log(f"  {tag}起点 {t:2d} (z_tab0={zt_s*1000:+6.1f} mm): 目标 {-r.fun:+.3f}")
        return best

    # ---- 叠放顺序：未显式给出时，按拟合质量在候选排列中选 ----
    order = chain
    if not cfg.get("support_chain"):
        # 只有 role=stack 的物体允许换序；target 若也在链上，位置由 objects.json 隐含（通常在顶）
        movable = [n for n in chain if role[n] == "stack"]
        if len(movable) >= 2:
            from itertools import permutations
            fixed = [(i, n) for i, n in enumerate(chain) if role[n] != "stack"]
            cand = []
            for perm in permutations(movable):
                c = list(perm)
                for i, n in fixed:
                    c.insert(i, n)
                cand.append(c)
            if len(cand) > 24:
                log(f"  候选顺序 {len(cand)} 个，超过 24，只试原始顺序")
                cand = [chain]
            best_o, best_r = None, None
            for o in cand:
                r = run(o, a.order_starts, 800, tag=f"顺序 {o} ")
                log(f"  顺序 {o}: 短跑 IoU 和 {-r.fun:.3f}")
                if best_r is None or r.fun < best_r.fun:
                    best_o, best_r = o, r
            order = best_o
        log(f"推断叠放顺序（底→顶）: {order}")

    # ---- 正式拟合 ----
    best = run(order, a.starts, 4000)
    z_tab, p = unpack(best.x)
    z0 = bottoms(z_tab, p, order)
    log(f"\n桌面高度 z_table = {z_tab:.4f} m（基座系）")

    shapes = {n: objs[n].get("shape", "box") for n in fit_names}
    result = {
        "frame": int(cfg["frame"]),
        "table_z": float(z_tab),
        "objects": {},
        "cameras": {c: {"extrinsics": ep[f"camera_extrinsics.{c}"][0].tolist()} for c in views
                    if f"camera_extrinsics.{c}" in ep},
        "intrinsics": cams.K,
        "extrinsics_status": cams.refined_status,
        "support_chain": order,
        "method": "多视角形状投影 vs SAM2 mask IoU 联合拟合（Nelder-Mead 多起点）；"
                  "叠放链式支撑；soft 约束来自 FK pinch 点",
    }
    vis = []
    for c in views:
        v = imgs[c].copy()
        for n in fit_names:
            cx, cy, yaw, sx, sy, h = p[n]
            mk = masks.get((c, n))
            if mk is not None:
                v[mk] = (0.5 * v[mk] + [0, 100, 0]).astype(np.uint8)
            pm = proj_mask(cams, c, shapes[n], cx, cy, yaw, sx, sy, z0[n], h)
            cnt, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cnt, -1, (0, 0, 255), 1)
        vis.append(cv2.resize(v, (640, 360), interpolation=cv2.INTER_CUBIC))
    for n in fit_names:
        cx, cy, yaw, sx, sy, h = p[n]
        ious = {}
        for c in views:
            mk = masks.get((c, n))
            if mk is not None and mk.sum() >= 8:
                ious[c] = round(float(iou(proj_mask(cams, c, shapes[n], cx, cy, yaw, sx, sy, z0[n], h), mk)), 3)
        entry = {"center": [float(cx), float(cy), float(z0[n] + h / 2)],
                 "yaw_rad": float(yaw), "size": [float(sx), float(sy), float(h)],
                 "shape": shapes[n], "role": role[n], "iou": ious,
                 "bottom_z": float(z0[n])}
        if shapes[n] == "arch":
            entry["shape_note"] = f"复合体近似：顶板 {ARCH_TOP_FRAC:.2f}h、腿宽 {ARCH_LEG_FRAC:.2f}sx（与 sim.py 一致）"
        if shapes[n] in ("cylinder", "prism"):
            entry["shape_degraded"] = f"{shapes[n]} 按箱体近似（mesh 化排在 P3，见 DEVELOPMENT_LOG §61.3）"
        result["objects"][n] = entry
        log(f"{n:16s} 中心 ({cx:.4f}, {cy:.4f}, {z0[n] + h / 2:.4f})  yaw {np.degrees(yaw):+6.1f}°  "
            f"尺寸 {sx*1000:.1f}×{sy*1000:.1f}×{h*1000:.1f} mm  底 z {z0[n]*1000:.1f}  IoU {ious}")

    # 物理一致性校验
    if target in result["objects"]:
        g = result["objects"][target]
        d = np.array(p_grasp) - np.array(g["center"])
        log(f"\n校验：抓取时 pinch 相对 {target} 中心偏差 {np.round(d * 1000, 1)} mm "
            f"（半尺寸 {np.round(np.array(g['size']) / 2 * 1000, 1)} mm）")
        result["check_grasp_pinch_offset_mm"] = (d * 1000).tolist()

    cv2.imwrite(os.path.join(out, "scene_fit.png"), np.hstack(vis))
    json.dump(result, open(os.path.join(out, "scene.json"), "w"), indent=2, ensure_ascii=False)
    log(f"写出 {out}/scene.json, scene_fit.png")

    # ---- 桌面显示/碰撞外形：由物体足迹估计（test4 是手动拟合的 mask 版）----
    if not a.no_table_display:
        xy = np.array([result["objects"][n]["center"][:2] for n in fit_names])
        half = np.array([[result["objects"][n]["size"][0], result["objects"][n]["size"][1]]
                         for n in fit_names]) / 2
        lo = (xy - half).min(0) - 0.18          # 0.18 m 边距
        hi = (xy + half).max(0) + 0.18
        td = {"center_xy": ((lo + hi) / 2).tolist(),
              "size_xy": (hi - lo).tolist(),
              "yaw_rad": 0.0,
              "table_z": float(z_tab),
              "apron_m": 0.12,
              "status": "estimated_from_object_footprint",
              "note": "由物体足迹外扩 0.18 m 估计，非 mask 拟合；test4 的 mask 版见其 "
                      "stage1_scene/table_display.json。若要精确桌形需补 SAM2 桌面 mask。",
              "method": "object footprint + margin"}
        json.dump(td, open(os.path.join(out, "table_display.json"), "w"), indent=2, ensure_ascii=False)
        log(f"写出 {out}/table_display.json（估计值：{np.round(td['size_xy'], 3).tolist()} m）")


if __name__ == "__main__":
    main()
