"""b2s 第 2 步：演示轨迹（末端 / 关节 / 夹爪 / 物体）的提取、交叉校验与修复。

blend 是人或 agent 搭的，出错很正常（ep71 实测两处：夹爪曲线整段全开；绿柱动画 f137-144 发散到
1e27 再变 NaN）。原则：**每条信号都用另一条独立信号核对；不一致就换源或修，修了必须记录**。

夹爪 g(t)（0 开 1 闭）候选，按可信度排序，取第一个通过一致性检查的：
  1. 数据集真值（DROID parquet，且关节角与 blend 逐帧吻合才算同一条）
  2. blend 自带的夹爪曲线（自定义属性 aperture/width/opening/jaw/angle）
  3. 从物体运动反推：物体被搬运期间闭合，前后各留 0.3 s 开合过渡（**估计值**）
  一致性检查：物体「被搬运」的帧（离开起点和终点各 >1 cm 且底面抬离 >5 mm）里，
  g ≥ 0.2 的比例必须 ≥ 90%。

物体轨迹修复：
  无效帧 = 非有限值 / 离基座 >3 m / 相邻有效帧速度 >3 m/s / 穿透桌面 >5 mm /
           抓持搬运中偏离参考抓取位姿 >1 cm（参考 = 抓持搬运帧上相对位姿的中位数）
  无效帧落在夹爪闭合区间 → 用 pinch 点刚性带动（抓取时刻的相对位姿）补
  否则 → 前后有效帧之间位置线性插值 + 旋转 slerp
  物体轨迹**只用于**：推导终态谓词、事件切分、和仿真回放做对照。仿真里的物体运动由物理产生。
"""
import numpy as np

from common import inv, rot_of
import sources as SRC

CARRY_DIST = 0.01
LIFT = 0.005
HOLD_LIFT_FRAC = 0.5     # 持物段 = carry 帧里抬升 > 50% 最高抬升的那部分（排除起手/放下过渡帧）
CLOSED = 0.2
CONSIST_MIN = 0.9
V_MAX = 3.0
GRIP_CMD_LEAD_S = 0.27
GRIP_SQUEEZE = 0.25       # DROID 标定值：在「实测值已停在物体宽度处」的基础上再挤这么多
GRIP_SQUEEZE_M = 0.010    # 无 DROID 指令通道时，挤压力按绝对开口量给（米）。
GRIP_PRECLOSE_S = 1.2     # 真实抓取前多久允许夹爪闭合（秒）；窗口外的闭合视为读数尖峰
FAILED_ATTEMPT_PREROLL_S = 1.5   # 裁掉失败尝试段时，从成功闭合点往前保留多少秒（含接近段）
# 取值依据（UMI ep68 实测）：杯子直径 52.7 mm，UMI 实测开口换算到 2F-85 是 54.5 mm
# （比杯子宽 1.8 mm，只是贴着），-2 mm 只到 52.5 mm 仍然夹不住 → 杯子被推走 330 mm 而不是被抬起。
# 改成 -10 mm 让夹爪真正咬进物体 8 mm，产生夹持力。
# 上限保护：不低于「物体宽度 - 20 mm」，避免细物体被压扁。
PEN_TOL = 0.005
RIGID_TOL = 0.01
LEAD_S = 0.3


def _clean_rot(M):
    M = M.copy()
    ok = np.isfinite(M).all((1, 2))
    for t in np.where(ok)[0]:
        M[t, :3, :3] = rot_of(M[t])
    return M


def object_tracks(ex, scene, T_base, D):
    out = {}
    top = scene["table"]["top_z"]
    for s, o in scene["objects"].items():
        M = _clean_rot(ex.M(o["blend_name"]))
        ok = np.isfinite(M).all((1, 2)) & (np.abs(M[:, :3, 3] - T_base[:3, 3]).max(1) < 3.0)
        # 穿透桌面 >5 mm（ep71 绿柱 f137 底面插进桌面 4 cm，数值上却是有限值）
        below = ex.M0(o["blend_name"])[2, 3] - o["bottom_z"]      # 对象原点到底面（原点不一定在包围盒中心）
        with np.errstate(invalid="ignore"):
            ok &= ~(M[:, 2, 3] - below < top - PEN_TOL)
        last = None
        for t in range(len(M)):
            if not ok[t]:
                continue
            if last is not None:
                v = np.linalg.norm(M[t, :3, 3] - M[last, :3, 3]) * ex.fps / (t - last)
                if v > V_MAX:
                    ok[t] = False
                    continue
            last = t
        bad = np.where(~ok)[0]
        if len(bad):
            D.add(f"track[{s}].invalid", [int(ex.frames[i]) for i in bad],
                  f"blend 动画 {len(bad)} 帧无效（非有限/越界/速度>{V_MAX} m/s）", warn=True)
        out[s] = {"M": M, "valid": ok}
    return out


def carry_frames(track, bottom0):
    """被搬运的帧：离开起点和终点都 >1 cm，且底面比起止的较低者高 >5 mm（只看有效帧）。"""
    M, ok = track["M"], track["valid"]
    vi = np.where(ok)[0]
    p = M[:, :3, 3]
    p0, p1 = p[vi[0]], p[vi[-1]]
    z_low = min(p0[2], p1[2])
    c = np.zeros(len(M), bool)
    c[vi] = (np.linalg.norm(p[vi] - p0, axis=1) > CARRY_DIST) & (np.linalg.norm(p[vi] - p1, axis=1) > CARRY_DIST) \
        & (p[vi, 2] > z_low + LIFT)
    track["lift"] = p[:, 2] - z_low          # 抬升剖面，供 hold_mask 用
    return c


def hold_mask(track, carry, frac=HOLD_LIFT_FRAC):
    """搬运帧里「真正被举着走」的那一段：抬升高度 > frac × 本段最大抬升。

    为什么需要它：carry 的两端是过渡帧 —— 开头刚夹上还没抬起来，结尾已经松开往下放。
    这两处夹爪本来就是开的，拿它们去和 g 比会得到「没夹住」的假阴性。
    实测 UMI ep68：84 个 carry 帧里有 12 帧（811–822）是释放过渡，g 从 0.195 掉到 0.071。
    """
    lift = np.asarray(track.get("lift"), float)
    idx = np.where(carry)[0]
    if len(idx) == 0 or not np.isfinite(lift[idx]).any():
        return carry
    lmax = float(np.nanmax(lift[idx]))
    if not np.isfinite(lmax) or lmax <= 0:
        return carry
    return carry & (lift >= frac * lmax)


def consistency(g, carry, closed=CLOSED):
    """夹爪信号与「持物」是否一致：持物段中闭合(g>=CLOSED)的比例。"""
    if carry.sum() == 0:
        return None
    return float(np.mean(g[carry] >= closed))


def blend_gripper_curves(ex):
    out = []
    for own, keys in ex.E["props_anim"].items():
        for k, v in keys.items():
            v = np.asarray(v, float)
            kl = k.lower()
            if kl in ("g", "open", "开口", "开合", "observation.gripper_position"):  # 0=开 / 1=闭
                out.append((f"blend:{own}.{k}", np.clip(v, 0, 1), 0))
            elif any(w in kl for w in ("aperture", "width", "opening")):
                a = v / 1000 if np.nanmax(np.abs(v)) > 1.5 else v       # mm → m
                g = 1 - np.clip(a / SRC.GRIPPER_MAX_APERTURE["robotiq_2f85"], 0, 1)
                out.append((f"blend:{own}.{k}", g, 0))
            elif any(w in kl for w in ("jaw", "angle", "grip")):
                out.append((f"blend:{own}.{k}", np.clip(v / 0.8, 0, 1), 1))   # 2F-85 driver 行程 ≈0.8 rad
    out.sort(key=lambda x: x[2])        # 开口类优先于角度类
    return [(n, g) for n, g, _ in out]


def infer_gripper(carry_any, fps):
    g = carry_any.astype(float)
    if not carry_any.any():
        return g
    lead = max(1, int(round(LEAD_S * fps)))
    idx = np.where(carry_any)[0]
    s, e = idx[0], idx[-1]
    g[s:e + 1] = 1.0
    for k in range(1, lead + 1):
        if s - k >= 0:
            g[s - k] = max(g[s - k], 1 - k / (lead + 1))
        if e + k < len(g):
            g[e + k] = max(g[e + k], 1 - k / (lead + 1))
    return g


def episode_index(ex, cfg, blend_dir):
    import json
    import os
    if cfg.get("episode_index") is not None:
        return int(cfg["episode_index"]), "b2s_config"
    v = ex.E["scene"]["props"].get("episode_index")
    if v is not None:
        return int(v), "blend 场景属性 episode_index"
    sj = os.path.join(blend_dir, "source.json")
    if os.path.exists(sj):
        j = json.load(open(sj))
        if "episode_index" in j:
            return int(j["episode_index"]), "blend 同目录 source.json"
    return None, "无"


def _slerp_fill(M, ok):
    from scipy.spatial.transform import Rotation as Rot, Slerp
    vi = np.where(ok)[0]
    out = M.copy()
    for t in np.where(~ok)[0]:
        a = vi[vi < t].max() if (vi < t).any() else vi.min()
        b = vi[vi > t].min() if (vi > t).any() else vi.max()
        if a == b:
            out[t] = M[a]
            continue
        al = (t - a) / (b - a)
        out[t] = np.eye(4)
        out[t, :3, 3] = (1 - al) * M[a, :3, 3] + al * M[b, :3, 3]
        out[t, :3, :3] = Slerp([0, 1], Rot.from_matrix([M[a, :3, :3], M[b, :3, :3]]))(al).as_matrix()
    return out


def repair_track(tr, g, ee, carry, D, name, frames, closed=CLOSED):
    """抓持且搬运的帧上，物体相对 pinch 应刚性不变：取 X 的中位数为参考抓取位姿，
    偏离 >1 cm 的帧判为无效（ep71 f107-113 物体比夹爪先抬起 3 cm），再统一修补。"""
    M, ok = tr["M"].copy(), tr["valid"].copy()
    held = carry & (g >= closed)
    X_ref = None
    hv = np.where(held & ok)[0]
    if len(hv) >= 3:
        X = np.array([inv(ee[t]) @ M[t] for t in hv])
        med = np.median(X[:, :3, 3], axis=0)
        k = hv[np.argmin(np.linalg.norm(X[:, :3, 3] - med, axis=1))]
        X_ref = inv(ee[k]) @ M[k]
        dev = np.linalg.norm(X[:, :3, 3] - X_ref[:3, 3], axis=1)
        bad_rigid = hv[dev > RIGID_TOL]
        tr["grasp_X"] = X_ref
        tr["grasp_X_spread_mm"] = float(np.median(dev) * 1000)
        if len(bad_rigid):
            ok[bad_rigid] = False
            D.add(f"track[{name}].rigid_violation", [int(frames[t]) for t in bad_rigid],
                  f"抓持搬运中相对 pinch 偏离参考抓取 >{RIGID_TOL*1000:.0f} mm（最大 {dev.max()*1000:.1f} mm）", warn=True)
    tr["valid_final"] = ok
    if ok.all():
        tr["M_fixed"] = M
        return
    rigid, interp = [], []
    filled = _slerp_fill(M, ok)
    for t in np.where(~ok)[0]:
        if X_ref is not None and g[t] >= closed:
            filled[t] = ee[t] @ X_ref
            rigid.append(int(frames[t]))
        else:
            interp.append(int(frames[t]))
    tr["M_fixed"] = filled
    D.add(f"track[{name}].repair", {"rigid_with_pinch": rigid, "interpolated": interp},
          "夹爪闭合期间 = pinch × 参考抓取位姿，其余前后有效帧插值+slerp", fallback=True)


def _segments(mask):
    """连续 True 段的 [起, 止]（含端点）列表。"""
    idx = np.where(np.asarray(mask, bool))[0]
    if len(idx) == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[idx[0]], idx[cuts + 1]])
    ends = np.concatenate([idx[cuts], [idx[-1]]])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def trim_failed_attempts(demo, scene, fps, D):
    """裁掉「真正的抓取」之前那段失败的尝试。

    为什么需要：一段人采的演示里常常先有一次夹空。实测 UMI ep68 —— 手持夹爪在 f348 合拢但没夹住
    （开口只剩 4.8 mm 却仍在杯子旁 3 cm 外），顺手把杯子推出 10 cm；f684 才真正夹住并抬起 5.6 cm。
    参考轨迹里这段位移是真的（杯子 f0 y=0.072 → f390 y=0.175），不是读数噪声。

    为什么仿真里必须裁：人手闭合后只有约 2 cm 宽，2F-85 全开 85 mm。同一条末端轨迹下，机器人的
    夹爪外壳（左右 coupler / spring_link）在 f434 就扫到杯子，一路把杯子蹭倒到 90°；而它离 pinch
    还有 6–19 cm —— 也就是说这不是「抓取参数不对」，是**尺寸差异导致第一次试探必然碰撞**。
    实测：整段回放杯子倾斜 90°、位移 172 mm；从成功抓取前 1.5 s 起回放，抬升 55.9 mm、
    位移 284 mm、四个谓词全过。

    裁剪条件（两条都满足才裁，避免误删真实数据）：
      ① hold（真把物体抬起来的那段）存在；
      ② hold 之前存在闭合段，且该段与 hold 不重叠 —— 那是没夹住的试探。
    裁剪点 = max(hold 起点 − FAILED_ATTEMPT_PREROLL_S, 最后一个失败段结束 + 1)。
    """
    hold = demo.get("_hold_any")
    if hold is None or not np.asarray(hold).any():
        return 0
    segs = _segments(hold)
    if not segs:
        return 0
    hs = segs[0][0]
    closed_segs = [s for s in _segments(demo["g"] >= demo.get("closed_threshold", CLOSED)) if s[1] < hs]
    if not closed_segs:
        return 0
    pre = int(round(FAILED_ATTEMPT_PREROLL_S * fps))
    t0 = max(hs - pre, closed_segs[-1][1] + 1, 0)
    if t0 <= 0:
        return 0
    n_drop = t0
    demo["trim_offset"] = int(t0)
    # 裁掉前缀：所有逐帧数组 + 事件。物体初始位姿随之移到裁剪后的第一帧（刚性物体的局部偏移不变）。
    for k in ("ee_world", "g", "g_cmd", "g_external", "g_hand"):
        if k in demo and demo[k] is not None and len(demo[k]) == len(demo["frames"]):
            demo[k] = np.asarray(demo[k])[t0:]
    demo["q"] = None if demo.get("q") is None else np.asarray(demo["q"])[t0:]
    demo["frames"] = np.asarray(demo["frames"])[t0:]
    for s in list(demo["objects"]):
        M = np.asarray(demo["objects"][s])
        cloc = M[0][:3, :3].T @ (np.array(scene["objects"][s]["center"]) - M[0][:3, 3])
        demo["objects"][s] = M[t0:]
        demo["objects_valid"][s] = np.asarray(demo["objects_valid"][s])[t0:]
        Mn = demo["objects"][s][0]
        c_old = np.asarray(scene["objects"][s]["center"], float)
        c_new = Mn[:3, :3] @ cloc + Mn[:3, 3]
        scene["objects"][s]["center"] = c_new.tolist()
        # bottom_z / top_z 是**物体顶点的世界包围盒**上下界（parse_scene 里算的），属于同一批
        # 随裁剪一起平移的量。漏掉它们 → 物体按旧包围盒摆进桌面。
        # 实测 UMI ep68：只更 center 不更 bottom/top，杯底被摆进桌面 1.000 mm，
        # V1 静置里杯子在 1 mm 深处做 1 s 周期的**不衰减**来回摇摆（|v|=12.4 mm/s、|w|=0.53 rad/s），
        # 怎么调 solref/solimp 都没用 —— 是初始位姿插进桌子里，不是接触参数问题。
        dz = float(c_new[2] - c_old[2])
        for k_ in ("bottom_z", "top_z"):
            if k_ in scene["objects"][s]:
                scene["objects"][s][k_] = float(scene["objects"][s][k_]) + dz
    if demo.get("events"):
        f_cut = float(demo["frames"][0])
        demo["events"] = [e for e in demo["events"] if e["frame"] >= f_cut]
    demo.pop("_hold_any", None)
    D.add("demo.trim_failed_attempt", {"dropped_frames": n_drop,
                                        "failed_close_segments": closed_segs[:4],
                                        "kept_from": int(demo["frames"][0])},
          f"成功抓取（hold {hs}）之前有 {len(closed_segs)} 段「合拢但没夹住」的试探，"
          f"那段人把物体推走了 {0:.0f}mm 级位移且机器人夹爪外壳比人手宽 4 倍、必然碰撞"
          f"→ 从成功抓取前 {FAILED_ATTEMPT_PREROLL_S}s 起裁剪，物体初始位姿同步移到裁剪首帧",
          warn=True)
    return t0


def build(ex, emb, robot, scene, cfg, D, blend_dir):
    T_base = scene["T_world_base"]
    demo = {"fps": ex.fps, "frames": ex.frames, "embodiment": emb["type"]}
    grip_cfg = cfg.get("gripper", {})
    closed_threshold = float(grip_cfg.get("closed_threshold", CLOSED))
    demo["closed_threshold"] = closed_threshold
    ext = cfg.get("demo")
    if ext:
        pass        # 末端轨迹来自外部数据，见下
    elif emb["type"] == "robot_arm":
        demo["q"] = robot["q"]
        demo["ee_world"] = np.array([T_base @ T for T in robot["fk"]["pinch"]])
        demo["ee_source"] = "blend 关节角 → MuJoCo FK（pinch site）"
    elif emb["type"] == "human_hand":
        T, g_hand, ap = SRC.blend_hand_to_canonical(ex, emb["armature"], D)
        demo["q"] = None
        demo["ee_world"], demo["g_hand"] = T, g_hand
        demo["ee_source"] = "blend 手骨架 → 拇指/食指捏合点（规范末端系）"
    elif emb["type"] == "handheld_gripper":
        M = _clean_rot(ex.M(emb["root"]))
        X = np.array(cfg.get("embodiment", {}).get("root_to_pinch", np.eye(4).tolist()))
        demo["q"] = None
        demo["ee_world"] = np.array([m @ X for m in M])
        demo["ee_source"] = f"手持夹爪根对象 {emb['root']!r} × root_to_pinch"
        if "root_to_pinch" not in cfg.get("embodiment", {}):
            D.add("ee.root_to_pinch", "单位阵", "未给定夹爪根对象到捏合点的变换，按根对象原点=捏合点", warn=True,
                  fallback=True)
    if ext:        # 外部数据文件（UMI zarr / EgoDex hdf5 / 通用 npz）覆盖 blend 里的末端轨迹
        A = np.array(ext.get("T_blend_from_data", np.eye(4).tolist()))
        demo["q"] = None
        if ext["type"] == "npz":    # 通用口：ee_world (T,4,4) 规范末端系 + g (T,) 实测开合
            z = np.load(ext["path"])
            T, g = z["ee_world"], z["g"]
        elif ext["type"] == "umi":
            T, g = SRC.load_umi(ext["path"], ext.get("episode", 0))
        elif ext["type"] == "egodex":
            T, g, _ = SRC.load_egodex(ext["path"], ext.get("hand", "right"))
        demo["ee_world"] = np.array([A @ t for t in T])
        demo["g_external"] = g
        demo["ee_source"] = f"外部 {ext['type']} 数据 {ext['path']}"
        if "T_blend_from_data" not in ext:
            D.add("demo.alignment", "单位阵", "外部数据坐标系与 blend 世界系的对齐未给定", warn=True, fallback=True)
    if "ee_source" not in demo:
        raise ValueError("Cannot extract end-effector motion: specify embodiment or an external demo in config")
    D.add("ee", demo["ee_source"], f"{len(demo['ee_world'])} 帧")

    tracks = object_tracks(ex, scene, T_base, D)
    carry = {s: carry_frames(tr, scene["objects"][s]["bottom_z"]) for s, tr in tracks.items()}
    moved = [s for s in tracks if carry[s].any()]
    carry_any = np.any([carry[s] for s in moved], axis=0) if moved else np.zeros(ex.T, bool)
    D.add("moved_objects", moved, f"被搬运帧数 {[int(carry[s].sum()) for s in moved]}")

    # 持物段（排除起手/放下的过渡帧）；夹爪一致性只在这段上判，否则释放时夹爪正常张开会被算成「没夹住」
    hold = {s: hold_mask(tr, carry[s]) for s, tr in tracks.items()}
    hold_any = np.any([hold[s] for s in moved], axis=0) if moved else np.zeros(ex.T, bool)
    if moved:
        D.add("hold_mask", {s: [int(np.where(hold[s])[0][0]), int(np.where(hold[s])[0][-1])]
                            for s in moved if hold[s].any()},
              f"持物段 = carry 帧中抬升 > {HOLD_LIFT_FRAC}×最大抬升；夹爪一致性只在此段统计"
              f"（carry {int(carry_any.sum())} 帧 → hold {int(hold_any.sum())} 帧）")

    # ---- 夹爪候选
    cands = []
    npz_ep = None
    if cfg.get("droid_trajectory"):
        npz_ep, why = SRC.load_droid_npz(cfg["droid_trajectory"], demo.get("q"), ex.frames, ex.fps)
        cands.append(("dataset:droid_npz", npz_ep["observation.state.gripper_position"], cfg["droid_trajectory"], why))
        demo["dataset"] = {"path": cfg["droid_trajectory"], "format": "droid_npz", "joints_verified": why}
        demo["dataset_instruction"] = npz_ep.get("language_instruction", [""])[0]
        D.add("dataset", cfg["droid_trajectory"], why)
    ep_idx, ep_why = episode_index(ex, cfg, blend_dir)
    if npz_ep is None and ep_idx is not None and demo.get("q") is not None:
        p, ep, why = SRC.find_droid_episode(ep_idx, demo["q"], roots=cfg.get("dataset_roots"))
        if p:
            cands.append(("dataset:droid", ep["observation.state.gripper_position"].ravel(), p, why))
            demo["dataset_episode"] = p
            demo["dataset"] = {"path": p, "episode_index": ep_idx, "joints_verified": why}
        else:
            D.add("dataset", None, f"episode {ep_idx}（来源：{ep_why}）：{why}", warn=True)
    for k in ("g_external", "g_hand"):
        if k in demo:
            cands.append((k, demo[k], None, "外部/人手数据"))
    for n, g in blend_gripper_curves(ex):
        cands.append((n, g, None, "blend 自定义属性曲线"))
    if grip_cfg.get("aperture_markers"):
        a, b = grip_cfg["aperture_markers"]
        width = np.linalg.norm(ex.M(a)[:, :3, 3] - ex.M(b)[:, :3, 3], axis=1)
        width -= float(grip_cfg.get("marker_offset_m", 0))
        if not np.isfinite(width).all() or (width < 0).any():
            raise ValueError("Invalid aperture marker distances")
        maximum = SRC.GRIPPER_MAX_APERTURE["robotiq_2f85"]
        g = 1 - np.clip(width / maximum, 0, 1)
        cands.append(("blend:aperture_markers", g, None, "指间标记距离换算到目标夹爪开口"))
        D.add("gripper.aperture", {"markers": [a,b], "range_m": [float(width.min()),float(width.max())]},
              "按配置的两指标记测量开口；超出目标夹爪最大开口的部分截断", warn=bool((width > maximum).any()))
    chosen = None
    report = []
    for n, g, p, why in cands:
        c = consistency(g, hold_any, closed_threshold)
        ok = c is None or c >= CONSIST_MIN
        report.append({"source": n, "consistency": c, "range": [float(np.nanmin(g)), float(np.nanmax(g))],
                       "passed": ok, "why": why})
        if ok and chosen is None:
            chosen = (n, g)
    for r in report:
        D.add(f"gripper.candidate[{r['source']}]", "通过" if r["passed"] else "不一致",
              f"搬运帧中闭合比例 {r['consistency']}；取值范围 {np.round(r['range'], 3).tolist()}",
              warn=not r["passed"])
    if chosen is None:
        g = infer_gripper(carry_any, ex.fps)
        chosen = ("inferred:object_motion", g)
        D.add("gripper", chosen[0], "所有候选都与物体运动不一致，按搬运区间反推（估计值，非真值）",
              fallback=True, warn=True)
    else:
        D.add("gripper", chosen[0], "第一个通过一致性检查的候选（按 数据集 > 外部/人手 > blend 排序）",
              fallback=chosen[0] != (cands[0][0] if cands else None))
    demo["g"], demo["g_source"], demo["g_candidates"] = np.asarray(chosen[1], float), chosen[0], report

    # ---- 夹爪**指令**（回放用）。实测开口在夹住物体后停在物体宽度处（ep71：实测 0.661、指令 0.911）；
    # 把实测值当位置伺服目标 = 目标开口≈物体宽度 → 几乎没有夹持力，物体滑落（ep71 实测：塔被碰倒）。
    gcmd, why = None, ""
    if chosen[0] == "dataset:droid_npz":
        gcmd = npz_ep["action.gripper_position"].copy()
        why = "已核对的 DROID NPZ action_dict.gripper_position；保留原始指令，不用实测开口替代"
    if demo.get("dataset") and chosen[0] == "dataset:droid":
        import droid_common as C
        ep = C.load_episode(demo["dataset"]["path"])
        if "action.gripper_position" in ep:
            gcmd = np.asarray(ep["action.gripper_position"], float).ravel()
            gm = demo["g"]
            lag = min(range(0, 10), key=lambda k: np.abs(gm[k:] - np.minimum(gcmd[:len(gm) - k], gm.max())).sum())
            sq = float(np.median((gcmd - gm)[carry_any])) if carry_any.any() else None
            why = (f"数据集 action.gripper_position（遥操作指令）；实测：指令领先 {lag} 帧、搬运段指令−实测 "
                   f"{sq if sq is None else round(sq, 3)}（推断常数为 {GRIP_CMD_LEAD_S}s / {GRIP_SQUEEZE}）")
    if gcmd is None:
        # 无指令通道：指令 = 实测提前 LEAD、闭合段 + SQUEEZE。两个常数由 DROID ep71 的
        # action vs observation 标定（指令领先 4 帧@15Hz；夹住后指令 0.911、实测 0.661）。
        # 合成 UMI 对照实测（同一条末端轨迹、retarget 后回放）：
        #   全闭 1.0（无论是否提前）→ 失败（2F-85 满行程把 27 mm 圆柱挤歪）；仅实测值 → 失败（无夹持力）
        #   实测+0.25 → 成功；再提前 4 帧 → 成功且穿透 4.3→2.6 mm
        L = int(round(float(grip_cfg.get("command_lead_s", GRIP_CMD_LEAD_S)) * ex.fps))
        gs = np.concatenate([demo["g"][L:], np.repeat(demo["g"][-1:], L)]) if L else demo["g"].copy()
        gcmd = gs.copy()
        closed = gs >= closed_threshold
        # 挤压力必须按**物理开口**给，不能直接搬 g 的增量：
        #   UMI 的 g 是 1 - w/80mm，仿真里的 2F-85 是 1 - w/85mm —— 同一个 g 在两个基准下相差 3.4 mm。
        #   ep71 的 +0.25 是在「DROID 实测 g 已停在物体宽度」的前提下标定的；UMI 也满足这个前提，
        #   所以这里把「实测开口 - GRIP_SQUEEZE_M」换算成指令 g，而不是给 g 加 0.25
        #   （直接 +0.25 会把 55 mm 的开口压到 37 mm，52.7 mm 的杯子会被挤歪推倒）。
        source_aperture = (SRC.GRIPPER_MAX_APERTURE["robotiq_2f85"] if chosen[0] == "blend:aperture_markers"
                           else float(grip_cfg.get("source_max_aperture_m", SRC.GRIPPER_MAX_APERTURE["umi"])))
        squeeze = float(grip_cfg.get("squeeze_m", GRIP_SQUEEZE_M))
        ap_meas = (1.0 - gs) * source_aperture
        # 挤压力只在**已经合到物体宽度**之后才施加。
        # 实测踩坑：接近段实测开口 55 mm（比杯径 52.7 mm 还宽 2 mm，本来就只是贴着），
        # 无条件减 GRIP_SQUEEZE_M 会把接近段压到 45 mm —— 夹爪还没靠近杯子就先收窄，
        # 一接触就把杯子撞飞。所以先夹到 ap_meas，只有当 ap_meas 已经 <= 物体宽度时才追加挤压力。
        # cylinder 的 size 是 [**半径**, 高]（sim.py 里就是 size=[r, h/2, 0] 这样用的）。
        # 踩过两次坑，方向相反，都错在同一个字段上：
        #   ① 写成 size[0]*2 的时候我以为它已经是直径 → obj_w=0.105，gate 永远成立
        #   ② 后来「修正」成 size[0] → obj_w=0.0264，gate 永远**不**成立，挤压力一次都没生效
        # 实测 UMI ep68：obj_w=26.4mm 时 gate 永假，指令开口停在 55.1mm（杯径 52.7mm），
        # 夹爪只是贴着杯子，重放时杯子几乎不动（抬升 0.6mm、倾斜 4.9°）。
        obj_w = None
        for s_ in scene["objects"]:
            sz = scene["objects"][s_].get("size") or scene["objects"][s_].get("dims")
            if sz:
                w = 2.0 * float(sz[0]) if scene["objects"][s_].get("shape") == "cylinder" \
                    else float(min(sz[0], sz[1]))
                obj_w = w if obj_w is None else min(obj_w, w)
        if obj_w is not None:
            ap_cmd = np.where(ap_meas <= obj_w + 0.004,
                              np.maximum(ap_meas - squeeze, obj_w - 0.020),
                              ap_meas)
        else:
            ap_cmd = np.maximum(ap_meas - squeeze, 0.0)
        g_squeeze = 1.0 - ap_cmd / SRC.GRIPPER_MAX_APERTURE["robotiq_2f85"]
        # ★ 用「物体真被抬起」校正闭合区间，而不是直接信 g 曲线。
        # UMI ep68 实测：g 在 367-371 帧有虚假尖峰（开口骤降到 4.8 mm），但那几帧杯子没动。
        # 照 g 闭合 → 夹爪在离杯子还差 30 mm 时就闭合，把杯子打飞 145 mm，
        # 真正的搬运段（741-803）开始时杯子已经不在原位了。
        gcmd[closed] = np.clip(g_squeeze[closed], 0, 1)
        if grip_cfg.get("filter_spurious_close", True) and hold_any.any() and closed.any():
            hi = np.where(hold_any)[0]
            w0 = max(0, hi[0] - int(round(GRIP_PRECLOSE_S * ex.fps)))
            w1 = min(len(closed), hi[-1] + 1)
            spurious = closed.copy()
            spurious[w0:w1] = False
            if spurious.any():
                # 回退必须是「张开」，不能回退成 gs —— gs 本身就是含尖峰的那条曲线，
                # 拿它当回退值等于什么都没做（实测 g_cmd[360] 仍是 0.94，对应开口 5 mm，
                # 机械臂还没到杯子就把夹爪合死了）。
                gcmd[spurious] = 0.0
                D.add("gripper.spurious_close",
                      {"frames": np.where(spurious)[0][:20].tolist(), "n": int(spurious.sum())},
                      f"g 曲线在这些帧闭合，但物体并未被抬起（hold 段 {int(hi[0])}-{int(hi[-1])}）"
                      f"→ 判为读数尖峰，指令回退为实测开口；否则夹爪会在对准前闭合把物体打飞",
                      warn=True)
        why = (f"无指令通道：实测提前 {L} 帧；挤压力按物理开口给 "
               f"（源开口基准 {source_aperture*1000:.0f}mm，减 {squeeze*1000:.0f}mm 后换算到 2F-85）")
        D.add("gripper.command", "inferred", why, fallback=True)
    else:
        D.add("gripper.command", "dataset", why)
    demo["g_cmd"] = gcmd
    held = demo["g"][carry_any] if carry_any.any() else np.array([])
    if len(held):
        demo["stall_aperture_m"] = float((1 - np.median(held)) * SRC.GRIPPER_MAX_APERTURE["robotiq_2f85"])

    for s, tr in tracks.items():
        repair_track(tr, demo["g"], demo["ee_world"], carry[s], D, s, ex.frames, closed_threshold)
    demo["objects"] = {s: tr["M_fixed"] for s, tr in tracks.items()}
    demo["objects_valid"] = {s: tr["valid_final"] for s, tr in tracks.items()}
    demo["grasp_X"] = {s: tr["grasp_X"] for s, tr in tracks.items() if "grasp_X" in tr}
    demo["moved"] = moved
    demo["_hold_any"] = hold_any
    # 事件：夹爪闭合/张开的帧
    gc = demo["g"] >= closed_threshold
    ev = []
    for t in range(1, ex.T):
        if gc[t] and not gc[t - 1]:
            ev.append({"frame": int(ex.frames[t]), "event": "grasp_close"})
        if not gc[t] and gc[t - 1]:
            ev.append({"frame": int(ex.frames[t]), "event": "release_open"})
    demo["events"] = ev
    D.add("events", ev, f"夹爪阈值 {closed_threshold}")
    return demo
