"""b2s 第 1b 步：场景解析（世界系 = blend 世界系；换到机器人基座系在 build_spec 里做）。

全部判断都写进 Decisions，没有任何按物体名的硬编码。b2s_config.json 可覆盖：
  "table": "<blend 对象名>"          指定桌面
  "objects": ["<blend 对象名>", ...]  指定任务物体
  "exclude": ["<blend 对象名>", ...]  排除
  "shapes": {"<blend 对象名>": "box|cylinder|prism|arch"}

★ 朝向一律取 blend 作者给的局部坐标系（matrix_world + 局部包围盒），**不做 OBB / PCA 拟合**
  （CLAUDE.md：OBB 被用户明确拒绝过）。blend 是人工/agent 建的，局部系本身就是语义朝向。

识别流程
  1. 候选物体：非机器人/夹爪/相机的网格，最大边 1–25 cm，离基座 1.1 m 内
  2. 候选桌面：静止、水平（最薄轴≈世界 z）、厚 <10 cm、面积 >0.04 m² 的网格
  3. 支撑关系：a 的底面与 b 的顶面相差 <6 mm 且 a 的质心落在 b 的水平投影内 → a on b
  4. 桌面 = 直接或间接支撑候选物体最多的那块；任务物体 = 以桌面为根的支撑树上的候选
  5. 形状：按体积填充率、凸包填充率、水平截面圆度、顶脊、底部开口分类到 sim.py 支持的
     box / cylinder / prism / arch；都不像 → 标 unsupported（按 box 建，quality_gate 要求替换资产）
  6. 静态障碍：桌面附近（桌面投影外扩 15 cm、桌面以上 0.8 m）其他静止网格 → 碰撞 box
  7. 相机：全部 CAMERA；挂在机器人末端下的 → mount=link7，其余 → world
"""
import math

import numpy as np

from common import inv, rot_of, scale_of, slug, srgb_from_linear

OBJ_MAX_DIM = (0.01, 0.25)
TABLE_MAX_THICK = 0.10
TABLE_MIN_AREA = 0.04
SUPPORT_TOL = 0.006
STATIC_PAD_XY = 0.15
STATIC_Z = 0.8
REACH = 1.1


def _frame(ex, name, fi=0):
    M = ex.M(name)[fi]
    S = scale_of(M)
    R = rot_of(M)
    v = np.array(ex.E["meshes"][name]["verts_local"], float) * S    # 作者局部系（已乘缩放）
    lo, hi = v.min(0), v.max(0)
    c_loc = (lo + hi) / 2
    return {"R": R, "t": M[:3, 3], "S": S, "v": v - c_loc, "dims": hi - lo,
            "center": M[:3, 3] + R @ c_loc}


def _world_verts(f):
    return f["v"] @ f["R"].T + f["center"]


def _in_hull_xy(p, pts):
    from scipy.spatial import Delaunay
    try:
        return bool(Delaunay(pts[:, :2]).find_simplex(p[:2]) >= 0)
    except Exception:
        lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
        return bool((p[:2] >= lo).all() and (p[:2] <= hi).all())


def _canon(f):
    """返回 (k_up, up_sign, tilt_deg, (ka, kb))：作者局部系中哪根轴朝上。"""
    z = f["R"].T @ np.array([0, 0, 1.0])       # 世界 z 在局部系的表达
    k = int(np.argmax(np.abs(z)))
    tilt = math.degrees(math.acos(min(1.0, abs(z[k]))))
    rest = [i for i in range(3) if i != k]
    return k, float(np.sign(z[k])), tilt, rest


def classify(ex, name, f, forced=None):
    k, sg, tilt, (ka, kb) = _canon(f)
    v = f["v"]
    d = f["dims"]
    u = v[:, k] * sg
    a, b = v[:, ka], v[:, kb]
    da, db, h = d[ka], d[kb], d[k]
    bbox_v = float(np.prod(d))
    me = ex.E["meshes"][name]
    vol = me["volume_local"] * float(np.prod(f["S"])) if me["volume_local"] else None
    fill = vol / bbox_v if vol else None
    try:
        from scipy.spatial import ConvexHull
        hull_fill = ConvexHull(v).volume / bbox_v
    except Exception:
        hull_fill = None
    r = np.sqrt((a / (da / 2)) ** 2 + (b / (db / 2)) ** 2)
    rim = r[r > 0.5]
    circ = float(np.mean(np.abs(rim - 1) < 0.06)) if len(rim) else 0.0
    corner = float(np.mean(r > 1.15))
    top = np.abs(u - h / 2) < 0.08 * h
    spread_a = np.ptp(a[top]) / da if top.any() else 1
    spread_b = np.ptp(b[top]) / db if top.any() else 1
    bot = np.abs(u + h / 2) < 0.08 * h
    inner_a = float(np.mean(np.abs(a[bot]) < 0.42 * da)) if bot.any() else 0
    inner_b = float(np.mean(np.abs(b[bot]) < 0.42 * db)) if bot.any() else 0
    feat = {"fill": fill, "hull_fill": hull_fill, "circularity": circ, "corner_frac": corner,
            "top_spread": [float(spread_a), float(spread_b)], "bottom_inner": [inner_a, inner_b],
            "aspect_ab": float(min(da, db) / max(da, db)), "tilt_deg": tilt, "n_verts": me["n_verts"]}
    if forced:
        shape, why = forced, "b2s_config 指定"
    elif corner < 0.02 and feat["aspect_ab"] > 0.9 and (
            circ > 0.8 or (hull_fill is not None and abs(hull_fill - math.pi / 4) < 0.05)):
        # 倒角圆柱的倒角环使圆度降到 ~0.75（ep71 绿柱），但凸包填充率≈π/4 且无角点同样能确定
        shape, why = "cylinder", f"水平截面圆度 {circ:.2f}、凸包填充 {hull_fill:.3f}（π/4=0.785）、无角点"
    elif hull_fill is not None and hull_fill < 0.65 and min(spread_a, spread_b) < 0.2:
        shape, why = "prism", f"凸包填充 {hull_fill:.2f}≈0.5、顶部成脊"
    elif hull_fill is not None and fill is not None and hull_fill > 0.85 and fill < 0.85 * hull_fill:
        shape, why = "arch", f"凸包填充 {hull_fill:.2f} 但实体填充 {fill:.2f}（下方有空腔）"
    elif (fill is not None and fill > 0.85) or (fill is None and hull_fill and hull_fill > 0.9):
        shape, why = "box", f"体积填充 {fill if fill is not None else hull_fill:.2f}"
    else:
        shape, why = "unsupported", f"不像任何支持的形状 fill={fill} hull={hull_fill}"
    # 选局部 x 轴（sim.py 约定）：prism = 脊方向；arch = 两腿分开的方向；其他取较长水平边
    if shape == "prism":
        kx = ka if spread_a > spread_b else kb
    elif shape == "arch":
        kx = ka if inner_a > inner_b else kb
    else:
        kx = ka if da >= db else kb
    ky = kb if kx == ka else ka
    xw = f["R"][:, kx]
    yaw = math.atan2(xw[1], xw[0])
    if shape == "cylinder":
        size = [float((da + db) / 4), float(h)]
    else:
        size = [float(d[kx]), float(d[ky]), float(h)]
    return {"shape": shape, "why": why, "size": size, "yaw": yaw, "tilt_deg": tilt, "features": feat}


def _color(ex, name):
    mats = ex.E["meshes"][name]["materials"]
    if not mats:
        return [0.6, 0.6, 0.6, 1.0], "无材质"
    m = mats[0]
    if m.get("image_mean_srgb"):
        return [*map(float, m["image_mean_srgb"]), 1.0], f"贴图均值（{m['name']}）"
    lin = m.get("ramp_mean_linear") if m.get("textured") and m.get("ramp_mean_linear") else m["base_color_linear"]
    return [*map(float, srgb_from_linear(lin)), 1.0], f"材质 {m['name']!r} 线性色→sRGB"


def parse(ex, emb, robot, cfg, D, ee_world=None):
    """ee_world: (T,3) 末端（pinch 点）世界轨迹，用于在没有物体动画时定位任务桌面。"""
    T_base = robot["T_world_base"] if robot else np.eye(4)
    base_xy = T_base[:2, 3]
    excl = set(cfg.get("exclude", []))
    # 机器人/夹爪/相机支架：骨架及其后代、法兰后代、末端执行器根
    robot_set = set()
    if emb.get("armature"):
        robot_set |= {emb["armature"], *ex.descendants(emb["armature"])}
    if emb.get("root"):
        robot_set |= {emb["root"], *ex.descendants(emb["root"])}
    for r in cfg.get("exclude_trees", []):          # 例如合成测试里把 blend 原有的机械臂整棵排除
        robot_set |= {r, *ex.descendants(r)}
    robot_set |= {n for n in ex.E["dynamic"] if any(a in robot_set for a in ex.ancestors(n))}

    meshes = [n for n in ex.E["meshes"] if n not in robot_set and n not in excl
              and not ex.O[n]["hide_render"] and ex.E["meshes"][n]["n_verts"] >= 4]
    F = {n: _frame(ex, n) for n in meshes}

    obj_c = [n for n in meshes if OBJ_MAX_DIM[0] <= F[n]["dims"].max() <= OBJ_MAX_DIM[1]
             and F[n]["dims"].min() > 0.004 and np.linalg.norm(F[n]["center"][:2] - base_xy) < REACH]
    tab_c = []
    for n in meshes:
        if ex.is_dynamic(n):
            continue
        k, _, tilt, rest = _canon(F[n])
        d = F[n]["dims"]
        if tilt < 3 and d[k] < TABLE_MAX_THICK and d[rest[0]] * d[rest[1]] > TABLE_MIN_AREA:
            tab_c.append(n)
    wv = {n: _world_verts(F[n]) for n in set(obj_c) | set(tab_c)}
    top = {n: wv[n][:, 2].max() for n in wv}
    bot = {n: wv[n][:, 2].min() for n in wv}

    def supports(a, b):
        return abs(bot[a] - top[b]) < SUPPORT_TOL and _in_hull_xy(F[a]["center"], wv[b])

    sup = {a: [b for b in obj_c + tab_c if b != a and supports(a, b)] for a in obj_c}

    def tree(root):
        got, frontier = set(), {root}
        while frontier:
            nxt = {a for a in obj_c if a not in got and any(b in frontier for b in sup[a])}
            got |= nxt
            frontier = nxt
        return got

    if cfg.get("table"):
        table = cfg["table"]
        D.add("table", table, "b2s_config 指定")
    else:
        if not tab_c:
            raise SystemExit("没有找到水平桌面候选；请在 b2s_config.json 指定 table")
        # 任务桌面 = 演示真正操作的那批物体所在的支撑面（不是「支撑物体最多」：
        #   实测 DROID ep71 里机器人推车顶板上的夹具/底座固定块比桌上积木还多，会被误选）
        trees = {t: tree(t) for t in tab_c}
        # nanmax：blend 动画可能有发散/NaN 帧（ep71 绿柱 f137-144 发散到 1e27 再变 NaN），这里只问「动没动」
        moved = [n for n in obj_c if ex.is_dynamic(n)
                 and np.nanmax(np.linalg.norm(ex.M(n)[:, :3, 3] - ex.M(n)[0, :3, 3], axis=1)) > 0.005]
        with_moved = [t for t in tab_c if any(m in trees[t] for m in moved)]
        if with_moved:
            table = max(with_moved, key=lambda t: len(trees[t]))
            D.add("table", table, f"支撑树包含被演示移动的物体 {moved}")
        elif ee_world is not None and len(ee_world):
            def dmin(t):
                ps = [F[o]["center"] for o in trees[t]]
                return min((np.linalg.norm(ee_world - p, axis=1).min() for p in ps), default=9.0)
            table = min(tab_c, key=dmin)
            D.add("table", table, f"blend 里没有物体动画；取支撑树离末端轨迹最近的（{dmin(table)*1000:.0f} mm）",
                  fallback=True)
        else:
            table = max(tab_c, key=lambda t: len(trees[t]))
            D.add("table", table, "无物体动画、无末端轨迹；取支撑物体最多的水平面", fallback=True, warn=True)
    if table not in F:
        F[table] = _frame(ex, table)
        wv[table] = _world_verts(F[table])
        top[table] = wv[table][:, 2].max()
    if cfg.get("objects"):
        objs = list(cfg["objects"])
        D.add("objects", objs, "b2s_config 指定")
    else:
        objs = sorted(tree(table), key=lambda n: (F[n]["center"][1], F[n]["center"][0]))
        D.add("objects", objs, f"以桌面为根的支撑树；候选 {len(obj_c)} 个中 {len(objs)} 个在树上")
        # 与 blend 的集合命名交叉核对（只提示，不改判）
        for cn, c in ex.E["collections"].items():
            low = cn.lower()
            if any(w in low for w in ("task", "object", "物体", "积木")):
                missing = [o for o in c["objects"] if o in ex.E["meshes"] and o not in objs]
                extra = [o for o in objs if o not in c["objects"]]
                if missing or extra:
                    D.add("objects.vs_collection", {"collection": cn, "missing": missing, "extra": extra},
                          "自动识别与 blend 集合不一致，请人工确认", warn=True)
                else:
                    D.add("objects.vs_collection", cn, "自动识别结果与 blend 集合完全一致")
    # 桌面几何（作者局部系）
    ft = F[table]
    k, _, _, (ka, kb) = _canon(ft)
    tdims = ft["dims"]
    xw = ft["R"][:, ka]
    table_out = {"blend_name": table, "top_z": float(top[table]), "center_xy": ft["center"][:2].tolist(),
                 "size_xy": [float(tdims[ka]), float(tdims[kb])], "yaw_rad": math.atan2(xw[1], xw[0]),
                 "thickness": float(tdims[k])}
    # 地面：桌面以下的大块水平网格中最低的顶面；没有就用 0.75 m 桌高
    floors = [n for n in meshes if n != table and not ex.is_dynamic(n)]
    fz = []
    for n in floors:
        f = F[n]
        kk, _, tilt, rr = _canon(f)
        if tilt < 3 and f["dims"][rr[0]] * f["dims"][rr[1]] > 1.0:
            z = (f["center"] + f["R"][:, kk] * f["dims"][kk] / 2 * np.sign(f["R"][2, kk]))[2]
            if z < table_out["top_z"] - 0.3:
                fz.append((z, n))
    if fz:
        z, n = max(fz)
        table_out["floor_z"] = float(z)
        D.add("floor", n, f"桌面以下面积 >1 m² 的水平面，顶面 z={z:.3f}")
    else:
        table_out["floor_z"] = table_out["top_z"] - 0.75
        D.add("floor", "估计", "没找到地面网格，取桌面以下 0.75 m", fallback=True)

    used = set()
    objects = {}
    for n in objs:
        f = F[n] if n in F else _frame(ex, n)
        c = classify(ex, n, f, cfg.get("shapes", {}).get(n))
        rgba, cwhy = _color(ex, n)
        s = slug(n, used)
        objects[s] = {"blend_name": n, **c, "center": f["center"].tolist(), "rgba": rgba,
                      "color_source": cwhy, "dynamic": ex.is_dynamic(n),
                      "on": [x for x in sup.get(n, [])], "bottom_z": float(bot.get(n, 0)),
                      "top_z": float(top.get(n, 0))}
        D.add(f"shape[{s}]", c["shape"], c["why"], warn=c["shape"] == "unsupported")
        if c["tilt_deg"] > 5:
            D.add(f"tilt[{s}]", round(c["tilt_deg"], 1), "物体初始倾斜；sim.py 只表达绕 z 的朝向", warn=True)
    # blend 名 → slug，支撑关系也换成 slug
    b2s = {o["blend_name"]: s for s, o in objects.items()}
    for o in objects.values():
        o["on"] = ["table" if x == table else b2s.get(x, x) for x in o["on"]]

    # 静态障碍
    tpoly = wv[table][:, :2]
    lo, hi = tpoly.min(0) - STATIC_PAD_XY, tpoly.max(0) + STATIC_PAD_XY
    static = []
    for n in meshes:
        if n in objs or n == table or ex.is_dynamic(n):
            continue
        f = F[n]
        w = _world_verts(f)
        if f["dims"].max() > 3.0 or f["dims"].max() < 0.01:
            continue
        if w[:, 2].max() < table_out["top_z"] + 0.002 or w[:, 2].min() > table_out["top_z"] + STATIC_Z:
            continue
        if (w[:, :2].max(0) < lo).any() or (w[:, :2].min(0) > hi).any():
            continue
        rgba, _ = _color(ex, n)
        static.append({"blend_name": n, "center": f["center"].tolist(), "half": (f["dims"] / 2).tolist(),
                       "R": f["R"].tolist(), "rgba": rgba})
    D.add("static", len(static), f"桌面附近静止网格 → 碰撞 box：{[s['blend_name'] for s in static][:12]}"
          + (" …" if len(static) > 12 else ""))

    cams = {}
    tip = set()
    if robot and robot.get("flange"):
        tip = {robot["flange"], robot["links"].get(7)}
    for cn, c in ex.E["cameras"].items():
        anc = set(ex.ancestors(cn))
        on_robot = bool(anc & robot_set) or bool(anc & tip)
        moving = ex.is_dynamic(cn)
        cams[cn] = {**c, "mount": "ee" if on_robot else "world", "moving": moving,
                    "stream": c["props"].get("video_stream")}
        if moving and not on_robot:
            D.add(f"camera[{cn}]", "动态世界相机", "不挂在机器人上却在动（ego 头戴相机）；仿真里固定为首帧位姿",
                  warn=True)
    return {"T_world_base": T_base, "table": table_out, "objects": objects, "static": static,
            "cameras": cams, "robot_set": sorted(robot_set)}
