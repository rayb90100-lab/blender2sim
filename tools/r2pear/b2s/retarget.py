"""b2s 第 2b 步：末端轨迹 → 目标机器人（Panda + 2F-85）关节轨迹。UMI / Ego / 任何没有机器人关节的演示。

输入：规范末端系（= 2F-85 pinch site：z 接近方向、y 闭合轴）的世界轨迹 ee_world (T,4,4)。
1. 基座摆位（b2s_config.json 的 "robot_base" 可直接给 4×4，跳过搜索）
   候选：以轨迹水平质心为圆心、半径 0.35–0.70 m 的圆周上 16 个方位 × 3 个高度（桌面 −5/0/+5 cm），
   基座朝向质心；排除基座落在桌面投影内部（>3 cm）或与静态障碍重叠的。
   评分：抽样帧 IK 可达比例（位置 <5 mm、姿态 <5°）优先，其次关节限位余量。
2. 逐帧 IK（阻尼最小二乘，上一帧热启动）
   二指夹爪绕接近轴 180° 对称：首帧两种都试，之后每帧在两者中取离上一帧关节最近的。
3. 报告：每帧位置/姿态残差、关节速度峰值；残差超标的帧写进决策（不静默）。
"""
import math

import numpy as np

from common import inv

POS_TOL, ROT_TOL = 0.005, math.radians(5)
RADII = (0.35, 0.45, 0.55, 0.65)
N_AZ = 16
DZ = (0.0, 0.05)      # 基座只能和桌面同高或更高；负值会让机械臂从桌子底下穿过（UMI 桌面高 176 mm 时实测踩中）
FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])     # 绕 z 转 180°
PANDA_TAU = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)   # N·m，与 sim.PANDA_TORQUE_LIMIT 一致
LIM_MARGIN = math.radians(3.0)   # 解离关节限位至少留 3°
LIM_PUSH = 0.5                   # 越界回推系数


class IK:
    def __init__(self, robot="panda"):
        """robot: "panda"（默认，DROID/test1 用）或 "xarm7"。

        xArm7 存在的理由：UMI 是人手持爪，手腕可贴桌面作业；Panda+2F-85 的腕→捏合点 0.263 m，
        末端离桌面 3 cm 时腕部必须高 29 cm，几何上做不到（实测 link7 穿桌 18.5 mm）。
        xArm7 自带夹爪只有 0.123 m，可下探到桌面高度。
        """
        import mujoco
        self.robot = robot
        if robot == "xarm7":
            import xarm7_common as X
            self.m, self.d = X.build_model()
            self.qa = X.arm_qadr(self.m)
            self.dofs = [int(dd) for dd in self.qa]
            self.sid = self.m.site(X.TCP_SITE).id
            self.lim = np.array([self.m.joint(j).range for j in X.ARM_JOINTS])
            self.home = np.array(X.HOME, float)
            self.arm_acts = X.arm_actuators(self.m)
            self.bind = X
        else:
            import droid_common as C
            self.m, self.d, _ = C.build_model()
            self.qa = C.arm_qadr(self.m)
            self.dofs = [int(self.m.joint(f"joint{i}").dofadr[0]) for i in range(1, 8)]
            self.sid = self.m.site(f"{C.GRIP_PREFIX}pinch").id
            self.lim = np.array([self.m.joint(f"joint{i}").range for i in range(1, 8)])
            self.home = np.array([0, -0.6, 0, -2.3, 0, 1.8, 0.785])
            self.arm_acts = None
            self.bind = C
        self.mj = mujoco

    # ---- 碰撞检查：直接用 MuJoCo 的精确碰撞检测，不自己造几何近似 ----
    # 早期实现手写了球/胶囊/圆柱/方盒/mesh 的最低点计算，连修 4 个 bug 仍不准确
    # （坐标系、size 语义、障碍物变换、mesh 被跳过），而 MuJoCo 本来就有精确的 GJK/EPA 碰撞。
    # 这里给 IK 模型临时拼一个「桌面 + 障碍物」的世界，用 mj_forward 后的 ncon/dist 判定。
    def collision_model(self, table, floor_z, obstacles=()):
        """返回 (model, data, robot_geoms) —— 带桌面/地面/障碍物碰撞体的可判碰模型（缓存）。"""
        key = (round(table["top_z"], 5), round(table["center_xy"][0], 5), round(table["center_xy"][1], 5),
               round(table["yaw_rad"], 5), round(table["size_xy"][0], 5), round(table["size_xy"][1], 5),
               round(floor_z, 5), tuple((round(o[0], 4), round(o[1], 4), round(o[2], 4),
                                         round(o[3], 4), round(o[4], 4)) for o in obstacles))
        if getattr(self, "_cm_key", None) == key:
            return self._cm
        import mujoco
        # 在「当前机型」的 MjSpec 上加桌面/地面/障碍物。两种机型都走 MjSpec.from_file，
        # 因为 MjModel 不可变、不能直接挂新几何。
        if self.robot == "xarm7":
            sp = mujoco.MjSpec.from_file(self.bind.XARM7_XML)
        else:
            sp = self.bind.build_spec()
        wb = sp.worldbody
        th = 0.02
        tb = wb.add_body(name="_table", pos=[table["center_xy"][0], table["center_xy"][1], table["top_z"] - th / 2])
        tb.quat = [math.cos(table["yaw_rad"] / 2), 0, 0, math.sin(table["yaw_rad"] / 2)]
        tb.add_geom(name="_table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[table["size_xy"][0] / 2, table["size_xy"][1] / 2, th / 2])
        fb = wb.add_body(name="_floor", pos=[0, 0, floor_z])
        fb.add_geom(name="_floor_g", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1])
        for i, (ox, oy, orad, ozlo, ozhi) in enumerate(obstacles):
            ob = wb.add_body(name=f"_obs{i}", pos=[ox, oy, (ozlo + ozhi) / 2])
            ob.add_geom(name=f"_obs{i}_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=[orad, max((ozhi - ozlo) / 2, 1e-3), 0])
        m = sp.compile()
        d = mujoco.MjData(m)
        prefix = ("link", "xarm_gripper", "left_", "right_") if self.robot == "xarm7" \
            else ("link", "gripper", "attachment")
        rg = [g for g in range(m.ngeom)
              if (m.body(m.geom_bodyid[g]).name or "").startswith(prefix)]
        self._cm_key, self._cm = key, (m, d, rg)
        return self._cm

    def penetration(self, q, table, floor_z, margin=0.0, obstacles=()):
        """机器人几何体与桌面/地面/障碍物的最深侵入（米，正 = 侵入）。用 MuJoCo 精确碰撞。"""
        import mujoco
        m, d, rg = self.collision_model(table, floor_z, obstacles)
        qa = [int(m.joint(f"joint{i}").dofadr[0]) for i in range(1, 8)]
        d.qpos[qa] = q
        mujoco.mj_forward(m, d)
        rgset = set(rg)
        worst = -1.0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in rgset or c.geom2 in rgset:
                worst = max(worst, float(-c.dist))
        return float(worst)

    def torque_margin(self, q):
        """这批姿态下关节的峰值扭矩占上限的比例（0–1，1 = 饱和）。

        用 MuJoCo 的重力补偿扭矩 qfrc_bias（qvel=0 时即静态重力扭矩）估计。
        实测：某些摆位下 j3 力矩饱和到 87 N·m 上限，静态跟位就差 11.7°。
        """
        m, d = self.m, self.d
        lim = np.array(PANDA_TAU)
        worst = 0.0
        d.qvel[:] = 0.0
        for k in range(len(q)):
            d.qpos[self.qa] = q[k]
            self.mj.mj_kinematics(m, d)
            self.mj.mj_comPos(m, d)
            self.mj.mj_comVel(m, d)
            self.mj.mj_rne(m, d, 0, d.qfrc_bias)
            worst = max(worst, float(np.max(np.abs(d.qfrc_bias[self.qa]) / lim)))
        return worst

    def fk(self, q):
        self.d.qpos[self.qa] = q
        self.mj.mj_kinematics(self.m, self.d)
        self.mj.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def solve(self, T, q0, iters=60, damp=1e-3):
        q = q0.copy()
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            p, R = self.fk(q)
            ep = T[:3, 3] - p
            Re = T[:3, :3] @ R.T
            from scipy.spatial.transform import Rotation as Rot
            er = Rot.from_matrix(Re).as_rotvec()
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            self.mj.mj_jacSite(self.m, self.d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.dofs], jr[:, self.dofs]])
            e = np.concatenate([ep, er])
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
            q = q + dq
            # 限位处理：直接 np.clip 会把解**死死压在限位边界上**（实测 j1 目标=-101.0 = 下限、
            # j4 目标=-166.0 = 下限）。MuJoCo 的位置伺服在限位处跟踪不了 —— 关节会被限位挡板
            # 顶住，回放跟踪误差 66°、姿态完全跑偏。这里改成「软限位」：接近边界时用一个
            # 反向弹簧把解推回来，保证解始终留出 LIM_MARGIN 的余量。
            over = q - (self.lim[:, 1] - LIM_MARGIN)
            under = (self.lim[:, 0] + LIM_MARGIN) - q
            q = q - np.maximum(over, 0) * LIM_PUSH + np.maximum(under, 0) * LIM_PUSH
            q = np.clip(q, self.lim[:, 0], self.lim[:, 1])
        p, R = self.fk(q)
        from scipy.spatial.transform import Rotation as Rot
        return q, float(np.linalg.norm(T[:3, 3] - p)), float(np.linalg.norm(
            Rot.from_matrix(T[:3, :3] @ R.T).as_rotvec()))


def _obstacles_in_base(scene, Tb, moved=()):
    """把演示过程中「静止不动」的物体换算成基座系的竖直圆柱，供碰撞粗判使用。

    只取非移动物体（被搬运的那个本来就要被夹爪接触，不能当障碍）；半径按水平尺寸的一半。
    """
    Ti = inv(Tb)
    out = []
    moved = set(moved or ())
    for s, o in (scene.get("objects") or {}).items():
        if s in moved:
            continue
        sz = o.get("dims") or o.get("size")
        if not sz:
            continue
        c = np.array([o["center"][0], o["center"][1], 0.0, 1.0])
        cb = Ti @ c
        rad = float(min(sz[0], sz[1])) / 2 if len(sz) >= 2 else 0.03
        zb_lo = float(Ti[2, 2] * o.get("bottom_z", 0.0) + Ti[2, 3])   # 高度也要过一遍同一变换
        zb_hi = float(Ti[2, 2] * o.get("top_z", 0.0) + Ti[2, 3])
        out.append((float(cb[0]), float(cb[1]), rad, zb_lo, zb_hi))
    return out


def _table_in_base(table, floor_z, Tb):
    """把世界系的桌面参数换算到**机器人基座系**。

    IK 用的 MuJoCo 模型里机器人基座就在原点，而 scene["table"] 是世界系的。
    实测踩坑：世界系 top_z=0.13、基座系 0.75（差 0.62 = 基座离地高度），
    拿世界系数值去比，碰撞检测永远认为「离桌面还有 0.62 m 余量」，
    于是选出让 link5 压进桌面 40 mm 的基座，回放时机械臂被桌子顶住、关节误差 114°。
    """
    Ti = inv(Tb)
    p = Ti @ np.array([*table["center_xy"], table["top_z"], 1.0])
    bottom = floor_z - Tb[2, 3]          # 地面也是世界系的，同样换算
    return ({"top_z": float(p[2]), "center_xy": p[:2].tolist(),
             "size_xy": table["size_xy"], "yaw_rad": table["yaw_rad"] - math.atan2(Tb[1, 0], Tb[0, 0])},
            float(bottom))


def _base_T(xy, z, yaw):
    T = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [xy[0], xy[1], z]
    return T


def _track(ik, Tb_list, q_init):
    """逐帧 IK，含 180° 对称选择。返回 q(T,7)、位置残差、姿态残差。"""
    qs, ep, er = [], [], []
    q = q_init
    for k, T in enumerate(Tb_list):
        best = None
        for TT in (T, T @ FLIP):
            qq, a, b = ik.solve(TT, q)
            cost = (a > POS_TOL) * 10 + (b > ROT_TOL) * 10 + np.abs(qq - q).sum() * (k > 0)
            if best is None or cost < best[0]:
                best = (cost, qq, a, b)
        q = best[1]
        qs.append(q)
        ep.append(best[2])
        er.append(best[3])
    return np.array(qs), np.array(ep), np.array(er)


def retarget(ee_world, scene, cfg, D, moved=()):
    ik = IK()
    top = scene["table"]["top_z"]
    floor = float(scene["table"].get("floor_z", top - 0.75))
    if cfg.get("robot_base"):
        cands = [(np.array(cfg["robot_base"]), 2)]      # tier=2：用户显式指定，优先级最高
        D.add("robot_base", "config", "b2s_config 指定基座位姿")
    else:
        c = ee_world[:, :2, 3].mean(0)
        tb = scene["table"]
        cy, sy = math.cos(tb["yaw_rad"]), math.sin(tb["yaw_rad"])
        Rt = np.array([[cy, -sy], [sy, cy]])
        hx, hy = tb["size_xy"][0] / 2, tb["size_xy"][1] / 2
        # 基座高度必须物理上说得通：要么和桌面同层（桌面 + 抬高量），要么落在地面上。
        #   旧实现 DZ 含 -0.05，允许基座在桌面**以下** —— DROID 桌面低、恰好没暴露；
        #   UMI 桌面高 176 mm，实测选中了桌面下方 50 mm 的悬空基座，机械臂整条从桌子底下穿过。
        zs = [(top + dz, 1) for dz in DZ] + [(floor, 0)]      # 第二项 = 优先级（桌面层优先于地面）
        cands = []
        for r in RADII:
            for k in range(N_AZ):
                a = 2 * math.pi * k / N_AZ
                xy = c + r * np.array([math.cos(a), math.sin(a)])
                lp = Rt.T @ (xy - np.array(tb["center_xy"]))
                inside = min(hx - abs(lp[0]), hy - abs(lp[1]))     # >0：在桌面投影内
                if inside > 0.03:
                    continue
                hit = False
                for st in scene["static"]:
                    if np.all(np.abs(np.array(st["center"][:2]) - xy) < np.array(st["half"][:2]) + 0.1):
                        hit = True
                        break
                if hit:
                    continue
                for z, tier in zs:
                    cands.append((_base_T(xy, z, math.atan2(*(c - xy)[::-1])), tier))
    # 抽样 24 帧太稀疏：实测漏掉了整段失败（抽样全过、全帧却 338 帧不可达）。
    # 这里对**全部帧**做一次 IK + 碰撞评估；候选数已由 16 方位 × 半径 × 高度压到百余个，
    # 用粗容差的热启动做一次即可（完整精度的 IK 在后面还会重跑一遍）。
    sub = np.arange(len(ee_world))
    _SUB_MAX = 400                      # 上限保护：帧数很多时按均匀子采样
    if len(sub) > _SUB_MAX:
        sub = np.linspace(0, len(ee_world) - 1, _SUB_MAX).astype(int)
    # 用「无碰撞帧数」而不是「最深侵入」当首要排序：只有少数帧撞桌时，回放照样会被那几帧顶住。
    scored = []
    for Tb, tier in cands:
        Ti = inv(Tb)
        _tb_base, _floor_base = _table_in_base(scene["table"], floor, Tb)
        _obs = _obstacles_in_base(scene, Tb, moved)
        q, ep, er = _track(ik, [Ti @ ee_world[t] for t in sub], ik.home)
        reach = float(np.mean((ep < POS_TOL) & (er < ROT_TOL)))
        margin = float(np.min(np.minimum(q - ik.lim[:, 0], ik.lim[:, 1] - q)))
        # 碰撞：每个抽样帧都测，统计「撞了的帧数」和「最深侵入」两个量
        pen = 0.0
        n_hit = 0
        for k in range(len(sub)):
            p = ik.penetration(q[k], _tb_base, _floor_base, obstacles=_obs)
            pen = max(pen, p)
            if p > 0.005:
                n_hit += 1
        # 力矩余量：Panda 关节 1–4 上限 87 N·m。IK 解出来不等于伺服跟得上。
        tau = ik.torque_margin(q)
        # 合格 = 零碰撞帧 **且** 可达 **且** 力矩不饱和。三条都必须满足：
        #   只看可达 → 钻进桌子；只看无碰撞 → 够不着轨迹（338 帧 IK 失败）。
        ok = (n_hit == 0) and (reach >= 0.999) and (tau < 0.9)
        scored.append((1 if ok else 0, -n_hit, -pen, tau, reach, tier, margin, Tb))
    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6]), reverse=True)
    ok, _nh, _negpen, tau, reach, _tier, margin, Tb = scored[0]
    pen = -_negpen
    n_hit = -_nh
    n_ok = sum(s[0] for s in scored)
    D.add("robot_base", {"xyz": Tb[:3, 3].round(3).tolist(), "yaw_deg": round(math.degrees(math.atan2(Tb[1, 0], Tb[0, 0])), 1)},
          f"{len(cands)} 个候选中「零碰撞帧且可达且力矩未饱和」的有 {n_ok} 个；选中：碰撞帧 "
          f"{n_hit}/{len(sub)}、最深侵入 {pen * 1000:.0f} mm、峰值力矩占上限 {tau:.0%}、"
          f"抽样帧可达 {reach:.0%}、关节限位余量 {math.degrees(margin):.1f}°"
          f"（基座 z 候选：桌面 {top:.3f} +{max(DZ)} m、地面 {floor:.3f} m）",
          fallback=not cfg.get("robot_base"), warn=(not ok))
    Ti = inv(Tb)
    q, ep, er = _track(ik, [Ti @ T for T in ee_world], ik.home)
    # 精解完必须**重新复核**碰撞：基座是拿抽样帧选的，全帧解出来可能仍有几帧撞桌
    # （实测踩过：选中时 0 mm，全帧解完 link5 压进桌面 39 mm，回放时机械臂被桌子顶住、跟踪误差 114°）。
    pen_full = 0.0
    bad_pen = []
    _tb_base, _floor_base = _table_in_base(scene["table"], floor, Tb)
    _obs = _obstacles_in_base(scene, Tb, moved)
    for t in range(len(q)):
        p = ik.penetration(q[t], _tb_base, _floor_base, obstacles=_obs)
        if p > 0.005:
            bad_pen.append(int(t))
        pen_full = max(pen_full, p)
    tau_full = ik.torque_margin(q)
    D.add("retarget.clearance",
          {"penetration_mm": round(pen_full * 1000, 1), "n_colliding": len(bad_pen),
           "bad_frames": bad_pen[:20], "peak_torque_frac": round(tau_full, 3)},
          "全帧精解后重新复核机器人与桌面/地面的净空；碰撞帧会被物理挡住，回放跟不上就是从这里来的",
          warn=(len(bad_pen) > 0 or tau_full >= 0.9))
    bad = np.where((ep > POS_TOL) | (er > ROT_TOL))[0]
    vmax = float(np.degrees(np.abs(np.diff(q, axis=0)).max()) * 1) if len(q) > 1 else 0.0
    D.add("retarget.ik", {"bad_frames": bad.tolist()[:40], "n_bad": int(len(bad)),
                          "pos_err_mm_max": round(float(ep.max() * 1000), 2),
                          "rot_err_deg_max": round(float(np.degrees(er.max())), 2),
                          "max_joint_step_deg": round(vmax, 2)},
          "逐帧 IK（上一帧热启动、180° 对称取近）", warn=len(bad) > 0)
    return {"T_world_base": Tb, "q": q, "pos_err": ep, "rot_err": er}
