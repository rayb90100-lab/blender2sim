"""由 task_spec 构建 MuJoCo 模型。

相对 test4（droid_stage2_physics.build）的改动，均来自 §61.2 随机动作压测：
  1. 加地面（test4 无地面：被挤飞的物体无限下落）
  2. 桌子按重建外形（table_display）建碰撞体 + 四条腿，而非 0.9×1.1 m 大板（同时消除 §57 Blender/MuJoCo 桌形不一致）
  3. Panda 关节 1–4 加力矩上限 87 N·m（Franka 官方规格；menagerie panda_nohand 只限了 5–7 的 12 N·m）
物体的初始位姿可由 overrides 覆盖（初始状态采样用）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import droid_common as C  # noqa: E402

TIMESTEP = 0.002
PANDA_TORQUE_LIMIT = [87, 87, 87, 87, 12, 12, 12]   # N·m，Franka Panda 官方关节力矩上限
GRIP_PREFIX = C.GRIP_PREFIX


def _quat_z(yaw):
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def build(spec, overrides=None, render_size=None):
    """返回 (model, data)。overrides: {obj: {"center": [...], "yaw_rad": ...}}，以及可选 "table_dz"。"""
    import mujoco
    overrides = overrides or {}
    assert abs(spec["robot"]["mount_yaw_deg"] - C.MOUNT_YAW_DEG) < 1e-9, "spec 与 droid_common 安装角不一致"
    s = C.build_spec()
    # 注意：**不要**在这里应用 spec["robot"]["T_world_base"]。
    # spec 里的 table / objects / 演示轨迹全部是**基座系**坐标，机器人就放在原点，
    # 这样才是自洽的（test1/test5/test6 都依赖这个约定）。T_world_base 只用于回报/可视化。
    # 曾经误以为「基座没被应用」，加上变换后机器人被挪到 (0.649,0.842,0.13)、
    # 而桌椅杯子仍在基座系原点附近 → 两边错位 1.07 m，机械臂全程碰不到物体。
    phys = spec.get("physics", {})
    # 实验用环境变量覆盖（参数扫描，不改 spec）：R2P_TIMESTEP, R2P_SOLREF="tc,dr"
    dt = float(os.environ.get("R2P_TIMESTEP", phys.get("timestep", TIMESTEP)))
    solref_override = os.environ.get("R2P_SOLREF")
    solref_override = [float(x) for x in solref_override.split(",")] if solref_override else phys.get("object_solref")
    s.option.timestep = dt
    solimp_override = os.environ.get("R2P_SOLIMP")
    solimp_override = [float(x) for x in solimp_override.split(",")] if solimp_override else phys.get("object_solimp")
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # noslip_iterations 由 spec.physics 决定，**不从代码默认值隐式继承**。
    # test5 的全部结论（50 个初始状态可解性、协议压测、9 组接触扫描、§62.4/§62.6 的数字）
    # 都是在 noslip=3 下得出的，spec 里必须显式写 3 才能重现。
    # 背景：noslip=3 在「圆柱立在小方块顶上」这类退化接触上会产生**不衰减的极限环**
    # （ep71 绿柱 |w| 在 0.173/0.258/0.540 间周期循环，1.5 s 周期、6 s 不衰减；
    # 真实摇晃必然衰减，这是数值解）。test6 因此显式写 0（|w| 降到 0.024）。
    # 环境变量 R2P_NOSLIP 仅用于实验覆盖，不落盘。
    if "noslip_iterations" not in phys:
        raise SystemExit(
            f"spec.physics 缺少 noslip_iterations。这是会让物理结论失效的参数，必须显式声明："
            f"test5 时代为 3，test6 起为 0。请在 {spec.get('task_id', 'spec')} 里补上。")
    s.option.noslip_iterations = int(os.environ.get("R2P_NOSLIP", phys["noslip_iterations"]))
    for i, lim in enumerate(PANDA_TORQUE_LIMIT, 1):
        a = s.actuator(f"actuator{i}")
        a.forcelimited = 1
        a.forcerange = [-lim, lim]
    # 4. 重力补偿（真机 Franka 控制器自带）：机器人各体 gravcomp=1，经关节执行器输出（计入力矩上限）。
    #    无补偿时位置伺服下垂：示教回放关节跟踪误差中位 1.5°、最大 4.6°（§62），工具点落差约 2 cm。
    for b_ in s.bodies:
        if b_.name and (b_.name.startswith("link") or b_.name.startswith(GRIP_PREFIX) or b_.name == "attachment"):
            b_.gravcomp = 1.0
    for j in C.PANDA_JOINTS:
        s.joint(j).actgravcomp = 1
    wb = s.worldbody

    tb = spec["table"]
    dz = float(overrides.get("table_dz", 0.0))
    top = tb["top_z"] + dz
    floor_z = tb["top_z"] - tb["height_m"]           # 地面不随桌面增强升降
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0, 0, floor_z], size=[3, 3, 0.1],
                rgba=[0.45, 0.45, 0.45, 1])
    sx, sy = tb["size_xy"]
    th = tb["thickness"]
    t = wb.add_body(name="table", pos=[*tb["center_xy"], top - th / 2], quat=_quat_z(tb["yaw_rad"]))
    wood = [0.55, 0.33, 0.20, 1]
    t.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, th / 2], rgba=wood,
               friction=[1.0, 0.005, 0.0001])
    ap = tb["apron_m"]
    # legs=False：blend 来源的场景自带桌下结构（b2s），不凭空建腿。缺省 True = test5/6 原行为
    for k, (ax, ay) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)) if tb.get("legs", True) else ()):
        leg_h = top - th - floor_z
        t.add_geom(name=f"table_leg{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, leg_h / 2],
                   pos=[ax * (sx / 2 - 0.03), ay * (sy / 2 - 0.03), -th / 2 - leg_h / 2], rgba=[0.2, 0.2, 0.2, 1])
    # 围板：仅显示（contype=0），真实围板在桌面下，机器人正常不会碰到
    for k, (px, py, hx, hy) in enumerate(((0, sy / 2 - 0.01, sx / 2, 0.01), (0, -sy / 2 + 0.01, sx / 2, 0.01),
                                          (sx / 2 - 0.01, 0, 0.01, sy / 2), (-sx / 2 + 0.01, 0, 0.01, sy / 2))
                                         if ap > 0 else ()):
        t.add_geom(name=f"table_apron{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, ap / 2],
                   pos=[px, py, -th / 2 - ap / 2], rgba=wood, contype=0, conaffinity=0)
    wb.add_light(pos=[0.5, 0, 1.5], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7])
    # 静态障碍（b2s 从 blend 桌面附近的静止网格导出：相机支架、挡板、夹具…）。缺省无 = 原行为
    for k, st in enumerate(spec.get("static", [])):
        wb.add_geom(name=f"static{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=st["half"], pos=st["center"],
                    quat=st["quat_wxyz"], rgba=st.get("rgba", [0.5, 0.5, 0.5, 1]), friction=[1.0, 0.005, 0.0001])

    for name, o in spec["objects"].items():
        ov = overrides.get(name, {})
        c = list(ov.get("center", o["center"]))
        c[2] += dz
        yaw = ov.get("yaw_rad", o["yaw_rad"])
        b = wb.add_body(name=name, pos=c, quat=_quat_z(yaw))
        b.add_freejoint(name=f"{name}_free")
        kw = dict(type=mujoco.mjtGeom.mjGEOM_BOX, density=o["density"], rgba=o["rgba"], friction=o["friction"],
                  condim=4, solref=solref_override or o["solref"])
        if solimp_override:
            kw["solimp"] = solimp_override
        if o["shape"] == "cylinder":
            # 视觉与碰撞分离，且**碰撞体用 cylinder + capsule 两者叠加**（§63.5、§63.10）：
            #   视觉 cylinder      外观准确，不参与接触
            #   碰撞 cylinder      提供两端的**平面**接触 → 直立时稳（只有 capsule 时端部是半球，
            #                      点接触，实测源场景的圆柱会从塔顶滑落到桌面）
            #   碰撞 capsule       提供侧面的**圆角**接触 → 侧躺时稳（只有 cylinder 时圆柱-平面
            #                      接触退化成一条线，实测穿透 2.36 mm、|w| 不衰减地增长）
            # 实测对照（隔离场景，静置 4 s）：
            #   直立：cyl  |w|=0.058 穿透 −0.55mm  | cyl+cap |w|=0.014 穿透 −0.01mm
            #   侧躺：cyl  |w|=0.954 穿透 −2.36mm  | cyl+cap |w|=0.002 穿透 −0.00mm
            r, h = o["size"]
            b.add_geom(name=f"{name}_vis", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, h / 2, 0],
                       rgba=o["rgba"], contype=0, conaffinity=0, group=0)
            inert = {**kw, "rgba": [0, 0, 0, 0]}
            b.add_geom(name=f"{name}_geom", size=[r, h / 2, 0],
                       **{**inert, "type": mujoco.mjtGeom.mjGEOM_CYLINDER})
            # capsule 的圆柱段长度 = h − 2r。**矮胖圆柱（2r > h）时它是负的**，不能截断成 0
            # 硬塞：cap = [r, 0] 就是一个半径 r 的球，比圆柱底面低 r − h/2，物体等于坐在一个球上
            # → 永远纯滚动、不衰减。实测 UMI ep68 的杯子 r=26.36mm、h=48.54mm：球比杯底低 2.09mm，
            # 回放放到桌上后 3 s 仍以 |w|=0.57 rad/s、|v|=0.015 m/s 打转（=ωr 纯滚动，倾角恒定 4.72°），
            # all_at_rest 永远判否。矮胖圆柱直接省略 capsule：侧躺时接触是宽面朝下，线接触退化不严重。
            cap_len = h / 2 - r
            if cap_len > 1e-4:
                b.add_geom(name=f"{name}_cap", size=[r, cap_len, 0],
                           **{**inert, "type": mujoco.mjtGeom.mjGEOM_CAPSULE})
            continue
        sx_, sy_, h = o["size"]
        if o["shape"] == "prism":
            # 屋顶形：屋脊沿局部 x，中心在包围盒中心。
            # **视觉与碰撞分离**：mesh 只用于渲染（contype=0，不参与接触），
            # 碰撞体用两个 box 拼（不可见）。原因见 DEVELOPMENT_LOG §63.5(a)：
            #   6 顶点凸包 mesh 与**平面**接触时 MuJoCo 接触生成退化 —— 同一 spec 同一位置，
            #   mesh 版静置 2 s 仍在抖（|v|=0.041、|w|=2.69、穿透 5.85 mm 且逐位不变），
            #   box 版 |v|=|w|=0.0000、穿透 0.06 mm。换 EULER/RK4/加阻尼/缩 dt 均无效。
            # 分离后实测：|v|=|w|=0.0000、穿透 0.00 mm、接触 4 个，且外观是真实斜棱柱。
            v = [[x, y, -h / 2] for x in (-sx_ / 2, sx_ / 2) for y in (-sy_ / 2, sy_ / 2)] + \
                [[-sx_ / 2, 0, h / 2], [sx_ / 2, 0, h / 2]]
            s.add_mesh(name=f"{name}_mesh", uservert=[q for p in v for q in p])
            b.add_geom(name=f"{name}_vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_mesh",
                       rgba=o["rgba"], contype=0, conaffinity=0, group=0)
            # 碰撞体：下截完整截面 + 上截窄脊；半透明 = 不参与渲染
            inert = [0, 0, 0, 0]
            kw = {**kw, "rgba": inert}
            lo_h = h * 0.5
            b.add_geom(name=f"{name}_lo", size=[sx_ / 2, sy_ / 2, lo_h / 2],
                       pos=[0, 0, -h / 2 + lo_h / 2], **kw)
            b.add_geom(name=f"{name}_hi", size=[sx_ / 2, sy_ / 4, (h - lo_h) / 2],
                       pos=[0, 0, -h / 2 + lo_h + (h - lo_h) / 2], **kw)
            continue
        if o["shape"] == "arch":
            ts, lh, lw = 0.35 * h, 0.65 * h, 0.30 * sx_
            b.add_geom(name=f"{name}_top", size=[sx_ / 2, sy_ / 2, ts / 2], pos=[0, 0, h / 2 - ts / 2], **kw)
            for sgn, tag in ((-1, "L"), (1, "R")):
                b.add_geom(name=f"{name}_leg{tag}", size=[lw / 2, sy_ / 2, lh / 2],
                           pos=[sgn * (sx_ / 2 - lw / 2), 0, -h / 2 + lh / 2], **kw)
        else:
            b.add_geom(name=f"{name}_geom", size=[sx_ / 2, sy_ / 2, h / 2], **kw)

    for cname, cam in spec["cameras"].items():
        parent = wb if cam["mount"] == "world" else s.body(cam["mount"])
        cm = parent.add_camera(name=cname, pos=cam["pos"], quat=cam["quat_wxyz"])
        cm.fovy = cam["fovy_deg"]
        K = cam.get("intrinsics")
        if K:
            # 带主点偏移的针孔内参（OpenCV 约定，像素中心在 .5）。MuJoCo 的 principal_pixel 是相对中心的偏移，
            # 且 **x 与 OpenCV 反号**：实测 principal_pixel = [(W/2-0.5)-cx, cy-(H/2-0.5)]，
            # 渲染小球与 OpenCV 投影偏差 <0.4 px（两组随机内参）。
            Wk, Hk = K["width"], K["height"]
            cm.resolution = [Wk, Hk]
            cm.focal_pixel = [K["fx"], K["fy"]]
            cm.principal_pixel = [(Wk / 2 - 0.5) - K["cx"], K["cy"] - (Hk / 2 - 0.5)]
            cm.sensor_size = [Wk * 1e-3, Hk * 1e-3]
    w, h = render_size or (spec["render"]["width"], spec["render"]["height"])
    s.visual.global_.offwidth = max(w, 640)
    s.visual.global_.offheight = max(h, 480)
    m = s.compile()
    return m, mujoco.MjData(m)


def reset(m, d, spec, settle_s=0.5):
    """机器人到 init_qpos、夹爪按 init_gripper，物体静置 settle_s 秒。"""
    settle_steps = int(round(settle_s / m.opt.timestep))
    import mujoco
    mujoco.mj_resetData(m, d)
    qa = C.arm_qadr(m)
    q0 = np.array(spec["robot"]["init_qpos"])
    d.qpos[qa] = q0
    d.ctrl[arm_actuators(m)] = q0
    d.ctrl[grip_actuator(m)] = 255 * spec["robot"]["init_gripper"]
    mujoco.mj_forward(m, d)
    for _ in range(settle_steps):
        mujoco.mj_step(m, d)


def arm_actuators(m):
    return [m.actuator(f"actuator{i}").id for i in range(1, 8)]


def grip_actuator(m):
    return m.actuator(f"{GRIP_PREFIX}fingers_actuator").id
