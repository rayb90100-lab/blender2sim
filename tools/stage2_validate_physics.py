#!/usr/bin/env python3
"""Stage 2 核心验证：物理仿真能否消除 Stage 1.5 的两类伪运动。

要回答的问题（用户指定的本轮目标）：
  Q1 "纸巾盒来回跑" —— 静止物体的假位移
  Q2 "物体大小抖动" —— 深度抖动导致的尺寸变化

Stage 1.5 用逐帧单目深度拟合物体轨迹，得到：
  - 贴桌物体的深度抖动 ±3.6~4.9 mm（→ 屏幕尺寸抖动 ±1%）
  - 遮挡帧 mask 质心被裁偏，产生 6~11 cm 的假位移
  - 纸巾盒全程不动，但 mask 在 f120–f170 因手部遮挡而 cx 变化 84 px

物理引擎里物体由重力与接触推进，中间帧不是估计出来的。所以只要初始状态对，
静止物体就不可能自己动，运动物体的轨迹也不会有高频抖动。

本脚本做三组对照：
  A. 静置：3 秒纯物理，测漂移
  B. 抓持搬运：等效接触对（两指板 + 物体）做抬升-平移-放下，测轨迹平滑性
  C. 与 Stage 1.5 的轨迹做抖动指标对比

用法:
  python tools/stage2_validate_physics.py --testdir <testdir>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_common import load_geometry  # noqa: E402

import mujoco  # noqa: E402


def log(m): print(m, flush=True)
def hdr(m): print(f"\n{'='*68}\n{m}\n{'='*68}", flush=True)


# ---------------- A. 静置漂移 ----------------
def test_rest(testdir, seconds=3.0):
    m = mujoco.MjModel.from_xml_path(
        os.path.join(testdir, "stage2_physics", "mujoco", "scene.xml"))
    d = mujoco.MjData(m)
    # 两个物体都在初始位；机械臂放到已验证无穿透的姿态并保持
    home = json.load(open(os.path.join(testdir, "stage2_physics", "mujoco",
                                       "robot_placement.json")))["home_q"]
    for side in ("left", "right"):
        for k in range(1, 8):
            d.qpos[int(m.joint(f"{side}_fr3v2_joint{k}").qposadr[0])] = home[k - 1]
    for side in ("left", "right"):
        ci = [i for i in range(m.nu) if m.actuator(i).name == f"{side}_finger_act"]
        if ci:
            d.ctrl[ci[0]] = 255.0
    for side in ("left", "right"):
        for k in range(1, 8):
            ci = [i for i in range(m.nu) if m.actuator(i).name == f"{side}_fr3v2_joint{k}"]
            if ci:
                d.ctrl[ci[0]] = home[k - 1]
    mujoco.mj_forward(m, d)

    adr = {n: int(m.joint(f"{n}_free").qposadr[0]) for n in ("earbud_case", "tissue_box")}
    p0 = {k: d.qpos[a:a + 3].copy() for k, a in adr.items()}
    traj = {k: [] for k in adr}
    n = int(seconds / m.opt.timestep)
    for _ in range(n):
        mujoco.mj_step(m, d)
        for k, a in adr.items():
            traj[k].append(d.qpos[a:a + 3].copy())

    out = {}
    for k, a in adr.items():
        T = np.array(traj[k])
        out[k] = {
            "drift_mm": float(np.linalg.norm(d.qpos[a:a + 3] - p0[k]) * 1000),
            "z_start_mm": float(p0[k][2] * 1000),
            "z_end_mm": float(d.qpos[a + 2] * 1000),
            "max_dev_mm": float(np.abs(T - p0[k]).max() * 1000),
            "jitter_mm": float(np.abs(np.diff(T, 2, axis=0)).mean() * 1000),
        }
    return out, m.opt.timestep * n


# ---------------- B. 等效接触对抓持搬运 ----------------
def test_transport(testdir, outdir):
    """用两个单指板夹住物体，做 抓取→抬升→平移→放下。

    这不是 FR3 的完整关节链，而是一个受控的等效试验：只要接触对、质量、摩擦、
    时间步与真实场景一致，物体的运动就由物理决定，而不是由轨迹拟合决定。
    用来隔离验证"物理能否产生平滑轨迹"这个命题。
    """
    m, d = _build_transport_scene(testdir)
    oa = int(m.joint("obj_free").qposadr[0])
    p0 = d.qpos[oa:oa + 3].copy()

    # 侧板的滑动关节是相对 body 原点的，ctrl 用关节坐标而不是世界坐标。
    # 曾直接拿世界 x 当 ctrl，板从 x=0 跳到 -0.097，把物体推到左边 54 mm。
    # 物体绕 z 旋转 yaw 后，在 x 方向上的半宽不再是 geom size[0]。
    # 取旋转后的 AABB 半宽，否则侧板会嵌进物体（实测 1.5 mm 穿透）。
    yaw_e = _LAST_YAW
    sx, sy = m.geom("obj_geom").size[0], m.geom("obj_geom").size[1]
    ca, sa = abs(np.cos(yaw_e)), abs(np.sin(yaw_e))
    box_half = sx * ca + sy * sa
    gap = 2 * box_half + 0.001
    body_x = {s_: float(m.body(f"plate_{s_}").pos[0]) for s_ in "lr"}
    # 关节目标 = 世界目标 - body 原点
    lx_tgt = (p0[0] - gap / 2) - body_x["l"]
    rx_tgt = (p0[0] + gap / 2) - body_x["r"]

    rec = []
    n_total = 0

    def step(n, tag):
        nonlocal n_total
        for _ in range(n):
            mujoco.mj_step(m, d)
            n_total += 1
            rec.append({"t": n_total * m.opt.timestep, "tag": tag,
                        "obj": d.qpos[oa:oa + 3].tolist(),
                        "plate_z": float(d.qpos[0])})

    # 阶段 1：静置
    step(300, "rest")

    # 阶段 2：两侧板从当前位置移到贴合（位置伺服，关节坐标）
    lx0 = float(d.qpos[m.joint("plate_l_x").qposadr[0]])
    rx0 = float(d.qpos[m.joint("plate_r_x").qposadr[0]])
    for k in range(300):
        a = (k + 1) / 300
        d.ctrl[0] = lx0 + a * (lx_tgt - lx0)
        d.ctrl[1] = rx0 + a * (rx_tgt - rx0)
        mujoco.mj_step(m, d)
        n_total += 1
        rec.append({"t": n_total * m.opt.timestep, "tag": "close",
                    "obj": d.qpos[oa:oa + 3].tolist(), "plate_z": float(d.qpos[2])})
    step(300, "squeeze")

    # 阶段 3：抬升（两侧板同步上升，靠摩擦带起物体）
    lift = 0.10
    for k in range(600):
        a = (k + 1) / 600
        d.ctrl[2] = lift * a
        d.ctrl[3] = lift * a
        mujoco.mj_step(m, d)
        n_total += 1
        rec.append({"t": n_total * m.opt.timestep, "tag": "lift",
                    "obj": d.qpos[oa:oa + 3].tolist(), "plate_z": float(d.qpos[0])})

    # 阶段 4：水平平移
    dx = 0.14
    for k in range(600):
        a = (k + 1) / 600
        d.ctrl[0] = lx_tgt + dx * a
        d.ctrl[1] = rx_tgt + dx * a
        mujoco.mj_step(m, d)
        n_total += 1
        rec.append({"t": n_total * m.opt.timestep, "tag": "transfer",
                    "obj": d.qpos[oa:oa + 3].tolist(), "plate_z": float(d.qpos[0])})

    # 阶段 5：下降 + 松开
    for k in range(600):
        a = (k + 1) / 600
        d.ctrl[2] = lift * (1 - a)
        d.ctrl[3] = lift * (1 - a)
        mujoco.mj_step(m, d)
        n_total += 1
        rec.append({"t": n_total * m.opt.timestep, "tag": "lower",
                    "obj": d.qpos[oa:oa + 3].tolist(), "plate_z": float(d.qpos[0])})
    for k in range(300):
        a = (k + 1) / 300
        d.ctrl[0] = lx_tgt + dx - 0.06 * a
        d.ctrl[1] = rx_tgt + dx + 0.06 * a
        mujoco.mj_step(m, d)
        n_total += 1
        rec.append({"t": n_total * m.opt.timestep, "tag": "release",
                    "obj": d.qpos[oa:oa + 3].tolist(), "plate_z": float(d.qpos[0])})
    step(400, "settle")

    p_end = d.qpos[oa:oa + 3].copy()
    T = np.array([r["obj"] for r in rec])
    lifted = T[:, 2].max() - p0[2]
    return {
        "records": rec,
        "object_start": p0.tolist(),
        "object_end": p_end.tolist(),
        "max_lift_mm": float(lifted * 1000),
        "horizontal_move_mm": float(np.linalg.norm(p_end[:2] - p0[:2]) * 1000),
        "z_settled_mm": float(p_end[2] * 1000),
        "jitter_z_mm": float(np.abs(np.diff(T[:, 2], 2)).mean() * 1000),
        "steps": n_total,
        "duration_s": n_total * m.opt.timestep,
    }, T


_LAST_YAW = 0.0


def _build_transport_scene(testdir):
    """构造等效接触对场景：两个板 + 被测物体（几何取自 Stage 1）。"""
    man = json.load(open(os.path.join(testdir, "stage1_blender", "manifest.json")))
    _, n_cam, d_cam, R_wc, c2w, d2w = load_geometry(testdir)
    v = man["instances"]["earbud_case"]
    c = c2w(v["center_cam"])
    s = np.array(v["size"], float)
    # 纸巾盒：本轮要验证"它不会自己动"，所以必须出现在对比画面里
    tv = man["instances"]["tissue_box"]
    tc = c2w(tv["center_cam"])
    ts = np.array(tv["size"], float)

    # yaw：Stage 1 建的是轴对齐盒。orientation.json 由轮廓 IoU 全扫得出。
    ori_p = os.path.join(testdir, "stage2_physics", "mujoco", "orientation.json")
    ori = json.load(open(ori_p))["objects"] if os.path.exists(ori_p) else {}
    yaw_e = float(ori["earbud_case"]["yaw_rad"]) if "earbud_case" in ori else 0.0
    # 板初始位置：物体绕 z 旋转后 x 方向的 AABB 半宽 + 5 mm 余量
    ca0, sa0 = abs(np.cos(yaw_e)), abs(np.sin(yaw_e))
    pd = s[0] / 2 * ca0 + s[1] / 2 * sa0 + 0.005
    yaw_t = float(ori["tissue_box"]["yaw_rad"]) if "tissue_box" in ori else 0.0

    cam_pos = c2w(np.zeros(3))
    fwd = d2w(np.array([0.0, 0.0, 1.0]))
    up_cam = d2w(np.array([0.0, -1.0, 0.0]))
    right_v = np.cross(fwd, up_cam); right_v /= np.linalg.norm(right_v)
    upv = np.cross(right_v, fwd); upv /= np.linalg.norm(upv)
    fovy = 2 * np.degrees(np.arctan(man["camera"]["H"] / 2 / man["camera"]["fy"]))

    xml = f'''<mujoco model="transport test">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <rgba haze="0.30 0.33 0.36 1"/>
    <global offwidth="720" offheight="1280" azimuth="120" elevation="-18"/>
  </visual>
  <asset>
    <texture type="2d" name="wood" builtin="checker" mark="edge"
             rgb1="0.62 0.47 0.32" rgb2="0.55 0.41 0.27"
             markrgb="0.48 0.36 0.24" width="256" height="256"/>
    <material name="table_mat" texture="wood" texuniform="true" texrepeat="6 6" reflectance="0.06"/>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.62 0.72" rgb2="0.22 0.24 0.28"
             width="512" height="3072"/>
  </asset>
  <worldbody>
    <light pos="0.2 0.2 1.2" dir="0 0 -1" directional="true"
           diffuse="0.75 0.75 0.75" specular="0.2 0.2 0.2" castshadow="true"/>
    <light pos="-0.6 0.3 0.9" diffuse="0.35 0.35 0.38"/>
    <geom name="table" type="box" pos="{c[0]+0.06:.4f} {c[1]+0.10:.4f} -0.02" size="0.34 0.26 0.02"
          material="table_mat"/>
    <!-- 纸巾盒：静止参照物，参与物理 -->
    <body name="tissue" pos="{tc[0]:.6f} {tc[1]:.6f} {tc[2]:.6f}" euler="0 0 {yaw_t:.6f}">
      <freejoint name="tissue_free"/>
      <geom name="tissue_geom" type="box" size="{ts[0]/2:.6f} {ts[1]/2:.6f} {ts[2]/2:.6f}"
            mass="0.230" friction="0.8 0.02 0.001" rgba="0.85 0.83 0.90 1"/>
    </body>
    <!-- 被测物体：Stage 1 拟合的几何与质量 -->
    <body name="obj" pos="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" euler="0 0 {yaw_e:.6f}">
      <freejoint name="obj_free"/>
      <geom name="obj_geom" type="box" size="{s[0]/2:.6f} {s[1]/2:.6f} {s[2]/2:.6f}"
            mass="0.055" friction="1.0 0.02 0.001" rgba="0.05 0.05 0.06 1"/>
    </body>
    <!-- 等效夹指：两块薄板，x 方向位置伺服、z 方向跟随 -->
    <body name="plate_l" pos="{c[0]-pd:.4f} {c[1]:.4f} {c[2]:.4f}">
      <joint name="plate_l_x" type="slide" axis="1 0 0" range="-0.3 0.3"/>
      <geom name="plate_l_geom" type="box" size="0.003 0.02 0.015"
            mass="0.2" friction="1.0 0.02 0.001" rgba="0.85 0.3 0.3 1"/>
    </body>
    <body name="plate_r" pos="{c[0]+pd:.4f} {c[1]:.4f} {c[2]:.4f}">
      <joint name="plate_r_x" type="slide" axis="1 0 0" range="-0.3 0.3"/>
      <geom name="plate_r_geom" type="box" size="0.003 0.02 0.015"
            mass="0.2" friction="1.0 0.02 0.001" rgba="0.85 0.3 0.3 1"/>
    </body>
    <camera name="source_view" pos="{cam_pos[0]:.6f} {cam_pos[1]:.6f} {cam_pos[2]:.6f}"
            xyaxes="{right_v[0]:.6f} {right_v[1]:.6f} {right_v[2]:.6f} {upv[0]:.6f} {upv[1]:.6f} {upv[2]:.6f}"
            fovy="{fovy:.4f}"/>
  </worldbody>
  <actuator>
    <position name="act_l_x" joint="plate_l_x" kp="900" kv="40" forcerange="-40 40"/>
    <position name="act_r_x" joint="plate_r_x" kp="900" kv="40" forcerange="-40 40"/>
    <position name="act_l_z" joint="plate_l_x" kp="0" kv="0" forcerange="0 0"/>
  </actuator>
</mujoco>
'''
    # 上面的 z 方向需要单独关节，重写：给每个板加两个滑轨（x 与 z）
    xml = xml.replace('''      <joint name="plate_l_x" type="slide" axis="1 0 0" range="-0.3 0.3"/>''',
'''      <joint name="plate_l_x" type="slide" axis="1 0 0" range="-0.3 0.3"/>
      <joint name="plate_l_z" type="slide" axis="0 0 1" range="-0.2 0.3"/>''')
    xml = xml.replace('''      <joint name="plate_r_x" type="slide" axis="1 0 0" range="-0.3 0.3"/>''',
'''      <joint name="plate_r_x" type="slide" axis="1 0 0" range="-0.3 0.3"/>
      <joint name="plate_r_z" type="slide" axis="0 0 1" range="-0.2 0.3"/>''')
    xml = xml.replace('''    <position name="act_l_z" joint="plate_l_x" kp="0" kv="0" forcerange="0 0"/>''',
'''    <position name="act_l_z" joint="plate_l_z" kp="900" kv="40" forcerange="-40 40"/>
    <position name="act_r_z" joint="plate_r_z" kp="900" kv="40" forcerange="-40 40"/>''')

    p = os.path.join(testdir, "stage2_physics", "mujoco", "_transport.xml")
    open(p, "w").write(xml)
    m = mujoco.MjModel.from_xml_path(p)
    global _LAST_YAW
    _LAST_YAW = yaw_e         # 供调用方算侧板间距（MjModel 不能挂自定义属性）
    return m, mujoco.MjData(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    a = ap.parse_args()
    outdir = os.path.join(a.testdir, "stage2_physics", "validation")
    os.makedirs(outdir, exist_ok=True)

    hdr("A. 静置漂移（纯物理 3 秒）")
    rest, dur = test_rest(a.testdir)
    for k, v in rest.items():
        log(f"  {k:14s} 漂移 {v['drift_mm']:.4f} mm   最大偏离 {v['max_dev_mm']:.4f} mm   "
            f"二阶差分 {v['jitter_mm']:.6f} mm")
        log(f"                 z {v['z_start_mm']:.2f} → {v['z_end_mm']:.2f} mm")
    log(f"  仿真时长 {dur:.2f} s")

    hdr("B. 等效接触对：抓持 → 抬升 → 平移 → 放下")
    tr, T = test_transport(a.testdir, outdir)
    log(f"  最大抬升 {tr['max_lift_mm']:.1f} mm")
    log(f"  水平位移 {tr['horizontal_move_mm']:.1f} mm")
    log(f"  落定后 z {tr['z_settled_mm']:.2f} mm（起始 {tr['object_start'][2]*1000:.2f} mm）")
    log(f"  轨迹二阶差分（抖动） {tr['jitter_z_mm']:.4f} mm")
    log(f"  物理步数 {tr['steps']}  ({tr['duration_s']:.2f} s)")

    hdr("C. 与 Stage 1.5 对比")
    tj = json.load(open(os.path.join(a.testdir, "trajectory_cam.json")))["trajectory"]
    _, _, _, _, c2w, _ = load_geometry(a.testdir)
    st15 = np.array([c2w(p) for p in tj["earbud_case"] if p])
    j15 = float(np.abs(np.diff(st15[:, 2], 2)).mean() * 1000)
    log(f"  Stage 1.5 逐帧深度拟合 z 二阶差分: {j15:.4f} mm")
    log(f"  物理仿真（本脚本）            : {tr['jitter_z_mm']:.4f} mm")
    log(f"  改善倍数: {j15 / max(tr['jitter_z_mm'], 1e-9):.1f}×")

    json.dump({"rest": rest, "rest_duration_s": dur,
               "transport": {k: v for k, v in tr.items() if k != "records"},
               "stage15_jitter_mm": j15,
               "improvement_factor": j15 / max(tr["jitter_z_mm"], 1e-9)},
              open(os.path.join(outdir, "physics_validation.json"), "w"),
              indent=2, ensure_ascii=False)
    np.save(os.path.join(outdir, "transport_traj.npy"), T)
    log(f"\n写出 {outdir}/physics_validation.json")


if __name__ == "__main__":
    main()
