#!/usr/bin/env python3
"""Stage 2：事件驱动的物理执行——把源视频的物体轨迹变成 MuJoCo 原生轨迹。

核心主张（本轮要验证的）：
  Stage 1.5 逐帧单目深度拟合出来的物体轨迹有 ±3.6~4.9 mm 抖动、遮挡处的假位移。
  物理引擎里物体由重力与接触推进，中间帧不是估计出来的，所以这些伪运动从
  根本上不存在——只要 (a) 事件时刻的物体位姿对，(b) 抓取真的成立。

做法（按 skill motion-execution / mujoco 的分阶段要求）：
  1. 读 Stage 1 的 manifest + 源轨迹，转到世界系
  2. 从物体轨迹反推右臂 TCP 参考（human_to_robot 的替代：无手部关键点，
     用物体轨迹 + 抓取偏置，manifest 里声明）
  3. 逐事件分段执行：静置 → 接近 → 闭合 → 抬升 → 搬运 → 放下 → 撤离
  4. 每步物理推进，记录物体/关节/接触力
  5. 独立回归：从磁盘模型重跑一遍，对比抖动指标

用法:
  python tools/stage2_execute.py --testdir <testdir> [--samples 96]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_common import load_geometry  # noqa: E402

import mujoco  # noqa: E402

OBJECT = "earbud_case"
GRASP_SIDE = "right"          # 右臂执行，左臂待机
# actuator ctrl 与指间距线性映射（实测）：gap = ctrl/255 * 80 mm
GRIP_OPEN = 255.0                  # 80 mm
GRIP_BOX = 255.0 * 0.024 / 0.080   # 半闭到 24 mm（盒宽 26.3 mm，先贴合）
GRIP_SHUT = 0.0                    # 全闭；盒子挡住后由接触建立夹持

# ★ 抓取高度：物体顶面 + 35 mm
# 实测（固定末端姿态扫开口）：手掌碰撞几何的最低点比 TCP 高 90 mm。
# 所以指尖不可能够到贴着桌面的物体中心——那样手掌一定会撞桌子或盒顶。
# 平行夹爪抓 26×59×46 mm 的小盒子，正常做法是夹它的**上沿**：
# 指尖落在盒顶上方，两指接触盒子上部侧壁。
# 之前用 0 和 53 mm 都失败过：0 推走 11 cm（手指框住盒身、指腹撞盒顶），
# 53 mm 时手指悬空没夹到。
GRASP_LIFT = 0.0231 + 0.035        # 半高 23.1 + 35 mm

# TCP：指尖中点相对 link8 的偏置，实测于 home 姿态（见 tools/stage2_measure_tcp.py）
TCP_OFFSET = np.array([-0.1233, -0.0415, -0.1021])

# 抓取姿态：link8 本地 +z 指向世界 -z（指尖朝下）。
# 实测：指尖方向 == link8 的 +z 轴（见下方注释），所以约束 z 列 = (0,0,-1)。
# 列是「本地轴在世界系中的方向」：R[:,2] 就是本地 +z。
# yaw 留自由——盒子接近各向同性，绕竖直轴的角度不影响抓取。
def grasp_rot(yaw=0.0):
    zc = np.array([0.0, 0.0, -1.0])
    xc = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    yc = np.cross(zc, xc)
    return np.stack([xc, yc, zc], axis=1)


GRASP_ROT = grasp_rot(0.0)

# 事件表（来自视频独立观察，见 stage2_physics/events/events.json）
EVENTS = [
    ("rest_before", 0, 80),
    ("approach", 80, 86),
    ("grasp", 86, 90),
    ("lift_transport", 90, 166),
    ("release", 166, 170),
    ("settle_after", 170, 244),
]


def log(m): print(m, flush=True)
def hdr(m): print(f"\n{'='*64}\n{m}\n{'='*64}", flush=True)


# ---------------- IK ----------------
def _tcp(m, d, bid):
    """当前指尖中点（TCP）世界坐标。"""
    f1 = m.body(f"{GRASP_SIDE}_left_finger").id
    f2 = m.body(f"{GRASP_SIDE}_right_finger").id
    return (d.xpos[f1] + d.xpos[f2]) / 2


def _rot_err(R_cur, R_des):
    """旋转误差向量（轴角），单位为弧度。"""
    Rerr = R_des @ R_cur.T
    ang = np.arccos(np.clip((np.trace(Rerr) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return np.zeros(3)
    s = np.sin(ang)
    if abs(s) < 1e-9:
        return np.zeros(3)
    ax = np.array([Rerr[2, 1] - Rerr[1, 2],
                   Rerr[0, 2] - Rerr[2, 0],
                   Rerr[1, 0] - Rerr[0, 1]]) / (2 * s)
    return ang * ax


def ik_to(m, d, qadr, target_pos, grip=0.04, want_rot=None, w_rot=0.25,
          restarts=12, seed=0, iters=300):
    """用 scipy 带边界最小二乘解 IK（位置 + 可选姿态）。

    手写 DLS 在这个问题上收敛很差（姿态误差长期卡在 46-66°），
    因为 6 维约束配 7 个关节、且阻尼步长和权重互相牵制。
    Stage 1 里用 least_squares 解盒拟合效果很好，这里沿用同一思路：
    残差 = [位置误差, w_rot × 旋转误差]，变量 = 7 个关节角，带关节限位。
    """
    from scipy.optimize import least_squares

    bid = m.body(f"{GRASP_SIDE}_fr3v2_link8").id
    fadr = [int(m.joint(f"{GRASP_SIDE}_finger_joint{k}").qposadr[0]) for k in (1, 2)]
    for a in fadr:
        d.qpos[a] = grip
    target = np.asarray(target_pos, float)
    jid = [int(np.flatnonzero(m.jnt_qposadr == a)[0]) for a in qadr]
    lo = np.array([m.jnt_range[k][0] for k in jid])
    hi = np.array([m.jnt_range[k][1] for k in jid])

    def resid(q):
        for k, a in enumerate(qadr):
            d.qpos[a] = q[k]
        for a in fadr:
            d.qpos[a] = grip
        mujoco.mj_forward(m, d)
        ep = target - _tcp(m, d, bid)
        if want_rot is None:
            return ep
        R = d.xmat[bid].reshape(3, 3)
        return np.concatenate([ep, w_rot * _rot_err(R, want_rot)])

    rng = np.random.default_rng(seed)
    best_q = np.array([d.qpos[a] for a in qadr])
    best_cost = np.inf
    for r in range(restarts):
        x0 = best_q if r == 0 else rng.uniform(lo, hi)
        try:
            sol = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                                xtol=1e-10, ftol=1e-10, gtol=1e-10,
                                max_nfev=iters)
        except Exception:
            continue
        # 只按位置误差选最优（姿态是软约束）
        for k, a in enumerate(qadr):
            d.qpos[a] = sol.x[k]
        for a in fadr:
            d.qpos[a] = grip
        mujoco.mj_forward(m, d)
        pe = float(np.linalg.norm(target - _tcp(m, d, bid)))
        if pe < best_cost:
            best_cost, best_q = pe, sol.x.copy()
    for k, a in enumerate(qadr):
        d.qpos[a] = best_q[k]
    for a in fadr:
        d.qpos[a] = grip
    mujoco.mj_forward(m, d)
    return best_cost


# ---------------- 场景加载 ----------------
def load(testdir, scene=None):
    p = scene or os.path.join(testdir, "stage2_physics", "mujoco", "scene.xml")
    m = mujoco.MjModel.from_xml_path(p)
    d = mujoco.MjData(m)
    return m, d


def arm_qadr(m, side):
    return [int(m.joint(f"{side}_fr3v2_joint{k}").qposadr[0]) for k in range(1, 8)]


def set_home(m, d, home_q):
    for side in ("left", "right"):
        qa = arm_qadr(m, side)
        for k, a in enumerate(qa):
            d.qpos[a] = home_q[k]
        d.qpos[int(m.joint(f"{side}_finger_joint1").qposadr[0])] = 0.04
        d.qpos[int(m.joint(f"{side}_finger_joint2").qposadr[0])] = 0.04
    d.ctrl[:7] = home_q
    d.ctrl[7:14] = home_q
    d.ctrl[14] = 0.04
    d.ctrl[15] = 0.04
    mujoco.mj_forward(m, d)


def obj_qadr(m, name):
    return int(m.joint(f"{name}_free").qposadr[0])


def record(m, d, tag, step, rec):
    e = obj_qadr(m, "earbud_case")
    t = obj_qadr(m, "tissue_box")
    bid = m.body(f"{GRASP_SIDE}_fr3v2_link8").id
    # 夹爪两侧接触力（世界系，指向物体的合力）
    fsum = np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        n1, n2 = m.geom(c.geom1).name or "", m.geom(c.geom2).name or ""
        if ("earbud" in n1 and "finger" in n2) or ("earbud" in n2 and "finger" in n1):
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            fsum += f[:3]
    rec.append({
        "tag": tag, "step": step, "t": step * m.opt.timestep,
        "earbud": d.qpos[e:e + 3].tolist(),
        "tissue": d.qpos[t:t + 3].tolist(),
        "ee": d.xpos[bid].tolist(),
        "finger1": float(d.qpos[int(m.joint(f"{GRASP_SIDE}_finger_joint1").qposadr[0])]),
        "grasp_force": float(np.linalg.norm(fsum)),
        "ncon": int(d.ncon),
    })


# ---------------- 关节空间驱动 ----------------
def drive_to(m, d, ctrl_idx, q_goal, n_steps, grip=None, rec=None, tag=""):
    """用 actuator 把关节平滑驱动到目标角度。

    不能直接赋 d.qpos：那是瞬移，物体会被挤压弹出（实测耳机盒被推到 (0.33, 0.19)）。
    这里把 ctrl 从当前值线性插值到目标，由位置伺服逐步推进。
    """
    q_from = np.array([d.ctrl[i] for i in ctrl_idx])
    q_goal = np.asarray(q_goal, float)
    for k in range(n_steps):
        alpha = (k + 1) / n_steps
        q = q_from + alpha * (q_goal - q_from)
        for idx, ci in enumerate(ctrl_idx):
            d.ctrl[ci] = q[idx]
        if grip is not None:
            for ci in grip[0]:
                d.ctrl[ci] = grip[1]
        mujoco.mj_step(m, d)
        if rec is not None:
            record(m, d, tag, k, rec)
    return


def settle(m, d, n, grip=None, rec=None, tag=""):
    for k in range(n):
        if grip is not None:
            for ci in grip[0]:
                d.ctrl[ci] = grip[1]
        mujoco.mj_step(m, d)
        if rec is not None:
            record(m, d, tag, k, rec)


# ---------------- 主流程 ----------------
def run(testdir, scene=None):
    man, n_cam, d_cam, R_wc, c2w, d2w = load_geometry(testdir)
    m, d = load(testdir, scene)
    placement = json.load(open(os.path.join(testdir, "stage2_physics", "mujoco",
                                            "robot_placement.json")))
    home_q = placement["home_q"]
    set_home(m, d, home_q)

    obj_adr = obj_qadr(m, OBJECT)
    obj0 = d.qpos[obj_adr:obj_adr + 3].copy()
    grasp_p = obj0 + np.array([0.0, 0.0, GRASP_LIFT])

    # actuator 索引
    ctrl_R = [i for i in range(m.nu) if m.actuator(i).name.startswith(f"{GRASP_SIDE}_fr3v2")]
    ctrl_f = [i for i in range(m.nu) if "finger_act" in m.actuator(i).name
              and GRASP_SIDE in m.actuator(i).name]

    rec = []
    hdr("阶段 A：静置与标定")
    settle(m, d, 500, grip=(ctrl_f, 0.04), rec=rec, tag="rest")
    log(f"  静置后耳机盒: {np.round(d.qpos[obj_adr:obj_adr+3], 5)}")
    log(f"  纸巾盒:       {np.round(d.qpos[obj_qadr(m, 'tissue_box'):+3], 5)}")

    qa = arm_qadr(m, GRASP_SIDE)
    pre = grasp_p + np.array([0.0, 0.0, 0.12])
    ik_to(m, d, qa, pre, grip=0.04, want_rot=GRASP_ROT, w_rot=0.25, restarts=12)
    q_pre = np.array([d.qpos[a] for a in qa]).copy()
    log(f"IK 预抓取位已解出（姿态约束 z 轴朝下）")

    hdr("阶段 B：接近 → 闭合")
    drive_to(m, d, ctrl_R, q_pre, 900, grip=(ctrl_f, 0.04), rec=rec, tag="approach")
    log(f"  接近后耳机盒: {np.round(d.qpos[obj_adr:obj_adr+3], 5)}")

    ik_to(m, d, qa, grasp_p, grip=0.04, want_rot=GRASP_ROT, w_rot=0.25, restarts=12)
    q_grasp = np.array([d.qpos[a] for a in qa]).copy()
    drive_to(m, d, ctrl_R, q_grasp, 700, grip=(ctrl_f, 0.04), rec=rec, tag="descend")
    log(f"  下探后耳机盒: {np.round(d.qpos[obj_adr:obj_adr+3], 5)}")

    settle(m, d, 300, grip=(ctrl_f, GRIP_BOX), rec=rec, tag="pre_close")
    log(f"  半闭后耳机盒: {np.round(d.qpos[obj_adr:obj_adr+3], 5)}")
    settle(m, d, 600, grip=(ctrl_f, GRIP_SHUT), rec=rec, tag="close")
    closed = d.qpos[obj_adr:obj_adr + 3].copy()
    log(f"  闭合后耳机盒: {np.round(closed, 5)}")
    log(f"  相对闭合前位移: {np.linalg.norm(closed - np.array(rec[-601]['earbud']))*1000:.2f} mm")
    f1 = m.body(f"{GRASP_SIDE}_left_finger").id
    f2 = m.body(f"{GRASP_SIDE}_right_finger").id
    log(f"  两指间距: {np.linalg.norm(d.xpos[f1]-d.xpos[f2])*1000:.1f} mm（盒宽 26.3 mm）")

    return m, d, rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    a = ap.parse_args()
    run(a.testdir)
