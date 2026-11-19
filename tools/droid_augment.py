#!/usr/bin/env python3
"""Stage 3 场景增强：参数化场景变体 + 按物体锚定的动作适配 + 原生物理验证。

skill augmentation.md：几何/布局修改后，依据对象局部抓点与目标调整动作，不只平移整条关节轨迹；
每个候选按原生执行验收，不用状态写回、附着、隐藏碰撞代替任务成功。

动作适配（分段锚定，源自 DROID 真实末端轨迹 cartesian_position）：
  t ≤ T_GRASP（f88，两侧指垫贴合）  : 末端位姿相对 绿方块 锚定
  t ≥ T_RELEASE（f142，张开）        : 末端位姿相对 塔顶红块 锚定
  中间（搬运）                       : 两种锚定变换按时间平滑过渡（平移线性、yaw 插值）
  每帧 T_ee_new = D(t) · T_ee_old(t)，D = 绕锚点竖直轴的 (Δx, Δy, Δz, Δyaw) 刚体变换
  → IK（scipy least_squares，热启动）→ 关节 → 物理（droid_stage2_physics.run, qfile）

参数（单位：m / deg）:
  green_dx, green_dy, green_dyaw   绿方块
  tower_dx, tower_dy, tower_dyaw   整座塔（拱/长板/红块同步，绕红块中心竖直轴）
  table_dz                         桌面及其上全部物体（机器人底座不动）

用法:
  python tools/droid_augment.py --testdir <T> --cand <id> --params '{"green_dx":0.03}'
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import droid_common as C  # noqa: E402

T_GRASP, T_RELEASE = 88, 142
TOWER = ("stack_arch", "stack_plank", "stack_cube")
PARAM_KEYS = ("green_dx", "green_dy", "green_dyaw", "tower_dx", "tower_dy", "tower_dyaw", "table_dz")


def log(m): print(m, flush=True)


def rotz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def delta_T(pivot, dx, dy, dz, dyaw):
    """绕 pivot 竖直轴转 dyaw，再平移 (dx,dy,dz) 的 4×4 变换。"""
    D = np.eye(4)
    Rz = rotz(dyaw)
    D[:3, :3] = Rz
    D[:3, 3] = np.asarray(pivot) - Rz @ np.asarray(pivot) + np.array([dx, dy, dz])
    return D


def sym_yaw(dyaw, period=90.0):
    """方块 90° 对称、平行夹爪 180° 对称：夹爪只需转与 dyaw 等价的最小角。"""
    return float((dyaw + period / 2) % period - period / 2)


# ---------------- 场景变体 ----------------
def make_scene(base, p):
    s = json.loads(json.dumps(base))
    o = s["objects"]
    dz = p.get("table_dz", 0.0)
    s["table_z"] = base["table_z"] + dz
    # 绿方块
    g = o["green_block"]
    g["center"] = [g["center"][0] + p.get("green_dx", 0.0), g["center"][1] + p.get("green_dy", 0.0), g["center"][2] + dz]
    g["yaw_rad"] = g["yaw_rad"] + np.radians(p.get("green_dyaw", 0.0))
    # 塔：绕红块中心竖直轴转，再平移
    piv = np.array(o["stack_cube"]["center"][:2])
    Rz = rotz(p.get("tower_dyaw", 0.0))[:2, :2]
    for n in TOWER:
        c = np.array(o[n]["center"])
        xy = piv + Rz @ (c[:2] - piv) + np.array([p.get("tower_dx", 0.0), p.get("tower_dy", 0.0)])
        o[n]["center"] = [float(xy[0]), float(xy[1]), float(c[2] + dz)]
        o[n]["yaw_rad"] = o[n]["yaw_rad"] + np.radians(p.get("tower_dyaw", 0.0))
    s["augment_params"] = p
    return s


# ---------------- 动作适配 ----------------
def adapt_ee(ep, base, p):
    X = ep["observation.state.cartesian_position"]
    n = len(X)
    dz = p.get("table_dz", 0.0)
    g0 = np.array(base["objects"]["green_block"]["center"])
    c0 = np.array(base["objects"]["stack_cube"]["center"])
    # 夹爪需转的角度：按方块 90° 对称取最小
    D_g = delta_T(g0, p.get("green_dx", 0.0), p.get("green_dy", 0.0), dz, sym_yaw(p.get("green_dyaw", 0.0)))
    # 放置：塔转 dyaw 后，绿方块落在红块上的相对朝向按 90° 对称取最小
    D_t = delta_T(c0, p.get("tower_dx", 0.0), p.get("tower_dy", 0.0), dz, sym_yaw(p.get("tower_dyaw", 0.0)))
    out = np.zeros((n, 4, 4))
    for t in range(n):
        T_old = C.pose6_to_T(X[t])
        if t <= T_GRASP:
            D = D_g
        elif t >= T_RELEASE:
            D = D_t
        else:
            a = (t - T_GRASP) / (T_RELEASE - T_GRASP)
            a = 0.5 - 0.5 * np.cos(np.pi * a)          # 平滑过渡
            dg, dt = D_g, D_t
            yaw_g = np.degrees(np.arctan2(dg[1, 0], dg[0, 0]))
            yaw_t = np.degrees(np.arctan2(dt[1, 0], dt[0, 0]))
            D = np.eye(4)
            D[:3, :3] = rotz((1 - a) * yaw_g + a * yaw_t)
            # 平移部分：对当前末端位置分别求两种变换后的位置再插值
            pg = dg[:3, :3] @ T_old[:3, 3] + dg[:3, 3]
            pt = dt[:3, :3] @ T_old[:3, 3] + dt[:3, 3]
            D[:3, 3] = (1 - a) * pg + a * pt - D[:3, :3] @ T_old[:3, 3]
        out[t] = D @ T_old
    return out


def ik_track(ep, T_ee):
    """逐帧 IK：目标为法兰位姿；首帧用真值关节热启动（场景未变的初态），之后用上一帧解。"""
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R
    J = ep["observation.state.joint_position"]
    X = ep["observation.state.cartesian_position"]
    m, d, _ = C.build_model()
    qa = C.arm_qadr(m)
    site = m.site("attachment_site").id
    lo = np.array([m.jnt_range[m.joint(j).id][0] for j in C.PANDA_JOINTS])
    hi = np.array([m.jnt_range[m.joint(j).id][1] for j in C.PANDA_JOINTS])
    d.qpos[qa] = J[0]
    mujoco.mj_kinematics(m, d)
    R_off = C.euler_to_R(X[0, 3:], "xyz").T @ d.site_xmat[site].reshape(3, 3)
    q = J[0].copy()
    Q = np.zeros((len(T_ee), 7))
    pe, re = [], []
    for t, Tt in enumerate(T_ee):
        tp, tR = Tt[:3, 3], Tt[:3, :3] @ R_off

        def resid(x):
            d.qpos[qa] = x
            mujoco.mj_kinematics(m, d)
            e_p = tp - d.site_xpos[site]
            e_r = R.from_matrix(tR @ d.site_xmat[site].reshape(3, 3).T).as_rotvec()
            return np.concatenate([e_p, 0.2 * e_r, 0.02 * (x - q)])

        sol = least_squares(resid, np.clip(q, lo, hi), bounds=(lo, hi), method="trf",
                            xtol=1e-10, ftol=1e-10, max_nfev=200)
        q = sol.x
        Q[t] = q
        d.qpos[qa] = q
        mujoco.mj_kinematics(m, d)
        pe.append(np.linalg.norm(tp - d.site_xpos[site]) * 1000)
        re.append(np.degrees(np.linalg.norm(R.from_matrix(tR @ d.site_xmat[site].reshape(3, 3).T).as_rotvec())))
    return Q, np.array(pe), np.array(re)


def run_candidate(testdir, cand, p, base=None):
    import droid_stage2_physics as P
    ep = C.load_episode(os.path.join(testdir, "input", "episode.parquet"))
    base = base or json.load(open(os.path.join(testdir, "stage1_scene", "scene.json")))
    out = os.path.join(testdir, "stage3_augment", "candidates", cand)
    os.makedirs(out, exist_ok=True)
    scene = make_scene(base, p)
    sp = os.path.join(out, "scene.json")
    json.dump(scene, open(sp, "w"), indent=2, ensure_ascii=False)
    T_ee = adapt_ee(ep, base, p)
    Q, pe, re = ik_track(ep, T_ee)
    np.save(os.path.join(out, "q_adapted.npy"), Q)
    os.environ["DROID_SCENE_JSON"] = sp
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rep = P.run(testdir, qfile=os.path.join(out, "q_adapted.npy"),
                        tag=os.path.relpath(out, os.path.join(testdir, "stage2_physics")))
    finally:
        os.environ.pop("DROID_SCENE_JSON", None)
    # 用与 test4 相同的判据评估
    crit = P.CRITERIA["rules"]
    metrics = {
        "task_success": bool(rep["task_success"]),
        "green_end_xy_err_mm": float(np.linalg.norm(rep["green_end_xy_vs_stack_mm"])),
        "green_end_z_err_mm": float(abs(rep["green_end_z_mm"] - rep["green_end_z_expected_on_stack_mm"])),
        "green_end_tilt_deg": float(np.max(np.abs(rep["green_end_tilt_deg"]))),
        "support_ratio": float(rep["green_on_stack_force_last1s_N"] / rep["green_weight_N"]),
        "grip_contact_frames": int(rep["grip_contact_frames"]),
        "stack_drift_mm": float(rep["stack_drift_mm"]),
        "tower_max_drift_mm": float(rep["tower_max_drift_mm"]),
        "tower_max_tilt_deg": float(rep["tower_max_tilt_deg"]),
        "min_contact_dist_mm": float(rep["min_contact_dist_mm"]),
    }
    fails = [k for k, r in crit.items() if not (
        (metrics[k] == r["value"]) if r["op"] == "eq" else
        (metrics[k] <= r["value"]) if r["op"] == "le" else (metrics[k] >= r["value"]))]
    res = {"candidate": cand, "params": p, "ik_pos_err_mm": {"median": float(np.median(pe)), "max": float(pe.max())},
           "ik_rot_err_deg": {"median": float(np.median(re)), "max": float(re.max())},
           "metrics": metrics, "failed_criteria": fails, "pass": not fails,
           "grip_contact_interval": rep["grip_contact_interval"]}
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=2, ensure_ascii=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--params", required=True)
    a = ap.parse_args()
    r = run_candidate(a.testdir, a.cand, json.loads(a.params))
    log(json.dumps({k: r[k] for k in ("candidate", "params", "ik_pos_err_mm", "pass", "failed_criteria",
                                      "grip_contact_interval")}, ensure_ascii=False))
    log(json.dumps(r["metrics"], ensure_ascii=False))
