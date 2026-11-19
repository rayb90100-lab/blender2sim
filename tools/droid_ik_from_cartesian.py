#!/usr/bin/env python3
"""从末端位姿解算关节角（师兄说的「解算」路线的后半段），与 DROID 关节真值对比。

链路：cartesian_position（法兰位姿, 欧拉 xyz）→ IK → 7 关节 → 物理执行
  - 数据集的 cartesian_position 是机器人自身记录，不是视觉估计；
    用 --noise_pos_mm / --noise_rot_deg 注入噪声，模拟「视频 → 末端位姿」的估计误差。
  - IK 初值：首帧用通用 home 姿态（不偷看真值），之后逐帧用上一帧解热启动。
  - Panda 7 自由度冗余：同一末端位姿可有不同肘部姿态（零空间），
    所以关节误差 ≠ 末端误差，两者分开报告。

用法:
  python tools/droid_ik_from_cartesian.py --testdir <testdir> [--noise_pos_mm 0 --noise_rot_deg 0 --tag clean]
输出 stage2_physics/ik_<tag>/q_ik.npy + ik_report.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import droid_common as C  # noqa: E402

PANDA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def log(m): print(m, flush=True)


def rot_err(Rc, Rd):
    from scipy.spatial.transform import Rotation as R
    return R.from_matrix(Rd @ Rc.T).as_rotvec()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--noise_pos_mm", type=float, default=0.0)
    ap.add_argument("--noise_rot_deg", type=float, default=0.0)
    ap.add_argument("--noise_mode", choices=["white", "smooth"], default="smooth")
    ap.add_argument("--tag", default="clean")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import mujoco
    from scipy.optimize import least_squares
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation as R

    T = a.testdir
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    Jgt = ep["observation.state.joint_position"]
    X = ep["observation.state.cartesian_position"].copy()
    n = len(X)
    m, d, _ = C.build_model()
    qa = C.arm_qadr(m)
    site = m.site("attachment_site").id
    lo = np.array([m.jnt_range[m.joint(j).id][0] for j in C.PANDA_JOINTS])
    hi = np.array([m.jnt_range[m.joint(j).id][1] for j in C.PANDA_JOINTS])

    # 法兰系 vs attachment_site 的固定旋转偏置（Stage 0 实测恒为 135°），用首帧真值标定一次。
    # 这是机器人模型常数，不是逐帧偷看真值。
    d.qpos[qa] = Jgt[0]
    mujoco.mj_kinematics(m, d)
    R_off = C.euler_to_R(X[0, 3:], "xyz").T @ d.site_xmat[site].reshape(3, 3)

    # ---- 注入噪声（模拟视觉估计误差）----
    rng = np.random.default_rng(a.seed)
    Xn = X.copy()
    if a.noise_pos_mm > 0 or a.noise_rot_deg > 0:
        npos = rng.normal(0, 1, (n, 3))
        nrot = rng.normal(0, 1, (n, 3))
        if a.noise_mode == "smooth":
            # 视觉估计误差通常是低频漂移 + 小抖动，不是独立白噪声
            npos = gaussian_filter1d(npos, 3, axis=0)
            nrot = gaussian_filter1d(nrot, 3, axis=0)
            npos /= npos.std(0, keepdims=True)
            nrot /= nrot.std(0, keepdims=True)
        Xn[:, :3] += npos * a.noise_pos_mm / 1000
        for i in range(n):
            Ri = C.euler_to_R(X[i, 3:], "xyz")
            Rn = R.from_rotvec(np.radians(a.noise_rot_deg) * nrot[i]).as_matrix() @ Ri
            Xn[i, 3:] = R.from_matrix(Rn).as_euler("xyz")

    # ---- 逐帧 IK ----
    W_ROT = 0.2          # 1 rad 姿态误差 ≈ 0.2 m 位置误差
    W_SMOOTH = 0.02      # 关节连续性（防止零空间跳变）
    q_prev = PANDA_HOME.copy()
    Q = np.zeros((n, 7))
    pos_err, ang_err = [], []
    for i in range(n):
        tgt_p = Xn[i, :3]
        tgt_R = C.euler_to_R(Xn[i, 3:], "xyz") @ R_off

        def resid(q):
            d.qpos[qa] = q
            mujoco.mj_kinematics(m, d)
            ep_ = tgt_p - d.site_xpos[site]
            er = rot_err(d.site_xmat[site].reshape(3, 3), tgt_R)
            return np.concatenate([ep_, W_ROT * er, W_SMOOTH * (q - q_prev)])

        starts = [q_prev] if i > 0 else [PANDA_HOME] + [rng.uniform(lo, hi) for _ in range(15)]
        best = None
        for x0 in starts:
            sol = least_squares(resid, np.clip(x0, lo, hi), bounds=(lo, hi), method="trf",
                                xtol=1e-10, ftol=1e-10, max_nfev=300)
            if best is None or sol.cost < best.cost:
                best = sol
        Q[i] = best.x
        q_prev = best.x
        # 相对「无噪声真值末端」的误差（衡量执行出来的末端离真实轨迹多远）
        d.qpos[qa] = best.x
        mujoco.mj_kinematics(m, d)
        pos_err.append(np.linalg.norm(X[i, :3] - d.site_xpos[site]) * 1000)
        Rt = C.euler_to_R(X[i, 3:], "xyz") @ R_off
        ang_err.append(np.degrees(np.linalg.norm(rot_err(d.site_xmat[site].reshape(3, 3), Rt))))

    pos_err, ang_err = np.array(pos_err), np.array(ang_err)
    jerr = np.degrees(np.abs(Q - Jgt))
    vel = np.degrees(np.abs(np.diff(Q, axis=0))) * 15
    vel_gt = np.degrees(np.abs(np.diff(Jgt, axis=0))) * 15
    out = os.path.join(T, "stage2_physics", f"ik_{a.tag}")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "q_ik.npy"), Q)
    rep = {
        "noise": {"pos_mm": a.noise_pos_mm, "rot_deg": a.noise_rot_deg, "mode": a.noise_mode, "seed": a.seed},
        "ee_pos_err_mm_vs_true": {"median": float(np.median(pos_err)), "max": float(pos_err.max())},
        "ee_rot_err_deg_vs_true": {"median": float(np.median(ang_err)), "max": float(ang_err.max())},
        "joint_err_deg_vs_gt": {"median_per_joint": np.median(jerr, 0).round(2).tolist(),
                                "max_per_joint": jerr.max(0).round(2).tolist(),
                                "overall_median": float(np.median(jerr))},
        "joint_speed_max_deg_s": {"ik": float(vel.max()), "gt": float(vel_gt.max())},
        "init": "首帧 Panda 通用 home + 15 个随机起点（不使用真值）；之后逐帧热启动",
        "R_offset_note": "法兰↔attachment_site 固定旋转偏置用首帧标定（模型常数 135°）",
    }
    json.dump(rep, open(os.path.join(out, "ik_report.json"), "w"), indent=2, ensure_ascii=False)
    log(f"[{a.tag}] 噪声 pos {a.noise_pos_mm} mm / rot {a.noise_rot_deg}°")
    log(f"  末端位置误差（vs 真实轨迹） 中位 {np.median(pos_err):.2f} mm  最大 {pos_err.max():.2f} mm")
    log(f"  末端姿态误差                中位 {np.median(ang_err):.2f}°   最大 {ang_err.max():.2f}°")
    log(f"  关节误差 vs 真值（逐关节中位，°） {np.median(jerr, 0).round(2).tolist()}")
    log(f"  关节误差 vs 真值（逐关节最大，°） {jerr.max(0).round(2).tolist()}")
    log(f"  最大关节速度 IK {vel.max():.1f}°/s  真值 {vel_gt.max():.1f}°/s")


if __name__ == "__main__":
    main()
