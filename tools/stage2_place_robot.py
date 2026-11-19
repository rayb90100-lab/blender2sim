#!/usr/bin/env python3
"""搜索 FR3 基座位姿，使单臂可达抓取点且无自碰撞。

用法:
  python tools/stage2_place_robot.py --testdir <testdir> [--apply]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_common import load_geometry  # noqa: E402

import mujoco  # noqa: E402

HOME_Q = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]


def build_variant(testdir, base_xy, base_yaw, out="scene.xml"):
    """用给定基座位姿重建 scene.xml，返回路径。"""
    src = os.path.join(testdir, "stage2_physics", "mujoco", "scene.xml")
    txt = open(src).read()
    import re
    txt = re.sub(
        r'<body name="robot_base"\s+pos="[^"]*"\s+euler="[^"]*">',
        f'<body name="robot_base" pos="{base_xy[0]:.4f} {base_xy[1]:.4f} 0.0" '
        f'euler="0 0 {base_yaw:.4f}">', txt, flags=re.S)
    p = os.path.join(testdir, "stage2_physics", "mujoco", out)
    open(p, "w").write(txt)
    return p


def evaluate(path, target, side="right", q=None):
    q = q or HOME_Q
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    # 两臂都必须设成合法姿态：留 qpos=0 时 link4/5/7 会互相穿插，
    # 这不是基座问题，而是零位本身是坏姿态
    for s_ in ("left", "right"):
        off = int(m.joint(f"{s_}_fr3v2_joint1").qposadr[0])
        d.qpos[off:off + 7] = q
        d.qpos[int(m.joint(f"{s_}_finger_joint1").qposadr[0])] = 0.04
        d.qpos[int(m.joint(f"{s_}_finger_joint2").qposadr[0])] = 0.04
    mujoco.mj_forward(m, d)

    ee = d.xpos[m.body(f"{side}_fr3v2_link8").id].copy()
    dist = float(np.linalg.norm(ee - np.asarray(target)))

    self_col = 0
    deepest = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        n1 = m.geom(c.geom1).name or ""
        n2 = m.geom(c.geom2).name or ""
        if any(k in n1 + n2 for k in ("table", "floor", "earbud", "tissue")):
            continue
        if c.dist < -0.001:
            self_col += 1
            deepest = min(deepest, c.dist)
    return dist, self_col, deepest, ee


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    man, n_cam, d_cam, R_wc, c2w, d2w = load_geometry(a.testdir)
    target = c2w(man["instances"]["earbud_case"]["center_cam"])
    # 抓取点略高于物体中心，且从上方进入
    grasp = target + np.array([0.0, 0.0, 0.02])
    print(f"抓取目标点（世界系）: {np.round(grasp,4)}")

    best = None
    for bx in [-0.30, -0.20, -0.10, 0.0, 0.10]:
        for by in [-0.34, -0.26, -0.18, -0.10]:
            for yaw in [0.0, 0.7854, 1.5708, 2.3562, 3.1416, -0.7854, -1.5708, -2.3562]:
                p = build_variant(a.testdir, (bx, by), yaw, out="_probe.xml")
                try:
                    dist, sc, dp, ee = evaluate(p, grasp)
                except Exception:
                    continue
                if sc == 0 and (best is None or dist < best[0]):
                    best = (dist, (bx, by), yaw, ee.copy())
    if os.path.exists(os.path.join(a.testdir, "stage2_physics", "mujoco", "_probe.xml")):
        os.remove(os.path.join(a.testdir, "stage2_physics", "mujoco", "_probe.xml"))

    if best is None:
        print("没有找到无自碰撞的可达位姿")
        return
    dist, (bx, by), yaw, ee = best
    print(f"\n最优: 基座=({bx}, {by}) yaw={np.degrees(yaw):.1f}°")
    print(f"  末端={np.round(ee,4)}  离抓取点={dist*100:.1f} cm")
    print(f"  （注意：这是 home 姿态下的距离，实际由 IK 收敛）")

    if a.apply:
        build_variant(a.testdir, (bx, by), yaw)
        json.dump({"base_pos": [bx, by, 0.0], "base_yaw": yaw,
                   "home_q": HOME_Q, "grasp_target": grasp.tolist(),
                   "init_ee_dist_m": dist,
                   "note": "基座位姿由 tools/stage2_place_robot.py 搜索，"
                           "判据=无自碰撞 + home 末端离抓取点最近；"
                           "实际可达性由 IK 收敛验证，不是这个距离"},
                  open(os.path.join(a.testdir, "stage2_physics", "mujoco",
                                    "robot_placement.json"), "w"), indent=2)
        print("\n已应用")


if __name__ == "__main__":
    main()
