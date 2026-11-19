#!/usr/bin/env python3
"""回归：在 r2pear 新仿真（地面 + 真实桌形 + 力矩上限）中按 DROID 关节真值回放，终态谓词应判成功。
证明 §61.2 的鲁棒性修改没有破坏原任务的物理可执行性。

用法: python tools/r2pear/check_demo_replay.py --spec <task_spec.json> --episode <episode.parquet> --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import droid_common as C  # noqa: E402
import predicates as PR  # noqa: E402
import sim as S  # noqa: E402
import spec as SP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mujoco
    spec = SP.load(a.spec)
    ep = C.load_episode(a.episode)
    J = ep["observation.state.joint_position"]
    G = ep["observation.state.gripper_position"].ravel()
    m, d = S.build(spec)
    S.reset(m, d, spec)
    arm, g = S.arm_actuators(m), S.grip_actuator(m)
    sub = int(round(1 / 15 / m.opt.timestep))
    min_dist = 0.0
    track_err = []
    qa = C.arm_qadr(m)
    for f in range(len(J) - 1):
        for k in range(sub):
            al = k / sub
            d.ctrl[arm] = (1 - al) * J[f] + al * J[f + 1]
            d.ctrl[g] = 255 * ((1 - al) * G[f] + al * G[f + 1])
            mujoco.mj_step(m, d)
            if d.ncon:
                min_dist = min(min_dist, min(d.contact[i].dist for i in range(d.ncon)))
        track_err.append(np.abs(d.qpos[qa] - J[f + 1]).max())
    for _ in range(int(round(1.0 / m.opt.timestep))):   # 1 s 静置后判定
        mujoco.mj_step(m, d)
    ok, detail = PR.evaluate(m, d, spec)
    rep = {"success": ok, "predicates": detail, "min_contact_dist_mm": round(min_dist * 1000, 2),
           "max_joint_tracking_err_deg": round(float(np.degrees(max(track_err))), 3),
           "median_joint_tracking_err_deg": round(float(np.degrees(np.median(track_err))), 3)}
    json.dump(rep, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
