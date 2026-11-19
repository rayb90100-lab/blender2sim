#!/usr/bin/env python3
"""非示教动作压测：随机关节目标（比 PEAR 位姿控制器更激进，受阻也不停）。
与 §61.2 test4 压测同参数（20 ep × 60 次 × 0.5 s，σ=0.35 rad，seed 0），用于修改前后对比。

用法: python tools/r2pear/stress.py --spec <task_spec.json> --out <json> [--episodes 20]
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import droid_common as C  # noqa: E402
import sim as S  # noqa: E402
import spec as SP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import mujoco
    spec = SP.load(a.spec)
    m, d = S.build(spec)
    qa, arm, g = C.arm_qadr(m), S.arm_actuators(m), S.grip_actuator(m)
    lim = np.array([m.jnt_range[m.joint(j).id] for j in C.PANDA_JOINTS])
    rng = np.random.default_rng(a.seed)
    bname = lambda gid: m.body(m.geom_bodyid[gid]).name.replace(S.GRIP_PREFIX, "")
    objs = list(spec["objects"])
    floor_z = spec["table"]["top_z"] - spec["table"]["height_m"]
    pen_hist, worst, pairs = [], {}, collections.Counter()
    max_speed, nan, off_table, fast_steps, steps = 0.0, 0, 0, 0, 0
    t0 = time.time()
    for e in range(a.episodes):
        S.reset(m, d, spec)
        q = np.array(spec["robot"]["init_qpos"])
        for k in range(a.rounds):
            q = np.clip(q + rng.normal(0, a.sigma, 7), lim[:, 0], lim[:, 1])
            d.ctrl[arm] = q
            d.ctrl[g] = rng.choice([0, 255])
            for _ in range(250):
                mujoco.mj_step(m, d)
                steps += 1
                if not np.isfinite(d.qpos).all():
                    nan += 1
                    break
                dmin = 0.0
                for i in range(d.ncon):
                    c = d.contact[i]
                    dmin = min(dmin, c.dist)
                    if c.dist < -0.005:
                        key = "–".join(sorted((bname(c.geom1), bname(c.geom2))))
                        pairs[key] += 1
                        worst[key] = min(worst.get(key, 0.0), c.dist * 1000)
                pen_hist.append(dmin * 1000)
                for n in objs:
                    j = m.joint(f"{n}_free")
                    v = float(np.linalg.norm(d.qvel[j.dofadr[0]:j.dofadr[0] + 3]))
                    max_speed = max(max_speed, v)
                    fast_steps += v > 3.0
            if nan:
                break
        for n in objs:
            if d.xpos[m.body(n).id][2] < spec["table"]["top_z"] - 0.05:
                off_table += 1
    ph = np.array(pen_hist)
    rep = {"params": vars(a), "steps": steps, "wall_s": round(time.time() - t0, 1), "nan_episodes": nan,
           "penetration_mm": {"min": round(float(ph.min()), 2), "p01": round(float(np.percentile(ph, 1)), 2),
                              "p001": round(float(np.percentile(ph, 0.1)), 2),
                              "frac_steps_below_5mm": round(float((ph < -5).mean()), 5)},
           "object_max_speed_mps": round(max_speed, 2), "object_steps_over_3mps": int(fast_steps),
           "objects_off_table_at_episode_end": off_table, "floor_z": floor_z,
           "pairs_over_5mm": {k: {"steps": v, "worst_mm": round(worst[k], 1)} for k, v in pairs.most_common(15)}}
    json.dump(rep, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
