#!/usr/bin/env python3
"""初始状态采样 + 逐个验收。

分布配置（init_distribution.json，由源 test 的单项扫描导出）:
  {"groups": {"<g>": {"objects": [...], "pivot": "<obj>"}}, "params": {"<g>.dx": [lo, hi], "<g>.dy":..., "<g>.dyaw_deg":...,
   "table_dz": [lo, hi]}, "seed": 0, "target_n": 50, "max_attempts": 300}
每个参数独立均匀采样（全部参数同时扰动）。

验收（全部通过才收录）：
  A 静置 0.5 s 后：所有物体中心在桌面内（距桌沿 ≥ 半个最大边长）、所有物体静止、初始接触无 >2 mm 穿透
  B 谓词中「与任务无关、初始即应成立」的部分成立（支撑结构 on/upright），且任务谓词本身未成立（不能开局即成功）
  C 特权 oracle 经 PEAR 协议动作可解（可解性证明）
输出 init_states.json：{"states": [{"id", "params", "overrides", "oracle_control_steps"}], "rejections": {...}}

用法: python tools/r2pear/init_states.py --spec <spec> --dist <init_distribution.json> --out <init_states.json>
"""
import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import env as E  # noqa: E402
import oracle as O  # noqa: E402
import predicates as PR  # noqa: E402
import spec as SP  # noqa: E402


def rotz(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def make_overrides(spec, dist, params):
    ov = {}
    for g, cfg in dist["groups"].items():
        dx, dy = params.get(f"{g}.dx", 0.0), params.get(f"{g}.dy", 0.0)
        dyaw = math.radians(params.get(f"{g}.dyaw_deg", 0.0))
        piv = np.array(spec["objects"][cfg["pivot"]]["center"][:2])
        for n in cfg["objects"]:
            o = spec["objects"][n]
            c = np.array(o["center"])
            xy = piv + rotz(dyaw) @ (c[:2] - piv) + [dx, dy]
            ov[n] = {"center": [float(xy[0]), float(xy[1]), float(c[2])], "yaw_rad": float(o["yaw_rad"] + dyaw)}
    if "table_dz" in params:
        ov["table_dz"] = params["table_dz"]
    return ov


def on_table(env):
    tb = env.spec["table"]
    c, yaw = np.array(tb["center_xy"]), tb["yaw_rad"]
    hx, hy = np.array(tb["size_xy"]) / 2
    for n, o in env.spec["objects"].items():
        p = env.d.xpos[env.m.body(n).id][:2]
        loc = rotz(-yaw) @ (p - c)
        margin = SP.footprint_half(o)
        if abs(loc[0]) > hx - margin or abs(loc[1]) > hy - margin:
            return False, n
    return True, None


def check(spec_path, spec, dist, params, task_obj, task_support):
    env = E.R2PEnv(spec_path, render=False)
    env.overrides = make_overrides(spec, dist, params)
    env.open()
    m, d = env.m, env.d
    ok, who = on_table(env)
    if not ok:
        return "A_off_table", None
    if not all(PR.at_rest(m, d, n) for n in spec["objects"]):
        return "A_not_at_rest", None
    if d.ncon and min(d.contact[i].dist for i in range(d.ncon)) < -0.002:
        return "A_initial_penetration", None
    succ, det = PR.evaluate(m, d, spec)
    if succ:
        return "B_already_solved", None
    structure = [p for p in det if p.get("obj") != task_obj and p["pred"] != "all_at_rest"]
    if not all(p["ok"] for p in structure):
        return "B_structure_broken", None
    O.solve(env, task_obj, task_support)
    succ, _ = PR.evaluate(m, d, spec)
    if not succ:
        return "C_oracle_failed", None
    return "ok", env.used_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--dist", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    dist = json.load(open(a.dist))
    rng = np.random.default_rng(dist.get("seed", 0))
    t_obj, t_sup = dist["task"]["obj"], dist["task"]["support"]
    states = [{"id": 0, "params": {}, "overrides": {}, "note": "源场景（重建原值）"}]
    st, steps = check(a.spec, spec, dist, {}, t_obj, t_sup)
    assert st == "ok", f"源场景未通过验收: {st}"
    states[0]["oracle_control_steps"] = steps
    rej = collections.Counter()
    attempts, t0 = 0, time.time()
    while len(states) < dist["target_n"] and attempts < dist["max_attempts"]:
        attempts += 1
        params = {k: float(rng.uniform(*v)) for k, v in dist["params"].items()}
        st, steps = check(a.spec, spec, dist, params, t_obj, t_sup)
        if st != "ok":
            rej[st] += 1
            continue
        states.append({"id": len(states), "params": {k: round(v, 4) for k, v in params.items()},
                       "overrides": make_overrides(spec, dist, params), "oracle_control_steps": steps})
    out = {"spec": os.path.relpath(a.spec, os.path.dirname(os.path.abspath(a.out))), "distribution": dist,
           "attempts": attempts, "accepted": len(states) - 1, "acceptance_rate": round((len(states) - 1) / max(attempts, 1), 3),
           "rejections": dict(rej), "wall_s": round(time.time() - t0, 1), "states": states}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k not in ("states", "distribution")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
