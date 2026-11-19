#!/usr/bin/env python3
"""接触参数扫描：每组 (timestep, 物体 solref) 测 示教回放成功 / oracle 成功 / 协议压测下挤压穿透 / 单控制步耗时。
地面（world）接触单列：物体掉落桌外时的瞬时穿透不影响任务判定。

用法: python tools/r2pear/sweep_contact.py --spec <spec> --episode <parquet> --dt 0.002 --solref 0.01,1 --out <json>
（由外部按组并行调用；参数经环境变量 R2P_TIMESTEP / R2P_SOLREF 传给 sim.build）
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--solref", required=True)
    ap.add_argument("--solimp", default="")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["R2P_TIMESTEP"], os.environ["R2P_SOLREF"] = a.dt, a.solref
    if a.solimp:
        os.environ["R2P_SOLIMP"] = a.solimp
    sys.path[:0] = [HERE, os.path.dirname(HERE)]
    import collections
    import mujoco
    import controller as CT
    import env as E
    import oracle as O
    rep = {"dt": float(a.dt), "solref": a.solref, "solimp": a.solimp or "default(0.9,0.95,0.001,0.5,2)"}

    tmp = a.out + ".replay.json"
    subprocess.run([sys.executable, os.path.join(HERE, "check_demo_replay.py"), "--spec", a.spec, "--episode",
                    a.episode, "--out", tmp], env=os.environ, check=True, stdout=subprocess.DEVNULL)
    r = json.load(open(tmp))
    rep["demo_replay"] = {"success": r["success"], "min_contact_mm": r["min_contact_dist_mm"]}

    e = E.R2PEnv(a.spec, render=False); e.open()
    res = O.solve(e, "green_block", "stack_cube")
    rep["oracle"] = {"success": bool(res["done"] and res["outcome"] == "done"), "control_steps": e.used_steps}

    rng = np.random.default_rng(0)
    squeeze, floor_pen, steps = [], [], 0
    t0 = time.time()
    for ep in range(a.episodes):
        env = E.R2PEnv(a.spec, render=False); env.open()
        m, d = env.m, env.d

        def watch(fmax):
            s_min = f_min = 0.0
            for i in range(d.ncon):
                c = d.contact[i]
                n1, n2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
                if "world" in (n1, n2):
                    f_min = min(f_min, c.dist)
                else:
                    s_min = min(s_min, c.dist)
            squeeze.append(s_min * 1000); floor_pen.append(f_min * 1000)
        for k in range(25):
            act = {"arm": {q: float(rng.normal(0, 8.0)) for q in ("x", "y", "z")}
                   | {q: float(rng.normal(0, 0.4)) for q in ("roll", "pitch", "yaw")},
                   "gripper": {"width": float(rng.choice([-1.0, 0.0, 1.0]))}}
            oc, n, _ = CT.PoseController(env.robot, act).move(env.budget - env.used_steps, on_control_step=watch)
            env.used_steps += n; steps += n
            if not np.isfinite(d.qpos).all():
                rep["nan"] = True
                break
            if env.used_steps >= env.budget:
                break
    sq, fl = np.array(squeeze), np.array(floor_pen)
    rep["protocol_stress"] = {
        "control_steps": steps, "ms_per_control_step": round((time.time() - t0) / steps * 1000, 2),
        "squeeze_pen_mm": {"min": round(float(sq.min()), 2), "p01": round(float(np.percentile(sq, 0.1)), 2),
                           "p1": round(float(np.percentile(sq, 1)), 2), "frac_below_3mm": round(float((sq < -3).mean()), 5)},
        "floor_pen_mm_min": round(float(fl.min()), 2)}
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
