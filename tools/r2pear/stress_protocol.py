#!/usr/bin/env python3
"""协议级压测：随机 PEAR 动作（位姿增量 + 夹爪）经控制器执行 —— 策略实际能产生的动作分布。
每个 episode 最多 25 轮（PEAR 真机评测上限）或用完控制步预算。

用法: python tools/r2pear/stress_protocol.py --spec <task_spec.json> --out <json> [--episodes 40]
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
import env as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--pos_cm", type=float, default=8.0)
    ap.add_argument("--rot_rad", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import mujoco
    rng = np.random.default_rng(a.seed)
    outcomes, reasons = collections.Counter(), collections.Counter()
    pen, speed, rounds, ctrl_steps, succ, off = [], [], 0, 0, 0, 0
    t0 = time.time()
    for e in range(a.episodes):
        env = E.R2PEnv(a.spec, render=False)
        env.open()
        m, d = env.m, env.d
        objs = list(env.spec["objects"])
        orig_step = mujoco.mj_step

        def watch(fmax):
            pen.append(min([d.contact[i].dist for i in range(d.ncon)] or [0.0]) * 1000)
            for n in objs:
                j = m.joint(f"{n}_free")
                speed.append(float(np.linalg.norm(d.qvel[j.dofadr[0]:j.dofadr[0] + 3])))

        for k in range(a.rounds):
            act = {"arm": {q: float(rng.normal(0, a.pos_cm)) for q in ("x", "y", "z")}
                   | {q: float(rng.normal(0, a.rot_rad)) for q in ("roll", "pitch", "yaw")},
                   "gripper": {"width": float(rng.choice([-1.0, 0.0, 1.0]))}}
            import controller as CT
            ctl = CT.PoseController(env.robot, act)
            oc, n, reason = ctl.move(env.budget - env.used_steps, on_control_step=watch)
            env.used_steps += n
            outcomes[oc] += 1
            reasons[reason] += 1
            rounds += 1
            ctrl_steps += n
            if not np.isfinite(d.qpos).all():
                outcomes["NAN"] += 1
                break
            if env.used_steps >= env.budget:
                break
        import predicates as PR
        ok, _ = PR.evaluate(m, d, env.spec)
        succ += ok
        off += sum(d.xpos[m.body(n).id][2] < env.spec["table"]["top_z"] - 0.05 for n in objs)
        env.close()
    ph, sp = np.array(pen), np.array(speed)
    rep = {"params": vars(a), "episodes": a.episodes, "rounds": rounds, "control_steps": ctrl_steps,
           "wall_s": round(time.time() - t0, 1),
           "wall_ms_per_control_step": round((time.time() - t0) / max(ctrl_steps, 1) * 1000, 2),
           "outcomes": dict(outcomes), "stop_reasons": {str(k): v for k, v in reasons.items()},
           "penetration_mm": {"min": round(float(ph.min()), 2), "p1": round(float(np.percentile(ph, 1)), 2),
                              "frac_below_3mm": round(float((ph < -3).mean()), 5)},
           "object_speed_mps": {"max": round(float(sp.max()), 3), "p999": round(float(np.percentile(sp, 99.9)), 3)},
           "objects_off_table_at_end": int(off), "random_successes": int(succ)}
    json.dump(rep, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
