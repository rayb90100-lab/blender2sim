#!/usr/bin/env python3
"""单项扰动扫描（通用）：逐参数沿一个轴扫，配合 oracle 可解性，给出「能通过验收的范围」。

用途：为 init_states 的采样分布定范围，而不是拍脑袋取 ±5 cm（test4 是 §55 手扫的）。

每个参数单独扫（其余保持 0），对每个取值检查：
  A 在桌内 / 静止 / 无初始穿透
  B 任务谓词未成立（不能开局即成功）
  C oracle 经 PEAR 协议动作可解
记录每段的失败原因，输出通过的连续区间。

用法: python tools/r2pear/sweep_params.py --testdir <T> --obj <target> --support <support> \
          [--n 9] [--out <json>]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import env as E  # noqa: E402
import init_states as IS  # noqa: E402
import oracle as O  # noqa: E402
import predicates as PR  # noqa: E402


def axis_values(param, n):
    """每个参数在自己的量纲上扫。角度单位是度，长度单位是米。"""
    span = {
        "target.dx": (-0.10, 0.10), "target.dy": (-0.10, 0.10),
        "tower.dx": (-0.08, 0.08), "tower.dy": (-0.08, 0.08),
        "table_dz": (-0.05, 0.06),
    }
    if param.endswith("dyaw_deg"):
        span[param] = (-60, 60)
    if param not in span:
        raise SystemExit(f"未知参数 {param}，请在 axis_values 里加范围")
    lo, hi = span[param]
    return list(np.linspace(lo, hi, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    T = a.testdir
    spec_path = os.path.join(T, "stage0_spec", "task_spec.json")
    spec = json.load(open(spec_path))

    # 分组：target 单独；decoration（叠放链上 target 下方的物体，会一起移动）
    below = [n for n in spec["objects"]
             if spec["objects"][n].get("role") == "stack"]
    groups = {"target": {"objects": [a.obj], "pivot": a.obj}}
    if below:
        groups["tower"] = {"objects": below, "pivot": below[-1]}
    base = {"task": {"obj": a.obj, "support": a.support}, "groups": groups}

    params = [f"{g}.{k}" for g in groups for k in ("dx", "dy", "dyaw_deg")] + ["table_dz"]
    res = {}
    print(f"{'param':18s} {'value':>8s}  {'结果':20s} oracle_steps")
    for p in params:
        rows = []
        for v in axis_values(p, a.n):
            dist = dict(base)
            st, steps = IS.check(spec_path, spec, dist, {p: float(v)}, a.obj, a.support)
            rows.append({"value": round(float(v), 4), "status": st, "oracle_steps": steps})
            print(f"{p:18s} {v:+8.3f}  {st:20s} {steps if steps else '-'}")
        okv = [r["value"] for r in rows if r["status"] == "ok"]
        if okv:
            # 取包含 0 的**连续**区间：从 0 出发向两侧走，遇到第一个失败值就停。
            # （早期版本用 max(v<=0)/min(v>=0)，因为 0 同时满足两个条件，永远得到 [0,0]。）
            vals = sorted(r["value"] for r in rows)          # 全轴按值排序
            stat = {r["value"]: r["status"] for r in rows}
            i0 = min(range(len(vals)), key=lambda i: abs(vals[i]))
            lo = hi = vals[i0]
            for i in range(i0 - 1, -1, -1):
                if stat[vals[i]] != "ok":
                    break
                lo = vals[i]
            for i in range(i0 + 1, len(vals)):
                if stat[vals[i]] != "ok":
                    break
                hi = vals[i]
            res[p] = {"ok_range": [lo, hi], "all_ok_values": okv,
                      "failures": {r["status"]: r["value"] for r in rows if r["status"] != "ok"}}
        else:
            res[p] = {"ok_range": None, "all_ok_values": [],
                      "failures": {r["status"]: r["value"] for r in rows}}
        print(f"  → {p} 通过区间 [{res[p]['ok_range']}]"
              f"   失败: {res[p]['failures']}\n")

    out = a.out or os.path.join(T, "stage3_init_states", "param_sweep.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"task": {"obj": a.obj, "support": a.support}, "n_per_axis": a.n, "params": res},
              open(out, "w"), indent=1, ensure_ascii=False)
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
