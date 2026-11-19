#!/usr/bin/env python3
"""b2s 第 5 步：param_sweep.json（sweep_params.py 的单项实测边界）→ init_distribution.json。

规则（通用，不看物体名）：
  每个参数取「包含 0 的连续通过区间」，向内收 10% 作为采样范围；
  区间退化成单点（扫描步长内就失败，如圆柱立在同宽方块上，§63.11）→ 取 ±0.4×扫描步长，标为兜底，
  靠 init_states.py 的逐样本验收（A 静置 / B 开局未成功 / C oracle 可解）剔除不可解样本。

用法: python tools/r2pear/b2s/init_dist.py --simdir <D> [--n 50]
"""
import argparse
import json
import os

SHRINK = 0.9
DEGEN_FRAC = 0.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simdir", required=True)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    d = os.path.join(a.simdir, "stage3_init_states")
    sw = json.load(open(os.path.join(d, "param_sweep.json")))
    spec = json.load(open(os.path.join(a.simdir, "stage0_spec", "task_spec.json")))
    obj, sup = sw["task"]["obj"], sw["task"]["support"]
    groups = {"target": {"objects": [obj], "pivot": obj}}
    below = [n for n, o in spec["objects"].items() if o.get("role") == "stack"]
    if below:
        groups["tower"] = {"objects": below, "pivot": below[-1]}
    params, notes = {}, {}
    for k, v in sw["params"].items():
        if k.split(".")[0] not in groups and k != "table_dz":
            continue
        lo, hi = v["ok_range"]
        vals = sorted(v["all_ok_values"] + [float(x) for x in v.get("failures", {}).values()])
        step = min((b - a_ for a_, b in zip(vals, vals[1:]) if b > a_), default=0.0)
        if hi - lo < 1e-9:
            r = DEGEN_FRAC * step
            params[k] = [-r, r]
            notes[k] = f"扫描区间退化为单点（步长 {step:g} 内即失败，失败原因 {v.get('failures')}）→ ±{r:g}，靠逐样本验收剔除"
        else:
            c = (lo + hi) / 2
            h = (hi - lo) / 2 * SHRINK
            params[k] = [c - h, c + h]
            notes[k] = f"实测通过区间 [{lo:g}, {hi:g}] 向内收 10%"
    dist = {"source": "b2s/init_dist.py ← stage3_init_states/param_sweep.json（sweep_params.py 单项实测）",
            "task": {"obj": obj, "support": sup}, "groups": groups, "params": params,
            "seed": 0, "target_n": a.n, "max_attempts": 12 * a.n, "range_note": notes}
    json.dump(dist, open(os.path.join(d, "init_distribution.json"), "w"), indent=1, ensure_ascii=False)
    for k in params:
        print(f"  {k:18s} {params[k]}   {notes[k]}")


if __name__ == "__main__":
    main()
