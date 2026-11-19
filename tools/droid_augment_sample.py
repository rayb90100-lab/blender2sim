#!/usr/bin/env python3
"""Stage 3 随机组合采样 + 同组合缩幅重试（skill augmentation.md §3）。

每个候选随机选 1–3 项参数，在单项测试得到的范围内均匀采样；失败时保持所选参数，
按 p ← α·p 缩幅（α=0.85，基准为 0 即原场景），最多 5 次。达到目标 N 或预算耗尽即停。
精确去重：参数四舍五入（位置 1 mm、角度 1°）后的签名相同计为重复。
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import droid_augment as A

RANGES = {  # 来自单项测试（DEVELOPMENT_LOG §55）
    "green_dx": (-0.08, 0.06), "green_dy": (-0.08, 0.08), "green_dyaw": (-45, 45),
    "tower_dx": (-0.06, 0.06), "tower_dy": (-0.06, 0.06), "tower_dyaw": (-45, 45),
    "table_dz": (-0.04, 0.06)}
CFG = {"seed": 0, "target_N": 3, "max_candidates": 8, "alpha": 0.85, "max_shrink": 5, "items_per_candidate": [1, 3]}


def sig(p):
    return json.dumps({k: round(v * 1000) if not k.endswith("yaw") else round(v) for k, v in sorted(p.items())})


def main(T):
    rng = np.random.default_rng(CFG["seed"])
    out = os.path.join(T, "stage3_augment")
    log, seen, passed = [], set(), []
    keys = list(RANGES)
    for ci in range(CFG["max_candidates"]):
        if len(passed) >= CFG["target_N"]:
            break
        k = int(rng.integers(CFG["items_per_candidate"][0], CFG["items_per_candidate"][1] + 1))
        chosen = list(rng.choice(keys, k, replace=False))
        p0 = {c: float(rng.uniform(*RANGES[c])) for c in chosen}
        rec = {"candidate_index": ci, "sampled": p0, "attempts": []}
        p = dict(p0)
        for s in range(CFG["max_shrink"] + 1):
            if sig(p) in seen:
                rec["attempts"].append({"shrink": s, "params": p, "duplicate": True})
                break
            seen.add(sig(p))
            cid = f"mech_c{ci}_s{s}"
            r = A.run_candidate(T, cid, p)
            rec["attempts"].append({"shrink": s, "alpha_total": CFG["alpha"] ** s, "params": p, "candidate": cid,
                                    "pass": r["pass"], "failed_criteria": r["failed_criteria"],
                                    "ik_max_mm": r["ik_pos_err_mm"]["max"]})
            print(f"ROW c{ci} s{s} {'PASS' if r['pass'] else 'FAIL'} {json.dumps({kk: round(v, 3) for kk, v in p.items()})} "
                  f"ik_max {r['ik_pos_err_mm']['max']:.2f}mm fails {r['failed_criteria']}", flush=True)
            if r["pass"]:
                passed.append(cid)
                rec["final"] = cid
                break
            p = {kk: v * CFG["alpha"] for kk, v in p.items()}
        log.append(rec)
    summary = {"config": CFG, "ranges": RANGES, "candidates": log, "passed": passed,
               "stats": {"sampled": len(log), "passed": len(passed),
                         "attempts": sum(len(r["attempts"]) for r in log),
                         "items_hist": {str(n): sum(1 for r in log if len(r["sampled"]) == n) for n in (1, 2, 3)}}}
    json.dump(summary, open(os.path.join(out, "mech_sampling.json"), "w"), indent=2, ensure_ascii=False)
    print("SAMPLE_DONE", json.dumps(summary["stats"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
