#!/usr/bin/env python3
"""汇总 IK 噪声扫描：每档噪声的末端误差、关节误差、物理任务成功率。

用法: python tools/droid_ik_sweep_summary.py --testdir <testdir>
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    a = ap.parse_args()
    base = os.path.join(a.testdir, "stage2_physics")
    rows = []
    for d in sorted(glob.glob(os.path.join(base, "ik_*"))):
        tag = os.path.basename(d)
        ik_p, ph_p = os.path.join(d, "ik_report.json"), os.path.join(d, "physics_report.json")
        if not (os.path.exists(ik_p) and os.path.exists(ph_p)):
            continue
        ik, ph = json.load(open(ik_p)), json.load(open(ph_p))
        m = re.match(r"ik_n([\d.]+)mm_([\d.]+)deg_s(\d+)", tag)
        lv = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
        rows.append({
            "tag": tag, "pos_mm": lv[0], "rot_deg": lv[1],
            "ee_pos_med": ik["ee_pos_err_mm_vs_true"]["median"],
            "ee_pos_max": ik["ee_pos_err_mm_vs_true"]["max"],
            "ee_rot_med": ik["ee_rot_err_deg_vs_true"]["median"],
            "joint_med": ik["joint_err_deg_vs_gt"]["overall_median"],
            "lift": ph["green_max_lift_mm"],
            "end_z": ph["green_end_z_mm"],
            "xy": float(np.linalg.norm(ph["green_end_xy_vs_stack_mm"])),
            "grip": ph["grip_contact_interval"],
            "pen": ph["min_contact_dist_mm"],
            "ok": ph["task_success"],
        })
    rows.sort(key=lambda r: (r["pos_mm"], r["tag"]))
    print(f"{'tag':26s} {'末端误差中位/最大(mm)':>18s} {'姿态(°)':>7s} {'关节(°)':>7s} "
          f"{'抬升':>6s} {'终z':>6s} {'xy偏':>6s} {'夹持区间':>11s} {'穿透':>6s} 成功")
    for r in rows:
        print(f"{r['tag']:26s} {r['ee_pos_med']:8.2f}/{r['ee_pos_max']:7.2f}      {r['ee_rot_med']:6.2f} "
              f"{r['joint_med']:7.2f} {r['lift']:6.0f} {r['end_z']:6.1f} {r['xy']:6.1f} "
              f"{str(r['grip']):>11s} {r['pen']:6.2f}  {'✓' if r['ok'] else '✗'}")
    print("\n按噪声档汇总：")
    levels = sorted({(r["pos_mm"], r["rot_deg"]) for r in rows})
    summ = []
    for lv in levels:
        g = [r for r in rows if (r["pos_mm"], r["rot_deg"]) == lv]
        sr = np.mean([r["ok"] for r in g])
        summ.append({"pos_mm": lv[0], "rot_deg": lv[1], "n": len(g), "success_rate": float(sr),
                     "ee_err_med_mm": float(np.median([r["ee_pos_med"] for r in g]))})
        print(f"  噪声 {lv[0]:5.1f} mm / {lv[1]:4.1f}°   n={len(g)}  成功率 {sr*100:5.1f}%  "
              f"末端误差中位 {summ[-1]['ee_err_med_mm']:.2f} mm")
    json.dump({"runs": rows, "levels": summ}, open(os.path.join(base, "ik_sweep_summary.json"), "w"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
