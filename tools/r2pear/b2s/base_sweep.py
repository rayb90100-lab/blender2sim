"""基座摆位对比：固定几个位置，各自解 IK + MuJoCo 精确判碰 + 完整回放。

不做 117 候选搜索——碰撞检测只要 0.1 s，没必要为它做粗筛（见 GLB/PIPELINE_LOG.md §3）。

用法：
    cd /home/ubuntu/workstation/RoboLab && source setup_env.sh
    MUJOCO_GL=egl $PY tools/r2pear/b2s/base_sweep.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from common import Extract, Decisions, load_config  # noqa: E402
import parse_scene as PS  # noqa: E402
import embodiment as EM  # noqa: E402
import demo as DM  # noqa: E402
import retarget as RT  # noqa: E402
import build_spec as BS  # noqa: E402
import validate as V  # noqa: E402

BLEND = "GLB/test2/UMI_ep68.blend"
EXTRACT = "GLB/test2/sim/stage0_extract/extract.json"
CFG = "GLB/test2/b2s_config.json"
OUT = "/tmp/base_sweep"


def base_T(xy, z, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, xy[0]], [s, c, 0, xy[1]], [0, 0, 1, z], [0, 0, 0, 1]], float)


def run_one(cfg, T, label):
    """给定基座 4x4，跑完整流程，返回指标。"""
    ex = Extract(EXTRACT)
    D = Decisions()
    emb = EM.detect(ex, cfg, D)
    scene = PS.parse(ex, emb, None, cfg, D, ee_world=None)
    demo = DM.build(ex, emb, None, scene, cfg, D, os.path.dirname(BLEND))

    ik = RT.IK()
    floor = float(scene["table"].get("floor_z"))
    tb_b, fl_b = RT._table_in_base(scene["table"], floor, T)
    obs = RT._obstacles_in_base(scene, T, demo.get("moved", ()))

    Ti = np.linalg.inv(T)
    q, ep, er = RT._track(ik, [Ti @ M for M in demo["ee_world"]], ik.home)
    bad = int(((ep > RT.POS_TOL) | (er > RT.ROT_TOL)).sum())

    pens = [ik.penetration(q[t], tb_b, fl_b, obstacles=obs) for t in range(len(q))]
    n_hit = sum(1 for p in pens if p > 0.005)
    pen_max = max(pens)
    tau = ik.torque_margin(q)
    lim_margin = float(np.min(np.minimum(q - ik.lim[:, 0], ik.lim[:, 1] - q)))

    res = {
        "label": label,
        "base_xyz": [round(float(T[i, 3]), 3) for i in range(3)],
        "ik_bad": bad,
        "pos_err_mm": round(float(ep.max() * 1000), 2),
        "rot_err_deg": round(float(np.degrees(er.max())), 2),
        "collision_frames": f"{n_hit}/{len(q)}",
        "penetration_mm": round(pen_max * 1000, 1),
        "torque_frac": round(tau, 3),
        "joint_limit_margin_deg": round(math.degrees(lim_margin), 1),
    }

    # 只有 IK 干净且零碰撞才真跑物理回放（省时间）
    if bad == 0 and n_hit == 0:
        scene["T_world_base"] = T
        demo["q"] = q
        os.makedirs(OUT, exist_ok=True)
        try:
            spec = BS.build(ex, scene, demo, q[0], {}, D, OUT, BLEND, robot=None, emb=emb)
            r2, _ = V.v2_replay(spec, demo)
            res["replay_success"] = bool(r2["success"])
            res["joint_track_deg"] = round(r2["joint_track_err_deg"]["median"], 1)
            res["cup_final_mm"] = (r2["vs_blend_reference"].get("mug_hollow_glazed_ceramic") or {}).get("final_mm")
        except Exception as e:
            res["replay_error"] = f"{type(e).__name__}: {str(e)[:60]}"
    else:
        res["replay"] = "skipped（IK 或碰撞不合格）"
    return res


def main():
    cfg = load_config(CFG)
    ex = Extract(EXTRACT)
    D0 = Decisions()
    emb = EM.detect(ex, cfg, D0)
    scene = PS.parse(ex, emb, None, cfg, D0, ee_world=None)
    demo = DM.build(ex, emb, None, scene, cfg, D0, os.path.dirname(BLEND))

    top = scene["table"]["top_z"]
    floor = scene["table"]["floor_z"]
    c = demo["ee_world"][:, :2, 3].mean(0)          # 轨迹质心
    print(f"桌面 z={top:.3f}  地面 z={floor:.3f}  轨迹质心 xy={np.round(c, 3).tolist()}")

    # 固定几个位置：地面高度 × 桌前三四个方位，朝向轨迹质心
    plans = []
    for r in (0.45, 0.60):
        for deg in (0, 90, 180, 270):
            a = math.radians(deg)
            xy = c + r * np.array([math.cos(a), math.sin(a)])
            plans.append((base_T(xy, floor, math.atan2(*(c - xy)[::-1])),
                          f"地面 r={r} 方位={deg}°"))
    # 再补一个「和桌面同高」的对照，看原方案为什么失败
    xy = c + 0.45 * np.array([0.0, -1.0])
    plans.append((base_T(xy, top, math.atan2(*(c - xy)[::-1])), "桌面同高 r=0.45 方位=-90°"))

    rows = []
    for T, label in plans:
        try:
            r = run_one(cfg, T, label)
        except Exception as e:
            r = {"label": label, "error": f"{type(e).__name__}: {str(e)[:70]}"}
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)

    json.dump(rows, open("/tmp/base_sweep.json", "w"), indent=1, ensure_ascii=False)
    ok = [r for r in rows if r.get("replay_success") or r.get("ik_bad") == 0
          and str(r.get("collision_frames", "")).startswith("0/")]
    print(f"\n=== 可用（IK 0 失败 + 0 碰撞帧）{len(ok)}/{len(rows)} ===")
    for r in ok:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
