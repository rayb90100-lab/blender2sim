"""test2 回放诊断：固定基座 XY/yaw，只扫「高度 × pinch 点」，离线评估回放结果。

比整条 run.py 快得多：基座搜索只做一次（retarget 内部用 24 帧抽样），
之后每个组合只跑一次全长度 IK + 一次 MuJoCo 回放。

用法：
    cd /home/ubuntu/workstation/RoboLab && source setup_env.sh
    MUJOCO_GL=egl $PY tools/r2pear/b2s/sweep_base_pinch.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

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

PINCHES = {
    "unit":  [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "pad":   [[0, 1, 0, 0.00650], [1, 0, 0, -0.00600], [0, 0, -1, -0.06600], [0, 0, 0, 1]],
    "cup":   [[0, 1, 0, 0.00651], [1, 0, 0, -0.02798], [0, 0, -1, -0.05576], [0, 0, 0, 1]],
    "trans": [[1, 0, 0, 0.00651], [0, 1, 0, -0.02798], [0, 0, 1, -0.05576], [0, 0, 0, 1]],
}

DZ_LIST = (-0.05, 0.0, 0.05, 0.10, 0.15)


def base_xy_from_last_run():
    """沿用上一次成功跑出的基座 XY/yaw，只让高度变，避免重复做 64 候选搜索。"""
    p = "GLB/test2/sim/stage0_spec/task_spec.json"
    if os.path.exists(p):
        tb = np.array(json.load(open(p))["robot"]["T_world_base"], float)
        return tb[:2, 3], math.atan2(tb[1, 0], tb[0, 0])
    return np.array([0.86, 0.80]), math.radians(-112.5)


def run_one(cfg, pinch, dz):
    ex = Extract(EXTRACT)
    D = Decisions()
    emb = EM.detect(ex, cfg, D)
    scene = PS.parse(ex, emb, None, cfg, D, ee_world=None)
    scene["T_world_base"] = np.eye(4)

    cfg2 = dict(cfg)
    cfg2["embodiment"] = dict(cfg["embodiment"])
    cfg2["embodiment"]["root_to_pinch"] = PINCHES[pinch]
    demo = DM.build(ex, emb, None, scene, cfg2, D, os.path.dirname(BLEND))

    xy, yaw = base_xy_from_last_run()
    top = scene["table"]["top_z"]
    Tb = RT._base_T(xy, top + dz, yaw)
    res = RT.retarget(demo["ee_world"], scene, {"robot_base": Tb.tolist()}, D)
    scene["T_world_base"] = res["T_world_base"]
    demo["q"] = res["q"]

    out = "/tmp/sweep_sim"
    os.makedirs(out, exist_ok=True)
    spec = BS.build(ex, scene, demo, res["q"][0], {}, D, out, BLEND, robot=None, emb=emb)
    r2, _ = V.v2_replay(spec, demo)
    tilt = [p.get("tilt_deg") for p in r2["predicates"] if p["pred"] == "upright"]
    return {
        "pinch": pinch, "dz": dz,
        "base_z": round(float(top + dz), 4),
        "ik_bad": int((res["pos_err"] > RT.POS_TOL).sum() + (res["rot_err"] > RT.ROT_TOL).sum()),
        "success": r2["success"],
        "tilt_deg": tilt[0] if tilt else None,
        "final_mm": (r2["vs_blend_reference"].get("mug_hollow_glazed_ceramic") or {}).get("final_mm"),
        "joint_med_deg": round(r2["joint_track_err_deg"]["median"], 1),
    }


def main():
    cfg = load_config(CFG)
    rows = []
    for pinch in PINCHES:
        for dz in DZ_LIST:
            try:
                r = run_one(cfg, pinch, dz)
            except Exception as e:
                r = {"pinch": pinch, "dz": dz, "error": f"{type(e).__name__}: {str(e)[:90]}"}
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(rows, open("/tmp/sweep_result.json", "w"), indent=1, ensure_ascii=False)
    ok = [r for r in rows if r.get("success")]
    print(f"\n=== 成功 {len(ok)}/{len(rows)} ===")
    for r in ok:
        print(json.dumps(r, ensure_ascii=False))
    if not ok:
        print("\n最接近的（倾角最小）：")
        cand = [r for r in rows if isinstance(r.get("tilt_deg"), (int, float))]
        for r in sorted(cand, key=lambda r: r["tilt_deg"])[:5]:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
