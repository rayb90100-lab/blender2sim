#!/usr/bin/env python3
"""Stage 3 增强验收组装：为一个候选生成 skill check_augmentation_gate.py 要求的全部实际证据。

步骤（全部由程序实际执行产生，不手填 pass）：
  1. config.json        {"candidate_id", "parameters": {"mechanical": 7 项, "appearance": …, "environment": …}}
  2. native/            物理程序运行时产出的原生证据，inputs.scenario = config.json
  3. render/            关键帧渲染（Blender + MuJoCo 单图 + render_manifest，含逐帧 qpos）
  4. media/receipts     droid_media_validator.py 实际解码与帧身份核对
  5. scene_audit.json   支撑 / 安装 / 显示—碰撞一致性（打开 Blender 工程，逐块比对世界包围盒）
  6. augmentation_gate.json → 运行 check_augmentation_gate.py

用法:
  python tools/droid_augment_gate.py --testdir T --cid ID [--parent PID] --kind mechanical|appearance|base
                                     --mech '{"tower_dyaw":-44.8}' [--appearance '{...}']
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SKILL_SCRIPTS = os.path.join(HERE, "..", "Real2Gym", "skills", "real2sim-prompt", "scripts")
sys.path.insert(0, SKILL_SCRIPTS)
import droid_augment as A  # noqa: E402
from evidence_checks import Evidence, digest  # noqa: E402

PY = sys.executable
MECH_KEYS = A.PARAM_KEYS
APPEAR_DEFAULT = {"table_material": "wood_medium", "light_tint": [1.0, 1.0, 1.0], "light_azimuth_deg": 0.0}
ENV_DEFAULT = {"background_rgb": [0.78, 0.80, 0.82]}
REQ_VIEWS = ("exterior_1_left", "exterior_2_left")
REQ_FRAMES = (88, 142)
EXPOSURE = "0.03"           # 原场景标定值（§46），外观增强固定曝光


def log(m): print(m, flush=True)


def gdir(T, cid):
    return os.path.join(T, "stage3_augment", "gates", cid)


def flatten(x, prefix=""):
    if isinstance(x, dict):
        r = {}
        for k, v in x.items():
            r.update(flatten(v, prefix + "/" + k))
        return r
    return {prefix: x}


def rebase(block, src_root, dst_root):
    """把以 src_root 为基准的证据引用改写成以 dst_root 为基准（哈希不变）。"""
    def fix(r):
        return {"path": os.path.relpath(os.path.join(src_root, r["path"]), dst_root), "sha256": r["sha256"]}
    out = {"inputs": {k: ([fix(x) for x in v] if k == "assets" else fix(v)) for k, v in block["inputs"].items()}}
    for k in ("run", "trajectory", "result"):
        out[k] = fix(block[k])
    return out


def build_candidate(T, cid, mech, appearance, env):
    import droid_stage2_physics as P
    G = gdir(T, cid)
    os.makedirs(G, exist_ok=True)
    mech_full = {k: float(mech.get(k, 0.0)) for k in MECH_KEYS}
    cfg = {"candidate_id": cid, "parameters": {"mechanical": mech_full, "appearance": appearance, "environment": env}}
    cfg_p = os.path.join(G, "config.json")
    json.dump(cfg, open(cfg_p, "w"), indent=2, ensure_ascii=False)

    # ---- 场景与动作 ----
    base = json.load(open(os.path.join(T, "stage1_scene", "scene.json")))
    nonzero = {k: v for k, v in mech_full.items() if abs(v) > 0}
    scene_p = os.path.join(G, "scene.json")
    json.dump(A.make_scene(base, nonzero), open(scene_p, "w"), indent=2, ensure_ascii=False)
    qfile = None
    if nonzero:
        ep = A.C.load_episode(os.path.join(T, "input", "episode.parquet"))
        qfile = os.path.join(G, "q_adapted.npy")
        if not os.path.exists(qfile):
            Q, pe, _ = A.ik_track(ep, A.adapt_ee(ep, base, nonzero))
            np.save(qfile, Q)
            json.dump({"ik_pos_err_mm_max": float(pe.max())}, open(os.path.join(G, "ik.json"), "w"))

    # ---- 原生物理证据（scenario = config.json）----
    os.environ["DROID_SCENE_JSON"] = scene_p
    os.environ["DROID_SCENARIO_FILE"] = cfg_p
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rep = P.run(T, qfile=qfile, tag=os.path.relpath(os.path.join(G, "physics"), os.path.join(T, "stage2_physics")),
                        native_out=os.path.join(G, "native"), gate_root=G, candidate_id=cid)
    finally:
        os.environ.pop("DROID_SCENE_JSON", None)
        os.environ.pop("DROID_SCENARIO_FILE", None)
    native = json.load(open(os.path.join(G, "native", "native_evidence_block.json")))
    result = json.load(open(os.path.join(G, "native", "result.json")))
    log(f"  [{cid}] 物理 {result['status']}  task_success={rep['task_success']}  "
        f"塔倾角 {rep['tower_max_tilt_deg']:.2f}°  穿透 {rep['min_contact_dist_mm']:.2f} mm")
    return cfg_p, native, result, scene_p, qfile


def render_and_validate(T, cid, scene_p, qfile, appearance, env, native):
    G = gdir(T, cid)
    ap = dict(appearance, **env)
    ap_p = os.path.join(G, "appearance.json")
    json.dump(ap, open(ap_p, "w"), indent=2)
    envs = dict(os.environ, DROID_SCENE_JSON=scene_p, DROID_APPEARANCE=ap_p, DROID_EXPOSURE=EXPOSURE,
                DROID_STAGE8_OUT=os.path.relpath(os.path.join(G, "render"), T),
                DROID_ROLLOUT_TAG=os.path.relpath(os.path.join(G, "render_rollout"), os.path.join(T, "stage2_physics")),
                MUJOCO_GL="egl")
    if qfile:
        envs["DROID_QFILE"] = qfile
    subprocess.run([PY, os.path.join(HERE, "droid_stage8_blender.py"), "--testdir", T, "--samples", "32",
                    "--only"] + [str(f) for f in REQ_FRAMES], env=envs, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    R = os.path.join(G, "render")
    rm_p = os.path.join(R, "render_manifest.json")
    rm = json.load(open(rm_p))

    def ref(p):
        return {"path": os.path.relpath(p, G), "sha256": digest(p)}

    display = {"scene": ref(os.path.join(R, "scene_stage8.blend")),
               "render_config": ref(os.path.join(R, "stage8_manifest.json")),
               "producer": ref(os.path.join(HERE, "droid_stage8_blender.py")),
               "assets": native["inputs"]["assets"]}
    disp_key = Evidence(G).display(display)
    traj_p = os.path.join(G, native["trajectory"]["path"])
    os.makedirs(os.path.join(G, "cameras"), exist_ok=True)
    os.makedirs(os.path.join(G, "receipts"), exist_ok=True)
    sys.path.insert(0, HERE)
    import droid_media_validator as V
    media, required = [], []
    for v in REQ_VIEWS:
        cam_p = os.path.join(G, "cameras", f"{v}.json")
        json.dump(rm["cameras"][v], open(cam_p, "w"), indent=2, ensure_ascii=False)
        for f in REQ_FRAMES:
            for r in ("blender", "mujoco"):
                img = os.path.join(R, "media", f"{r}_{v}_f{f:03d}.png")
                rc = os.path.join(G, "receipts", f"{r}_{v}_f{f:03d}.json")
                rec = V.validate(img, cam_p, traj_p, rm_p, v, f, r, cid, disp_key, rc, gate_root=G)
                if not (rec["decoded"] and rec["frame_identity_verified"]):
                    log(f"  !! 媒体校验未通过 {r} {v} f{f}: {rec['checks']}")
                required.append({"view": v, "frame": f, "renderer": r})
                media.append({"view": v, "frame": f, "renderer": r, "file": ref(img), "camera": ref(cam_p),
                              "validation": ref(rc)})
    return display, disp_key, media, required


def scene_audit(T, cid, scene_p, qfile, native, display, disp_key):
    """支撑 / 安装 / 显示—碰撞一致性：实际计算，写证据文件。"""
    import mujoco
    import droid_stage2_physics as P
    import droid_common as C
    G = gdir(T, cid)
    ep = C.load_episode(os.path.join(T, "input", "episode.parquet"))
    os.environ["DROID_SCENE_JSON"] = scene_p
    try:
        m, d, scene = P.build(T, ep, 130.0)
    finally:
        os.environ.pop("DROID_SCENE_JSON", None)
    traj = np.load(os.path.join(G, native["trajectory"]["path"]))["qpos"]

    def contacts(q):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            out.append((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name, c.dist))
        return out

    # 支撑：首记录帧（静置后）各块与预期支撑体接触、穿入 ≤1.5 mm
    want = [("green_block", "table"), ("stack_arch", "table"), ("stack_plank", "stack_arch"), ("stack_cube", "stack_plank")]
    c0 = contacts(traj[0])
    sup = {}
    for a, b in want:
        ds = [x[2] for x in c0 if {x[0], x[1]} == {a, b}]
        sup[f"{a}->{b}"] = {"in_contact": bool(ds), "min_dist_mm": float(min(ds) * 1000) if ds else None}
    sup_ok = all(v["in_contact"] and v["min_dist_mm"] >= -1.5 for v in sup.values())
    # 安装：机器人（非夹爪手指）全程不与桌面/塔穿插；桌面上表面低于机器人底座
    robot_bodies = {m.body(i).name for i in range(m.nbody) if m.body(i).name.startswith(("link", "g_"))}
    worst = 0.0
    for q in traj[::3]:
        for a, b, dist in contacts(q):
            pair = {a, b}
            if pair & robot_bodies and pair & {"table", "stack_arch", "stack_plank"}:
                worst = min(worst, dist)
    mount = {"robot_base": "Panda link0 固定于世界原点（不随桌面移动）", "table_top_z": scene["table_z"],
             "robot_vs_table_tower_min_dist_mm": float(worst * 1000)}
    mount_ok = worst >= -1.0
    # 显示—碰撞：打开 Blender 工程，比较每个积木在 f88 的世界包围盒与 MuJoCo 几何
    import bpy
    from mathutils import Vector
    bpy.ops.wm.open_mainfile(filepath=os.path.join(G, "render", "scene_stage8.blend"))
    bpy.context.scene.frame_set(88)
    q88 = traj[88 - 1]
    d.qpos[:] = q88
    mujoco.mj_forward(m, d)
    align = {}
    for g in range(m.ngeom):
        bn = m.body(m.geom_bodyid[g]).name
        nm = m.geom(g).name
        if not (bn.startswith("stack_") or bn == "green_block") or nm not in bpy.data.objects:
            continue
        ob = bpy.data.objects[nm]
        bb = np.array([list(ob.matrix_world @ Vector(c)) for c in ob.bound_box])
        R_ = d.geom_xmat[g].reshape(3, 3)
        hs = m.geom_size[g]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * hs
        mj = corners @ R_.T + d.geom_xpos[g]
        err = float(np.abs(np.r_[bb.min(0) - mj.min(0), bb.max(0) - mj.max(0)]).max() * 1000)
        align[nm] = {"aabb_max_abs_diff_mm": err}
    align_ok = bool(align) and all(v["aabb_max_abs_diff_mm"] < 1.0 for v in align.values())
    ev_p = os.path.join(G, "scene_audit_evidence.json")
    json.dump({"support": sup, "mounting": mount, "display_collision_alignment": align}, open(ev_p, "w"),
              indent=2, ensure_ascii=False)
    ev = {"path": os.path.relpath(ev_p, G), "sha256": digest(ev_p)}
    run = json.load(open(os.path.join(G, native["run"]["path"])))
    audit = {"candidate_id": cid, "inputs_sha256": run["inputs_sha256"], "display_inputs_sha256": disp_key,
             "checks": {
                 "support": {"status": "pass" if sup_ok else "fail", "evidence": ev,
                             "observation": "静置后各块与预期支撑体接触：" + "；".join(
                                 f"{k} {v['min_dist_mm']:.2f} mm" if v["in_contact"] else f"{k} 无接触" for k, v in sup.items())},
                 "mounting": {"status": "pass" if mount_ok else "fail", "evidence": ev,
                              "observation": f"机器人底座固定原点；机器人与桌面/塔全程最小距离 {worst*1000:.2f} mm"},
                 "display_collision_alignment": {"status": "pass" if align_ok else "fail", "evidence": ev,
                                                 "observation": "Blender 显示与 MuJoCo 几何 f88 世界包围盒最大差 "
                                                 + ", ".join(f"{k} {v['aabb_max_abs_diff_mm']:.3f} mm" for k, v in align.items())}}}
    ap = os.path.join(G, "scene_audit.json")
    json.dump(audit, open(ap, "w"), indent=2, ensure_ascii=False)
    return ap, sup_ok and mount_ok and align_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--cid", required=True)
    ap.add_argument("--parent", default=None)
    ap.add_argument("--kind", required=True, choices=["base", "mechanical", "appearance"])
    ap.add_argument("--mech", default="{}")
    ap.add_argument("--appearance", default="{}")
    ap.add_argument("--environment", default="{}")
    ap.add_argument("--basis", default="用户要求 Stage 3 场景增强（原话：『嗯，就这么做吧，先1再2最后3』『嗯，去吧』）")
    a = ap.parse_args()
    T = a.testdir
    appearance = dict(APPEAR_DEFAULT, **json.loads(a.appearance))
    env = dict(ENV_DEFAULT, **json.loads(a.environment))
    cfg_p, native, result, scene_p, qfile = build_candidate(T, a.cid, json.loads(a.mech), appearance, env)
    if a.kind == "base":
        log(f"  [{a.cid}] 基线原生证据已生成（作为父候选）")
        return
    display, disp_key, media, required = render_and_validate(T, a.cid, scene_p, qfile, appearance, env, native)
    audit_p, audit_ok = scene_audit(T, a.cid, scene_p, qfile, native, display, disp_key)
    G = gdir(T, a.cid)
    PG = gdir(T, a.parent)
    parent_cfg = os.path.join(PG, "config.json")
    parent_native = rebase(json.load(open(os.path.join(PG, "native", "native_evidence_block.json"))), PG, G)
    before = flatten(json.load(open(parent_cfg))["parameters"])
    after = flatten(json.load(open(cfg_p))["parameters"])
    changes = {k: {"before": before[k], "after": after[k]} for k in before if before[k] != after[k]}

    def ref(p):
        return {"path": os.path.relpath(p, G), "sha256": digest(p)}

    gate = {"candidate_id": a.cid, "task_scope": "augmentation", "physics_required": True, "scope_basis": a.basis,
            "kind": a.kind, "parent_config": ref(parent_cfg), "current_config": ref(cfg_p),
            "declared_changes": changes, "parent_native": parent_native, "native_evidence": native,
            "scene_audit": ref(audit_p), "display_inputs": display, "required_media": required, "media": media}
    gp = os.path.join(G, "augmentation_gate.json")
    json.dump(gate, open(gp, "w"), indent=2, ensure_ascii=False)
    r = subprocess.run([PY, os.path.join(SKILL_SCRIPTS, "check_augmentation_gate.py"), gp],
                       capture_output=True, text=True)
    res = json.loads(r.stdout)
    log(f"  [{a.cid}] 审计 {'pass' if audit_ok else 'FAIL'}  验收 {'PASS' if res['pass'] else 'FAIL'} "
        f"{res.get('errors') or ''}  媒体 {res.get('media_count', 0)}")


if __name__ == "__main__":
    main()
