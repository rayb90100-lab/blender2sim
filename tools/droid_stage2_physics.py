#!/usr/bin/env python3
"""test4 Stage 2：DROID 关节真值驱动的原生物理执行。

- 机器人：Panda 位置伺服跟踪 observation.state.joint_position（15 fps 线性插值到物理步长）
- 夹爪：Robotiq 2F-85 actuator ctrl = 255 × gripper_position
- 物体：Stage 1 拟合的箱体，freejoint，只受重力与接触推进
  —— 不写 qpos、不加 weld、不附着（skill mujoco.md：状态回放/附着不能冒充原生成功）

输出 stage2_physics/：rollout.npz、physics_report.json、对比视频（真实 vs MuJoCo 同相机）

用法:
  python tools/droid_stage2_physics.py --testdir <testdir> [--render]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import droid_common as C  # noqa: E402
import droid_fit_intrinsics as F  # noqa: E402

FFMPEG = "/home/ubuntu/mambaforge/envs/E2E/bin/ffmpeg"
CAMS = ("exterior_1_left", "exterior_2_left", "wrist_left")
EXT_CAMS = ("exterior_1_left", "exterior_2_left")
# 腕部相机：挂 link7，相对位姿 = link7(q0)^-1 · 首帧外参（stage0_replay/wrist_cam_rel_link7.json）；
# LeRobot 移植版腕部外参整段恒定，只代表首帧基座系位姿（DEVELOPMENT_LOG §47）。
WRIST_FX = 150.0   # 估计：指尖轮廓在 fx 145–160 间最吻合
W, H = 320, 180

# 物理参数：玩具木块，未称重 → 估计值（木材密度 ~600 kg/m³）
DENSITY = 600.0
FRICTION = (1.0, 0.005, 0.0001)
# 木块接触刚度：默认 solref 时间常数 0.02 s 对 20–30 g 小木块冲击太软，
# f142 落放瞬间穿透 3.54 mm（超出事先定的 3 mm 上限）。0.01 s → 0.70 mm（§52）。
# 取 0.01 而非 0.005：后者接近 MuJoCo 建议下限 2×timestep=0.004 s，稳定余量小。
BLOCK_SOLREF = (0.01, 1.0)


def log(m): print(m, flush=True)


def build(testdir, ep, fx):
    import mujoco
    from scipy.spatial.transform import Rotation as R
    # 允许用环境变量覆盖场景（失败/扰动实验用，不改动正式 scene.json）
    sp = os.environ.get("DROID_SCENE_JSON", os.path.join(testdir, "stage1_scene", "scene.json"))
    scene = json.load(open(sp))
    zt = scene["table_z"]
    spec = C.build_spec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 3
    wb = spec.worldbody

    # 桌面：上表面 = z_table；避开机器人基座（基座在原点，桌面从 x=0.15 起）
    tb = wb.add_body(name="table", pos=[0.60, 0.0, zt - 0.02])
    tb.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.45, 0.55, 0.02],
                rgba=[0.55, 0.33, 0.20, 1], friction=list(FRICTION))
    wb.add_light(pos=[0.5, 0, 1.5], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7])

    colors = {"green_block": [0.02, 0.32, 0.12, 1], "block_stack": [0.80, 0.22, 0.03, 1],
              "stack_arch": [0.80, 0.18, 0.03, 1], "stack_plank": [0.90, 0.80, 0.05, 1],
              "stack_cube": [0.85, 0.15, 0.03, 1]}
    for name, o in scene["objects"].items():
        sx, sy, h = o["size"]
        q = R.from_euler("z", o["yaw_rad"]).as_quat()
        b = wb.add_body(name=name, pos=o["center"], quat=[q[3], q[0], q[1], q[2]])
        b.add_freejoint(name=f"{name}_free")
        kw = dict(type=mujoco.mjtGeom.mjGEOM_BOX, density=DENSITY, rgba=colors.get(name, [0.7] * 3 + [1]),
                  friction=list(FRICTION), condim=4, solref=list(BLOCK_SOLREF))
        if o.get("shape") == "arch":
            # ∩ 形拱：顶板（厚 0.35h，全长）+ 两腿（各占长度 30%，高 0.65h）。比例为近似（§49）
            ts, lh, lw = 0.35 * h, 0.65 * h, 0.30 * sx
            b.add_geom(name=f"{name}_top", size=[sx / 2, sy / 2, ts / 2], pos=[0, 0, h / 2 - ts / 2], **kw)
            for sgn, tag in ((-1, "L"), (1, "R")):
                b.add_geom(name=f"{name}_leg{tag}", size=[lw / 2, sy / 2, lh / 2],
                           pos=[sgn * (sx / 2 - lw / 2), 0, -h / 2 + lh / 2], **kw)
        else:
            b.add_geom(name=f"{name}_geom", size=[sx / 2, sy / 2, h / 2], **kw)

    for c in EXT_CAMS:
        ext = ep[f"camera_extrinsics.{c}"][0]
        cam = wb.add_camera(name=c, pos=ext[:3], quat=F.cam_quat_mj(ext[3:]))
        cam.fovy = float(np.degrees(2 * np.arctan(H / 2 / fx)))
    rel = json.load(open(os.path.join(testdir, "stage0_replay", "wrist_cam_rel_link7.json")))
    wc = spec.body("link7").add_camera(name="wrist_left", pos=rel["pos"], quat=rel["quat_wxyz"])
    wc.fovy = float(np.degrees(2 * np.arctan(H / 2 / WRIST_FX)))
    spec.visual.global_.offwidth = 960
    spec.visual.global_.offheight = 540
    m = spec.compile()
    build.last_spec = spec
    return m, mujoco.MjData(m), scene


def contact_force(m, d, body_a, geom_b_prefix):
    """body_a 与名字以 prefix 开头的 body 之间的法向力之和（N）。"""
    import mujoco
    ba = m.body(body_a).id
    tot = 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        other = b2 if b1 == ba else (b1 if b2 == ba else None)
        if other is None:
            continue
        if m.body(other).name.startswith(geom_b_prefix):
            mujoco.mj_contactForce(m, d, i, f6)
            tot += abs(f6[0])
    return tot


def run(testdir, render=False, qfile=None, tag=None, on_frame=None, native_out=None, gate_root=None,
        review_scene=None, candidate_id=None):
    """on_frame(m, d, src_frame)：每个源帧物理推进后调用（环节 8 用来记录几何位姿）。"""
    import mujoco
    import cv2
    ep = C.load_episode(os.path.join(testdir, "input", "episode.parquet"))
    K = json.load(open(os.path.join(testdir, "stage0_replay", "intrinsics_adopted.json")))
    m, d, scene = build(testdir, ep, K["fx"])
    J = ep["observation.state.joint_position"]
    if qfile:                                   # 用 IK 解算的关节代替真值
        J = np.load(qfile)
        log(f"  关节来源: {qfile}")
    G = ep["observation.state.gripper_position"].ravel()
    n_src = len(J)
    fps = 15.0
    dt = m.opt.timestep
    sub = int(round(1 / fps / dt))           # 每源帧的物理步数（33）
    qa = C.arm_qadr(m)
    arm_act = [m.actuator(f"actuator{i}").id for i in range(1, 8)]
    grip_act = m.actuator("g_fingers_actuator").id

    # 初态：关节 = 首帧，夹爪张开，物体静置
    d.qpos[qa] = J[0]
    d.ctrl[arm_act] = J[0]
    d.ctrl[grip_act] = 255 * G[0]
    mujoco.mj_forward(m, d)
    for _ in range(250):                       # 0.5 s 静置，让接触落定（不计入时间映射）
        mujoco.mj_step(m, d)

    ga = int(m.joint("green_block_free").qposadr[0])
    # 支撑目标：塔顶块（拆分后为 stack_cube；旧单箱体场景兼容 block_stack）
    tower = [n for n in ("stack_arch", "stack_plank", "stack_cube") if n in scene["objects"]] or ["block_stack"]
    top_name = tower[-1]
    tadr = {n: int(m.joint(f"{n}_free").qposadr[0]) for n in tower}
    sa = tadr[top_name]
    g0 = d.qpos[ga:ga + 3].copy()
    s0 = d.qpos[sa:sa + 3].copy()
    tower0 = {n: d.qpos[a_:a_ + 7].copy() for n, a_ in tadr.items()}

    qpos_init = d.qpos.copy()
    rec = {"qpos": [], "t": [], "src": [], "green": [], "stack": [], "pinch": [], "grip_force": [],
           "green_on_stack": [], "green_on_table": [], "qpos_arm": [], "pen": [],
           **{f"tower_{n}": [] for n in tower}}
    pinch = m.site("g_pinch").id
    renderers = {}
    frames = {c: [] for c in CAMS}
    if render:
        renderers = {c: mujoco.Renderer(m, 540, 960) for c in CAMS}

    for f in range(n_src - 1):
        for k in range(sub):
            a = k / sub
            d.ctrl[arm_act] = (1 - a) * J[f] + a * J[f + 1]
            d.ctrl[grip_act] = 255 * ((1 - a) * G[f] + a * G[f + 1])
            mujoco.mj_step(m, d)
        rec["t"].append(d.time)
        rec["qpos"].append(d.qpos.copy())
        rec["src"].append(f + 1)
        rec["green"].append(d.qpos[ga:ga + 7].copy())
        rec["stack"].append(d.qpos[sa:sa + 7].copy())
        rec["pinch"].append(d.site_xpos[pinch].copy())
        rec["qpos_arm"].append(d.qpos[qa].copy())
        rec["grip_force"].append(contact_force(m, d, "green_block", "g_"))
        rec["green_on_stack"].append(contact_force(m, d, "green_block", top_name))
        for n, a_ in tadr.items():
            rec[f"tower_{n}"].append(d.qpos[a_:a_ + 7].copy())
        rec["green_on_table"].append(contact_force(m, d, "green_block", "table"))
        rec["pen"].append(min([d.contact[i].dist for i in range(d.ncon)] or [0.0]))
        if on_frame is not None:
            on_frame(m, d, f + 1)
        if render:
            for c in CAMS:
                r = renderers[c]
                r.update_scene(d, camera=c)
                frames[c].append(cv2.resize(r.render(), (640, 360), interpolation=cv2.INTER_AREA))

    for k in rec:
        rec[k] = np.array(rec[k])
    out = os.path.join(testdir, "stage2_physics", tag) if tag else os.path.join(testdir, "stage2_physics")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "rollout.npz"), **rec)

    # ---- 判定 ----
    gz = rec["green"][:, 2]
    to = scene["objects"][top_name]
    st_top = to["center"][2] + to["size"][2] / 2
    gh = scene["objects"]["green_block"]["size"][2]
    fin = rec["green"][-1, :3]
    stk = rec["stack"][-1, :3]
    lift_mm = (gz.max() - g0[2]) * 1000
    hold = rec["grip_force"] > 0.5
    on_stack = rec["green_on_stack"][-15:].mean()
    from scipy.spatial.transform import Rotation as R
    tilt = np.degrees(R.from_quat(rec["green"][-1, [4, 5, 6, 3]]).as_euler("xyz")[:2])
    report = {
        "duration_s": float(rec["t"][-1]),
        "src_frames": n_src,
        "green_start": g0.tolist(), "green_end": fin.tolist(),
        "green_max_lift_mm": float(lift_mm),
        "green_end_z_expected_on_stack_mm": float((st_top + gh / 2) * 1000),
        "green_end_z_mm": float(fin[2] * 1000),
        "green_end_xy_vs_stack_mm": ((fin[:2] - stk[:2]) * 1000).tolist(),
        "green_end_tilt_deg": tilt.tolist(),
        "stack_drift_mm": float(np.linalg.norm(stk - s0) * 1000),
        "grip_contact_frames": int(hold.sum()),
        "grip_contact_interval": [int(rec["src"][hold][0]), int(rec["src"][hold][-1])] if hold.any() else None,
        "green_on_stack_force_last1s_N": float(on_stack),
        "green_weight_N": float(m.body("green_block").mass[0] * 9.81),
        "min_contact_dist_mm": float(rec["pen"].min() * 1000),
    }
    # ---- 塔的完整性：任务核心是「放上去且不碰倒」----
    def tilt_deg(q_wxyz, q0_wxyz):
        Rn = R.from_quat(q_wxyz[[1, 2, 3, 0]])
        R0 = R.from_quat(q0_wxyz[[1, 2, 3, 0]])
        z_now, z0 = Rn.apply([0, 0, 1]), R0.apply([0, 0, 1])
        return float(np.degrees(np.arccos(np.clip(z_now @ z0, -1, 1))))
    tw = {}
    for n in tower:
        T_ = rec[f"tower_{n}"]
        drift = np.linalg.norm(T_[:, :3] - tower0[n][:3], axis=1) * 1000
        tl = np.array([tilt_deg(q, tower0[n][3:7]) for q in T_[:, 3:7]])
        tw[n] = {"max_drift_mm": float(drift.max()), "end_drift_mm": float(drift[-1]),
                 "max_tilt_deg": float(tl.max()), "end_tilt_deg": float(tl[-1]),
                 "mass_g": float(m.body(n).mass[0] * 1000)}
    report["support_block"] = top_name
    report["tower"] = tw
    report["tower_max_drift_mm"] = max(v["max_drift_mm"] for v in tw.values())
    report["tower_max_tilt_deg"] = max(v["max_tilt_deg"] for v in tw.values())
    report["tower_intact"] = bool(all(v["end_drift_mm"] < 10 and v["end_tilt_deg"] < 10 for v in tw.values()))
    report["task_success"] = bool(abs(fin[2] - (st_top + gh / 2)) < 0.012 and
                                  np.linalg.norm(fin[:2] - stk[:2]) < 0.03 and
                                  on_stack > 0.5 * report["green_weight_N"] and report["tower_intact"])
    json.dump(report, open(os.path.join(out, "physics_report.json"), "w"), indent=2, ensure_ascii=False)
    for k, v in report.items():
        log(f"  {k:36s} {np.round(v, 2) if isinstance(v, (float, list)) else v}")

    if native_out:
        write_native_evidence(testdir, native_out, gate_root, review_scene, candidate_id, m, ep, J, G, sub,
                              qpos_init, rec, report, qfile)

    if render:
        for r in renderers.values():
            r.close()
        caps = {c: cv2.VideoCapture(os.path.join(testdir, "input", "videos", f"{c}.mp4")) for c in CAMS}
        for c in caps.values():
            c.read()                          # 物理第 i 帧对应源帧 i+1
        p = os.path.join(out, "compare_real_vs_mujoco.mp4")
        vw = None
        for i in range(len(frames[CAMS[0]])):
            row = []
            for c in CAMS:
                ok, real = caps[c].read()
                real = cv2.resize(real, (640, 360), interpolation=cv2.INTER_CUBIC)
                sim = cv2.cvtColor(frames[c][i], cv2.COLOR_RGB2BGR)
                tile = np.hstack([real, np.full((360, 4, 3), 40, np.uint8), sim])
                cv2.putText(tile, f"{c} REAL", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(tile, "MUJOCO native physics" + (" (IK joints)" if qfile else " (joint GT)"), (652, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
                cv2.putText(tile, f"src f{rec['src'][i]:03d}  grip={G[rec['src'][i]]:.2f}  "
                            f"Fgrip={rec['grip_force'][i]:.1f}N", (8, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                row.append(tile)
            fr = np.vstack(row)
            if vw is None:
                vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fr.shape[1], fr.shape[0]))
            vw.write(fr)
            if rec["src"][i] in (40, 84, 110, 142, n_src - 1):
                cv2.imwrite(os.path.join(out, f"compare_f{rec['src'][i]:03d}.png"), fr)
        vw.release()
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", p, "-c:v", "libx264", "-crf", "20",
                        "-pix_fmt", "yuv420p", p.replace(".mp4", "_h264.mp4")], check=True)
        log(f"  视频 {p.replace('.mp4', '_h264.mp4')}")
    return report


# ---------------- 原生执行证据（skill native-and-augmentation-gates.md §2）----------------
# 判据阈值按任务容差设定（绿方块半宽约 16 mm），理由见 DEVELOPMENT_LOG §48；不是按已有结果倒推。
CRITERIA = {"rules": {
    "task_success": {"op": "eq", "value": True},
    "green_end_xy_err_mm": {"op": "le", "value": 15.0},       # 落点在堆顶面内：< 绿方块半宽
    "green_end_z_err_mm": {"op": "le", "value": 10.0},        # 终态高度 = 堆顶 + 半高
    "green_end_tilt_deg": {"op": "le", "value": 5.0},         # 平放、未翻倒
    "support_ratio": {"op": "ge", "value": 0.5},              # 末 1 s 堆承载 ≥ 一半重力
    "grip_contact_frames": {"op": "ge", "value": 30},         # 真实夹持约 54 帧（f88–142），至少过半
    "stack_drift_mm": {"op": "le", "value": 5.0},             # 支撑块（塔顶）终态未被推动
    "tower_max_drift_mm": {"op": "le", "value": 10.0},        # 塔各块全程最大位移 < 约 1/3 块宽
    "tower_max_tilt_deg": {"op": "le", "value": 10.0},        # 塔各块全程最大倾角（翻倒判据）
    "min_contact_dist_mm": {"op": "ge", "value": -3.0},       # 软接触穿入上限
}}


def write_native_evidence(testdir, out, gate_root, review_scene, candidate_id, m, ep, J, G, sub,
                          qpos_init, rec, report, qfile):
    import hashlib
    import time
    import mujoco
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Real2Gym",
                                    "skills", "real2sim-prompt", "scripts"))
    from evidence_checks import Evidence, digest
    os.makedirs(out, exist_ok=True)
    gate_root = os.path.abspath(gate_root)

    def ref(p):
        return {"path": os.path.relpath(os.path.abspath(p), gate_root), "sha256": digest(p)}

    def dump(name, obj):
        p = os.path.join(out, name)
        json.dump(obj, open(p, "w"), indent=2, ensure_ascii=False)
        return p

    # model：实际编译的 MjSpec 导出
    model_p = os.path.join(out, "model.xml")
    open(model_p, "w").write(build.last_spec.to_xml())
    # control：手臂关节目标、夹爪命令、插值方式
    ctrl_p = os.path.join(out, "control.npz")
    np.savez(ctrl_p, arm_targets=J, gripper_cmd=G, substeps=np.array(sub))
    init_p = dump("initial_state.json", {"qpos": qpos_init.tolist(),
                                         "settle_steps": 250, "note": "首帧关节 + 夹爪张开，静置 0.5 s 后开始计时"})
    simcfg_p = dump("sim_config.json", {"timestep": float(m.opt.timestep), "integrator": int(m.opt.integrator),
                                        "cone": int(m.opt.cone), "noslip_iterations": int(m.opt.noslip_iterations),
                                        "mujoco_version": mujoco.__version__, "seed": None, "deterministic": True,
                                        "control_interp": "源帧间线性插值，每源帧 %d 物理步" % sub})
    crit_p = dump("criteria.json", CRITERIA)
    common_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "droid_common.py")
    scen_p = os.environ.get("DROID_SCENARIO_FILE") or dump("scenario.json", {
        "candidate_id": candidate_id,
        "stage1_scene": json.load(open(os.path.join(testdir, "stage1_scene", "scene.json"))),
        "mount_yaw_deg": C.MOUNT_YAW_DEG, "density": DENSITY, "friction": list(FRICTION),
        "block_solref": list(BLOCK_SOLREF),
        "joint_source": qfile or "observation.state.joint_position (DROID 真值)",
        "droid_common_sha256": digest(common_p)})
    # 增强验收要求 native.inputs.scenario == 当前候选配置文件（check_augmentation_gate.py）
    producer = os.path.abspath(__file__)
    # assets：模型实际引用的网格文件
    names = {mh.file for mh in build.last_spec.meshes if mh.file}
    assets = []
    for d_ in (os.path.join(C.MENAGERIE, "franka_emika_panda", "assets"),
               os.path.join(C.MENAGERIE, "robotiq_2f85", "assets")):
        for n in sorted(names):
            p = os.path.join(d_, os.path.basename(n))
            if os.path.exists(p):
                assets.append(ref(p))
    inputs = {"model": ref(model_p), "control": ref(ctrl_p), "initial_state": ref(init_p),
              "sim_config": ref(simcfg_p), "criteria": ref(crit_p), "scenario": ref(scen_p),
              "producer": ref(producer), "assets": assets}
    if review_scene:
        inputs["review_scene"] = ref(review_scene)
    key = Evidence(gate_root).inputs(inputs)

    traj_p = os.path.join(out, "trajectory.npz")
    np.savez(traj_p, qpos=np.asarray(rec["qpos"]), time_s=np.asarray(rec["t"]))
    tr = ref(traj_p)
    run_id = f"{candidate_id}-{int(time.time())}"
    run_p = dump("run.json", {"candidate_id": candidate_id, "run_id": run_id, "inputs_sha256": key,
                              "trajectory_sha256": tr["sha256"], "mode": "native", "completed": True,
                              "frame_count": int(len(rec["qpos"]))})
    stk_top = None
    metrics = {
        "task_success": bool(report["task_success"]),
        "green_end_xy_err_mm": float(np.linalg.norm(report["green_end_xy_vs_stack_mm"])),
        "green_end_z_err_mm": float(abs(report["green_end_z_mm"] - report["green_end_z_expected_on_stack_mm"])),
        "green_end_tilt_deg": float(np.max(np.abs(report["green_end_tilt_deg"]))),
        "support_ratio": float(report["green_on_stack_force_last1s_N"] / report["green_weight_N"]),
        "grip_contact_frames": int(report["grip_contact_frames"]),
        "stack_drift_mm": float(report["stack_drift_mm"]),
        "tower_max_drift_mm": float(report["tower_max_drift_mm"]),
        "tower_max_tilt_deg": float(report["tower_max_tilt_deg"]),
        "min_contact_dist_mm": float(report["min_contact_dist_mm"]),
    }
    ok = all(
        (metrics[k] == r["value"]) if r["op"] == "eq" else
        (metrics[k] <= r["value"]) if r["op"] == "le" else (metrics[k] >= r["value"])
        for k, r in CRITERIA["rules"].items())
    res_p = dump("result.json", {"candidate_id": candidate_id, "run_id": run_id, "inputs_sha256": key,
                                 "trajectory_sha256": tr["sha256"], "criteria_sha256": inputs["criteria"]["sha256"],
                                 "status": "pass" if ok else "fail", "metrics": metrics})
    block = {"inputs": inputs, "run": ref(run_p), "trajectory": tr, "result": ref(res_p)}
    json.dump(block, open(os.path.join(out, "native_evidence_block.json"), "w"), indent=2, ensure_ascii=False)
    log(f"  原生证据 -> {out}  status={'pass' if ok else 'fail'}  metrics={ {k: round(v, 2) if isinstance(v, float) else v for k, v in metrics.items()} }")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--qfile", default=None, help="IK 关节轨迹 .npy（默认用数据集真值）")
    ap.add_argument("--tag", default=None, help="输出子目录")
    ap.add_argument("--native_out", default=None, help="写原生执行证据的目录")
    ap.add_argument("--gate_root", default=None, help="review_gate.json 所在目录（证据路径相对它）")
    ap.add_argument("--review_scene", default=None, help="当前复核的 Blender 工程")
    ap.add_argument("--candidate_id", default=None)
    a = ap.parse_args()
    run(a.testdir, a.render, a.qfile, a.tag, native_out=a.native_out, gate_root=a.gate_root,
        review_scene=a.review_scene, candidate_id=a.candidate_id)
