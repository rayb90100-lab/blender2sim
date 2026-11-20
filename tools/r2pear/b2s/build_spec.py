"""b2s 第 3 步：scene + demo → r2pear task_spec.json（机器人基座系）+ demo.npz。

谓词从演示**终态**推（不读指令文本猜）：
  对每个被搬运的物体 o：终态底面落在谁的顶面上（≤6 mm 且质心在其水平投影内）→ on(o, 支撑)
                         released(o)；终态倾角 <15° → upright(o, 15°)
角色：target = 被搬运物体；support = 终态支撑；stack = target 初始支撑链；其余 distractor。
"""
import math
import os

import numpy as np

from common import inv, quat_wxyz, rot_of
import droid_common as C

RENDER_W = 512
SUPPORT_TOL = 0.006
UPRIGHT_MAX = 15.0
PHYSICS = {"timestep": 0.002, "object_solref": [0.01, 1.0], "object_solimp": [0.99, 0.999, 0.001, 0.5, 2],
           "noslip_iterations": 0,
           "noslip_note": "0：避免「圆柱站在方块顶上」类退化接触的不衰减极限环（DEVELOPMENT_LOG §63.5b）。"}


def _yaw_in_base(yaw_world, T_base):
    v = np.array([math.cos(yaw_world), math.sin(yaw_world), 0])
    vb = T_base[:3, :3].T @ v
    return math.atan2(vb[1], vb[0])


def intrinsics_px(cam, W, H):
    """Blender 镜头 → OpenCV 像素内参（与 blend 里 intrinsics_320px 属性逐位一致，已核对）。"""
    fit = cam["sensor_fit"]
    if fit == "VERTICAL" or (fit == "AUTO" and H > W):
        f = cam["lens"] / (cam["sensor_height"] if fit == "VERTICAL" else cam["sensor_width"]) * H
        ref = H
    else:
        f = cam["lens"] / cam["sensor_width"] * W
        ref = W
    sx, sy = cam["shift"]
    return {"fx": f, "fy": f, "cx": W / 2 - 0.5 - sx * ref, "cy": H / 2 - 0.5 + sy * ref, "width": W, "height": H}


def final_supports(ex, scene, demo, obj):
    """终态 obj 坐在谁上：用最后一帧（修复后）各物体的世界顶点。"""
    def wv(s):
        o = scene["objects"][s]
        n = o["blend_name"]
        M0 = ex.M0(n)
        S = np.linalg.norm(M0[:3, :3], axis=0)
        v = np.array(ex.E["meshes"][n]["verts_local"]) * S
        M = demo["objects"][s][-1]
        return v @ M[:3, :3].T + M[:3, 3]
    me = wv(obj)
    bot = me[:, 2].min()
    c = me.mean(0)
    from parse_scene import _in_hull_xy
    best = []
    for s in scene["objects"]:
        if s == obj:
            continue
        w = wv(s)
        if abs(bot - w[:, 2].max()) < SUPPORT_TOL and _in_hull_xy(c, w):
            best.append(s)
    if not best and abs(bot - scene["table"]["top_z"]) < SUPPORT_TOL:
        best = ["table"]
    R = demo["objects"][obj][-1][:3, :3]
    tilt = math.degrees(math.acos(min(1, abs(R[2, 2]))))
    return best, tilt


def _cloc(scene, demo, s):
    """对象原点 → 包围盒中心（局部系）。sim 的 body 放在包围盒中心，blend 对象原点不一定在那。"""
    M0 = demo["objects"][s][0]
    return M0[:3, :3].T @ (np.array(scene["objects"][s]["center"]) - M0[:3, 3])


def blend_binding(ex, scene, demo, robot, emb, cfg=None):
    """Web UI 的 Blender 视图直接驱动原 blend 所需的映射（b2s/blend_worker.py 读）。"""
    b = {"objects": {s: {"blend_name": o["blend_name"], "cloc": _cloc(scene, demo, s).tolist()}
                     for s, o in scene["objects"].items()},
         "cameras": {}, "robot": None}
    cfg = cfg or {}
    for s, o in scene["objects"].items():
        root = cfg.get("visual_roots", {}).get(o["blend_name"])
        if root:
            if root not in ex.ancestors(o["blend_name"]):
                raise ValueError(f"Visual root {root!r} is not an ancestor of {o['blend_name']!r}")
            offset = int(demo.get("trim_offset", 0))
            Mr = ex.M(root)[offset]
            b["objects"][s] = {"blend_name": root,
                "cloc": (rot_of(Mr).T @ (np.asarray(o["center"]) - Mr[:3, 3])).tolist()}
    if robot is not None:
        b["robot"] = {"axes": [np.asarray(a).tolist() for a in robot["axes"]],
                      "sign": np.asarray(robot["sign"]).tolist(), "offset": np.asarray(robot["offset"]).tolist(),
                      "q_mj = sign * q_bone + offset": True}
        if emb.get("representation") == "object_joints":
            from object_motion import gripper_binding
            b["robot"].update(representation="object_joints", root=emb["base"], objects=emb["chain"],
                              gripper=gripper_binding(ex, emb))
        else:
            b["robot"].update(armature=emb["armature"], bones=emb["chain"])
    else:
        # Physics exclusions also contain background furniture. Hide only the
        # demonstrator replaced by the simulated robot, preserving the scene.
        root = emb.get("root") or emb.get("armature")
        b["hide"] = [root, *ex.descendants(root)] if root else []
    for root in cfg.get("hide_trees", []):
        b.setdefault("hide", []).extend([root, *ex.descendants(root)])
    return b


def build(ex, scene, demo, robot_q0, cfg, D, out_dir, blend_path, robot=None, emb=None):
    T_base = scene["T_world_base"]
    Tb_inv = inv(T_base)
    tb = scene["table"]
    c_t = Tb_inv @ np.array([*tb["center_xy"], tb["top_z"], 1])
    table = {"top_z": float(c_t[2]), "center_xy": c_t[:2].tolist(), "size_xy": tb["size_xy"],
             "yaw_rad": _yaw_in_base(tb["yaw_rad"], T_base), "thickness": tb["thickness"], "apron_m": 0.0,
             "height_m": float(tb["top_z"] - tb["floor_z"]), "legs": False,
             "status": f"blend 对象 {tb['blend_name']!r}（作者局部系）",
             "legs_note": "blend 自带桌下结构；桌面以下不建腿，避免凭空多出障碍"}
    # 角色与谓词
    moved = demo["moved"]
    success, roles = [], {s: "distractor" for s in scene["objects"]}
    for s in moved:
        sup, tilt = final_supports(ex, scene, demo, s)
        roles[s] = "target"
        if not sup:
            D.add(f"success[{s}]", None, "终态没有坐在任何物体或桌面上（可能还在手里）", warn=True)
            continue
        if len(sup) > 1:
            D.add(f"success[{s}].multi_support", sup, "终态同时接触多个支撑，取第一个", warn=True)
        success.append({"pred": "on", "obj": s, "support": sup[0]})
        success.append({"pred": "released", "obj": s})
        if tilt < UPRIGHT_MAX:
            success.append({"pred": "upright", "obj": s, "max_tilt_deg": UPRIGHT_MAX})
        if sup[0] != "table":
            roles[sup[0]] = "support"
        # 初始支撑链
        chain = scene["objects"][s]["on"]
        while chain and chain[0] != "table":
            if roles.get(chain[0]) == "distractor":
                roles[chain[0]] = "stack"
            chain = scene["objects"][chain[0]]["on"]
        D.add(f"success[{s}]", success[-3:] if tilt < UPRIGHT_MAX else success[-2:],
              f"终态底面落在 {sup[0]} 顶面上，倾角 {tilt:.1f}°")
    if not success:
        D.add("success", [], "没有推出任何任务谓词；需在 b2s_config.json 里写 success", warn=True)
    if cfg.get("success"):
        success = cfg["success"]
        D.add("success", success, "b2s_config 覆盖")

    objects = {}
    # ---- 竖直就位：sim.py 只表达「直立 + 绕 z 朝向」，所以物体的 z 必须由支撑面反推，
    # 不能沿用 blend 里那个（可能倾斜的）包围盒中心。
    # 实测 UMI ep68：blend 首帧杯子被前一次夹空推歪 2.4°，包围盒中心比直立中心低 1.000 mm；
    # 按中心摆 → 杯底插进桌面 1.0 mm，V1 静置里杯子在 1 mm 深处做 1 s 周期**不衰减**的来回摇摆
    # （|v|=12.4 mm/s、|w|=0.53 rad/s），调 solref/solimp/dt 全部无效 —— 是初始位姿，不是接触参数。
    # 就位规则：z_center = 支撑面顶面 + h/2（支撑可以是桌面，也可以是另一个物体）。
    # 在 test1 的五个物体上，这条规则与原作者给出的 center 差值为 0.000 mm（全部精确一致），
    # 所以对已有场景没有任何副作用。
    def _obj_h(o):
        return float(o["size"][1]) if o["shape"] == "cylinder" else float(o["size"][2])

    def _support_top(s, depth=0):
        o = scene["objects"][s]
        sups = [x for x in (o.get("on") or []) if x == "table" or x in scene["objects"]]
        best = None
        for x in sups:
            if x == "table":
                z = float(scene["table"]["top_z"])
            elif depth < 8:
                z = _support_top(x, depth + 1) + _obj_h(scene["objects"][x])
            else:
                continue
            best = z if best is None else max(best, z)
        return best

    for s, o in scene["objects"].items():
        c_world = np.array([*o["center"], 1.0], float)
        shape = o["shape"]
        h = _obj_h(o)
        z_sup = _support_top(s)
        if z_sup is None:
            z_sup = float(o["bottom_z"])
        dz = (z_sup + h / 2) - float(c_world[2])
        c_world[2] += dz
        c = Tb_inv @ c_world
        entry = {"shape": "box" if shape == "unsupported" else shape, "size": o["size"],
                 "center": c[:3].tolist(), "yaw_rad": _yaw_in_base(o["yaw"], T_base), "rgba": o["rgba"],
                 "density": 600.0, "friction": [1.0, 0.005, 0.0001], "solref": [0.01, 1.0], "role": roles[s],
                 "blend_name": o["blend_name"], "shape_evidence": o["why"], "color_source": o["color_source"]}
        if abs(dz) > 1e-4:
            entry["seated_dz_mm"] = round(dz * 1000, 2)
            D.add(f"seat[{s}]", round(dz * 1000, 2),
                  f"blend 里的姿态不是直立（包围盒中心与「支撑面+h/2」差 {dz * 1000:+.1f} mm），"
                  f"sim 只表达直立 → 按支撑面就位，避免底面插进支撑面", warn=True)
        if shape == "unsupported":
            entry["shape_degraded"] = "形状不在 sim.py 支持集内，按包围盒建 box；需要替换资产"
        objects[s] = entry
        if s in cfg.get("object_overrides", {}):
            allowed = {"grasp_z_min_cm"}
            overrides = cfg["object_overrides"][s]
            if set(overrides) - allowed:
                raise ValueError(f"Unsupported object overrides for {s}: {set(overrides) - allowed}")
            entry.update(overrides)

    # 相机
    W = RENDER_W
    sw, sh = ex.E["scene"]["resolution"]
    H = int(round(W * sh / sw))
    cams, n_world = {}, 0
    use = [n for n, c in scene["cameras"].items() if c.get("stream")] or list(scene["cameras"])
    for cn in sorted(use, key=lambda n: (scene["cameras"][n]["mount"] != "world", scene["cameras"][n]["stream"] or n)):
        c = scene["cameras"][cn]
        Mc = ex.M(cn).copy()
        Mc[:, :3, :3] = np.array([rot_of(m) for m in Mc])
        # demo 可能已被 trim_failed_attempts 裁掉前缀（demo["q"] 比 blend 帧数短）。
        # 手持相机的外参是逐帧的对齐量，必须跟着裁，否则 zip 从第 0 帧开始配 → 帧号整体错位。
        # 实测 UMI ep68：裁掉前 683 帧后不裁 Mc，wrist 相对 link7 的离散从 0.1 mm 变成 736 mm。
        _off = int(demo.get("trim_offset", 0))
        if c["mount"] == "ee" and _off:
            Mc = Mc[_off:_off + len(demo["q"])] if len(Mc) > _off else Mc[:1]
        K = intrinsics_px(c, W, H)
        fovy = math.degrees(2 * math.atan(H / 2 / K["fy"]))
        if c["mount"] == "ee" and demo.get("q") is not None:
            import embodiment as EM
            fk7 = EM._mj_fk(demo["q"])[7]
            X = np.array([inv(T_base @ a) @ b for a, b in zip(fk7, Mc)])
            spread = float(np.linalg.norm(X[:, :3, 3] - X[0, :3, 3], axis=1).max() * 1000)
            Xm = X[len(X) // 2]
            name = "wrist" if "wrist" not in cams else f"wrist_{len(cams)}"
            cams[name] = {"mount": "link7", "pos": Xm[:3, 3].tolist(), "quat_wxyz": quat_wxyz(Xm[:3, :3]),
                          "fovy_deg": fovy, "intrinsics": K, "blend_camera": cn, "source_camera": c["stream"],
                          "rigid_spread_mm": spread}
            D.add(f"camera[{name}]", cn, f"挂在机器人末端；相对 link7 逐帧离散 {spread:.3f} mm", warn=spread > 1)
        else:
            Tc = Tb_inv @ Mc[0]
            name = "agentview" if n_world == 0 else f"agentview_{n_world + 1}"
            n_world += 1
            cams[name] = {"mount": "world", "pos": Tc[:3, 3].tolist(), "quat_wxyz": quat_wxyz(Tc[:3, :3]),
                          "fovy_deg": fovy, "intrinsics": K, "blend_camera": cn, "source_camera": c["stream"]}
            D.add(f"camera[{name}]", cn, f"世界固定，fovy {fovy:.1f}°，cx/cy 偏离中心 "
                  f"{K['cx'] - (W / 2 - 0.5):+.1f}/{K['cy'] - (H / 2 - 0.5):+.1f} px @ {W}x{H}")
    for name, camera in cfg.get("additional_cameras", {}).items():
        if name in cams:
            raise ValueError(f"Additional camera would overwrite extracted camera: {name}")
        cams[name] = {**camera, "source_camera": None,
                      "blend_camera": f"b2s_virtual_{name}", "virtual": True}
        D.add(f"camera[{name}]", "virtual", "配置的仿真观察相机，无对应真实源视角")
    static = []
    for st in scene["static"]:
        c = Tb_inv @ np.array([*st["center"], 1])
        R = T_base[:3, :3].T @ np.array(st["R"])
        static.append({"name": st["blend_name"], "center": c[:3].tolist(), "half": st["half"],
                       "quat_wxyz": quat_wxyz(R), "rgba": st["rgba"]})

    instr = cfg.get("instruction") or ex.E["scene"]["props"].get("instruction") or demo.get("dataset_instruction")
    if not instr and demo.get("dataset"):
        ep = C.load_episode(demo["dataset"]["path"])
        instr = ep["language_instruction"][0]
    if not instr:
        sj = os.path.join(os.path.dirname(blend_path), "source.json")
        if os.path.exists(sj):
            import json
            instr = json.load(open(sj)).get("instruction")
    spec = {
        "schema_version": 1,
        "task_id": cfg.get("task_id") or os.path.splitext(os.path.basename(blend_path))[0].lower(),
        "instruction": instr or "",
        "source": {"blend": os.path.abspath(blend_path), "generator": "tools/r2pear/b2s (blend → sim)",
                   "dataset": demo.get("dataset"), "embodiment": demo["embodiment"],
                   "gripper_source": demo["g_source"], "ee_source": demo["ee_source"]},
        "frame": "机器人基座系，z 向上，单位 m",
        "robot": {"model": "panda_nohand + robotiq_2f85 (menagerie)", "mount_yaw_deg": C.MOUNT_YAW_DEG,
                  "init_qpos": [float(x) for x in robot_q0], "init_gripper": float(demo["g"][0]),
                  "T_world_base": T_base.tolist()},
        "table": table, "objects": objects, "static": static, "cameras": cams,
        "render": {"width": W, "height": H},
        "predicate_source": {"derived_from": "演示终态的支撑关系（b2s/build_spec.final_supports）"},
        "success": success,
        "step_budget": {"control_hz": int(round(demo["fps"])), "max_control_steps": 600, "max_rounds": 25},
        "physics": dict(PHYSICS),
        "blend_binding": blend_binding(ex, scene, demo, robot, emb or {}, cfg),
    }
    spec["source"]["timing"] = {"source_fps": float(demo.get("source_fps", demo["fps"])),
                                  "playback_fps": float(demo["fps"]),
                                  "time_scale": float(cfg.get("retarget_time_scale", 1.0))}
    for cn, c in spec["cameras"].items():
        spec["blend_binding"]["cameras"][cn] = c["blend_camera"]
    if not instr:
        D.add("instruction", None, "blend/数据集/source.json 都没有任务指令", warn=True)
    # 被夹物体宽度 vs 夹爪停止开口（几何一致性；不改几何，只报告）
    if demo.get("stall_aperture_m") is not None:
        for s in moved:
            o = objects[s]
            Xg = demo.get("grasp_X", {}).get(s)
            if o["shape"] == "cylinder":
                w = 2 * o["size"][0]
            elif Xg is not None:
                # size is in the classified shape frame (longest edge first),
                # not necessarily the author's local axes. Measure the actual
                # mesh span along the closing axis at the demonstrated grasp.
                yo = Xg[:3, :3].T @ np.array([0, 1.0, 0])
                n = scene["objects"][s]["blend_name"]
                vertices = np.asarray(ex.E["meshes"][n]["verts_local"])
                vertices = vertices * np.linalg.norm(ex.M0(n)[:3, :3], axis=0)
                w = float(np.ptp(vertices @ yo))
            else:
                continue
            ap = demo["stall_aperture_m"]
            bad = abs(w - ap) > 0.003
            D.add(f"grasp_width[{s}]", {"object_mm": round(w * 1000, 1), "stall_aperture_mm": round(ap * 1000, 1)},
                  "物体宽度与夹爪停止开口差 >3 mm：blend 尺寸或夹爪标定有误" if bad else "一致（≤3 mm）", warn=bad)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "demo.npz"), q=demo["q"] if demo.get("q") is not None else np.zeros((0, 7)),
             g=demo["g"], g_cmd=demo["g_cmd"], ee_world=demo["ee_world"], fps=demo["fps"],
             frames=np.array(demo["frames"]),
             **{f"cloc__{s}": _cloc(scene, demo, s) for s in demo["objects"]},
             **{f"obj__{s}": M for s, M in demo["objects"].items()},
             **{f"valid__{s}": v for s, v in demo["objects_valid"].items()})
    return spec
