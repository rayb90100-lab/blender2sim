#!/usr/bin/env python3
"""把一个已过三门槛的 DROID test（如 test4）导出为通用 task_spec.json。

这是唯一允许知道「源 test 布局」的脚本；物体名、颜色、成功谓词通过参数/默认表给出，导出后下游只读 spec。

角色来源有两档（向后兼容）：
  - scene.json 的物体带 role（target/stack/support/distractor，test6 起）：谓词按 role 推导，
    stack 的叠放顺序取 scene.json 的 stack_order（无则按名称排序）
  - 否则退回 test4 的旧规则：--target / --support 指定，其余全部视为塔

相机有两档（同一套 droid_cams 约定）：
  - stage0_replay/camera_refined.json 存在 → 用精修后的外参 + fx（test6 起）
  - 否则用数据集原始外参 + intrinsics_adopted.json 的 fx（test4 的行为，逐字节不变）

用法:
  python tools/r2pear/export_droid.py --src <test4 dir> --out <test5>/stage0_spec/task_spec.json \
      --task_id droid_ep17_green_on_stack --target green_block --support stack_cube
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import droid_common as C  # noqa: E402
import droid_fit_intrinsics as F  # noqa: E402
import droid_cams as DC  # noqa: E402

# 与 test4 droid_stage2_physics 一致的外观/材料（估计值，未称重）
# 仅作兜底：优先用 scene.json 里物体自带的 rgba（test6 起由 stage1 拟合写入）
COLORS = {"green_block": [0.02, 0.32, 0.12, 1], "stack_arch": [0.80, 0.18, 0.03, 1],
          "stack_plank": [0.90, 0.80, 0.05, 1], "stack_cube": [0.85, 0.15, 0.03, 1]}
DEFAULT_RGBA = [0.7, 0.7, 0.7, 1]
DENSITY, FRICTION, SOLREF = 600.0, [1.0, 0.005, 0.0001], [0.01, 1.0]
W, H, WRIST_FX = 320, 180, 150.0

# noslip_iterations 必须写进 spec（不然物理结论无法复现）。
# test5 的全部数字基于 3；test6 起默认 0 —— 圆柱站在方块顶上时 noslip=3 会产生不衰减极限环
# （§63.5），设 0 后 |w| 从 0.540 降到 0.024。用 --physics '{"noslip_iterations":3}' 覆盖。
NOSLIP_DEFAULT = 0
NOSLIP_NOTE = ("0：避免「圆柱站在方块顶上」这类退化接触的不衰减极限环（§63.5b）。"
               "test5 时代为 3；复现 §62 的产物须显式传 --physics '{\"noslip_iterations\":3}'。")


def resolve_success(preds, objects, target, support):
    """校验人工给出的谓词表，并检查它与 --target/--support 一致。"""
    out = []
    for p in preds:
        q = dict(p)
        q.setdefault("pred", "")
        if not q["pred"]:
            raise SystemExit(f"success 条目缺 pred: {p}")
        if q["pred"] in ("on", "released", "upright"):
            if q.get("obj") not in objects:
                raise SystemExit(f"success 引用了不存在的物体: {p}")
        if q["pred"] == "on" and q.get("support") not in objects:
            raise SystemExit(f"success 的 support 不存在: {p}")
        if q["pred"] == "on" and q.get("obj") == target and q.get("support") != support:
            print(f"  WARN success 里 on({target}, {q.get('support')}) 与 --support={support} 不一致")
        out.append(q)
    used = {p["obj"] for p in out if "obj" in p}
    if target not in used:
        print(f"  WARN success 没有提到 --target={target}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task_id", required=True)
    ap.add_argument("--target", required=True, help="被搬运物体")
    ap.add_argument("--support", required=True, help="目标支撑物体")
    ap.add_argument("--scene", default=None,
                    help="scene.json 路径；默认 stage1_scene/scene.json。并行工作或多解对比时显式指定")
    ap.add_argument("--table", default=None,
                    help="table_display.json 路径；默认 stage1_scene/table_display.json")
    ap.add_argument("--physics", default=None,
                    help='JSON 覆盖 physics 字段，例如 \'{"noslip_iterations":3}\' 复现 test5 时代')
    a = ap.parse_args()

    src = a.src
    ep = C.load_episode(os.path.join(src, "input", "episode.parquet"))
    source = json.load(open(os.path.join(src, "input", "source.json")))
    scene_path = a.scene or os.path.join(src, "stage1_scene", "scene.json")
    table_path = a.table or os.path.join(src, "stage1_scene", "table_display.json")
    scene = json.load(open(scene_path))
    # scene 自带 table 时优先（merge_scene.py 的输出），否则读独立的 table 文件
    table = scene.get("table") or json.load(open(table_path))
    print(f"  scene: {scene_path}")
    K = json.load(open(os.path.join(src, "stage0_replay", "intrinsics_adopted.json")))
    wrist = json.load(open(os.path.join(src, "stage0_replay", "wrist_cam_rel_link7.json")))

    objects = {}
    for n, o in scene["objects"].items():
        # rgba：scene.json 自带（test6 起）> test4 名称表 > 中性灰
        rgba = o.get("rgba") or COLORS.get(n, DEFAULT_RGBA)
        objects[n] = {"shape": o.get("shape", "box"), "size": o["size"], "center": o["center"],
                      "yaw_rad": o["yaw_rad"], "rgba": rgba,
                      "density": DENSITY, "friction": FRICTION, "solref": SOLREF}
        # 形状降级必须显式标注，不静默（§61.3：mesh 化排在 P3）
        if o.get("shape_degraded"):
            objects[n]["shape_degraded"] = o["shape_degraded"]
        if o.get("role"):
            objects[n]["role"] = o["role"]
    missing_rgba = sorted(n for n in objects if not scene["objects"][n].get("rgba")
                          and n not in COLORS)
    if missing_rgba:
        print(f"WARN scene.json 缺 rgba 且不在 test4 名称表：{missing_rgba} → 用中性灰")
        print("     （外观同步是 P3 的事；当前仅影响观感，不影响物理与谓词）")

    # ---- 相机：与 droid_cams 完全一致（标定优先级由 Cams 决定）----
    # 用 dc.mj_cam 而不是手拼：保证 test4 无标定时结果与原实现一致。
    # Cams.source() 报告实际生效的那一档：objects_pnp / robot / stage0_rot / dataset / wrist_rel
    dc = DC.Cams(src, ep)
    fov = lambda fx: float(np.degrees(2 * np.arctan(H / 2 / fx)))
    cams = {}
    for c, alias in (("exterior_1_left", "agentview"),):
        pos, quat, fovy = dc.mj_cam(c)
        cams[alias] = {"mount": "world", "source_camera": c, "pos": pos, "quat_wxyz": quat,
                       "fovy_deg": fovy, "intrinsics_status": K["status"],
                       "extrinsics_status": dc.source(c),
                       "calibration_note": ("6-DoF 自标定；未通过门槛的标定不会出现在这里"
                                            if dc.source(c) in ("robot", "objects") else
                                            "未做 6-DoF 标定，退到小旋转精修/数据集外参")}
    cams["wrist"] = {"mount": "link7", "source_camera": "wrist_left", "pos": wrist["pos"],
                     "quat_wxyz": wrist["quat_wxyz"], "fovy_deg": fov(WRIST_FX),
                     "intrinsics_status": "estimated",
                     "extrinsics_status": "handeye_from_dataset_f0"}

    # ---- 成功谓词 ----
    # 优先级：objects.json 的 success（人工确认过的任务语义）> 按 role 推导 > test4 旧规则。
    # 不自己加论文/指令没有的约束：ep71 的 objects.json 明确写了 success_note
    #「指令只要求圆柱放到长条上；终态画面中圆柱竖立（f171），故加 upright。
    #   不要求原小塔保持完好（指令未提）」，因此**不**给塔体加 upright/on。
    has_role = any(o.get("role") for o in scene["objects"].values())
    cfg_success = scene.get("success")
    if cfg_success:
        tmp = resolve_success(cfg_success, objects, a.target, a.support)
        print(f"  成功谓词取自 scene.success（{len(tmp)} 条）："
              f"{[p['pred']+':'+p.get('obj','') for p in tmp]}")
        if scene.get("success_note"):
            print(f"    note: {scene['success_note']}")
        pred_src = {"derived_from": "scene.json success（人工确认）",
                    "note": scene.get("success_note", "")}
        chain = tmp
    elif has_role:
        role = {n: o["role"] for n, o in scene["objects"].items()}
        target_obj = next(n for n in objects if role[n] == "target")
        support_obj = next(n for n in objects if role[n] == "support")
        if (target_obj, support_obj) != (a.target, a.support):
            raise SystemExit(f"--target/--support ({a.target},{a.support}) 与 scene 的 role "
                             f"({target_obj},{support_obj}) 不一致")
        # 叠放链（底→顶）：scene 优先给 support_chain（含站在别人上面的 target），否则取 role=stack
        full = scene.get("support_chain") or [n for n in objects if role[n] == "stack"]
        if set(full) - set(objects):
            raise SystemExit(f"support_chain 含未知物体: {set(full) - set(objects)}")

        # 三类物体要分开处理，不能一概而论：
        #   below  = 链上 target 下方的那摞（ep71 的 拱+黄块）。target 被搬走后它们**允许坍塌**，
        #            所以不加 upright，也不加它们之间的 on。
        #   above  = 链上 target 上方（ep71 没有）。它们本来压在被搬物体上，target 一走必然掉落，
        #            同样不约束。
        #   others = 桌面上的 distractor / support，不加约束（允许被推开）。
        # 唯一稳定约束是：**塔体本身不能整体被推倒** —— 用 below 的最底层「仍在桌面上」来表达。
        if target_obj in full:
            i_t = full.index(target_obj)
            below, above = full[:i_t], full[i_t + 1:]
        else:
            below, above = [], []
        #
        # 关键：below 里的物体（拱、黄块）**不是**可以随便移动的 —— 视频里它们全程没动，
        # 只是塔会因 target 离开而失去顶层。因此对 below 施加 upright（它们不该被推倒），
        # 但不施加「黄块压在拱上」这类堆叠完整性谓词（target 一走，上面那层本来就空了）。
        chain = [{"pred": "upright", "obj": n, "max_tilt_deg": 10.0} for n in below]
        # below 内部的支撑关系仍然成立（拱托着黄块，不因 target 离开而改变）
        chain += [{"pred": "on", "obj": below[i + 1], "support": below[i]}
                  for i in range(len(below) - 1)]
        pred_src = {"target": target_obj, "support": support_obj,
                    "stack_ordered": full,
                    "upright_constrained": below,
                    "unconstrained_above_target": above,
                    "derived_from": "scene.json role + support_chain"}
        ignored = [n for n in objects if role[n] == "distractor"]
        if ignored:
            print(f"  role=distractor 的物体不加谓词（允许被推开）：{ignored}")
        if below:
            print(f"  塔体 {below} 加 upright（不该被推倒）；不约束它上方（target 搬走后本来就空）")
        if above:
            print(f"  {above} 在 target 上方，target 一走必然掉落，不约束")
    else:
        stack = [n for n in objects if n not in (a.target,)]
        order = stack
        chain = ([{"pred": "upright", "obj": n, "max_tilt_deg": 10.0} for n in stack]
                 + [{"pred": "on", "obj": stack[i + 1], "support": stack[i]}
                    for i in range(len(stack) - 1)])
        pred_src = {"derived_from": "legacy: all-but-target is a tower"}

    spec = {
        "schema_version": 1,
        "task_id": a.task_id,
        "instruction": source["instruction"],
        "source": {"dataset": source["dataset"], "episode_index": source["episode_index"],
                   "reconstruction": os.path.relpath(src, os.path.dirname(os.path.abspath(a.out)))},
        "frame": "Panda 基座系，z 向上，单位 m",
        "robot": {"model": "panda_nohand + robotiq_2f85 (menagerie)", "mount_yaw_deg": C.MOUNT_YAW_DEG,
                  "init_qpos": ep["observation.state.joint_position"][0].tolist(),
                  "init_gripper": float(ep["observation.state.gripper_position"][0])},
        "table": {"top_z": table["table_z"], "center_xy": table["center_xy"], "size_xy": table["size_xy"],
                  "yaw_rad": table["yaw_rad"], "thickness": 0.03, "apron_m": table["apron_m"],
                  "height_m": 0.75, "height_status": "estimated（DROID 未提供桌高，取常见办公桌 0.75 m）",
                  "status": table.get("status", "mask_iou_fit")},
        "objects": objects,
        "cameras": cams,
        "render": {"width": 512, "height": 288},
        "predicate_source": pred_src,
        # 终态谓词：只看最终状态，不依赖示教轨迹
        # 谓词来源互斥：cfg_success 已含 on/released，chain 里只剩额外约束（如 upright）；
        # 无 cfg_success 时才由 role 推导补上 on/released 打头。
        "success": (chain if cfg_success else
                    [{"pred": "on", "obj": a.target, "support": a.support},
                     {"pred": "released", "obj": a.target}, *chain]),
        "step_budget": {"control_hz": 15, "max_control_steps": 600, "max_rounds": 25},
        # 接触参数扫描（§62，stage1_robust/contact_sweep）：tc/dt 不改善尾部穿透；solimp 0.99 使 >3 mm 比例 −40%、零成本
        "physics": {"timestep": 0.002, "object_solref": SOLREF, "object_solimp": [0.99, 0.999, 0.001, 0.5, 2],
                    "noslip_iterations": NOSLIP_DEFAULT,
                    "noslip_note": NOSLIP_NOTE},
    }
    # 显式传入的 physics 覆盖（例如测试旧产物时必须钉住 noslip_iterations=3）
    if a.physics:
        over = json.loads(a.physics)
        spec["physics"].update(over)
        print(f"  physics 覆盖: {over}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(spec, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {a.out}: {len(objects)} objects, {len(cams)} cameras, {len(spec['success'])} predicates")


if __name__ == "__main__":
    main()
