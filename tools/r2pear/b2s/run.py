#!/usr/bin/env python3
"""b2s：任意 .blend → r2pear 可交互 MuJoCo 环境（PEAR StatefulEnv 接口 + Web UI 直接可用）。

  python tools/r2pear/b2s/run.py --blend <X.blend> --out <simdir> [--config b2s_config.json] [--videos <dir>]

产物（与 Real2Gym/tests/testN 的布局一致，web_ui.py --testdir <simdir> 直接能开）：
  <simdir>/stage0_extract/extract.json     blend 原始事实（bpy 提取）
  <simdir>/stage0_spec/task_spec.json      r2pear spec（机器人基座系）
  <simdir>/stage0_spec/demo.npz            演示：关节 q、夹爪 g、末端、物体参考轨迹（修复后）+ 有效帧掩码
  <simdir>/stage0_spec/decisions.json      每一个自动判断：选择、依据、是否兜底/警告
  <simdir>/stage0_spec/scene_parse.json    解析中间结果（形状特征等）
  <simdir>/stage1_validate/                静置 / 回放 / 源视频对照

具身形态（自动识别，b2s_config.json 的 "embodiment" 可强制）：
  robot_arm         blend 里有机械臂骨架（DROID）：关节真值 → FK 校验 → 直接回放
  human_hand        blend 里有人手骨架（Ego）：捏合点轨迹 → retarget.py（基座摆位 + IK）
  handheld_gripper  blend 里有手持夹爪（UMI）：夹爪位姿轨迹 → retarget.py
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import PY, Decisions, Extract, dump_json, load_config  # noqa: E402
import build_spec as BS  # noqa: E402
import demo as DM  # noqa: E402
import embodiment as EM  # noqa: E402
import parse_scene as PS  # noqa: E402


def suggest_port(simdir):
    """给 Web UI 建议一个端口。

    端口是**按测试目录固定分配**的（见根 README）：证据目录 test1→8091、test2→8092，
    新开 testN→809N。不在 evidence/testN 下的目录退回 8080。原来这里写死 8112，
    和 README 的分配表冲突，照着打印出来的命令跑会开到没登记过的端口上。
    """
    head = os.path.basename(os.path.dirname(os.path.abspath(simdir)))
    d = os.path.basename(os.path.abspath(simdir))
    for name in (head, d):
        if name.startswith("test") and name[4:].isdigit():
            return 8090 + int(name[4:])
    return 8080


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="缺省找 blend 同目录的 b2s_config.json")
    ap.add_argument("--videos", default=None, help="源视频目录（文件名 = 相机的 video_stream），缺省 = blend 同目录")
    ap.add_argument("--reextract", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--env-tests", action="store_true", help="再跑反向对照 / 扰动扫描 / 初始状态采样（约 2 分钟）")
    ap.add_argument("--n-init", type=int, default=50)
    a = ap.parse_args()
    bdir = os.path.dirname(os.path.abspath(a.blend))
    cfg = load_config(a.config or os.path.join(bdir, "b2s_config.json"))
    xd, sd = os.path.join(a.out, "stage0_extract"), os.path.join(a.out, "stage0_spec")
    os.makedirs(xd, exist_ok=True)
    os.makedirs(sd, exist_ok=True)
    xp = os.path.join(xd, "extract.json")
    if a.reextract or not os.path.exists(xp) or os.path.getmtime(xp) < os.path.getmtime(a.blend):
        print("== extract（bpy）")
        subprocess.run([PY, os.path.join(HERE, "extract_blend.py"), "--blend", a.blend, "--out", xp], check=True)
    ex = Extract(xp)
    D = Decisions()
    print("== 具身识别")
    emb = EM.detect(ex, cfg, D)
    robot = EM.extract_robot(ex, emb, D) if emb["type"] == "robot_arm" else None
    print("== 场景解析")
    import numpy as np
    ee_hint = np.array([robot["T_world_base"] @ T for T in robot["fk"]["pinch"]])[:, :3, 3] if robot else None
    scene = PS.parse(ex, emb, robot, cfg, D, ee_world=ee_hint)
    print("== 演示轨迹")
    demo = DM.build(ex, emb, robot, scene, cfg, D, bdir)
    n_trim = DM.trim_failed_attempts(demo, scene, demo["fps"], D)
    if n_trim:
        print(f"   [裁前段] 丢弃前 {n_trim} 帧「没夹住的试探」（成功抓取之前的那次夹空）")
    if robot is None:
        print("== retarget（无机器人关节：基座摆位 + IK）")
        import retarget as RT
        res = RT.retarget(demo["ee_world"], scene, cfg, D, moved=demo.get("moved", ()))
        scene["T_world_base"] = res["T_world_base"]
        demo["q"] = res["q"]
        q0 = res["q"][0]
    else:
        q0 = demo["q"][0]
    print("== spec")
    spec = BS.build(ex, scene, demo, q0, cfg, D, sd, a.blend, robot=robot, emb=emb)
    dump_json(spec, os.path.join(sd, "task_spec.json"))
    dump_json({k: v for k, v in scene.items() if k != "robot_set"}, os.path.join(sd, "scene_parse.json"))
    D.dump(os.path.join(sd, "decisions.json"))
    n_fb = sum(i["fallback"] for i in D.items)
    n_w = sum(i["warn"] for i in D.items)
    print(f"spec → {sd}/task_spec.json   决策 {len(D.items)} 条（兜底 {n_fb}、警告 {n_w}）")
    if not a.skip_validate:
        print("== validate")
        subprocess.run([PY, os.path.join(HERE, "validate.py"), "--simdir", a.out, "--videos", a.videos or bdir],
                       check=True)
    on = [p for p in spec["success"] if p["pred"] == "on"]
    if a.env_tests and on:
        # r2pear 的环境级验收（与 test5/6 相同的三项）：反向对照 → 单项扰动扫描 → 初始状态采样
        R2P = os.path.dirname(HERE)
        sp = os.path.join(sd, "task_spec.json")
        obj, sup = on[0]["obj"], on[0]["support"]
        os.makedirs(os.path.join(a.out, "stage2_env"), exist_ok=True)
        os.makedirs(os.path.join(a.out, "stage3_init_states"), exist_ok=True)
        print("== 反向对照（含 oracle 正例）")
        subprocess.run([PY, os.path.join(R2P, "check_negatives.py"), "--spec", sp, "--obj", obj, "--support", sup,
                        "--out", os.path.join(a.out, "stage2_env", "negatives.json")], check=True,
                       stdout=subprocess.DEVNULL)
        print("== 单项扰动扫描 + 初始状态采样")
        subprocess.run([PY, os.path.join(R2P, "sweep_params.py"), "--testdir", a.out, "--obj", obj, "--support", sup],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run([PY, os.path.join(HERE, "init_dist.py"), "--simdir", a.out, "--n", str(a.n_init)], check=True)
        d3 = os.path.join(a.out, "stage3_init_states")
        subprocess.run([PY, os.path.join(R2P, "init_states.py"), "--spec", sp, "--dist",
                        os.path.join(d3, "init_distribution.json"), "--out", os.path.join(d3, "init_states.json")],
                       check=True, stdout=subprocess.DEVNULL)
        import json
        neg = json.load(open(os.path.join(a.out, "stage2_env", "negatives.json")))
        ist = json.load(open(os.path.join(d3, "init_states.json")))
        res = neg["cases"]
        print(f"   反向对照 {sum(v['pass'] for v in res.values())}/{len(res)} 通过；"
              f"初始状态 {len(ist['states'])} 个，验收率 {ist.get('acceptance_rate')}")
    elif a.env_tests:
        print("== 跳过环境级验收：谓词里没有 on(obj, support)，oracle 只支持 pick-place 模板")
    print(f"\nWeb UI:  MUJOCO_GL=egl {PY} {os.path.join(os.path.dirname(HERE), 'web_ui.py')} "
          f"--testdir {a.out} --port {suggest_port(a.out)}")


if __name__ == "__main__":
    main()
