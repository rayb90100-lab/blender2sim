"""Individual b2s stages.

Each function mutates one part of :class:`PipelineContext` and leaves the
existing domain implementation responsible for the actual conversion logic.
The stage order and subprocess commands match the original entry point.
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from common import PY, Decisions, Extract, dump_json, load_config
import build_spec as BS
import demo as DM
import embodiment as EM
import parse_scene as PS

from .models import PipelineContext, PipelineOptions

HERE = Path(__file__).resolve().parent.parent
R2P_ROOT = HERE.parent


def load_context(options: PipelineOptions) -> PipelineContext:
    """Create the workspace and load the extracted Blender facts."""
    options.validate()
    cfg_path = options.config or (options.blend_dir / "b2s_config.json")
    options.extract_dir.mkdir(parents=True, exist_ok=True)
    options.spec_dir.mkdir(parents=True, exist_ok=True)
    extractor = HERE / "extract_blend.py"
    needs_extract = (
        options.reextract
        or not options.extract_json.exists()
        or options.extract_json.stat().st_mtime < max(
            options.blend.stat().st_mtime, extractor.stat().st_mtime))
    if needs_extract:
        print("== extract（bpy）")
        subprocess.run(
            [PY, str(extractor), "--blend", str(options.blend), "--out", str(options.extract_json)],
            check=True,
        )
    return PipelineContext(
        options=options,
        cfg=load_config(str(cfg_path)),
        extract=Extract(str(options.extract_json)),
        decisions=Decisions(),
        outputs={"extract": options.extract_json},
    )


def identify_embodiment(ctx: PipelineContext) -> None:
    """Detect the source embodiment and run the DROID registration audit."""
    print("== 具身识别")
    ctx.embodiment = EM.detect(ctx.extract, ctx.cfg, ctx.decisions)
    if ctx.embodiment["type"] == "robot_arm":
        ctx.robot = EM.extract_robot(ctx.extract, ctx.embodiment, ctx.decisions)
    import registration
    ctx.registration_report = registration.audit_droid(
        ctx.extract, ctx.embodiment, ctx.robot, ctx.cfg)
    if ctx.registration_report["status"] == "not_applicable":
        return
    report_path = ctx.options.out / "registration-report.json"
    dump_json(ctx.registration_report, report_path)
    ctx.outputs["registration"] = report_path
    ctx.decisions.add(
        "registration",
        ctx.registration_report.get("recommendation", ctx.registration_report["status"]),
        "先按 DROID 实测法兰检查整体偏移，再检查扣除整体配准后的抓取残差；见 registration-report.json",
        warn=ctx.registration_report.get("warning", False),
    )
    print("   配准检查:", ctx.registration_report.get(
        "recommendation", ctx.registration_report["status"]))


def parse_scene(ctx: PipelineContext) -> None:
    """Parse the table, objects, cameras and static scene facts."""
    print("== 场景解析")
    ee_hint = None
    if ctx.robot is not None:
        ee_hint = np.array([
            ctx.robot["T_world_base"] @ T for T in ctx.robot["fk"]["pinch"]
        ])[:, :3, 3]
    ctx.scene = PS.parse(
        ctx.extract, ctx.embodiment, ctx.robot, ctx.cfg, ctx.decisions,
        ee_world=ee_hint)


def build_demo(ctx: PipelineContext) -> None:
    """Build the source trajectory, trim failed attempts and retarget if needed."""
    print("== 演示轨迹")
    ctx.demo = DM.build(
        ctx.extract, ctx.embodiment, ctx.robot, ctx.scene, ctx.cfg,
        ctx.decisions, str(ctx.options.blend_dir))
    n_trim = DM.trim_failed_attempts(
        ctx.demo, ctx.scene, ctx.demo["fps"], ctx.decisions)
    if n_trim:
        print(f"   [裁前段] 丢弃前 {n_trim} 帧「没夹住的试探」（成功抓取之前的那次夹空）")
    if ctx.robot is None:
        print("== retarget（无机器人关节：基座摆位 + IK）")
        import retarget as RT
        result = RT.retarget(
            ctx.demo["ee_world"], ctx.scene, ctx.cfg, ctx.decisions,
            moved=ctx.demo.get("moved", ()))
        ctx.scene["T_world_base"] = result["T_world_base"]
        ctx.demo["q"] = result["q"]
        ctx.q0 = result["q"][0]
    else:
        ctx.q0 = ctx.demo["q"][0]
    time_scale = float(ctx.cfg.get("retarget_time_scale", 1.0))
    if not np.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("retarget_time_scale must be finite and positive")
    if time_scale != 1.0:
        ctx.demo["source_fps"] = ctx.demo["fps"]
        ctx.demo["fps"] /= time_scale
        ctx.decisions.add(
            "demo.time_scale", time_scale,
            f"保持路径及开合采样，执行时间拉长 {time_scale:g} 倍；"
            f"源 {ctx.demo['source_fps']:g} fps → 回放 {ctx.demo['fps']:g} fps",
        )


def write_spec(ctx: PipelineContext) -> None:
    """Build and persist task_spec, scene_parse and decisions."""
    print("== spec")
    ctx.spec = BS.build(
        ctx.extract, ctx.scene, ctx.demo, ctx.q0, ctx.cfg, ctx.decisions,
        str(ctx.options.spec_dir), str(ctx.options.blend),
        robot=ctx.robot, emb=ctx.embodiment)
    dump_json(ctx.spec, ctx.options.spec_json)
    scene_path = ctx.options.spec_dir / "scene_parse.json"
    dump_json({k: v for k, v in ctx.scene.items() if k != "robot_set"}, scene_path)
    decisions_path = ctx.options.spec_dir / "decisions.json"
    ctx.decisions.dump(str(decisions_path))
    ctx.outputs.update(spec=ctx.options.spec_json, scene=scene_path, decisions=decisions_path)
    n_fb = sum(item["fallback"] for item in ctx.decisions.items)
    n_w = sum(item["warn"] for item in ctx.decisions.items)
    print(f"spec → {ctx.options.spec_dir}/task_spec.json   决策 {len(ctx.decisions.items)} 条（兜底 {n_fb}、警告 {n_w}）")


def validate(ctx: PipelineContext) -> None:
    """Run the existing V1/V2/V3 validation subprocess."""
    if ctx.options.skip_validate:
        return
    print("== validate")
    videos = ctx.options.videos or ctx.options.blend_dir
    subprocess.run(
        [PY, str(HERE / "validate.py"), "--simdir", str(ctx.options.out),
         "--videos", str(videos)],
        check=True,
    )


def run_environment_tests(ctx: PipelineContext) -> None:
    """Run optional negative cases, parameter sweeps and initial-state sampling."""
    if not ctx.options.env_tests:
        return
    on = [p for p in ctx.spec["success"] if p["pred"] == "on"]
    if not on:
        print("== 跳过环境级验收：谓词里没有 on(obj, support)，oracle 只支持 pick-place 模板")
        return
    R2P = str(R2P_ROOT)
    spec_path = str(ctx.options.spec_json)
    obj, support = on[0]["obj"], on[0]["support"]
    stage2 = ctx.options.out / "stage2_env"
    stage3 = ctx.options.out / "stage3_init_states"
    stage2.mkdir(exist_ok=True)
    stage3.mkdir(exist_ok=True)
    print("== 反向对照（含 oracle 正例）")
    subprocess.run(
        [PY, os.path.join(R2P, "check_negatives.py"), "--spec", spec_path,
         "--obj", obj, "--support", support, "--out", str(stage2 / "negatives.json")],
        check=True, stdout=subprocess.DEVNULL)
    print("== 单项扰动扫描 + 初始状态采样")
    subprocess.run(
        [PY, os.path.join(R2P, "sweep_params.py"), "--testdir", str(ctx.options.out),
         "--obj", obj, "--support", support],
        check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        [PY, str(HERE / "init_dist.py"), "--simdir", str(ctx.options.out),
         "--n", str(ctx.options.n_init)],
        check=True)
    subprocess.run(
        [PY, os.path.join(R2P, "init_states.py"), "--spec", spec_path,
         "--dist", str(stage3 / "init_distribution.json"),
         "--out", str(stage3 / "init_states.json")],
        check=True, stdout=subprocess.DEVNULL)
    neg = json.load(open(stage2 / "negatives.json"))
    states = json.load(open(stage3 / "init_states.json"))
    cases = neg["cases"]
    print(f"   反向对照 {sum(v['pass'] for v in cases.values())}/{len(cases)} 通过；"
          f"初始状态 {len(states['states'])} 个，验收率 {states.get('acceptance_rate')}")


def run_speed_diagnosis(ctx: PipelineContext) -> None:
    """Run the opt-in last-resort timing diagnosis after normal validation."""
    if ctx.options.skip_validate:
        return
    from speed_diagnosis import run_diagnosis
    run_diagnosis(str(ctx.options.out), ctx.options.last_resort_speed_diagnosis)
