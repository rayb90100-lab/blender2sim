#!/usr/bin/env python3
"""Export a b2s scene as a portable interactive environment and Web3D replay.

python tools/r2pear/export_scene.py --testdir runs/test1 --out outputs/test1-export \
    --scene-id b2s_test1 --dataset DROID --episode ep71 --title '方块搬放' \
    --viewer-assets local/export-vendor

No publication happens here. An incomplete/failed export must not be published.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "b2s"))
import droid_common as C
import sim as S
import spec as SP
import validate as V

Y_UP = np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(root, path):
    return {"path": Path(path).relative_to(root).as_posix(), "sha256": sha(path)}


class Recorder:
    """Read poses from a separate MjData: recording must not change the rollout."""
    def __init__(self):
        self.rows = []

    def __call__(self, frame, model, data):
        import mujoco
        if not self.rows:
            self.model = model
            self.scratch = mujoco.MjData(model)
            self.state_type = mujoco.mjtState.mjSTATE_INTEGRATION
            self.size = mujoco.mj_stateSize(model, self.state_type)
        state = np.empty(self.size)
        mujoco.mj_getState(model, data, state, self.state_type)
        mujoco.mj_setState(model, self.scratch, state, self.state_type)
        mujoco.mj_forward(model, self.scratch)
        d = self.scratch
        self.rows.append({"time": float(d.time), "integration_state": state,
                          "qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "ctrl": d.ctrl.copy(),
                          "geom_xpos": d.geom_xpos.copy(),
                          "geom_quat_xyzw": Rotation.from_matrix(d.geom_xmat.reshape(-1, 3, 3)).as_quat(),
                          "body_xpos": d.xpos.copy(), "body_quat_wxyz": d.xquat.copy()})

    def arrays(self):
        a = {key: np.asarray([r[key] for r in self.rows]) for key in self.rows[0]}
        a["timestamps_seconds"] = a["time"] - a["time"][0]
        return a


def matrix(position, quat_xyzw):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    t[:3, 3] = position
    return t


def initial_pose(trace, model, spec):
    bodies = {f"geom{i}": matrix(trace["geom_xpos"][0, i], trace["geom_quat_xyzw"][0, i]).ravel().tolist()
              for i in range(model.ngeom)}
    objects = {}
    for name in spec["objects"]:
        bid = model.body(name).id
        q = trace["body_quat_wxyz"][0, bid]
        objects[name] = matrix(trace["body_xpos"][0, bid], q[[1, 2, 3, 0]]).ravel().tolist()
    return {"bodies": bodies, "state": {"q": trace["qpos"][0, C.arm_qadr(model)].tolist(),
            "grip": float(trace["qpos"][0, model.joint(C.GRIP_PREFIX + "right_driver_joint").qposadr[0]]),
            "objects": objects}}


def copy_runtime(bundle, src, spec):
    runtime = bundle / "interactive"
    tools = runtime / "tools/r2pear"
    (tools / "b2s").mkdir(parents=True)
    modules = ["web_ui.py", "env.py", "sim.py", "spec.py", "controller.py", "predicates.py", "oracle.py",
               "blender_live.py", "blender_worker.py", "environment_prompt.md"]
    for name in modules:
        p = HERE / name
        if p.exists():
            shutil.copy2(p, tools / name)
    for name in ["blend_worker.py", "export_display.py", "object_motion.py"]:
        shutil.copy2(HERE / "b2s" / name, tools / "b2s" / name)
    shutil.copy2(HERE.parent / "droid_common.py", runtime / "tools/droid_common.py")
    shutil.copy2(HERE / "export_runtime.py", runtime / "run.py")
    for family in ["franka_emika_panda", "robotiq_2f85"]:
        shutil.copytree(Path(C.MENAGERIE) / family, runtime / "assets/mujoco_menagerie" / family,
                        ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"))
    stage = runtime / "scene/stage0_spec"
    stage.mkdir(parents=True)
    (runtime / "source").mkdir()
    shutil.copy2(spec["source"]["blend"], runtime / "source/scene.blend")
    portable = copy.deepcopy(spec)
    portable.pop('experimental_preview', None)
    portable["source"]["blend"] = "../../source/scene.blend"
    if isinstance(portable["source"].get("dataset"), dict):
        portable["source"]["dataset"].pop("path", None)
    dump(stage / "task_spec.json", portable)
    shutil.copy2(src / "stage0_spec/demo.npz", stage / "demo.npz")
    (runtime / "requirements.txt").write_text(
        "mujoco==3.3.7\nnumpy==1.26.4\nscipy==1.17.1\npillow==12.3.0\nopencv-python-headless==4.11.0.86\n")
    (runtime / "requirements-blender.txt").write_text("-r requirements.txt\nbpy==4.5.4\n")
    return runtime


def export_mjcf(runtime, spec, initial):
    m, d, ms = S.build(spec, return_spec=True)
    # Preserve the explicit XML's precision; re-serializing a compiled spec can
    # round source poses and alter replay contacts.
    tree = (ET.parse(spec['model_xml']['path']).getroot() if spec.get('model_xml')
            else ET.fromstring(ms.to_xml()))
    folder = runtime / "model"
    folder.mkdir()
    compiler = tree.find("compiler")
    compiler.set("meshdir", "assets")
    compiler.set("texturedir", "assets")
    for element in tree.iter():
        name = element.get("file")
        if not name:
            continue
        family = "robotiq_2f85" if element.get("name", "").startswith(C.GRIP_PREFIX) else "franka_emika_panda"
        source = Path(name)
        if not source.is_absolute():
            source = Path(C.MENAGERIE) / family / "assets" / name
        if not source.is_file():
            raise FileNotFoundError(f"Cannot package MJCF asset: {source}")
        # Convex decompositions from different objects commonly share filenames.
        filename = sha(source)[:16] + '-' + source.name
        dest = folder / "assets" / family / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha(dest) != sha(source):
            raise ValueError(f"Asset name collision: {dest}")
        shutil.copy2(source, dest)
        element.set("file", f"{family}/{filename}")
    # A native viewer can reset to this keyframe without knowing our Python API.
    keyframe = tree.find("keyframe")
    if keyframe is not None:
        tree.remove(keyframe)
    keyframe = ET.SubElement(tree, "keyframe")
    ET.SubElement(keyframe, "key", name="task_initial", time=str(float(initial["time"][0])),
                  qpos=" ".join(map(str, initial["qpos"][0])),
                  qvel=" ".join(map(str, initial["qvel"][0])),
                  ctrl=" ".join(map(str, initial["ctrl"][0])))
    ET.indent(tree)
    ET.ElementTree(tree).write(folder / "scene.xml", encoding="unicode")
    if spec.get('model_xml'):
        path = runtime / 'scene/stage0_spec/task_spec.json'
        portable = json.loads(path.read_text())
        portable['model_xml'] = {'path': '../../model/scene.xml', 'sha256': sha(folder/'scene.xml')}
        dump(path, portable)


def export_replay(bindings, trace, model, spec):
    base = np.asarray(spec["robot"]["T_world_base"])
    channels = []
    for binding in bindings:
        if binding["kind"] == "geom":
            i = binding["id"]
            positions, quats = trace["geom_xpos"][:, i], trace["geom_quat_xyzw"][:, i]
        else:
            i = model.body(binding["name"]).id
            positions = trace["body_xpos"][:, i]
            quats = trace["body_quat_wxyz"][:, i][:, [1, 2, 3, 0]]
        transforms = np.repeat(np.eye(4)[None], len(positions), axis=0)
        transforms[:, :3, :3] = Rotation.from_quat(quats).as_matrix()
        transforms[:, :3, 3] = positions
        transforms = Y_UP @ base @ transforms @ Y_UP.T
        rotations = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
        for i in range(1, len(rotations)):
            if np.dot(rotations[i - 1], rotations[i]) < 0:
                rotations[i] *= -1
        channels.append({"node": binding["node"], "logical_body_id": binding.get("name", binding["node"]),
                         "translation": transforms[:, :3, 3].tolist(), "rotation_xyzw": rotations.tolist()})
    times = trace["timestamps_seconds"].tolist()
    return {"schema": "real2gym.current_native_replay.v1", "timestamps_seconds": times,
            "frame_count": len(times), "duration_seconds": times[-1], "interpolation": "step",
            "coordinate_system": "glTF_Y_up_meters", "transform_scope": "Independent scene-root local transforms",
            "native_indices": list(range(len(times))), "native_sample_times_seconds": times,
            "channels": channels}


def check_display(glb, replay):
    with glb.open("rb") as f:
        magic, version, size = struct.unpack("<4sII", f.read(12))
        n, kind = struct.unpack("<I4s", f.read(8))
        doc = json.loads(f.read(n))
    assert magic == b"glTF" and version == 2 and size == glb.stat().st_size and kind == b"JSON"
    roots = set(doc["scenes"][doc.get("scene", 0)]["nodes"])
    nodes = {node.get("name"): (i, node) for i, node in enumerate(doc["nodes"])}
    times = np.asarray(replay["timestamps_seconds"])
    assert times[0] == 0 and np.all(np.diff(times) > 0)
    for c in replay["channels"]:
        i, node = nodes[c["node"]]
        assert i in roots, f"Animated node is not a root: {c['node']}"
        p, q = np.asarray(c["translation"]), np.asarray(c["rotation_xyzw"])
        assert p.shape == (len(times), 3) and q.shape == (len(times), 4)
        assert np.isfinite(p).all() and np.isfinite(q).all()
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-6)
        actual = (np.asarray(node["matrix"]).reshape(4, 4).T if "matrix" in node else
                  matrix(node.get("translation", [0, 0, 0]), node.get("rotation", [0, 0, 0, 1])))
        np.testing.assert_allclose(actual, matrix(p[0], q[0]), atol=3e-5)
    return {"ok": True, "nodes": len(nodes), "channels": len(replay["channels"]), "samples": len(times)}


def export_video(bundle, trace, model, spec, fps=15, camera_name="orbit"):
    import mujoco
    from PIL import Image
    d = mujoco.MjData(model)
    mujoco.mj_setState(model, d, trace["integration_state"][0], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, d)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    look = np.mean([o["center"] for o in spec["objects"].values()], axis=0)
    look[2] += .03
    camera.lookat[:] = look
    eye = d.cam_xpos[model.camera("agentview").id]
    camera.azimuth = float(np.degrees(np.arctan2(eye[1] - look[1], eye[0] - look[0])))
    camera.elevation, camera.distance = -32, 0.8
    if camera_name != "orbit":
        if camera_name not in spec["cameras"] or spec["cameras"][camera_name]["mount"] != "world":
            raise ValueError("Export camera must be orbit or a fixed world camera")
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = model.camera(camera_name).id
    w, h = 640, 360
    video = bundle / "mujoco-sim.mp4"
    stderr = (bundle / "ffmpeg.log").open("w")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                          "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
                         stdin=subprocess.PIPE, stderr=stderr)
    renderer = mujoco.Renderer(model, h, w)
    count = math.ceil(trace["timestamps_seconds"][-1] * fps) + 1
    try:
        for frame in range(count):
            t = min(frame / fps, trace["timestamps_seconds"][-1])
            i = int(np.searchsorted(trace["timestamps_seconds"], t, side="right") - 1)
            mujoco.mj_setState(model, d, trace["integration_state"][i], mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(model, d)
            renderer.update_scene(d, camera=camera)
            # MuJoCo azimuth points opposite to the eye-to-target direction.
            # Export the actual rendered camera rather than re-deriving it.
            actual_eye = np.mean([renderer.scene.camera[i].pos.copy() for i in (0, 1)], axis=0)
            rendered_camera = renderer.scene.camera[0]
            actual_forward = rendered_camera.forward.copy()
            actual_up = rendered_camera.up.copy()
            fovy = math.degrees(2 * math.atan(
                (rendered_camera.frustum_top - rendered_camera.frustum_bottom) /
                (2 * rendered_camera.frustum_near)))
            pixels = renderer.render()
            if float(np.max(pixels.std(axis=(0, 1)))) < 2:
                raise RuntimeError("Export camera sees a nearly uniform image; choose a visible world camera with --camera")
            if frame == 0:
                Image.fromarray(pixels).save(bundle / "mujoco-poster.png")
            if frame == count - 1:
                Image.fromarray(pixels).save(bundle / "mujoco-final.png")
            p.stdin.write(pixels.tobytes())
            if frame % 30 == 0:
                print(f"Video frame {frame + 1}/{count}", flush=True)
    finally:
        renderer.close()
        p.stdin.close()
        code = p.wait()
        stderr.close()
    if code:
        raise RuntimeError("ffmpeg failed; see ffmpeg.log")
    transform = Y_UP @ np.asarray(spec["robot"]["T_world_base"])
    point = lambda p: (transform @ np.r_[p, 1])[:3].tolist()
    up = (transform[:3, :3] @ actual_up).tolist()
    target = actual_eye + actual_forward * float(np.linalg.norm(actual_eye - look))
    return {"eye": point(actual_eye), "target": point(target), "up": up,
            "fovy_deg": fovy, "source_camera": camera_name}, {"frames": count, "fps": fps}


def verify_relocated(bundle, python):
    # A new filesystem path and unrelated cwd catch hidden absolute-path dependencies.
    with tempfile.TemporaryDirectory(prefix="b2s-export-check-") as tmp:
        dest = Path(tmp) / "relocated environment"
        shutil.copytree(bundle / "interactive", dest)
        env = dict(os.environ)
        for k in ["MENAGERIE", "ROBOLAB_DATA", "PEAR_CODE", "PYTHONPATH"]:
            env.pop(k, None)
        result = subprocess.run([python, str(dest / "run.py"), "--check", "--software-render"],
                                cwd=tmp, env=env, capture_output=True, text=True, timeout=120)
        (bundle / "relocation.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("Relocated runtime check failed; see relocation.log")
        return json.loads(result.stdout.strip().splitlines()[-1])


def write_readme(bundle, spec):
    (bundle / "README.txt").write_text(f"""{spec['instruction']}

展示回放：
  python -m http.server 8095 --bind 127.0.0.1 --directory .
  浏览器访问 http://localhost:8095/preview.html。远程机器请转发对应端口。
  GLB + replay JSON 与 real2gym.current_native_replay.v1 格式对齐。
  回放只播放已记录位姿；视频和轨迹来自同一次 MuJoCo 演示回放，包含结束后的稳定阶段。

实时交互（推荐 Python 3.11）：
  cd interactive
  python3.11 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  .venv/bin/python run.py --port 8094
  浏览器访问 http://localhost:8094，远程机器请转发对应端口。
  Linux 无 GPU 时可加 --software-render，需要系统提供 Mesa/EGL。
  可控制机器人、开合夹爪、重置、执行配置的自动动作（原始轨迹回放或 oracle），物理与任务判定实时运行。
  若要开启原 Blender 视图：pip install -r requirements-blender.txt。
  环境迁移自检：python run.py --check --software-render。

原生 MuJoCo：
  model/scene.xml 及其 assets 位于 interactive/ 下，可直接加载。
  加载后用 mj_resetDataKeyframe(model, data, model.key('task_initial').id) 恢复任务初态。
  控制器、动作协议和成功判定由随包 Python 代码提供，XML 本身不包含这些任务逻辑。

内容与证据：
  scene.json：展示文件索引；interactive/ 与 interactive-environment.tar.gz 内容相同。
  interactive/scene/stage0_spec/：任务配置、原示范关节轨迹。
  interactive/scene/native_trace.npz：此次物理回放采样的状态和位姿。
  interactive/source/scene.blend：原场景；本包使用相对路径。
  validation.json：静置、原示范回放、GLB 绑定及迁移交互测试结果。
  checksums.json：文件 SHA256。此包不包含虚拟环境或完整原始数据集。

限制：碰撞体采用当前管线的几何近似；成功不代表无穿透。
  请查看 validation.json 的 min_contact_dist_mm。test2 杯碟为薄圆柱近似，杯把仅作视觉跟随。
  展示 GLB 的机器人使用 MuJoCo 刚性网格，背景和任务物体使用原 Blender 几何及可导出的材质。
  glTF 不支持所有 Blender 程序化材质，外观可能与原 Blender 渲染不同。
  MuJoCo 观测不含完整实验室背景；交互 Blender 视图保留原场景。

来源：输入的 Blender 场景与 blender2sim 管线；具体数据集见 scene.json，文件校验见 provenance.json。
  如有 SCENE_NOTES.txt，请同时阅读其中的来源、场景适配和验证边界。
  Panda/Robotiq 资产来自 MuJoCo Menagerie。
  模型许可证保留在 interactive/assets/mujoco_menagerie/ 各目录。
""", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testdir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--viewer-assets", required=True, type=Path, help="Pinned Three.js r170 files (see export README)")
    ap.add_argument("--scene-notes", type=Path, help="Scene provenance and limitations, included in both delivery and runtime")
    ap.add_argument("--camera", default="orbit", help="Video/Web3D camera: orbit or a named fixed world camera")
    a = ap.parse_args()
    if a.scene_notes is not None and not a.scene_notes.is_file():
        raise SystemExit(f"Scene notes not found: {a.scene_notes}")
    src, out = a.testdir.resolve(), a.out.resolve()
    if out.exists():
        raise SystemExit(f"Output exists; choose a new directory: {out}")
    for key in os.environ:
        if key.startswith("R2P_") and key in {"R2P_TIMESTEP", "R2P_SOLREF", "R2P_SOLIMP", "R2P_NOSLIP"}:
            raise SystemExit(f"Unset physics override before exporting: {key}")
    spec = SP.load(src / "stage0_spec/task_spec.json")
    from export_assets.fetch_viewer import check as check_viewer
    check_viewer(a.viewer_assets)
    if not spec.get("blend_binding"):
        raise SystemExit("This exporter requires a b2s scene with blend_binding")
    out.mkdir(parents=True)
    dump(out / "export-status.json", {"complete": False})
    runtime = copy_runtime(out, src, spec)
    if a.scene_notes is not None:
        shutil.copy2(a.scene_notes, out / "SCENE_NOTES.txt")
        shutil.copy2(a.scene_notes, runtime / "SCENE_NOTES.txt")
    print("Recording physics...", flush=True)
    recorder = Recorder()
    with np.load(src / "stage0_spec/demo.npz") as data:
        v2, _ = V.v2_replay(spec, dict(data), on_frame=recorder, capture_settle=True)
    trace = recorder.arrays()
    np.savez_compressed(runtime / "scene/native_trace.npz", **trace)
    with np.load(src / 'stage0_spec/demo.npz') as data:
        v1 = V.v1_settle(spec, demo=dict(data))
    if not v1["ok"] or not v2["success"]:
        dump(out / "validation.json", {"V1_settle": v1, "V2_replay": v2})
        raise RuntimeError("Scene validation failed; refusing a successful export status")
    pose = initial_pose(trace, recorder.model, spec)
    dump(out / "initial-pose.json", pose)
    export_mjcf(runtime, spec, trace)
    # Keep pre-correction failures and explicit adjustments reviewable in the
    # artifact; a stable normalized initial state must not erase source drift.
    for name in ('initialization-report.json', 'registration-report.json'):
        if (src/name).is_file():
            shutil.copy2(src/name, out/name)
            shutil.copy2(src/name, runtime/'scene'/name)
    print("Exporting original Blender surroundings and bound rigid meshes...", flush=True)
    glb = out / "mujoco-web3d-scene.glb"
    with (out / "blender-export.log").open("w") as log:
        subprocess.run([sys.executable, str(runtime / "tools/r2pear/b2s/blend_worker.py"),
                        "--testdir", str(runtime / "scene"), "--export-glb", str(glb),
                        "--state-json", str(out / "initial-pose.json")],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
    bindings = json.loads(Path(str(glb) + ".bindings.json").read_text())
    replay = export_replay(bindings, trace, recorder.model, spec)
    replay["identity"] = {"scene_id": a.scene_id, "backend": "mujoco", "candidate_id": "validated_export"}
    dump(out / "mujoco-web3d-replay.json", replay)
    display_check = check_display(glb, replay)
    print("Rendering video from the recorded states...", flush=True)
    camera, video = export_video(out, trace, recorder.model, spec, camera_name=a.camera)
    manifest = {"schema": "real2gym.current_native_web3d_manifest.v1", "identity": replay["identity"],
                "scene_ref": ref(out, glb), "replay_ref": ref(out, out / "mujoco-web3d-replay.json"),
                "native_trace_ref": ref(out, runtime / "scene/native_trace.npz"),
                "frame_count": replay["frame_count"], "duration_seconds": replay["duration_seconds"],
                "camera": camera, "display_only": True, "browser_physics": False,
                "interactive_runtime": "interactive/run.py", "binding_rows": bindings,
                "source_grade": "automated_physics_and_portability_validated", "video": video}
    dump(out / "mujoco-web3d-manifest.json", manifest)
    (out / "viewer").mkdir()
    for name in ["three.module.js", "GLTFLoader.js", "OrbitControls.js", "BufferGeometryUtils.js", "LICENSE"]:
        shutil.copy2(a.viewer_assets / name, out / "viewer" / name)
    # Preserve upstream source files except this relative addon import.
    p = out / "viewer/GLTFLoader.js"
    p.write_text(p.read_text().replace("../utils/BufferGeometryUtils.js", "./BufferGeometryUtils.js"))
    shutil.copy2(HERE / "export_assets/preview.html", out / "preview.html")
    write_readme(out, spec)
    print("Checking relocated interactive bundle...", flush=True)
    portability = verify_relocated(out, sys.executable)
    validation = {"V1_settle": v1, "V2_replay": v2, "display": display_check, "relocation": portability}
    dump(out / "validation.json", validation)
    shutil.copy2(out / "validation.json", runtime / "validation.json")
    shutil.copy2(out / "README.txt", runtime / "README.txt")
    # The Blender worker imports runtime modules during export. Its bytecode
    # caches contain machine-local filenames and do not belong in a delivery.
    for cache in runtime.rglob("__pycache__"):
        shutil.rmtree(cache)
    with tarfile.open(out / "interactive-environment.tar.gz", "w:gz") as tar:
        tar.add(runtime, arcname="interactive")
    files = {key: {"file": name, "sha256": sha(out / name)} for key, name in {
        "mujoco-web3d-scene": glb.name, "mujoco-web3d-replay": "mujoco-web3d-replay.json",
        "mujoco-web3d-manifest": "mujoco-web3d-manifest.json", "mujoco-sim": "mujoco-sim.mp4",
        "mujoco-poster": "mujoco-poster.png", "interactive-environment": "interactive-environment.tar.gz",
        "validation": "validation.json"}.items()}
    dump(out / "scene.json", {"scene_id": a.scene_id, "dataset": a.dataset, "task": a.title,
                              "episode_label": a.episode, "episode_label_scope": "source dataset episode",
                              "source_scope": "blender2sim validated native replay and interactive runtime", "files": files})
    dump(out / "provenance.json", {"task_spec_sha256": sha(src / "stage0_spec/task_spec.json"),
                                   "demo_sha256": sha(src / "stage0_spec/demo.npz"),
                                   "source_blend_sha256": sha(spec["source"]["blend"]),
                                   "exporter_sha256": sha(__file__), "viewer_version": "three.js r170"})
    dump(out / "export-status.json", {"complete": True, "validated": True})
    dump(out / "checksums.json", {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
                                  if p.is_file() and p.name != "checksums.json"})
    print(json.dumps({"output": str(out), "success": True, "samples": len(trace["time"]),
                      "automatic_steps": portability.get('automatic_steps', portability.get('oracle_steps'))}), flush=True)


if __name__ == "__main__":
    main()
