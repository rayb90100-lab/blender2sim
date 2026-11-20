"""Portable exports: coordinate convention, node bindings and non-mutating capture."""
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import export_scene as EX
import spec as SP


def test_explicit_mjcf_packages_same_named_convex_assets_and_uses_relative_model(tmp_path):
    import hashlib
    import mujoco
    sources = []
    for folder, extent in [('a', 1), ('b', 2)]:
        path = tmp_path/folder/'convex-000.obj'
        path.parent.mkdir()
        path.write_text(f'v 0 0 0\nv {extent} 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n')
        sources.append(path)
    xml = tmp_path/'source.xml'
    xml.write_text(f'''<mujoco><compiler/><asset>
        <mesh name="a" file="{sources[0]}"/><mesh name="b" file="{sources[1]}"/>
        </asset><worldbody><geom type="mesh" mesh="a"/><geom type="mesh" mesh="b" pos="3 0 0"/></worldbody></mujoco>''')
    spec = {'model_xml': {'path': str(xml), 'sha256': hashlib.sha256(xml.read_bytes()).hexdigest()}}
    runtime = tmp_path/'runtime'; stage = runtime/'scene/stage0_spec'; stage.mkdir(parents=True)
    EX.dump(stage/'task_spec.json', spec)
    model = mujoco.MjModel.from_xml_path(str(xml)); data = mujoco.MjData(model)
    EX.export_mjcf(runtime, spec, {'time': [0.], 'qpos': [data.qpos], 'qvel': [data.qvel], 'ctrl': [data.ctrl]})
    files = list((runtime/'model/assets').rglob('*.obj'))
    assert len(files) == 2 and files[0].name != files[1].name
    portable = json.loads((stage/'task_spec.json').read_text())['model_xml']
    assert portable['path'] == '../../model/scene.xml'
    assert portable['sha256'] == EX.sha(runtime/'model/scene.xml')
    relocated = mujoco.MjModel.from_xml_path(str(runtime/'model/scene.xml'))
    assert relocated.nmesh == 2


def test_source_replay_qualification_cannot_be_substituted_with_oracle_success():
    from export_assets.publish import automatic_succeeded
    assert not automatic_succeeded({'automatic_kind': 'source_demo', 'oracle_success': True,
                                    'automatic_success': True, 'source_replay': {'complete': False, 'success': True}})
    assert automatic_succeeded({'automatic_kind': 'source_demo', 'automatic_success': True,
                                 'source_replay': {'complete': True, 'success': True}})


def test_recording_does_not_advance_or_modify_physics():
    import mujoco
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    d = mujoco.MjData(m)
    for _ in range(20):
        mujoco.mj_step(m, d)
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    before = np.empty(mujoco.mj_stateSize(m, kind))
    after = np.empty_like(before)
    mujoco.mj_getState(m, d, before, kind)
    recorder = EX.Recorder()
    recorder(0, m, d)
    mujoco.mj_getState(m, d, after, kind)
    np.testing.assert_array_equal(before, after)
    assert recorder.arrays()["timestamps_seconds"][0] == 0
    np.testing.assert_array_equal(recorder.arrays()["qpos"][0], d.qpos)


def test_replay_converts_base_translation_and_z_up_rotation():
    # Native +z becomes glTF +y, native +y becomes glTF -z.
    trace = {"geom_xpos": np.array([[[1., 2, 3]], [[1., 2, 4]]]),
             "geom_quat_xyzw": np.array([[[0., 0, 0, 1]], [[0., 0, 1, 0]]]),
             "timestamps_seconds": np.array([0., 1.])}
    base = np.eye(4)
    base[:3, 3] = [10, 20, 30]
    r = EX.export_replay([{"kind": "geom", "id": 0, "node": "robot"}], trace, None,
                         {"robot": {"T_world_base": base.tolist()}})
    channel = r["channels"][0]
    np.testing.assert_allclose(channel["translation"], [[11, 33, -22], [11, 34, -22]])
    rotation = EX.Rotation.from_quat(channel["rotation_xyzw"][1])
    np.testing.assert_allclose(rotation.apply([1, 0, 0]), [-1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(rotation.apply([0, 1, 0]), [0, 1, 0], atol=1e-12)


def test_display_validation_rejects_nested_animation_root(tmp_path):
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": [{"name": "parent", "children": [1]}, {"name": "moving"}]}
    b = json.dumps(doc).encode()
    b += b" " * (-len(b) % 4)
    p = tmp_path / "scene.glb"
    p.write_bytes(struct.pack("<4sII", b"glTF", 2, 20 + len(b)) + struct.pack("<I4s", len(b), b"JSON") + b)
    replay = {"timestamps_seconds": [0, 1], "channels": [{"node": "moving", "translation": [[0, 0, 0]]*2,
                                                         "rotation_xyzw": [[0, 0, 0, 1]]*2}]}
    with pytest.raises(AssertionError, match="not a root"):
        EX.check_display(p, replay)


def test_relative_blend_path_is_resolved_against_spec_not_cwd(tmp_path, monkeypatch):
    doc = {"schema_version": 1, "source": {"blend": "../../source/scene.blend"},
           "objects": {}, "cameras": {}, "success": []}
    p = tmp_path / "scene/stage0_spec/task_spec.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(doc))
    monkeypatch.chdir("/tmp")
    assert SP.load(p)["source"]["blend"] == str(tmp_path / "source/scene.blend")


def test_publication_ignores_generated_bytecode_but_rejects_changed_assets(tmp_path):
    from export_assets import publish
    EX.dump(tmp_path / "export-status.json", {"complete": True, "validated": True})
    EX.dump(tmp_path / "validation.json", {"V1_settle": {"ok": True}, "V2_replay": {"success": True},
            "display": {"ok": True}, "relocation": {"ok": True, "oracle_success": True}})
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"validated model")
    EX.dump(tmp_path / "checksums.json", {p.name: publish.digest(p) for p in tmp_path.iterdir()})
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "runtime.cpython-311.pyc").write_bytes(b"generated by running the bundle")
    publish.verify(tmp_path)
    asset.write_bytes(b"changed model")
    with pytest.raises(AssertionError, match="Checksum mismatch"):
        publish.verify(tmp_path)


def test_publication_never_overwrites_existing_scene(tmp_path, monkeypatch):
    from export_assets import publish
    target = tmp_path / "existing-scene"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("existing scene")
    monkeypatch.setattr(sys, "argv", ["publish.py", "--bundle", str(tmp_path / "unused"),
                                     "--browser-report", str(tmp_path / "unused.json"),
                                     "--destination", str(target)])
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        publish.main()
    assert marker.read_text() == "existing scene"
