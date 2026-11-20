"""Real physics regression: a harmless-looking proxy overlap moves the target at reset."""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate


def scene(monkeypatch, neighbor_x):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option gravity="0 0 0" timestep=".002"/>
      <worldbody>
        <body name="target"><freejoint name="target_free"/>
          <geom name="target_surface" type="sphere" size=".01" mass=".1"/></body>
        <body name="neighbor" pos="{neighbor_x} 0 0">
          <geom name="neighbor_proxy" type="sphere" size=".01"/></body>
      </worldbody></mujoco>''')
    monkeypatch.setattr(validate.S, 'build', lambda spec: (model, mujoco.MjData(model)))
    def reset(m, d, spec, settle_s=0):
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
    monkeypatch.setattr(validate.S, 'reset', reset)
    return {'objects': {'target': {}}, 'robot': {'T_world_base': np.eye(4)}}


def test_initial_overlap_reports_cause_and_horizontal_motion(monkeypatch):
    report = validate.v1_settle(scene(monkeypatch, .015), seconds=.1)
    assert not report['ok']
    assert report['initial_overlaps'][0]['bodies'] == ['target', 'neighbor']
    assert abs(report['initial_overlaps'][0]['dist_mm'] + 5) < 1e-8
    obj = report['objects']['target']
    assert obj['delta_xyz_mm'][0] < -2
    assert obj['horizontal_drift_mm'] > 2
    assert obj['peak_drift_mm'] >= obj['drift_mm']
    np.testing.assert_allclose(obj['initial_position_base_m'], [0, 0, 0])


def test_stable_scene_preserves_source_to_compiled_offset(monkeypatch):
    spec = scene(monkeypatch, .025)
    source = np.eye(4)[None]
    source[0, 0, 3] = .023
    report = validate.v1_settle(spec, seconds=.1, demo={'obj__target': source})
    assert report['ok']
    assert not report['initial_overlaps']
    obj = report['objects']['target']
    assert obj['drift_mm'] == 0
    np.testing.assert_allclose(obj['source_reference']['source_to_initial_xyz_mm'], [-23, 0, 0])
    # A stable compiled state does not prove faithful source placement.
    np.testing.assert_allclose(obj['source_reference']['source_to_final_xyz_mm'], [-23, 0, 0])
