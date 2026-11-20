"""Object hierarchy input: FK, source alignment and keyed visual transforms."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Extract, Decisions
import embodiment as EM
import sources
import demo
from object_motion import interpolate_pose, gripper_binding


def fixture(tmp_path):
    q = np.tile([0., -.5, 0., -2.3, 0., 1.6, .8], (5, 1))
    q += np.arange(5)[:, None]*np.arange(1, 8)[None, :]*.003
    fk = EM._mj_fk(q)
    objects = {'base': {'type': 'MESH', 'parent': None, 'matrix_world': np.eye(4).tolist()}}
    basis, dynamic = {}, {}
    for i in range(7):
        name = f'joint{i}'
        objects[name] = {'type': 'EMPTY', 'parent': f'joint{i-1}' if i else 'base',
                         'matrix_world': fk[i+1][0].tolist(),
                         'props': {'source_field': 'observation.joint_position', 'source_column_zero_based': i}}
        mats = np.tile(np.eye(4), (5, 1, 1))
        mats[:, :3, :3] = Rotation.from_rotvec(q[:, i, None]*np.array([0, 0, 1])).as_matrix()
        basis[name] = mats.tolist()
        dynamic[name] = fk[i+1].tolist()
    path = tmp_path/'extract.json'
    path.write_text(json.dumps({'objects': objects, 'dynamic': dynamic, 'object_basis': basis,
                               'armatures': {}, 'scene': {'fps': 15}, 'frames': list(range(1, 6))}))
    return Extract(path), q


def test_object_chain_uses_shared_fk_and_preserves_joints(tmp_path):
    ex, q = fixture(tmp_path)
    emb = EM.detect(ex, {}, Decisions())
    assert emb['type'] == 'robot_arm' and emb['representation'] == 'object_joints'
    robot = EM.extract_robot(ex, emb, Decisions())
    assert robot['fk_ok']
    np.testing.assert_allclose(robot['q'], q, atol=1e-10)


def test_object_chain_rejects_duplicate_columns_and_wrong_hierarchy(tmp_path):
    ex, _ = fixture(tmp_path)
    ex.O['joint6']['props']['source_column_zero_based'] = 5
    with pytest.raises(ValueError, match='unique source columns'):
        EM.detect(ex, {}, Decisions())
    ex.O['joint6']['props']['source_column_zero_based'] = 6
    ex.O['joint3']['parent'] = 'base'
    with pytest.raises(ValueError, match='hierarchy'):
        EM.detect(ex, {}, Decisions())


def test_object_chain_rejects_moving_joint_origin(tmp_path):
    ex, _ = fixture(tmp_path)
    emb = EM.detect(ex, {}, Decisions())
    ex.E['object_basis']['joint2'][2][0][3] += .01
    with pytest.raises(ValueError, match='moving joint origins'):
        EM.extract_robot(ex, emb, Decisions())


def test_droid_npz_checks_joints_time_and_preserves_commands(tmp_path):
    ex, q = fixture(tmp_path)
    path = tmp_path/'trajectory.npz'
    g = np.array([0, .2, .6, .6, 0.])
    command = np.array([0, .5, .9, .1, 0.])
    np.savez(path, **{'observation.joint_position': q, 'observation.gripper_position': g[:, None],
                     'action_dict.gripper_position': command[:, None], 'frame_index': np.arange(5),
                     'timestamp_s_nominal': np.arange(5)/15})
    ep, _ = sources.load_droid_npz(path, q, ex.frames, 15)
    np.testing.assert_array_equal(ep['action.gripper_position'], command)
    with pytest.raises(ValueError, match='joints differ'):
        sources.load_droid_npz(path, q+.01, ex.frames, 15)
    with pytest.raises(ValueError, match='timebase'):
        sources.load_droid_npz(path, q, ex.frames, 30)
    with pytest.raises(ValueError, match='frame_index'):
        sources.load_droid_npz(path, q, np.arange(5), 15)
    ex.E['props_anim'] = {'flange': {'observation.gripper_position': g.tolist()}}
    np.testing.assert_array_equal(demo.blend_gripper_curves(ex)[0][1], g)


def test_gripper_binding_rejects_time_motion_unrelated_to_opening(tmp_path):
    ex, _ = fixture(tmp_path)
    ex.O['finger'] = {'type': 'EMPTY', 'parent': 'joint6', 'matrix_world': np.eye(4).tolist()}
    g = np.array([0., .5, .5, 0., .5])
    ex.E['props_anim'] = {'joint6': {'observation.gripper_position': g.tolist()}}
    mats = np.tile(np.eye(4), (5, 1, 1)); mats[:, 1, 3] = .04*(1-g)
    ex.E['object_basis']['finger'] = mats.tolist()
    emb = EM.detect(ex, {}, Decisions())
    b = gripper_binding(ex, emb)
    pose = interpolate_pose(b['values'], b['channels']['finger'], .25)
    assert pose[1, 3] == pytest.approx(.03)
    ex.E['object_basis']['finger'][2][0][3] = .02
    with pytest.raises(ValueError, match='not a function'):
        gripper_binding(ex, emb)


def test_explicit_gripper_property_keeps_normalized_validation(tmp_path):
    ex, _ = fixture(tmp_path)
    ex.O['finger'] = {'type': 'EMPTY', 'parent': 'joint6', 'matrix_world': np.eye(4).tolist()}
    g = np.array([0., .5, .5, 0., .5])
    ex.E['props_anim'] = {'joint6': {'measured_closure': g.tolist()}}
    mats = np.tile(np.eye(4), (5, 1, 1)); mats[:, 1, 3] = .04*(1-g)
    ex.E['object_basis']['finger'] = mats.tolist()
    emb = EM.detect(ex, {}, Decisions())
    with pytest.raises(ValueError, match='needs an animated'):
        gripper_binding(ex, emb)
    emb['gripper_property'] = {'object': 'joint6', 'property': 'measured_closure'}
    binding = gripper_binding(ex, emb)
    assert binding['property'] == 'measured_closure'
    assert interpolate_pose(binding['values'], binding['channels']['finger'], .25)[1, 3] == pytest.approx(.03)
    ex.E['props_anim']['joint6']['measured_closure'][1] = 2
    with pytest.raises(ValueError, match='Invalid normalized'):
        gripper_binding(ex, emb)
