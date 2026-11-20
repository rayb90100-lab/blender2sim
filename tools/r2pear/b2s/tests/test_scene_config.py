"""Regression coverage for thin, grouped objects and wide-jaw demonstrations."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Extract, Decisions
import parse_scene
import build_spec
import embodiment
import demo


def scene_extract(tmp_path):
    objects, meshes = {}, {}
    def box(name, center, dims, parent=None):
        M = np.eye(4); M[:3, 3] = center
        objects[name] = {'type': 'MESH', 'parent': parent, 'hide_render': False,
                         'matrix_world': M.tolist()}
        verts = np.array(list(itertools.product((-0.5, 0.5), repeat=3))) * dims
        meshes[name] = {'verts_local': verts.tolist(), 'n_verts': 8,
                        'volume_local': float(np.prod(dims)), 'materials': []}
    box('table', [0, 0, -.02], [.8, .8, .04])
    box('phone_mesh', [-.2, 0, .004], [.16, .078, .008], 'phone_root')
    box('lid', [.15, 0, .001], [.29, .28, .002])
    for name in ['phone_root', '右夹爪 · 工具中心', 'left_gripper']:
        M = np.eye(4)
        if name == 'phone_root': M[:3, 3] = [-.2, 0, 0]
        objects[name] = {'type': 'EMPTY', 'parent': None, 'matrix_world': M.tolist()}
    start = np.eye(4); end = start.copy(); end[0, 3] = .1
    data = {'objects': objects, 'meshes': meshes, 'frames': [1, 2], 'scene': {'fps': 30},
            'dynamic': {'右夹爪 · 工具中心': [start.tolist(), end.tolist()]},
            'armatures': {}, 'cameras': {}, 'collections': {}}
    p = tmp_path/'extract.json'; p.write_text(json.dumps(data))
    return Extract(p)


def test_explicit_thin_support_uses_its_actual_surface(tmp_path):
    ex = scene_extract(tmp_path)
    cfg = {'table': 'table', 'objects': ['phone_mesh', 'lid'],
           'object_names': {'phone_mesh': 'phone', 'lid': 'box_lid'},
           'shapes': {'phone_mesh': 'box', 'lid': 'box'}}
    scene = parse_scene.parse(ex, {'type': 'handheld_gripper', 'root': '右夹爪 · 工具中心'},
                              None, cfg, Decisions())
    lid = scene['objects']['box_lid']
    assert lid['on'] == ['table']
    assert abs(lid['top_z'] - .002) < 1e-9
    tracks = {s: ex.M(o['blend_name']) for s, o in scene['objects'].items()}
    tracks['phone'][-1, :3, 3] = [.15, 0, .006]
    sup, _ = build_spec.final_supports(ex, scene, {'objects': tracks}, 'phone')
    assert sup == ['box_lid']
    binding = build_spec.blend_binding(ex, scene, {'objects': tracks}, None,
                                      {'root': '右夹爪 · 工具中心'},
                                      {'visual_roots': {'phone_mesh': 'phone_root'}, 'hide_trees': ['left_gripper']})
    assert binding['objects']['phone']['blend_name'] == 'phone_root'
    np.testing.assert_allclose(binding['objects']['phone']['cloc'], [0, 0, .004])
    assert 'left_gripper' in binding['hide']
    assert 'table' not in binding['hide']


def test_chinese_gripper_and_wide_object_closure(tmp_path):
    ex = scene_extract(tmp_path)
    assert embodiment.detect(ex, {}, Decisions())['root'] == '右夹爪 · 工具中心'
    # A 78 mm object is firmly contacted before a normalized 20% closure.
    g = np.array([0., 1 - .078/.085, 1 - .078/.085, 0.])
    carry = np.array([False, True, True, False])
    assert demo.consistency(g, carry, closed=.04) == 1
    assert demo.consistency(g, carry) == 0


def test_retarget_checks_background_and_includes_gripper():
    import retarget
    ik = retarget.IK()
    table = {'top_z': -2., 'center_xy': [3., 3.], 'size_xy': [.5, .5], 'yaw_rad': 0.}
    q = ik.home.copy()
    ik.fk(q)
    center = ik.d.xpos[ik.m.body('link4').id].copy()
    empty = ik.penetration(q, table, -3.)
    box = (*center, .10, .10, .10, 1., 0., 0., 0.)
    blocked = ik.penetration(q, table, -3., static_boxes=[box])
    assert blocked > max(.02, empty + .01)
    m, d, robot_geoms = ik.collision_model(table, -3., static_boxes=[box])
    assert m.geom('g_left_pad1').id in robot_geoms
    assert m.geom('g_right_pad1').id in robot_geoms
    # Changing box poses must rebuild the cached collision model.
    moved = (*[v + 10 for v in center], .10, .10, .10, 1., 0., 0., 0.)
    assert ik.penetration(q, table, -3., static_boxes=[moved]) < blocked


def test_static_boxes_transform_with_robot_base():
    import retarget
    base = np.eye(4); base[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]; base[:3,3] = [1,2,3]
    scene = {'static': [{'center': [1,3,3], 'half': [.2,.3,.4], 'R': np.eye(3).tolist()}]}
    box = retarget._static_in_base(scene, base)[0]
    np.testing.assert_allclose(box[:3], [1,0,0], atol=1e-12)
    np.testing.assert_allclose(box[3:6], [.2,.3,.4])
    from scipy.spatial.transform import Rotation
    q = np.array(box[6:10])
    np.testing.assert_allclose(Rotation.from_quat(q[[1,2,3,0]]).as_matrix(),base[:3,:3].T,atol=1e-12)


def test_long_object_grasp_keeps_short_edge_between_jaws():
    import oracle
    import math
    chosen = oracle._grip_yaw(0., math.pi/2, quarter_turns=False)
    assert abs(math.sin(chosen)) < 1e-12
    assert abs(oracle._grip_yaw(0., math.pi/2) - math.pi/2) < 1e-12
