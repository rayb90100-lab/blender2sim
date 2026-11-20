import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from registration import compare_tracks


def tracks():
    poses = np.repeat(np.eye(4)[None], 20, axis=0)
    poses[:, :3, 3] = np.random.default_rng(0).normal(size=(20, 3))*.1
    return poses


def test_global_translation_is_not_misclassified_as_local_grasp_error():
    raw = tracks(); authored = raw.copy(); authored[:, 0, 3] += .03
    r = compare_tracks(authored, raw, authored[:, :3, 3], raw[:, :3, 3])
    assert r['global']['warning'] and not r['grasp']['warning']
    assert not r['automatic_scene_translation_applied']
    assert abs(r['global']['reference_to_authored_translation_mm'][0]-30) < 1e-8


def test_local_tool_offset_detected_with_correct_global_alignment():
    raw = tracks(); local = raw[:, :3, 3]+[.02, 0, 0]
    r = compare_tracks(raw, raw, local, raw[:, :3, 3])
    assert not r['global']['warning'] and r['grasp']['warning']
    assert abs(r['grasp']['residual_median_mm']-20) < 1e-8
    assert 'check_tool_geometry' in r['recommendation']


def test_global_rotation_is_removed_before_local_comparison():
    raw = tracks(); transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('z', 5, degrees=True).as_matrix()
    authored = transform @ raw
    r = compare_tracks(authored, raw, authored[:, :3, 3], raw[:, :3, 3])
    assert r['global']['warning'] and not r['grasp']['warning']


def test_missing_grasp_evidence_is_not_a_pass():
    raw = tracks(); r = compare_tracks(raw, raw)
    assert r['grasp']['status'] == 'unmeasured'
    assert r['recommendation'] == 'check_grasp_reference'
