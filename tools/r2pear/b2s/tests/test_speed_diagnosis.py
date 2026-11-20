import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import speed_diagnosis as SD


def inputs():
    spec = {'source': {'timing': {'source_fps': 30, 'playback_fps': 30, 'time_scale': 1}},
            'objects': {'phone': {}}}
    demo = {'q': np.zeros((4, 7)), 'g_cmd': np.zeros(4), 'fps': np.array(30.)}
    validation = {'V1_settle': {'ok': True}, 'V2_replay': {'success': False}}
    return spec, demo, validation


@pytest.mark.parametrize('case,reason,expected', [
    ('default', None, 'regular_diagnosis_not_exhausted'),
    ('default', ' ', 'regular_diagnosis_not_exhausted'),
    ('unstable', 'checked', 'fix_initial_geometry_and_stability_first'),
    ('passed', 'checked', 'configured_replay_already_passes'),
    ('missing', 'checked', 'regular_replay_validation_missing'),
    ('timing_missing', 'checked', 'source_timing_provenance_missing'),
    ('timing_mismatch', 'checked', 'source_timing_provenance_inconsistent'),
    ('no_trajectory', 'checked', 'invalid_or_missing_demonstration'),
])
def test_regular_checks_prevent_slowdown(monkeypatch, case, reason, expected):
    spec, demo, validation = inputs()
    if case == 'unstable': validation['V1_settle']['ok'] = False
    if case == 'passed': validation['V2_replay']['success'] = True
    if case == 'missing': validation.pop('V2_replay')
    if case == 'timing_missing': spec['source'] = {}
    if case == 'timing_mismatch': demo['fps'] = 10
    if case == 'no_trajectory': demo['q'] = np.zeros((0, 7))
    def forbidden(*a, **kw):
        pytest.fail('Skipped diagnosis must not launch physics or slowdown trials')
    monkeypatch.setattr(SD.V, 'v1_settle', forbidden)
    monkeypatch.setattr(SD.V, 'v2_replay', forbidden)
    r = SD.diagnose(spec, demo, validation, reason)
    assert r['skip_reason'] == expected
    assert not r['trials'] and not r['automatic_changes_applied']


def test_stale_validation_cannot_hide_current_initial_instability(monkeypatch):
    spec, demo, validation = inputs()
    monkeypatch.setattr(SD.V, 'v1_settle', lambda *a, **kw: {'ok': False})
    monkeypatch.setattr(SD.V, 'v2_replay', lambda *a, **kw: pytest.fail('unstable scene'))
    assert SD.diagnose(spec, demo, validation, 'checked')['skip_reason'] == 'current_initial_state_invalid'


@pytest.mark.parametrize('success_fps', [30, 15, 10, None])
def test_bounded_source_relative_trials_preserve_inputs(monkeypatch, success_fps):
    spec, demo, validation = inputs()
    # Already adapted input must not cause another compounded 2x/3x slowdown.
    spec['source']['timing'].update(playback_fps=10, time_scale=3)
    demo['fps'] = 10.
    before = copy.deepcopy((spec, demo, validation))
    rates = []
    monkeypatch.setattr(SD.V, 'v1_settle', lambda *a, **kw: {'ok': True})
    def replay(s, d, **kw):
        rates.append(d['fps'])
        np.testing.assert_array_equal(d['q'], demo['q'])
        np.testing.assert_array_equal(d['g_cmd'], demo['g_cmd'])
        return {'success': d['fps'] == success_fps}, {'phone': [np.eye(4)] * 4}
    monkeypatch.setattr(SD.V, 'v2_replay', replay)
    report = SD.diagnose(spec, demo, validation, 'registration, geometry and contact checks exhausted')
    assert rates == ([30] if success_fps == 30 else [30, 15, 10])
    assert report['status'] == ('original_speed_passed' if success_fps == 30 else
                                'all_speeds_failed' if success_fps is None else 'slower_replay_passed')
    if success_fps in (15, 10):
        assert report['smallest_successful_time_scale'] == 30 / success_fps
    assert spec == before[0] and validation == before[2]
    for k, value in demo.items(): np.testing.assert_array_equal(value, before[1][k])
    assert not report['automatic_changes_applied']
