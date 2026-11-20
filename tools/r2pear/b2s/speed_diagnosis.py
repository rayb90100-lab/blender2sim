"""Last-resort timing diagnosis; never changes a scene, demonstration or acceptance result."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import validate as V
import spec as SP
from common import dump_json


def diagnose(spec, demo, validation, reason=None):
    report = {
        'status': 'skipped', 'reason_for_last_resort': reason,
        'trials': [], 'automatic_changes_applied': False,
        'scope': 'Timing sensitivity only; slowed success is not original-timing success or a root-cause proof.',
    }

    def skip(why):
        report['skip_reason'] = why
        return report

    if not reason or not reason.strip():
        return skip('regular_diagnosis_not_exhausted')
    if validation.get('V1_settle', {}).get('ok') is not True:
        return skip('fix_initial_geometry_and_stability_first')
    success = validation.get('V2_replay', {}).get('success')
    if success is True:
        return skip('configured_replay_already_passes')
    if success is not False:
        return skip('regular_replay_validation_missing')
    q = np.asarray(demo.get('q', []))
    g = np.asarray(demo.get('g_cmd', demo.get('g', [])))
    if (q.ndim != 2 or q.shape[1] != 7 or len(q) < 2 or g.shape != (len(q),)
            or not np.isfinite(q).all() or not np.isfinite(g).all()):
        return skip('invalid_or_missing_demonstration')
    timing = spec.get('source', {}).get('timing', {})
    try:
        source_fps = float(timing['source_fps'])
        configured_fps = float(demo['fps'])
        scale = float(timing['time_scale'])
        playback_fps = float(timing['playback_fps'])
    except (KeyError, ValueError, TypeError):
        return skip('source_timing_provenance_missing')
    if (not all(np.isfinite(x) and x > 0 for x in [source_fps, configured_fps, scale, playback_fps])
            or not np.isclose(playback_fps, configured_fps)
            or not np.isclose(source_fps / scale, configured_fps)):
        return skip('source_timing_provenance_inconsistent')
    report.update(source_fps=source_fps, configured_fps=configured_fps,
                  configured_time_scale=scale, source_samples=len(q))
    # Recheck current inputs, including when called directly against an older validation report.
    report['fresh_initial_validation'] = V.v1_settle(spec, demo=demo)
    if not report['fresh_initial_validation']['ok']:
        return skip('current_initial_state_invalid')

    def trial(time_scale):
        contacts = {n: {'any_gripper_contact_frames': [], 'bilateral_contact_frames': []}
                    for n in spec['objects']}
        samples = 0
        interval = None

        def capture(f, m, d):
            nonlocal samples, interval
            samples += 1
            interval = round(time_scale / source_fps / m.opt.timestep) * m.opt.timestep
            touching = {n: set() for n in contacts}
            for c in d.contact:
                bodies = {m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2)}
                for n in bodies.intersection(touching):
                    touching[n].update(b for b in bodies if b in ('g_left_pad', 'g_right_pad'))
            for n, pads in touching.items():
                if pads:
                    contacts[n]['any_gripper_contact_frames'].append(f)
                if len(pads) == 2:
                    contacts[n]['bilateral_contact_frames'].append(f)

        result, trajectories = V.v2_replay(spec, {**demo, 'fps': source_fps / time_scale},
                                           on_frame=capture)
        return {'time_scale': time_scale, 'speed_ratio': 1 / time_scale,
                'fps': source_fps / time_scale, 'executed_samples': samples,
                'physics_seconds_per_interval': interval, 'result': result,
                'max_lift_mm': {n: float((max(T[2, 3] for T in ts) - ts[0][2, 3]) * 1000)
                                for n, ts in trajectories.items()},
                'contacts': contacts, 'contact_frame_indexing': 'zero_based; pad contact does not prove stable grasp'}

    report['trials'].append(trial(1))
    if report['trials'][0]['result']['success']:
        report['status'] = 'original_speed_passed'
        report['recommendation'] = 'No slowdown needed; investigate discrepancy with prior configured validation.'
        return report
    # Bounded fallback: all preceding normal checks have already completed; no unbounded search.
    for time_scale in (2, 3):
        report['trials'].append(trial(time_scale))
    passed = [t['time_scale'] for t in report['trials'][1:] if t['result']['success']]
    report.update(status='slower_replay_passed' if passed else 'all_speeds_failed',
                  successful_time_scales=passed,
                  smallest_successful_time_scale=min(passed) if passed else None,
                  recommendation=('Timing adaptation candidate only. Keep original failure; review tracking and '
                                  'contacts before explicitly configuring a separate adapted scene.' if passed else
                                  'Slowdown did not solve the task; return to geometry, trajectory and control diagnosis.'))
    return report


def run_diagnosis(simdir, reason=None):
    root = Path(simdir)
    paths = {'spec': root / 'stage0_spec/task_spec.json', 'demo': root / 'stage0_spec/demo.npz',
             'validation': root / 'stage1_validate/validate.json'}
    if not paths['validation'].is_file():
        report = {'status': 'skipped', 'skip_reason': 'regular_validation_missing',
                  'automatic_changes_applied': False, 'trials': []}
    else:
        with np.load(paths['demo'], allow_pickle=False) as archive:
            demo = dict(archive)
        report = diagnose(SP.load(str(paths['spec'])), demo,
                          json.loads(paths['validation'].read_text()), reason)
    report['input_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in paths.items() if path.is_file()}
    out = root / 'stage4_diagnostics/speed-diagnosis.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(report, str(out))
    print(f"速度诊断: {report['status']} ({report.get('skip_reason', '')}) → {out}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simdir', required=True)
    parser.add_argument('--reason', required=True, help='Record conventional checks attempted and why timing is the last resort.')
    args = parser.parse_args()
    run_diagnosis(args.simdir, args.reason)
