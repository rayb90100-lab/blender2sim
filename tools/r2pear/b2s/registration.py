"""DROID-first registration audit. Reports offsets; never edits measured motion.

Global robot alignment and local tool/grasp discrepancies are separate checks.
Thresholds are configurable workflow defaults, not calibration specifications.
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def compare_tracks(authored, reference, grasp_authored=None, grasp_reference=None,
                   global_mm=5., global_deg=2., grasp_mm=5., residual_ratio=3.):
    a, r = np.asarray(authored, float), np.asarray(reference, float)
    if a.shape != r.shape or a.ndim != 3 or a.shape[1:] != (4, 4) or not len(a):
        raise ValueError('Aligned nonempty Nx4x4 pose tracks are required')
    if not np.isfinite(a).all() or not np.isfinite(r).all():
        raise ValueError('Nonfinite registration input')
    relative = a[:, :3, :3] @ r[:, :3, :3].transpose(0, 2, 1)
    rotation = Rotation.from_matrix(relative).mean().as_matrix()
    translation = np.median(a[:, :3, 3] - r[:, :3, 3] @ rotation.T, axis=0)
    residual = a[:, :3, 3] - (r[:, :3, 3] @ rotation.T + translation)
    pos_error = np.linalg.norm(a[:, :3, 3]-r[:, :3, 3], axis=1)*1000
    rotation_error = np.degrees(Rotation.from_matrix(relative).magnitude())
    residual_mm = float(np.percentile(np.linalg.norm(residual, axis=1)*1000, 95))
    warning = bool(np.percentile(pos_error, 95) > global_mm or np.percentile(rotation_error, 95) > global_deg)
    report = {'thresholds': {'global_position_mm': global_mm, 'global_rotation_deg': global_deg,
                            'grasp_mm': grasp_mm, 'relative_ratio': residual_ratio},
              'global': {'median_position_error_mm': float(np.median(pos_error)),
                         'p95_position_error_mm': float(np.percentile(pos_error, 95)),
                         'p95_rotation_error_deg': float(np.percentile(rotation_error, 95)),
                         'reference_to_authored_translation_mm': (translation*1000).tolist(),
                         'reference_to_authored_rotation': rotation.tolist(),
                         'aligned_residual_p95_mm': residual_mm, 'warning': warning},
              'grasp': {'status': 'unmeasured'}, 'automatic_scene_translation_applied': False}
    if (grasp_authored is None) != (grasp_reference is None):
        raise ValueError('Both grasp-point tracks are required')
    if grasp_authored is not None:
        ga, gr = np.asarray(grasp_authored), np.asarray(grasp_reference)
        if ga.shape != gr.shape or ga.ndim != 2 or ga.shape[1] != 3 or not len(ga):
            raise ValueError('Aligned nonempty Nx3 grasp-point tracks are required')
        if not np.isfinite(ga).all() or not np.isfinite(gr).all():
            raise ValueError('Nonfinite grasp points')
        delta = ga - (gr @ rotation.T + translation)
        gap = float(np.median(np.linalg.norm(delta, axis=1))*1000)
        limit = max(grasp_mm, residual_ratio*residual_mm)
        report['grasp'] = {'status': 'evaluated', 'residual_median_mm': gap,
                           'residual_vector_median_mm': (np.median(delta, axis=0)*1000).tolist(),
                           'comparison_limit_mm': limit, 'warning': bool(gap > limit)}
    report['recommendation'] = ('resolve_global_registration_then_recheck_grasp' if warning else
        'check_tool_geometry_then_test_scene_translation_with_unchanged_DROID' if report['grasp'].get('warning') else
        'check_grasp_reference' if report['grasp']['status'] == 'unmeasured' else 'no_detected_registration_warning')
    return report


def audit_droid(ex, emb, robot, cfg):
    from common import rot_of
    from sources import load_droid_npz
    if not cfg.get('droid_trajectory') or robot is None:
        return {'status': 'not_applicable', 'reason': 'No aligned DROID NPZ robot source'}
    path = Path(cfg['droid_trajectory'])
    load_droid_npz(path, robot['q'], ex.frames, ex.fps)
    config = cfg.get('registration', {})
    flange = emb.get('flange')
    if not flange:
        return {'status': 'incomplete', 'warning': True,
                'reason': 'Specify embodiment.flange to audit global registration against DROID'}
    with np.load(path) as raw:
        if 'observation.cartesian_position' not in raw:
            return {'status': 'incomplete', 'warning': True, 'reason': 'DROID Cartesian observations missing'}
        cartesian = raw['observation.cartesian_position']
        measured = np.repeat(np.eye(4)[None], len(cartesian), axis=0)
        measured[:, :3, 3] = cartesian[:, :3]
        measured[:, :3, :3] = Rotation.from_euler('xyz', cartesian[:, 3:]).as_matrix()
        command = raw['action_dict.gripper_position'].ravel()
    authored = np.linalg.inv(robot['T_world_base']) @ ex.M(flange)
    authored[:, :3, :3] = np.array([rot_of(t) for t in authored])
    ga = gr = None
    offset = config.get('grasp_point_in_flange_m')
    frames = []
    if offset is not None:
        offset = np.asarray(offset, float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('registration.grasp_point_in_flange_m must be finite length 3')
        rising = np.flatnonzero((command[1:] > .5) & (command[:-1] <= .5))+1
        if len(rising):
            indices = np.arange(max(0, rising[0]-3), min(len(command), rising[0]+7))
            ga = (authored[indices, :3, :3] @ offset[:, None]).squeeze(-1)+authored[indices, :3, 3]
            gr = np.array(robot['fk']['pinch'])[indices, :3, 3]
            frames = np.asarray(ex.frames)[indices].tolist()
    report = compare_tracks(authored, measured, ga, gr,
                            global_mm=float(config.get('global_warn_mm', 5)),
                            global_deg=float(config.get('global_warn_deg', 2)),
                            grasp_mm=float(config.get('grasp_warn_mm', 5)),
                            residual_ratio=float(config.get('relative_ratio', 3)))
    report.update(status='complete', droid_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  reference='Raw measured DROID flange pose and unchanged measured joints; native tool FK',
                  grasp_reference_grade=config.get('grasp_reference_grade', 'authored_estimate'),
                  grasp_frames=frames,
                  limitations='Robot alignment is not scene/object calibration. A local discrepancy can also be tool geometry; retain original motion and validate any scene correction by contacts and full replay.')
    report['warning'] = bool(report['global']['warning'] or report['grasp'].get('warning') or not frames)
    return report
