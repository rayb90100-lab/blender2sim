"""Counterfactual registration tests with unchanged original DROID q/g_cmd.

These are diagnostic scene hypotheses, not calibrated reconstructions. Source
Blender object animation is never used to drive physical objects during replay.
Run from the repository root after sourcing local/env.sh.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'b2s')]
import validate
import spec as SP
import sim as S
import droid_common as C


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', type=Path, default=Path('runs/test4-preview'))
    ap.add_argument('--out', type=Path, default=Path('outputs/test4-registration'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = SP.load(args.scene/'stage0_spec/task_spec.json')
    source_xml = Path(spec['model_xml']['path']).read_bytes()
    (args.out/'source-model.xml').write_bytes(source_xml)
    dump(args.out/'source-spec.json', spec)
    demo_path = args.scene/'stage0_spec/demo.npz'
    demo = dict(np.load(demo_path))
    m, d = S.build(spec)
    S.reset(m, d, spec, settle_s=0)
    ring0 = d.body('blue_ring').xpos.copy()
    normal = d.body('table').xmat.reshape(3, 3)[:, 2].copy()
    project = np.eye(3)-np.outer(normal, normal)
    # First original command that exceeds half closure; q is unchanged.
    f = int(np.flatnonzero(demo['g_cmd'] > .5)[0])
    d.qpos[C.arm_qadr(m)] = demo['q'][f]
    mujoco.mj_forward(m, d)
    pinch = d.site('g_pinch').xpos.copy()
    to_ring = project @ (ring0-pinch)
    # Source blue ring's geometric wall centerline, not an oracle grasp hint.
    # The initial experiment has source radial fit data beside its convex assets.
    root = ET.fromstring(source_xml)
    visual_file = root.find("./asset/mesh[@name='ring_original']").get('file')
    asset_info = json.loads((Path(visual_file).parent/'collision-assets.json').read_text())
    radius = asset_info['torus_major_radius_m']
    shift = -(np.linalg.norm(to_ring)-radius)*to_ring/np.linalg.norm(to_ring)
    reference = np.array([np.linalg.inv(spec['robot']['T_world_base']) @ T for T in demo['obj__blue_ring']])
    authored_shift = project @ (reference[142, :3, 3]-reference[0, :3, 3])
    cases = {
        'baseline': ([], np.zeros(3)),
        'animation_offset_only': (['blue_ring'], authored_shift),
        'trajectory_ring_only': (['blue_ring'], shift),
        'trajectory_task_assembly': (['table', *spec['objects']], shift),
    }
    manifest = {'source_model_sha256': hashlib.sha256(source_xml).hexdigest(),
                'demo_sha256': hashlib.sha256(demo_path.read_bytes()).hexdigest(),
                'frame_used_for_alignment': f+1, 'native_pinch_base_m': pinch.tolist(),
                'ring_initial_base_m': ring0.tolist(), 'table_normal_base': normal.tolist(),
                'wall_centerline_radius_m': radius,
                'hypothesis_shift_base_mm': (shift*1000).tolist(),
                'note': 'Position hypothesis from a single grasp; not camera/object calibration. All q/g_cmd unchanged.',
                'cases': {}}
    for name, (bodies, delta) in cases.items():
        directory = args.out/name
        directory.mkdir(exist_ok=True)
        tree = ET.fromstring(source_xml)
        for body_name in bodies:
            body = tree.find(f".//body[@name='{body_name}']")
            pos = np.fromstring(body.get('pos'), sep=' ')+delta
            body.set('pos', ' '.join(f'{x:.12g}' for x in pos))
        ET.indent(tree)
        path = directory/'model.xml'
        path.write_text(ET.tostring(tree, encoding='unicode')+'\n')
        candidate = copy.deepcopy(spec)
        candidate['model_xml'] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        m, d = S.build(candidate)
        S.reset(m, d, candidate, settle_s=0)
        before = d.body('blue_ring').xpos.copy()
        initial_contacts = validate.contact_pairs(m, d)
        S.reset(m, d, candidate)
        after = d.body('blue_ring').xpos.copy()
        rows, states = [], []

        def on_frame(frame, model, data):
            states.append(data.qpos.copy())
            if 115 <= frame <= 230:
                ring_id = model.body('blue_ring').id
                touch = set()
                for c in data.contact:
                    ids = [int(c.geom1), int(c.geom2)]
                    if any(model.geom_bodyid[i] == ring_id for i in ids):
                        for i in ids:
                            if model.geom(i).name.startswith('g_'):
                                touch.add(model.geom(i).name)
                rows.append({'frame': frame+1, 'robot_contact_geoms': sorted(touch),
                             'ring_position_m': data.xpos[ring_id].tolist()})

        replay, traj = validate.v2_replay(candidate, demo, on_frame=on_frame)
        ring = np.array(traj['blue_ring'])[:, :3, 3]
        held = ring[160:215, 2]-ring[0, 2]
        report = {'translated_bodies': bodies, 'translation_base_mm': (delta*1000).tolist(),
                  'initial_overlaps': initial_contacts, 'reset_ring_displacement_mm': ((after-before)*1000).tolist(),
                  'max_lift_mm': float((ring[:, 2].max()-ring[0, 2])*1000),
                  'min_lift_frames161_215_mm': float(held.min()*1000),
                  'V2': replay, 'grasp_contacts': rows}
        dump(directory/'results.json', report)
        np.savez_compressed(directory/'trace.npz', qpos=np.array(states), **{n: np.array(T) for n, T in traj.items()})
        manifest['cases'][name] = {k: report[k] for k in ['translation_base_mm', 'reset_ring_displacement_mm',
                                                       'max_lift_mm', 'min_lift_frames161_215_mm']}
        manifest['cases'][name]['success'] = replay['success']
        dump(args.out/'manifest.json', manifest)
        print(name, json.dumps(manifest['cases'][name]), flush=True)


if __name__ == '__main__':
    main()
