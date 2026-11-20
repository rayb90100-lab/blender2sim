"""Explicitly correct a source scene's settling poses; preserve before/after evidence.

This changes reconstructed initial object poses, not recorded robot commands.
It is a scene correction, never evidence that the original authored poses passed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'b2s')]
import sim as S
import spec as SP
import validate as V


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=3.)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('Refusing to overwrite an existing scene')
    if not np.isfinite(a.seconds) or a.seconds <= 0:
        ap.error('seconds must be positive and finite')
    spec = SP.load(a.source/'stage0_spec/task_spec.json')
    if not spec.get('model_xml'):
        raise ValueError('Explicit source MJCF is required')
    demo_path = a.source/'stage0_spec/demo.npz'
    demo = dict(np.load(demo_path))
    before = V.v1_settle(spec, seconds=a.seconds, demo=demo)
    m, d = S.build(spec)
    S.reset(m, d, spec, settle_s=0)
    poses = {n: (d.body(n).xpos.copy(), d.body(n).xquat.copy()) for n in spec['objects']}
    for _ in range(round(a.seconds/m.opt.timestep)):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    tree = ET.parse(spec['model_xml']['path'])
    output = copy.deepcopy(spec)
    output.pop('experimental_preview', None)
    corrections = {}
    for name in spec['objects']:
        adr = int(m.joint(name+'_free').qposadr[0])
        p, q = d.qpos[adr:adr+3], d.qpos[adr+3:adr+7]
        body = tree.find(f"./worldbody/body[@name='{name}']")
        if body is None:
            raise ValueError(f'Correction requires a root free body: {name}')
        body.set('pos', ' '.join(map(str, p)))
        body.set('quat', ' '.join(map(str, q)))
        corrections[name] = {'translation_mm': ((p-poses[name][0])*1000).tolist(),
                             'original_position_m': poses[name][0].tolist(),
                             'original_quaternion': poses[name][1].tolist(),
                             'initialized_position_m': p.tolist(), 'initialized_quaternion': q.tolist()}
        output['objects'][name]['center'] = p.tolist()
    (a.out/'stage0_spec').mkdir(parents=True)
    tree.write(a.out/'model.xml', encoding='unicode')
    output['model_xml'] = {'path': '../model.xml', 'sha256': hashlib.sha256((a.out/'model.xml').read_bytes()).hexdigest()}
    output['initialization_adjustment'] = {'method': f'{a.seconds:g} s physical settling, explicit pose correction',
        'source_acceptance_ok': before['ok'], 'report': 'initialization-report.json',
        'note': 'Does not establish camera/object calibration; original reference offsets remain recorded.'}
    (a.out/'stage0_spec/task_spec.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    shutil.copy2(demo_path, a.out/'stage0_spec/demo.npz')
    for name in ('registration-report.json',):
        if (a.source/name).exists():
            shutil.copy2(a.source/name, a.out/name)
    output = SP.load(a.out/'stage0_spec/task_spec.json')
    report = {'source_spec_sha256': hashlib.sha256((a.source/'stage0_spec/task_spec.json').read_bytes()).hexdigest(),
              'demo_sha256': hashlib.sha256(demo_path.read_bytes()).hexdigest(),
              'before': before, 'corrections': corrections, 'after': V.v1_settle(output, demo=demo)}
    report['replay'], _ = V.v2_replay(output, demo)
    (a.out/'initialization-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if not report['after']['ok'] or not report['replay']['success']:
        raise SystemExit('Corrected scene did not pass; reports retained, do not publish')
    print(json.dumps({'prepared': str(a.out), 'initial_state_corrected': True, 'original_commands_unchanged': True}))


if __name__ == '__main__':
    main()
