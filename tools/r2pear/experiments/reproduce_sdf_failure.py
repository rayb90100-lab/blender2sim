"""Reproduce a saved SDF failure without rerunning the robot trajectory."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    model = mujoco.MjModel.from_binary_path(str(args.run/'replay.mjb'))
    with np.load(args.run/'replay-trace.npz') as trace:
        qpos = trace['final_qpos'].copy()
    report = {'mujoco_version': mujoco.__version__, 'tests': {}}
    # First reproduce using the engine's unmodified contact filtering.
    for diagnostic in (False, True):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        last_pair = None

        def contact_filter(m, d, g1, g2):
            nonlocal last_pair
            # A callback REPLACES the default bitmask filter. Preserve it.
            allowed = ((m.geom_contype[g1] & m.geom_conaffinity[g2]) or
                       (m.geom_contype[g2] & m.geom_conaffinity[g1]))
            if not allowed:
                return 1
            last_pair = (g1, g2)
            return 0

        previous = mujoco.get_mjcb_contactfilter()
        mujoco.set_mjcb_contactfilter(contact_filter if diagnostic else None)
        try:
            mujoco.mj_forward(model, data)
            result = {'reproduced': False}
        except mujoco.FatalError as exc:
            result = {'reproduced': True, 'error': str(exc)}
            if last_pair is not None:
                result['last_collision_pair'] = [
                    {'id': int(g), 'name': model.geom(g).name,
                     'body': model.body(int(model.geom_bodyid[g])).name,
                     'type': int(model.geom_type[g]),
                     'contype': int(model.geom_contype[g]),
                     'conaffinity': int(model.geom_conaffinity[g]),
                     'mesh': model.mesh(int(model.geom_dataid[g])).name
                     if model.geom_type[g] in (7, 8) else None}
                    for g in last_pair]
        finally:
            mujoco.set_mjcb_contactfilter(previous)
        report['tests']['diagnostic' if diagnostic else 'default_filter'] = result
    report['note'] = 'Last accepted candidate before error; default-filter reproduction is authoritative.'
    output = json.dumps(report, indent=2, allow_nan=False)+'\n'
    (args.run/'failure-reproduction.json').write_text(output)
    print(output)


if __name__ == '__main__':
    main()
