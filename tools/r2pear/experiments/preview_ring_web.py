"""Run the existing WebUI against a saved experimental collision model.

The override exists only in this process. Production model construction and the
WebUI layout remain unchanged; the preview spec records the exact model hash.
"""
import hashlib
from pathlib import Path
import sys

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sim
import web_ui


def preview_build(spec, overrides=None, render_size=None, return_spec=False):
    preview = spec.get('experimental_preview', {})
    if preview.get('kind') != 'test4_coacd_18':
        raise ValueError('This launcher requires the test4 CoACD preview spec')
    if overrides:
        raise ValueError('Experimental XML preview does not support random initial states')
    path = Path(preview['model_xml'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != preview['model_sha256']:
        raise ValueError('Preview model changed; regenerate the preview manifest')
    source = mujoco.MjSpec.from_file(str(path))
    if render_size:
        source.visual.global_.offwidth = max(render_size[0], 640)
        source.visual.global_.offheight = max(render_size[1], 480)
    model = source.compile()
    data = mujoco.MjData(model)
    return (model, data, source) if return_spec else (model, data)


if __name__ == '__main__':
    sim.build = preview_build
    web_ui.main()
