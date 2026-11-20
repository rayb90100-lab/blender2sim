"""Offline high-quality rendering; live service settings are not changed."""
import json
import math
import os
from pathlib import Path
import time


def render_sequence(requests, frames_dir, cameras, apply_state):
    import bpy
    scene = bpy.context.scene
    dest = Path(frames_dir)
    dest.mkdir(parents=True, exist_ok=True)
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = int(os.environ.get('R2P_BLEND_SAMPLES', '64'))
    scene.cycles.use_denoising = True
    scene.render.use_persistent_data = True
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = int(os.environ.get('R2P_RENDER_THREADS', '8'))
    backend = os.environ.get('R2P_BLEND_DEVICE', 'OPTIX')
    if backend == 'CPU':
        scene.cycles.device = 'CPU'
        devices = ['CPU']
    else:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = backend
        prefs.get_devices()
        devices = []
        for device in prefs.devices:
            device.use = device.type == backend
            if device.use:
                devices.append(device.name)
        if not devices:
            raise RuntimeError(f'No {backend} devices available')
        scene.cycles.device = 'GPU'
        scene.cycles.denoising_use_gpu = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.compression = 15
    rows = [json.loads(line) for line in Path(requests).read_text().splitlines() if line.strip()]
    if rows and rows[0].get('presentation_camera'):
        from mathutils import Matrix
        camera = rows[0]['presentation_camera']
        data = bpy.data.cameras.new('b2s_hd_export_view')
        data.sensor_fit = 'VERTICAL'
        data.lens = data.sensor_height / (2 * math.tan(math.radians(camera['fovy_deg']) / 2))
        data.clip_start = .001
        ob = bpy.data.objects.new('b2s_hd_export_view', data)
        scene.collection.objects.link(ob)
        ob.matrix_world = Matrix(camera['matrix_world'])
        cameras[rows[0]['cam']] = ob
    timings = []
    for i, req in enumerate(rows):
        target = dest / f"{req['source_index']:06d}.png"
        apply_state(req)
        scene.camera = cameras[req['cam']]
        scene.render.resolution_x, scene.render.resolution_y = req['w'], req['h']
        scene.render.resolution_percentage = 100
        scene.render.filepath = str(target.resolve())
        start = time.monotonic()
        bpy.ops.render.render(write_still=True)
        timings.append(time.monotonic() - start)
        print(f"HD frame {i+1}/{len(rows)} source={req['source_index']} seconds={timings[-1]:.3f}", flush=True)
    (dest.parent / 'render-device.json').write_text(json.dumps({
        'engine': 'CYCLES', 'device_backend': backend, 'devices': devices,
        'samples': scene.cycles.samples, 'denoising': True, 'render_seconds': timings,
        'denoising_use_gpu': scene.cycles.denoising_use_gpu,
        'blender_version': bpy.app.version_string}, indent=2))
