#!/usr/bin/env python3
"""Render an additional Blender HD video from a validated export's recorded physics.

No simulation is rerun. Output frames hold the most recent recorded state;
encoding at 30 fps does not invent additional physics samples.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageStat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_scene as EX
import sim
import spec as SP


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--camera', default='agentview')
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--device', choices=['OPTIX', 'CUDA', 'CPU'], default='OPTIX')
    ap.add_argument('--preview-only', action='store_true')
    ap.add_argument('--encode-only', action='store_true', help='Resume encoding from a complete matching PNG sequence')
    a = ap.parse_args()
    if min(a.width, a.height, a.fps, a.samples) <= 0 or a.width % 2 or a.height % 2:
        ap.error('Positive dimensions/fps/samples and even width/height required')
    bundle, out = a.bundle.resolve(), a.out.resolve()
    if out.exists() and not a.encode_only:
        raise SystemExit(f'Refusing to overwrite {out}')
    if a.encode_only and (not out.is_dir() or a.preview_only):
        raise SystemExit('--encode-only requires an existing full-sequence output')
    from export_assets.publish import verify
    verify(bundle)
    source = bundle / 'interactive/scene'
    sp = source / 'stage0_spec/task_spec.json'
    spec = SP.load(sp)
    presentation = None
    if a.camera == 'export_view':
        camera = json.loads((bundle/'mujoco-web3d-manifest.json').read_text())['camera']
        conversion = EX.Y_UP[:3,:3].T
        eye, target, up = [conversion @ np.asarray(camera[k]) for k in ('eye','target','up')]
        forward = target-eye; forward /= np.linalg.norm(forward)
        right = np.cross(forward,up); right /= np.linalg.norm(right)
        up = np.cross(right,forward)
        matrix = np.eye(4); matrix[:3,:3] = np.column_stack((right,up,-forward)); matrix[:3,3] = eye
        presentation = {'matrix_world': matrix.tolist(), 'fovy_deg': camera.get('fovy_deg',45)}
    elif spec['cameras'][a.camera]['mount'] != 'world':
        raise ValueError('Use a fixed world camera for this presentation video')
    m, _ = sim.build(spec)
    with np.load(source / 'native_trace.npz') as archive:
        trace = dict(archive)
    ts = trace['timestamps_seconds']
    assert ts[0] == 0 and np.all(np.diff(ts) > 0)
    count = int(math.ceil(float(ts[-1]) * a.fps - 1e-8)) + 1
    times = np.minimum(np.arange(count) / a.fps, ts[-1])
    indices = np.clip(np.searchsorted(ts, times + 1e-9, side='right') - 1, 0, len(ts)-1)
    selected = sorted(set(indices.tolist()))
    if a.preview_only:
        selected = sorted({0, len(ts)//2, len(ts)-1})
    out.mkdir(parents=True, exist_ok=a.encode_only)
    EX.dump(out/'status.json', {'complete': False, 'preview_only': a.preview_only})
    requests = []
    for index in selected:
        row = {k: v[index:index+1] for k, v in trace.items()}
        pose = EX.initial_pose(row, m, spec)
        pose.update(source_index=index, cam=a.camera, w=a.width, h=a.height)
        if presentation:
            pose['presentation_camera'] = presentation
        requests.append(json.dumps(pose) + '\n')
    request_text = ''.join(requests)
    if a.encode_only:
        assert (out/'requests.jsonl').read_text() == request_text, 'Resume source/camera/settings differ'
        device = json.loads((out/'render-device.json').read_text())
        assert device['samples'] == a.samples and device['engine'] == 'CYCLES'
    else:
        (out/'requests.jsonl').write_text(request_text)
    env = dict(os.environ)
    env.update(R2P_BLEND_ENGINE='CYCLES', R2P_BLEND_SAMPLES=str(a.samples),
               R2P_BLEND_DEVICE=a.device, CUDA_VISIBLE_DEVICES=str(a.gpu),
               R2P_RENDER_THREADS='8')
    worker = HERE/'b2s/blend_worker.py'
    renderer_hashes = {p.name:EX.sha(p) for p in [Path(__file__),worker,HERE/'b2s/render_sequence.py']}
    if not a.encode_only:
        with (out/'render.log').open('w') as log:
            subprocess.run([sys.executable, str(worker), '--testdir', str(source),
                            '--render-sequence', str(out/'requests.jsonl'), '--frames-dir', str(out/'frames')],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for index in selected:
        with Image.open(out/'frames'/f'{index:06d}.png') as image:
            assert image.size == (a.width, a.height)
            w,h=image.size
            assert max(ImageStat.Stat(image.crop((w//10,h//5,9*w//10,4*h//5))).stddev) >= 2, 'Uniform/occluded frame'
    if a.preview_only:
        EX.dump(out/'status.json', {'complete': True, 'preview_only': True, 'source_indices': selected})
        print(json.dumps({'preview': str(out), 'indices': selected}), flush=True)
        return
    delivery = out/'delivery'
    delivery.mkdir(exist_ok=a.encode_only)
    video = delivery/'blender-hd.mp4'
    with (out/'ffmpeg.log').open('w') as log:
        ff = subprocess.Popen(['ffmpeg','-v','error','-y','-f','image2pipe','-framerate',str(a.fps),
                               '-vcodec','png','-threads','2','-i','-','-an','-c:v','libx264','-threads','2',
                               '-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
                              stdin=subprocess.PIPE, stderr=log)
        try:
            last=None; pixels=None
            for index in indices:
                if index != last:
                    pixels=(out/'frames'/f'{index:06d}.png').read_bytes();last=index
                ff.stdin.write(pixels)
        finally:
            ff.stdin.close()
            code=ff.wait()
        if code:
            raise RuntimeError('ffmpeg encoding failed')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
          '-count_frames','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration',
          '-of','json',str(video)],text=True))['streams'][0]
    assert probe['codec_name']=='h264' and int(probe['nb_read_frames'])==count
    assert (probe['width'],probe['height'])==(a.width,a.height) and probe['r_frame_rate']==f'{a.fps}/1'
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True,stderr=subprocess.PIPE)
    shutil.copy2(out/'frames'/'000000.png',delivery/'blender-hd-poster.png')
    shutil.copy2(out/'frames'/f'{len(ts)-1:06d}.png',delivery/'blender-hd-final.png')
    device=json.loads((out/'render-device.json').read_text())
    report={'ok':True,'scene_id':json.loads((bundle/'scene.json').read_text())['scene_id'],
            'camera':a.camera,'video':probe,'source_state_count':len(ts),'rendered_state_count':len(selected),
            'source_duration_seconds':float(ts[-1]),'source_indices_per_video_frame':indices.tolist(),
            'time_sampling':'zero-order hold of the published MuJoCo states; no new physics or interpolation',
            'source_hashes':{name:EX.sha(bundle/name) for name in ['scene.json','interactive/scene/native_trace.npz',
                             'interactive/scene/stage0_spec/task_spec.json','interactive/source/scene.blend']},
            'renderer_hashes':renderer_hashes,
            'render':device,'video_sha256':EX.sha(video)}
    EX.dump(delivery/'render-report.json',report)
    (delivery/'README.txt').write_text(
        f"Blender 高清视频：{a.width}×{a.height}，{a.fps} fps，H.264，Cycles {a.samples} samples + denoising。\n"
        f"相机：{a.camera}。动作来自本场景已发布的 MuJoCo native_trace.npz，保持原物理轨迹和时序。\n"
        "30 fps 封装使用最近的已记录状态；没有重新仿真、加速动作或生成额外物理采样。\n"
        "原 Blender 的场景材质、灯光和背景保留；机器人与物体由记录的物理状态驱动。\n"
        "来源校验、实际设备、帧数和渲染参数见 render-report.json。原任务的物理/标定限制仍然适用。\n")
    EX.dump(out/'status.json', {'complete':True,'preview_only':False,'validated':True})
    print(json.dumps({'output':str(delivery),'frames':count,'ok':True}),flush=True)


if __name__ == '__main__':
    main()
