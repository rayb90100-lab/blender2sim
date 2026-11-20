#!/usr/bin/env python3
"""Compare actual source footage (left) with the existing Blender HD video (right).

Frame correspondence follows the HD report's recorded-state indices, including
trimmed source frames, retarget slowdown and final source-frame holds. It never
uses global duration fitting or regenerates physics/Blender images.
"""
import argparse
from fractions import Fraction
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_assets.publish import digest, verify
from export_assets.video_validation import validate_video
from export_assets.compare_validation import frame_mapping, validate_compare, SOURCE_FILES


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration',
        '-of', 'json', str(path)], text=True))['streams'][0]


def read_exact(stream, count):
    data = bytearray()
    while len(data) < count:
        block = stream.read(count - len(data))
        if not block:
            raise RuntimeError(f'Unexpected video EOF: {len(data)}/{count} bytes')
        data.extend(block)
    return bytes(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--blender-video-dir', type=Path)
    ap.add_argument('--source-video', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--source-frame-offset', type=int, default=0, help='Zero-based frame retained at demo start')
    ap.add_argument('--source-fps', type=float, required=True, help='Source sampling rate; declare assumptions in provenance')
    ap.add_argument('--source-provenance', required=True, help='Dataset/episode, extraction and correspondence evidence')
    ap.add_argument('--source-view', required=True)
    ap.add_argument('--view-note', required=True, help='Explain matching or different camera viewpoints')
    ap.add_argument('--source-crop', nargs=4, type=int, metavar=('X','Y','W','H'))
    a = ap.parse_args()
    scene, out, real = a.bundle.resolve(), a.out.resolve(), a.source_video.resolve()
    hd_dir = a.blender_video_dir.resolve() if a.blender_video_dir else scene/'blender-hd'
    if out.exists():
        raise SystemExit(f'Refusing to overwrite {out}')
    if not np.isfinite(a.source_fps) or a.source_fps <= 0:
        ap.error('Source fps must be finite and positive')
    verify(scene)
    hd_report = validate_video(hd_dir, scene, digest)
    real_info = probe(real)
    with np.load(scene/'interactive/scene/stage0_spec/demo.npz') as z:
        demo_frames, playback_fps = len(z['q']), float(z['fps'])
    indices = frame_mapping(hd_report['source_indices_per_video_frame'], demo_frames,
                            a.source_frame_offset, int(real_info['nb_read_frames']))
    hd_info = probe(hd_dir/'blender-hd.mp4')
    assert len(indices) == int(hd_info['nb_read_frames'])
    rate = Fraction(hd_info['r_frame_rate']); fps = float(rate)
    out.mkdir(parents=True)
    width, height, header = 1920, 640, 100
    panel_w, panel_h = 960, 540
    base_filter = f'scale={panel_w}:{panel_h}:force_original_aspect_ratio=decrease,pad={panel_w}:{panel_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1'
    crop = ''
    if a.source_crop:
        x,y,w,h=a.source_crop
        assert min(x,y)>=0 and min(w,h)>0 and x+w<=real_info['width'] and y+h<=real_info['height']
        crop=f'crop={w}:{h}:{x}:{y},'
    def decoder(path, filters):
        return subprocess.Popen(['ffmpeg','-v','error','-threads','2','-i',str(path),'-an',
            '-vf',filters,'-vsync','0','-threads','2','-f','rawvideo','-pix_fmt','rgb24','-'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    trim=f'trim=start_frame={a.source_frame_offset}:end_frame={a.source_frame_offset+demo_frames},setpts=PTS-STARTPTS,'
    raw_real = subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(real),'-an',
        '-vf',crop+trim+base_filter,'-vsync','0','-threads','2','-f','rawvideo','-pix_fmt','rgb24','-'])
    frame_bytes = panel_w*panel_h*3
    assert len(raw_real) == demo_frames*frame_bytes
    hd = decoder(hd_dir/'blender-hd.mp4',base_filter)
    enc = subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',
        f'{width}x{height}','-framerate',str(rate),'-i','-','-an','-c:v','libx264','-threads','2',
        '-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'compare.mp4')],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title=ImageFont.truetype(font_path,24);small=ImageFont.truetype(font_path,17)
    try:
        for i, real_index in enumerate(indices):
            canvas=Image.new('RGB',(width,height),(20,24,30));draw=ImageDraw.Draw(canvas)
            local_index=int(real_index)-a.source_frame_offset
            left=Image.frombytes('RGB',(panel_w,panel_h),raw_real[local_index*frame_bytes:(local_index+1)*frame_bytes])
            right=Image.frombytes('RGB',(panel_w,panel_h),read_exact(hd.stdout,frame_bytes))
            canvas.paste(left,(0,header));canvas.paste(right,(panel_w,header))
            draw.text((18,10),'REAL  |  '+a.source_view,font=title,fill=(240,240,240))
            draw.text((978,10),'BLENDER HD  |  '+hd_report['camera'],font=title,fill=(240,240,240))
            draw.text((18,44),f'Source frame {int(real_index)+1}   t={real_index/a.source_fps:.3f}s',font=small,fill=(185,205,220))
            draw.text((978,44),f'Playback t={i/fps:.3f}s   {a.view_note}',font=small,fill=(185,205,220))
            hold = hd_report['source_indices_per_video_frame'][i] >= demo_frames
            note = 'Source ended: holding its final retained frame during simulation settling' if hold else f'Frame correspondence | source {a.source_fps:g} fps -> demo {playback_fps:g} fps | not a calibration metric'
            draw.text((18,74),note,font=small,fill=(225,205,155))
            draw.line((959,header,959,height),fill=(220,220,220),width=2)
            if i==0:canvas.save(out/'compare-poster.png')
            if i in {len(indices)//2,len(indices)-1}:canvas.save(out.parent/(out.name+f'-review-{i}.png'))
            enc.stdin.write(canvas.tobytes())
        enc.stdin.close()
        error=enc.stderr.read();assert enc.wait()==0,error.decode(errors='replace')
        assert hd.stdout.read()==b''
        error=hd.stderr.read();assert hd.wait()==0,error.decode(errors='replace')
    finally:
        for process in (hd,enc):
            if process.poll() is None:process.kill();process.wait()
    info=probe(out/'compare.mp4')
    assert int(info['nb_read_frames'])==len(indices) and (info['width'],info['height'])==(width,height)
    subprocess.run(['ffmpeg','-v','error','-i',str(out/'compare.mp4'),'-f','null','-'],check=True,stderr=subprocess.PIPE)
    hashes={n:digest(hd_dir/n.split('/',1)[1] if n.startswith('blender-hd/') else scene/n) for n in SOURCE_FILES}
    report={'ok':True,'scene_id':json.loads((scene/'scene.json').read_text())['scene_id'],'video':info,
        'video_sha256':digest(out/'compare.mp4'),'source_hashes':hashes,
        'real_source':{'filename':real.name,'sha256':digest(real),'frame_count':int(real_info['nb_read_frames']),
            'sampling_fps':a.source_fps,'video':real_info,'crop_xywh':a.source_crop,'view':a.source_view,'provenance':a.source_provenance},
        'alignment':{'source_frame_offset':a.source_frame_offset,'demo_frames':demo_frames,'playback_fps':playback_fps,
            'source_indices_per_video_frame':indices.tolist(),'method':'HD recorded-state frame index -> retained source frame; final frame held during settling'},
        'view_note':a.view_note,'layout':'real left / existing Blender HD right; aspect ratios preserved; silent H264',
        'generator_sha256':digest(Path(__file__))}
    (out/'compare-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'COMPARE_README.txt').write_text('左右对比：左侧真实视频，右侧本目录既有Blender高清视频。\n'
        '按HD记录状态对应到示范帧；保留裁剪偏移、慢速重定向与结尾定格，不按整片时长强行拉伸。\n'
        f'源视频：{a.source_provenance}\n相机说明：{a.view_note}\n'
        f'源采样率{a.source_fps:g}fps；示范执行{playback_fps:g}fps；零基裁剪偏移{a.source_frame_offset}帧。\n'
        '两侧保持宽高比；真实低分辨率画面放大不会增加细节。无音轨。\n'
        '帧对应不等于运动精确复现或相机标定通过；源视角/具身可能不同，原场景限制仍适用。\n'
        'compare-report.json记录逐帧映射、真实视频来源和哈希、HD与物理轨迹绑定及解码验证结果。\n')
    validate_compare(out,scene,digest,hd_dir)
    print(json.dumps({'output':str(out),'frames':len(indices),'ok':True},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
