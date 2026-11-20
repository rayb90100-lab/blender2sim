"""HD supplements must remain tied to the published physics and preserve existing artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2]/'export_assets/publish_video.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    scene=tmp_path/'scene'; scene.mkdir()
    (scene/'scene.json').write_text(json.dumps({'scene_id':'fixture'}))
    (scene/'trace.npz').write_bytes(b'validated trajectory')
    (scene/'interactive-environment.tar.gz').write_bytes(b'original environment')
    (scene/'browser-validation.json').write_text('{"ok": true}')
    (scene/'export-status.json').write_text('{"complete": true, "validated": true}')
    (scene/'validation.json').write_text(json.dumps({'V1_settle':{'ok':True},'V2_replay':{'success':True},
            'display':{'ok':True},'relocation':{'ok':True,'oracle_success':True}}))
    (scene/'checksums.json').write_text(json.dumps({p.name:sha(p) for p in scene.iterdir()}))
    video=tmp_path/'video';video.mkdir()
    for name in ('blender-hd.mp4','blender-hd-poster.png','blender-hd-final.png','README.txt'):
        (video/name).write_bytes(name.encode())
    report={'ok':True,'scene_id':'fixture','video':{'codec_name':'h264'},
            'video_sha256':sha(video/'blender-hd.mp4'),'source_hashes':{'trace.npz':sha(scene/'trace.npz')}}
    (video/'render-report.json').write_text(json.dumps(report))
    return scene,video


def run(scene,video):
    return subprocess.run([sys.executable,str(SCRIPT),'--video-dir',str(video),'--scene-dir',str(scene)],
                          capture_output=True,text=True)


def test_hd_publication_preserves_original_artifacts_and_extends_hashes(tmp_path):
    scene,video=fixture(tmp_path)
    original={p.name:p.read_bytes() for p in scene.iterdir() if p.name!='checksums.json'}
    result=run(scene,video)
    assert result.returncode==0,result.stderr
    for name,data in original.items():
        assert (scene/name).read_bytes()==data
    hashes=json.loads((scene/'checksums.json').read_text())
    assert all(sha(scene/name)==digest for name,digest in hashes.items())
    assert 'blender-hd/blender-hd.mp4' in hashes


def test_hd_publication_rejects_other_trajectory_without_mutating_scene(tmp_path):
    scene,video=fixture(tmp_path)
    report=json.loads((video/'render-report.json').read_text());report['source_hashes']['trace.npz']='0'*64
    (video/'render-report.json').write_text(json.dumps(report))
    before=(scene/'checksums.json').read_bytes()
    result=run(scene,video)
    assert result.returncode!=0 and 'Different source' in result.stderr
    assert not (scene/'blender-hd').exists() and (scene/'checksums.json').read_bytes()==before


def test_hd_publication_refuses_to_overwrite_existing_video(tmp_path):
    scene,video=fixture(tmp_path)
    (scene/'blender-hd').mkdir();marker=scene/'blender-hd/keep.txt';marker.write_text('keep')
    result=run(scene,video)
    assert result.returncode!=0 and 'Refusing to overwrite' in result.stderr
    assert marker.read_text()=='keep'


def initial_publish(scene, target, video=None, comparison=None):
    args=[sys.executable,str(SCRIPT.with_name('publish.py')),'--bundle',str(scene),
          '--browser-report',str(scene/'browser-validation.json'),'--destination',str(target)]
    if video is not None:
        args += ['--blender-video-dir',str(video)]
    if comparison is not None:
        args += ['--compare-dir',str(comparison)]
    return subprocess.run(args,capture_output=True,text=True)


def prepare_browser_report(scene):
    for name in ('mujoco-web3d-scene.glb','mujoco-web3d-replay.json'):
        (scene/name).write_bytes(name.encode())
    browser={'ok':True,'browser_errors':[],
             'interactive':{'oracle_success':True,'reset_steps':0,'blender_frame':True},
             'display':{'maxPositionError':0},
             'artifact_hashes':{name:sha(scene/name) for name in ('scene.json','mujoco-web3d-scene.glb',
                              'mujoco-web3d-replay.json','interactive-environment.tar.gz')}}
    (scene/'browser-validation.json').write_text(json.dumps(browser))
    (scene/'checksums.json').write_text(json.dumps({p.name:sha(p) for p in scene.iterdir() if p.name!='checksums.json'}))


def test_initial_publication_requires_blender_video(tmp_path):
    scene,_=fixture(tmp_path);prepare_browser_report(scene)
    target=tmp_path/'published'
    result=initial_publish(scene,target)
    assert result.returncode!=0 and 'Blender HD video is required' in result.stderr
    assert not target.exists()


def test_initial_publication_includes_matching_video_atomically(tmp_path):
    scene,video=fixture(tmp_path);prepare_browser_report(scene)
    target=tmp_path/'published'
    comparison=compare_fixture(scene,video,tmp_path/'comparison')
    result=initial_publish(scene,target,video,comparison)
    assert result.returncode==0,result.stderr
    assert (target/'blender-hd/blender-hd.mp4').read_bytes()==(video/'blender-hd.mp4').read_bytes()
    hashes=json.loads((target/'checksums.json').read_text())
    assert all(sha(target/name)==digest for name,digest in hashes.items())
    # A bundle which already includes its video satisfies the same requirement.
    result=initial_publish(target,tmp_path/'republished')
    assert result.returncode==0,result.stderr


def test_initial_publication_rejects_video_from_another_trajectory(tmp_path):
    scene,video=fixture(tmp_path);prepare_browser_report(scene)
    report=json.loads((video/'render-report.json').read_text());report['source_hashes']['trace.npz']='0'*64
    (video/'render-report.json').write_text(json.dumps(report))
    target=tmp_path/'published'
    result=initial_publish(scene,target,video)
    assert result.returncode!=0 and 'Different source' in result.stderr
    assert not target.exists()


def compare_fixture(scene, video, dest):
    from importlib.util import spec_from_file_location, module_from_spec
    module_spec=spec_from_file_location('compare_validation_fixture',SCRIPT.with_name('compare_validation.py'))
    module=module_from_spec(module_spec);module_spec.loader.exec_module(module)
    dest.mkdir()
    stage=scene/'interactive/scene/stage0_spec';stage.mkdir(parents=True)
    np.savez(stage/'demo.npz',q=np.zeros((3,7)))
    np.savez(scene/'interactive/scene/native_trace.npz',time=np.arange(4))
    info=json.loads((video/'render-report.json').read_text())
    info['source_indices_per_video_frame']=[0,1,2,3]
    info['video'].update(nb_read_frames='4',r_frame_rate='30/1')
    (video/'render-report.json').write_text(json.dumps(info))
    for name in ('compare.mp4','compare-poster.png','COMPARE_README.txt'):
        (dest/name).write_bytes(name.encode())
    report={'ok':True,'scene_id':'fixture','video':{'codec_name':'h264','nb_read_frames':'4','r_frame_rate':'30/1'},
      'video_sha256':sha(dest/'compare.mp4'),
      'source_hashes':{n:sha(video/n.split('/',1)[1] if n.startswith('blender-hd/') else scene/n) for n in module.SOURCE_FILES},
      'alignment':{'source_frame_offset':5,'source_indices_per_video_frame':[5,6,7,7]},
      'real_source':{'frame_count':8,'sha256':'e'*64,'provenance':'test footage'}}
    (dest/'compare-report.json').write_text(json.dumps(report))
    (scene/'checksums.json').write_text(json.dumps({p.relative_to(scene).as_posix():sha(p) for p in scene.rglob('*') if p.is_file() and p.name!='checksums.json'}))
    return dest


def test_initial_publication_requires_comparison(tmp_path):
    scene,video=fixture(tmp_path);prepare_browser_report(scene)
    result=initial_publish(scene,tmp_path/'published',video)
    assert result.returncode!=0 and 'comparison is required' in result.stderr
    assert not (tmp_path/'published').exists()


def test_comparison_mismatched_hd_is_rejected_before_publication(tmp_path):
    scene,video=fixture(tmp_path);prepare_browser_report(scene)
    comparison=compare_fixture(scene,video,tmp_path/'comparison')
    report=json.loads((comparison/'compare-report.json').read_text())
    report['source_hashes']['blender-hd/blender-hd.mp4']='0'*64
    (comparison/'compare-report.json').write_text(json.dumps(report))
    result=initial_publish(scene,tmp_path/'published',video,comparison)
    assert result.returncode!=0 and 'Different comparison source' in result.stderr
    assert not (tmp_path/'published').exists()
