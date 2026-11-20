"""Validate that a Blender video supplement belongs to a given scene bundle."""
import json

VIDEO_FILES = {'blender-hd.mp4', 'blender-hd-poster.png', 'blender-hd-final.png',
               'render-report.json', 'README.txt'}


def validate_video(source, scene, digest):
    assert {p.name for p in source.iterdir()} == VIDEO_FILES, 'Unexpected supplement contents'
    assert all(p.is_file() and not p.is_symlink() for p in source.iterdir())
    report = json.loads((source/'render-report.json').read_text())
    assert report['ok'] and report['video']['codec_name'] == 'h264'
    assert report['video_sha256'] == digest(source/'blender-hd.mp4')
    assert report['scene_id'] == json.loads((scene/'scene.json').read_text())['scene_id']
    assert report['source_hashes'], 'Missing video source binding'
    for name, sha in report['source_hashes'].items():
        path = (scene/name).resolve()
        assert path.is_relative_to(scene.resolve()) and digest(path) == sha, f'Different source: {name}'
    return report
