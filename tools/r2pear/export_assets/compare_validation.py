"""Validate source binding and frame correspondence of a real/Blender comparison."""
import json
import numpy as np

COMPARE_FILES = {'compare.mp4', 'compare-poster.png', 'compare-report.json', 'COMPARE_README.txt'}
SOURCE_FILES = {'scene.json', 'interactive/scene/native_trace.npz',
                'interactive/scene/stage0_spec/demo.npz',
                'blender-hd/blender-hd.mp4', 'blender-hd/render-report.json'}


def frame_mapping(hd_indices, demo_frames, source_offset, source_frames):
    indices = np.asarray(hd_indices)
    if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('HD state indices must be a nonempty integer vector')
    if demo_frames < 2 or source_offset < 0 or source_offset + demo_frames > source_frames:
        raise ValueError('Real source does not cover the complete retained demo')
    if indices[0] != 0 or np.any(np.diff(indices) < 0):
        raise ValueError('HD states must start at zero and be monotonic')
    return np.minimum(indices, demo_frames - 1) + source_offset


def validate_compare(source, scene, digest, video_dir=None):
    assert all((source/n).is_file() and not (source/n).is_symlink() for n in COMPARE_FILES), 'Missing comparison files'
    report = json.loads((source/'compare-report.json').read_text())
    assert report['ok'] and report['video']['codec_name'] == 'h264'
    assert report['video_sha256'] == digest(source/'compare.mp4'), 'Comparison video changed'
    assert report['scene_id'] == json.loads((scene/'scene.json').read_text())['scene_id']
    assert set(report['source_hashes']) == SOURCE_FILES, 'Incomplete comparison source binding'
    video_dir = video_dir or scene/'blender-hd'
    for name, expected in report['source_hashes'].items():
        path = video_dir/name.split('/', 1)[1] if name.startswith('blender-hd/') else scene/name
        assert digest(path) == expected, f'Different comparison source: {name}'
    hd = json.loads((video_dir/'render-report.json').read_text())
    with np.load(scene/'interactive/scene/stage0_spec/demo.npz') as demo:
        count = len(demo['q'])
    mapping = frame_mapping(hd['source_indices_per_video_frame'], count,
                            report['alignment']['source_frame_offset'], report['real_source']['frame_count'])
    assert mapping.tolist() == report['alignment']['source_indices_per_video_frame'], 'Incorrect frame alignment'
    assert int(report['video']['nb_read_frames']) == len(mapping) == int(hd['video']['nb_read_frames'])
    assert report['video']['r_frame_rate'] == hd['video']['r_frame_rate']
    assert report['real_source']['sha256'] and report['real_source']['provenance'], 'Missing real footage provenance'
    return report
