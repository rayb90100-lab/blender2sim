#!/usr/bin/env python3
"""Export compact DROID modeling inputs from a cached RLDS Example and source manifest.

Three left-camera videos + source.json + trajectory.npz; calibration.json only
when a matching raw HDF5/metadata pair is provided. No TensorFlow dependency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

STREAMS = {'exterior_1_left': 'exterior_image_1_left',
           'exterior_2_left': 'exterior_image_2_left', 'wrist_left': 'wrist_image_left'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def varint(buf, pos):
    value = shift = 0
    while shift < 70:
        byte = buf[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
        shift += 7
    raise ValueError('Invalid protobuf varint')


def fields(buf):
    pos = 0
    while pos < len(buf):
        tag, pos = varint(buf, pos)
        wire = tag & 7
        if wire == 0:
            value, pos = varint(buf, pos)
        else:
            if wire == 2:
                size, pos = varint(buf, pos)
            elif wire in (1, 5):
                size = 8 if wire == 1 else 4
            else:
                raise ValueError(f'Unsupported protobuf wire type {wire}')
            if pos + size > len(buf):
                raise ValueError('Truncated protobuf')
            value, pos = buf[pos:pos + size], pos + size
        yield tag >> 3, wire, value


def read_example(raw):
    features = next(v for n, _, v in fields(memoryview(raw)) if n == 1)
    result = {}
    for n, _, entry in fields(features):
        if n != 1:
            continue
        item = {n: v for n, _, v in fields(entry)}
        kind, _, values = next(fields(item[2]))
        if kind == 1:
            data = [bytes(v) for n, _, v in fields(values) if n == 1]
        elif kind == 2:
            data = np.concatenate([np.frombuffer(v, dtype='<f4') for n, _, v in fields(values) if n == 1])
        elif kind == 3:
            numbers = []
            for n, wire, value in fields(values):
                if n != 1:
                    continue
                if wire == 0:
                    numbers.append(value)
                else:
                    pos = 0
                    while pos < len(value):
                        number, pos = varint(value, pos)
                        numbers.append(number)
            data = np.array([n if n < 2**63 else n - 2**64 for n in numbers], dtype=np.int64)
        else:
            raise ValueError(f'Unsupported Feature type {kind}')
        result[bytes(item[1]).decode()] = data
    return result


def trajectory(episode, frames, fps):
    result = {}
    for key, value in episode.items():
        if not key.startswith('steps/') or isinstance(value, list):
            continue
        arr = value.reshape(frames, -1)
        if not np.isfinite(arr).all():
            raise ValueError(f'Non-finite trajectory field: {key}')
        result[key.removeprefix('steps/').replace('/', '.')] = arr
    if result['observation.joint_position'].shape != (frames, 7):
        raise ValueError('Expected seven measured joints per frame')
    result['frame_index'] = np.arange(frames)
    result['timestamp_s_nominal'] = np.arange(frames) / fps
    return result


def raw_calibration(hdf5, metadata_path, arrays, source):
    import h5py
    metadata = json.loads(metadata_path.read_text())
    frames = len(arrays['frame_index'])
    # Match the original recording path as well as all measured states. Frame
    # count alone cannot distinguish episodes from the same robot/camera setup.
    original = source['episode_metadata/file_path'][0]
    if not original.endswith('/' + metadata['lab'] + '/' + metadata['hdf5_path']):
        raise ValueError('Raw metadata belongs to a different recording')
    if metadata['trajectory_length'] != frames:
        raise ValueError('Raw metadata frame count differs')
    cameras = {}
    with h5py.File(hdf5, 'r') as f:
        for key, rawkey in [('joint_position', 'joint_positions'),
                            ('cartesian_position', 'cartesian_position'),
                            ('gripper_position', 'gripper_position')]:
            actual = f['observation/robot_state/' + rawkey][:].reshape(frames, -1)
            expected = arrays['observation.' + key]
            if not np.array_equal(actual.astype(expected.dtype), expected):
                raise ValueError(f'Raw trajectory differs from RLDS: {key}')
        for role in ('wrist', 'ext1', 'ext2'):
            serial = str(metadata[role + '_cam_serial'])
            prefix = 'observation/camera_extrinsics/' + serial + '_left'
            values = f[prefix][:]
            if values.shape != (frames, 6) or not np.isfinite(values).all():
                raise ValueError(f'Invalid camera extrinsics: {serial}')
            camera = {'raw_role': role, 'eye': 'left', 'source_field': prefix,
                      'extrinsics_per_frame': values.tolist()}
            offset = prefix + '_gripper_offset'
            if offset in f:
                camera['gripper_offset_per_frame'] = f[offset][:].tolist()
            cameras[serial] = camera
        def timestamps(name, dataset):
            if isinstance(dataset, h5py.Dataset):
                value = dataset[:]
                if value.shape[0] != frames or not np.isfinite(value).all():
                    raise ValueError(f'Invalid raw timestamp field: {name}')
                arrays['raw.timestamp.' + name.replace('/', '.')] = value
        if 'observation/timestamp' in f:
            f['observation/timestamp'].visititems(timestamps)
    return {
        'schema_version': 1, 'raw_episode_uuid': metadata['uuid'], 'frames': frames,
        'original_recording_path': original,
        'raw_hdf5_sha256': digest(hdf5), 'raw_metadata_sha256': digest(metadata_path),
        'measured_states_match_rlds': True,
        'intrinsics': None,
        'intrinsics_status': 'Not included: not found in the inspected raw HDF5; SVO/supplement not extracted.',
        'cameras_by_serial': cameras,
        'video_camera_serials': {'wrist_left.mp4': str(metadata['wrist_cam_serial']),
                                'exterior_1_left.mp4': None, 'exterior_2_left.mp4': None},
        'notes': [
            'Extrinsics are preserved raw six-component vectors; no Blender coordinate conversion was applied.',
            'External video-to-serial correspondence is unverified; raw ext1/ext2 labels must not be assumed to match RLDS exterior_1/exterior_2.',
            'Per-frame extrinsics align with trajectory frame_index. Wrist extrinsics may move; do not use only the first frame.',
            'Raw calibration accuracy and projection alignment have not been validated. Intrinsics must match the video resolution.',
        ],
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def export(record, source_path, out, raw_h5=None, raw_metadata=None):
    source = json.loads(source_path.read_text())
    if digest(record) != source['record_sha256']:
        raise ValueError('Example checksum differs from source.json')
    if bool(raw_h5) != bool(raw_metadata):
        raise ValueError('Provide both --raw-h5 and --raw-metadata')
    episode = read_example(record.read_bytes())
    frames = source['frames']
    fps = source['timebase']['video_fps']
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid video fps')
    for stream in STREAMS.values():
        if len(episode['steps/observation/' + stream]) != frames:
            raise ValueError('Video and trajectory lengths differ')
    arrays = trajectory(episode, frames, fps)
    calibration = raw_calibration(raw_h5, raw_metadata, arrays, source) if raw_h5 else None
    # Existing calibration must be deliberately preserved by supplying its raw
    # inputs again, rather than silently downgrading an enriched package.
    if (out/'calibration.json').exists() and calibration is None:
        raise ValueError('Existing calibration requires matching raw inputs on re-export')
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.droid-extract-', dir=out) as temp:
        stage = Path(temp)
        videos = {}
        for camera, field in STREAMS.items():
            name = camera + '.mp4'
            target = out/name
            if target.exists():
                if digest(target) != source['videos'][name]['sha256']:
                    raise ValueError(f'Existing video checksum differs: {name}')
            else:
                target = stage/name
                cmd = ['ffmpeg', '-v', 'error', '-f', 'image2pipe', '-framerate', str(fps),
                       '-vcodec', 'mjpeg', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
                       '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
                with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
                    for jpg in episode['steps/observation/' + field]:
                        proc.stdin.write(jpg)
                    proc.stdin.close()
                    if proc.wait() != 0:
                        raise RuntimeError(f'Video encoding failed: {name}')
            videos[name] = {'codec': 'h264', 'sha256': digest(target), 'frames': frames}
        np.savez_compressed(stage/'trajectory.npz', **arrays)
        source['videos'] = videos
        source['trajectory'] = {
            'file': 'trajectory.npz', 'sha256': digest(stage/'trajectory.npz'),
            'fields': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()},
            'modeling_joint_field': 'observation.joint_position',
            'notes': ['observation.* contains measured states; action_dict.* contains commands, not measured motion.',
                      'frame_index is zero-based; Blender frame = frame_index + 1 at the video fps.',
                      'timestamp_s_nominal is derived from video fps, not the original capture clock.',
                      'raw.timestamp.* retains original clock fields and units; clocks are not synchronized or converted here.'],
        }
        source['camera_calibration_present'] = calibration is not None
        source['calibration'] = {'status': 'partial_extrinsics_only' if calibration else 'unavailable',
                                 'intrinsics_present': False}
        if calibration:
            write_json(stage/'calibration.json', calibration)
            source['calibration'].update(file='calibration.json', sha256=digest(stage/'calibration.json'))
        else:
            source['calibration']['reason'] = 'RLDS has no calibration; no matching raw calibration supplied. No estimated values substituted.'
        source['timebase']['original_timestamps_present'] = any(k.startswith('raw.timestamp.') for k in arrays)
        source['readme'] = 'Three left-camera H.264 videos plus measured/commanded trajectory.npz. Available raw calibration is optional and may be partial; see calibration status and notes.'
        write_json(stage/'source.json', source)
        # All validation/encoding succeeds before replacing any delivery files;
        # publish the manifest last. Existing video bytes are never rewritten.
        for path in sorted(stage.iterdir(), key=lambda p: p.name == 'source.json'):
            path.replace(out/path.name)
    return {'directory': str(out), 'frames': frames, 'calibration': source['calibration']['status']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--record-example', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True, help='Source manifest with original record SHA256, frames, fps and recording path')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--raw-h5', type=Path)
    ap.add_argument('--raw-metadata', type=Path)
    args = ap.parse_args()
    print(json.dumps(export(args.record_example, args.source, args.out, args.raw_h5, args.raw_metadata), ensure_ascii=False))


if __name__ == '__main__':
    main()
