"""Publish a verified export into a new shared directory, without overwriting scenes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(root):
    status = json.loads((root / 'export-status.json').read_text())
    assert status.get('complete') and status.get('validated'), 'Incomplete export'
    check = json.loads((root / 'validation.json').read_text())
    assert check['V1_settle']['ok'] and check['V2_replay']['success']
    assert check['display']['ok'] and check['relocation']['ok'] and automatic_succeeded(check['relocation'])
    hashes = {name: value for name, value in json.loads((root / 'checksums.json').read_text()).items()
              if '__pycache__' not in Path(name).parts and Path(name).suffix != '.pyc'}
    for name, expected in hashes.items():
        p = (root / name).resolve()
        assert p.is_relative_to(root.resolve()), name
        assert digest(p) == expected, f'Checksum mismatch: {name}'
    # Running Python creates bytecode caches; these are not delivery artifacts.
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.relative_to(root).parts and p.suffix != '.pyc'}
    assert actual == set(hashes) | {'checksums.json'}, 'Unaccounted files in bundle'


def automatic_succeeded(report):
    """Keep source-demo qualification distinct from historical oracle reports."""
    if report.get('automatic_kind') == 'source_demo':
        replay = report.get('source_replay') or {}
        return bool(report.get('automatic_success') and replay.get('complete') and replay.get('success'))
    if report.get('automatic_kind') not in (None, 'oracle'):
        return False
    return bool(report.get('automatic_success', report.get('oracle_success', False)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--browser-report', required=True, type=Path)
    ap.add_argument('--destination', required=True, type=Path)
    ap.add_argument('--blender-video-dir', type=Path, help='Required HD video delivery, unless already included as bundle/blender-hd')
    ap.add_argument('--compare-dir', type=Path, help='Required real/Blender comparison, unless already included in bundle')
    a = ap.parse_args()
    root, target = a.bundle.resolve(), a.destination.resolve()
    if target.exists():
        raise SystemExit(f'Refusing to overwrite: {target}')
    video = a.blender_video_dir.resolve() if a.blender_video_dir else root/'blender-hd'
    if not video.is_dir():
        raise SystemExit('Blender HD video is required before publication; render it and pass --blender-video-dir')
    verify(root)
    if __package__:
        from .video_validation import VIDEO_FILES, validate_video
    else:
        from video_validation import VIDEO_FILES, validate_video
    validate_video(video, root, digest)
    if __package__:
        from .compare_validation import COMPARE_FILES, validate_compare
    else:
        from compare_validation import COMPARE_FILES, validate_compare
    comparison = a.compare_dir.resolve() if a.compare_dir else root
    if not all((comparison/name).is_file() for name in COMPARE_FILES):
        raise SystemExit('Real/Blender comparison is required before publication; generate it and pass --compare-dir')
    validate_compare(comparison, root, digest, video)
    browser = json.loads(a.browser_report.read_text())
    assert browser['ok'] and not browser['browser_errors']
    assert automatic_succeeded(browser['interactive']) and browser['interactive']['reset_steps'] == 0
    assert browser['interactive']['blender_frame'] and browser['display']['maxPositionError'] < 1e-9
    required = {'scene.json', 'mujoco-web3d-scene.glb', 'mujoco-web3d-replay.json', 'interactive-environment.tar.gz'}
    assert set(browser['artifact_hashes']) == required
    for name, expected in browser['artifact_hashes'].items():
        assert digest(root / name) == expected, f'Browser test is for a different artifact: {name}'
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + target.name + '-', dir=target.parent) as stage:
        payload = Path(stage) / 'payload'
        shutil.copytree(root, payload, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        verify(payload)
        if (payload/'blender-hd').exists():
            assert all(digest(payload/'blender-hd'/name) == digest(video/name) for name in VIDEO_FILES), 'Conflicting Blender videos'
        else:
            shutil.copytree(video, payload/'blender-hd')
        validate_video(payload/'blender-hd', payload, digest)
        for name in COMPARE_FILES:
            if (payload/name).exists():
                assert digest(payload/name) == digest(comparison/name), f'Conflicting comparison: {name}'
            else:
                shutil.copy2(comparison/name, payload/name)
        validate_compare(payload, payload, digest)
        shutil.copy2(a.browser_report, payload / 'browser-validation.json')
        checksums = {p.relative_to(payload).as_posix(): digest(p) for p in sorted(payload.rglob('*'))
                     if p.is_file() and p.name != 'checksums.json'}
        (payload / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
        verify(payload)
        if target.exists():
            raise FileExistsError(target)
        os.rename(payload, target)
    print(json.dumps({'published': str(target), 'verified': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
