"""Add a source-bound HD video supplement without changing existing scene artifacts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from publish import digest, verify
from video_validation import VIDEO_FILES, validate_video


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video-dir', required=True, type=Path)
    ap.add_argument('--scene-dir', required=True, type=Path)
    a = ap.parse_args()
    source, scene = a.video_dir.resolve(), a.scene_dir.resolve()
    target = scene/'blender-hd'
    if target.exists():
        raise SystemExit(f'Refusing to overwrite {target}')
    verify(scene)
    original_checksums = (scene/'checksums.json').read_bytes()
    validate_video(source, scene, digest)
    with tempfile.TemporaryDirectory(prefix='.blender-hd-', dir=scene.parent) as tmp:
        stage=Path(tmp)/'payload'
        shutil.copytree(source,stage)
        checksums=json.loads(original_checksums)
        for name in VIDEO_FILES:
            assert digest(stage/name)==digest(source/name)
            checksums[f'blender-hd/{name}']=digest(stage/name)
        new_index=Path(tmp)/'checksums.json'
        new_index.write_text(json.dumps(checksums,indent=2)+'\n')
        assert (scene/'checksums.json').read_bytes()==original_checksums, 'Scene changed during publication'
        if target.exists():
            raise FileExistsError(target)
        os.rename(stage,target)
        try:
            os.replace(new_index,scene/'checksums.json')
        except BaseException:
            shutil.rmtree(target)
            raise
    verify(scene)
    print(json.dumps({'published':str(target),'verified':True},ensure_ascii=False))


if __name__=='__main__':
    main()
