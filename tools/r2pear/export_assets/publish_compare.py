"""Append a verified comparison to an existing public scene without altering its replay/runtime."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

try:
    from .publish import digest, verify
    from .compare_validation import COMPARE_FILES, validate_compare
    from .video_validation import validate_video
except ImportError:
    from publish import digest, verify
    from compare_validation import COMPARE_FILES, validate_compare
    from video_validation import validate_video


def publish(source, scene):
    source,scene=source.resolve(),scene.resolve()
    if any((scene/n).exists() for n in COMPARE_FILES):
        raise FileExistsError('Refusing to overwrite an existing comparison')
    verify(scene)
    validate_video(scene/'blender-hd',scene,digest)
    validate_compare(source,scene,digest)
    assert {p.name for p in source.iterdir()} == COMPARE_FILES, 'Unexpected comparison supplement files'
    original=(scene/'checksums.json').read_bytes()
    added=[]
    with tempfile.TemporaryDirectory(prefix='.compare-',dir=scene.parent) as temp:
        stage=Path(temp)/'payload';shutil.copytree(source,stage)
        hashes=json.loads(original)
        for name in COMPARE_FILES:
            assert digest(stage/name)==digest(source/name)
            hashes[name]=digest(stage/name)
        index=Path(temp)/'checksums.json';index.write_text(json.dumps(hashes,indent=2)+'\n')
        assert (scene/'checksums.json').read_bytes()==original,'Scene changed during publication'
        try:
            for name in sorted(COMPARE_FILES):
                if (scene/name).exists():raise FileExistsError(scene/name)
                os.rename(stage/name,scene/name);added.append(name)
            os.replace(index,scene/'checksums.json')
            verify(scene);validate_compare(scene,scene,digest)
        except BaseException:
            for name in added:(scene/name).unlink(missing_ok=True)
            (scene/'checksums.json').write_bytes(original)
            raise
    return {'published':str(scene/'compare.mp4'),'verified':True,'existing_artifacts_unchanged':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compare-dir',required=True,type=Path)
    ap.add_argument('--scene-dir',required=True,type=Path)
    a=ap.parse_args();print(json.dumps(publish(a.compare_dir,a.scene_dir),ensure_ascii=False))

if __name__=='__main__':main()
