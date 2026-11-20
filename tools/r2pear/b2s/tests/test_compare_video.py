"""Comparison sampling must respect source trims and simulation settling."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from export_assets.compare_validation import frame_mapping
from export_assets.publish_compare import publish
from test_hd_video import fixture, compare_fixture, run, sha


def test_trimmed_umi_sampling_and_final_hold():
    assert frame_mapping([0,0,1,239,240,241,255],241,683,924).tolist()==[683,683,684,922,923,923,923]
    with pytest.raises(ValueError,match='complete retained demo'):
        frame_mapping([0,1,2],241,683,923)


def test_slowdown_follows_recorded_state_indices_not_video_duration():
    indices=np.repeat(np.arange(160),3).tolist()+[160,161]
    result=frame_mapping(indices,160,0,160)
    assert result[0]==0 and result[300]==100 and result[-1]==159
    with pytest.raises(ValueError,match='monotonic'):
        frame_mapping([0,2,1],3,0,3)


def test_supplement_preserves_existing_bundle_and_rejects_overwrite(tmp_path):
    scene,video=fixture(tmp_path)
    comparison=compare_fixture(scene,video,tmp_path/'comparison')
    assert run(scene,video).returncode==0
    original={p.relative_to(scene).as_posix():p.read_bytes() for p in scene.rglob('*') if p.is_file() and p.name!='checksums.json'}
    result=publish(comparison,scene)
    assert result['verified']
    for name,data in original.items():assert (scene/name).read_bytes()==data
    hashes=json.loads((scene/'checksums.json').read_text())
    assert all(sha(scene/name)==value for name,value in hashes.items())
    with pytest.raises(FileExistsError,match='overwrite'):
        publish(comparison,scene)


def test_invalid_frame_mapping_does_not_touch_public_scene(tmp_path):
    scene,video=fixture(tmp_path)
    comparison=compare_fixture(scene,video,tmp_path/'comparison')
    assert run(scene,video).returncode==0
    p=comparison/'compare-report.json';r=json.loads(p.read_text())
    r['alignment']['source_indices_per_video_frame']=[0,1,2,3];p.write_text(json.dumps(r))
    before=(scene/'checksums.json').read_bytes()
    with pytest.raises(AssertionError,match='frame alignment'):
        publish(comparison,scene)
    assert not (scene/'compare.mp4').exists() and (scene/'checksums.json').read_bytes()==before
