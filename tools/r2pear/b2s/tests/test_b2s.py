"""b2s 单元测试：覆盖 ep71 实跑没走到的分支（关节符号/零点修复、人手→末端、ARKit 轴变换、形状分类）。

  source setup_env.sh
  $B2S_PYTHON -m pytest -q tools/r2pear/b2s/tests       # 有 pytest
  $B2S_PYTHON tools/r2pear/b2s/tests/test_b2s.py        # 没有 pytest 时

依赖 evidence/test1/sim/stage0_extract/extract.json（ep71 的 blend 提取结果）。
"""
import copy
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import Decisions, Extract, rot_angle_deg  # noqa: E402
import embodiment as EM  # noqa: E402
import parse_scene as PS  # noqa: E402
import sources as SRC  # noqa: E402

# 仓库根 = tests/ -> b2s/ -> r2pear/ -> tools/ -> <repo>
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
EXTRACT = os.environ.get("B2S_EXTRACT") or os.path.join(
    _REPO, "evidence", "test1", "sim", "stage0_extract", "extract.json")


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_visual_binding_preserves_excluded_background():
    """Excluding furniture from physics must not hide it in Blender."""
    import build_spec
    ex = Extract(os.path.join(_REPO, "evidence", "test2", "sim", "stage0_extract", "extract.json"))
    root = "UMI  |  记录的末端坐标系 · 6DoF"
    furniture = "Rear table | 深木工作台"
    scene = {"objects": {}, "robot_set": [root, furniture]}
    binding = build_spec.blend_binding(ex, scene, {}, None, {"root": root})
    assert root in binding["hide"]
    assert set(ex.descendants(root)) <= set(binding["hide"])
    assert furniture not in binding["hide"]
    assert "Saucer  |  蓝色杯碟" not in binding["hide"]


def test_joint_sign_offset_repair():
    """把 J3 的旋转取反、J5 加 30° 零点偏移后，FK 校验必须失败并自动拟合回 s=-1 / c=-30°。"""
    ex = Extract(EXTRACT)
    D = Decisions()
    emb = EM.detect(ex, {}, D)
    ref = EM.extract_robot(ex, emb, D)["q"]
    ex2 = copy.deepcopy(ex)
    arm = ex2.E["armatures"][emb["armature"]]
    ch = emb["chain"]
    for name, fn in ((ch[2], lambda R, ax: R.T),
                     (ch[4], lambda R, ax: R @ _axis_rot(ax, math.radians(30)))):
        _, ax, _, _ = EM.bone_joint_angles(arm["basis"][name])
        B = np.array(arm["basis"][name])
        for t in range(len(B)):
            B[t, :3, :3] = fn(B[t, :3, :3], ax)
        arm["basis"][name] = B.tolist()
    r = EM.extract_robot(ex2, emb, Decisions())
    assert r["fk_ok"], r["fk_spread"]
    assert r["sign"][2] == -1 and abs(math.degrees(r["offset"][4]) + 30) < 0.5, (r["sign"], np.degrees(r["offset"]))
    assert np.abs(r["q"] - ref).max() < 2e-3


def _axis_rot(ax, a):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_rotvec(np.asarray(ax) * a).as_matrix()


def test_hand_to_canonical_roundtrip():
    """由规范末端位姿合成拇指尖/食指尖/手腕，再换算回来，位姿与开口必须还原。"""
    rng = np.random.default_rng(0)
    from scipy.spatial.transform import Rotation as Rot
    T = np.repeat(np.eye(4)[None], 20, 0)
    T[:, :3, :3] = Rot.random(20, random_state=1).as_matrix()
    T[:, :3, 3] = rng.uniform(-0.5, 0.5, (20, 3))
    ap = rng.uniform(0.01, 0.08, 20)
    y, z, p = T[:, :3, 1], T[:, :3, 2], T[:, :3, 3]
    thumb, index = p - y * ap[:, None] / 2, p + y * ap[:, None] / 2
    wrist = p - z * 0.09
    T2, g, ap2 = SRC.hand_to_canonical(wrist, thumb, index)
    assert np.abs(ap2 - ap).max() < 1e-9
    assert np.abs(T2[:, :3, 3] - p).max() < 1e-9
    assert np.abs(T2[:, :3, :3] - T[:, :3, :3]).max() < 1e-9      # 不用 arccos：接近 0 时精度只有 ~1e-6°
    assert np.all((g >= 0) & (g <= 1))


def test_arkit_to_zup():
    """ARKit 的 +y（向上）必须变成 +z；右手系保持（行列式 +1）。"""
    A = SRC.ARKIT_TO_ZUP
    assert np.allclose(A @ [0, 1, 0], [0, 0, 1]) and abs(np.linalg.det(A) - 1) < 1e-12


def test_umi_to_canonical():
    T, g = SRC.umi_to_canonical(np.zeros((3, 3)), np.zeros((3, 3)), np.array([0.0, 0.04, 0.08]))
    assert np.allclose(g, [1.0, 0.5, 0.0]) and np.allclose(T[:, :3, :3], np.eye(3))


def test_shapes_ep71():
    """ep71 五个物体的形状与局部 x 轴约定（棱柱 = 脊方向；拱 = 两腿分开方向）。"""
    ex = Extract(EXTRACT)
    D = Decisions()
    emb = EM.detect(ex, {}, D)
    r = EM.extract_robot(ex, emb, D)
    sc = PS.parse(ex, emb, r, {}, D)
    got = {k: v["shape"] for k, v in sc["objects"].items()}
    assert got == {"orange_triangular_prism": "prism", "yellow_rectangular_block": "box",
                   "green_bridge_block": "arch", "yellow_cube": "box", "green_cylinder": "cylinder"}, got
    arch = sc["objects"]["green_bridge_block"]
    assert arch["size"][0] > arch["size"][1]          # 58 mm 方向是两腿方向


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
