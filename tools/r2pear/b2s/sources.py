"""b2s 演示数据源适配器：DROID / UMI / EgoDex → 统一的规范末端轨迹。

★ 规范末端坐标系（所有数据源都换算到这里，= menagerie robotiq_2f85 的 pinch site 坐标系）：
    原点 = 两指之间的捏合点；+z = 接近方向（从手腕指向指尖）；±y = 两指闭合轴
  夹爪量 g ∈ [0,1]：0 = 全开，1 = 全闭（= DROID gripper_position 的语义；sim ctrl = 255·g）

  DROID   parquet（LeRobot 移植版 cadene/droid_1.0.1）：关节真值 + gripper_position，直接用
  UMI     replay_buffer.zarr：robot0_eef_pos / robot0_eef_rot_axis_angle / robot0_gripper_width
          TCP 坐标轴约定按 UMI 仓库：z 沿夹爪伸出方向（**未在真实数据上验证**，见 UMI_TCP_TO_PINCH）
  EgoDex  hdf5：transforms/<joint> (N,4,4) 为 ARKit 世界系（**y 向上**），需转 z 向上；
          手 → 夹爪：捏合点 = 拇指尖与食指尖中点，开口 = 两者距离（**未在真实数据上验证**）

读文件（需要 zarr / h5py）与换算（只用 numpy）分开：本机 real2gym 环境没装 zarr/h5py，
且版本锁死（numpy<2），装之前先确认不会被顶掉 numpy。换算部分可以单独测试。
"""
import glob
import os

import numpy as np

GRIPPER_MAX_APERTURE = {"robotiq_2f85": 0.085, "umi": 0.08, "human": 0.10}
# UMI TCP → 规范末端系的旋转（列 = 规范系 x,y,z 在 UMI TCP 系中的表达）。默认恒等，未验证。
UMI_TCP_TO_PINCH = np.eye(3)
# ARKit（y 向上）→ z 向上：x 不变，y→z，z→-y
ARKIT_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)

# 数据集搜索根。ROBOLAB_DATA 指向权重/数据集所在目录（见 setup_env.sh），
# 额外目录用 B2S_DATASET_ROOTS（os.pathsep 分隔）追加。
_root = os.environ.get("ROBOLAB_DATA")
DEFAULT_ROOTS = ([os.path.join(_root, "datasets")] if _root else []) + \
    [p for p in os.environ.get("B2S_DATASET_ROOTS", "").split(os.pathsep) if p]


# ------------------------------------------------------------------ DROID
def load_droid_npz(path, q_ref, frames, fps):
    """Read compact modeling data only after checking its joint/time alignment."""
    import hashlib
    import json
    from pathlib import Path
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        q = z['observation.joint_position']
        if q_ref is None or q.shape != np.shape(q_ref) or not np.isfinite(q).all():
            raise ValueError('DROID NPZ requires an aligned robot joint trajectory')
        err = float(np.max(np.abs(q - q_ref)))
        if err > 1e-4:
            raise ValueError(f'DROID NPZ joints differ from Blender: {err:.6g} rad')
        if not np.array_equal(z['frame_index'], np.asarray(frames) - 1):
            raise ValueError('DROID NPZ frame_index does not match Blender frames')
        if not np.allclose(z['timestamp_s_nominal'], np.arange(len(q))/fps, atol=1e-7, rtol=0):
            raise ValueError('DROID NPZ nominal timebase differs from Blender fps')
        ep = {'observation.state.joint_position': q.copy()}
        for src, dst in [('observation.gripper_position', 'observation.state.gripper_position'),
                         ('action_dict.gripper_position', 'action.gripper_position')]:
            g = z[src].ravel()
            if g.shape != (len(q),) or not np.isfinite(g).all() or np.any((g < 0) | (g > 1)):
                raise ValueError(f'Invalid DROID gripper channel: {src}')
            ep[dst] = g.copy()
    manifest = path.with_name('source.json')
    if manifest.exists():
        source = json.loads(manifest.read_text())
        expected = source.get('trajectory', {}).get('sha256')
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('DROID trajectory checksum differs from source.json')
        ep['language_instruction'] = source.get('instructions', [''])
    return ep, f'NPZ 实测关节逐帧最大差 {err:.3g} rad；帧号、标称 fps 与 Blender 一致'


def find_droid_episode(episode_index, q_ref=None, roots=None, max_files=2000):
    """按 episode 编号找 parquet。q_ref 给定时用关节角逐帧比对确认是同一条（防止编号同名不同源）。
    返回 (path, ep_dict, evidence) 或 (None, None, 原因)。"""
    import droid_common as C
    roots = (roots or []) + DEFAULT_ROOTS
    pats = [f"episode_{int(episode_index):06d}.parquet", "episode.parquet"]
    cands = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for p in pats:
            cands += glob.glob(os.path.join(r, "**", p), recursive=True)[:max_files]
    seen, tried = set(), []
    for p in cands:
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        seen.add(rp)
        try:
            import pyarrow.parquet as pq
            cols = pq.read_schema(p).names
            if "episode_index" not in cols or "observation.state.joint_position" not in cols:
                continue
            idx = pq.read_table(p, columns=["episode_index"]).column(0).to_pylist()
            if int(idx[0]) != int(episode_index):
                continue
            ep = C.load_episode(p)
        except Exception as e:
            tried.append(f"{p}: {e}")
            continue
        J = ep["observation.state.joint_position"]
        if q_ref is not None:
            if len(J) != len(q_ref):
                tried.append(f"{p}: 帧数 {len(J)} ≠ {len(q_ref)}")
                continue
            err = float(np.abs(J - q_ref).max())
            if err > 1e-3:
                tried.append(f"{p}: 关节角最大差 {err:.4f} rad")
                continue
            return p, ep, f"episode_index={episode_index}，关节角与 blend 逐帧最大差 {err:.2e} rad"
        return p, ep, f"episode_index={episode_index}（未用关节角核对）"
    return None, None, f"没找到匹配的 parquet；尝试过 {len(seen)} 个：{tried[:3]}"


# ------------------------------------------------------------------ UMI
def umi_to_canonical(eef_pos, eef_rotvec, gripper_width, max_aperture=GRIPPER_MAX_APERTURE["umi"]):
    from scipy.spatial.transform import Rotation as Rot
    T = np.repeat(np.eye(4)[None], len(eef_pos), 0)
    T[:, :3, :3] = Rot.from_rotvec(eef_rotvec).as_matrix() @ UMI_TCP_TO_PINCH
    T[:, :3, 3] = eef_pos
    g = 1 - np.clip(np.asarray(gripper_width).ravel() / max_aperture, 0, 1)
    return T, g


def load_umi(zarr_path, episode=0):
    import zarr   # noqa: 需要 zarr<3
    root = zarr.open(zarr_path, mode="r")
    ends = np.asarray(root["meta/episode_ends"])
    s = 0 if episode == 0 else int(ends[episode - 1])
    e = int(ends[episode])
    d = root["data"]
    return umi_to_canonical(np.asarray(d["robot0_eef_pos"][s:e]), np.asarray(d["robot0_eef_rot_axis_angle"][s:e]),
                            np.asarray(d["robot0_gripper_width"][s:e]))


# ------------------------------------------------------------------ 人手（EgoDex / blend 手骨架共用）
def hand_to_canonical(wrist, thumb_tip, index_tip, knuckle=None, max_aperture=GRIPPER_MAX_APERTURE["human"]):
    """wrist (T,3)、thumb_tip (T,3)、index_tip (T,3)、knuckle (T,3) 可选（食指/中指掌指关节）。
    z = 手腕 → 捏合点（无 knuckle 时）或 手腕 → 掌指关节；y = 拇指尖 → 食指尖（正交化）。"""
    p = (thumb_tip + index_tip) / 2
    ap = np.linalg.norm(index_tip - thumb_tip, axis=1)
    z = (knuckle if knuckle is not None else p) - wrist
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    y = index_tip - thumb_tip
    y -= (y * z).sum(1, keepdims=True) * z
    n = np.linalg.norm(y, axis=1, keepdims=True)
    # 两指并拢时闭合轴退化：沿用上一帧
    for t in range(len(y)):
        if n[t] < 1e-4:
            y[t] = y[t - 1] if t else np.array([0, 1.0, 0])
            n[t] = 1
    y /= n
    x = np.cross(y, z)
    T = np.repeat(np.eye(4)[None], len(p), 0)
    T[:, :3, 0], T[:, :3, 1], T[:, :3, 2], T[:, :3, 3] = x, y, z, p
    g = 1 - np.clip(ap / max_aperture, 0, 1)
    return T, g, ap


EGODEX_JOINTS = {"right": ("rightHand", "rightThumbTip", "rightIndexFingerTip", "rightIndexFingerKnuckle"),
                 "left": ("leftHand", "leftThumbTip", "leftIndexFingerTip", "leftIndexFingerKnuckle")}


def load_egodex(h5_path, hand="right"):
    import h5py   # noqa
    with h5py.File(h5_path, "r") as f:
        tr = f["transforms"]
        P = [np.asarray(tr[j])[:, :3, 3] @ ARKIT_TO_ZUP.T for j in EGODEX_JOINTS[hand]]
        conf = {j: np.asarray(f["confidences"][j]) for j in EGODEX_JOINTS[hand] if "confidences" in f}
    T, g, ap = hand_to_canonical(*P)
    return T, g, {"aperture_m": ap, "confidence": conf}


def blend_hand_to_canonical(ex, armature, D):
    """blend 里的人手骨架：按骨骼名找手腕/拇指尖/食指尖/食指掌指关节（世界位姿由 extract 提供）。"""
    W = ex.E["armatures"][armature]["world"]
    names = list(W)
    low = {n: n.lower() for n in names}

    def find(*keys, prefer_last=True):
        c = [n for n in names if all(k in low[n] for k in keys)]
        return (c[-1] if prefer_last else c[0]) if c else None
    wrist = find("wrist", prefer_last=False) or find("hand", prefer_last=False)
    thumb = find("thumb", "tip") or find("thumb")
    index = find("index", "tip") or find("index")
    knuck = find("index", "knuckle", prefer_last=False) or find("index", prefer_last=False)
    D.add("hand.bones", {"wrist": wrist, "thumb_tip": thumb, "index_tip": index, "knuckle": knuck},
          "按骨骼名关键词匹配；不对请在 b2s_config.json 的 embodiment.bones 指定", warn=True)
    P = [np.array(W[b])[:, :3, 3] for b in (wrist, thumb, index)]
    K = np.array(W[knuck])[:, :3, 3] if knuck and knuck != index else None
    return hand_to_canonical(*P, knuckle=K)
