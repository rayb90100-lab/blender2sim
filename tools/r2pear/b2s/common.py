"""b2s 公共工具：读 extract.json、矩阵、命名、颜色空间、决策记录。"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R2P = os.path.dirname(HERE)
TOOLS = os.path.dirname(R2P)
for p in (HERE, R2P, TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)

# b2s 会用这个解释器起 bpy 子进程（extract_blend.py / validate.py 等）。
# 必须在**装了 bpy + mujoco 的那个环境**里，默认取当前解释器；
# 需要指定别的（比如带 GPU 的 conda env）就设 B2S_PYTHON。
PY = os.environ.get("B2S_PYTHON") or sys.executable


class Extract:
    """extract.json 的只读视图。所有矩阵返回 np.ndarray(4,4)，world 系 = blend 世界系。"""

    def __init__(self, path):
        self.path = path
        self.E = json.load(open(path))
        self.O = self.E["objects"]
        self.frames = self.E["frames"]
        self.T = len(self.frames)
        self.fps = float(self.E["scene"]["fps"])

    def M0(self, name):
        return np.array(self.O[name]["matrix_world"])

    def M(self, name):
        """(T,4,4)：动态对象逐帧，静态对象复制首帧。"""
        if name in self.E["dynamic"]:
            return np.array(self.E["dynamic"][name])
        return np.repeat(self.M0(name)[None], self.T, 0)

    def is_dynamic(self, name):
        return name in self.E["dynamic"]

    def children(self, name):
        return [n for n, o in self.O.items() if o["parent"] == name]

    def descendants(self, name):
        out, st = [], [name]
        while st:
            c = self.children(st.pop())
            out += c
            st += c
        return out

    def ancestors(self, name):
        out = []
        p = self.O[name]["parent"]
        while p:
            out.append(p)
            p = self.O[p]["parent"]
        return out

    def verts_world(self, name, frame_idx=0):
        v = np.array(self.E["meshes"][name]["verts_local"], float)
        M = self.M(name)[frame_idx] if self.is_dynamic(name) else self.M0(name)
        return v @ M[:3, :3].T + M[:3, 3]


def scale_of(M):
    return np.linalg.norm(M[:3, :3], axis=0)


def rot_of(M):
    return M[:3, :3] / scale_of(M)


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    o = np.eye(4)
    o[:3, :3] = R.T
    o[:3, 3] = -R.T @ t
    return o


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def quat_wxyz(R):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(R).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def srgb_from_linear(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def slug(name, used):
    """'Green cylinder | 被抓取物' -> 'green_cylinder'。纯非 ASCII 名 -> obj_k。保证唯一。"""
    base = name.split("|")[0]
    s = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
    if not s:
        s = "obj"
    s0, k = s, 1
    while s in used:
        k += 1
        s = f"{s0}_{k}"
    used.add(s)
    return s


class Decisions:
    """每个自动判断都记一条：选了什么、依据、是否为兜底。最终写进 decisions.json，并在控制台打印。"""

    def __init__(self):
        self.items = []

    def add(self, key, choice, evidence, fallback=False, warn=False):
        self.items.append({"key": key, "choice": choice, "evidence": evidence, "fallback": bool(fallback),
                           "warn": bool(warn)})
        tag = "兜底" if fallback else ("警告" if warn else "决策")
        print(f"  [{tag}] {key}: {choice}  ← {evidence}")

    def dump(self, path):
        json.dump(self.items, open(path, "w"), indent=1, ensure_ascii=False, default=_np_default)


def _np_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def dump_json(obj, path):
    json.dump(obj, open(path, "w"), indent=1, ensure_ascii=False, default=_np_default)


def load_config(path):
    """可选的人工覆盖：b2s_config.json。任何字段缺省都走自动判断。"""
    if path and os.path.exists(path):
        return json.load(open(path))
    return {}
