#!/usr/bin/env python3
"""DROID (Panda + Robotiq 2F-85) 公共模块：模型装配、episode 读取、坐标约定。

装配用 mujoco.MjSpec.attach，不手工拼 XML：test3 手拼时踩过 include 嵌套、
default class 重名、actuator 不继承 biastype 等一串坑（DEVELOPMENT_LOG §34）。

资产（menagerie 官方，未改造几何）：
  franka_emika_panda/panda_nohand.xml   Panda 7-DoF，link7 下有 attachment_site（法兰 +0.107 m）
  robotiq_2f85/2f85.xml                 Robotiq 2F-85，site "pinch" 为两指间中心（base +0.145 m）
"""
import os

import numpy as np

MENAGERIE = os.environ.get("MENAGERIE", "/media/data1/RoboLab_data/assets/mujoco_menagerie")
PANDA_XML = f"{MENAGERIE}/franka_emika_panda/panda_nohand.xml"
GRIPPER_XML = f"{MENAGERIE}/robotiq_2f85/2f85.xml"
GRIP_PREFIX = "g_"

PANDA_JOINTS = [f"joint{i}" for i in range(1, 8)]

# ★ Robotiq 在法兰上的安装偏航修正（2026-11-17 由腕部相机确定）。
# menagerie panda_nohand 的 attachment body 自带 135° 绕 z 旋转（为原装 Franka Hand 设计）。
# DROID 用转接件装 Robotiq 2F-85，实际安装角为 90°，即需额外 -45°。
# 证据：腕部相机位姿由 cartesian_position × 手眼推出、与夹爪模型无关；
#      在 {0, ±45, ±90}° 中只有 -45° 时渲染夹爪两指与真实腕部画面的左右对称开合一致。
# 影响：夹爪闭合方向；不影响法兰 FK 与 pinch 点位置（pinch 在夹爪中轴上）。
MOUNT_YAW_DEG = -45.0


def build_spec(extra_worldbody_xml=None):
    """返回 MjSpec：Panda + Robotiq 2F-85（挂在 attachment_site）。"""
    import mujoco
    from scipy.spatial.transform import Rotation as R
    arm = mujoco.MjSpec.from_file(PANDA_XML)
    grip = mujoco.MjSpec.from_file(GRIPPER_XML)
    if MOUNT_YAW_DEG:
        att = arm.body("attachment")
        q0 = np.array(att.quat)                                    # wxyz
        q = (R.from_quat([q0[1], q0[2], q0[3], q0[0]]) * R.from_euler("z", MOUNT_YAW_DEG, degrees=True)).as_quat()
        att.quat = [q[3], q[0], q[1], q[2]]
    site = arm.site("attachment_site")
    arm.attach(grip, site=site, prefix=GRIP_PREFIX)
    return arm


def build_model():
    import mujoco
    spec = build_spec()
    m = spec.compile()
    return m, mujoco.MjData(m), spec


def load_episode(path):
    """读 DROID episode parquet，返回 dict[str, np.ndarray]。"""
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    out = {}
    for c in t.column_names:
        v = t.column(c).to_pylist()
        try:
            out[c] = np.array(v, dtype=float)
        except (ValueError, TypeError):
            out[c] = v
    return out


def euler_to_R(rpy, order="xyz"):
    from scipy.spatial.transform import Rotation as R
    return R.from_euler(order, rpy).as_matrix()


def pose6_to_T(p, order="xyz"):
    T = np.eye(4)
    T[:3, :3] = euler_to_R(p[3:6], order)
    T[:3, 3] = p[:3]
    return T


def arm_qadr(m):
    return [int(m.joint(j).qposadr[0]) for j in PANDA_JOINTS]
