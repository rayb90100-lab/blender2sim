"""xArm7 + 自带夹爪（menagerie ufactory_xarm7）的装配与末端约定。

**为什么需要它**：UMI 是人手持爪采集，手腕可以贴着桌面上方 3cm 作业。
Panda + Robotiq 2F-85 的「腕→捏合点」是 0.263 m，末端要离桌面 3cm 时腕部必须在桌面上方 29cm，
几何上无法复现（实测 link7 穿进桌面 18.5mm）。
xArm7 + 自带夹爪只有 0.123 m，末端的可达下探深度比 Panda 好 14cm。

与 droid_common.py 并存：那个模块是 DROID/Panda 专用的，test1/test5/test6 都依赖它，**不要改它**。
这里只提供 xArm7 的等价接口，供 b2s 的 retarget 换机型时使用。
"""
import os

import numpy as np

MENAGERIE = os.environ.get("MENAGERIE", "/media/data1/RoboLab_data/assets/mujoco_menagerie")
XARM7_XML = f"{MENAGERIE}/ufactory_xarm7/xarm7.xml"

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
N_ARM = 7
TCP_SITE = "link_tcp"
# 夹爪：menagerie 的 xarm7.xml 自带，两指由 left/right 驱动关节控制
GRIP_JOINTS = ("left_driver_joint", "right_driver_joint")
HOME = np.array([0.0, -0.5, 0.0, 1.2, 0.0, 1.0, 0.0])


def build_model():
    """返回 (model, data)。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(XARM7_XML)
    return m, mujoco.MjData(m)


def arm_qadr(m):
    return np.array([int(m.joint(j).dofadr[0]) for j in ARM_JOINTS])


def arm_actuators(m):
    """机械臂 7 个位置执行器的索引（不含夹爪）。"""
    out = []
    for j in ARM_JOINTS:
        try:
            out.append(int(m.actuator(j).id))
        except Exception:
            pass
    if not out:
        out = list(range(N_ARM))
    return np.array(out, int)


def grip_actuators(m):
    """夹爪执行器索引；menagerie 的 xarm7 用关节名或 driver 名，做一次兜底匹配。"""
    out = []
    for a in range(m.nu):
        n = (m.actuator(a).name or "").lower()
        if "gripper" in n or "finger" in n or "drive" in n:
            out.append(a)
    return np.array(out, int)


def pinch_pose(m, d):
    """TCP site 的世界位姿。"""
    sid = m.site(TCP_SITE).id
    T = np.eye(4)
    T[:3, :3] = d.site_xmat[sid].reshape(3, 3)
    T[:3, 3] = d.site_xpos[sid]
    return T


def wrist_to_pinch(m, d):
    """腕部（最后一个 arm body）到 TCP 的距离，用于评估「能否贴近桌面作业」。"""
    import mujoco
    mujoco.mj_forward(m, d)
    last = m.body(m.site_bodyid[m.site(TCP_SITE).id]).name
    b = m.body(last).id
    return float(np.linalg.norm(d.site_xpos[m.site(TCP_SITE).id] - d.xpos[b]))


if __name__ == "__main__":
    import mujoco
    m, d = build_model()
    d.qpos[arm_qadr(m)] = HOME
    mujoco.mj_forward(m, d)
    print("xArm7 装配成功")
    print("  腕→TCP 距离", round(wrist_to_pinch(m, d), 4), "m")
    print("  关节数", m.njnt, "执行器", m.nu)
    print("  arm 执行器", arm_actuators(m).tolist())
    print("  夹爪执行器", grip_actuators(m).tolist())
