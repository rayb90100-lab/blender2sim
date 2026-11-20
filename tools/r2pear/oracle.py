"""特权脚本解题器：读仿真真值位姿，但只通过 PEAR 协议动作（位姿增量 + 夹爪）执行。
用途：证明某初始状态在**环境自身接口**下可解（可解性证明），不用于训练。
适用任务模板：pick(obj) → place_on(support)。
"""
import math

import numpy as np

import controller as CT
import spec as SP

APPROACH_CM = 10.0
GRASP_Z_MIN_CM = 2.5          # 工具点距桌面下限：指尖比工具点低约 2 cm，再低会撞桌（外力保护会停）
PLACE_CLEAR_CM = 0.8          # 放置时绿方块底面高于支撑顶面的余量


def _obj(env, name):
    m, d = env.m, env.d
    b = m.body(name).id
    p = d.xpos[b].copy()
    R = d.xmat[b].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    return p, yaw, env.spec["objects"][name]


def _grip_yaw(obj_yaw, cur_yaw, quarter_turns=True):
    """指垫夹合方向 = 物体局部 y：工具 yaw ≡ 物体 yaw (mod 90°)；取离当前最近、且在 ±π/2 内的。"""
    cands = [obj_yaw + k * math.pi / 2 for k in range(-4, 5) if quarter_turns or k % 2 == 0]
    cands = [c for c in cands if abs(c) <= math.pi / 2 + 1e-6] or cands
    return min(cands, key=lambda c: abs(c - cur_yaw))


def _offset_world(env, name, key):
    """Optional metric point offset in the object's own body frame."""
    offset = np.asarray(env.spec['objects'][name].get(key, [0., 0., 0.]), dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError(f'{name}.{key} must contain three finite distances in metres')
    rotation = env.d.xmat[env.m.body(name).id].reshape(3, 3)
    return rotation @ offset


def _goto(env, x, y, z, yaw, dwidth=0.0, log=None):
    st = env.robot.state()["arm"]
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
    act = {"arm": {"x": x - st["x"], "y": y - st["y"], "z": z - st["z"],
                   "roll": wrap(math.pi - st["roll"]), "pitch": -st["pitch"], "yaw": wrap(yaw - st["yaw"])},
           "gripper": {"width": dwidth}}
    act["arm"] = {k: round(float(v), 3) for k, v in act["arm"].items()}
    res = env.step(act)
    if log is not None:
        log.append({"action": act, "outcome": res["outcome"], "stop_reason": res.get("stop_reason"),
                    "state": res["state"], "used": res["used_control_steps"]})
    return res


def solve(env, obj, support, log=None):
    z0 = env.robot.z0
    p, oyaw, o = _obj(env, obj)
    # Long thin objects cannot be grasped across their long edge. For shapes
    # where both horizontal dimensions fit, preserve the original yaw choice.
    quarter_turns = o["shape"] == "cylinder" or max(o["size"][:2]) <= 0.085
    off_center = 'grasp_offset_local_m' in o
    if off_center:
        quarter_turns = False
    yaw = _grip_yaw(oyaw, env.robot.state()["arm"]["yaw"], quarter_turns)
    grasp = p + _offset_world(env, obj, 'grasp_offset_local_m')
    gx, gy = grasp[0] * 100, grasp[1] * 100
    gz = max((grasp[2] - z0) * 100, o.get("grasp_z_min_cm", GRASP_Z_MIN_CM))
    steps = [
        ("open_above", lambda: _goto(env, gx, gy, gz + APPROACH_CM, yaw, +1.0, log)),
        ("descend", lambda: _goto(env, gx, gy, gz, yaw, 0.0, log)),
        ("close", lambda: _goto(env, gx, gy, gz, yaw, -1.0, log)),
    ]
    res = None
    for name, f in steps:
        res = f()
        if res["done"]:
            return res
    # 抬起与放置目标基于当前（可能被扰动后的）支撑位姿
    sp, syaw, so = _obj(env, support)
    top_cm = (sp[2] - z0) * 100 + SP.height(so) * 50
    lift = max(top_cm + SP.height(o) * 100 + APPROACH_CM, gz + APPROACH_CM)
    held_off = gz - ((p[2] - z0) * 100)                      # 工具点相对物体中心的竖直偏移（抓得偏高时为正）
    place_z = top_cm + SP.height(o) * 50 + PLACE_CLEAR_CM + held_off
    pyaw = _grip_yaw(syaw, env.robot.state()["arm"]["yaw"], quarter_turns)
    res = _goto(env, gx, gy, lift, yaw, 0.0, log)
    if res['done']:
        return res
    place = sp + _offset_world(env, support, 'place_offset_local_m')
    px, py = place[:2] * 100
    place_z += (place[2] - sp[2]) * 100
    if off_center:
        # Preserve the grasp actually achieved, including lateral displacement
        # during closure. Place the object center at the requested support point,
        # rather than putting the off-center tool point there.
        tool_p, tool_R = env.robot.tool()
        object_p, _, _ = _obj(env, obj)
        relative = tool_R.T @ (object_p - tool_p)
        desired_object = place.copy()
        desired_object[2] += SP.height(so)/2 + SP.height(o)/2 + PLACE_CLEAR_CM/100
        target = desired_object - CT.rpy_to_matrix(math.pi, 0., pyaw) @ relative
        px, py = target[:2] * 100
        place_z = (target[2] - z0) * 100
    for name, f in [
        ("transfer", lambda: _goto(env, px, py, lift, pyaw, 0.0, log)),
        ("lower", lambda: _goto(env, px, py, place_z, pyaw, 0.0, log)),
        ("release", lambda: _goto(env, px, py, place_z, pyaw, +1.0, log)),
        ("retreat", lambda: _goto(env, px, py, place_z + APPROACH_CM, pyaw, 0.0, log)),
    ]:
        res = f()
        if res["done"]:
            return res
    return res
