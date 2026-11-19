"""位姿增量控制器（PEAR 协议）：动作给末端位姿与夹爪开口的「变化量」，控制器闭环执行到到达或受阻。

参照 PEAR pear/envs/robolab/controller.py（同为 DROID 的 Franka + Robotiq 2F-85）：
  - 15 Hz 控制步；路点每步最多前进 STEP_M / STEP_RAD；阻尼最小二乘 IK → 关节位置伺服目标
  - 连续 STALL_STEP_N 个控制步无进展 → 停在原地（outcome = stopped）
本项目新增（§61.2 压测）：
  - 外力保护：机器人受到的环境净接触力 > CONTACT_STOP_N 即停（模拟 Franka 碰撞反射；夹持力两指相互抵消，不触发）

工具点 = Robotiq 的 g_pinch site（两指垫中点），其坐标系 z = 指向指尖，y = 两指夹合方向。
状态读数：位置为基座系、z 从桌面量起（cm）；姿态为固定轴 roll/pitch/yaw（rad）；width = 指垫开口（cm）。
"""
import math

import numpy as np

SCHEMA = {"arm": ("x", "y", "z", "roll", "pitch", "yaw"), "gripper": ("width",)}
DECIMALS = {"x": 1, "y": 1, "z": 1, "roll": 2, "pitch": 2, "yaw": 2, "width": 1}

CONTROL_HZ = 15
STEP_M = 0.02
STEP_RAD = math.radians(10)
LEAD_M, LEAD_RAD = 2 * STEP_M, STEP_RAD
POS_TOL_M, ROT_TOL_RAD = 0.005, 0.02
PROGRESS_M, PROGRESS_RAD = 0.001, 0.005
STALL_STEP_N = 4
REST_MPS, REST_RADPS = 0.03, 0.15
GRIP_STILL_M, GRIP_STILL_STEP_N = 0.0005, 3
MAX_ACTION_STEP_N = 100
IK_DAMPING, IK_ITERS, IK_TOL_M = 0.01, 12, 1e-4
CONTACT_STOP_N = 30.0           # Franka 默认笛卡尔碰撞阈值量级（~20 N）；取 30 N，记为估计
PAD_GAP_CLOSED_M = 0.0149       # 指垫中心距在空夹闭合时的值；width = 中心距 − 此值（实测，§62）
PAD_BODIES = ("g_left_pad", "g_right_pad")
# Robotiq 连杆闭合时指尖沿夹爪轴线前伸；闭合动作把目标沿轴线回退此距离，使指尖在原处合拢（同 PEAR robolab 的 PAD_ADVANCE_M）
PAD_ADVANCE_M = 0.013            # 本模型实测（§62）：空夹闭合时指垫最低点比张开多下探 13.1 mm（PEAR robolab 同款 13.6 mm）


def rpy_to_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_to_rpy(R):
    sp = min(1.0, max(-1.0, -float(R[2, 0])))
    if abs(sp) < 1 - 1e-9:
        return np.array([math.atan2(R[2, 1], R[2, 2]), math.asin(sp), math.atan2(R[1, 0], R[0, 0])])
    return np.array([0.0, math.asin(sp), math.atan2(-R[0, 1], R[1, 1])])


def rot_vec(R_to, R_from):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(R_to @ R_from.T).as_rotvec()


class Robot:
    """对 MjModel/MjData 的机器人视图：读工具位姿、开口、外力；写关节目标。"""

    def __init__(self, m, d, table_top_z, arm_act, grip_act, arm_qadr, arm_dofadr):
        import mujoco
        self.mj, self.m, self.d = mujoco, m, d
        self.z0 = table_top_z
        self.arm_act, self.grip_act, self.qa, self.da = arm_act, grip_act, arm_qadr, arm_dofadr
        self.site = m.site("g_pinch").id
        self.pads = [m.body(n).id for n in PAD_BODIES]
        self.lo = m.jnt_range[[m.dof_jntid[i] for i in arm_dofadr], 0]
        self.hi = m.jnt_range[[m.dof_jntid[i] for i in arm_dofadr], 1]
        self.robot_bodies = set(self._robot_bodies())

    def _robot_bodies(self):
        root = self.m.body("link0").id
        out = []
        for b in range(self.m.nbody):
            x = b
            while x != 0 and x != root:
                x = self.m.body_parentid[x]
            if x == root:
                out.append(b)
        return out

    def tool(self):
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def tool_vel(self):
        v = np.zeros(6)
        self.mj.mj_objectVelocity(self.m, self.d, self.mj.mjtObj.mjOBJ_SITE, self.site, v, 0)
        return v[3:], v[:3]                                          # 线速度, 角速度（世界系）

    def gap(self):
        return float(np.linalg.norm(self.d.xpos[self.pads[0]] - self.d.xpos[self.pads[1]]))

    def width_cm(self):
        return max(0.0, (self.gap() - PAD_GAP_CLOSED_M) * 100)

    def external_force(self):
        """环境对机器人的净接触力（世界系向量）；夹持的两指对向力相互抵消。"""
        f6, tot = np.zeros(6), np.zeros(3)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if r1 == r2:
                continue                                              # 自碰撞或与机器人无关
            self.mj.mj_contactForce(self.m, self.d, i, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]                     # 接触系 → 世界系（作用在 geom2 上）
            tot += fw if r2 else -fw
        return tot

    def self_collision(self):
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if (self.m.geom_bodyid[c.geom1] in self.robot_bodies and self.m.geom_bodyid[c.geom2] in self.robot_bodies
                    and c.dist < -0.002):
                return True
        return False

    def state(self):
        p, R = self.tool()
        x, y, z = p[0] * 100, p[1] * 100, (p[2] - self.z0) * 100
        r, pi, ya = matrix_to_rpy(R)
        vals = {"x": x, "y": y, "z": z, "roll": r, "pitch": pi, "yaw": ya, "width": self.width_cm()}
        return {part: {q: round(float(vals[q]), DECIMALS[q]) + 0.0 for q in qs} for part, qs in SCHEMA.items()}

    def ik(self, p_t, R_t, q):
        """阻尼最小二乘 IK，从 q 出发迭代 IK_ITERS 次；用临时 MjData 不扰动仿真。"""
        m, mj = self.m, self.mj
        dd = mj.MjData(m)
        dd.qpos[:] = self.d.qpos
        J = np.zeros((6, m.nv))
        for _ in range(IK_ITERS):
            dd.qpos[self.qa] = q
            mj.mj_kinematics(m, dd)
            mj.mj_comPos(m, dd)
            p = dd.site_xpos[self.site]
            R = dd.site_xmat[self.site].reshape(3, 3)
            e = np.concatenate([p_t - p, rot_vec(R_t, R)])
            if np.linalg.norm(e[:3]) < IK_TOL_M and np.linalg.norm(e[3:]) < 10 * IK_TOL_M:
                break
            mj.mj_jacSite(m, dd, J[:3], J[3:], self.site)
            Ja = J[:, self.da]
            dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + IK_DAMPING ** 2 * np.eye(6), e)
            q = np.clip(q + dq, self.lo, self.hi)
        return q


class PoseController:
    def __init__(self, robot, action):
        self.r = robot
        p, R = robot.tool()
        st = robot.state()
        a = action["arm"]
        tgt = {k: st["arm"][k] + float(a[k]) for k in SCHEMA["arm"]}
        self.p_goal = np.array([tgt["x"] / 100, tgt["y"] / 100, tgt["z"] / 100 + robot.z0])
        self.R_goal = rpy_to_matrix(tgt["roll"], tgt["pitch"], tgt["yaw"])
        dw = float(action["gripper"]["width"])
        self.grip_cmd = None if dw == 0 else (0.0 if dw > 0 else 255.0)      # 开到底 / 闭到底（同 LIBERO 用法）
        if self.grip_cmd == 255.0 and robot.d.ctrl[robot.grip_act] < 128:      # 由开到合：沿夹爪轴线回退
            self.p_goal = self.p_goal - self.R_goal[:, 2] * PAD_ADVANCE_M
        self.wp_p, self.wp_R = p.copy(), R.copy()
        self.q_cmd = robot.d.qpos[robot.qa].copy()

    def _advance_waypoint(self):
        p, R = self.r.tool()
        if np.linalg.norm(self.wp_p - p) > LEAD_M or np.linalg.norm(rot_vec(self.wp_R, R)) > LEAD_RAD:
            return
        dp = self.p_goal - self.wp_p
        n = np.linalg.norm(dp)
        self.wp_p = self.wp_p + (dp if n <= STEP_M else dp / n * STEP_M)
        rv = rot_vec(self.R_goal, self.wp_R)
        ang = np.linalg.norm(rv)
        from scipy.spatial.transform import Rotation as Rot
        step = rv if ang <= STEP_RAD else rv / ang * STEP_RAD
        self.wp_R = Rot.from_rotvec(step).as_matrix() @ self.wp_R

    def move(self, budget, on_control_step=None):
        """返回 (outcome, n_control_steps, stop_reason)。outcome ∈ reached | stopped | budget | timeout。"""
        r, mj = self.r, self.r.mj
        sub = int(round(1 / CONTROL_HZ / r.m.opt.timestep))
        if self.grip_cmd is not None:
            r.d.ctrl[r.grip_act] = self.grip_cmd
        least_p = least_r = math.inf
        stall = grip_still = 0
        stopping, reason = False, None
        prev_gap = r.gap()
        f_start = float(np.linalg.norm(r.external_force()))
        for k in range(min(MAX_ACTION_STEP_N, budget)):
            if not stopping:
                self._advance_waypoint()
                self.q_cmd = r.ik(self.wp_p, self.wp_R, r.d.qpos[r.qa].copy())
            r.d.ctrl[r.arm_act] = self.q_cmd
            fmax = 0.0
            p_now, _ = r.tool()
            u = self.wp_p - p_now
            un = np.linalg.norm(u)
            for _ in range(sub):
                mj.mj_step(r.m, r.d)
                F = r.external_force()
                fmax = max(fmax, float(np.linalg.norm(F)))
                # 只拦「顶着障碍继续推」：外力在指令运动方向上的反向分量超阈值；纯转动/纯夹爪动作看外力增量
                if un > 1e-3:
                    push = -float(F @ u) / un
                else:
                    push = float(np.linalg.norm(F)) - f_start
                if not stopping and (push > CONTACT_STOP_N or r.self_collision()):
                    stopping, reason = True, "contact_force" if push > CONTACT_STOP_N else "self_collision"
                    self.q_cmd = r.d.qpos[r.qa].copy()                 # 反射：停在当前关节位置
                    r.d.ctrl[r.arm_act] = self.q_cmd
            if on_control_step:
                on_control_step(fmax)
            p, R = r.tool()
            lin, ang = r.tool_vel()
            pe, re = float(np.linalg.norm(self.p_goal - p)), float(np.linalg.norm(rot_vec(self.R_goal, R)))
            slow = np.linalg.norm(lin) < REST_MPS and np.linalg.norm(ang) < REST_RADPS
            g = r.gap()
            grip_still = grip_still + 1 if abs(g - prev_gap) < GRIP_STILL_M else 0
            prev_gap = g
            grip_done = self.grip_cmd is None or grip_still >= GRIP_STILL_STEP_N
            reached = pe < POS_TOL_M and re < ROT_TOL_RAD
            if slow and grip_done and (reached or stopping):
                return ("stopped" if stopping else "reached"), k + 1, reason
            if stopping:
                continue
            if reached or pe < least_p - PROGRESS_M or re < least_r - PROGRESS_RAD:
                least_p, least_r, stall = min(least_p, pe), min(least_r, re), 0
            else:
                stall += 1
            if not reached and stall >= STALL_STEP_N:
                stopping, reason = True, "no_progress"
                self.q_cmd = r.d.qpos[r.qa].copy()
        n = min(MAX_ACTION_STEP_N, budget)
        return ("budget" if n >= budget else "timeout"), n, reason
