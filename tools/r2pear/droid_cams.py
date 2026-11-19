"""DROID 相机模型（通用）：外部相机 = 数据集外参 · 精修小旋转（camera_refined.json），腕部 = FK(link7, t) · rel。
所有投影均为 OpenCV 约定（x 右 y 下 z 前），像素坐标基于 320×180。"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import droid_common as C  # noqa: E402

W, H = 320, 180
EXT = ("exterior_1_left", "exterior_2_left")
FLIP = np.diag([1.0, -1.0, -1.0])


def _rot(deg):
    from scipy.spatial.transform import Rotation as R
    return R.from_euler("xyz", deg, degrees=True).as_matrix()


class Cams:
    def __init__(self, testdir, ep=None):
        import mujoco
        self.mj = mujoco
        self.T = testdir
        self.ep = ep or C.load_episode(os.path.join(testdir, "input", "episode.parquet"))
        od = os.path.join(testdir, "stage0_replay")
        # camera_refined.json 是 test6 才引入的外参精修步骤；test4 无此文件，回退原始外参
        rp = os.path.join(od, "camera_refined.json")
        self.refined = json.load(open(rp)) if os.path.exists(rp) else {}
        self.refined_status = "refined" if self.refined else "dataset_extrinsics_unrefined"
        # 6-DoF 标定（优先于 camera_refined 的小旋转）：
        #   机器人自标定 camera_calib_robot.json —— 有效帧 ≥ 30 且质心误差中位 ≤ 2 px 才采纳
        #   物体 PnP 标定 camera_calib_objects.json —— 其 accepted 为真才采纳
        self.calib = {}
        pr = os.path.join(od, "camera_calib_robot.json")
        if os.path.exists(pr):
            for c, v in json.load(open(pr)).items():
                e = v["centroid_err_px"]
                if e["frames"] >= 30 and e["after_median"] <= 2.0:
                    self.calib[c] = {**v, "source": "robot"}
        po = os.path.join(od, "camera_calib_objects.json")
        if os.path.exists(po):
            for c, v in json.load(open(po)).items():
                if c not in self.calib and v.get("accepted"):
                    self.calib[c] = {**v, "source": "objects"}
        self.K = json.load(open(os.path.join(od, "intrinsics_adopted.json")))
        self.wrist_rel = json.load(open(os.path.join(od, "wrist_cam_rel_link7.json")))
        self.m, self.d, _ = C.build_model()
        self.qa = C.arm_qadr(self.m)

    def source(self, cam):
        if cam in self.calib:
            return self.calib[cam]["source"]
        return "stage0_rot" if cam in self.refined else ("dataset" if cam in EXT else "wrist_rel")

    def fx(self, cam):
        if cam in self.calib:
            return self.calib[cam]["fx"]
        if cam in self.refined:
            return self.refined[cam]["fx"]
        if cam in EXT:
            # 无精修时：优先 intrinsics_adopted 的逐相机 fx，其次全局 fx
            return self.K.get("per_camera", {}).get(cam, self.K["fx"])
        return self.K["wrist_fx"]

    def pose_cv(self, cam, frame=0):
        """返回 (R_cv, t)：相机系 → 基座系，OpenCV 轴。"""
        if cam in self.calib:
            return np.array(self.calib[cam]["R_cv"]), np.array(self.calib[cam]["t"])
        if cam in EXT:
            ext = self.ep[f"camera_extrinsics.{cam}"][0]
            Rcv = C.euler_to_R(ext[3:], "xyz")
            if cam in self.refined:
                Rcv = Rcv @ _rot(self.refined[cam]["delta_rot_deg_xyz_cv"])
            return Rcv, np.asarray(ext[:3], float)
        from scipy.spatial.transform import Rotation as R
        self.d.qpos[self.qa] = self.ep["observation.state.joint_position"][frame]
        self.mj.mj_kinematics(self.m, self.d)
        b = self.m.body("link7").id
        R7, p7 = self.d.xmat[b].reshape(3, 3), self.d.xpos[b]
        q = self.wrist_rel["quat_wxyz"]
        Rrel = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        Rmj = R7 @ Rrel
        return Rmj @ FLIP, p7 + R7 @ np.asarray(self.wrist_rel["pos"])

    def mj_cam(self, cam, frame=0):
        """MuJoCo 世界相机参数 (pos, quat_wxyz, fovy_deg)；腕部返回相对 link7 的参数。"""
        from scipy.spatial.transform import Rotation as R
        fov = float(np.degrees(2 * np.arctan(H / 2 / self.fx(cam))))
        if cam in EXT:
            Rcv, t = self.pose_cv(cam)
            q = R.from_matrix(Rcv @ FLIP).as_quat()
            return t.tolist(), [q[3], q[0], q[1], q[2]], fov
        return self.wrist_rel["pos"], self.wrist_rel["quat_wxyz"], fov

    def project(self, cam, P, frame=0):
        Rcv, t = self.pose_cv(cam, frame)
        P = np.atleast_2d(P)
        pc = (P - t) @ Rcv
        f = self.fx(cam)
        uv = np.stack([W / 2 + f * pc[:, 0] / pc[:, 2], H / 2 + f * pc[:, 1] / pc[:, 2]], 1)
        uv[pc[:, 2] <= 0.02] = np.nan
        return uv

    def pinch(self, frame):
        import mujoco
        m, d, _ = C.build_model() if not hasattr(self, "_pm") else self._pm
        self._pm = (m, d, None)
        d.qpos[C.arm_qadr(m)] = self.ep["observation.state.joint_position"][frame]
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[m.site("g_pinch").id].copy()
