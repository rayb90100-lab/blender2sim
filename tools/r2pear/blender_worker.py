#!/usr/bin/env python3
"""Blender EEVEE 渲染子进程（供 web_ui 调用）。

为什么单独一个进程：
  bpy 与 MuJoCo/Renderer 同进程会互相干扰（EGL 上下文、GL 状态、线程安全），
  实测同一进程里先开 MuJoCo Renderer 再渲染 bpy 会报 GL 错误或直接崩。
  拆成子进程后各用各的 GL 上下文，稳定。

协议（stdin/stdout 各一行 JSON，NDJSON）：
  请求: {"pose": [{"matrix": [...16], ...}], "az":, "el":, "dist":, "w":, "h":, "quality":}
  响应: {"ok": true, "jpeg_b64": "..."}  或  {"ok": false, "err": "..."}

用法（由 web_ui 启动，不手动调）:
  MUJOCO_GL=egl python tools/r2pear/blender_worker.py --testdir <T>
"""
import argparse
import base64
import json
import math
import os
import sys

import numpy as np


def _fix_wrist_camera(spec, cam_objs, m, d):
    """把非 world 挂载的相机（wrist：mount="link7"）设成 MuJoCo 算出的世界位姿。

    MuJoCo 的 camera 坐标系与 Blender 相机坐标系一致（+X 右 / +Y 上 / 视向 −Z），
    所以 cam_xmat 可以直接当 matrix_world 的旋转部分 —— test4 的
    droid_stage8_blender.py 也是这么做的（`cams["wrist_left"].matrix_world = Matrix(M)`）。
    """
    import bpy
    from mathutils import Matrix

    names = {m.camera(i).name: i for i in range(m.ncam)}
    for cname, cam in spec["cameras"].items():
        if cam.get("mount", "world") == "world" or cname not in cam_objs:
            continue
        if cname not in names:
            raise SystemExit(f"spec 相机 {cname} 挂载在 {cam['mount']}，但 MuJoCo 模型里没有同名相机")
        i = names[cname]
        M = np.eye(4)
        M[:3, :3] = np.asarray(d.cam_xmat[i]).reshape(3, 3)
        M[:3, 3] = np.asarray(d.cam_xpos[i])
        cam_objs[cname].matrix_world = Matrix(M.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    a = ap.parse_args()
    T = a.testdir

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import spec as SP
    import blender_live as B

    spec = SP.load(os.path.join(T, "stage0_spec", "task_spec.json"))
    geoms, m, d = B.collect(T, spec, 0.0)
    sc, objs, cams = B.build_blender(geoms, spec, engine="BLENDER_EEVEE_NEXT", res_scale=1)

    import bpy
    from mathutils import Vector

    # 轨道相机：一个独立的 Blender 相机，由请求里的 az/el/dist 驱动
    cd = bpy.data.cameras.new("orbit")
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = 36.0
    cd.angle_y = math.radians(45.0)
    orbit = bpy.data.objects.new("orbit", cd)
    sc.collection.objects.link(orbit)
    sc.camera = orbit

    look = np.mean([o["matrix"][:3, 3] for o in geoms if o["body"] in spec["objects"]], axis=0) \
        if geoms else np.array([0.5, 0.0, 0.0])

    # 相机表：spec 里的相机由 build_blender 建好，这里补一个 az/el/dist 驱动的 orbit。
    cam_objs = dict(cams)
    cam_objs["orbit"] = orbit
    cam_fovy = {n: math.radians(c["fovy_deg"]) for n, c in spec["cameras"].items()}
    cam_fovy["orbit"] = math.radians(45.0)

    # **腕部相机必须用世界位姿**：spec 里的 wrist 记录的是 link7 局部系的相对位姿
    # （mount="link7", pos/quat 相对法兰）。build_blender 里按 mount=="world" 才写位姿，
    # 所以腕部相机停在原点、朝向默认 —— 实测画面朝天，只看到桌面底面和机械臂。
    # 修法：直接从 MuJoCo 取 cam_xpos / cam_xmat（它已经把局部位姿换算成世界位姿）。
    _fix_wrist_camera(spec, cam_objs, m, d)

    out = sys.stdout
    # 关键：bpy 在 **C 层直接往 fd 1 打印**渲染进度（"Fra:1 Mem:...", "Time: ..."），
    # Python 层的 sys.stdout 重定向挡不住它，会把 NDJSON 协议污染成
    # "Fra:1 Mem:..." + JSON 混排，父进程 readline() 拿到进度行 → JSONDecodeError。
    # 做法：先把 fd 1 复制到一个私有 fd 用于写响应，再把 fd 1 指向 /dev/null。
    _real_fd = os.dup(1)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 1)
    os.close(_devnull)
    out = os.fdopen(_real_fd, "w", buffering=1)
    sys.stdout = open(os.devnull, "w")     # 兜住 Python 层的 print

    # 子进程里没有 MuJoCo 模型，只能靠父进程每次把世界位姿推过来。
    # 早期版本不推：场景永远停在启动那一刻，oracle/手动操作后画面不动，
    # 表现为「切换了 Blender 但画面没变」。
    obj_objs = [o for o in objs if o is not None]
    obj_names = [o.name for o in obj_objs]
    print(f"[worker] 可更新对象 {len(obj_objs)} 个", file=sys.stderr)

    def _m4(flat):
        """16 个数（行主序）→ mathutils.Matrix。

        注意 mathutils.Matrix 吃的是 **2D 序列**（4 行 4 列），
        直接传扁平的 16 个数会报 "expects no args or a single arg containing 2-4 numeric sequences"。
        """
        import mathutils
        return mathutils.Matrix([list(flat[i * 4:(i + 1) * 4]) for i in range(4)])

    def sync_bodies(req):
        """req["bodies"]: {geom_or_obj_name: [16 floats]} —— 只更新位姿，不重建场景。"""
        mats = req.get("bodies") or {}
        if not mats:
            return
        for name, m16 in mats.items():
            ob = sc.objects.get(name)
            if ob is None:
                continue
            ob.matrix_world = _m4(m16)
        # 机械臂/腕部相机也要跟着动
        if req.get("wrist_matrix"):
            cam = cam_objs.get("wrist")
            if cam is not None:
                cam.matrix_world = _m4(req["wrist_matrix"])

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            out.write(json.dumps({"ok": False, "err": f"bad json: {e}"}) + "\n")
            out.flush()
            continue
        try:
            sync_bodies(req)
            cam_name = str(req.get("cam", "orbit"))
            w = int(req.get("w", 512))
            h = int(req.get("h", 288))
            sc.render.resolution_x, sc.render.resolution_y = w, h

            if cam_name == "orbit":
                az = math.radians(float(req.get("az", 36.0)))
                el = math.radians(float(req.get("el", -30.0)))
                dist = float(req.get("dist", 0.6))
                # MuJoCo 约定：azimuth 绕 z 轴，elevation 负值=从上往下看
                x = look[0] + dist * math.cos(el) * math.cos(az)
                y = look[1] + dist * math.cos(el) * math.sin(az)
                z = look[2] - dist * math.sin(el)
                orbit.location = (x, y, z)
                direction = Vector(tuple(look)) - Vector((x, y, z))
                orbit.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

            if cam_name not in cam_objs:
                raise KeyError(f"未知相机 {cam_name}，可选 {list(cam_objs)}")
            sc.camera = cam_objs[cam_name]

            sc.render.image_settings.file_format = "JPEG"
            sc.render.image_settings.quality = int(req.get("quality", 80))
            tmp = "/tmp/r2p_%s_%d.jpg" % (cam_name, os.getpid())
            sc.render.filepath = tmp
            bpy.ops.render.render(write_still=True)
            with open(tmp, "rb") as fh:
                data = fh.read()
            out.write(json.dumps({"ok": True, "cam": cam_name,
                                  "jpeg_b64": base64.b64encode(data).decode()}) + "\n")
        except Exception as e:
            import traceback
            out.write(json.dumps({"ok": False, "err": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"}) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
