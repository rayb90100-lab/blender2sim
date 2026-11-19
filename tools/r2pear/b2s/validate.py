"""b2s 第 4 步：验证（只读 task_spec.json + demo.npz，与 blend 无关）。

V1 静置   reset 后 3 s：各物体漂移、末速度、最深穿透
V2 回放   演示关节 + 夹爪按 demo fps 线性插值驱动位置伺服，物体运动全由物理产生；
          结束后静置 1 s 判谓词。同时逐帧记录物体位姿，与 blend 参考轨迹（仅有效帧）比较
V3 对照   每个有源视频的相机：源视频帧 | MuJoCo 回放帧 | 50% 叠加，输出对照视频 + 关键帧拼图

用法: python tools/r2pear/b2s/validate.py --simdir <D> [--videos <dir>] [--no-video]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dump_json  # noqa: E402
import droid_common as C  # noqa: E402
import predicates as PR  # noqa: E402
import sim as S  # noqa: E402
import spec as SP  # noqa: E402

FF = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def penetration(m, d):
    n = min(int(d.ncon), len(d.contact))
    return min([d.contact[i].dist for i in range(n)] + [0.0])


def obj_pose(m, d, n):
    b = m.body(n).id
    T = np.eye(4)
    T[:3, :3] = d.xmat[b].reshape(3, 3)
    T[:3, 3] = d.xpos[b]
    return T


def v1_settle(spec, seconds=3.0):
    import mujoco
    m, d = S.build(spec)
    S.reset(m, d, spec, settle_s=0.0)
    p0 = {n: d.xpos[m.body(n).id].copy() for n in spec["objects"]}
    pen = 0.0
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        pen = min(pen, penetration(m, d))
    out = {}
    for n in spec["objects"]:
        j = m.joint(f"{n}_free")
        v = d.qvel[j.dofadr[0]:j.dofadr[0] + 6]
        out[n] = {"drift_mm": float(np.linalg.norm(d.xpos[m.body(n).id] - p0[n]) * 1000),
                  "speed_mm_s": float(np.linalg.norm(v[:3]) * 1000), "spin_rad_s": float(np.linalg.norm(v[3:]))}
    worst = max(o["drift_mm"] for o in out.values())
    return {"objects": out, "max_drift_mm": worst, "min_contact_dist_mm": pen * 1000,
            "ok": bool(worst < 2.0 and pen > -0.003 and all(o["speed_mm_s"] < 5 for o in out.values()))}


def v2_replay(spec, demo, renderers=None, on_frame=None):
    import mujoco
    m, d = S.build(spec)
    S.reset(m, d, spec)
    arm, g = S.arm_actuators(m), S.grip_actuator(m)
    qa = C.arm_qadr(m)
    J = demo["q"]
    G = demo["g_cmd"] if "g_cmd" in demo else demo["g"]       # 回放用指令通道（见 demo.py）
    fps = float(demo["fps"])
    sub = int(round(1 / fps / m.opt.timestep))
    names = list(spec["objects"])
    traj = {n: [obj_pose(m, d, n)] for n in names}
    track, pen = [], 0.0
    if on_frame:
        on_frame(0, m, d)
    for f in range(len(J) - 1):
        for k in range(sub):
            al = k / sub
            d.ctrl[arm] = (1 - al) * J[f] + al * J[f + 1]
            d.ctrl[g] = 255 * ((1 - al) * G[f] + al * G[f + 1])
            mujoco.mj_step(m, d)
            pen = min(pen, penetration(m, d))
        track.append(float(np.abs(d.qpos[qa] - J[f + 1]).max()))
        for n in names:
            traj[n].append(obj_pose(m, d, n))
        if on_frame:
            on_frame(f + 1, m, d)
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    ok, detail = PR.evaluate(m, d, spec)
    # 与 blend 参考轨迹比较（仅有效帧；参考是 blend 作者的近似，不是真值）
    cmp = {}
    for n in names:
        ref = demo.get(f"obj__{n}")
        if ref is None:
            continue
        val = demo[f"valid__{n}"]
        T_base = np.array(spec["robot"]["T_world_base"])
        Tb_inv = np.linalg.inv(T_base)
        sim_p = np.array([T[:3, 3] for T in traj[n]])
        cl = demo.get(f"cloc__{n}", np.zeros(3))
        ref_p = np.array([(Tb_inv @ M)[:3, :3] @ cl + (Tb_inv @ M)[:3, 3] for M in ref])
        e = np.linalg.norm(sim_p - ref_p, axis=1)[val]
        cmp[n] = {"median_mm": float(np.median(e) * 1000), "max_mm": float(e.max() * 1000),
                  "final_mm": float(e[-1] * 1000) if val[-1] else None}
    return {"success": bool(ok), "predicates": detail, "min_contact_dist_mm": pen * 1000,
            "joint_track_err_deg": {"median": float(np.degrees(np.median(track))),
                                    "max": float(np.degrees(max(track)))},
            "vs_blend_reference": cmp}, traj


def read_video(path, W, H):
    p = subprocess.run([FF, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-"],
                       capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8)
    return a.reshape(-1, H, W, 3)


def ffmpeg_ready():
    """ffmpeg 是否可用。缺它时 V3 无法出片，但 V1/V2 仍然有效——不能因此整条命令崩掉。"""
    return shutil.which(FF) is not None or os.path.isabs(FF) and os.access(FF, os.X_OK)


def v3_compare(spec, demo, simdir, videos_dir):
    import mujoco
    W, H = spec["render"]["width"], spec["render"]["height"]
    cams = {c: v for c, v in spec["cameras"].items() if v.get("source_camera")}
    src = {}
    for c, v in cams.items():
        p = os.path.join(videos_dir, f"{v['source_camera']}.mp4")
        if os.path.exists(p):
            src[c] = read_video(p, W, H)
    if not src:
        return {"skipped": f"{videos_dir} 下没有与相机同名的源视频"}
    frames = {c: [] for c in src}
    rend = {}

    def on_frame(f, m, d):
        if not rend:
            for c in src:
                rend[c] = mujoco.Renderer(m, H, W)
        for c in src:
            rend[c].update_scene(d, camera=c)
            frames[c].append(rend[c].render().copy())
    _, _ = v2_replay(spec, demo, on_frame=on_frame)
    out = {}
    os.makedirs(simdir, exist_ok=True)
    rows = []
    for c in src:
        n = min(len(src[c]), len(frames[c]))
        A, B = src[c][:n], np.array(frames[c][:n])
        mix = ((A.astype(np.uint16) + B) // 2).astype(np.uint8)
        vid = np.concatenate([A, B, mix], axis=2)
        p = os.path.join(simdir, f"compare_{c}.mp4")
        proc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                 f"{vid.shape[2]}x{H}", "-r", str(spec["step_budget"]["control_hz"]), "-i", "-",
                                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", p], stdin=subprocess.PIPE)
        proc.stdin.write(vid.tobytes())
        proc.stdin.close()
        proc.wait()
        key = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
        rows.append(np.concatenate([vid[k] for k in key], axis=0))
        out[c] = {"video": p, "frames": n, "keyframes": key}
    from PIL import Image
    Image.fromarray(np.concatenate(rows, axis=1)).save(os.path.join(simdir, "compare_keyframes.jpg"), quality=88)
    for r in rend.values():
        r.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simdir", required=True)
    ap.add_argument("--videos", default=None)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    spec = SP.load(os.path.join(a.simdir, "stage0_spec", "task_spec.json"))
    demo = dict(np.load(os.path.join(a.simdir, "stage0_spec", "demo.npz")))
    vd = os.path.join(a.simdir, "stage1_validate")
    os.makedirs(vd, exist_ok=True)
    rep = {"V1_settle": v1_settle(spec)}
    print("V1 静置:", json.dumps({k: rep["V1_settle"][k] for k in ("max_drift_mm", "min_contact_dist_mm", "ok")}))
    if len(demo["q"]):
        r2, _ = v2_replay(spec, demo)
        rep["V2_replay"] = r2
        print("V2 回放:", json.dumps({k: r2[k] for k in ("success", "min_contact_dist_mm", "joint_track_err_deg")},
                                    ensure_ascii=False))
        for p in r2["predicates"]:
            print("   ", p)
        print("   与 blend 参考轨迹:", r2["vs_blend_reference"])
        # V3 出片要 ffmpeg。缺了不该让 V1/V2 的结果一起丢掉（原来会抛 FileNotFoundError，
        # 且 validate.json 根本没写成），所以这里先判可用性，缺了明确记进报告。
        if a.no_video:
            rep["V3_compare"] = {"skipped": "--no-video"}
        elif not a.videos:
            rep["V3_compare"] = {"skipped": "没给 --videos"}
        elif not ffmpeg_ready():
            rep["V3_compare"] = {"skipped": f"找不到 ffmpeg（FFMPEG={FF}）；"
                                            "装一个或 export FFMPEG=/path/to/ffmpeg 后重跑"}
        else:
            rep["V3_compare"] = v3_compare(spec, demo, vd, a.videos)
            print("V3 对照:", rep["V3_compare"])
        if "skipped" in rep.get("V3_compare", {}):
            print("V3 对照: 跳过 —", rep["V3_compare"]["skipped"])
    else:
        rep["V2_replay"] = {"skipped": "没有关节轨迹（需先 retarget）"}
    # 没重出 V3 时保留上一轮已有的记录，避免一次 --no-video 就把证据里的对照视频信息抹掉
    vp = os.path.join(vd, "validate.json")
    if "skipped" in rep.get("V3_compare", {}) and os.path.exists(vp):
        try:
            old = json.load(open(vp))
            if isinstance(old.get("V3_compare"), dict) and "skipped" not in old["V3_compare"]:
                rep["V3_compare"] = old["V3_compare"]
                print("V3 对照: 沿用上一轮已有的记录（未重新生成）")
        except (OSError, ValueError):
            pass
    dump_json(rep, vp)


if __name__ == "__main__":
    main()
