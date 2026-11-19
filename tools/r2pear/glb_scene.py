#!/usr/bin/env python3
"""GLB 场景 -> MuJoCo 可交互环境（+ Blender 精修渲染）。

这是 r2pear 管线的「几何来源」替换：此前几何来自多视角 box 拟合（test5-8），
外参和尺度都不可信（test8 外参精修 IoU 仅 0.475，物体 IoU 全 0）。GLB 自带
米制尺度和照片贴图，绕开了那条最容易失败的路径。

**沿用 sim.py 的既有约定**（不另起一套）：
  - 视觉与碰撞分离：视觉 geom 一律 contype=0，不参与接触
  - 碰撞体用解析几何（box/capsule）拼，不用 mesh 凸包 —— 理由见 DEVELOPMENT_LOG
    §63.5/§63.10：凸包与平面接触会退化（棱柱静置仍抖 |w|=2.69、穿透 5.85 mm）

用法:
  PY=/home/ubuntu/mambaforge/envs/real2gym/bin/python
  source setup_env.sh

  # 1. 导出（内部调 Blender）
  $PY tools/r2pear/glb_scene.py export \
      --glb GLB/办公室场景.glb --out /media/data1/RoboLab_data/glb_env \
      --bodies "鼠标=Wireless mouse / molded shell, wheel and side buttons" \
               "耳机盒=Earbud case / hinged black shell" \
      --static "桌面=Near worktop"

  # 2. 建 MuJoCo 场景 + 物理 + 渲染
  MUJOCO_GL=egl $PY tools/r2pear/glb_scene.py build \
      --out /media/data1/RoboLab_data/glb_env --drop "鼠标=0.15" "耳机盒=0.20"

  # 3. 只重建 XML（改了碰撞参数后）
  $PY tools/r2pear/glb_scene.py xml --out /media/data1/RoboLab_data/glb_env
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_BLENDER_PY = os.environ.get("B2S_PYTHON") or sys.executable
EXTRACT = os.path.join(HERE, "glb_extract.py")

# 场景视觉的默认材质颜色（GLB 没给贴图或贴图未导出时兜底）
FALLBACK_RGBA = [0.55, 0.55, 0.58, 1.0]


# ---------------------------------------------------------------- export
def _pairs(items, what):
    """["标签=值", ...] -> {标签: 值}。用 JSON 传给 Blender，避免节点名里的逗号/等号破坏解析。"""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"{what} 参数格式应为 标签=值，收到 {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def cmd_export(a):
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    bodies = _pairs(a.bodies, "--bodies")     # 标签 -> GLB 节点名
    static = _pairs(a.static, "--static")
    cmd = [a.blender_python, EXTRACT, "--",
           "--glb", os.path.abspath(a.glb), "--out", out,
           "--bodies", json.dumps(list(bodies.values()), ensure_ascii=False),
           "--static", json.dumps(static, ensure_ascii=False)]
    print("$", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in (p.stdout or "").splitlines():
        if line.startswith("GLB_EXPORT_OK") or "Error" in line:
            print(" ", line)
    if p.returncode != 0:
        print((p.stderr or "")[-3000:], file=sys.stderr)
        raise SystemExit(f"[export] blender 退出码 {p.returncode}")
    path = os.path.join(out, "glb_export.json")
    rep = json.load(open(path))
    # body 是按 GLB 节点名导出的；把结果映射回用户给的标签
    by_node = rep["bodies"]
    mapped = {}
    for tag, node in bodies.items():
        if node in by_node:
            mapped[tag] = by_node[node]
        else:
            print(f"  WARN 刚体未导出: {tag} ({node})")
    rep["bodies"] = mapped
    rep["body_nodes"] = bodies
    json.dump(rep, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[export] parts={rep['n_parts']} bodies={len(rep['bodies'])} "
          f"textures={len(rep['textures'])} extent={rep['bounds']['extent']}")
    if rep["warnings"]:
        print("  warnings:", *rep["warnings"], sep="\n    ")
    return rep


# ---------------------------------------------------------------- xml
def load(a):
    out = os.path.abspath(a.out)
    rep_path = os.path.join(out, "glb_export.json")
    if not os.path.exists(rep_path):
        raise SystemExit(f"缺少 {rep_path}，先跑 export")
    return out, json.load(open(rep_path))


def build_xml(out, rep, drops=None, timestep=0.002, noslip=0):
    drops = drops or {}
    parts = rep["parts"]
    L = []
    A = L.append

    A('<?xml version="1.0"?>')
    A('<mujoco model="glb_env">')
    A('  <compiler meshdir="." autolimits="true"/>')
    A(f'  <option timestep="{timestep}" noslip_iterations="{noslip}" '
      f'integrator="implicitfast" cone="elliptic"/>')
    A('  <visual>')
    A('    <global offwidth="960" offheight="540"/>')
    A('    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>')
    A('    <map znear="0.01" zfar="60"/>')
    A('  </visual>')
    A('  <asset>')

    # 贴图 + 材质
    for i, t in enumerate(rep["textures"]):
        A(f'    <texture name="tex{i}" type="2d" file="textures/{t["file"]}"/>')
    tex_by_file = {t["file"]: i for i, t in enumerate(rep["textures"])}
    mat_seen = set()
    for i, p in enumerate(parts):
        name = f"m{i}"
        if name in mat_seen:
            continue
        mat_seen.add(name)
        if p.get("texture") and p["texture"] in tex_by_file:
            ti = tex_by_file[p["texture"]]
            A(f'    <material name="{name}" texture="tex{ti}" texuniform="true" '
              f'texrepeat="1 1" rgba="1 1 1 1" specular="0.15" shininess="0.2"/>')
        else:
            r, g, b, al = p["rgba"]
            A(f'    <material name="{name}" rgba="{r:.3f} {g:.3f} {b:.3f} {al:.3f}" '
              f'specular="0.15" shininess="0.2"/>')

    # 刚体材质：各自用自己的贴图/基色，不能共用片材材质
    body_mat = {}
    for k, (tag, b) in enumerate(rep["bodies"].items()):
        name = f"bm{k}"
        body_mat[tag] = name
        tex = b.get("texture")
        if tex and tex in tex_by_file:
            A(f'    <material name="{name}" texture="tex{tex_by_file[tex]}" '
              f'texuniform="true" texrepeat="1 1" rgba="1 1 1 1" '
              f'specular="0.15" shininess="0.2"/>')
        else:
            r, g, bl, al = b.get("rgba") or FALLBACK_RGBA
            A(f'    <material name="{name}" rgba="{r:.3f} {g:.3f} {bl:.3f} {al:.3f}" '
              f'specular="0.15" shininess="0.2"/>')

    # mesh：场景片材 + 刚体
    for i, p in enumerate(parts):
        A(f'    <mesh name="part{i}" file="parts/{p["file"]}" inertia="shell"/>')
    for tag, b in rep["bodies"].items():
        mesh_name = "body_" + tag.replace(" ", "_").replace("/", "_")
        A(f'    <mesh name="{mesh_name}" file="bodies/{b["file"]}" '
          f'scale="1 1 1"/>')
    A('  </asset>')

    A('  <worldbody>')
    A('    <light pos="1.0 1.0 2.5" dir="-0.3 -0.3 -1" diffuse="0.65 0.65 0.65"/>')
    A('    <light pos="0.2 -0.6 2.2" dir="0 -0.2 -1" diffuse="0.35 0.35 0.35"/>')

    # 场景视觉：整片材只渲染，不碰物理。
    # 不指定 mass 且 contype=0 -> MuJoCo 不算体积/凸包（实测 hull=False，编译毫秒级）
    for i, p in enumerate(parts):
        A(f'    <geom name="room{i}" type="mesh" mesh="part{i}" '
          f'material="m{i}" contype="0" conaffinity="0" group="1"/>')

    # 静态碰撞体：解析 box，不用 mesh
    for tag, s in rep.get("static", {}).items():
        cx, cy, cz = s["center"]
        hx, hy, hz = s["half"]
        A(f'    <body name="static_{tag}" pos="{cx:.5f} {cy:.5f} {cz:.5f}">')
        A(f'      <geom name="{tag}_col" type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}" '
          f'friction="1.0 0.005 0.0001" rgba="0.62 0.45 0.28 0.35"/>')
        A('    </body>')

    # 动态刚体：视觉 mesh + 解析碰撞体（只支持长方体近似，见 DEVELOPMENT_LOG）
    for tag, b in rep["bodies"].items():
        mesh_name = "body_" + tag.replace(" ", "_").replace("/", "_")
        cx, cy, cz = b["center_world"]
        dz = float(drops.get(tag, 0.0))
        hx, hy, hz = (d / 2 for d in b["dims"])
        A(f'    <body name="obj_{tag}" pos="{cx:.5f} {cy:.5f} {cz + dz:.5f}">')
        A(f'      <freejoint name="j_{tag}"/>')
        A(f'      <geom name="{tag}_vis" type="mesh" mesh="{mesh_name}" '
          f'contype="0" conaffinity="0" group="0" material="{body_mat[tag]}"/>')
        A(f'      <geom name="{tag}_col" type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}" '
          f'density="600" friction="1.0 0.005 0.0001" solref="0.01 1.0" condim="4" '
          f'rgba="0.8 0.2 0.2 0.0"/>')
        A('    </body>')
    A('  </worldbody>')
    A('</mujoco>')

    path = os.path.join(out, "scene.xml")
    open(path, "w").write("\n".join(L) + "\n")
    return path


def cmd_xml(a):
    out, rep = load(a)
    drops = dict(x.split("=", 1) for x in (a.drop or []))
    p = build_xml(out, rep, drops, a.timestep, a.noslip)
    print(f"[xml] {p}")


# ---------------------------------------------------------------- build
def cmd_build(a):
    import numpy as np
    import mujoco

    out, rep = load(a)
    drops = dict(x.split("=", 1) for x in (a.drop or []))
    xml = build_xml(out, rep, drops, a.timestep, a.noslip)

    rep_out = {"xml": xml, "drops": drops}
    try:
        m = mujoco.MjModel.from_xml_path(xml)
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"[build] 编译失败: {type(e).__name__}: {e}")
        raise SystemExit(1)

    hull = [m.mesh(i).name for i in range(m.nmesh) if m.mesh_graphadr[i] >= 0]
    rep_out.update({
        "compiled": True,
        "ngeom": m.ngeom, "nmesh": m.nmesh, "nbody": m.nbody, "nq": m.nq,
        "faces": int(m.mesh_facenum.sum()),
        "hull_meshes": hull,
        "hull_note": "hull_meshes 应只有 body_*；场景片材若出现在这里说明在算凸包",
    })

    # ---------- 静置 + 落下 ----------
    tags = list(rep["bodies"].keys())
    bid = {t: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj_{t}") for t in tags}
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    z0 = {t: float(d.xpos[bid[t]][2]) for t in tags}
    n = int(round(a.seconds / m.opt.timestep))
    maxpen = 0.0
    for _ in range(n):
        mujoco.mj_step(m, d)
        if d.ncon:
            maxpen = min(maxpen, float(d.contact.dist[: d.ncon].min()))

    rep_out["settle"] = {}
    for t in tags:
        dof = int(m.body_dofadr[bid[t]])
        rep_out["settle"][t] = {
            "z0": round(z0[t], 4), "z": round(float(d.xpos[bid[t]][2]), 4),
            "dropped_mm": round((z0[t] - float(d.xpos[bid[t]][2])) * 1000, 1),
            "speed_mm_s": round(float(np.linalg.norm(d.qvel[dof:dof + 3])) * 1000, 2),
            "spin_rad_s": round(float(np.linalg.norm(d.qvel[dof + 3:dof + 6])), 4),
            "tilt_deg": round(float(np.degrees(
                2 * np.arccos(min(1.0, abs(d.xquat[bid[t]][0]))))), 1),
        }
    rep_out["max_penetration_mm"] = round(maxpen * 1000, 2)

    # ---------- 渲染 ----------
    if a.render:
        try:
            import cv2
            r = mujoco.Renderer(m, a.height, a.width)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = a.lookat
            cam.distance = a.distance
            cam.azimuth = a.azimuth
            cam.elevation = a.elevation
            r.update_scene(d, cam)
            img = r.render()
            png = os.path.join(out, a.png)
            cv2.imwrite(png, img[:, :, ::-1])
            rep_out["render"] = {"file": png, "std": round(float(img.std()), 2),
                                 "note": "std 过低说明画面空白"}
            r.close()
        except Exception as e:
            rep_out["render"] = {"error": f"{type(e).__name__}: {e}"}

    json.dump(rep_out, open(os.path.join(out, "build_report.json"), "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps(rep_out, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="GLB -> OBJ 片材 + 刚体 + 贴图（走 Blender）")
    e.add_argument("--glb", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--bodies", nargs="*", default=[],
                   help='"标签=GLB节点名"，可多个')
    e.add_argument("--static", nargs="*", default=[],
                   help='"标签=GLB节点名"，生成解析 box 碰撞体')
    e.add_argument("--blender-python", default=DEFAULT_BLENDER_PY)
    e.set_defaults(func=cmd_export)

    x = sub.add_parser("xml", help="按 glb_export.json 重建 scene.xml")
    x.add_argument("--out", required=True)
    x.add_argument("--drop", nargs="*", default=[])
    x.add_argument("--timestep", type=float, default=0.002)
    x.add_argument("--noslip", type=int, default=0)
    x.set_defaults(func=cmd_xml)

    b = sub.add_parser("build", help="编译 + 物理 + 渲染")
    b.add_argument("--out", required=True)
    b.add_argument("--drop", nargs="*", default=[],
                   help='"标签=米"，初始抬高')
    b.add_argument("--seconds", type=float, default=3.0)
    b.add_argument("--timestep", type=float, default=0.002)
    b.add_argument("--noslip", type=int, default=0)
    b.add_argument("--render", action="store_true")
    b.add_argument("--png", default="render.png")
    b.add_argument("--width", type=int, default=960)
    b.add_argument("--height", type=int, default=540)
    b.add_argument("--lookat", type=float, nargs=3, default=[0.2, 0.0, 0.78])
    b.add_argument("--distance", type=float, default=1.2)
    b.add_argument("--azimuth", type=float, default=135.0)
    b.add_argument("--elevation", type=float, default=-20.0)
    b.set_defaults(func=cmd_build)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
