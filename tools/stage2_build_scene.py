#!/usr/bin/env python3
"""生成 Stage 2 的 MuJoCo 场景：双 FR3 + 原装 Franka Hand + 桌面 + 两个物体。

依据：
- skill target-robot-selection.md：第一种配置固定为「双 FR3 + Franka Hand 原装夹爪」，
  使用厂商描述中的标准手掌、夹指、安装与开合关系，保持真实尺寸。
- skill mujoco.md：显示网格与碰撞代理分开；核心背景同步导入；记录质量/惯量/摩擦来源。

资产来源（不改造厂商几何，只做拼装）：
  fr3_duo.xml     menagerie franka_fr3_duo，派生自 franka_description v2.9.0
                  commit 7aeeddc449edf8d62b594f9e36a81da53e7796f9
  hand body       menagerie franka_emika_panda/hand.xml（原装 Franka Hand，非自制代理）

几何全部来自 Stage 1 的 manifest（相机系 -> 世界系，见 tools/stage2_common.py）。

用法:
  python tools/stage2_build_scene.py --testdir <testdir>
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_common import load_geometry  # noqa: E402

MENAGERIE = os.environ.get("MENAGERIE",
                           "/media/data1/RoboLab_data/assets/mujoco_menagerie")
FR3_DUO = f"{MENAGERIE}/franka_fr3_duo"
PANDA = f"{MENAGERIE}/franka_emika_panda"

BASE_POS = (0.0, -0.16, 0.0)
BASE_YAW = 0.0

# 物体物理参数：真实物体未称重，按类别估计，manifest 里标注 estimated
OBJECT_PHYS = {
    "earbud_case": {"mass": 0.055, "friction": [1.0, 0.02, 0.001], "rgba": "0.05 0.05 0.06 1",
                    "note": "TWS 耳机盒典型质量 50-60 g（估计，未称重）"},
    "tissue_box": {"mass": 0.230, "friction": [0.8, 0.02, 0.001], "rgba": "0.85 0.83 0.90 1",
                   "note": "抽纸盒含纸典型质量 200-260 g（估计，未称重）"},
}


def log(m): print(m, flush=True)


def indent(s, n):
    pad = " " * n
    return "\n".join(pad + ln if ln.strip() else ln for ln in s.splitlines())


def slice_tag(src, tag):
    """取出 <tag>...</tag> 的完整片段（含标签本身）。"""
    o = src.find(f"<{tag}")
    if o == -1:
        return ""
    gt = src.index(">", o)
    if src[gt - 1] == "/":
        return src[o:gt + 1]
    body_start = gt + 1
    depth = 1
    k = body_start
    pat_o, pat_c = f"<{tag}", f"</{tag}>"
    while depth:
        no = src.find(pat_o, k)
        nc = src.find(pat_c, k)
        if nc == -1:
            raise RuntimeError(f"{tag} 未闭合")
        if no != -1 and no < nc:
            depth += 1
            k = no + len(pat_o)
        else:
            depth -= 1
            k = nc + len(pat_c)
    return src[o:k]


def extract_hand_body():
    """从 panda.xml 抽 hand body，并做侧别重命名。"""
    src = open(f"{PANDA}/panda.xml").read()
    marker = '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">'
    o = src.index(marker)
    # 从 hand 起点开始找配对 </body>（它内含两个 finger body）
    depth, k = 1, o + len(marker)
    while depth:
        no = src.find("<body", k)
        nc = src.find("</body>", k)
        if nc == -1:
            raise RuntimeError("hand body 未闭合")
        if no != -1 and no < nc:
            depth += 1
            k = no + len("<body")
        else:
            depth -= 1
            k = nc + len("</body>")
    h = src[o:k]
    for old, new in [
        ('<body name="hand"', '<body name="{SIDE}_hand"'),
        ('name="left_finger"', 'name="{SIDE}_left_finger"'),
        ('name="right_finger"', 'name="{SIDE}_right_finger"'),
        ('name="finger_joint1"', 'name="{SIDE}_finger_joint1"'),
        ('name="finger_joint2"', 'name="{SIDE}_finger_joint2"'),
    ]:
        h = h.replace(old, new)
    return h


def build(testdir):
    man, n_cam, d_cam, R_wc, c2w, d2w = load_geometry(testdir)
    outdir = os.path.join(testdir, "stage2_physics", "mujoco")
    adir = os.path.join(outdir, "assets")
    os.makedirs(adir, exist_ok=True)

    # ---- 复制用到的资产 ----
    for f in os.listdir(f"{FR3_DUO}/assets"):
        shutil.copy2(os.path.join(FR3_DUO, "assets", f), adir)
    for f in os.listdir(f"{PANDA}/assets"):
        if f.startswith(("hand", "finger")):
            shutil.copy2(os.path.join(PANDA, "assets", f), adir)
    log(f"资产: {len(os.listdir(adir))} 个文件 -> {adir}")

    # ---- 几何 ----
    # yaw：Stage 1 建的是轴对齐盒，真实物体是斜放的。
    # orientation.json 由 tools/stage2_fix_orientation.py 用轮廓 IoU 全扫 ±90° 得出。
    ori_p = os.path.join(outdir, "orientation.json")
    ori = {}
    if os.path.exists(ori_p):
        ori = json.load(open(ori_p)).get("objects", {})
    objs = {}
    for name, v in man["instances"].items():
        c = c2w(v["center_cam"])
        s = np.array(v["size"], float)
        if name in ori:
            s = np.array(ori[name]["size_used"], float)
        yaw = float(ori[name]["yaw_rad"]) if name in ori else 0.0
        objs[name] = {"center": c.tolist(), "size": s.tolist(), "yaw": yaw}
        log(f"{name:14s} center={np.round(c,4)} size={np.round(s,4)} "
            f"yaw={np.degrees(yaw):+6.1f}° 底面z={c[2]-s[2]/2:+.5f}")

    tbl_x, tbl_y, tbl_th = 0.80, 0.70, 0.04
    tbl_cx, tbl_cy = 0.0, 0.32

    cam_pos = c2w(np.zeros(3))
    fwd = d2w(np.array([0.0, 0.0, 1.0]))
    up_cam = d2w(np.array([0.0, -1.0, 0.0]))
    right = np.cross(fwd, up_cam); right /= np.linalg.norm(right)
    upv = np.cross(right, fwd); upv /= np.linalg.norm(upv)
    fovy = 2 * np.degrees(np.arctan(man["camera"]["H"] / 2 / man["camera"]["fy"]))

    src = open(f"{FR3_DUO}/fr3_duo.xml").read()

    # fr3_duo 的 default 树（class fr3_duo 及其嵌套 visual/collision）
    dflt = slice_tag(src, "default")
    # Panda Hand 的 default class（joint damping / finger slide range）
    panda_dflt = slice_tag(open(f"{PANDA}/panda.xml").read(), "default")
    # fr3_duo 已有 class="visual"/"collision"，panda 的同名 class 会冲突，加前缀
    panda_dflt = panda_dflt.replace('class="visual"', 'class="hand_visual"')
    panda_dflt = panda_dflt.replace('class="collision"', 'class="hand_collision"')

    asset_inner = slice_tag(src, "asset")
    asset_inner = asset_inner[asset_inner.index(">") + 1: asset_inner.rindex("</asset>")]

    # hand 需要的 mesh 在 panda 的 asset 里，手工补
    hand_meshes = """
    <mesh name="hand_c" file="hand.stl"/>
    <mesh name="hand_0" file="hand_0.obj"/>
    <mesh name="hand_1" file="hand_1.obj"/>
    <mesh name="hand_2" file="hand_2.obj"/>
    <mesh name="hand_3" file="hand_3.obj"/>
    <mesh name="hand_4" file="hand_4.obj"/>
    <mesh name="finger_0" file="finger_0.obj"/>
    <mesh name="finger_1" file="finger_1.obj"/>
    <material name="off_white" rgba="0.901961 0.921569 0.929412 1"/>"""

    wb = slice_tag(src, "worldbody")
    wb_inner = wb[wb.index(">") + 1: wb.rindex("</worldbody>")]

    hand = extract_hand_body()
    for side in ("left", "right"):
        tag = f'<body name="{side}_fr3v2_link8" pos="0 0 0.107">'
        if tag not in wb_inner:
            raise RuntimeError(f"找不到 {tag}")
        wb_inner = wb_inner.replace(
            tag, tag + "\n" + indent(hand.replace("{SIDE}", side), 22), 1)

    actuators = slice_tag(src, "actuator")
    act_inner = actuators[actuators.index(">") + 1: actuators.rindex("</actuator>")]
    # 参数照抄 menagerie franka_emika_panda/hand.xml 的原装 actuator8。
    # ★ 必须显式写 dyntype="none" biastype="affine"：原文件里这两个属性在
    #   <default class="panda"> 的 <general> 里，我的 actuator 不在那个 class 下，
    #   不继承就成了 biastype=none -> biasprm 完全失效 -> 力只有 gain*ctrl，
    #   ctrl=0 时闭合力为 0，手指永远停在张开位（实测 actfrc 恒为 0.000）。
    #   force = gainprm[0]*ctrl + biasprm[1]*length + biasprm[2]*velocity
    #   原文件把 ctrlrange (0,0.04) 重映射到 (0,255)，gain=0.04*100/255。
    #   ctrl=255 -> 张开，ctrl=0 -> 闭合（力 = -100*length）。
    # 我自己配的 gain=1 / ctrlrange 0-0.04 力太小，克服不了 -100*length，
    # 手指永远卡在 0.04 张不开也合不上。
    finger_acts = "".join(f"""
    <general name="{s}_finger_act" tendon="split_{s}" dyntype="none" biastype="affine"
             forcerange="-100 100" ctrlrange="0 255"
             gainprm="0.01568627451 0 0" biasprm="0 -100 -10"/>"""
                          for s in ("left", "right"))
    tendons = "".join(f"""
    <fixed name="split_{s}">
      <joint joint="{s}_finger_joint1" coef="0.5"/>
      <joint joint="{s}_finger_joint2" coef="0.5"/>
    </fixed>""" for s in ("left", "right"))
    equals = "".join(f"""
    <joint joint1="{s}_finger_joint1" joint2="{s}_finger_joint2"
           solimp="0.95 0.99 0.001" solref="0.005 1"/>""" for s in ("left", "right"))

    obj_xml = []
    for name, v in objs.items():
        c, s = v["center"], v["size"]
        ph = OBJECT_PHYS[name]
        obj_xml.append(f'''
      <body name="{name}" pos="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}"
            euler="0 0 {v.get('yaw', 0.0):.6f}">
        <freejoint name="{name}_free"/>
        <geom name="{name}_geom" type="box" size="{s[0]/2:.6f} {s[1]/2:.6f} {s[2]/2:.6f}"
              mass="{ph['mass']}" friction="{' '.join(str(x) for x in ph['friction'])}"
              rgba="{ph['rgba']}"/>
      </body>''')

    xml = f'''<mujoco model="Real2Gym stage2 test3">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <!-- cone/impratio 保持 MuJoCo 默认（pyramidal / 1.0）。
       曾设 elliptic + impratio=10，手指一碰物体就被过强的摩擦锥锁死在张开位，
       ctrl=0 也合不拢（实测行程只有 1 mm）。原版 hand.xml 用的是默认值。 -->
  <option timestep="0.002" integrator="implicitfast"/>

  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <rgba haze="0.25 0.28 0.32 1"/>
    <global azimuth="120" elevation="-18" offwidth="720" offheight="1280"/>
  </visual>

  <default>
{indent(dflt[dflt.index(">") + 1: dflt.rindex("</default>")], 4)}
{indent(panda_dflt[panda_dflt.index(">") + 1: panda_dflt.rindex("</default>")], 4)}
    <default class="handgeom">
      <geom contype="1" conaffinity="1" condim="4"
            friction="1.0 0.02 0.001" solimp="0.95 0.99 0.001" solref="0.005 1"/>
    </default>
  </default>

  <asset>
{indent(asset_inner, 4)}
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.62 0.72" rgb2="0.22 0.24 0.28"
             width="512" height="3072"/>
    <texture type="2d" name="floor_tex" builtin="flat" rgb1="0.34 0.35 0.37" width="128" height="128"/>
    <texture type="2d" name="wood" builtin="checker" mark="edge"
             rgb1="0.62 0.47 0.32" rgb2="0.55 0.41 0.27"
             markrgb="0.48 0.36 0.24" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texuniform="true"/>
    <material name="table_mat" texture="wood" texuniform="true" texrepeat="4 4" reflectance="0.06"/>
{indent(hand_meshes.strip(), 4)}
  </asset>

  <worldbody>
    <light pos="0.3 0.2 2.0" dir="0 0 -1" directional="true"
           diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2" castshadow="true"/>
    <light pos="-1.0 0.6 1.6" diffuse="0.35 0.35 0.38"/>

    <geom name="floor" size="0 0 0.05" type="plane" material="floor_mat"/>

    <!-- 桌面：上表面 z=0 -->
    <body name="table" pos="{tbl_cx:.4f} {tbl_cy:.4f} {-tbl_th:.4f}">
      <geom name="table_top" type="box" size="{tbl_x/2:.4f} {tbl_y/2:.4f} {tbl_th/2:.4f}"
            material="table_mat"/>
    </body>
{"".join(obj_xml)}

    <!-- 双 FR3 + 原装 Franka Hand（厂商几何，未改造）-->
    <body name="robot_base" pos="{BASE_POS[0]:.4f} {BASE_POS[1]:.4f} {BASE_POS[2]:.4f}"
          euler="0 0 {BASE_YAW:.4f}">
{indent(wb_inner.strip(), 6)}
    </body>

    <camera name="source_view" pos="{cam_pos[0]:.6f} {cam_pos[1]:.6f} {cam_pos[2]:.6f}"
            xyaxes="{right[0]:.6f} {right[1]:.6f} {right[2]:.6f} {upv[0]:.6f} {upv[1]:.6f} {upv[2]:.6f}"
            fovy="{fovy:.4f}"/>
  </worldbody>

  <tendon>{tendons}
  </tendon>

  <equality>{equals}
  </equality>

  <actuator>
{indent(act_inner.strip(), 4)}{finger_acts}
  </actuator>
</mujoco>
'''
    sp = os.path.join(outdir, "scene.xml")
    open(sp, "w").write(xml)
    log(f"写出 {sp} ({len(xml)/1024:.0f} KB)")

    meta = {
        "world_frame": "Z 向上，桌面上表面为 z=0；由支撑面法线构造，见 tools/stage2_common.py",
        "camera": {"pos": cam_pos.tolist(), "right": right.tolist(), "up": upv.tolist(),
                   "fovy_deg": float(fovy),
                   "note": "由 Stage 1 manifest 的相机系位姿转到世界系，未重新标定"},
        "objects": objs,
        "object_physics": OBJECT_PHYS,
        "table": {"size": [tbl_x, tbl_y, tbl_th], "center": [tbl_cx, tbl_cy, -tbl_th / 2]},
        "robot": {"model": "dual FR3 (fr3_duo) + Franka Hand",
                  "base_pos": list(BASE_POS), "base_yaw": BASE_YAW,
                  "source": "menagerie franka_fr3_duo, franka_description v2.9.0 "
                            "commit 7aeeddc449edf8d62b594f9e36a81da53e7796f9",
                  "end_effector": "Franka Hand original (menagerie franka_emika_panda/hand.xml)",
                  "note": "原装平行夹爪，非自制代理；开口 0-0.04 m"},
        "physics": {"timestep": 0.002, "integrator": "implicitfast",
                    "note": "2 ms 步长用于抓取接触；时间与源视频映射见 trajectory/"},
    }
    json.dump(meta, open(os.path.join(outdir, "scene_meta.json"), "w"), indent=2,
              ensure_ascii=False)
    return sp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    a = ap.parse_args()
    build(a.testdir)
