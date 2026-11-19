# blender2sim

从 Blender 场景出发，构建 MuJoCo 可交互仿真环境的管线。

```
.blend  ──b2s──>  task_spec.json + demo.npz  ──>  PEAR StatefulEnv  ──>  Web UI（MuJoCo + Blender 双视图）
                                                          │
                                        原生轨迹导回 Blender ┘
```

## 它做什么

给定一个 `.blend`（里面有机械臂动画、或者一段人手/手持夹爪的演示），产出一个可以直接
用 RL / 回放驱动的 MuJoCo 环境，并能在浏览器里交互：

- **识别具身**：机械臂骨架 / 人手 / 手持夹爪，三种自动判别
- **解析场景**：桌面、支撑树、物体形状（box/cylinder/prism/arch）、静态障碍、相机
- **提取演示**：关节角、夹爪指令、末端轨迹，自动修坏帧
- **重定向**（无机器人关节时）：基座摆位搜索 + 逐帧 IK
- **验证**：静置漂移 / 真值回放 / 与源视频三栏对照
- **交互**：Web UI 里 jog 机械臂、开合夹爪、看 oracle 自动解题；Blender 视图保留原场景材质光照

## 快速开始

```bash
# 1. 环境
pip install -r requirements.txt
source setup_env.sh          # 设置 B2S_PYTHON / ROBOLAB_DATA / MENAGERIE 等

# 2. 跑一条 .blend
$B2S_PYTHON tools/r2pear/b2s/run.py --blend <X.blend> --out <simdir> --env-tests

# 3. 打开交互界面
MUJOCO_GL=egl $B2S_PYTHON tools/r2pear/web_ui.py --testdir <simdir> --port 8091
```

详细用法见 [`tools/r2pear/b2s/README.md`](tools/r2pear/b2s/README.md)。

## 目录

```
tools/r2pear/                核心管线
  b2s/                       .blend → PEAR 环境（主链路）
    run.py                   入口
    extract_blend.py         bpy 子进程，提取 blend 事实
    embodiment.py            具身识别 + FK 校验 + 关节符号/零点拟合
    parse_scene.py           场景解析
    demo.py                  演示轨迹提取与修复
    retarget.py              基座摆位 + IK
    build_spec.py            生成 task_spec.json
    validate.py              V1/V2/V3 验证
    init_dist.py             初始状态分布
    blend_worker.py          Web UI 的 Blender 渲染子进程
    tests/                   单测
  env.py                     PEAR StatefulEnv 封装
  sim.py                     MuJoCo 场景构建与步进
  oracle.py                  pick-place 解题器
  predicates.py              on / released / upright
  web_ui.py                  交互界面（EGL + MJPEG）
  glb_scene.py               GLB → MuJoCo 静态场景（旁支，无机器人）
tools/umi_extract/           UMI 数据集抽 episode
tools/egodex/                EgoDex 数据集拉取
tools/droid_*.py             DROID 数据集辅助

evidence/                    可复现的证据（见下）
docs/                        踩坑史与进度
```

## 证据

`evidence/` 下是两条**端到端跑通**的记录，压缩后仅 13 MB，可以直接复跑：

| 目录 | 数据 | 具身 | 状态 |
|---|---|---|---|
| `test1/` | DROID ep71（DROID 0071.blend） | 机械臂直连（关节真值） | ✅ V1/V2 全通过 |
| `test2/` | UMI ep68（真实采集） | 手持夹爪重定向 | ⚠️ 谓词 3/4 通过，见下 |

**test1** 的数字：

```
V1 静置    max_drift 0.357 mm    penetration −0.126 mm    ok=true
V2 回放    success=true   关节跟踪中位 0.72° / 最大 4.42°
           4 个谓词全过（on force 0.766 N vs 自重 0.769 N）
```

**test2** 的现状（不要当成已通过）：

```
on / released / upright    ✅
all_at_rest                ❌
V1 静置  杯子 speed 12.74 mm/s（阈值 5）
V2      最低接触 −7.13 mm，与 blend 参考轨迹中位偏 13.9 mm
```

根因怀疑是 `PIPELINE_LOG.md §5` 记的那个结构性冲突：UMI 是人手持爪、手腕可贴桌面上方
3 cm 作业，而 Panda 的 link5/link6 直径 8 cm。这个演示对 Panda 是否物理可行还没定论。

复跑验证（不需要重新跑 bpy 提取）：

```bash
source setup_env.sh
$B2S_PYTHON tools/r2pear/b2s/validate.py --simdir evidence/test1/sim
$B2S_PYTHON tools/r2pear/b2s/tests/test_b2s.py
```

## 不支持 / 未验证

- 目标机型只有 **Panda + 2F-85**
- 形状只有 box / cylinder / prism / arch，其他按包围盒兜底并告警
- 物体只绕 z 轴旋转，初始倾斜会告警
- 谓词只有 on / released / upright；oracle 只支持 pick-place
- 静态障碍用局部包围盒近似，不是精确网格
- **UMI / EgoDex 的 TCP 轴向换算没在真实数据上验证过**（test2 是第一次真实数据尝试，仍在进行中）
- RL 用的观测是 MuJoCo 渲染，**没有实验室背景**；Blender 视图只给人看

## 依赖版本

组合是实测锁死的，升级前先读 `docs/DEVELOPMENT_LOG.md §2–5`：

```
python 3.11      bpy 只发 cp311 wheel
numpy 1.26.4     bpy 要求 <2.0
torch 2.11.0     flex_gemm 需要 triton>=3.2
bpy 4.5.4 LTS
mujoco 3.3.7
```

## 移植

换服务器要改的东西见 [`docs/PORTING.md`](docs/PORTING.md)。

## 致谢

管线建立在 [Real2Gym](https://github.com/real2gym/Real2Gym) 与
[PEAR](https://github.com/mit-han-lab/pear) 之上；机器人模型来自
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)。
数据来源：DROID、UMI、EgoDex。
