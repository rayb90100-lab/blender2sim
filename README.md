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
  controller.py              位姿增量 → 关节指令
  spec.py                    task_spec 读写
  init_states.py             初始状态采样
  sweep_params.py            单项扰动扫描
  check_negatives.py         反向对照
  glb_scene.py               GLB → MuJoCo 静态场景（旁支，无机器人）
  glb_extract.py             GLB 资产提取（bpy）
  blender_live.py / blender_worker.py   旧式 Blender 渲染 worker
tools/droid_common.py        机械臂模型加载 + 位姿工具（主链路依赖）
tools/umi_extract/           UMI 数据集抽 episode
tools/egodex/                EgoDex 数据集拉取

evidence/                    可复现的证据（见下）
```

内部文档（几轮踩坑的完整记录、进度交接、移植清单）暂不随仓库发布。

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

这套管线在 2026-11-19 从原始 `DROID_0071.blend` **重跑过一遍完整流程**
（extract → 具身 → 解析 → 演示 → spec → validate，非读取缓存），结果：

```
V1 静置    max_drift 0.358 mm    penetration −0.148 mm    ok=true
V2 回放    success=true    关节跟踪中位 0.72° / 最大 4.42°    4 谓词全过
V3 对照    三路视频各 172 帧
决策记录   30 条（兜底 1、警告 2，均为数据质量问题的自动修复，已记录）
```

与仓库内 `evidence/` 的差异（0.02 mm / 0.3 µm）来自 MuJoCo 求解顺序的非确定性，
不影响任何判定。

**test2** 的现状（不要当成已通过）：

```
on / released / upright    ✅
all_at_rest                ❌
V1 静置  杯子 speed 12.74 mm/s（阈值 5）
V2      最低接触 −7.13 mm，与 blend 参考轨迹中位偏 13.9 mm
```

根因怀疑是结构性冲突：UMI 是人手持爪、手腕可贴桌面上方 3 cm 作业，
而 Panda 的 link5/link6 直径 8 cm。这个演示对 Panda 是否物理可行还没定论。

## 数据集（复现用）

两条证据的原始数据都在公开数据集里。`evidence/` 里的产物可以直接复跑验证，
只有想**从头重建场景**时才需要下载原始数据。

### test1：DROID ep71

| 项 | 值 |
|---|---|
| 数据集 | `cadene/droid_1.0.1`（LeRobot 移植的完整 DROID 1.0.1） |
| HuggingFace | https://huggingface.co/datasets/cadene/droid_1.0.1 |
| 国内镜像 | https://hf-mirror.com/datasets/cadene/droid_1.0.1 |
| 许可 | apache-2.0，无需申请 |
| Episodes | 76k 条，27 TB。**只需下 ep71 那一个 parquet** |
| 定位 | `**/episode_000071.parquet`（`sources.py::find_droid_episode` 按这个模式找） |
| 指令 | Pick up the green block from the yellow cube and put it on the yellow cuboid. |
| 视频 | 三路 320×180 @ 15 fps：`exterior_1_left` / `exterior_2_left` / `wrist_left` |

```bash
# 用镜像下（官方源在国内约 37 KB/s）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download cadene/droid_1.0.1 --repo-type dataset \
  --include "*episode_000071*" --local-dir $ROBOLAB_DATA/datasets/droid
```

**场景文件不在仓库里**（二进制，7.4 MB），打包成 zip 放在 release 资产里：

```
https://github.com/rayb90100-lab/blender2sim/releases/tag/v0.1.0-blends
```

| 资产 | 内含 | 大小 | zip SHA256 | 解压后 .blend SHA256 |
|---|---|---|---|---|
| `DROID_0071.blend.zip` | `DROID_0071.blend` | 6.80 MB | `4cd1dd22abaed5b4c787986ab2a6bc52e8f2098db81739f0646ea4ade002479e` | `25838c75bec78e4e5b3eb3c14ca8c1f09b27c2e06a7086dfd180976ab9a4cf99` |
| `UMI_ep68.blend.zip` | `UMI_ep68.blend` | 0.61 MB | `699922b6299200033c131f66677521bccc24b652735456cbe200c01ab8d18523` | `d9ffdae16a0c91c2668c9eef99291e8dbbb998f416fbfd41771801edf52e3658` |

解压后直接跑：

```bash
unzip DROID_0071.blend.zip
$B2S_PYTHON tools/r2pear/b2s/run.py --blend DROID_0071.blend --out <simdir> --env-tests
```

`.blend` 是 Blender 4.x 默认的 zstd 压缩格式（文件头 `28 b5 2f fd`），这是正常的，
Blender 直接能打开，不需要额外处理。

`DROID_0071.blend` 由 GPT 从 parquet + 三路视频生成（`extract.json` 里记着
`blend: GLB/test1/DROID_0071.blend`）。**生成过程不在本仓库**，因此无法在此重新生成，
请直接用 release 里的文件。资产本身完整，`evidence/test1` 的全部结果都是基于它跑出来的。

### test2：UMI ep68

| 项 | 值 |
|---|---|
| 数据集 | UMI `cup_in_the_wild`（zarr_datasets） |
| 下载 | https://real.stanford.edu/umi/data/zarr_datasets/cup_in_the_wild.zarr.zip |
| 大小 | 17.99 GB（305 episodes / 184958 帧） |
| Episode | 68，帧区间 `[49453, 50377)`，924 帧 |
| 具身 | 单臂（只有 `robot0_*`，全库无 `robot1_*`） |

```bash
curl -L -o $ROBOLAB_DATA/downloads/cup_in_the_wild.zarr.zip \
  https://real.stanford.edu/umi/data/zarr_datasets/cup_in_the_wild.zarr.zip

# 抽 ep68 → zarr + 规范末端 npz + 腕部相机 mp4
$B2S_PYTHON tools/umi_extract/extract_umi_episode.py \
  --src $ROBOLAB_DATA/downloads/cup_in_the_wild.zarr.zip --episode 68 --out <outdir>
```

⚠️ **本仓库的证据不是从这个 URL 的文件生成的。** 本地副本是 `cup_in_the_lab.zarr.zip`
（4.2 GB，解压 4.47 GB），与官方 `cup_in_the_wild`（17.99 GB）**文件名不同、大小差 4 倍，
且从未做过内容比对**。字段结构一致，但 ep68 的具体内容可能不同。
引用时请以自己下载的官方文件为准，或以此说明为参照重新抽取。

⚠️ **为什么选 ep68**：这份数据集 305 条里只有 ep68 有真实抓取动作。
全库夹爪宽度四分位是 `[0.052, 0.055, 0.063, 0.081, 0.083, 0.083]` m ——
绝大多数 episode 夹爪停在张开状态。出现过 < 30 mm 的**只有 ep68**（15 帧，最闭合 4.8 mm）。

场景文件同样在 release 里：`UMI_ep68.blend.zip`（0.61 MB，SHA256 见上）。
它只含场景与演示者，不含机器人——具身是 `handheld_gripper`，走 `retarget.py` 重定向到 Panda。

复跑验证（不需要原始数据，`evidence/` 里已有产物）：

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

组合是实测锁死的，不要随手升级（原因见下）：

```
python 3.11      bpy 只发 cp311 wheel，Python 版本被它决定
numpy 1.26.4     bpy 要求 <2.0；装了 numpy 2.x 的包会把它顶上去
torch 2.11.0     flex_gemm 需要 triton>=3.2（torch 2.11 自带 3.6）
bpy 4.5.4 LTS
mujoco 3.3.7
```

## 移植

换服务器要改的只有几个环境变量，`source setup_env.sh` 后按提示设：

| 变量 | 含义 |
|---|---|
| `B2S_PYTHON` | 装了 bpy + mujoco + torch 的解释器 |
| `ROBOLAB_DATA` | 权重 / 数据集 / 缓存的根目录 |
| `MENAGERIE` | mujoco_menagerie 路径（需要 `franka_emika_panda` 和 `robotiq_2f85`） |
| `FFMPEG` | ffmpeg 可执行文件（`setup_env.sh` 会自动探测） |

**仓库里不含**外部资产：menagerie 机器人模型、MoGe/sam2 权重、以及 DROID/UMI/EgoDex
数据集都要自己准备。缺 menagerie 会报
`ValueError: ParseXML: Error opening file ... panda_nohand.xml`。

### 装完自查

不需要数据集，这三条应该全过：

```bash
source setup_env.sh

# ① 单测（4 个：关节符号修复 / 人手换算 / ARKit 轴 / 形状分类）
$B2S_PYTHON tools/r2pear/b2s/tests/test_b2s.py

# ② 端到端回放（期望 ok=true / success=true）
$B2S_PYTHON tools/r2pear/b2s/validate.py --simdir evidence/test1/sim

# ③ Web UI
MUJOCO_GL=egl $B2S_PYTHON tools/r2pear/web_ui.py --testdir evidence/test1/sim --port 8091
```

## 致谢

管线建立在 [Real2Gym](https://github.com/real2gym/Real2Gym) 与
[PEAR](https://github.com/mit-han-lab/pear) 之上；机器人模型来自
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)。
数据来源：DROID、UMI、EgoDex。
