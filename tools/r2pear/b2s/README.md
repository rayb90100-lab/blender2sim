# b2s：任意 .blend → PEAR 可交互 MuJoCo 环境

```bash
cd /home/ubuntu/workstation/RoboLab && source setup_env.sh
PY=/home/ubuntu/mambaforge/envs/real2gym/bin/python
MUJOCO_GL=egl $PY tools/r2pear/b2s/run.py --blend <X.blend> --out <simdir> [--config b2s_config.json] [--env-tests]
MUJOCO_GL=egl $PY tools/r2pear/web_ui.py --testdir <simdir> --port 8112      # 交互界面（MuJoCo + 原 blend 精修视图）
```

`--env-tests` 会再跑反向对照、扰动扫描和初始状态采样，大约需要 2 分钟。产物目录和 `Real2Gym/tests/testN` 的布局一致，`env.py`（PEAR StatefulEnv）、`oracle.py` 和 `web_ui.py` 都可以直接使用。

## 流水线

| 步骤 | 文件 | 做什么 |
|---|---|---|
| 0 提取 | `extract_blend.py`（bpy） | 只提取事实：对象、网格、骨骼逐帧、自定义属性动画、相机、灯光。动态对象靠 5 个探测帧自动发现 |
| 1a 具身 | `embodiment.py` | 识别 robot_arm / human_hand / handheld_gripper。机械臂需提取关节角，再用 **MuJoCo FK 做恒定偏移校验**；不一致时用束搜索拟合各关节的符号和零点 |
| 1b 场景 | `parse_scene.py` | 桌面 = 被演示移动的物体所在的支撑面；物体 = 以桌面为根的支撑树；形状分类到 box/cylinder/prism/arch；桌面附近的静止网格作为静态障碍；相机 |
| 2 演示 | `demo.py`、`sources.py` | 夹爪候选交叉校验（数据集 > 外部/人手 > blend > 由物体运动反推）；夹爪**指令**通道；物体坏帧检测与修复 |
| 2b 重定向 | `retarget.py` | 没有机器人关节时（UMI/Ego）：搜索基座位置，再逐帧 IK（考虑 180° 对称） |
| 3 spec | `build_spec.py` | 生成 r2pear `task_spec.json`（机器人基座系）。谓词由演示终态的支撑关系推出。另写 `demo.npz` 和 `blend_binding` |
| 4 验证 | `validate.py` | V1 静置；V2 真值回放（物体运动全部由物理产生），之后判谓词；V3 和源视频三栏对照 |
| 5 初始状态 | `init_dist.py` | 把 `sweep_params.py` 实测的单项扰动边界换算成 `init_states.py` 的采样分布 |
| UI | `blend_worker.py` | Web UI 的 Blender 视图：打开原 blend，用 MuJoCo 状态驱动骨骼、物体和夹爪驱动属性 |

每一个自动判断都记在 `stage0_spec/decisions.json`，字段是 `choice`、`evidence`、`fallback`、`warn`，**没有静默兜底**。

## 规范末端系（所有数据源统一换算到这里）

规范末端系就是 menagerie robotiq_2f85 的 `pinch` site：原点在两指之间，+z 为接近方向，±y 为闭合轴。夹爪量 g 取 0 = 开、1 = 闭，与 DROID `gripper_position` 的语义相同。

| 数据源 | 状态 |
|---|---|
| DROID（blend 骨架 + parquet） | ✅ ep71 端到端验证 |
| UMI（`replay_buffer.zarr`） | ⚠️ 换算已单测；**没在真实数据上验证**，TCP 轴向按 UMI 仓库约定（`UMI_TCP_TO_PINCH`） |
| EgoDex（hdf5，ARKit y 向上） | ⚠️ 手→夹爪换算和轴变换已单测；**没在真实数据上验证** |
| 通用 npz（`ee_world` + `g`） | ✅ 用 ep71 末端轨迹冒充 UMI，重定向后回放成功（`GLB/test1/sim_synth_umi`） |

本机 real2gym 环境**没装 zarr / h5py**。装之前要先确认不会把 numpy 顶到 2.x，之前装 rerun 就踩过这个坑。

## b2s_config.json（全部可选，放在 blend 同目录，或用 `--config` 指定）

```json
{
  "embodiment": {"type": "robot_arm|human_hand|handheld_gripper", "root": "...", "root_to_pinch": [[4x4]]},
  "table": "<blend 对象名>", "objects": ["..."], "exclude": ["..."], "exclude_trees": ["..."],
  "shapes": {"<blend 对象名>": "box|cylinder|prism|arch"},
  "success": [{"pred": "on", "obj": "...", "support": "..."}],
  "instruction": "...", "episode_index": 71, "dataset_roots": ["..."],
  "demo": {"type": "npz|umi|egodex", "path": "...", "T_blend_from_data": [[4x4]]},
  "robot_base": [[4x4]]
}
```

## 从 ep71 学到的规则（都已写进代码注释）

1. **夹爪回放必须用指令，不能用实测值**。实测值停在物体宽度处（0.661）；拿它当位置伺服目标，夹爪几乎没有夹持力，塔会被碰倒。数据集里的指令是 0.911。没有指令通道时，用"实测提前 0.27 s、闭合段 +0.25"推断；直接用全闭 1.0 会把 27 mm 的圆柱挤歪。
2. **关节零点在本连杆上不可观测**，要到下一个连杆才看得出来，所以修复时要和下一个关节的符号一起解。还存在运动学等价解（腕部翻转），需要用关节限位来打破平局。
3. **blend 对象的原点不一定在包围盒中心**（拱和棱柱的原点在底部）。所有"中心"都按包围盒算，`cloc` 记录两者之间的偏移。
4. blend 的物体动画是作者的近似：可能发散成 NaN、穿进桌面、在夹爪前面先抬起来。只用它来推终态谓词和做对照；仿真里的运动由物理产生。
5. MuJoCo 的 `principal_pixel` 是相对图像中心的偏移，**x 方向和 OpenCV 反号**：`[(W/2−0.5)−cx, cy−(H/2−0.5)]`，实测误差 <0.4 px。

## 已知限制

- 目标机型只有 Panda + 2F-85。
- 形状只支持 box / cylinder / prism / arch，其他形状按包围盒建 box，并在决策里告警，要求替换资产。
- 物体只能绕 z 轴旋转，初始倾斜会告警。
- 谓词只有 on / released / upright；oracle 只支持 pick-place 模板。
- 静态障碍用作者给的局部包围盒近似（ep71 有 55 个 box），不是精确网格。
- 最后一个关节的零点只靠连杆链观测不到。
- RL 实际用的观测（MuJoCo 渲染）**没有实验室背景**，Blender 视图只用于人看。
