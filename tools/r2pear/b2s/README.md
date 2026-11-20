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
| 6 最后诊断（显式启用） | `speed_diagnosis.py` | 常规排查仍无解后，原速失败才对照半速/1⁄3速度；只写诊断报告，不替换轨迹或正常验收 |
| UI | `blend_worker.py` | Web UI 的 Blender 视图：打开原 blend，用 MuJoCo 状态驱动骨骼、物体和夹爪驱动属性 |

每一个自动判断都记在 `stage0_spec/decisions.json`，字段是 `choice`、`evidence`、`fallback`、`warn`，**没有静默兜底**。

## 最后才做速度诊断

先检查来源与帧对应、整体配准、工具/抓取点、碰撞与初态、夹持指令和控制跟踪。
这些检查仍不能解决回放失败时，带上排查说明运行：

```bash
python tools/r2pear/b2s/run.py --blend <scene.blend> --config <config.json> --out <simdir> \
  --last-resort-speed-diagnosis "已复核配准、工具、碰撞和夹持；初态稳定，原速仍失去双侧接触"
```

该阶段放在普通 V1/V2/V3 及本次启用的 `--env-tests` 之后。默认不扫描，报告记录跳过原因；
`--skip-validate` 不允许同时启用诊断。排查说明是操作人员的判断记录，程序不能自动证明所有修复手段已穷尽。
初态验证失败、回放验证缺失、当前配置回放已成功、来源时序不明时均跳过；真正开始前再检查当前初态。

原速对照仍失败才测试源时序的2倍、3倍执行时长；倍数始终相对于 `source.timing.source_fps`，
不会在已有慢速上再成倍降低速度。每个实验重新初始化，保持关节目标、开合采样、几何和控制参数一致。
报告为 `stage4_diagnostics/speed-diagnosis.json`，包含输入SHA256、原速与降速结果、跟踪误差、
抬升高度、左右指垫接触帧、实际采样数及物理步长量化后的间隔。双侧接触不等于已经稳定抓住。

可以对已有失败场景单独补做同一诊断：

```bash
python tools/r2pear/b2s/speed_diagnosis.py --simdir <simdir> --reason "已完成的常规排查及未解决问题"
```

诊断不修改 `task_spec.json`、`demo.npz`、`validate.json`，也不自动发布。降速通过只是适配候选，
不证明原时序通过，不自动证明具体根因；采用适配版时应另设配置并保留原失败报告。
`retarget_time_scale` 仍可用于明确选择过的场景配置，例如test3现有三倍执行时长版本。

## Blender 转仿真的初态验收

不能只看渲染外观或静置后的稳定性。test4 曾因陪衬红环退化成实心箱体，在机器人动作前把蓝环推偏约 23 mm。
先检查完整求值网格的孔洞、相邻物间隙及完整旋转；桌面也必须保留基座系中的倾斜。
同一托盘的底板和边框应属于同一刚体，避免把互相嵌合的构件当作会互相推挤的独立物体。

`validate.py` 的 V1 从 **零静置** 开始，保存初始重叠的 geom/body 对、XYZ 位移、水平位移、
全程最大位移及最终转角；即使物体后来回到原处，超过 2 mm 的瞬时位移也不通过。
提供 `demo.npz` 时还报告 Blender 首帧→编译初态→静置终态的位置差，避免预先固化静置姿态掩盖偏移。
源 Blender 位姿是建模参考，不是已标定真值；这部分偏差需单独审阅，V1 的 `ok` 仍表示物理静置条件。
发现偏移先修几何、坐标和刚体组合，再回放未修改的原始关节及夹爪指令；不以关闭物体间碰撞、
挪动物体迎合轨迹或 oracle 成功代替原始轨迹验收。目标物和陪衬物都要检查。

### DROID 整体配准与抓取残差

提供 `droid_trajectory` 与 `embodiment.flange` 时，输入阶段自动写 `registration-report.json`：

1. 以 DROID 实测法兰位姿为依据，检查 Blender 多帧法兰的位置、旋转及固定整体变换；默认
   P95 位置误差 >5 mm 或旋转误差 >2° 告警。机器人世界位置本身不等于场景偏移。
2. 有显式夹持点参考时，扣除整体变换再比较它与原始关节 FK 的工具位置；残差中位值超过
   `max(5 mm, 3 × 整体拟合残差P95)` 时告警，区分全局错位与局部工具/抓取问题。
3. 先检查工具中心、安装、尺寸和碰撞，再以原始 DROID 轨迹为约束测试任务区的位置修正。
   保持原始 q/g_cmd 和时序，记录修正前后结果、接触与初态漂移；只有完整回放验证有效才采用修正。
   局部残差不能直接视为全场平移量，未标定的建模点也不能当作物体真值。缺参考时明确报告未测。

阈值是可配置的工程告警默认值，不是行业标定标准。配置示例：
`registration: {global_warn_mm: 5, global_warn_deg: 2, grasp_warn_mm: 5, relative_ratio: 3,
grasp_point_in_flange_m: [x,y,z], grasp_reference_grade: "authored_estimate_not_calibrated"}`。
test4 的原始机器人整体一致、局部夹持参考差约20.2 mm；经对照采用任务区约30.2 mm的位置修正。
这是轨迹约束下的重建修正，不是重新定义 DROID 真值。发布时保留告警和调整依据。

## 规范末端系（所有数据源统一换算到这里）

规范末端系就是 menagerie robotiq_2f85 的 `pinch` site：原点在两指之间，+z 为接近方向，±y 为闭合轴。夹爪量 g 取 0 = 开、1 = 闭，与 DROID `gripper_position` 的语义相同。

| 数据源 | 状态 |
|---|---|
| DROID（blend 骨架 + parquet） | ✅ ep71 端到端验证 |
| DROID（父子对象关节链 + trajectory.npz） | test4 已通过输入/FK/Blender 绑定验证；圆环物理待验证 |
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

## DROID 父子对象关节输入

父子对象机械臂可指定 `embodiment.type="robot_arm"`、`representation="object_joints"`、
`base` 和可选 `flange`。关节链通过七个对象的 `source_field="observation.joint_position"` 与
`source_column_zero_based=0..6` 读取，也可显式指定按序排列的 `embodiment.chain`。
当前支持固定基座、单轴旋转关节及固定中间节点；移动基座、变化的关节原点、缩放或未建模的
中间节点动画会被拒绝。仍需通过与骨架分支相同的 MuJoCo FK 校验。

`droid_trajectory` 可指向精简资料中的 `trajectory.npz`。先核对关节、帧号、fps 和已有校验值，
再读取实测夹爪与原始控制指令。Blender 的键帧夹爪需提供归一化的
`observation.gripper_position` 属性，或在 `embodiment.gripper_property` 显式指定
`{"object": "执行器对象名", "property": "measured_closure"}`。显式属性也必须是0开/1闭的
有限归一化值，其他夹爪局部动画必须能由该属性唯一确定；原始控制指令仍从已核对的DROID数据读取。
详见 [`test4_README.txt`](../../../configs/test4_README.txt)；该适配不要求把对象层级转换为骨架。

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
