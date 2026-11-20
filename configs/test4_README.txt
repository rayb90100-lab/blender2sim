test4 | DROID_s00000_r00000 | 蓝色圆环放入木托盘
================================================

当前状态：已发布到 /root/data/simulation/DROID-蓝色圆环放入木托盘-test4。
本地同源完整包保留在 outputs/test4-release-ready/。
原始DROID轨迹在约30.2mm位置修正及显式初态归位后通过；V1、迁移及浏览器验收通过。
附1920×1080/30fps Blender视频，校验与修正前后报告齐全；公共包335个文件复核通过，未推送GitHub。
9004运行该发布包的本地副本。单次任务成功不等于完成相机标定或随机初态验收。
最新交付说明见 test4_RELEASE_READY.txt；随包限制见 test4_RELEASE_NOTES.txt。
以下为输入适配阶段及早期实验记录，不能替代最新候选的验证结果。
源 Blender：data/scenes/DROID_s00000_r00000.blend，保持用户上传文件不变。
SHA256：e4b479bff697d6dab5dbee0f560ae39531b0ecebc4d6271d32e30db948eeb1c1
源数据：data/droid/DROID_s00000_r00000/trajectory.npz，284 帧、15 fps。
模型采用七个对象旋转组成的父子关节链，不含 Armature；不需要返工成骨架。

输入阶段复现（从项目根目录）：
  source local/env.sh
  "$B2S_PYTHON" tools/r2pear/b2s/run.py \
    --blend data/scenes/DROID_s00000_r00000.blend --out runs/test4-input \
    --config configs/test4_droid_s00000_r00000_input.json --skip-validate

该输出是适配诊断产物，不是合格的仿真环境。蓝环等几何仍会落到原有 unsupported→box 退化，
条纹细环还会被旧分类器判为 cylinder；这些表示都会丢失中孔，不能用来验收抓取成功。
输入适配阶段没有修改碰撞/摩擦/质量参数。后续实验只在 outputs/test4-ring-experiments/
替换任务环并比较 CoACD/SDF，没有回写此输入阶段 spec，也没有运行 V1/V2、oracle 或发布。

输入适配：
- extract_blend 增加逐帧局部 matrix_basis；run.py 在提取器变更后自动刷新缓存。
- embodiment 支持 object_joints 表示，按显式 chain 或七个来源列标签读取关节；
  检查父子顺序、固定基座与单轴旋转，继续复用 robot_arm 的 FK 校验与符号/零点检查。
- 七关节与原始 NPZ 最大差 9.9924e-8 rad，FK 最大离散约 0.0003（mm/deg）。
- droid_trajectory 在帧号/fps/关节/校验值匹配后读取；demo.npz 的 q 为 (284,7)，
  g 与原始 observation.gripper_position 完全一致，g_cmd 与 action_dict.gripper_position 完全一致。
  不把本组 s/r 编号误当成 LeRobot episode_index，不再使用推测夹爪指令。
- Blender 仍显示原机械臂网格，按局部关节角驱动，保留父子关系和 parent inverse。
  10 个夹爪局部动画对象按实测闭合量插值；实测范围外采用端点段视觉外推，不能称为机构标定。
  本数据实测闭合范围约 [0,0.669604]；[0,1] 的端点只检查有限性，超出实测范围的机构精度未验证。
- wrist 正确绑定 link7，逐帧相对位置离散 0.00026189 mm；source_view 属性与原视频名称关联。
  相机与场景尺度仍是建模拟合值，不是原始数据提供的标定。
- 搬迁交付的运行时代码已纳入 object_motion.py；旧骨架、手持夹爪分支保留。

验证：
- 284 帧、85 个机器人子树对象，对比源 Blender 世界矩阵，最大位置误差 4.106e-7 m，
  最大矩阵元素误差 1.083e-6。包含机械臂、夹爪连杆和腕部相机。
- 实际 blend_worker 主视角/wrist 在第 1、165、284 帧均成功渲染，已检查抓取及放下画面。
- 新增关节/FK、错误列标签及层级、移动关节原点、NPZ 时间/关节对齐、夹爪属性绑定反向测试。
  全部 b2s 测试 28 passed（23.75 s）；test1/test2/test3 原 spec 的 Blender 首帧也均正常出图。
- 证据 local/test4-input-validation/binding-validation.json、local/test4-input-tests.log、
  local/test4-input-pipeline.log；最初未适配的失败记录 local/test4-baseline.log。

后续物理需关注：
- 模型圆环外径约 92.71 mm、内径约 37.34 mm、高约 26.42 mm（采样网格估计）；
  抓取是夹环壁，实测停止开口约 28.5 mm。旧 grasp_width 外径比较不适用于这种抓法。
- 原演示修复逻辑还会调整蓝环 221–223 帧的参考物体姿态；未改原 Blender 或 q/g/g_cmd。
  后续需检查原始释放接触时序，不能将参考物体轨迹修复当作物理成功。
- 某些陪衬圆环初态倾斜/堆叠；当前通用表示只有初始 yaw，这也须在物理阶段核查。
- 圆环处理资料与候选方案见 test4_collision_research.txt。先确定几何，再检验接触，不预先调摩擦蒙过。
