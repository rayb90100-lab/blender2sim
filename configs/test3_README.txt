test3 | EgoDex test/basic_pick_place/1 | 手机放入浅纸盒盖
=====================================================

源资产：data/scenes/EgoDex_basic_pick_place_0001.blend（原上传名 EGO.blend，仅改名）。
源数据：data/egodex/basic_pick_place_1/，含官方 MP4、HDF5 和 source.json 校验收据。
通过场景属性、160 帧/30 fps 以及真实视频首尾布局核对来源；与公共展示 E01 玩具入蓝盘不是同一场景。

复现（项目依赖和 Menagerie 已安装）：
  source local/env.sh
  "$B2S_PYTHON" tools/r2pear/b2s/run.py \
    --blend data/scenes/EgoDex_basic_pick_place_0001.blend --out runs/test3 \
    --config configs/test3_egodex_basic_pick_place_0001.json
  "$B2S_PYTHON" tools/r2pear/web_ui.py --testdir runs/test3 --host 127.0.0.1 --port 8093
本机已由 local/test3-service.sh 管理，避免重复启动。

场景配置与新增通用能力：
- 中文夹爪识别；右夹爪为演示者，左夹爪隐藏并排除物理。
- 手机中框作为单刚体碰撞近似，visual_roots 驱动整个手机父对象，镜头和按键不再拆成独立物体。
- 明确配置的薄盒底不受自动候选厚度/尺寸过滤，能够参与初态支撑关系和终态谓词推导。
- 四面盒壁及背景墙/搁板保留碰撞，不能删掉障碍以获得通过。
- 夹爪开口由两指动画标记测量：78.24–104.24 mm，映射到 85 mm 目标夹爪并明确报告开口截断。
- 78 mm 宽物体不能套用固定 20% 的闭合阈值；使用 0.04，命令提前 2 个源帧，挤压偏置 4 mm。
  此例禁用针对 UMI 噪声的闭合尖峰过滤，避免删掉缓慢接近/放下过程中的真实闭合动作。
- 基座搜索使用与运行模型一致的背景有向 box，并包含 g_ 夹爪几何；合格候选优先较低力矩。
  当前基座由该搜索得到并写入配置，重建时仍重新检查碰撞与 IK。
- 原速回放抓取失败；6 组有界对照表明 3 倍执行时间、原 4 mm 挤压即可成功。
  retarget_time_scale=3 保留路径和开合采样；160 帧从源 30 fps 改为执行 10 fps。
- oracle 对过长矩形仅允许绕 z 的 180° 抓取等价姿态，避免沿 163 mm 长边夹持；
  phone.grasp_z_min_cm=0.65 使薄手机侧面进入指垫接触区，不使用原 2.5 cm 的默认抓取高度。
- 腕部相机为 additional_cameras 配置的虚拟相机，不是 EgoDex 的真实相机；Blender 同样支持该视图。
  已修复新建相机默认 10 cm 近裁剪切掉夹爪连接机构的问题：near_clip_m=0.001，
  同姿态前后对照确认连接件恢复。相机位置未改，白色腕部仍在当前视野之外。

验证（2026-11-19）：
- V1：通过，最大物体漂移 0.064468 mm，最深接触 -1.298721 mm。
- V2：通过，on(phone, box_lid)、released、upright、all_at_rest 全满足。
  最深接触 -1.346703 mm；关节跟踪中位 0.7924°、最大 13.8465°。
  相对 Blender 手机参考轨迹位置差：中位 2.0939 mm、最大 31.6641 mm、末态 2.1304 mm。
- 独立交互 oracle：90 步成功；浏览器实际按钮 oracle：83 步成功。
- 浏览器重置后步数 0，MuJoCo 两路及 orbit 正常；Blender agentview/wrist/orbit 出图正常，无脚本错误。
- 当前服务：8093；8091/test1 和 8092/test2 不受影响。

证据：
  runs/test3/stage0_spec/{task_spec.json,decisions.json,demo.npz}
  runs/test3/stage1_validate/validate.json
  local/test3-baseline.log、local/test3-pipeline.log：最初解析失败、漏查背景碰撞的失败记录。
  runs/test3-static-check/：修好背景碰撞但原速回放失败。
  local/test3-contact-diagnosis.json、local/test3-replay-probe.json：接触诊断和速度/夹持对照。
  local/test3-oracle.json、local/test3-browser/validation.json 及其截图。

验证边界：
- 本次通过的是 EgoDex 来源的 Blender 夹爪动画 → Panda/MuJoCo 场景与交互。
- HDF5 人手适配器可读出 160 帧有限末端位姿，但原始人手与 Blender 的度量坐标对齐尚未验证，
  本次不使用原始人手 HDF5 作为控制轨迹，不宣称该分支已端到端通过。
- Blender 尺度来自单目估计；碰撞为当前几何近似，成功不等于高精度接触。
- 源头戴相机运动在当前仿真中固定为首帧；V3 源视频标定对照未通过，此轮报告明确为 skipped。
- 改动尚未提交/推送；test3 已完成展示及交互包导出，发布记录见下。

三种来源的共用范围：
- test1 / DROID：Blender 机器人关节动画经过 FK 校验后直接回放，夹爪指令来自匹配的 DROID 数据。
- test2 / UMI：Blender 中的手持夹爪位姿及开合属性 → 基座搜索、IK → Panda；
  当前杯碟任务还使用场景专用补齐脚本恢复杯碟、视觉跟随和多点接触。
- test3 / EgoDex：Blender 中的手持夹爪位姿及两指开口标记 → 同一基座搜索、IK → Panda；
  需要显式物体/视觉根配置、夹持适配、薄手机抓取高度及三倍执行时间。
- 三者共用场景提取/解析、任务 spec、MuJoCo、控制器、谓词、Web/Blender 显示和导出框架。
  这是跨来源 Blender 场景转换的初步通用性证据，不是三种原始数据格式都已全自动转换的证明。
  目前仅三个场景，任务都属于抓取放置，目标机器人也相同；更多 DROID 场景及原始人手分支仍待验证。

最终导出与发布（2026-11-19）：
- 公共目录：/root/data/simulation/EgoDex-手机放入浅纸盒盖-test3/
- 本地包：outputs/test3-delivery/；浏览器证据：local/test3-delivery-browser-publish/。
- 导出使用原有 agentview_2 世界固定全览相机，避开默认环绕位置的墙面遮挡；
  视频、Web3D 预览均已检查首尾图。展示回放 170 个状态样本、时长 16.9 s，MP4 为 255 帧/15 fps。
- 交互归档包含近裁剪修复、相对路径源 Blender、MJCF/机器人模型、任务 spec、控制与重置代码。
- V1/V2 通过，独立迁移 oracle 90 步成功；实际浏览器 oracle 83 步成功，重置后 0 步，脚本错误 0。
  浏览器已检查 Blender agentview/wrist/orbit 三路；旧的整幅墙面遮挡导出会被可见性检查拒绝。
- 原有及新增测试共 17 passed（29.91 秒），git diff --check 通过。
- 包约 111.48 MiB，交互压缩归档约 15.72 MiB；发布时核对全部文件哈希，拒绝覆盖已有目录。
- 9003 整页空白曾因隧道指向服务器 9003 而非 8093；本机增加 9003→8093 回环兼容转发，
  用户已确认恢复。它复用同一实例，入口脚本 local/test3-forward-alias.py，无开机自启。

高清补充视频：
- 已追加到公共目录 blender-hd/blender-hd.mp4，1920×1080、30 fps、508 帧、16.933333 s，约 3.52 MiB。
- 使用 Cycles 64 samples + 降噪及原有 agentview_2 相机，保留同一条已验证物理记录的时序。
  30 fps 为显示采样，未增加原始 10 Hz 轨迹的物理信息；首尾高清图、来源与编码验证报告同目录提供。
- 原 scene.json、展示回放和交互归档未改；公共 checksums.json 已纳入新增补充文件。

原速复测（2026-11-20）：
- 当前代码从原 Blender 重新提取/解析/IK/验证，配置
  configs/test3_egodex_basic_pick_place_0001_original_speed.json，唯一配置差异为 retarget_time_scale=1。
- 输出 runs/test3-original-speed/；新 q 和 g_cmd 与原 runs/test3 逐元素完全相同，fps=30。
- 原速 V1 通过；V2 失败，手机最大抬升 0.05713 mm，关节误差中位 2.5092°/最大 46.9894°。
  手机未进入盒盖，相对 Blender 末态误差 413.4431 mm。
- 同一新场景/同一目标仅改 fps=10 对照：成功，抬升 139.6508 mm，跟踪最大误差 13.8465°。
- 证据 stage1_validate/speed-comparison.json；原速视频 original-speed.mp4 为 960×540/30fps，
  190帧，含结束静置。物理子步存在取整；源30Hz对应每区间0.034秒。
- 原速试验时 local/test3-service.sh 曾暂切 runs/test3-original-speed，8093/9003。
  运行 spec 单独配置 web_playback 并绑定 demo SHA，按钮执行原速重定向轨迹而非 oracle。
  重建后网页回放绑定需重新添加；恢复旧网页可用 B2S_TESTDIR="$PWD/runs/test3" bash local/test3-service.sh restart。
- 1×指物理执行时序，Blender逐步等待渲染仍可能增加墙钟耗时。公共三倍时长版本未替换。

原速复测后恢复（2026-11-20）：
- 按用户要求，test3服务默认恢复 runs/test3，demo为10Hz，执行时长为源30Hz的3倍。
  运行spec绑定现有demo哈希，网页按钮为“回放 test3 轨迹（1/3 速度）”，实际执行这条轨迹。
  公共发布包未改，原速实验保留。
- 管线内已有retarget_time_scale参数和决策记录；validate.py会输出跟踪误差和任务结果。
  速度扫描/接触时序对照目前仍是test3专项诊断脚本，不是run.py自动诊断环节，亦不会失败后自动降速。

速度诊断主流程接入（2026-11-20，替代上节“专项脚本”状态）：
- run.py末尾现已接入speed_diagnosis.py，默认跳过；常规排查无解后使用
  --last-resort-speed-diagnosis "已排查项和剩余问题"启用。
- 在常规验证和已启用的环境测试之后执行，仅做报告，不自动应用降速。
- test3完整入口回归见runs/test3-speed-diagnosis-smoke/：原速失败，半速与1/3速度成功，
  stage1_validate仍保留原速失败，demo仍30Hz。现有服务和公共三倍时长版本未改。
