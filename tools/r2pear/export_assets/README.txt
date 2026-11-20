可复用导出流程
================

要求：Python 3.11、项目运行依赖、bpy、ffmpeg、Menagerie 模型，及已经验证的 b2s 场景。
导出不会修改输入场景、现有 Web 服务或公共目录。输出目录必须尚不存在。
发布必需交付：展示回放 + 可交互环境 + Blender 高清视频 + 真实/Blender左右对比视频。
缺少同源且验证通过的Blender视频或compare视频，publish.py均拒绝发布。

1. 一次性获取并校验离线展示依赖（固定 Three.js r170，附许可证）：
   python tools/r2pear/export_assets/fetch_viewer.py --out local/export-vendor

2. 使用项目环境导出（先设置 MENAGERIE，及服务器适用的 EGL 配置）：
   python tools/r2pear/export_scene.py --testdir runs/test1 --out outputs/test1-export \
     --scene-id b2s_test1_droid_ep71 --dataset DROID --episode ep71 \
     --title '绿色物块放到黄色长方体' --viewer-assets local/export-vendor
   python tools/r2pear/export_scene.py --testdir runs/test2-cup-saucer --out outputs/test2-export \
     --scene-id b2s_test2_umi_ep68 --dataset UMI --episode ep68 \
     --title '蓝杯放到蓝色杯碟' --viewer-assets local/export-vendor
   python tools/r2pear/export_scene.py --testdir runs/test3 --out outputs/test3-delivery \
     --scene-id b2s_test3_egodex_basic_pick_place_1 --dataset EgoDex --episode test/basic_pick_place/1 \
     --title '手机放入浅纸盒盖' --viewer-assets local/export-vendor \
     --scene-notes configs/test3_export_notes.txt --camera agentview_2
   --scene-notes 可将场景来源、适配项及验证边界同时打入交付目录和交互归档。
   --camera 可指定已有的世界固定相机，避开默认环绕相机落在墙后或墙内的情况。

3. 可选的浏览器依赖（只用于验证，不随包要求安装）：
   pip install playwright==1.51.0
   python -m playwright install chromium
   系统还需满足 Chromium 所需运行库。

4. 在浏览器检查 WebGL 回放和真实交互按钮，保存截图及带文件校验值的测试报告：
   python tools/r2pear/export_assets/verify_browser.py --bundle outputs/test1-export --out local/test1-export-browser
   python tools/r2pear/export_assets/verify_browser.py --bundle outputs/test2-export --out local/test2-export-browser
   脚本使用临时本机端口，结束后关闭；不会占用原来的 8091/8092。

5. 制作并检查 Blender 高清视频（必需，具体说明见下方）：
   python tools/r2pear/render_blender_video.py --bundle outputs/test1-export \
     --out outputs/test1-blender-hd --camera agentview --gpu 0
   先在独立目录加 --preview-only 检查首/中/末取景，再渲染全片；test2 使用 export_view，test3 使用 agentview_2。

6. 用真实视频与既有Blender HD生成左右对比（必需）：
   python tools/r2pear/render_compare_video.py --bundle outputs/test1-export \
     --blender-video-dir outputs/test1-blender-hd/delivery \
     --source-video <对应真实视频.mp4> --source-fps 15 --source-frame-offset 0 \
     --source-provenance '<数据集/episode、来源和帧对应依据>' --source-view exterior_1_left \
     --view-note 'Corresponding exterior camera; approximate calibration' --out outputs/test1-compare
   先检查真实来源与动作帧；工具根据HD的物理记录状态索引取源帧，不按整段视频时长拉伸。
   --source-frame-offset是零基裁剪量；UMI/test2为683，保留原视频第684～924帧。
   EgoDex/test3源30fps、执行10fps，按状态索引自然对应3倍慢速。仿真结束静置时左侧定格末帧。
   保持左右原比例，鱼眼不强行去畸变。不同机位必须写在view-note；该对比不是标定误差验收。
   必要时--source-crop X Y W H仅提取已确认的真实面板，必须在provenance中说明二次编码来源。
   输出compare.mp4、compare-poster.png、compare-report.json、COMPARE_README.txt。

7. 验证通过后连同两种视频一次发布（显式操作，目标必须不存在）：
   python tools/r2pear/export_assets/publish.py --bundle outputs/test1-export \
     --browser-report local/test1-export-browser/browser-validation.json \
     --blender-video-dir outputs/test1-blender-hd/delivery \
     --compare-dir outputs/test1-compare \
     --destination /root/data/simulation/DROID-绿色物块放到黄色长方体-test1
   test2/test3 同理。脚本核对静置、回放、迁移、浏览器、视频来源和全部文件校验值后，先复制到临时目录再改名。
   若输入包已有经过校验的blender-hd/与compare文件，可分别省略--blender-video-dir和--compare-dir。
   任何测试失败、文件损坏、测试报告对应其他产物或目标已存在都会拒绝发布。

导出产物：
  scene.json / mujoco-web3d-{scene.glb,replay.json,manifest.json}
  mujoco-sim.mp4 / mujoco-poster.png / mujoco-final.png
  preview.html / viewer/                 离线依赖，不需要外部 CDN
  interactive/                          可运行代码、模型、源 blend、示范与本次回放轨迹
  interactive-environment.tar.gz         上述环境的便携归档
  validation.json / provenance.json / checksums.json / export-status.json
  blender-hd/                           发布时加入：Blender 高清 MP4、首尾 PNG、说明及验证报告
  compare.mp4 / compare-poster.png      左真实、右既有Blender HD；1920×640，保持比例
  compare-report.json / COMPARE_README.txt  来源哈希、逐帧对应、裁剪/速度/视角说明

验证边界：
  原生 XML 和 spec 构建模型的一致性检查、真实控制/重置/oracle、模型重建后渲染、
  改变工作目录和解耦模型路径后的运行、GLB 节点/坐标/首帧位姿，以及浏览器回放与交互。
  迁移验证复用已安装的 Python 依赖；不会声称在全新 OS 上做过安装测试。
  Three.js 回放只显示已记录物理状态，实时仿真由 interactive/run.py 提供。
  支持带 blend_binding 的 b2s 场景。当前验证了 test1～test5，不代表任意新任务都可自动通过。
  test3 的来源和限制见 configs/test3_export_notes.txt；发布包附 SCENE_NOTES.txt。
  GLB 使用原场景几何及支持的材质，程序化材质可能与 Blender 渲染不同。
  模型近似、接触穿透和相机近似等限制会继续保留在 validation.json 和导出说明中。

必需的 Blender 高清视频（不修改实时服务）：
  python tools/r2pear/render_blender_video.py --bundle outputs/test3-delivery \
    --out outputs/test3-blender-hd --camera agentview_2 --gpu 0
  默认 1920×1080、30 fps、Cycles 64 samples、OptiX 与降噪，需可用的 NVIDIA 渲染设备。
  可先加 --preview-only 在另一个输出目录检查首、中、末帧；--device CPU 可使用 CPU。
  --camera export_view 可复用已发布 Web3D 的展示相机；test2 使用此角度避免主相机被机械臂遮挡。
  视频保留 native_trace.npz 的时序，用最近的物理状态采样到 30 fps，不重新仿真或插值生成动作。
  输出下 delivery/ 为可发布的 MP4、首尾 PNG、README 和来源/编码验证报告；frames/ 留作本地证据。
  若 PNG 全部完成但编码中断，可用同一命令加 --encode-only 恢复；会先核对原状态请求和每帧尺寸。
  以下追加工具仅用于补齐以前已发布的旧场景；新场景必须在初次发布时包含视频：
  python tools/r2pear/export_assets/publish_video.py --video-dir outputs/test3-blender-hd/delivery \
    --scene-dir /root/data/simulation/EgoDex-手机放入浅纸盒盖-test3
  发布器核对原始场景和轨迹哈希，只新增 blender-hd/ 并更新 checksums.json，拒绝覆盖已有补充目录。
  原 scene.json、展示回放、交互归档及其浏览器验证绑定均保持不变。

DROID/test4 发布补充（2026-11-20）
- 同名凸分解网格按内容哈希区分，显式MJCF转换为随包相对路径；迁移不得依赖实验输出目录。
- 自动交互验证使用场景配置的执行路径：source_demo检查完整原始回放，oracle检查oracle。
  验证报告的automatic_kind/automatic_success区分两者，不将原始回放结果写成oracle成功。
- registration-report.json记录整体/局部偏移告警与修正依据；initialization-report.json保留
  未修正初态的失败、逐物体修正量及修正后V1/V2。显式场景归位不能被描述为原建模初态通过。
- test4场景与边界见configs/test4_RELEASE_NOTES.txt；源关节和夹爪指令仍保持DROID记录。

程序化颜色兼容：GLB导出时对Noise/Voronoi/Musgrave驱动的Color Ramp基础色使用控制点均值，
避免glTF不支持该节点链时把目标物变白。仅改变临时导出进程，图像纹理与原始blend不改；
近似记录在*.glb.materials.json中。仍需用真实浏览器检查物体颜色，不能只验文件和动画绑定。

为已有公共场景补齐compare（不修改现有环境、视频、scene.json和浏览器报告）：
  python tools/r2pear/export_assets/publish_compare.py --compare-dir outputs/test5-compare \
    --scene-dir /root/data/simulation/DROID-红甜椒从盘中取出放到桌面-test5
补充发布器检查已发布HD/物理来源和帧映射，只新增四个compare文件并扩展checksums.json，
拒绝覆盖既有compare。源真实视频不另复制进包，来源和SHA256保留在compare-report.json。
五个场景的具体来源与限制见configs/COMPARE_VIDEOS.txt。
