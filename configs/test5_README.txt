test5 | DROID_s00000_r00008 | 红甜椒从浅盘取出放到桌面
====================================================
当前已跑通并发布到 /root/data/simulation/DROID-红甜椒从盘中取出放到桌面-test5。
9005 → 8095，服务 local/test5-service.sh；默认加载 outputs/test5-release-ready/interactive/scene。
473个文件发布后复核通过；完整交付与视频说明见test5_RELEASE_READY.txt。
源场景不改：data/scenes/DROID_s00000_r00008.blend。
详细数据、物理假设、未标定项见 test5_RELEASE_NOTES.txt。

复现（项目根目录 source local/env.sh）：
1. .venv/bin/python tools/r2pear/b2s/run.py \
     --blend data/scenes/DROID_s00000_r00008.blend --out runs/test5-input \
     --config configs/test5_droid_s00000_r00008_input.json --skip-validate
   这是输入诊断阶段，box碰撞退化导致V1失败，不能直接发布。
2. tools/r2pear/experiments/extract_collision_mesh.py 按配置中四个对象分别导出至
   outputs/test5-collision/{red_pepper,metal_plate,small_fruit,brown_item}/。
   PYTHONPATH=local/ring-experiment-deps .venv/bin/python \
     tools/r2pear/experiments/prepare_convex_collision.py --assets <对象目录> \
     --threshold-m .001 --density <600；金属浅盘7800>
3. .venv/bin/python tools/r2pear/experiments/prepare_test5_scene.py
   生成 runs/test5-preview，保留完整源网格位姿、开启多点接触，原始回放成功但初态漂移超标。
4. .venv/bin/python tools/r2pear/experiments/prepare_release_initial.py \
     --source runs/test5-preview --out <新的release目录>
   当前 runs/test5-release 已生成；不要覆盖。显式修正静置初态，报告保留修正前失败。
5. B2S_TESTDIR=<场景目录> bash local/test5-service.sh restart
   自动补齐9005兼容入口；其他test服务不动。

当前证据：
- runs/test5-input/registration-report.json：整体P95约0.000135mm；建模夹持点局部差30.64mm。
- runs/test5-preview/collision-report.json：凸分解、质量估计和完整位姿。
- runs/test5-release/initialization-report.json：修正前后V1及原始轨迹V2。
- outputs/test5-diagnostics/source-replay.json：逐帧双指接触、102.65mm抬升与最终通过。
- outputs/test5-release-ready/：最终完整交付，含修复颜色的GLB、交互归档和Blender高清视频。
- local/test5-delivery-browser-final/：最终包真实浏览器验收通过；local/test5-publication.json为公共副本校验。
- local/test5-web-check/：9005首页/身份与MuJoCo/Blender各三路真实图片。
- local/test5-tests.log：37项测试通过。

共同管线：robot_arm/FK → DROID原始关节与指令 → spec → MuJoCo → 同源Blender/WebUI。
本次新增的是显式夹爪属性名配置；复用已有父子对象关节链、visual_roots、凸分解、初态报告及导出流程。
没有用oracle替代原始轨迹，也没有根据单次抓取平移任务区。单初态通过不是完整标定或随机泛化验收。
