圆环碰撞实验（项目根目录执行）
============================

独立实验工具，不改生产 spec、不启动服务。结论见 configs/test4_collision_experiments.txt。
需要已有 runs/test4-input/stage0_spec/{task_spec.json,demo.npz}；输入流程见 configs/test4_README.txt。
依赖使用当前项目 .venv 中的 bpy/numpy/scipy/mujoco；新增包隔离安装：

  source local/env.sh
  local/bootstrap/bin/uv pip install --python .venv/bin/python \
    --target local/ring-experiment-deps --no-deps coacd==1.0.14 trimesh==5.2.1

以下使用新的 outputs/test4-ring-repro/，避免覆盖原实验记录。
提取完整任务环并生成两档凸分解（mesh.npz 相同）：

  .venv/bin/python tools/r2pear/experiments/extract_collision_mesh.py \
    --blend data/scenes/DROID_s00000_r00000.blend --object '蓝环 · 九点防滑纹' \
    --out outputs/test4-ring-repro/assets
  mkdir -p outputs/test4-ring-repro/assets-2mm
  cp outputs/test4-ring-repro/assets/mesh.npz outputs/test4-ring-repro/assets-2mm/
  cp outputs/test4-ring-repro/assets/mesh-source.json outputs/test4-ring-repro/assets-2mm/
  PYTHONPATH=local/ring-experiment-deps .venv/bin/python tools/r2pear/experiments/prepare_ring_collision.py \
    --assets outputs/test4-ring-repro/assets --threshold-m .001
  PYTHONPATH=local/ring-experiment-deps .venv/bin/python tools/r2pear/experiments/prepare_ring_collision.py \
    --assets outputs/test4-ring-repro/assets-2mm --threshold-m .002

默认四项局部测试（probe/rest/drop/pinch），plane 支撑，同时运行完整场景：

  .venv/bin/python tools/r2pear/experiments/compare_ring_collision.py \
    --assets outputs/test4-ring-repro/assets --out outputs/test4-ring-repro/coacd --method coacd --full-scene
  .venv/bin/python tools/r2pear/experiments/compare_ring_collision.py \
    --assets outputs/test4-ring-repro/assets --out outputs/test4-ring-repro/sdf --method sdf --full-scene

更换 box 支撑，仅影响局部夹具，不改变完整场景原桌面。相同开关用于两条路线：

  .venv/bin/python tools/r2pear/experiments/compare_ring_collision.py \
    --assets outputs/test4-ring-repro/assets --out outputs/test4-ring-repro/coacd-box --method coacd --support box
  .venv/bin/python tools/r2pear/experiments/compare_ring_collision.py \
    --assets outputs/test4-ring-repro/assets --out outputs/test4-ring-repro/sdf-box --method sdf --support box
  .venv/bin/python tools/r2pear/experiments/compare_ring_collision.py \
    --assets outputs/test4-ring-repro/assets-2mm --out outputs/test4-ring-repro/coacd-2mm-box \
    --method coacd --support box --full-scene

SDF 碰撞网格预生成距离场的补充尝试（约 208 MB 的 MJB；--cases 空列表跳过局部试验）：

  .venv/bin/python tools/r2pear/experiments/compare_ring_collision.py \
    --assets outputs/test4-ring-repro/assets --out outputs/test4-ring-repro/sdf-prepared-meshes \
    --method sdf --cases --full-scene --prepare-mesh-sdf
  .venv/bin/python tools/r2pear/experiments/reproduce_sdf_failure.py \
    --run outputs/test4-ring-repro/sdf-prepared-meshes

结果 results.json 必须同时检查 complete、error 和 success：脚本正常退出仅表示记录已写出，
不代表物理或任务通过。局部是否稳定、持住及松手返回各自独立记录；完整场景中断时 success=null。
每项动力学/回放默认墙钟预算 180 s。模型编译不包含在此预算内。
XML 引用本地绝对网格路径，只用于本机诊断，不能当作可移植发布包。
各轨迹 NPZ 存 qpos；完整回放另存 final_qpos，便于单状态错误复现。
此工具记录简化实验；正式发布仍必须走原验收/导出流程，并附 Blender 高清视频。
