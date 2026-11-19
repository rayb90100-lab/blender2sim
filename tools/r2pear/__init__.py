"""r2pear：把 Real2Gym 式重建结果封装成 PEAR 协议兼容的 MuJoCo 环境（test5 起，论文外扩展，见 DEVELOPMENT_LOG §61–）。

只读通用任务规格 task_spec.json，不引用具体 test 的物体名/帧号：
  spec.py        读写与校验 task_spec.json
  sim.py         由 spec 构建 MjModel（机器人、地面、桌子、物体、相机、力矩上限）
  controller.py  末端位姿增量 → 路点 + 阻尼最小二乘 IK → 关节位置伺服（参照 PEAR robolab/libero 控制器）
  predicates.py  终态谓词（on / upright / released …）
  env.py         R2PEnv：PEAR StatefulEnv 接口（open / step / close）
"""
