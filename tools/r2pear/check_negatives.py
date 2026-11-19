#!/usr/bin/env python3
"""反向对照：未完成任务时谓词必须判失败（防止环境「送分」）。

  N0 不动：初始状态
  N1 放在桌上：抓起绿方块，放到塔旁 10 cm 的桌面
  N2 推倒塔：张开夹爪，从侧面横扫塔
  N3 未松手：放到塔顶但不张开夹爪
  P  正例：oracle 完整执行
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import env as E  # noqa: E402
import oracle as O  # noqa: E402
import predicates as PR  # noqa: E402


def fresh(spec):
    e = E.R2PEnv(spec, render=False)
    e.open()
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}

    def record(tag, e, expect):
        ok, det = PR.evaluate(e.m, e.d, e.spec)
        res[tag] = {"success": ok, "expected": expect, "pass": ok == expect,
                    "failed_predicates": [f"{p['pred']}({p.get('obj', '')})" for p in det if not p["ok"]]}

    e = fresh(a.spec); record("N0_no_action", e, False)

    e = fresh(a.spec)
    p, yaw, _o = O._obj(e, a.obj)
    y = O._grip_yaw(yaw, e.robot.state()["arm"]["yaw"])
    gx, gy, gz = p[0] * 100, p[1] * 100, max((p[2] - e.robot.z0) * 100, O.GRASP_Z_MIN_CM)
    for args in ((gx, gy, gz + 10, y, 1.0), (gx, gy, gz, y, 0.0), (gx, gy, gz, y, -1.0), (gx, gy, gz + 10, y, 0.0),
                 (gx, gy - 10, gz + 10, y, 0.0), (gx, gy - 10, gz + 0.5, y, 0.0), (gx, gy - 10, gz + 0.5, y, 1.0),
                 (gx, gy - 10, gz + 10, y, 0.0)):
        O._goto(e, *args)
    record("N1_placed_on_table", e, False)

    e = fresh(a.spec)
    sp, syaw, _so = O._obj(e, a.support)
    sx, sy = sp[0] * 100, sp[1] * 100
    for args in ((sx, sy - 12, 15, 0.0, 0.0), (sx, sy - 12, 4, 0.0, 0.0), (sx, sy + 12, 4, 0.0, 0.0),
                 (sx, sy + 12, 15, 0.0, 0.0)):
        O._goto(e, *args)
    record("N2_swept_tower", e, False)

    e = fresh(a.spec)
    log = []
    orig = O._goto
    # oracle 去掉 release/retreat：用 lower 后直接判定
    steps_done = []

    def no_release(env, x, y, z, yaw, dwidth=0.0, log=None):
        if len(steps_done) >= 6:                     # 第 7 步 release 起不执行
            return {"done": False}
        steps_done.append(1)
        return orig(env, x, y, z, yaw, dwidth, log)
    O._goto = no_release
    O.solve(e, a.obj, a.support)
    O._goto = orig
    for _ in range(500):
        e.robot.mj.mj_step(e.m, e.d)
    record("N3_not_released", e, False)

    e = fresh(a.spec)
    r = O.solve(e, a.obj, a.support)
    record("P_oracle", e, True)

    allpass = all(v["pass"] for v in res.values())
    out = {"all_pass": allpass, "cases": res}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
