"""终态谓词：只看仿真当前状态（环境侧可用真值；策略看不到）。"""
import numpy as np

GRIP_PREFIX = "g_"
REST_LIN, REST_ANG = 0.02, 0.5         # m/s, rad/s：判为静止
ON_FORCE_FRAC = 0.5                    # 支撑法向力 ≥ 50% 自重
TILT_DEFAULT = 10.0


def _body_contact_force(m, d, body_a, pred_b):
    """累加 body_a 与满足 pred_b 的 body 之间的法向接触力。

    注意 d.ncon 与实际可读的 contact 数组可能不一致：mujoco 用的是固定大小的
    contact buffer（nconmax），reset 之后立即读、或离散步中间读，会拿到
    ncon 大于已填充条目的状态 → d.contact[i] 抛 IndexError。
    实测：web_ui 里 POST /cmd {"t":"reset"} 之后紧接着 GET /state 就崩，
    异常把整个 worker 线程打死，之后命令队列再也不消费（表现为"点了没反应"）。
    这里逐条 try，读不到就跳过 —— 少一条接触力对判定无实质影响。
    """
    import mujoco
    a = m.body(body_a).id
    f6 = np.zeros(6)
    total = 0.0
    ncon = min(int(d.ncon), len(d.contact))
    for i in range(ncon):
        try:
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        except (IndexError, TypeError):
            continue
        if a not in (b1, b2):
            continue
        other = m.body(b2 if b1 == a else b1).name
        if pred_b(other):
            mujoco.mj_contactForce(m, d, i, f6)
            total += abs(f6[0])
    return total


def tilt_deg(m, d, obj):
    R = d.xmat[m.body(obj).id].reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), -1, 1))))   # 箱体对称：上下颠倒也算竖直


def at_rest(m, d, obj):
    j = m.joint(f"{obj}_free")
    v = d.qvel[j.dofadr[0]:j.dofadr[0] + 6]
    return bool(np.linalg.norm(v[:3]) < REST_LIN and np.linalg.norm(v[3:]) < REST_ANG)


def evaluate(m, d, spec):
    """返回 (success: bool, detail: list[dict])。所有谓词成立且所有物体静止才算成功。"""
    out = []
    for p in spec["success"]:
        if p["pred"] == "on":
            w = m.body(p["obj"]).subtreemass[0] * 9.81
            f = _body_contact_force(m, d, p["obj"], lambda n: n == p["support"])
            above = d.xpos[m.body(p["obj"]).id][2] > d.xpos[m.body(p["support"]).id][2]
            ok = bool(f >= ON_FORCE_FRAC * w and above)
            out.append({**p, "ok": ok, "force_N": round(f, 3), "weight_N": round(w, 3)})
        elif p["pred"] == "released":
            f = _body_contact_force(m, d, p["obj"], lambda n: n.startswith(GRIP_PREFIX))
            out.append({**p, "ok": bool(f < 0.05), "grip_force_N": round(f, 3)})
        elif p["pred"] == "upright":
            t = tilt_deg(m, d, p["obj"])
            out.append({**p, "ok": bool(t <= p.get("max_tilt_deg", TILT_DEFAULT)), "tilt_deg": round(t, 2)})
    rest = all(at_rest(m, d, n) for n in spec["objects"])
    out.append({"pred": "all_at_rest", "ok": rest})
    return all(x["ok"] for x in out), out
