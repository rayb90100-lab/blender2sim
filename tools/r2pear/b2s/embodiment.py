"""b2s 第 1a 步：识别演示者的具身形态（robot_arm / human_hand / handheld_gripper），提取关节真值并校验。

robot_arm（DROID 类）
  支持骨架关节及带来源列标签/显式配置的父子对象关节链，两者共用下述 FK 校验。
  在 blend 里找「骨架中逐帧单轴旋转的骨骼链」，链长 = 关节数。每根骨骼的关节角取
  matrix_basis 的旋转向量在主轴上的投影（主轴由 SVD 求得，不假设是 X/Y/Z 哪一个）。
  校验（这是判断 blend 是否靠得住的核心）：
    把关节角喂给 MuJoCo 目标机型做 FK，对每个连杆 i 求 X_i(t) = T_mj_i(t)^-1 · T_blend_i(t)。
    若 blend 的连杆和 MuJoCo 的连杆是同一刚体，X_i 必须**逐帧恒定**（与网格原点放在哪里无关）。
    不恒定 → 关节方向或零点和 MuJoCo 不一致 → 逐关节拟合 q_mj = s·q + c（s∈{±1}），再验一次。
  机器人基座：链根骨骼所在骨架对象（或其 link0 网格）的世界矩阵。

human_hand（Ego 类）/ handheld_gripper（UMI 类）
  没有机器人关节，只能给出末端轨迹，交给 retarget.py 做基座摆位 + IK。
  识别规则见 detect()；识别不了时用 b2s_config.json 的 "embodiment" 字段手动指定。

已支持的目标机型：panda（menagerie panda_nohand + robotiq_2f85，即 r2pear/sim.py 的装配）。
"""
import numpy as np

from common import inv, rot_angle_deg, rot_of

HAND_WORDS = ("thumb", "index", "middle", "ring", "pinky", "little", "palm", "wrist", "metacarpal", "phalan")
GRIPPER_WORDS = ("gripper", "umi", "finger", "jaw", "夹爪")
ROBOT_DOF = {"panda": 7}


def _rotvec(R):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(R).as_rotvec()


def bone_joint_angles(basis):
    """basis: (T,4,4) 骨骼局部变换 → (angle(T), axis(3), off_axis_rms_deg, range_deg)。"""
    R = np.array(basis)[:, :3, :3]
    rv = np.array([_rotvec(r) for r in R])
    if np.abs(rv).max() < 1e-9:
        return np.zeros(len(R)), np.array([0, 0, 1.0]), 0.0, 0.0
    _, _, vt = np.linalg.svd(rv, full_matrices=False)
    ax = vt[0]
    # 主轴符号：让分量绝对值最大的那一维为正（Blender 骨骼一般绕局部 Y 或 Z）
    ax = ax * np.sign(ax[np.argmax(np.abs(ax))])
    ang = rv @ ax
    off = rv - np.outer(ang, ax)
    return ang, ax, float(np.degrees(np.sqrt((off ** 2).sum(1).mean()))), float(np.degrees(np.ptp(ang)))


def _bone_chain(arm):
    """返回从根到叶、逐帧有旋转的骨骼链（最长的那条）。"""
    bones = arm["bones"]
    kids = {}
    for b, v in bones.items():
        kids.setdefault(v["parent"], []).append(b)

    def longest(b):
        best = [b]
        for c in kids.get(b, []):
            p = [b] + longest(c)
            if len(p) > len(best):
                best = p
        return best

    roots = kids.get(None, [])
    chains = [longest(r) for r in roots]
    chain = max(chains, key=len) if chains else []
    return [b for b in chain if bone_joint_angles(arm["basis"][b])[3] > 0.5]


def detect(ex, cfg, D):
    forced = cfg.get("embodiment")
    cands = []
    for an, arm in ex.E["armatures"].items():
        names = " ".join(arm["bones"]).lower()
        n_hand = sum(w in names for w in HAND_WORDS)
        chain = _bone_chain(arm)
        if n_hand >= 3 and len(arm["bones"]) >= 15:
            cands.append(("human_hand", an, {"n_bones": len(arm["bones"]), "hand_words": n_hand}))
        elif len(chain) >= 5:
            cands.append(("robot_arm", an, {"chain": chain}))
    if forced:
        if forced.get("representation") == "object_joints":
            return object_joint_embodiment(ex, forced, D)
        typ = forced["type"]
        D.add("embodiment", typ, "b2s_config.json 手动指定")
        return {"type": typ, **forced}
    arms = [c for c in cands if c[0] == "robot_arm"]
    hands = [c for c in cands if c[0] == "human_hand"]
    if arms:
        typ, an, info = max(arms, key=lambda c: len(c[2]["chain"]))
        D.add("embodiment", "robot_arm", f"骨架 {an!r} 有 {len(info['chain'])} 根逐帧单轴旋转的骨骼")
        return {"type": "robot_arm", "armature": an, "chain": info["chain"]}
    if any(o.get("props", {}).get("source_field") == "observation.joint_position" for o in ex.O.values()):
        return object_joint_embodiment(ex, {}, D)
    if hands:
        typ, an, info = hands[0]
        D.add("embodiment", "human_hand", f"骨架 {an!r} {info['n_bones']} 根骨骼，含手部关键词 {info['hand_words']} 类")
        return {"type": "human_hand", "armature": an}
    # 手持夹爪：名字含 gripper/umi 且逐帧在动、没有父级在动的刚体
    dyn = [n for n in ex.E["dynamic"] if any(w in n.lower() for w in GRIPPER_WORDS)]
    roots = [n for n in dyn if not any(a in ex.E["dynamic"] for a in ex.ancestors(n))]
    if roots:
        D.add("embodiment", "handheld_gripper", f"无机械臂骨架；动态根对象 {roots[0]!r} 名字含夹爪关键词",
              warn=True)
        return {"type": "handheld_gripper", "root": roots[0]}
    D.add("embodiment", None, "没找到机械臂骨架、人手骨架或手持夹爪；请在 b2s_config.json 指定 embodiment", warn=True)
    return {"type": None}


def object_joint_embodiment(ex, config, D):
    """Accept an explicitly ordered object chain or authored DROID column tags."""
    chain = config.get("chain")
    if chain is None:
        indexed = [(o["props"].get("source_column_zero_based"), name) for name, o in ex.O.items()
                   if o.get("props", {}).get("source_field") == "observation.joint_position"]
        if len(indexed) != 7 or {i for i, _ in indexed} != set(range(7)):
            raise ValueError('Object joints need unique source columns 0..6, or an explicit embodiment.chain')
        chain = [name for _, name in sorted(indexed)]
    if len(chain) != 7 or len(set(chain)) != 7 or any(n not in ex.O for n in chain):
        raise ValueError('Object joint chain must contain seven distinct existing objects')
    for a, b in zip(chain, chain[1:]):
        if a not in ex.ancestors(b):
            raise ValueError(f'Joint hierarchy is not ordered: {a} is not an ancestor of {b}')
    base = config.get("base")
    if base is None:
        # URDF-like layouts often put a fixed-axis empty between base and J1.
        base = next((n for n in ex.ancestors(chain[0])
                     if ex.O[n]["type"] == "MESH" and not ex.is_dynamic(n)), None)
        if base:
            D.add("robot.base", base, "首关节最近的静止网格祖先；后续必须通过 FK 校验", fallback=True)
    if base not in ex.ancestors(chain[0]):
        raise ValueError('Set embodiment.base to a fixed ancestor of the first object joint')
    if not np.allclose(ex.M(base), ex.M(base)[0], atol=1e-7, rtol=0):
        raise ValueError('Moving robot bases are not supported')
    # Fixed spacers are allowed, independently animated spacers/constraints are not.
    hierarchy = set()
    for name in chain:
        for n in [name, *ex.ancestors(name)]:
            hierarchy.add(n)
            if n == base:
                break
    for n in hierarchy:
        if ex.O[n].get("constraints") or ex.O[n].get("parent_type", "OBJECT") != "OBJECT":
            raise ValueError(f'Unsupported constrained or bone-parented object joint: {n}')
        basis = ex.E.get("object_basis", {}).get(n)
        if n not in chain and basis is not None and not np.allclose(basis, basis[0], atol=1e-7, rtol=0):
            raise ValueError(f'Unmodeled animated spacer in joint chain: {n}')
    D.add("embodiment", "robot_arm", "父子对象关节链；保留 robot_arm 分支并进行同一套 FK 校验")
    return {**config, "type": "robot_arm", "representation": "object_joints", "chain": chain,
            "base": base, "root": base}


# ------------------------------------------------------------------ robot_arm
def robot_links(ex, emb):
    """连杆 i（1..n）= 父级为第 i 根链骨骼的第一个网格；link0 = 父级为骨架对象本身的网格（没有则用骨架）。"""
    if emb.get("representation") == "object_joints":
        return {0: emb["base"], **{i: n for i, n in enumerate(emb["chain"], 1)}}, emb.get("flange")
    an = emb["armature"]
    links = {}
    for i, b in enumerate(emb["chain"], 1):
        ms = [n for n, o in ex.O.items() if o["parent"] == an and o["parent_type"] == "BONE"
              and o["parent_bone"] == b and o["type"] == "MESH"]
        if ms:
            links[i] = max(ms, key=lambda n: ex.E["meshes"][n]["n_verts"])
    base = [n for n, o in ex.O.items() if o["parent"] == an and o["parent_type"] == "OBJECT" and o["type"] == "MESH"]
    links[0] = max(base, key=lambda n: ex.E["meshes"][n]["n_verts"]) if base else an
    flange = [n for n, o in ex.O.items() if o["parent"] == an and o["parent_bone"] == emb["chain"][-1]
              and o["type"] == "EMPTY"]
    return links, (flange[0] if flange else None)


_FK_CACHE = None


def _mj_fk(q):
    """(T,7) → {i: (T,4,4)} MuJoCo 连杆世界位姿（基座在原点）+ pinch site。"""
    import mujoco
    import droid_common as C
    global _FK_CACHE
    if _FK_CACHE is None:                       # 编译一次复用（束搜索要调几千次）
        _FK_CACHE = C.build_model()
    m, d, _ = _FK_CACHE
    qa = C.arm_qadr(m)
    out = {i: [] for i in range(0, 8)}
    out["pinch"] = []
    sid = m.site(f"{C.GRIP_PREFIX}pinch").id
    for qt in q:
        d.qpos[:] = 0
        d.qpos[qa] = qt
        mujoco.mj_kinematics(m, d)
        for i in range(0, 8):
            T = np.eye(4)
            b = m.body(f"link{i}").id
            T[:3, :3] = d.xmat[b].reshape(3, 3)
            T[:3, 3] = d.xpos[b]
            out[i].append(T)
        T = np.eye(4)
        T[:3, :3] = d.site_xmat[sid].reshape(3, 3)
        T[:3, 3] = d.site_xpos[sid]
        out["pinch"].append(T)
    return {k: np.array(v) for k, v in out.items()}


def _offset_spread(Tw_base, mj, bl):
    """X(t) = (Tbase·Tmj)^-1 · Tbl 的逐帧离散度：平移 mm、旋转 deg（相对中位帧）。"""
    X = np.array([inv(Tw_base @ a) @ b for a, b in zip(mj, bl)])
    ref = X[len(X) // 2]
    dt = np.linalg.norm(X[:, :3, 3] - ref[:3, 3], axis=1).max() * 1000
    dr = max(rot_angle_deg(ref[:3, :3].T @ x[:3, :3]) for x in X)
    return float(dt), float(dr), X


def extract_robot(ex, emb, D, robot="panda"):
    objects = emb.get("representation") == "object_joints"
    basis = ex.E.get("object_basis", {}) if objects else ex.E["armatures"][emb["armature"]]["basis"]
    n = ROBOT_DOF[robot]
    chain = emb["chain"]
    if len(chain) != n:
        D.add("robot.dof", len(chain), f"骨骼链长 {len(chain)} ≠ {robot} 的 {n} 个关节", warn=True)
        raise SystemExit("目标机型与 blend 机械臂自由度不符；在 b2s_config.json 里指定 embodiment.chain")
    Q, axes = [], []
    for b in chain:
        if b not in basis:
            raise ValueError(f'Missing local joint transforms for {b}; re-run extraction')
        matrices = np.asarray(basis[b], float)
        if objects:
            if matrices.shape != (ex.T, 4, 4) or not np.isfinite(matrices).all():
                raise ValueError(f'Invalid local joint transforms: {b}')
            if not np.allclose(matrices[:, :3, 3], matrices[0, :3, 3], atol=1e-7, rtol=0):
                raise ValueError(f'Prismatic/moving joint origins are not supported: {b}')
            if not np.allclose(matrices[:, :3, :3].transpose(0,2,1) @ matrices[:, :3, :3], np.eye(3), atol=2e-6):
                raise ValueError(f'Scaled/sheared object joint basis is not supported: {b}')
        ang, ax, off, rng = bone_joint_angles(matrices)
        if objects:
            ang = np.unwrap(ang)
            if off > 0.5:
                raise ValueError(f'Object joint is not single-axis: {b} ({off:.3f} deg RMS)')
        Q.append(ang)
        axes.append(ax)
        if off > 0.5:
            D.add(f"robot.joint[{b}]", "非单轴", f"离轴旋转 RMS {off:.2f}°，关节角可能不准", warn=True)
    q_raw = np.array(Q).T
    links, flange = robot_links(ex, emb)
    # 基座：link0 对象（若无则骨架对象）的世界矩阵，去掉缩放
    Mb = ex.M(links[0])[0].copy()
    Mb[:3, :3] = rot_of(Mb)
    links_bl = {i: ex.M(nm) for i, nm in links.items() if i > 0}
    for i in links_bl:
        links_bl[i][:, :3, :3] = np.array([rot_of(M) for M in links_bl[i]])

    def score(q):
        fk = _mj_fk(q)
        return {i: _offset_spread(Mb, fk[i], links_bl[i])[:2] for i in links_bl}, fk

    sgn, off = np.ones(n), np.zeros(n)
    rep, fk = score(q_raw)
    worst = max(max(v) for v in rep.values())
    D.add("robot.fk_check(raw)", {i: [round(a, 3), round(b, 3)] for i, (a, b) in rep.items()},
          "各连杆 X_i(t) 离散度 [mm, deg]；同一刚体应≈0")
    if worst > 2.0:
        # 逐关节拟合符号与零点。★ 可观测性：连杆 j+1 与关节 j 一起刚性转动，关节 j 的常数零点偏移
        # 只让 X_{j+1} 整体转一个固定角、X 仍恒定 —— **在连杆 j+1 上不可观测**，要到连杆 j+2 才显现。
        # 符号 s_j 在连杆 j+1 上可观测（取反会让 X 随 2q 转动）。所以第 j 步用连杆 j+1 同时解 (c_{j-1}, s_j)；
        # 最后一个关节的零点只靠连杆链不可观测，保持 0 并告警。（单元测试 test_joint_sign_offset_repair）
        from scipy.optimize import minimize_scalar
        sub = np.linspace(0, ex.T - 1, min(ex.T, 40)).astype(int)
        # ★ 等价解：如 (c_{j-1}=180°, s_j 取反) 在连杆 j+1 上离散度同样≈0，要到更远的连杆才分得开
        #   （单元测试实测：贪心在平局处选错，连杆 4 起平移离散 321 mm）。
        #   所以用束搜索：每步保留 BEAM 个候选，代价 = 到当前为止所有连杆离散度之和。
        BEAM = 8

        def cost(sg, of, upto):
            fkj = _mj_fk(q_raw[sub] * sg + of)
            return sum(sum(_offset_spread(Mb, fkj[i], links_bl[i][sub])[:2]) for i in range(1, upto + 1)
                       if i in links_bl)

        beam = [(0.0, np.ones(n), np.zeros(n))]
        for j in range(n):
            if j + 1 not in links_bl:
                continue
            grid = np.radians(np.arange(-180, 180, 15)) if j > 0 else [0.0]
            cand = []
            for _, sg0, of0 in beam:
                for s in (1.0, -1.0):
                    for c in grid:
                        sg, of = sg0.copy(), of0.copy()
                        sg[j] = s
                        if j > 0:
                            of[j - 1] = c
                        cand.append((cost(sg, of, j + 1), sg, of))
            cand.sort(key=lambda x: x[0])
            beam = []
            for e, sg, of in cand[:BEAM]:
                if j > 0:          # 细化 c_{j-1}
                    def f(c, sg=sg, of=of):
                        o = of.copy()
                        o[j - 1] = c
                        return cost(sg, o, j + 1)
                    r = minimize_scalar(f, bounds=(of[j - 1] - 0.3, of[j - 1] + 0.3), method="bounded")
                    of = of.copy()
                    of[j - 1] = r.x
                    e = r.fun
                beam.append((e, sg, of))
            beam.sort(key=lambda x: x[0])
        # ★ 腕部等价解：(q5+π, −q6, q7+π) 类组合让所有连杆的 X_i 同样恒定（单元测试实测选中过这种解，
        #   关节角越限 323°）。代价相近（+0.5 内）的候选里按 关节越限量 → |零点| 之和 → 取反个数 选最合理的。
        import droid_common as C
        mm, _, _ = _FK_CACHE or C.build_model()
        lim = np.array([mm.joint(f"joint{i}").range for i in range(1, 8)])

        def plaus(b):
            q = q_raw * b[1] + b[2]
            viol = np.maximum(lim[:, 0] - q.min(0), 0).sum() + np.maximum(q.max(0) - lim[:, 1], 0).sum()
            return (round(float(viol), 3), float(np.abs(np.angle(np.exp(1j * b[2]))).sum()), int((b[1] < 0).sum()))
        near = [b for b in beam if b[0] <= beam[0][0] + 0.5]
        _, sgn, off = min(near, key=plaus)
        if len(near) > 1:
            D.add("robot.joint_map.ambiguity", len(near), f"{len(near)} 个运动学等价解，按关节限位合理性选取", warn=True)
        D.add("robot.joint_map.last_offset", 0.0, "末关节零点只靠连杆链不可观测，保持 0（需末端工具几何才能定）",
              warn=True)
        rep, fk = score(q_raw * sgn + off)
        worst = max(max(v) for v in rep.values())
        D.add("robot.joint_map", {"sign": sgn.tolist(), "offset_deg": np.degrees(off).round(3).tolist()},
              f"原始关节角与 MuJoCo 不一致，逐关节拟合后离散度最大 {worst:.3f}", fallback=True)
    q = q_raw * sgn + off
    ok = worst <= 2.0
    D.add("robot.fk_check", "通过" if ok else "失败",
          f"最大离散度 {worst:.4f}（阈值 2 mm / 2°）", warn=not ok)
    if objects and not ok:
        raise ValueError('Object-joint FK validation failed; refusing to bind this robot')
    # 关节限位检查
    import droid_common as C
    m, _, _ = C.build_model()
    lim = np.array([m.joint(f"joint{i}").range for i in range(1, 8)])
    viol = np.maximum(lim[:, 0] - q.min(0), q.max(0) - lim[:, 1])
    if (viol > 1e-3).any():
        D.add("robot.joint_limits", np.degrees(viol).round(2).tolist(), "关节角越过 MuJoCo 限位（deg）", warn=True)
    return {"robot": robot, "q": q, "q_raw": q_raw, "sign": sgn, "offset": off, "T_world_base": Mb, "axes": axes,
            "links": links, "flange": flange, "fk_spread": rep, "fk_ok": ok, "fk": fk}
