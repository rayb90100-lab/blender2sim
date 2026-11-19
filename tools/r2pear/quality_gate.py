#!/usr/bin/env python3
"""Stage 1 质量门：拟合失败报警 + 模型替换建议（通用，只读产物，不重跑拟合）。

为什么必须有这一层（test7 ep54 实测教训）：
  1) `droid_stage1_masks.py` 的候选过滤只查 15 <= area <= MAX_AREA，**没有分数下限**。
     黄块在黄木桌上，提示点偏 3 px 就分割到桌面，score 0.032、area 511 —— 仍在区间内被
     选中并写进 masks.npz。12/12 全部 "status: ok"，乍看全过，实际是静默失败。
  2) `droid_stage1_scene.py` 对未知 shape（test7 的 "cylinder_container"）静默回退成箱体，
     且 degraded 分支只覆盖 cylinder/prism，未知值连 note 都不写。
  3) 拟合尺寸可能错 2-3 倍（ep54 的 25 mm 方块被拟合成 69x76x26 mm）而没有任何报警。

★ 关于 IoU 的来源（踩过的坑，别重犯）：
  `scene.json` 顶层的 `objects[n]["iou"]` 是**单相机拟合时的自评**，ext1 坏掉时那里会出现
  一堆 0.0（test6 就是：ext1 0.683 / ext2 0.0 / wrist 0.0）。
  test6 实际采用的是 `scene_ext2_wrist.json`，真实 IoU 在 `fit.iou` 里，键为 `<obj>|<cam>`，
  数值 0.375-0.79。本脚本因此按以下优先级取 IoU：
     1) <scene>.fit.iou（键 <obj>|<cam>）—— 多视角联合拟合的自评，最可靠
     2) <scene>.objects[n].iou（键 <cam>）
  `--scene` 可显式指定实际采用的文件；缺省按 merged/ → mine/ → scene.json 搜索。

用法:
  python tools/r2pear/quality_gate.py --testdir <T> [--scene stage1_scene/xxx.json] [--strict]

退出码: 0 = pass；1 = reject（或 --strict 下 review）
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

# ---------------------------------------------------------------- 阈值
# 依据（§63.4 / test6 实测）：好解单视角 IoU 0.375-0.79、中位 0.65；坏解（test7）0.0-0.41。
THRESH = {
    "mask_score_min": 0.50,          # SAM2 置信度下限；test7 坏 mask 是 0.032/0.037
    "mask_centroid_px_max": 12.0,    # 提示点 → mask 质心；test6 好解 3.0-7.5 px
    "fit_iou_min": 0.30,             # 单视角硬下限；test6 最低 0.375
    "fit_iou_median_min": 0.45,      # 全视角中位；test6 中位 0.65
    "reproj_centroid_px_max": 12.0,  # 3D 中心投回图像的质心误差（§63.4 主指标）
    "size_ratio_min": 0.40,          # 拟合/先验（最大边长比）
    "size_ratio_max": 2.50,          # test7 实测 2.5-3.5× 明确是坏解
}

SUPPORTED_SHAPES = {"box", "arch", "cylinder", "prism"}
DEGRADED_SHAPES = {"cylinder", "prism"}          # sim.py 里按箱体近似
CONTAINER_HINTS = ("cup", "bowl", "pot", "box_", "bin", "basket",
                   "container", "mug", "bottle", "flask", "basin", "tray", "sink")


def find_scene(testdir, explicit=None):
    od = os.path.join(testdir, "stage1_scene")
    if explicit:
        p = explicit if os.path.isabs(explicit) else os.path.join(testdir, explicit)
        return p if os.path.exists(p) else None
    for pat in ("merged/scene*.json", "mine/scene.json", "scene_ext2_wrist.json", "scene.json"):
        hits = sorted(glob.glob(os.path.join(od, pat)))
        if hits:
            return hits[-1]
    return None


def iou_map(scene):
    """返回 {(obj, cam): iou}，按上文的优先级取。同时返回来源说明。"""
    out, src = {}, "none"
    fit = (scene.get("fit") or {}).get("iou") or {}
    if fit:
        for k, v in fit.items():
            if "|" in k and v is not None:
                o, c = k.split("|", 1)
                out[(o, c)] = float(v)
        if out:
            src = "fit.iou"
    if not out:
        for n, ob in (scene.get("objects") or {}).items():
            for c, v in (ob.get("iou") or {}).items():
                if v is not None:
                    out[(n, c)] = float(v)
        if out:
            src = "objects[].iou"
    return out, src


def is_container_name(name, shape):
    if shape not in SUPPORTED_SHAPES:
        return True
    low = name.lower()
    return any(h in low for h in CONTAINER_HINTS)


def mask_checks(masks, info, th):
    res = {}
    for key in sorted(masks.files):
        mk = masks[key].astype(bool)
        obj, cam = key.split("|", 1)
        rec, reasons = {"cam": cam}, []
        ys, xs = np.nonzero(mk)
        if len(xs) == 0:
            res.setdefault(obj, {})[cam] = {"cam": cam, "grade": "fail",
                                            "reasons": ["mask 为空"]}
            continue
        cen = (float(xs.mean()), float(ys.mean()))
        rec.update(area=int(mk.sum()), centroid=[round(cen[0], 1), round(cen[1], 1)])
        meta = (info or {}).get(key, {})
        rec["score"] = meta.get("score")
        s = meta.get("score")
        if s is not None and s < th["mask_score_min"]:
            reasons.append(f"mask 分数 {s:.3f} < {th['mask_score_min']}"
                           f"（疑分割到背景/桌面，需重取提示点）")
            rec["fatal"] = True
        p = meta.get("prompt")
        if p:
            d = float(np.hypot(cen[0] - p[0], cen[1] - p[1]))
            rec["prompt_offset_px"] = round(d, 1)
            if d > th["mask_centroid_px_max"]:
                reasons.append(f"提示点距 mask 质心 {d:.1f} px > {th['mask_centroid_px_max']}"
                               f"（物体与背景对比低，提示点要精确到 2-3 px）")
        if meta.get("status") not in (None, "ok"):
            reasons.append(f"mask 状态 {meta.get('status')}")
            rec["fatal"] = True
        rec["grade"] = "fail" if rec.get("fatal") else ("warn" if reasons else "ok")
        rec["reasons"] = reasons
        res.setdefault(obj, {})[cam] = rec
    return res


def reproj_checks(testdir, scene, masks, th):
    """C: 质心重投影误差 —— **主指标**（§63.4：好解 1.2-7.2 px，坏解 13-23 px）。

    为什么必须自己算、不能信 scene.json 里的 iou：
      ep88 实测——6 个物体里 4 个质心误差 1.4-8.8 px（拟合正确），
      另 2 个（同色小黄块）误差 19-55 px，被优化器牺牲了；
      而它们的自报 IoU 是**精确的 0.0**，肉眼只看 IoU 会误判成"分割坏了"，
      实际是**拟合把它们推到了别处**。质心误差能直接指出是哪一个、偏了多少。

    需要 droid_cams（相机参数）；缺失时该项跳过并记 warn。

    ★ 只检查**参与拟合的相机**（scene.fit.views）。原因：test6 的 ext1 外参已被判定
      不可用（§63.1），实际采用的是 ext2+wrist 的解；若把 ext1 也算进来，
      同一套好解会被判 3 个物体 "被拟合到错误位置"（16-18 px）——这是误报。
    """
    import droid_cams as DC
    views = ((scene.get("fit") or {}).get("views")
             or ["exterior_1_left", "exterior_2_left", "wrist_left"])
    try:
        cams = DC.Cams(testdir)
    except Exception as e:
        return {n: {"grade": "warn", "reasons": [f"无法构建相机模型，跳过：{type(e).__name__}"]}
                for n in (scene.get("objects") or {})}
    out = {}
    for name, ob in (scene.get("objects") or {}).items():
        c = ob.get("center")
        if not c:
            out[name] = {"grade": "warn", "reasons": ["scene 缺 center"]}
            continue
        errs, reasons = {}, []
        for cam in views:
            key = f"{name}|{cam}"
            if key not in masks.files:
                continue
            mk = masks[key].astype(bool)
            ys, xs = np.nonzero(mk)
            if len(xs) == 0:
                continue
            cen = np.array([xs.mean(), ys.mean()])
            uv = np.asarray(cams.project(cam, np.asarray(c, float)[None, :])[0])
            if np.isnan(uv).any():
                continue
            e = float(np.linalg.norm(uv - cen))
            errs[cam] = round(e, 1)
            if e > th["reproj_centroid_px_max"]:
                reasons.append(f"{cam} 质心重投影 {e:.1f} px > {th['reproj_centroid_px_max']}"
                               f"（3D 中心投到 ({uv[0]:.0f},{uv[1]:.0f})，mask 质心在 ({cen[0]:.0f},{cen[1]:.0f})"
                               f" → 该物体被拟合到了错误位置）")
        med = float(np.median(list(errs.values()))) if errs else None
        out[name] = {"grade": "fail" if reasons else "ok",
                     "views_checked": list(views),
                     "centroid_err_px": errs,
                     "median_px": None if med is None else round(med, 1),
                     "reasons": reasons}
    return out


def fit_checks(scene, th, size_prior, imap, isrc):
    res = {}
    for name, ob in (scene.get("objects") or {}).items():
        shape = ob.get("shape", "box")
        reasons, warns = [], []
        need, suggest = False, False

        # 形状可表达性 —— 「模型替换」判断的主入口
        if shape not in SUPPORTED_SHAPES:
            reasons.append(f"shape='{shape}' 不在支持集 {sorted(SUPPORTED_SHAPES)} 内，管线静默回退成"
                           f"箱体（无法表达中空/开口/曲面）→ 必须替换模型资产")
            need = True
        elif shape in DEGRADED_SHAPES:
            warns.append(f"shape='{shape}' 按箱体近似（外观与碰撞都不精确）→ 建议替换资产")
            suggest = True
        if is_container_name(name, shape) and shape in SUPPORTED_SHAPES:
            warns.append(f"名称/任务暗示这是容器，但管线只有实心 {shape}；腔内无法碰撞 → 建议替换资产")
            suggest = True

        # IoU（按 <obj>|<cam>）
        per = {c: v for (o, c), v in imap.items() if o == name}
        vals = list(per.values())
        med = float(np.median(vals)) if vals else 0.0
        if vals:
            bad = {c: v for c, v in per.items() if v < th["fit_iou_min"]}
            if bad:
                reasons.append(f"拟合 IoU 过低 {bad} < {th['fit_iou_min']}")
            if med < th["fit_iou_median_min"]:
                reasons.append(f"IoU 中位 {med:.3f} < {th['fit_iou_median_min']}")

        # 尺寸自洽（最大边长比，避免 cube 与 box 语义混淆）
        size = ob.get("size") or []
        ratio = None
        prior = (size_prior or {}).get(name)
        if prior and len(size) >= 1:
            h_fit, h_pri = max(float(x) for x in size), max(float(x) for x in prior)
            if h_pri > 0:
                ratio = h_fit / h_pri
                if not (th["size_ratio_min"] <= ratio <= th["size_ratio_max"]):
                    reasons.append(f"最大边长 {h_fit*1000:.1f} / 先验 {h_pri*1000:.1f} mm = {ratio:.2f}×，"
                                   f"超出 [{th['size_ratio_min']}, {th['size_ratio_max']}]"
                                   f"（多视角约束不足或先验错）")

        grade = "fail" if reasons else ("warn" if warns else "ok")
        r = {"shape": shape, "iou": per, "iou_median": round(med, 3),
             "size_m": size, "size_ratio_vs_prior": None if ratio is None else round(ratio, 2),
             "grade": grade, "reasons": reasons + warns}
        if need:
            r["need_asset_replacement"] = True
        if suggest:
            r["suggest_asset_replacement"] = True
        res[name] = r
    return res, isrc


def cross_view_checks(scene, imap):
    """跨视角一致性——**只作参考，不参与 overall 判定**。

    实测（2026-11-18）：这个指标区分不了好解与坏解。
      test6（已验收）：green_arch 极差 0.371、red_prism 0.202、green_cylinder 0.072
      test7（坏解）  ：yellow_block 0.117、green_block 0.172
    已验收场景的极差反而更大（被遮挡的 arch 只在腕部视角有高 IoU），
    因为极差主要反映「物体在各视角下的可见面积差异」，不是拟合质量。
    所以这里只输出数值供人看，不设 fail 判定 —— 早期版本据此误判 green_arch 为 fail。
    """
    out = {}
    for n in {o for o, _ in imap}:
        vals = [v for (o, c), v in imap.items() if o == n]
        if len(vals) < 2:
            out[n] = {"grade": "ok", "note": "可用视角 < 2，跳过"}
            continue
        out[n] = {"iou_spread": round(float(max(vals) - min(vals)), 3),
                  "grade": "ok", "note": "仅参考，不参与判定"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--scene", default=None, help="实际采用的 scene json（相对 testdir 或绝对）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true", help="review 也按失败处理")
    a = ap.parse_args()

    th = dict(THRESH)
    T = a.testdir
    od = os.path.join(T, "stage1_scene")
    mp = os.path.join(od, "masks.npz")
    sp = find_scene(T, a.scene)
    if not os.path.exists(mp) or not sp:
        print(f"缺少产物：masks={os.path.exists(mp)} scene={sp}", file=sys.stderr)
        return 1
    masks = np.load(mp)
    ip = os.path.join(od, "masks.json")
    info = json.load(open(ip)) if os.path.exists(ip) else {}
    op = os.path.join(od, "objects.json")
    objs = json.load(open(op)) if os.path.exists(op) else {}
    scene = json.load(open(sp))
    size_prior = objs.get("size_prior_m") or objs.get("size_prior") or {}

    imap, isrc = iou_map(scene)
    mres = mask_checks(masks, info, th)
    fres, isrc = fit_checks(scene, th, size_prior, imap, isrc)
    cres = cross_view_checks(scene, imap)
    rres = reproj_checks(T, scene, masks, th)

    # 三组都要计入：早期版本漏了 cross_view，导致打印 [xview fail] 而 overall 仍是 REVIEW
    grades = [r["grade"] for o in mres.values() for r in o.values()] + \
             [r["grade"] for r in fres.values()] + \
             [r["grade"] for r in cres.values() if r["grade"] == "fail"] + \
             [r["grade"] for r in rres.values()]
    n_fail, n_warn = grades.count("fail"), grades.count("warn")
    overall = "reject" if n_fail else ("review" if n_warn else "pass")

    report = {
        "testdir": os.path.abspath(T), "scene_file": os.path.relpath(sp, T),
        "iou_source": isrc, "thresholds": th, "overall": overall,
        "counts": {"fail": n_fail, "warn": n_warn, "ok": grades.count("ok")},
        "need_asset_replacement": sorted(n for n, r in fres.items() if r.get("need_asset_replacement")),
        "suggest_asset_replacement": sorted(n for n, r in fres.items() if r.get("suggest_asset_replacement")),
        "masks": mres, "objects": fres, "reproj": rres, "cross_view": cres,
    }
    out = a.out or os.path.join(od, "quality_gate.json")
    json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)

    print(f"=== 质量门 {os.path.basename(T.rstrip('/'))} → {overall.upper()} ===")
    print(f"scene={report['scene_file']}  IoU来源={isrc}")
    print(f"fail {n_fail} / warn {n_warn} / ok {grades.count('ok')}\n")
    for obj, cams in mres.items():
        for cam, r in cams.items():
            if r["grade"] != "ok":
                print(f"[mask {r['grade']:4s}] {obj}|{cam}")
                for x in r["reasons"]:
                    print(f"        - {x}")
    for obj, r in fres.items():
        if r["grade"] != "ok":
            print(f"[fit  {r['grade']:4s}] {obj}  (shape={r['shape']}, IoU中位 {r['iou_median']})")
            for x in r["reasons"]:
                print(f"        - {x}")
    for obj, r in rres.items():
        if r["grade"] != "ok":
            print(f"[reproj {r['grade']:4s}] {obj}  (中位 {r.get('median_px')} px)")
            for x in r["reasons"]:
                print(f"        - {x}")
    for obj, r in cres.items():
        if r["grade"] != "ok":
            print(f"[xview {r['grade']:4s}] {obj}: " + "; ".join(r["reasons"]))
    if report["need_asset_replacement"]:
        print(f"\n★ 需替换模型资产：{report['need_asset_replacement']}")
    if report["suggest_asset_replacement"]:
        print(f"☆ 建议替换资产：{report['suggest_asset_replacement']}")
    print(f"\n写出 {out}")
    if overall == "reject" or (overall == "review" and a.strict):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
