"""task_spec.json 的读取与基本校验。"""
import json

PREDICATES = {"on": ("obj", "support"), "released": ("obj",), "upright": ("obj",)}


SHAPES = ("box", "arch", "cylinder", "prism")


def height(o):
    """物体高度（m）。cylinder 的 size = [r, h]，其余 = [sx, sy, h]。"""
    return float(o["size"][-1])


def footprint_half(o):
    """底面外接半宽（m），用于桌内判定。"""
    if o["shape"] == "cylinder":
        return float(o["size"][0])
    return float(max(o["size"][:2]) / 2)


def grip_half(o):
    """可夹持方向的半宽（m）：cylinder 为半径，箱类为较短边之半。"""
    if o["shape"] == "cylinder":
        return float(o["size"][0])
    return float(min(o["size"][:2]) / 2)


def load(path):
    spec = json.load(open(path))
    assert spec.get("schema_version") == 1, "unknown schema_version"
    names = set(spec["objects"])
    for p in spec["success"]:
        assert p["pred"] in PREDICATES, p
        for k in PREDICATES[p["pred"]]:
            if k in ("obj", "support"):
                ok = p[k] in names or (k == "support" and p[k] == "table")   # b2s：放到桌面上
                assert ok, f"predicate references unknown object {p[k]}"
    for o in spec["objects"].values():
        assert o["shape"] in SHAPES, o["shape"]
    for c in spec["cameras"].values():
        assert c["mount"] in ("world", "link7"), c
    return spec
