#!/usr/bin/env python3
"""从 EgoDex 官方 test.zip 按条目抓取单条 episode（HTTP Range，不下载 17 GB 整包）。

为什么不用 unzip / gdown / 整包下载：
  - 官方分发就是一个 17.3 GB 的 test.zip，3243 条 episode 全在里面
  - S3 支持 Range，中央目录已经缓存成 entries JSON（tools/tmp_ego_dl/zip_entries.json），
    每条 episode 的本地头偏移 (lho) 已知 → 直接取该条目的字节区间即可
  - 只需要 hdf5（几百 KB）+ mp4（几 MB）

下载完逐条校验：解压长度 == 中央目录里的 usize，且 CRC32 == 中央目录里的 crc。
任何不一致直接报错，不写文件（不静默降级）。

用法:
  python tools/egodex/fetch.py --episode 1 --out GLB/test3/egodex_basic_pick_place_1
  python tools/egodex/fetch.py --refresh-index          # 重新拉中央目录（torrent 更新后用）
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request
import zlib

URL = "https://ml-site.cdn-apple.com/datasets/egodex/test.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "test_zip_entries.json")


# ---------------------------------------------------------------- 中央目录
def _range(a, b, retries=3):
    """闭区间 Range 读取。S3 大偏移需要重试（偶发 5xx）。"""
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=180) as f:
                data = f.read()
            if len(data) != b - a + 1:
                raise IOError(f"short read {len(data)} != {b - a + 1}")
            return data
        except Exception as ex:      # noqa: BLE001 —— 网络抖动都要重试
            last = ex
            time.sleep(1.5 * (k + 1))
    raise SystemExit(f"Range {a}-{b} 失败: {last}")


def _remote_size():
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as f:
        return int(f.headers["Content-Length"])


def parse_central_directory():
    """返回 [{name, method, csize, usize, crc, lho}]。支持 ZIP64。"""
    sz = _remote_size()
    tail = _range(max(0, sz - (1 << 20)), sz - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise SystemExit("尾部找不到 EOCD")
    n = struct.unpack("<H", tail[i + 10:i + 12])[0]
    cdsize = struct.unpack("<I", tail[i + 12:i + 16])[0]
    cdoff = struct.unpack("<I", tail[i + 16:i + 20])[0]
    loc = tail.rfind(b"PK\x06\x07")
    if loc >= 0:
        z64 = struct.unpack("<Q", tail[loc + 8:loc + 16])[0]
        z = _range(z64, z64 + 55)
        assert z[:4] == b"PK\x06\x06", z[:4]
        n = struct.unpack("<Q", z[32:40])[0]
        cdsize = struct.unpack("<Q", z[40:48])[0]
        cdoff = struct.unpack("<Q", z[48:56])[0]
    cd = _range(cdoff, cdoff + cdsize - 1)
    out, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (_, _, flag, method, _, _, crc, csize, usize, nlen, elen, clen,
         _, _, _, lho) = struct.unpack("<HHHHHHIIIHHHHHII", cd[p + 4:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        # ZIP64 扩展字段：把 0xFFFFFFFF 的占位值换成真值（大文件才会出现）
        if 0xFFFFFFFF in (usize, csize, lho):
            q = 0
            while q + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    v = extra[q + 4:q + 4 + hsz]
                    off = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", v[off:off + 8])[0]; off += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", v[off:off + 8])[0]; off += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", v[off:off + 8])[0]; off += 8
                q += 4 + hsz
        out.append({"name": name, "method": method, "csize": csize, "usize": usize,
                    "crc": crc, "lho": lho})
        p += 46 + nlen + elen + clen
    if len(out) != n:
        print(f"⚠️  中央目录条目数 {len(out)} != EOCD 声明 {n}", file=sys.stderr)
    return out


def load_index(refresh=False):
    if not refresh and os.path.exists(INDEX):
        return json.load(open(INDEX))
    print("拉取中央目录（Range，约 1 MB）…", file=sys.stderr)
    e = parse_central_directory()
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    json.dump(e, open(INDEX, "w"))
    return e


# ---------------------------------------------------------------- 单条目
def fetch_entry(entry, out_path):
    """取一条 zip 条目，解压到 out_path，校验长度和 CRC32。"""
    lho = entry["lho"]
    hdr = _range(lho, lho + 29)
    if hdr[:4] != b"PK\x03\x04":
        raise SystemExit(f"{entry['name']}: 本地头签名不对 {hdr[:4]}")
    nlen, elen = struct.unpack("<HH", hdr[26:30])
    start = lho + 30 + nlen + elen
    raw = _range(start, start + entry["csize"] - 1)
    if entry["method"] == 8:
        data = zlib.decompress(raw, -15)
    elif entry["method"] == 0:
        data = raw
    else:
        raise SystemExit(f"{entry['name']}: 未支持的压缩方法 {entry['method']}")
    if len(data) != entry["usize"]:
        raise SystemExit(f"{entry['name']}: 长度 {len(data)} != {entry['usize']}")
    crc = zlib.crc32(data) & 0xFFFFFFFF
    if crc != entry["crc"]:
        raise SystemExit(f"{entry['name']}: CRC32 不匹配 {crc:08x} != {entry['crc']:08x}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)
    return {"path": out_path, "bytes": len(data), "crc32": f"{crc:08x}",
            "sha256": hashlib.sha256(data).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="basic_pick_place")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--out", required=True, help="产物目录")
    ap.add_argument("--no-video", dest="video", action="store_false",
                    help="只下 hdf5，不下 mp4（默认两者都下）")
    ap.set_defaults(video=True)
    ap.add_argument("--refresh-index", action="store_true")
    a = ap.parse_args()

    idx = {x["name"]: x for x in load_index(a.refresh_index)}
    want = [(f"test/{a.task}/{a.episode}.hdf5", f"{a.task}_{a.episode}.hdf5"),
            (f"test/{a.task}/{a.episode}.mp4", f"{a.task}_{a.episode}.mp4")]
    if not a.video:
        want = want[:1]

    got = []
    for name, fn in want:
        e = idx.get(name)
        if e is None:
            avail = sorted(int(k.rsplit("/", 1)[1][:-6]) for k in idx
                           if k.startswith(f"test/{a.task}/") and k.endswith(".hdf5"))
            raise SystemExit(f"zip 里没有 {name}；该任务已有 episode: {avail[:20]}{' …' if len(avail) > 20 else ''}")
        print(f"下载 {name}  ({e['usize'] / 1e6:.2f} MB 解压后) …", file=sys.stderr)
        got.append({"zip_entry": name, **fetch_entry(e, os.path.join(a.out, fn))})
        print(f"  ✓ {got[-1]['path']}  sha256={got[-1]['sha256'][:16]}…", file=sys.stderr)

    meta = {
        "source": {"dataset": "EgoDex", "url": URL, "task": a.task, "episode": a.episode,
                   "zip_entry": [g["zip_entry"] for g in got],
                   "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "method": "HTTP Range over zip central directory (未下载整包)",
                   "zip_size_bytes": _remote_size()},
        "files": got,
    }
    p = os.path.join(a.out, "source.json")
    json.dump(meta, open(p, "w"), indent=1, ensure_ascii=False)
    print(f"溯源写入 {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
