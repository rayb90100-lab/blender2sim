"""从 UMI replay_buffer.zarr.zip 里抽一条单臂 episode，落成标准 zarr v2 store + 规范末端 npz。

用法：
    PY=/home/ubuntu/mambaforge/envs/real2gym/bin/python
    $PY tools/umi_extract/extract_umi_episode.py --episode 68 --out GLB/test2

为什么直接读 zip 而不是 import zarr：
  本机 real2gym 环境的 zarr 2.18.2 与 numcodecs 0.16.6 不匹配（import 即 ImportError:
  cannot import name 'cbuffer_sizes'），且 numpy 锁在 1.26.4 不能装 imagecodecs。
  这里只用 zipfile + numpy 读，写成未压缩的 zarr v2 store（compressor: null），
  任何装了 zarr 的环境都能直接 zarr.open() 读回。

相机通道（data/camera0_rgb）是 JPEG-XL（imagecodecs_jpegxl, level 99），本机无解码器，
见 --with-camera 的说明。
"""
import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np

SRC_DEFAULT = "/media/data1/RoboLab_data/downloads/cup_in_the_lab.zarr.zip"
SRC_URL = "https://real.stanford.edu/umi/data/zarr_datasets/cup_in_the_wild.zarr.zip"
UMI_MAX_APERTURE = 0.08
UMI_TCP_TO_PINCH = np.eye(3)  # 与 tools/r2pear/b2s/sources.py 保持一致


def zarr_array_meta(shape, chunks, dtype, compressor=None):
    return {
        "chunks": list(chunks),
        "compressor": compressor,
        "dtype": dtype,
        "fill_value": None,
        "filters": None,
        "order": "C",
        "shape": list(shape),
        "zarr_format": 2,
    }


def zarr_chunk_name(idx):
    """zarr v2 的 chunk key：各维索引用 '.' 连接。"""
    return ".".join(str(i) for i in idx)


class ZipZarr:
    """只读的 zarr v2 store（后端是一个 zip 文件）。"""

    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
        self.names = set(self.z.namelist())

    def meta(self, key):
        raw = self.z.read(key.rstrip("/") + "/.zarray")
        return json.loads(raw.decode())

    def read_array(self, key):
        m = self.meta(key)
        shape = tuple(m["shape"])
        chunks = tuple(m["chunks"])
        cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        out = np.full((shape[0], cols), np.nan, dtype=m["dtype"])
        prefix = key.rstrip("/") + "/"
        for n in self.names:
            if not n.startswith(prefix) or n.endswith((".zarray", ".zattrs")):
                continue
            idx = int(n[len(prefix):].split(".")[0])
            v = np.frombuffer(self.z.read(n), dtype=m["dtype"]).reshape(-1, cols)
            lo = idx * chunks[0]
            hi = min(lo + chunks[0], shape[0])
            out[lo:hi] = v[: hi - lo]
        if len(shape) == 1:
            return out[:, 0]
        return out.reshape(shape)


class DirZarr:
    """写未压缩的 zarr v2 store。"""

    def __init__(self, root):
        self.root = root

    def _p(self, key):
        return os.path.join(self.root, key.strip("/"))

    def create(self, key, data, chunks=None):
        data = np.ascontiguousarray(data)
        chunks = chunks or (data.shape[0],) + tuple(data.shape[1:])
        p = self._p(key)
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, ".zarray"), "w") as f:
            json.dump(zarr_array_meta(data.shape, chunks, data.dtype.str), f, indent=4)
        with open(os.path.join(p, ".zattrs"), "w") as f:
            json.dump({}, f)
        # 只在第 0 维切块：chunk key 的其余维索引全填 0
        step = chunks[0]
        flat = data.reshape(data.shape[0], -1)
        n_rest_dims = max(len(data.shape) - 1, 0)
        for i, lo in enumerate(range(0, data.shape[0], step)):
            hi = min(lo + step, data.shape[0])
            idx = (i,) + (0,) * n_rest_dims
            with open(os.path.join(p, zarr_chunk_name(idx)), "wb") as f:
                f.write(np.ascontiguousarray(flat[lo:hi]).tobytes())

    def marker(self, key="", attrs=None):
        os.makedirs(self._p(key), exist_ok=True)
        with open(os.path.join(self._p(key), ".zgroup"), "w") as f:
            json.dump({"zarr_format": 2}, f)
        if attrs is not None:
            with open(os.path.join(self._p(key), ".zattrs"), "w") as f:
                json.dump(attrs, f, indent=2, ensure_ascii=False)


def umi_to_canonical(eef_pos, eef_rotvec, gripper_width, max_aperture=UMI_MAX_APERTURE):
    """与 b2s/sources.py:umi_to_canonical 逐行等价（不在本环境 import 该模块，避免额外依赖）。"""
    from scipy.spatial.transform import Rotation as Rot
    T = np.repeat(np.eye(4)[None], len(eef_pos), 0)
    T[:, :3, :3] = Rot.from_rotvec(eef_rotvec).as_matrix() @ UMI_TCP_TO_PINCH
    T[:, :3, 3] = eef_pos
    g = 1 - np.clip(np.asarray(gripper_width).ravel() / max_aperture, 0, 1)
    return T, g


def write_mp4(frames, path, fps=30):
    """frames (T,H,W,3) uint8 → H.264 mp4，供人直接看。

    UMI 的手腕相机是 30 fps 录制（zarr 里没有帧率字段，按 GoPro 常见 30fps 标注，
    见 source.json 的 fps_note）；写视频用 matplotlib 自带的 ffmpeg 不保证存在，
    这里直接调系统/conda 的 ffmpeg。
    """
    import subprocess
    import tempfile
    t, h, w, _ = frames.shape
    ff = None
    for cand in (os.environ.get("FFMPEG"), "/home/ubuntu/mambaforge/envs/E2E/bin/ffmpeg", "ffmpeg"):
        if not cand:
            continue
        if cand == "ffmpeg":
            from shutil import which
            if which("ffmpeg"):
                ff = which("ffmpeg")
                break
        elif os.path.exists(cand):
            ff = cand
            break
    if ff is None:
        print("  [warn] 找不到 ffmpeg，跳过 mp4 导出")
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=os.path.dirname(path) or ".") as f:
        tmp = f.name
    cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", tmp]
    p = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        print(f"  [warn] ffmpeg 失败: {p.stderr.decode()[:300]}")
        os.unlink(tmp)
        return None
    os.replace(tmp, path)
    return path


def decode_jxl(blob):
    """JPEG-XL → uint8 RGB。依赖 pillow-jxl-plugin（模块名 pillow_jxl，import 即注册到 PIL）。"""
    try:
        import pillow_jxl  # noqa: F401  仅为了向 Pillow 注册 JXL 解码器
    except ImportError as e:
        raise ImportError("缺 pillow-jxl-plugin：pip install pillow-jxl-plugin") from e
    from PIL import Image
    im = Image.open(io.BytesIO(blob))
    return np.asarray(im.convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--episode", type=int, default=68)
    ap.add_argument("--out", default="GLB/test2")
    ap.add_argument("--with-camera", action="store_true",
                    help="同时抽 camera0_rgb（需要 JPEG-XL 解码器，本机默认没有）")
    args = ap.parse_args()

    src = ZipZarr(args.src)
    ends = np.frombuffer(src.z.read("meta/episode_ends/0"), dtype="<i8").copy()
    ends = ends[ends > 0]
    ep = args.episode
    if not 0 <= ep < len(ends):
        sys.exit(f"episode {ep} 超出范围 0..{len(ends)-1}")
    s = 0 if ep == 0 else int(ends[ep - 1])
    e = int(ends[ep])
    n = e - s
    print(f"源 {args.src}")
    print(f"episode {ep}: 帧区间 [{s}, {e})，共 {n} 帧")

    pos = src.read_array("data/robot0_eef_pos")[s:e]
    rot = src.read_array("data/robot0_eef_rot_axis_angle")[s:e]
    gfw = src.read_array("data/robot0_gripper_width")[s:e]
    print(f"robot0_eef_pos {pos.shape}  rotvec {rot.shape}  gripper {gfw.shape}")
    if np.isnan(pos).any() or np.isnan(rot).any() or np.isnan(gfw).any():
        sys.exit("源数据里有 NaN，抽帧区间不完整")

    os.makedirs(args.out, exist_ok=True)

    # ---- 1. 标准 zarr v2 store（单臂，只含 robot0_*）----
    zroot = os.path.join(args.out, f"umi_ep{ep}.zarr")
    dst = DirZarr(zroot)
    os.makedirs(zroot, exist_ok=True)
    dst.marker("")
    dst.marker("meta")
    dst.marker("data")
    dst.create("meta/episode_ends", np.array([n], dtype="<i8"))
    dst.create("data/robot0_eef_pos", pos.astype("<f4"), chunks=(min(631, n), 3))
    dst.create("data/robot0_eef_rot_axis_angle", rot.astype("<f4"), chunks=(min(631, n), 3))
    dst.create("data/robot0_gripper_width", gfw.astype("<f4").reshape(n, 1), chunks=(min(631, n), 1))
    print(f"写入 {zroot}")

    # ---- 2. 规范末端系 npz（和 GLB/test1/sim_synth_umi/umi_like_demo.npz 同结构）----
    T, g = umi_to_canonical(pos, rot, gfw)
    npz = os.path.join(args.out, f"umi_ep{ep}_canonical.npz")
    np.savez(npz, ee_world=T.astype(np.float64), g=g.astype(np.float64))
    print(f"写入 {npz}  ee_world {T.shape}  g {g.shape}")

    # ---- 3. 相机（可选）----
    cam_note = None
    cam_shape = None
    if args.with_camera:
        prefix = "data/camera0_rgb/"
        frames = []
        for i in range(s, e):
            key = f"{prefix}{i}.0.0.0"
            if key not in src.names:
                sys.exit(f"缺相机 chunk: {key}")
            frames.append(decode_jxl(src.z.read(key)))
        frames = np.stack(frames)
        cam_shape = frames.shape
        dst.create("data/camera0_rgb", frames, chunks=(1,) + frames.shape[1:])
        print(f"写入相机 {frames.shape}")
        cam_note = f"camera0_rgb 已抽取 {frames.shape}"

        # 同时导一份 H.264 mp4（与 test1 的 *_left.mp4 同用途，便于直接看）
        mp4 = write_mp4(frames, os.path.join(args.out, f"wrist_camera_ep{ep}.mp4"))
        print(f"写入 {mp4}")
    else:
        cam_note = "camera0_rgb（JPEG-XL）未抽取"

    # ---- 4. 溯源 ----
    info = {
        "source_zip": args.src,
        "source_url": SRC_URL,
        "dataset": "UMI cup_in_the_wild (zarr_datasets)",
        "episode_index": ep,
        "frame_range": [s, e],
        "n_frames": n,
        "fields": ["data/robot0_eef_pos", "data/robot0_eef_rot_axis_angle",
                   "data/robot0_gripper_width"] + (["data/camera0_rgb"] if args.with_camera else []),
        "arm": "single (robot0_*)",
        "gripper_width_m": {"min": float(gfw.min()), "max": float(gfw.max()),
                            "mean": float(gfw.mean())},
        "gripper_closed_frames_lt_30mm": int((gfw.ravel() < 0.03).sum()),
        "g": {"min": float(g.min()), "max": float(g.max())},
        "conventions": {
            "UMI_MAX_APERTURE": UMI_MAX_APERTURE,
            "UMI_TCP_TO_PINCH": UMI_TCP_TO_PINCH.tolist(),
            "note": "与 tools/r2pear/b2s/sources.py 一致；UMI TCP 轴向仍未在真实数据上验证",
        },
        "camera_note": cam_note,
        "fps_note": "zarr 里没有帧率字段；UMI 手腕相机是 GoPro 录制，按 30 fps 写 mp4（假设）",
        "camera": {
            "key": "data/camera0_rgb",
            "shape": list(cam_shape) if cam_shape is not None else None,
            "codec_in_source": "JPEG-XL (imagecodecs_jpegxl, level 99)",
            "view": "手腕/夹爪第一视角（鱼眼）",
            "streams_in_this_zarr": 1,
            "note": "本 zarr 只有 camera0_rgb 一路；UMI 硬件论文里是夹爪两侧各一台 GoPro，"
                    "这份发布只保留了其中一路",
        },
    }
    with open(os.path.join(args.out, f"umi_ep{ep}_source.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"写入 {os.path.join(args.out, f'umi_ep{ep}_source.json')}")
    print(json.dumps({k: v for k, v in info.items() if k != "conventions"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
