#!/usr/bin/env python3
"""增强验收媒体校验器：实际解码图像、核对尺寸，并核对帧身份。

帧身份：渲染清单里记录的该帧完整 qpos，必须与原生轨迹 trajectory.npz 中同一源帧的 qpos 一致
（物理确定性重跑；容差 1e-9）。通过才写 decoded / frame_identity_verified = true。
"""
import hashlib, json, sys
import numpy as np
from PIL import Image


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def validate(image, camera, trajectory, render_manifest, view, frame, renderer, candidate_id,
             display_inputs_sha256, out, gate_root=None):
    with Image.open(image) as im:
        dims = list(im.size)
        im.verify()
    with Image.open(image) as im:
        arr = np.asarray(im.convert("RGB"))
    nonblank = bool(arr.std() > 3.0)
    rm = json.load(open(render_manifest))
    traj = np.load(trajectory)
    q_render = np.array(rm["qpos_by_frame"][str(frame)])
    q_native = traj["qpos"][frame - 1]
    ident = bool(q_render.shape == q_native.shape and np.allclose(q_render, q_native, atol=1e-9))
    cam = json.load(open(camera))
    exp_dims = cam["resolution"] if renderer == "blender" else cam["mujoco_resolution"]
    rec = {"candidate_id": candidate_id, "trajectory_sha256": sha(trajectory),
           "display_inputs_sha256": display_inputs_sha256, "media_sha256": sha(image),
           "camera_sha256": sha(camera), "dimensions": dims, "view": view, "frame": frame, "renderer": renderer,
           "decoded": bool(nonblank and dims == exp_dims), "frame_identity_verified": ident,
           "validator": {"path": __import__("os").path.relpath(__import__("os").path.abspath(__file__), gate_root or "."),
                         "sha256": sha(__import__("os").path.abspath(__file__))},
           "checks": {"nonblank_std": float(arr.std()), "expected_dims": exp_dims,
                      "qpos_max_abs_diff": float(np.abs(q_render - q_native).max()) if q_render.shape == q_native.shape else None}}
    json.dump(rec, open(out, "w"), indent=2, ensure_ascii=False)
    return rec


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(validate(*a[:5], int(a[5]), a[6], a[7], a[8], a[9]), ensure_ascii=False))
