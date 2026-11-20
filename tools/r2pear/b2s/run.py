#!/usr/bin/env python3
"""Command-line entry point for the Blender-to-Sim (b2s) pipeline.

The entry point owns argument parsing only. Stage orchestration lives in
``b2s/pipeline`` and the conversion algorithms remain in the existing domain
modules, so the same pipeline can later be called from Python or a job queue.

    python tools/r2pear/b2s/run.py --blend <X.blend> --out <simdir>
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline import PipelineOptions, run_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blend", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None,
                    help="缺省找 blend 同目录的 b2s_config.json")
    ap.add_argument("--videos", type=Path, default=None,
                    help="源视频目录（文件名 = 相机的 video_stream），缺省 = blend 同目录")
    ap.add_argument("--reextract", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--env-tests", action="store_true",
                    help="再跑反向对照 / 扰动扫描 / 初始状态采样（约 2 分钟）")
    ap.add_argument("--n-init", type=int, default=50)
    ap.add_argument("--last-resort-speed-diagnosis", metavar="REASON",
                    help="常规排查无解后才启用：最后对照原速/半速/1⁄3速度；不自动改轨迹")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    options = PipelineOptions(
        blend=a.blend,
        out=a.out,
        config=a.config,
        videos=a.videos,
        reextract=a.reextract,
        skip_validate=a.skip_validate,
        env_tests=a.env_tests,
        n_init=a.n_init,
        last_resort_speed_diagnosis=a.last_resort_speed_diagnosis,
    )
    try:
        return run_pipeline(options)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
