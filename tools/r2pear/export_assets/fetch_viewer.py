"""Fetch pinned, hash-checked Three.js r170 sources for offline exported viewers."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

FILES = {
    "three.module.js": ("build/three.module.js", "ce1fa418de16a19495a9f72495580e3015d7745c296d3ce0485897f902ddedfb"),
    "GLTFLoader.js": ("examples/jsm/loaders/GLTFLoader.js", "45139faddd5aaf48ed2d62203d976e5cbd703db1a592de40527f9f6cf58abd44"),
    "OrbitControls.js": ("examples/jsm/controls/OrbitControls.js", "80efaadea4f8a636a65fb0bd08bfef62f3d93a0bb94e2e7500f23176c5c07f4e"),
    "BufferGeometryUtils.js": ("examples/jsm/utils/BufferGeometryUtils.js", "c25b7930e570e9ec56173cd3b866ec8d2e10016630db3937efb439daf1cedbf6"),
    "LICENSE": ("LICENSE", "4c40a1ef62450b857c3b2aaf294936304cd552d965fbcd9d32d4c5bcf4ba4454"),
}


def check(directory):
    for name, (_, expected) in FILES.items():
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unexpected viewer dependency hash: {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (source, expected) in FILES.items():
        path = args.out / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        data = urllib.request.urlopen("https://raw.githubusercontent.com/mrdoob/three.js/r170/" + source,
                                      timeout=60).read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Download checksum failed: {name}")
        path.write_bytes(data)
    check(args.out)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
