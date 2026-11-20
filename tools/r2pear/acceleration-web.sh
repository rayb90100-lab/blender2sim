#!/usr/bin/env bash
set -e
if [ "$#" -lt 2 ]; then
  echo "usage: $0 <testdir> <port>" >&2
  exit 2
fi
cd "$(dirname "$0")/../.."
source local/env.sh
export R2P_USE_GPU=1
export R2P_BLEND_ENGINE=BLENDER_EEVEE_NEXT
export R2P_BLEND_SAMPLES="${R2P_BLEND_SAMPLES:-16}"
export R2P_BLEND_RUNTIME_BAKE=1
unset LIBGL_ALWAYS_SOFTWARE __EGL_VENDOR_LIBRARY_FILENAMES
export EGL_PLATFORM=surfaceless
exec "$B2S_PYTHON" tools/r2pear/web_ui.py \
  --testdir "$(realpath "$1")" --host 127.0.0.1 --port "$2" --fps "${B2S_FPS:-5}"
