# blender2sim 环境变量 —— source 这个文件即可。
#
# 只需要管下面三个变量：B2S_PYTHON / ROBOLAB_DATA / MENAGERIE。
# 三个都有探测，找不到时会明确 WARN，不会静默指向一个装不了东西的解释器。
# 想跳过探测就自己先 export，脚本不会覆盖已有的值。

# ---- 解释器 ----------------------------------------------------------------
# 装 bpy + mujoco + torch 的那个 python，b2s 用它起 bpy 子进程。
#
# 光看 `command -v python3` 不够：系统 python3 一般没有 bpy，选中了要等跑到
# extract 阶段才炸。这里对候选逐个试 import，取第一个能导入 bpy+mujoco 的。
_b2s_has_deps() {
  [ -n "${1:-}" ] && [ -x "$1" ] || return 1
  "$1" -c 'import bpy, mujoco' >/dev/null 2>&1
}

if [ -z "${B2S_PYTHON:-}" ]; then
  for _cand in "$(command -v python3 2>/dev/null)" \
               "$HOME/mambaforge/envs/real2gym/bin/python" \
               "$HOME/miniconda3/envs/real2gym/bin/python" \
               "$HOME/anaconda3/envs/real2gym/bin/python" \
               "$HOME/mambaforge/envs/b2s/bin/python" \
               "$HOME/miniconda3/envs/b2s/bin/python"; do
    if _b2s_has_deps "$_cand"; then export B2S_PYTHON="$_cand"; break; fi
  done
  unset _cand
fi
if [ -n "${B2S_PYTHON:-}" ] && _b2s_has_deps "$B2S_PYTHON"; then
  :
elif [ -n "${B2S_PYTHON:-}" ]; then
  echo "WARN: B2S_PYTHON=$B2S_PYTHON 里 import bpy/mujoco 失败，b2s 会在 extract 阶段崩；" >&2
  echo "      装法见上文的 requirements 说明，或 export B2S_PYTHON=/path/to/right/python" >&2
else
  export B2S_PYTHON="$(command -v python3 2>/dev/null)"
  echo "WARN: 没找到装了 bpy+mujoco 的解释器（试过 python3 和常见 conda env 路径）。" >&2
  echo "      B2S_PYTHON 暂时指向 $B2S_PYTHON，跑 b2s 之前先 export B2S_PYTHON=/path/to/python。" >&2
fi
unset -f _b2s_has_deps

# ---- 数据根 ----------------------------------------------------------------
# 权重、数据集、下载缓存、运行产物都放这里。要留出至少 100 GB（单条 UMI 数据 4-18 GB）。
# 探测顺序：已导出的值 → $HOME/roblab_data → /media/data1/RoboLab_data（本机在工作区时的位置）
if [ -z "${ROBOLAB_DATA:-}" ]; then
  for _cand in "$HOME/roblab_data" /media/data1/RoboLab_data; do
    if [ -d "$_cand" ]; then export ROBOLAB_DATA="$_cand"; break; fi
  done
  unset _cand
  export ROBOLAB_DATA="${ROBOLAB_DATA:-$HOME/roblab_data}"
fi
[ -d "$ROBOLAB_DATA" ] || echo "WARN: ROBOLAB_DATA=$ROBOLAB_DATA 不存在；数据集和缓存会写不进去。" >&2
export HF_HOME="${HF_HOME:-$ROBOLAB_DATA/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROBOLAB_DATA/cache/torch}"

# ---- mujoco_menagerie ------------------------------------------------------
# 需要 franka_emika_panda。github.com/google-deepmind/mujoco_menagerie
export MENAGERIE="${MENAGERIE:-$ROBOLAB_DATA/assets/mujoco_menagerie}"
if [ ! -d "$MENAGERIE/franka_emika_panda" ]; then
  echo "WARN: MENAGERIE=$MENAGERIE 下没有 franka_emika_panda/，validate 会报" >&2
  echo "      ValueError: ParseXML: Error opening file ... panda_nohand.xml" >&2
fi

# ---- ffmpeg ----------------------------------------------------------------
# validate.py 的 V3 对照出片要用。优先级：$FFMPEG → PATH 里的 ffmpeg → 常见 conda env。
if [ -z "${FFMPEG:-}" ]; then
  for _cand in "$(command -v ffmpeg 2>/dev/null)" \
               "$HOME/mambaforge/envs/E2E/bin/ffmpeg" \
               "$HOME/miniconda3/envs/E2E/bin/ffmpeg" \
               /usr/bin/ffmpeg /usr/local/bin/ffmpeg; do
    if [ -n "$_cand" ] && [ -x "$_cand" ]; then export FFMPEG="$_cand"; break; fi
  done
  unset _cand
fi
[ -n "${FFMPEG:-}" ] || echo "WARN: 没找到 ffmpeg，validate.py 的 V3 对照会跳过；请 export FFMPEG=/path/to/ffmpeg" >&2

# ---- CUDA（按机器实际情况改）------------------------------------------------
if [ -d /usr/local/cuda ]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
# 无头渲染一律走 EGL，不需要 Xvfb / DISPLAY
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
