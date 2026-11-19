# blender2sim 环境变量 —— source 这个文件即可。
#
# 换机器时要改的只有下面三个路径变量；代码里不再有写死的绝对路径。
# 本机原值见 docs/PORTING.md。

# ---- 解释器 ----------------------------------------------------------------
# 装了 bpy + mujoco + torch 的那个 python。b2s 用它起 bpy 子进程。
export B2S_PYTHON="${B2S_PYTHON:-$(command -v python3)}"

# ---- 数据根 ----------------------------------------------------------------
# 权重、数据集、下载缓存、运行产物都放这里。要留出至少 100 GB（单条 UMI 数据 4-18 GB）。
export ROBOLAB_DATA="${ROBOLAB_DATA:-$HOME/roblab_data}"
export HF_HOME="${HF_HOME:-$ROBOLAB_DATA/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROBOLAB_DATA/cache/torch}"

# ---- mujoco_menagerie ------------------------------------------------------
# 需要 franka_emika_panda。github.com/google-deepmind/mujoco_menagerie
export MENAGERIE="${MENAGERIE:-$ROBOLAB_DATA/assets/mujoco_menagerie}"

# ---- ffmpeg ----------------------------------------------------------------
# validate.py 的出片要用。优先级：$FFMPEG → PATH 里的 ffmpeg → 常见 conda env。
# 本机原值是 /home/ubuntu/mambaforge/envs/E2E/bin/ffmpeg（系统 PATH 里没有）。
if [ -z "${FFMPEG:-}" ]; then
  for _cand in "$(command -v ffmpeg 2>/dev/null)" \
               "$HOME/mambaforge/envs/E2E/bin/ffmpeg" \
               /usr/bin/ffmpeg /usr/local/bin/ffmpeg; do
    if [ -n "$_cand" ] && [ -x "$_cand" ]; then export FFMPEG="$_cand"; break; fi
  done
  unset _cand
fi
[ -n "${FFMPEG:-}" ] || echo "WARN: 没找到 ffmpeg，validate.py 的出片会失败；请 export FFMPEG=/path/to/ffmpeg" >&2

# ---- CUDA（按机器实际情况改）------------------------------------------------
if [ -d /usr/local/cuda ]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
# 无头渲染一律走 EGL，不需要 Xvfb / DISPLAY
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
