#!/usr/bin/env python3
"""交互式仿真界面（零依赖：标准库 http.server + 已有 cv2/mujoco）。

为什么不用 mujoco.viewer / rerun：
  - 本机 DISPLAY 为空、无 Xvfb → mujoco.viewer 起不来
  - rerun-sdk 强制 numpy>=2，与本环境（bpy/MoGe 要求 numpy<2）冲突，装不了
本方案用 EGL 离屏渲染 + MJPEG 流，浏览器里看，VSCode 远程端口转发即可访问。

界面：左两路相机画面，右控制面板（立柱式 jog、夹爪、重置、看 oracle 解题）。
动作协议与环境完全一致：PEAR 的位姿增量 {arm:{x,y,z,roll,pitch,yaw}, gripper:{width}}。

线程模型（MuJoCo/MjData 非线程安全）：
  唯一锁 mu 保护 (m, d)；动作线程独占执行 step；渲染线程独占 Renderer。
  http 请求只投递命令，不直接碰仿真。

用法:
  python tools/r2pear/web_ui.py --testdir <T> [--port 8080] [--init-id 0]
  然后浏览器打开 http://<机器>:8080（VSCode 会自动转发端口）
"""
import argparse
import io
import json
import socket
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import controller as CT  # noqa: E402
import droid_common as C  # noqa: E402
import env as E  # noqa: E402
import oracle as O  # noqa: E402
import predicates as PR  # noqa: E402
import spec as SP  # noqa: E402


class _OracleCancelled(Exception):
    """reset 时用来中断正在跑 oracle 的信号（不是错误）。"""


class BlenderProxy:
    """把 Blender EEVEE 渲染放到子进程里。

    为什么必须分进程：bpy 和 MuJoCo 的 EGL/GL 上下文在同一进程里互相干扰，
    实测同进程先开 MuJoCo Renderer 再渲染 bpy 会报 GL 错误或直接崩溃。
    子进程各用各的上下文，稳定。

    协议：NDJSON over stdin/stdout（见 blender_worker.py）。
    调用方（HTTP 线程）持一把独立锁，避免多个请求交错写管道。
    """

    def __init__(self, testdir, python=None, timeout=60.0):
        self.testdir = testdir
        self.python = python or sys.executable
        self.timeout = timeout
        self.lock = threading.Lock()
        self.proc = None
        self.err = None
        self.last_ms = None

    def _spawn(self):
        import subprocess
        env = {**os.environ, "MUJOCO_GL": "egl"}
        worker = os.path.join(HERE, "blender_worker.py")
        # b2s 从 blend 生成的环境：直接驱动原 blend（资产/材质/灯光全保留），见 b2s/blend_worker.py
        try:
            _sp = SP.load(os.path.join(self.testdir, "stage0_spec", "task_spec.json"))
            if _sp.get("blend_binding") and os.path.exists(_sp.get("source", {}).get("blend", "")):
                worker = os.path.join(HERE, "b2s", "blend_worker.py")
        except (OSError, ValueError):
            pass
        # stderr 落到文件而不是 DEVNULL：子进程出错时能查（早期丢进 DEVNULL，
        # 表现为「无响应」，完全看不到原因）
        self.logpath = os.path.join("/tmp", f"blender_worker_{os.getpid()}.log")
        self._logfh = open(self.logpath, "w")
        self.proc = subprocess.Popen(
            [self.python, worker, "--testdir", self.testdir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._logfh,
            text=True, bufsize=1, env=env)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def render(self, az, el, dist, w, h, quality=80, cam="orbit", pose=None):
        """返回 JPEG bytes；失败抛异常。首次调用会启进程（约 3–6 s）。

        cam: "orbit"（az/el/dist 驱动）或 spec 里的固定相机名（agentview / wrist）。
        三个视角共用同一个 Blender 进程与场景，切换相机不重建。
        """
        with self.lock:
            if not self.alive():
                self._spawn()
            body = {"cam": cam, "az": az, "el": el, "dist": dist,
                    "w": w, "h": h, "quality": quality}
            if pose:
                # 子进程没有 MuJoCo 模型：每次渲染前把当前位姿推过去，
                # 否则画面永远停在子进程启动那一刻（oracle/操作后不动）。
                body["bodies"] = pose.get("bodies") or {}
                body["wrist_matrix"] = pose.get("wrist_matrix")
                body["state"] = pose.get("state")          # b2s/blend_worker 用：关节角/夹爪/物体位姿
            req = json.dumps(body)
            t0 = time.time()
            try:
                self.proc.stdin.write(req + "\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
            except (BrokenPipeError, OSError) as e:
                self.proc = None
                raise RuntimeError(f"blender 子进程断开: {e}")
            self.last_ms = (time.time() - t0) * 1000
            if not line:
                self.proc = None
                raise RuntimeError("blender 子进程无响应")
            rep = json.loads(line)
            if not rep.get("ok"):
                raise RuntimeError(rep.get("err", "unknown"))
            import base64
            return base64.b64decode(rep["jpeg_b64"])

    def close(self):
        if self.alive():
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None


class Sim:
    """一个环境实例 + 两个后台线程。"""

    def __init__(self, spec_path, init_states=None, init_id=0, width=512, height=288):
        self.spec_path = spec_path
        self.init_states = init_states
        self.init_id = init_id
        self.width, self.height = width, height
        self.mu = threading.RLock()
        self.cmd_mu = threading.Lock()
        self.cmd_q = []
        self.running = True
        self.note = ""
        self.blender = None          # 由 main() 注入 BlenderProxy（可选）
        # 轨道相机（自由转视角）。初始值在渲染线程首帧时由 agentview 的真实位姿反推。
        self.orbit = {"lookat": [0.5, 0.0, 0.0], "azimuth": 36.0, "elevation": -30.0, "distance": 0.6}
        # Blender 模式下 oracle 的步进闸门：前端每画完一帧就 +1，oracle 等它变化才走下一步。
        # 目的：**不丢帧**。Blender 单帧 118 ms（≈8.5 fps），而 oracle 默认按 80 Hz 跑，
        # 两者差一个数量级 —— 结果 68 步里只有 5 帧被画出来，看起来"一闪而过/跳步"。
        # 这里不降低 oracle 的速度参数，而是让 oracle **等渲染**，每步都看得见。
        self.render_seq = 0               # 前端渲染完成计数
        self.blender_live_frame = None    # oracle 每步渲染的 agentview JPEG 缓存
        self.blender_mode = False         # 前端当前是否处于 Blender 模式
        self.reset()

    # ---------- 生命周期 ----------
    def reset(self, init_id=None):
        # **先建好新 env 再替换**：早期版本先 close 旧的、再赋值新的，
        # 中间那一瞬间 self.env 指向已关闭的对象（m/d 被释放），
        # 渲染线程若刚好进锁就会拿到 e.m == None → AttributeError，
        # 而渲染线程没有容错，一崩就永久停摆（三张图全卡死、点按钮没反应）。
        # **先停掉正在跑的 oracle**：oracle 在独立线程里跑，会一直持有旧 env 的
        # `e` 并往里写。若 reset 直接换掉 self.env，两个线程就同时改仿真，
        # 且 `_oracle_running` 一直是 True —— 之后点 oracle 只会得到"正在执行"。
        self._cancel_oracle()
        with self.mu:
            if init_id is not None:
                self.init_id = int(init_id)
            new = E.R2PEnv(self.spec_path, self.init_states, init_id=self.init_id, render=False)
            new.open()
            old = getattr(self, "env", None)
            self.env = new
            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass
            self.log = []
            self.reset_count = getattr(self, "reset_count", 0) + 1
            self.note = f"已重置场景（{self.reset_count}）"
            self.frames = {}
            self.blender_live_frame = None
            self.last_source_replay = None
            self.pose_snapshot = self._pose_snapshot_locked()
            ok, det = PR.evaluate(self.env.m, self.env.d, self.env.spec)
            self.last_eval = {"success": ok, "predicates": det}

    # ---------- 命令 ----------
    def submit(self, cmd):
        with self.cmd_mu:
            if isinstance(cmd, dict) and cmd.get("t") == "reset":
                self.cmd_q.clear()
                if getattr(self, "_oracle_running", False):
                    self._oracle_cancel = True
            self.cmd_q.append(cmd)

    def drain(self, limit=1):
        with self.cmd_mu:
            out, self.cmd_q = self.cmd_q[:limit], self.cmd_q[limit:]
        return out

    # ---------- 状态快照 ----------
    def snapshot(self):
        with self.mu:
            e = self.env
            ok, det = PR.evaluate(e.m, e.d, e.spec)
            return {
                "task": e.spec["task_id"],
                "instruction": e.spec["instruction"],
                "state": e.robot.state(),
                "used_steps": e.used_steps,
                "budget": e.budget,
                "success": bool(ok),
                "predicates": det,
                "note": self.note,
                "init_id": self.init_id,
                "queue": len(self.cmd_q),
                "reset_count": self.reset_count,
                "automatic_action_label": e.spec.get('web_playback', {}).get('label', '让 oracle 解一遍'),
                "automatic_action_help": (e.spec.get('web_playback', {}).get('help', '读取原始关节和夹爪指令；物体仍由物理驱动。')
                                          if e.spec.get('web_playback', {}).get('type') == 'source_demo'
                                          else 'oracle 是特权脚本（用真值位姿），只用于可解性演示。'),
                "source_replay": getattr(self, 'last_source_replay', None),
            }

    # ---------- 动作执行 ----------
    def do_action(self, act):
        self._cancel_oracle()
        with self.mu:
            e = self.env
            try:
                res = e.step(act)
            except Exception as ex:
                self.note = f"动作被拒: {type(ex).__name__}: {ex}"
                return
            self.note = f"{res['outcome']}" + (f" ({res['stop_reason']})" if res.get("stop_reason") else "")
            self.last_eval = {"success": res["done"], "predicates": res.get("predicates", [])}

    def do_oracle(self):
        """特权解题：只用于演示/可解性验证，不是策略。

        **不能整个解题过程都持锁**：早期版本用 `with self.mu` 包住 O.solve，
        oracle 一跑（几秒）HTTP 线程的渲染请求全被阻塞，第三张图就冻住了。
        改为把锁下沉到每个控制步内部，步与步之间释放，渲染线程就能插进来。
        """
        if self.env.spec.get('web_playback', {}).get('type') == 'source_demo':
            return self.do_source_replay()
        e = self.env
        on = next((p for p in e.spec["success"] if p["pred"] == "on"), None)
        if on is None:
            self.note = "spec 里没有 on 谓词，无法自动解题"
            return
        orig = e.step
        interval = 1.0 / max(float(getattr(self, "oracle_hz", 80.0)), 1.0)

        def paced_step(action):
            def paced_cb(_fmax):
                # Physics holds the lock; release it between control steps so
                # reset, state requests and rendering remain responsive.
                self.pose_snapshot = self._pose_snapshot_locked()
                self.mu.release()
                try:
                    if self._oracle_cancel:
                        raise _OracleCancelled()
                    if self.blender_mode:
                        try:
                            self._render_now()
                        except Exception:
                            # A failed frame must not change the physics action.
                            pass
                    else:
                        time.sleep(interval)
                    if self._oracle_cancel:
                        raise _OracleCancelled()
                finally:
                    self.mu.acquire()

            with self.mu:
                if self._oracle_cancel:
                    raise _OracleCancelled()
                ctl = CT.PoseController(e.robot, action)
                outcome, n, reason = ctl.move(e.budget - e.used_steps, on_control_step=paced_cb)
                e.used_steps += n
                ok, detail = PR.evaluate(e.m, e.d, e.spec)
                done = ok or e.used_steps >= e.budget
                if ok:
                    outcome = "done"
                res = e._result([None] * n, outcome, done, success=ok)
                res["stop_reason"] = reason
                res["predicates"] = detail
                return res

        e.step = paced_step
        try:
            O.solve(e, on["obj"], on["support"])
            self.note = "oracle 执行完毕"
        except _OracleCancelled:
            self.note = "oracle 已取消"
        except Exception as ex:
            self.note = f"oracle 失败: {type(ex).__name__}: {ex}"
        finally:
            e.step = orig

    def do_source_replay(self):
        """Replay recorded joint/gripper commands, without oracle planning or object playback."""
        import hashlib
        from pathlib import Path
        import mujoco
        import sim as S
        e = self.env
        path = Path(self.spec_path).parent/'demo.npz'
        config = e.spec['web_playback']
        replay_label = config.get('label', '回放原始 DROID 轨迹')
        self.last_source_replay = {'complete': False, 'source': 'demo.npz'}
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != config['demo_sha256']:
                raise ValueError('原始轨迹文件校验不一致')
            with np.load(path, allow_pickle=False) as archive:
                q, g = archive['q'].copy(), archive['g_cmd'].copy()
                fps = float(archive['fps'])
            if (q.ndim != 2 or q.shape[1] != 7 or len(q) < 2 or g.shape != (len(q),)
                    or not np.isfinite(q).all() or not np.isfinite(g).all()
                    or not np.isfinite(fps) or fps <= 0 or np.any((g < 0) | (g > 1))):
                raise ValueError('原始轨迹格式或数值无效')
            arm, grip = S.arm_actuators(e.m), S.grip_actuator(e.m)
            sub = round(1/fps/e.m.opt.timestep)
            if sub < 1:
                raise ValueError('轨迹采样率超过当前物理步频')
            def update_view():
                if self._oracle_cancel:
                    raise _OracleCancelled()
                if getattr(self, 'source_replay_pacing', True):
                    started = time.monotonic()
                    if self.blender_mode:
                        try:
                            self._render_now()
                        except Exception as exc:
                            self.renderer_error = str(exc)
                    time.sleep(max(0., 1/fps - (time.monotonic()-started)))
            for frame in range(len(q)-1):
                if self._oracle_cancel:
                    raise _OracleCancelled()
                with self.mu:
                    for k in range(sub):
                        alpha = k/sub
                        e.d.ctrl[arm] = (1-alpha)*q[frame] + alpha*q[frame+1]
                        e.d.ctrl[grip] = 255*((1-alpha)*g[frame] + alpha*g[frame+1])
                        mujoco.mj_step(e.m, e.d)
                    e.used_steps = frame+1
                    self.pose_snapshot = self._pose_snapshot_locked()
                    self.note = f'{replay_label}：{frame+2}/{len(q)} 帧'
                update_view()
            for k in range(round(1/e.m.opt.timestep)):
                if self._oracle_cancel:
                    raise _OracleCancelled()
                with self.mu:
                    mujoco.mj_step(e.m, e.d)
                    if (k+1) % sub == 0:
                        self.pose_snapshot = self._pose_snapshot_locked()
                if (k+1) % sub == 0:
                    update_view()
            with self.mu:
                ok, predicates = PR.evaluate(e.m, e.d, e.spec)
                self.pose_snapshot = self._pose_snapshot_locked()
                self.last_source_replay = {'complete': True, 'source': 'demo.npz', 'sha256': digest,
                    'frames': len(q), 'source_fps': fps, 'physics_seconds_per_interval': sub*e.m.opt.timestep,
                    'success': bool(ok), 'predicates': predicates}
                self.note = f'{replay_label}结束：{"任务成功" if ok else "未完成任务"}（{len(q)} 帧）'
        except _OracleCancelled:
            self.note = '原始轨迹回放已取消'
        except Exception as exc:
            self.last_source_replay.update(error=f'{type(exc).__name__}: {exc}')
            self.note = f'原始轨迹回放失败：{type(exc).__name__}: {exc}'

    def _render_now(self):
        """让 Blender 画一帧当前状态（oracle 在 Blender 模式下每步调用）。

        画出来的 JPEG **存进缓存**，前端下次请求 agentview 时直接拿走，
        不再重复渲染 —— 否则 oracle 和前端抢同一个 Blender 子进程，
        两个 118 ms 的渲染排队，前端被挤到 0.44 s/轮（实测 2.3 fps）。
        """
        Bp = self.blender
        if Bp is None or not Bp.alive():
            return
        with self.mu:
            generation = self.reset_count
            pose = self._pose_snapshot_locked()
        data = Bp.render(0.0, 0.0, 0.6, 384, 216, 72, cam="agentview", pose=pose)
        with self.mu:
            if self.reset_count == generation:
                self.blender_live_frame = (data, time.time())

    def _cancel_oracle(self, wait_s=20.0):
        """请求 oracle 停下并等它退出。reset 必须调用，否则两个线程抢同一个 env。"""
        if not getattr(self, "_oracle_running", False):
            return
        self._oracle_cancel = True
        t0 = time.time()
        while getattr(self, "_oracle_running", False) and time.time() - t0 < wait_s:
            time.sleep(0.02)
        if getattr(self, "_oracle_running", False):
            raise TimeoutError("动作仍在停止中，请稍后重置")
        self._oracle_cancel = False

    def start_oracle_async(self):
        """在独立线程里跑 oracle，并**让步**给渲染。

        为什么需要：oracle 一次动作会连跑最多 100 个控制步且整段持锁，
        实测单次 orbit 渲染等了 275 ms、三秒采样只拿到 1 帧 —— 第三张图看起来冻住。
        （oracle 总耗时仅 0.56 s / 68 步，本身不慢。）
        做法：oracle 放到线程里，同时把每个物理子步的等待让出去；
        锁仍由 locked_step 保证一致性，但渲染线程能在这两个动作之间插进来。
        """
        if getattr(self, "_oracle_running", False):
            self.note = "oracle 正在执行"
            return
        if self.env.spec.get('web_playback', {}).get('type') == 'source_demo':
            # Start from the same scene initialization as the offline replay.
            self.reset()
        self._oracle_running = True
        self._oracle_cancel = False

        def job():
            try:
                self.do_oracle()
            finally:
                self._oracle_running = False

        self.note = (self.env.spec['web_playback'].get('label', '回放原始 DROID 轨迹') + '中…' if self.env.spec.get('web_playback', {}).get('type') == 'source_demo'
                     else "oracle 执行中…")
        threading.Thread(target=job, daemon=True).start()

    # ---------- 线程 ----------
    def worker(self):
        while self.running:
            cmds = self.drain(1)
            if not cmds:
                time.sleep(0.005)
                continue
            c = cmds[0]
            # **单条命令的异常不能打死线程**：早期版本 worker 里任何异常
            # （实测是 reset 后立刻读谓词触发 IndexError，以及消息里少了 "t" 键）
            # 都会让整个 worker 退出，之后命令队列永远不消费 ——
            # 用户看到的就是"点了没反应、oracle 不跑"，而且毫无提示。
            try:
                t = c.get("t") if isinstance(c, dict) else None
                if t == "act":
                    self.do_action(c["act"])
                elif t == "oracle":
                    self.start_oracle_async()
                elif t == "reset":
                    self.reset(c.get("init_id"))
                else:
                    self.note = f"未知命令 {t!r}（期望 act / oracle / reset）"
            except Exception as ex:
                import traceback
                self.note = f"命令 {c!r} 失败: {type(ex).__name__}: {ex}"
                traceback.print_exc()

    # ---------- 轨道相机（按需渲染，不进命令队列）----------
    # 早期版本把 orbit 也塞进命令队列：worker 每 5ms 只取 1 条，拖动时鼠标每帧都发请求，
    # 队列越积越长，画面就明显滞后。现在改为 HTTP 线程直接写参数 + 直接渲染。
    def set_orbit(self, az=None, el=None, dist=None):
        with self.mu:
            o = self.orbit
            if az is not None:
                o["azimuth"] = float(az)
            if el is not None:
                o["elevation"] = max(-89.0, min(89.0, float(el)))
            if dist is not None:
                o["distance"] = max(0.08, min(3.0, float(dist)))

    def render_orbit(self, quality=80, scale=1.0):
        """渲染一帧轨道相机。用自己的 Renderer（与后台流分开），避免互相排队。

        scale < 1 时以更小分辨率渲染（拖动中用它降延迟）。实测瓶颈不在渲染
        （512x288 渲染+编码仅 1.0 ms，384x216 为 0.8 ms），而在 HTTP 往返；
        小图能省掉传输时间，但主要收益来自前端「单请求在途」避免请求堆积。
        """
        import cv2
        import mujoco
        w = max(64, int(self.width * scale))
        h = max(36, int(self.height * scale))
        with self.mu:
            e = self.env
            key = (w, h)
            if (getattr(self, "_orb_r", None) is None or getattr(self, "_orb_size", None) != key
                    or getattr(self, "_orb_model", None) is not e.m):
                if getattr(self, "_orb_r", None) is not None:
                    try:
                        self._orb_r.close()
                    except Exception:
                        pass
                self._orb_r = mujoco.Renderer(e.m, h, w)
                self._orb_size = key
                self._orb_model = e.m
            if getattr(self, "_orb_cam", None) is None:
                # 只初始化一次（不能放在「尺寸变化」分支里：拖动中切分辨率会把视角重置）
                cid = e.m.camera("agentview").id
                p = e.d.cam_xpos[cid].copy()
                objc = np.array([o["center"] for o in e.spec["objects"].values()])
                look = objc.mean(0) if len(objc) else np.array([0.5, 0.0, 0.0])
                dxy = p[:2] - look[:2]
                self.orbit["lookat"] = [float(x) for x in look]
                self.orbit["distance"] = 0.6
                self.orbit["azimuth"] = float(np.degrees(np.arctan2(dxy[1], dxy[0])))
                self.orbit["elevation"] = -30.0
                self._orb_cam = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(self._orb_cam)
            o = self.orbit
            c = self._orb_cam
            c.lookat[:] = o["lookat"]
            c.azimuth = o["azimuth"]
            c.elevation = o["elevation"]
            c.distance = o["distance"]
            self._orb_r.update_scene(e.d, camera=c)
            img = cv2.cvtColor(self._orb_r.render(), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def close_orbit(self):
        with self.mu:
            r = getattr(self, "_orb_r", None)
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
            self._orb_r = None

    def _pose_snapshot_locked(self):
        """Capture the current model state while self.mu is held."""
        e = self.env
        m, d = e.m, e.d
        bodies = {}
        for g in range(m.ngeom):
            mid = m.geom_matid[g]
            rgba = m.mat_rgba[mid] if mid >= 0 else m.geom_rgba[g]
            if m.geom_group[g] > 2 or float(rgba[3]) <= 0.01:
                continue
            T = np.eye(4)
            T[:3, :3] = d.geom_xmat[g].reshape(3, 3)
            T[:3, 3] = d.geom_xpos[g]
            bodies[f"geom{g}"] = T.ravel().tolist()
        wrist = None
        if "wrist" in e.spec["cameras"]:
            wid = m.camera("wrist").id
            T = np.eye(4)
            T[:3, :3] = d.cam_xmat[wid].reshape(3, 3)
            T[:3, 3] = d.cam_xpos[wid]
            wrist = T.ravel().tolist()
        state = {"q": d.qpos[C.arm_qadr(m)].tolist(), "objects": {}}
        try:
            state["grip"] = float(d.qpos[m.joint(f"{C.GRIP_PREFIX}right_driver_joint").qposadr[0]])
        except KeyError:
            state["grip"] = None
        for name in e.spec["objects"]:
            bid = m.body(name).id
            T = np.eye(4)
            T[:3, :3] = d.xmat[bid].reshape(3, 3)
            T[:3, 3] = d.xpos[bid]
            state["objects"][name] = T.ravel().tolist()
        return {"bodies": bodies, "wrist_matrix": wrist, "state": state}

    def renderer_thread(self, fps=20):
        """后台流：三张图一起渲染（agentview / wrist / orbit）。

        轨道相机也放进这里，而不是只靠 /frame 按需渲染 —— 因为 oracle 执行时
        没人去请求 orbit 帧，第三张图就会停在旧画面（用户反馈）。
        三张图同源同步，观感一致。
        拖动时前端仍可用 /frame?cam=orbit 走「单请求设参数+取帧」的快速路径
        （延迟 0.8 ms），两条路径共用同一把锁与同一个相机状态，不冲突。
        """
        import cv2
        import mujoco
        r = None
        render_model = None
        cam = mujoco.MjvCamera()
        try:
            while self.running:
              try:
                with self.mu:
                    e = self.env
                    if e is None or getattr(e, "m", None) is None:
                        continue          # reset 中途，跳过这一帧
                    if r is None or render_model is not e.m:
                        if r is not None:
                            r.close()
                        r = mujoco.Renderer(e.m, self.height, self.width)
                        render_model = e.m
                        # 初始视角：lookat 取**场景中心**（物体质心均值），绕它转才不会转出画面。
                        # 方位角沿用 agentview 的水平方向，俯仰角与距离用固定初值 ——
                        # 不要用「相机到质心的方向」反算，那条方向与 agentview 视线不同，
                        # 反算出的 elevation 符号会错，实测会把相机压到桌面以下。
                        cid = e.m.camera("agentview").id
                        p = e.d.cam_xpos[cid].copy()
                        objc = np.array([o["center"] for o in e.spec["objects"].values()])
                        look = objc.mean(0) if len(objc) else np.array([0.5, 0.0, 0.0])
                        dxy = p[:2] - look[:2]
                        self.orbit["lookat"] = [float(x) for x in look]
                        self.orbit["distance"] = 0.6
                        self.orbit["azimuth"] = float(np.degrees(np.arctan2(dxy[1], dxy[0])))
                        # MuJoCo 约定：elevation 为负时相机在目标**上方**俯视（-30° 实测正常）
                        self.orbit["elevation"] = -30.0
                        mujoco.mjv_defaultCamera(cam)
                    o = self.orbit
                    cam.lookat[:] = o["lookat"]
                    cam.azimuth = o["azimuth"]
                    cam.elevation = o["elevation"]
                    cam.distance = o["distance"]
                    frames = {}
                    for c in e.CAMERAS:
                        r.update_scene(e.d, camera=c)
                        frames[c] = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)  # Renderer 返回 RGB
                    r.update_scene(e.d, camera=cam)
                    frames["orbit"] = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
                jpg = {}
                for c, f in frames.items():
                    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    jpg[c] = buf.tobytes()
                with self.mu:
                    if self.env is not e:
                        continue
                    self.frames = jpg
                    self.pose_snapshot = self._pose_snapshot_locked()
                time.sleep(1.0 / fps)
              except Exception:
                # 单帧失败只记一笔，不终止渲染线程 ——
                # 早期没有这层，任何一次异常（实测是 reset 竞态）都会让三张图永久卡死。
                import traceback
                self.renderer_error = traceback.format_exc()
                traceback.print_exc()
                time.sleep(0.1)
        except Exception:
            import traceback
            self.renderer_error = traceback.format_exc()
            traceback.print_exc()                    # 打到 stderr，便于从日志里看到
        finally:
            if r is not None:
                r.close()


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>r2pear 交互仿真</title>
<style>
:root{--bg:#14171a;--panel:#1d2126;--line:#2c3238;--fg:#e6e9ec;--dim:#8b959e;--ok:#3ecf8e;--bad:#ff6b6b;--acc:#4a9eff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial}
/* 布局：左侧三张画面占满剩余宽度，右侧面板 sticky 常驻 —— 滚动画布时按键始终可见 */
.wrap{display:grid;grid-template-columns:minmax(320px,1fr) 340px;gap:14px;padding:14px;align-items:start}
.cams{min-width:0;display:flex;flex-direction:column;gap:12px}
.cam{position:relative;margin:0}
.cam img{width:100%;display:block;border-radius:6px;background:#000;border:1px solid var(--line)}
.cam .tag{position:absolute;left:8px;top:8px;background:#000a;padding:2px 8px;border-radius:4px;font-size:12px;color:#ffd75e}
.orbit{position:relative;cursor:grab}
.orbit.drag{cursor:grabbing}
.orbit .tag{color:#8ad6ff}
.orbit .hint2{position:absolute;right:8px;bottom:8px;background:#000a;padding:2px 8px;border-radius:4px;font-size:11px;color:var(--dim)}
.overlay-btn{position:absolute;right:8px;top:8px;background:#000c;border:1px solid #3d454e;
  color:#8ad6ff;border-radius:5px;padding:4px 10px;font-size:12px;cursor:pointer}
.overlay-btn:hover{background:#1d2a36;border-color:#5a7fa0}
.overlay-btn.on{background:#1e3a1e;border-color:#2c6b2c;color:#9be89b}
/* sticky：面板贴在视口顶部，页面滚动时不动 */
.side{display:flex;flex-direction:column;gap:12px;position:sticky;top:14px;
      max-height:calc(100vh - 28px);overflow-y:auto;padding-right:2px}
/* 窄屏（手机/窄窗）退回单列：面板移到下方，仍然完整可用 */
@media (max-width:900px){
  .wrap{grid-template-columns:1fr}
  .side{position:static;max-height:none;overflow:visible}
}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
h3{margin:0 0 10px;font-size:13px;color:var(--dim);font-weight:600;letter-spacing:.04em;text-transform:uppercase}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
button{background:#252b32;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:9px 0;cursor:pointer;font-size:14px}
button:hover{background:#2f3740;border-color:#3d454e}
button:active{background:#3a434d}
button.wide{grid-column:span 3}
button.ghost{background:transparent;border-color:transparent;cursor:default;opacity:.35}
button.acc{background:#1e3a5f;border-color:#2d5680}
button.acc:hover{background:#24476f}
button.warn{background:#3d2323;border-color:#5c3030}
button.warn:hover{background:#4d2b2b}
button.go{background:#1e4a35;border-color:#2c6b4d}
button.go:hover{background:#245a41}
.row{display:flex;gap:8px;align-items:center;margin-top:8px}
.row label{color:var(--dim);font-size:12px;min-width:64px}
select,input[type=range]{flex:1;background:#252b32;color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:4px}
pre{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--dim);white-space:pre-wrap;word-break:break-all}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-family:ui-monospace,monospace;font-size:12px}
.kv b{color:var(--dim);font-weight:400}
.pred{font-family:ui-monospace,monospace;font-size:12px;padding:2px 0}
.pred.ok{color:var(--ok)} .pred.no{color:var(--bad)}
.big{font-size:20px;font-weight:600;margin-bottom:6px}
.big.ok{color:var(--ok)} .big.no{color:#ffb454}
.hint{color:var(--dim);font-size:11px;margin-top:6px}
</style></head><body>
<div class="wrap">
  <div class="cams">
    <div class="cam"><span class="tag" id="tag0">agentview（主视角）</span>
      <img id="c0" src="/stream?cam=agentview" style="max-height:32vh;object-fit:contain"></div>
    <div class="cam"><span class="tag" id="tag1">wrist（腕部）</span>
      <img id="c1" src="/stream?cam=wrist" style="max-height:32vh;object-fit:contain"></div>
    <div class="cam orbit" id="orbitBox">
      <span class="tag" id="orbitTag">自由视角（MuJoCo · 拖动旋转 · 滚轮缩放）</span>
      <!-- orbit 帧由 drawOrbit() 从 /frame?cam=orbit 拉取填入（也走后台缓存），
           不用 src 指向 MJPEG：那条路径已废弃，留着会显示空白 -->
      <img id="c2" alt="自由视角加载中…" draggable="false"
           style="max-height:32vh;min-height:120px;object-fit:contain">
      <span class="hint2" id="orbitInfo"></span>
      <button id="bswitch" class="overlay-btn"
              onclick="toggleBlender()">全部切到 Blender 精修</button>
    </div>
  </div>
  <div class="side">
    <div class="card">
      <div class="big no" id="verdict">未完成</div>
      <pre id="instr" style="margin-bottom:8px"></pre>
      <div class="kv">
        <b>工具点</b><span id="pos">-</span>
        <b>夹爪</b><span id="grip">-</span>
        <b>步数</b><span id="steps">-</span>
      </div>
    </div>

    <div class="card">
      <h3>平移（厘米）</h3>
      <!-- 6 个键两行三列：上排 Z+ / X+ / Z−，下排 Y+ / X− / Y−
           键位：↑↓ 控 Z，←→ 控 Y，+ − 控 X -->
      <div class="grid">
        <button onclick="jog('z', +1)">↑ Z+</button>
        <button onclick="jog('x', +1)">X+ 前</button>
        <button onclick="jog('z', -1)">↓ Z−</button>
        <button onclick="jog('y', +1)">← Y+</button>
        <button onclick="jog('x', -1)">X− 后</button>
        <button onclick="jog('y', -1)">Y− 右 →</button>
      </div>
      <div class="row"><label>步长</label>
        <select id="step">
          <option value="0.5">0.5 cm</option>
          <option value="1" selected>1 cm</option>
          <option value="2">2 cm</option>
          <option value="5">5 cm</option>
        </select>
      </div>
    </div>

    <div class="card">
      <h3>旋转（度）</h3>
      <div class="grid">
        <button onclick="rot('yaw', -5)">yaw −5</button>
        <button onclick="rot('pitch', +5)">pitch +5</button>
        <button onclick="rot('yaw', +5)">yaw +5</button>
        <button onclick="rot('roll', -5)">roll −5</button>
        <button onclick="rot('pitch', -5)">pitch −5</button>
        <button onclick="rot('roll', +5)">roll +5</button>
      </div>
    </div>

    <div class="card">
      <h3>夹爪</h3>
      <div class="grid">
        <button class="acc" onclick="grip(+1)">夹爪张开</button>
        <button class="acc" onclick="grip(-1)">夹爪闭合</button>
        <button class="acc" onclick="grip(0)">保持</button>
      </div>
    </div>

    <div class="card">
      <h3>其它</h3>
      <div class="grid">
        <button class="warn wide" onclick="post('reset')">重置场景</button>
        <button id="autoRun" class="go wide" onclick="post('oracle')">自动执行</button>
      </div>
      <div id="autoHelp" class="hint"></div>
    </div>

    <div class="card">
      <h3>谓词</h3>
      <div id="preds"></div>
      <pre id="note" style="margin-top:8px"></pre>
    </div>
  </div>
</div>
<script>
const S = {x:0,y:0,z:0,roll:0,pitch:0,yaw:0};
function post(t, extra={}){
  fetch('/cmd', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({t, ...extra})}).catch(()=>{});
}
// 下拉框的 value 本身就是**厘米**，而 PEAR 协议的动作增量也是**厘米**：
// controller.PoseController 里是 `tgt[k] = state[k] + action[k]`，
// 而 state 的位置是 cm（`x = p[0]*100`）。两边单位一致，**不要再除以 100**。
// （曾误以为动作是米，改成 /100 后每次 jog 只有 0.01 cm，表现为"按了没反应"。）
function stepSize(){ return parseFloat(document.getElementById('step').value); }
function jog(axis, sign){
  if(typeof axis !== 'string') return;
  const a = {arm:{x:0,y:0,z:0,roll:0,pitch:0,yaw:0}, gripper:{width:0}};
  a.arm[axis] = sign * stepSize();
  post('act', {act:a});
}
function rot(axis, deg){
  const a = {arm:{x:0,y:0,z:0,roll:0,pitch:0,yaw:0}, gripper:{width:0}};
  a.arm[axis] = deg * Math.PI/180;
  post('act', {act:a});
}
function grip(sign){
  const a = {arm:{x:0,y:0,z:0,roll:0,pitch:0,yaw:0}, gripper:{width:sign}};
  post('act', {act:a});
}
// 夹爪开合**切换**（空格键用）。width>0 张开、<0 闭合，0 保持。
// 判断当前状态的依据是 /state 里报的开口宽度：宽 > 4 cm 视为张开。
let gripOpen = true;
function gripToggle(){
  // 先按已知状态取反，避免等一次 /state 往返
  gripOpen = !gripOpen;
  grip(gripOpen ? +1 : -1);
}
async function poll(){
  try{
    const r = await fetch('/state'); const d = await r.json();
    const p = d.state.arm, g = d.state.gripper;
    document.getElementById('instr').textContent = d.instruction;
    document.getElementById('autoRun').textContent = d.automatic_action_label || '让 oracle 解一遍';
    document.getElementById('autoHelp').textContent = d.automatic_action_help || '';
    document.getElementById('pos').textContent =
      `(${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)}) cm`;
    document.getElementById('grip').textContent = g.width.toFixed(1)+' cm';
    // 用实测开口同步开合状态：>=4cm 记为张开。这样空格第一次按必然朝正确方向。
    gripOpen = g.width >= 4.0;
    document.getElementById('steps').textContent = d.used_steps+' / '+d.budget;
    const v = document.getElementById('verdict');
    v.textContent = d.success ? '任务成功' : '未完成';
    v.className = 'big ' + (d.success ? 'ok' : 'no');
    document.getElementById('preds').innerHTML = d.predicates.map(x =>
      `<div class="pred ${x.ok?'ok':'no'}">${x.ok?'✓':'✗'} ${x.pred}(${x.obj||''})</div>`).join('');
    document.getElementById('note').textContent = d.note || '';
  }catch(e){}
}
setInterval(poll, 400); poll();
// 页面一加载就主动拉一帧轨道相机，不必等用户先拖一下
window.addEventListener('load', ()=>{ dirty = true; drawOrbit(); });
// 空闲时也定时刷新轨道视角：oracle 执行 / 手动 jog 时场景在动，
// 只在拖动时才拉帧的话第三张图会一直停在旧画面（用户反馈的问题）。
// 拖动中由 queueOrbit 驱动，这里只在非拖动状态补充。
setInterval(()=>{ if(!drag){ preview = false; dirty = true; drawOrbit(); } }, 120);
// ---- 键盘：方向键 / ± 号 / WASD，支持长按连续移动 ----
// 长按逻辑：keydown 立即执行一次，之后若一直按住则每 120 ms 重复一次。
// 浏览器自身的系统级重复（e.repeat）不可靠且延迟约 500 ms，所以自己计时。
const KEYMAP = {
  'ArrowUp':    ['z', +1], 'ArrowDown':  ['z', -1],
  'ArrowLeft':  ['y', +1], 'ArrowRight': ['y', -1],
  '+':          ['x', +1], '=':          ['x', +1],   // = 是 + 的无 shift 键，一并接受
  '-':          ['x', -1], '_':          ['x', -1],
  'w':          ['x', +1], 's':          ['x', -1],
  'a':          ['y', +1], 'd':          ['y', -1],
  'q':          ['z', +1], 'e':          ['z', -1],
};
const REPEAT_MS = 80;
const held = new Map();          // key -> timer id

function startKey(key, mods){
  const m = KEYMAP[key];
  if(!m) return false;
  if(held.has(key)) return true;             // 已经在重复，忽略系统 repeat
  jog(...m);                                 // 按下立刻走一步
  held.set(key, setInterval(()=>jog(...m), REPEAT_MS));   // 长按持续
  return true;
}
function stopKey(key){
  const t = held.get(key);
  if(t !== undefined){ clearInterval(t); held.delete(key); }
}
document.addEventListener('keydown', e=>{
  // 不要抢输入框/下拉框的键
  const tag = (e.target.tagName || '').toLowerCase();
  if(tag === 'input' || tag === 'select' || tag === 'textarea') return;
  // 空格 = 夹爪开合**切换**（按一次换一个状态，不是按住才动）
  if(e.key === ' '){
    if(!e.repeat){ gripToggle(); }
    e.preventDefault(); return;
  }
  if(e.key === 'Enter'){ grip(-1); e.preventDefault(); return; }
  if(startKey(e.key)) e.preventDefault();
});
document.addEventListener('keyup', e=>{
  stopKey(e.key);
  // Shift 松开时 "+" 会变回 "="，两个都清掉，避免卡住不停走
  stopKey(e.key === '=' ? '+' : e.key);
});
// 失焦时清空所有按住状态，否则切走再回来会一直动
window.addEventListener('blur', ()=>{ for(const k of [...held.keys()]) stopKey(k); });
document.addEventListener('visibilitychange', ()=>{ if(document.hidden){ for(const k of [...held.keys()]) stopKey(k); } });

// ---- 轨道相机：拖动旋转、滚轮缩放 ----
// MuJoCo 约定：azimuth 绕 z 轴，elevation 俯仰（正值从上往下看），distance 到 lookat 的距离。
// 鼠标水平拖动 → azimuth；垂直拖动 → elevation。用指针事件，触摸屏也能用。
const orbit = {az: null, el: null, dist: null};
const box = document.getElementById('orbitBox');
const img = document.getElementById('c2');
const info = document.getElementById('orbitInfo');
let drag = null;
// 拖动/缩放：**一次请求同时带参数和取帧**（早期版本是 POST 参数 + GET 图，两次往返，
// 所以感觉"要拖动一段距离才反应"）。配合「单请求在途 + 最新覆盖」，永远只渲染最新姿态。
// 拖动中用半分辨率预览（preview=1），松手后补一张全质量图。
// 非拖动时按需定时刷新（oracle 执行时画面才会动），但只在有变化/播放中才拉。
let inflight = false, dirty = true, preview = false, lastQ = null;
// MuJoCo / Blender 视角切换。Blender EEVEE 约 8 fps，明显慢于 MuJoCo，
// 所以默认用 MuJoCo；切到 Blender 后拖动会变卡，这是引擎速度决定的上限。
let useBlender = false;
let blenderTimer = null;
// Drop late responses across mode switches, including Blender -> MuJoCo -> Blender.
let viewGeneration = 0;
function toggleBlender(){
  useBlender = !useBlender;
  viewGeneration++;
  dirty = true;
  const b = document.getElementById('bswitch');
  const ot = document.getElementById('orbitTag');
  b.classList.toggle('on', useBlender);
  b.textContent = useBlender ? '切回 MuJoCo' : '全部切到 Blender 精修';
  ot.textContent = useBlender ? '自由视角（Blender EEVEE · 拖动旋转）'
                              : '自由视角（MuJoCo · 拖动旋转 · 滚轮缩放）';
  document.getElementById('tag0').textContent =
    useBlender ? 'agentview（Blender EEVEE）' : 'agentview（主视角）';
  document.getElementById('tag1').textContent =
    useBlender ? 'wrist（Blender EEVEE）' : 'wrist（腕部）';
  if(useBlender){
    // 三张图都改由 Blender 出：agentview / wrist 由定时器刷新，
    // orbit 跟随拖动（也走 Blender）。
    // 注意 Blender 约 9 fps（110 ms/帧），三张图串行渲染会更慢，
    // 所以给固定视角设 250 ms 间隔，够看、不抢死 CPU。
    drawOrbit();
    fetch('/mode?blender=1', {cache:'no-store'}).catch(()=>{});
    clearInterval(blenderTimer);
    blenderTimer = setInterval(drawFixedBlender, 120);
  }else{
    clearInterval(blenderTimer); blenderTimer = null;
    fetch('/mode?blender=0', {cache:'no-store'}).catch(()=>{});
    document.getElementById('c0').src = '/stream?cam=agentview&_=' + performance.now();
    document.getElementById('c1').src = '/stream?cam=wrist&_=' + performance.now();
    dirty = true; drawOrbit();
  }
}
// 固定视角的 Blender 帧：单请求在途，避免堆积
// Blender 帧：**串行**请求两路，画完一轮再排下一轮。
// 为什么串行而不是并发：Blender 子进程是单线程的（单帧 118 ms），
// 并发请求只会在它内部排队，前端却以为自己在并行，反而堆积。
// 更重要的是：oracle 在 Blender 模式下**等渲染**推进（见后端 render_seq），
// 一轮渲染 = 两路都更新完，节奏才和画面严格对应，不会出现"走了好几步画一帧"。
let fixedInflight = 0;
function drawFixedBlender(){
  if(!useBlender || fixedInflight > 0) return;
  fixedInflight = 1;
  const generation = viewGeneration;
  const views = [['c0','agentview'], ['c1','wrist']];
  let i = 0;
  const next = () => {
    if(!useBlender || generation !== viewGeneration || i >= views.length){ fixedInflight = 0; return; }
    const [id, view] = views[i++];
    fetch(`/frame?cam=blender&view=${view}&w=384&h=216&quality=72&_=${performance.now()}`,
          {cache:'no-store'})
      .then(r => r.ok ? r.blob() : null)
      .then(b => {
        if(!b || !useBlender || generation !== viewGeneration) return;
        const el = document.getElementById(id);
        if(el.dataset.url) URL.revokeObjectURL(el.dataset.url);
        const u = URL.createObjectURL(b);
        el.dataset.url = u; el.src = u;
      })
      .catch(()=>{})
      .finally(next);
  };
  next();
}
function orbitQS(){
  const p = new URLSearchParams({
    cam: useBlender ? 'blender' : 'orbit', preview: preview?1:0, _: performance.now()});
  if(useBlender) p.set('view', 'orbit');
  if(orbit.az !== null) p.set('az', orbit.az.toFixed(2));
  if(orbit.el !== null) p.set('el', orbit.el.toFixed(2));
  if(orbit.dist !== null) p.set('dist', orbit.dist.toFixed(3));
  return '/frame?' + p.toString();
}
function drawOrbit(){
  if(inflight){ dirty = true; return; }
  if(!dirty) return;
  dirty = false; inflight = true;
  const generation = viewGeneration;
  fetch(orbitQS(), {cache:'no-store'})
    .then(r => r.ok ? r.blob() : null)
    .then(b => {
      if(b && generation === viewGeneration){
        if(img.dataset.url) URL.revokeObjectURL(img.dataset.url);
        const u = URL.createObjectURL(b);
        img.dataset.url = u; img.src = u;
      }
    })
    .catch(()=>{})
    .finally(()=>{ inflight = false; if(dirty) requestAnimationFrame(drawOrbit); });
}
function queueOrbit(prev){
  preview = !!prev;
  info.textContent = `az ${orbit.az===null?'-':orbit.az.toFixed(0)}°  el ${orbit.el===null?'-':orbit.el.toFixed(0)}°  d ${orbit.dist===null?'-':orbit.dist.toFixed(2)}m`;
  dirty = true; requestAnimationFrame(drawOrbit);
}
box.addEventListener('pointerdown', e=>{
  // 点在右上角的切换按钮上时不要开始拖动 —— 早期版本没判断，
  // setPointerCapture 把后续事件全吃掉，按钮的 click 永远不触发（用户反馈"点了没反应"）
  if(e.target.closest('button')) return;
  drag = {x:e.clientX, y:e.clientY};
  box.classList.add('drag');
  box.setPointerCapture(e.pointerId);
  e.preventDefault();
});
box.addEventListener('pointermove', e=>{
  if(!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag = {x:e.clientX, y:e.clientY};
  if(orbit.az === null) orbit.az = 36;      // 与后端初始值一致，首次拖动前先取已知值
  if(orbit.el === null) orbit.el = -30;
  orbit.az -= dx * 0.5;
  orbit.el = Math.max(-89, Math.min(89, orbit.el + dy * 0.4));
  queueOrbit(true);          // 拖动中用半分辨率预览，保证跟手
});
box.addEventListener('pointerup', e=>{
  drag = null; box.classList.remove('drag');
  try{ box.releasePointerCapture(e.pointerId); }catch(_){}
  queueOrbit(false);         // 松手后补一张全质量图
});
box.addEventListener('wheel', e=>{
  e.preventDefault();
  if(orbit.dist === null) orbit.dist = 0.6;
  orbit.dist = Math.max(0.08, Math.min(3.0, orbit.dist * (e.deltaY > 0 ? 1.1 : 0.9)));
  queueOrbit(true);
  clearTimeout(box._wt);
  box._wt = setTimeout(()=>queueOrbit(false), 180);   // 滚轮停下后补全质量
}, {passive:false});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    sim = None
    # BaseHTTPRequestHandler 默认 HTTP/1.0：每个请求新建 TCP 连接、用完即关。
    # 拖动轨道相机时每帧一个请求，连接建立成了主要开销 —— 实测单帧渲染+编码仅 1 ms，
    # 但 HTTP 往返约 42 ms，几乎全是连接开销。改 1.1 + keep-alive 后复用连接。
    protocol_version = "HTTP/1.1"
    # Nagle 算法 + 对端延迟确认（delayed ACK）会让每个小响应卡住 ~40 ms ——
    # 实测单帧渲染+编码只 1 ms，但每个请求稳定 40 ms，正是 delayed-ACK 定时器。
    # 关掉 Nagle（=TCP_NODELAY）后小包立即发出。
    disable_nagle_algorithm = True

    def log_message(self, *a):
        pass  # 静默，避免刷屏

    def _send(self, code, body, ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        u = urlparse(self.path)          # 必须用 path 部分比较：self.path 含查询串，
        path, q = u.path, parse_qs(u.query)   # 直接 != "/frame" 会永远不匹配（曾导致 404）
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/mode":
            self.sim.blender_mode = (q.get("blender") or ["0"])[0] == "1"
            return self._send(200, json.dumps({"blender_mode": self.sim.blender_mode}).encode(),
                              "application/json; charset=utf-8")
        if path == "/state":
            return self._send(200, json.dumps(self.sim.snapshot(), ensure_ascii=False).encode(),
                              "application/json; charset=utf-8")
        if path == "/stream":
            return self._mjpeg((q.get("cam") or ["agentview"])[0])
        if path == "/frame":
            cam = (q.get("cam") or ["agentview"])[0]
            if cam == "blender":
                # 三个视角都走 Blender：请求里用 "view" 指定 agentview / wrist / orbit。
                Bp = self.sim.blender
                if Bp is None:
                    return self._send(503, b"blender proxy not configured")
                try:
                    view = (q.get("view") or ["orbit"])[0]
                    o = self.sim.orbit
                    az = float((q.get("az") or [o["azimuth"]])[0])
                    el = float((q.get("el") or [o["elevation"]])[0])
                    dist = float((q.get("dist") or [o["distance"]])[0])
                    prev = (q.get("preview") or ["0"])[0] == "1"
                    if prev:
                        w, h, qual = max(96, self.sim.width // 2), max(54, self.sim.height // 2), 60
                    else:
                        w, h, qual = self.sim.width, self.sim.height, 82
                    # oracle 在 Blender 模式下每步都渲染好了 agentview，
                    # 这里直接复用，避免重复渲染把前端挤慢。
                    if view == "agentview":
                        with self.sim.mu:
                            live = getattr(self.sim, "blender_live_frame", None)
                        if live is not None and time.time() - live[1] < 1.0:
                            return self._send(200, live[0], "image/jpeg")
                    with self.sim.mu:
                        pose = getattr(self.sim, "pose_snapshot", None)
                    data = Bp.render(az, el, dist, w, h, qual, cam=view, pose=pose)
                    self.sim.orbit.update({"azimuth": az, "elevation": el, "distance": dist})
                    # 通知 oracle：这一步已经画出来了，可以走下一步（见 oracle 的步进闸门）
                    self.sim.render_seq += 1
                    return self._send(200, data, "image/jpeg")
                except Exception as ex:
                    return self._send(500, f"blender render failed: {ex}".encode())
            if cam == "orbit":
                # 两种路径：
                #  - 带 az/el/dist（拖动中）：一次请求完成「设参数 + 渲染取帧」，不走后台流，
                #    延迟 ~0.8 ms，保证跟手。
                #  - 不带参数（空闲轮询）：直接取后台流缓存的 orbit 帧，与另外两张图同步，
                #    这样 oracle 执行时画面也会跟着动。
                try:
                    if "az" in q or "el" in q or "dist" in q:
                        self.sim.set_orbit(
                            az=(q.get("az") or [None])[0],
                            el=(q.get("el") or [None])[0],
                            dist=(q.get("dist") or [None])[0])
                        prev = (q.get("preview") or ["0"])[0] == "1"
                        return self._send(200, self.sim.render_orbit(quality=62 if prev else 82,
                                                                     scale=0.5 if prev else 1.0),
                                          "image/jpeg")
                    with self.sim.mu:
                        cached = getattr(self.sim, "frames", {}).get("orbit")
                    if cached:
                        return self._send(200, cached, "image/jpeg")
                    return self._send(200, self.sim.render_orbit(), "image/jpeg")
                except Exception as ex:
                    return self._send(500, f"orbit render failed: {ex}".encode())
            with self.sim.mu:
                fr = getattr(self.sim, "frames", {})
                data = fr.get(cam)
            if data:
                return self._send(200, data, "image/jpeg")
            return self._send(503, b"frame not ready")   # 渲染线程还没出首帧
        self._send(404, b"not found")

    def _mjpeg(self, cam):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while self.sim.running:
                with self.sim.mu:
                    fr = getattr(self.sim, "frames", {})
                    data = fr.get(cam)
                if data:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                time.sleep(1 / 20)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        # /cmd   动作与重置 → 进命令队列（需要在 worker 里独占仿真执行）
        # /orbit 轨道相机参数 → **直接内联写入**，不进队列，否则拖动时延迟累积
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, b"bad json")
        if self.path == "/orbit":
            self.sim.set_orbit(az=body.get("azimuth"), el=body.get("elevation"),
                               dist=body.get("distance"))
            return self._send(200, b'{"ok":true}', "application/json")
        if self.path != "/cmd":
            return self._send(404, b"not found")
        self.sim.submit(body)
        self._send(200, b'{"ok":true}', "application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--init-id", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()

    T = a.testdir
    spec_path = os.path.join(T, "stage0_spec", "task_spec.json")
    init_states = os.path.join(T, "stage3_init_states", "init_states.json")
    if not os.path.exists(init_states):
        init_states = None

    sim = Sim(spec_path, init_states, init_id=a.init_id)
    # Blender EEVEE 渲染走子进程（bpy 与 MuJoCo 的 GL 上下文不能同进程共存）。
    # 懒启动：第一次切到 Blender 视角时才 spawn（约 3–6 s）；不用就不占资源。
    sim.blender = BlenderProxy(T)
    threading.Thread(target=sim.worker, daemon=True).start()
    threading.Thread(target=sim.renderer_thread, args=(a.fps,), daemon=True).start()
    time.sleep(0.6)  # 等首帧

    Handler.sim = sim

    class Server(ThreadingHTTPServer):
        # listen backlog 默认只有 5。页面一打开要并发拉 3 路画面 + /state，
        # 加上浏览器预连接和反复刷新，5 个槽位瞬间占满 → 新连接被内核丢弃，
        # 表现为"网页打不开/一直转圈"，而服务端日志一片正常（因为请求根本没进来）。
        request_queue_size = 128
        daemon_threads = True
        allow_reuse_address = True
        # Explicit IPv4 bind addresses (including loopback) need an IPv4 socket.
        address_family = (socket.AF_INET if a.host not in ("0.0.0.0", "", "::")
                          and ":" not in a.host else socket.AF_INET6)

        def server_bind(self):
            # 双栈必须显式关掉 IPV6_V6ONLY，否则只监听 ::，IPv4 连不进来。
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except OSError:
                pass
            super().server_bind()

    # **必须绑 IPv6 通配地址**：本机 `localhost` 只解析到 `::1`（见 /etc/hosts），
    # 而早期绑的是 0.0.0.0（纯 IPv4）。VSCode 的端口转发连的是 "localhost:PORT"，
    # 解析成 ::1 后连不上，日志里刷满
    #   [error] [remote-connection] Failed to connect tunnel to localhost:8122
    # 浏览器里就是"打不开"，而服务端一切正常（ss 显示在监听、curl 用 IPv4 打得通）。
    bind_host = "::" if a.host in ("0.0.0.0", "", "::") else a.host
    srv = Server((bind_host, a.port), Handler)
    print(f"r2pear 交互界面: http://{a.host}:{a.port}  （VSCode 会自动转发端口）")
    print(f"  task: {sim.env.spec['task_id']}")
    print(f"  init_states: {'有' if init_states else '无（只用源场景）'}  初始 init_id={a.init_id}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sim.running = False
        srv.server_close()


if __name__ == "__main__":
    main()
