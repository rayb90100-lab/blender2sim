"""R2PEnv：Real2Gym 重建场景 → PEAR StatefulEnv 接口（open / step / close）。

env_args: {"spec_path": <task_spec.json>, "init_states_path": <init_states.json 或 None>, "init_id": int}
reward：二值，仅在 episode 因成功结束的那一步为 1（PEAR 做法：R(τ)=1 当且仅当仿真报告成功）。
不依赖 verl：PEAR 仓库存在时继承其 StatefulEnv，否则用同签名的本地基类（便于本机单独测试）。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
PEAR_CODE = os.environ.get("PEAR_CODE", os.path.join(os.path.dirname(os.path.dirname(HERE)), "PEAR", "code"))
if os.path.isdir(PEAR_CODE):
    sys.path.insert(0, PEAR_CODE)

import droid_common as C  # noqa: E402
import controller as CT  # noqa: E402
import predicates as PR  # noqa: E402
import sim as S  # noqa: E402
import spec as SP  # noqa: E402

try:
    from pear.sim.env import StatefulEnv  # noqa: E402
    HAVE_PEAR = True
except ImportError:                       # 本地占位：同名方法
    HAVE_PEAR = False

    class StatefulEnv:
        pass

PROMPT_PATH = Path(HERE) / "environment_prompt.md"


class R2PEnv(StatefulEnv):
    SCHEMA = CT.SCHEMA
    CAMERAS = ("agentview", "wrist")
    ENVIRONMENT_PROMPT = PROMPT_PATH.read_text().strip() if PROMPT_PATH.exists() else ""
    CLIENT_CLASS_PATH = "pear.sim.client.StatefulClient"
    SERVER_RESOURCES = {"num_cpus": 1}
    START_TIMEOUT_S = 300.0
    OPEN_TIMEOUT_S = 120.0
    STEP_TIMEOUT_S = 120.0

    def __init__(self, spec_path, init_states_path=None, init_id=0, render=True):
        self.spec = SP.load(spec_path)
        self.overrides = {}
        if init_states_path:
            states = json.load(open(init_states_path))["states"]
            self.overrides = states[int(init_id)]["overrides"]
        self.init_id = int(init_id)
        self.render_on = render
        self.m = self.d = self.robot = None
        self.renderer = None
        self.used_steps = 0
        self.budget = self.spec["step_budget"]["max_control_steps"]

    # ---- PEAR 接口 ----
    @classmethod
    def setup_env(cls, context):
        """一台机器一个仿真进程，用 R2P_PYTHON（real2gym 环境）运行；MuJoCo 走 EGL。
        【未在 PEAR 集群上验证】—— 本机只做单进程测试，不跑 PEAR 训练。"""
        from pear.sim.process import SimulatorProcess
        variables = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "OMP_NUM_THREADS": "1",
                     "PYTHONPATH": os.pathsep.join([HERE, os.path.dirname(HERE)])}
        proc = SimulatorProcess("sim", cls, os.environ.get("R2P_PYTHON", sys.executable), [], variables,
                                SimulatorProcess.PEAR_ROOT, context.log_dir)
        return [proc], 1

    def open(self):
        self.m, self.d = S.build(self.spec, self.overrides)
        S.reset(self.m, self.d, self.spec)
        m = self.m
        qa = C.arm_qadr(m)
        da = [int(m.joint(j).dofadr[0]) for j in C.PANDA_JOINTS]
        top = self.spec["table"]["top_z"] + float(self.overrides.get("table_dz", 0.0))
        self.robot = CT.Robot(m, self.d, top, S.arm_actuators(m), S.grip_actuator(m), qa, da)
        self.used_steps = 0
        return self._result([], None, False)

    def step(self, action):
        self._check(action)
        ctl = CT.PoseController(self.robot, action)
        outcome, n, reason = ctl.move(self.budget - self.used_steps)
        self.used_steps += n
        ok, detail = PR.evaluate(self.m, self.d, self.spec)
        done = ok or self.used_steps >= self.budget
        if ok:
            outcome = "done"
        res = self._result([None] * n, outcome, done, success=ok)
        res["stop_reason"] = reason
        res["predicates"] = detail
        return res

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = self.m = self.d = None

    @classmethod
    def task_name(cls, env_args):
        return json.load(open(env_args["spec_path"]))["task_id"]

    # ---- 内部 ----
    def _check(self, action):
        if not isinstance(action, dict) or set(action) != set(self.SCHEMA):
            raise ValueError(f"action must map exactly {list(self.SCHEMA)}")
        for part, qs in self.SCHEMA.items():
            if set(action[part]) != set(qs):
                raise ValueError(f"{part} must give exactly {qs}")
            for v in action[part].values():
                if type(v) not in (int, float) or not np.isfinite(v):
                    raise ValueError("values must be finite numbers")

    def render(self):
        import mujoco
        w, h = self.spec["render"]["width"], self.spec["render"]["height"]
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.m, h, w)
        out = []
        for c in self.CAMERAS:
            self.renderer.update_scene(self.d, camera=c)
            out.append(self.renderer.render().copy())
        return out

    def _result(self, native, outcome, done, success=False):
        images = None
        if not done and self.render_on:
            from PIL import Image
            images = [Image.fromarray(x) for x in self.render()]
        return {"images": images, "state": self.robot.state(), "actions": native,
                "rewards": [0.0] * max(0, len(native) - 1) + ([1.0 if success else 0.0] if native else []),
                "done": done, "executed": len(native), "outcome": outcome,
                "instruction": self.spec["instruction"], "used_control_steps": self.used_steps}
