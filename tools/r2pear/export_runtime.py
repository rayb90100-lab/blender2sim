"""Portable bundle launcher; copied to interactive/run.py by export_scene.py."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    ap = argparse.ArgumentParser(description="Run or verify this exported MuJoCo environment")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--software-render", action="store_true")
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    os.environ["MENAGERIE"] = str(root / "assets/mujoco_menagerie")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if a.software_render:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["EGL_PLATFORM"] = "surfaceless"
        mesa = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
        if mesa.exists():
            os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(mesa)
        os.environ["R2P_BLEND_ENGINE"] = "CYCLES"
        os.environ.setdefault("R2P_BLEND_SAMPLES", "8")
    sys.path.insert(0, str(root / "tools/r2pear"))
    import web_ui as W
    if not a.check:
        sys.argv = ["web_ui.py", "--testdir", str(root / "scene"), "--host", a.host,
                    "--port", str(a.port), "--fps", str(a.fps)]
        W.main()
        return

    import mujoco
    import numpy as np
    s = W.Sim(str(root / "scene/stage0_spec/task_spec.json"), width=256, height=144)
    try:
        initial = s.env.d.qpos.copy()
        before = s.snapshot()
        assert not before["success"], "Initial state already satisfies task"
        # Load the standalone MJCF using only its packaged assets.
        m = mujoco.MjModel.from_xml_path(str(root / "model/scene.xml"))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("task_initial").id)
        mujoco.mj_forward(m, d)
        for field in ("nq", "nv", "nu", "nbody", "ngeom", "ncam"):
            assert getattr(m, field) == getattr(s.env.m, field), field
        np.testing.assert_allclose(d.qpos, initial, atol=1e-12)
        for field in ("body_mass", "body_inertia", "body_gravcomp", "geom_friction",
                      "geom_solref", "actuator_gainprm", "actuator_forcerange"):
            np.testing.assert_allclose(getattr(m, field), getattr(s.env.m, field), rtol=2e-5, atol=1e-9)
        assert m.opt.enableflags == s.env.m.opt.enableflags
        assert m.opt.timestep == s.env.m.opt.timestep
        for _ in range(100):
            mujoco.mj_step(m, d)
        assert np.isfinite(d.qpos).all()
        act = {"arm": dict.fromkeys(W.CT.SCHEMA["arm"], 0.0), "gripper": {"width": 0.0}}
        act["arm"]["z"] = 3.0
        s.do_action(act)
        assert s.env.used_steps > 0 and not np.allclose(s.env.d.qpos, initial)
        s.reset()
        np.testing.assert_allclose(s.env.d.qpos, initial, atol=1e-12)
        # Exercise the same asynchronous control/reset path used by the UI.
        s.oracle_hz = 10
        s.start_oracle_async()
        time.sleep(0.15)
        s.reset()
        assert not s._oracle_running and s.env.used_steps == 0
        np.testing.assert_allclose(s.env.d.qpos, initial, atol=1e-12)
        source_demo = s.env.spec.get('web_playback', {}).get('type') == 'source_demo'
        if source_demo:
            s.source_replay_pacing = False
            s.do_source_replay()
        else:
            on = next(p for p in s.env.spec["success"] if p["pred"] == "on")
            W.O.solve(s.env, on["obj"], on["support"])
        solved = s.snapshot()
        assert solved["success"], solved
        s.reset()
        assert s.render_orbit().startswith(b"\xff\xd8")
        s.reset()
        assert s.render_orbit().startswith(b"\xff\xd8")
        result = {"ok": True, "task": before["task"], "initial_success": before["success"],
                  "automatic_kind": 'source_demo' if source_demo else 'oracle',
                  "automatic_success": solved["success"], "automatic_steps": solved["used_steps"],
                  "source_replay": solved.get('source_replay') if source_demo else None,
                  "predicates": solved["predicates"], "reset_steps": s.env.used_steps,
                  "standalone_mjcf": True, "render_after_reset": True,
                  "multiccd": bool(m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD))}
        if not source_demo:
            result.update(oracle_success=solved['success'], oracle_steps=solved['used_steps'])
        print(json.dumps(result, ensure_ascii=False))
    finally:
        s.close_orbit()
        s.env.close()


if __name__ == "__main__":
    main()
