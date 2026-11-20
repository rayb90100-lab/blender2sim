"""Optional end-to-end verification of a bundle, using Playwright Chromium.

Launches temporary loopback servers, verifies WebGL replay and interactive UI,
captures screenshots, and stops both servers. Requires playwright + Chromium.
"""
import argparse
from io import BytesIO
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    bundle, out = a.bundle.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(bundle)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    service = None
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--enable-unsafe-swiftshader",
                                                            "--use-gl=angle", "--use-angle=swiftshader"])
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_port}/preview.html")
            page.wait_for_function("window.b2sPreview?.ready", timeout=90000)
            def check_canvas():
                from PIL import Image, ImageStat
                pixels = page.locator("canvas").screenshot()
                frame = Image.open(BytesIO(pixels)).convert("RGB")
                # Locator screenshots can include overlaid HTML controls. Only
                # inspect the scene interior, excluding the header and footer.
                w, h = frame.size
                interior = frame.crop((w // 10, h // 5, 9 * w // 10, 4 * h // 5))
                spread = max(ImageStat.Stat(interior).stddev)
                assert spread >= 2, "Web3D camera sees a nearly uniform image; check camera occlusion"
                return spread
            initial_spread = check_canvas()
            page.screenshot(path=str(out / "display-initial.png"))
            page.locator("#play").click()
            page.wait_for_timeout(500)
            assert float(page.locator("#time").input_value()) > 0
            result = page.evaluate("""() => {
                const p=window.b2sPreview, end=p.replay.timestamps_seconds.at(-1);
                p.seek(end);
                let maxError=0;
                for(const {node,c} of p.channels) {
                    const target=c.translation.at(-1);
                    node.position.toArray().forEach((v,i)=>maxError=Math.max(maxError,Math.abs(v-target[i])));
                }
                return {channels:p.channels.length,duration:end,maxPositionError:maxError};
            }""")
            page.wait_for_timeout(200)
            result["canvas_spatial_std"] = {"initial": initial_spread, "final": check_canvas()}
            page.screenshot(path=str(out / "display-final.png"))
            assert result["maxPositionError"] < 1e-9 and not errors, errors
            page.close()

            env = dict(os.environ)
            for k in ["MENAGERIE", "ROBOLAB_DATA", "PEAR_CODE", "PYTHONPATH"]:
                env.pop(k, None)
            with tempfile.TemporaryDirectory(prefix="b2s-http-check-") as tmp, (out / "service.log").open("w") as log:
                service = subprocess.Popen([sys.executable, str(bundle / "interactive/run.py"),
                                            "--port", str(port), "--software-render"], cwd=tmp, env=env,
                                           stdout=log, stderr=log, start_new_session=True)
                base = f"http://127.0.0.1:{port}"
                for _ in range(100):
                    try:
                        urllib.request.urlopen(base + "/state", timeout=1).close()
                        break
                    except OSError:
                        if service.poll() is not None:
                            raise RuntimeError("Bundle web service failed to start")
                        time.sleep(.1)
                page = browser.new_page(viewport={"width": 1280, "height": 1000})
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(base, wait_until="domcontentloaded")
                page.wait_for_function("document.querySelector('#c0').naturalWidth > 0")
                page.screenshot(path=str(out / "interactive-initial.png"))
                initial = page.evaluate("async()=>await(await fetch('/state')).json()")
                with page.expect_response(lambda r: r.url.endswith('/cmd') and r.request.method == 'POST'):
                    page.locator('#autoRun').click()
                def wait_state(predicate):
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        with urllib.request.urlopen(base + '/state', timeout=10) as response:
                            state = json.load(response)
                        if predicate(state):
                            return state
                        time.sleep(.1)
                    raise AssertionError(state)
                spec = json.loads((bundle / 'interactive/scene/stage0_spec/task_spec.json').read_text())
                source_demo = spec.get('web_playback', {}).get('type') == 'source_demo'
                solved = wait_state(lambda s: bool((s.get('source_replay') or {}).get('complete'))
                                    if source_demo else s['success'])
                assert solved['success'] and solved['used_steps'] > 0
                page.screenshot(path=str(out / "interactive-success.png"))
                with page.expect_response(lambda r: r.url.endswith('/cmd') and r.request.method == 'POST'):
                    page.get_by_role("button", name="重置场景").click()
                reset = wait_state(lambda s: s['reset_count'] > solved['reset_count'] and s['used_steps'] == 0)
                assert reset["used_steps"] == 0 and not reset["success"]
                spec = json.loads((bundle / "interactive/scene/stage0_spec/task_spec.json").read_text())
                blender_views = [view for view in ("agentview", "wrist") if view in spec["cameras"]] + ["orbit"]
                for view in blender_views:
                    with urllib.request.urlopen(base + "/frame?cam=blender&view=" + view, timeout=90) as r:
                        jpg = r.read()
                        assert r.status == 200 and jpg.startswith(b"\xff\xd8")
                        name = "interactive-blender.jpg" if view == "orbit" else f"interactive-blender-{view}.jpg"
                        (out / name).write_bytes(jpg)
                assert not errors, errors
                page.close()
            browser.close()
        report = {"ok": True, "display": result, "browser_errors": errors,
                  "artifact_hashes": {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
                                      for name in ['scene.json', 'mujoco-web3d-scene.glb', 'mujoco-web3d-replay.json',
                                                   'interactive-environment.tar.gz']},
                  "interactive": {"automatic_kind": 'source_demo' if source_demo else 'oracle',
                                  "automatic_success": solved['success'], "automatic_steps": solved['used_steps'],
                                  "source_replay": solved.get('source_replay') if source_demo else None,
                                  "reset_steps": reset["used_steps"], "blender_frame": True,
                                  "blender_views": blender_views}}
        if not source_demo:
            report['interactive'].update(oracle_success=solved['success'], oracle_steps=solved['used_steps'])
        (out / "browser-validation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    finally:
        if service is not None and service.poll() is None:
            os.killpg(service.pid, signal.SIGTERM)
            service.wait(timeout=10)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
