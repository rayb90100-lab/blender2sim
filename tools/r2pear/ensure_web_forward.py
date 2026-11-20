"""Ensure the loopback 900x -> 809x alias used by VS Code remote previews."""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ADDRESSES = ('127.0.0.1', '127.0.1.1', '::1')
ROOT = Path(__file__).resolve().parents[2]


def task_at(host, port):
    host = f'[{host}]' if ':' in host else host
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://{host}:{port}/state', timeout=2) as response:
        return json.load(response)['task']


async def serve(listen_port, target_port):
    async def handle(reader, writer):
        upstream = None
        tasks = []
        try:
            remote, upstream = await asyncio.wait_for(
                asyncio.open_connection('127.0.0.1', target_port), timeout=5)

            async def pump(source, target):
                while data := await source.read(65536):
                    target.write(data)
                    await target.drain()

            tasks = [asyncio.create_task(pump(reader, upstream)),
                     asyncio.create_task(pump(remote, writer))]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (OSError, asyncio.TimeoutError) as exc:
            print(f'forward connection ended: {exc}', flush=True)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for stream in (writer, upstream):
                if stream is not None:
                    stream.close()
                    try:
                        await stream.wait_closed()
                    except OSError:
                        pass

    servers = []
    try:
        for host in ADDRESSES:
            servers.append(await asyncio.start_server(handle, host, listen_port))
        print(f'loopback {listen_port} -> 127.0.0.1:{target_port} ready', flush=True)
        await asyncio.gather(*(server.serve_forever() for server in servers))
    finally:
        for server in servers:
            server.close()
        await asyncio.gather(*(server.wait_closed() for server in servers))


def ensure(listen_port, target_port):
    # Wait for the web process to finish compiling its model.
    deadline = time.monotonic() + 30
    while True:
        try:
            expected = task_at('127.0.0.1', target_port)
            break
        except (OSError, ValueError, KeyError):
            if time.monotonic() >= deadline:
                raise RuntimeError(f'WebUI {target_port} did not become ready')
            time.sleep(.2)

    def healthy():
        try:
            return all(task_at(host, listen_port) == expected for host in ADDRESSES)
        except (OSError, ValueError, KeyError):
            return False

    if healthy():
        print(f'forward verified: localhost:{listen_port} -> task {expected}')
        return
    logs = ROOT/'local'
    logs.mkdir(exist_ok=True)
    logfile = logs/f'forward-{listen_port}.log'
    with logfile.open('ab', buffering=0) as log:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), '--port', str(target_port),
             '--listen-port', str(listen_port), '--serve'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    for _ in range(30):
        time.sleep(.1)
        if proc.poll() is not None:
            raise RuntimeError(f'Forward port occupied or startup failed; inspect {logfile}')
        if healthy():
            (logs/f'forward-{listen_port}.pid').write_text(str(proc.pid)+'\n')
            print(f'forward ready: localhost:{listen_port} -> 127.0.0.1:{target_port}; task {expected}')
            return
    proc.terminate()
    proc.wait(timeout=5)
    raise RuntimeError(f'Forward health check failed; inspect {logfile}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--listen-port', type=int)
    parser.add_argument('--serve', action='store_true')
    args = parser.parse_args()
    if args.listen_port is None and not 8090 <= args.port <= 8099:
        parser.error('Default mapping requires 809x; otherwise specify --listen-port')
    listen_port = args.listen_port if args.listen_port is not None else args.port + 910
    if not (0 < listen_port < 65536 and 0 < args.port < 65536) or listen_port == args.port:
        parser.error('Ports must be distinct valid TCP ports')
    if args.serve:
        asyncio.run(serve(listen_port, args.port))
    else:
        ensure(listen_port, args.port)


if __name__ == '__main__':
    main()
