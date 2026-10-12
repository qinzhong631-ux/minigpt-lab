"""启动真实本地服务并通过 Gradio 客户端调用；默认结束后关闭服务。"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-running", action="store_true")
    args = parser.parse_args()
    # 公开仓库不包含本机日志；首次运行时创建输出目录。
    (ROOT / "evidence").mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, PYTHONUTF8="1", GRADIO_ANALYTICS_ENABLED="False", NO_PROXY="127.0.0.1,localhost")
    url = f"http://127.0.0.1:{port}"
    log = (ROOT / "evidence/app_server.txt").open("w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-X", "utf8", "-u", "app.py", "--device", "cpu", "--port", str(port)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    success = False
    try:
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError(f"服务提前退出 {proc.returncode}，见 evidence/app_server.txt")
            try:
                with urlopen(url + "/config", timeout=2) as response:
                    config = json.loads(response.read())
                break
            except (OSError, ValueError):
                time.sleep(0.5)
        else:
            raise TimeoutError("Gradio 未在时限内启动")
        from gradio_client import Client
        client = Client(url, verbose=False)
        output = client.predict("Alice", 40, 0.8, 0, 42, api_name="/generate")
        repeat = client.predict("Alice", 40, 0.8, 0, 42, api_name="/generate")
        assert isinstance(output, str) and output.startswith("Alice") and len(output) == 45
        assert output == repeat
        record = {"url": url, "pid": proc.pid, "http_config_ok": True, "component_count": len(config["components"]), "api_generate_ok": True, "same_seed_repeat_equal": True, "output": output, "server_kept_running": args.keep_running}
        (ROOT / "evidence/app_http_verification.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        success = True
    finally:
        if not args.keep_running or not success:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    main()
