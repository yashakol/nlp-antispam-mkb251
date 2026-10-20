"""Synthetic CLI/API smoke and local timings. Never trains or reads a real corpus."""
import argparse
import base64
import datetime
import json
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def summary(values):
    ordered = sorted(values)
    return {"n": len(values), "median_ms": statistics.median(values),
            "p95_ms_nearest_rank": ordered[max(0, __import__("math").ceil(len(values) * .95) - 1)],
            "min_ms": min(values), "max_ms": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New evidence directory; no overwrite or uncertain-run retry")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "started.json").write_text(json.dumps({
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "synthetic_only": True, "training": False, "real_test_evaluation": False}) + "\n")
    imported = time.perf_counter()
    from antispam import SpamFilter
    import uvicorn
    from api import app
    import_ms = (time.perf_counter() - imported) * 1000
    started = time.perf_counter()
    model = SpamFilter()
    load_ms = (time.perf_counter() - started) * 1000
    payload = {"subject": "Project meeting", "body": "Please review the agenda for tomorrow."}
    for _ in range(5):
        model.predict_text(**payload)
    warm = []
    for _ in range(30):
        started = time.perf_counter()
        direct = model.predict_text(**payload)
        warm.append((time.perf_counter() - started) * 1000)
    cli = subprocess.run([sys.executable, "-B", str(ROOT / "antispam.py"), "predict",
                          "--subject", payload["subject"], "--body", payload["body"]],
                         capture_output=True, text=True)
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout) == direct
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                         access_log=False, log_level="error", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    startup = time.perf_counter()
    thread.start()
    base = f"http://127.0.0.1:{port}"
    def request(value):
        req = urllib.request.Request(base + "/predict", data=json.dumps(value).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            assert response.status == 200
            return json.load(response)
    try:
        deadline = time.monotonic() + 15
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("FastAPI startup did not complete")
            time.sleep(.02)
        startup_ms = (time.perf_counter() - startup) * 1000
        with urllib.request.urlopen(base + "/health", timeout=5) as response:
            health = json.load(response)
        assert health == model.info()
        for _ in range(5):
            assert request(payload) == direct
        timings = []
        for _ in range(30):
            started = time.perf_counter()
            assert request(payload) == direct
            timings.append((time.perf_counter() - started) * 1000)
        raw = b"Subject: Project meeting\n\nPlease review the agenda for tomorrow."
        assert request({"eml_base64": base64.b64encode(raw).decode("ascii")}) == model.predict_eml(raw)
        try:
            request({"body": 123})
            raise AssertionError("Invalid request was accepted")
        except urllib.error.HTTPError as error:
            assert error.code == 400
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
    assert not thread.is_alive(), "Owned smoke-test server did not stop cleanly"
    result = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "passed",
        "synthetic_only": True, "model_hash": health["model_sha256"], "threshold": model.threshold,
        "global_python_changed": False, "model_retrained": False, "assignment_1600_evaluated": False,
        "python": platform.python_version(), "platform": platform.system(),
        "architecture": platform.machine(),
        "dependencies": {name: metadata.version(name) for name in
                         ("fastapi", "uvicorn", "numpy", "scipy", "scikit-learn", "joblib", "pydantic")},
        "imports_ms": import_ms, "model_load_after_imports_ms": load_ms,
        "warm_inference": summary(warm), "local_api_startup_in_process_ms": startup_ms,
        "local_http_round_trip": summary(timings),
        "checks": {"CLI_parity": True, "health_200": True, "predict_200": True,
                   "EML_base64_parity": True, "invalid_request_400": True,
                   "owned_server_stopped": True},
        "method": "One short synthetic message, batch1,5warmups,30serial measurements; local127.0.0.1 HTTP includes client/server/JSON overhead and a fresh connection per request.",
        "limitations": "Smoke timing only; not a production benchmark, throughput test or model-quality estimate."}
    (args.output_dir / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    output = [
        "$ python smoke_check.py --output-dir NEW_EVIDENCE_DIRECTORY",
        "PASS: frozen V4 loaded, CLI parity, GET /health 200, POST /predict 200, invalid request 400",
        f"Python {result['python']}; FastAPI {result['dependencies']['fastapi']}; Uvicorn {result['dependencies']['uvicorn']}",
        f"Model load after imports: {load_ms:.3f} ms",
        f"Warm inference (30 synthetic requests): median {statistics.median(warm):.3f} ms; p95 {result['warm_inference']['p95_ms_nearest_rank']:.3f} ms",
        f"Local HTTP (30 synthetic requests): median {statistics.median(timings):.3f} ms; p95 {result['local_http_round_trip']['p95_ms_nearest_rank']:.3f} ms",
        "Local socket only; owned server stopped. No training or assignment-set evaluation.",
        "Synthetic smoke measurements, not a production benchmark."]
    (args.output_dir / "terminal.txt").write_text("\n".join(output) + "\n")
    print("\n".join(output), flush=True)


if __name__ == "__main__":
    main()
