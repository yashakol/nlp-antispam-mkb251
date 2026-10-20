"""Synthetic smoke tests only. No real corpus, fitting, or old-test evaluation."""
import base64
import json
import math
import os
import socket
import time
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from antispam import SpamFilter, evaluate, evaluation_rows
from service import predict_payload


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SpamFilter()
        cls.examples = [json.loads(s) for s in
                        (ROOT / "examples/synthetic_messages.jsonl").read_text().splitlines()]

    def test_frozen_model_loads_with_both_feature_branches(self):
        self.assertEqual(self.model.bundle["fit_rows"], 28493)
        self.assertEqual(self.model.threshold, 0.17315834063189364)
        self.assertEqual(len(self.model.bundle["features"]["word"].vocabulary_), 40000)
        self.assertEqual(len(self.model.bundle["features"]["char"].vocabulary_), 25000)
        self.assertEqual(self.model.bundle["features"]["char_weight"], 0.5)
        self.assertEqual(self.model.bundle["classifier"].C, 1)

    def test_synthetic_eml_and_decoded_text_agree(self):
        r = self.examples[0]
        message = EmailMessage()
        message["Subject"] = r["subject"]
        message["X-Spam-Flag"] = "Synthetic excluded envelope marker"
        message.set_content(r["body"])
        a = self.model.predict_eml(message.as_bytes())
        b = self.model.predict_text(r["subject"], r["body"])
        self.assertEqual(a, b)
        self.assertTrue(math.isfinite(a["score"]))
        self.assertEqual(a["predicted_label"], int(a["score"] >= self.model.threshold))

    def test_jsonl_and_eml_evaluation_agree_on_synthetic_inputs(self):
        a = evaluate(self.model, evaluation_rows(jsonl=ROOT / "examples/synthetic_messages.jsonl"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("ham", "spam"):
                (root / name).mkdir()
            for r in self.examples:
                message = EmailMessage()
                message["Subject"] = r["subject"]
                message.set_content(r["body"])
                name = "spam" if r["label"] else "ham"
                (root / name / "synthetic.eml").write_bytes(message.as_bytes())
            b = evaluate(self.model, evaluation_rows(data_dir=root))
        self.assertEqual(a["metrics"], b["metrics"])
        self.assertEqual(a["metrics"]["n"], 2)

    def test_cli_runs_outside_package_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-B", str(ROOT / "antispam.py")]
            for arguments in (["info"], ["predict", "--subject", "Meeting", "--body", "Agenda tomorrow"],
                              ["evaluate", "--jsonl", str(ROOT / "examples/synthetic_messages.jsonl")]):
                result = subprocess.run(command + arguments, cwd=directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsInstance(json.loads(result.stdout), dict)

    def test_api_validation_and_eml_payload(self):
        for bad in (None, {}, {"body": 9}, {"eml_base64": "not base64"}, {"path": "/anything"}):
            with self.assertRaises(ValueError):
                predict_payload(self.model, bad)
        raw = b"Subject: Synthetic example\n\nProject agenda tomorrow"
        result = predict_payload(self.model, {"eml_base64": base64.b64encode(raw).decode("ascii")})
        self.assertEqual(result, self.model.predict_eml(raw))

    @unittest.skipUnless(os.environ.get("NLP_HTTP_SMOKE") == "1",
                         "Opt-in network check: set NLP_HTTP_SMOKE=1 on a host permitting local sockets")
    def test_local_http_round_trip(self):
        import uvicorn
        from api import app
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    self.fail("FastAPI server failed to start")
                time.sleep(0.02)
            with urllib.request.urlopen(base + "/health", timeout=5) as response:
                self.assertEqual(json.load(response)["status"], "ok")
            payload = {"subject": "Synthetic local API", "body": "Project agenda"}
            request = urllib.request.Request(base + "/predict", data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(json.load(response), self.model.predict_text(**payload))
            bad = urllib.request.Request(base + "/predict", data=b'{"body": 123}',
                                         headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(bad, timeout=5)
            self.assertEqual(error.exception.code, 400)
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            sock.close()
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
