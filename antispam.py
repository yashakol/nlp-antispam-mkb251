"""Portable V4 inference and evaluation. No automatic training or downloads."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import argparse
import json
import math
import sys
import threading
from pathlib import Path
import joblib
from threadpoolctl import threadpool_limits
from maildata import MAX_BYTES, feature_text, parse_email, sha256
from experiment_common import metrics, score_bundle

ROOT = Path(__file__).resolve().parent


class SpamFilter:
    def __init__(self):
        self.manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
        path = ROOT / "models" / self.manifest["model_file"]
        if not path.is_file():
            raise ValueError("Frozen model is missing. Copy models/v4.joblib from the private project package; weights are not included in the public repository.")
        if sha256(path) != self.manifest["model_sha256"]:
            raise ValueError("Bundled model checksum mismatch")
        if sha256(ROOT / "maildata.py") != self.manifest["maildata_sha256"]:
            raise ValueError("Frozen preprocessing checksum mismatch")
        self.threshold = self.manifest["threshold"]
        self.bundle = joblib.load(path)
        if self.bundle["representation"] != "unchanged_v2":
            raise ValueError("Unexpected model representation")
        self.lock = threading.Lock()

    def predict_normalized(self, texts):
        if not texts:
            return []
        with self.lock, threadpool_limits(limits=2):
            scores = score_bundle(self.bundle, texts)
        output = []
        for score in scores:
            value = float(score)
            if not math.isfinite(value):
                raise ValueError("Model returned a nonfinite score")
            label = int(value >= self.threshold)
            output.append({"label": "spam" if label else "ham", "predicted_label": label,
                           "score": value, "threshold": self.threshold,
                           "score_kind": "decision_margin_not_probability", "model_version": "V4"})
        return output

    def predict_text(self, subject, body, html=False):
        if not isinstance(subject, str) or not isinstance(body, str) or not isinstance(html, bool):
            raise ValueError("subject/body must be strings and html must be boolean")
        if len(subject.encode("utf-8")) + len(body.encode("utf-8")) > MAX_BYTES:
            raise ValueError("Decoded message exceeds 2 MB limit")
        return self.predict_normalized([feature_text(subject, body, html=html)])[0]

    def predict_eml(self, raw):
        if not isinstance(raw, bytes):
            raise ValueError("EML input must be bytes")
        return self.predict_normalized([parse_email(raw)["text"]])[0]

    def info(self):
        return {"status": "ok", "model_version": "V4", "threshold": self.threshold,
                "model_sha256": self.manifest["model_sha256"],
                "score_kind": "decision_margin_not_probability"}


def read_eml(path):
    with Path(path).open("rb") as f:
        raw = f.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Email exceeds 2 MB limit")
    return raw


def evaluation_rows(jsonl=None, data_dir=None):
    rows = []
    if jsonl:
        with Path(jsonl).open(encoding="utf-8") as f:
            for number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                if not isinstance(r, dict):
                    raise ValueError(f"JSONL line {number}: expected an object")
                label = r.get("label")
                if type(label) is not int or label not in (0, 1):
                    raise ValueError(f"JSONL line {number}: label must be integer 0 or 1")
                subject, body = r.get("subject", ""), r.get("body", "")
                html = r.get("html", False)
                if not isinstance(subject, str) or not isinstance(body, str) or not isinstance(html, bool):
                    raise ValueError(f"JSONL line {number}: invalid decoded text fields")
                if len(subject.encode("utf-8")) + len(body.encode("utf-8")) > MAX_BYTES:
                    raise ValueError(f"JSONL line {number}: message exceeds 2 MB")
                rows.append((label, feature_text(subject, body, html=html)))
                if len(rows) > 100_000:
                    raise ValueError("Evaluation is limited to 100000 rows")
    else:
        root = Path(data_dir)
        for name, label in (("ham", 0), ("spam", 1)):
            folder = root / name
            if not folder.is_dir():
                raise ValueError("Expected both ham/ and spam/ directories")
            for path in sorted(folder.rglob("*.eml")):
                if path.is_file():
                    rows.append((label, parse_email(read_eml(path))["text"]))
                if len(rows) > 100_000:
                    raise ValueError("Evaluation is limited to 100000 rows")
    if not rows:
        raise ValueError("No evaluation rows found")
    return rows


def evaluate(model, rows):
    labels = [r[0] for r in rows]
    predictions = []
    for start in range(0, len(rows), 128):
        predictions.extend(x["predicted_label"] for x in
                           model.predict_normalized([r[1] for r in rows[start:start + 128]]))
    return {"model": model.info(), "metrics": metrics(labels, predictions),
            "matrix_order": "[[TN,FP],[FN,TP]]",
            "note": "Metrics describe only supplied data; this command does not establish independence."}


def parser():
    p = argparse.ArgumentParser(description="Frozen NLP anti-spam V4: inference, evaluation and local API.")
    commands = p.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Predict one EML or decoded Subject/body.")
    predict.add_argument("--eml", type=Path)
    predict.add_argument("--subject", default="")
    body = predict.add_mutually_exclusive_group()
    body.add_argument("--body")
    body.add_argument("--body-file", type=Path, help="UTF-8 decoded body")
    predict.add_argument("--html", action="store_true", help="Decoded body is HTML")
    evaluation = commands.add_parser("evaluate", help="Evaluate user-supplied labeled data.")
    inputs = evaluation.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--jsonl", type=Path, help="Decoded subject/body/label JSONL")
    inputs.add_argument("--data-dir", type=Path, help="Directory containing ham/*.eml and spam/*.eml")
    evaluation.add_argument("--output", type=Path, help="Create a new aggregate JSON report; never overwrite")
    commands.add_parser("info", help="Validate and describe the bundled model")
    serve = commands.add_parser("serve", help="Start local FastAPI/Uvicorn HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "serve":
            if not 0 <= args.port <= 65535:
                raise ValueError("Port must be between 0 and 65535")
            from service import serve
            serve(args.host, args.port)
            return 0
        model = SpamFilter()
        if args.command == "info":
            result = model.info()
        elif args.command == "predict":
            if args.eml:
                if args.subject or args.body is not None or args.body_file or args.html:
                    raise ValueError("--eml cannot be combined with decoded text options")
                result = model.predict_eml(read_eml(args.eml))
            else:
                body = args.body_file.read_text(encoding="utf-8") if args.body_file else args.body
                if body is None and not args.subject:
                    raise ValueError("Supply --eml, --subject, --body or --body-file")
                result = model.predict_text(args.subject, body or "", args.html)
        else:
            if args.output and args.output.exists():
                raise ValueError("Output already exists; choose a new filename")
            result = evaluate(model, evaluation_rows(args.jsonl, args.data_dir))
            if args.output:
                with args.output.open("x", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
                    f.write("\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
