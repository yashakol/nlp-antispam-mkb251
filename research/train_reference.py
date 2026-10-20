"""Portable reference for the historical single fit; not run during packaging.

Requires separately prepared normalized train/calibration JSONL with text, label,
source_id and group. It never opens evaluation data or overwrites the bundled V4.
This does not reproduce corpus acquisition or heuristic grouping by itself.
"""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_common import (load_rows, fit_features, fit_classifier, transform_features,
                               score_classifier, threshold_selection, sha256, save, now)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT / "research/protocol_v4.json").read_text())
    train, cal = load_rows(args.train), load_rows(args.calibration)
    assert len(train) == 28493 and len(cal) == 7555
    assert not {r["group"] for r in train} & {r["group"] for r in cal}
    save(args.output_dir / "fit.started.json",
         {"utc": now(), "train_sha256": sha256(args.train),
          "calibration_sha256": sha256(args.calibration),
          "seed": protocol["seed"], "fit_limit": 1})
    features, _, x = fit_features([r["text"] for r in train], protocol)
    classifier, warnings = fit_classifier(protocol["candidate"], x,
                                         np.asarray([r["label"] for r in train]), protocol["seed"])
    if warnings:
        save(args.output_dir / "failed.json", {"convergence_warnings": warnings})
        return
    scores = score_classifier(classifier, transform_features(features, [r["text"] for r in cal]))
    selected = threshold_selection([r["label"] for r in cal], scores, [r["source_id"] for r in cal])
    save(args.output_dir / "calibration.json", selected)
    bundle = {"features": features, "classifier": classifier, "candidate": protocol["candidate"],
              "representation": "unchanged_v2", "fit_rows": len(train), "fit_count": 1}
    joblib.dump(bundle, args.output_dir / "reference_fit.joblib", compress=3)
    # No refit and no evaluation. A failed calibration never starts another trial.
    print(json.dumps({"passed_calibration": selected["selected"] is not None,
                      "selected": selected["selected"]}))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
