"""Frozen feature construction, metrics and calibration threshold rules for V2."""
import csv
import datetime
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from maildata import char_preprocess, sha256

ROOT = Path(__file__).resolve().parent
SOURCES = ("trec07", "enron", "ling")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def log(event, **values):
    print(json.dumps({"event": event, "utc": now(), **values}), flush=True)


def load_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    n = tn + fp + fn + tp
    return {
        "n": n, "ham": tn + fp, "spam": tp + fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tn + tp) / n if n else None,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else None,
    }


def breakdown(y, scores, sources, threshold):
    y, scores, sources = np.asarray(y), np.asarray(scores), np.asarray(sources)
    pred = (scores >= threshold).astype(int)
    return {
        "overall": metrics(y, pred),
        "sources": {
            source: metrics(y[sources == source], pred[sources == source])
            for source in sorted(set(sources.tolist()))
        },
    }


def make_vectorizers(protocol):
    words = dict(protocol["features"]["words"])
    words["ngram_range"] = tuple(words["ngram_range"])
    words["dtype"] = np.float64
    chars = dict(protocol["features"]["characters"])
    chars.pop("text_limit")
    chars.pop("weight")
    chars["ngram_range"] = tuple(chars["ngram_range"])
    chars["dtype"] = np.float64
    chars["preprocessor"] = char_preprocess
    return TfidfVectorizer(**words), TfidfVectorizer(**chars)


def fit_features(texts, protocol):
    """Only caller-supplied training texts can enter fit_transform."""
    word, char = make_vectorizers(protocol)
    x_word = word.fit_transform(texts)
    x_char = char.fit_transform(texts)
    weight = protocol["features"]["characters"]["weight"]
    x_both = hstack([x_word, x_char * weight], format="csr")
    return {"word": word, "char": char, "char_weight": weight}, x_word, x_both


def transform_features(features, texts, both=True):
    word = features["word"].transform(texts)
    if not both:
        return word
    char = features["char"].transform(texts)
    return hstack([word, char * features["char_weight"]], format="csr")


def fit_classifier(candidate, x, labels, seed):
    if candidate["classifier"] == "LinearSVC":
        clf = LinearSVC(C=candidate["C"], dual=candidate["dual"],
                        max_iter=candidate["max_iter"], random_state=seed)
    else:
        clf = LogisticRegression(C=candidate["C"], solver=candidate["solver"],
                                 max_iter=candidate["max_iter"], random_state=seed)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x, labels)
    convergence = [str(w.message) for w in observed
                   if issubclass(w.category, ConvergenceWarning)]
    return clf, convergence


def score_classifier(clf, x):
    if hasattr(clf, "predict_proba"):
        positive = int(np.flatnonzero(clf.classes_ == 1)[0])
        return np.asarray(clf.predict_proba(x)[:, positive], dtype=float)
    assert clf.classes_.tolist() == [0, 1]
    return np.asarray(clf.decision_function(x), dtype=float)


def score_bundle(bundle, texts):
    x = transform_features(bundle["features"], texts,
                           both=bundle["candidate"]["features"] == "word+char")
    return score_classifier(bundle["classifier"], x)


def threshold_selection(labels, scores, sources):
    """Enumerate all distinct scores; never split equal-score observations.

    Integer comparisons implement the frozen 1% / 2% / 80% constraints.
    Returns diagnostics even when no threshold satisfies all constraints.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    sources = np.asarray(sources)
    assert len(labels) == len(scores) == len(sources) and len(labels)
    assert set(labels.tolist()) <= {0, 1} and np.isfinite(scores).all()
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
                 len(sorted_scores) - 1]
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), sorted_scores[ends]]
    assert np.isfinite(thresholds).all()

    def counts(mask):
        return np.r_[0, np.cumsum(mask[order], dtype=np.int64)[ends]]

    ham = int(np.sum(labels == 0))
    spam = int(np.sum(labels == 1))
    assert ham and spam
    fp = counts(labels == 0)
    tp = counts(labels == 1)
    fpr = fp / ham
    recall = tp / spam
    fpr_ok = fp * 100 <= ham
    source_arrays = {}
    recalls = []
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        source_ham = int(np.sum(mask & (labels == 0)))
        source_spam = int(np.sum(mask & (labels == 1)))
        source_fp = counts(mask & (labels == 0))
        source_tp = counts(mask & (labels == 1))
        source_arrays[source] = (source_ham, source_spam, source_fp, source_tp)
        if source_ham >= 100:
            fpr_ok &= source_fp * 50 <= source_ham
        if source_spam:
            recalls.append(source_tp / source_spam)
    macro_recall = np.mean(recalls, axis=0)
    recall_ok = tp * 5 >= spam * 4
    passed = fpr_ok & recall_ok

    def best(mask):
        indices = np.flatnonzero(mask)
        if not len(indices):
            return None
        return int(max(indices, key=lambda i: (
            macro_recall[i], recall[i], -fpr[i], thresholds[i])))

    def describe(index):
        if index is None:
            return None
        threshold = float(thresholds[index])
        result = breakdown(labels, scores, sources, threshold)
        result.update({"threshold": threshold, "mean_source_recall": float(macro_recall[index]),
                       "passes_fpr_constraints": bool(fpr_ok[index]),
                       "passes_recall_constraint": bool(recall_ok[index])})
        return result

    # Diagnostic under recall floor: lowest overall FPR, then objective/tie rules.
    recall_indices = np.flatnonzero(recall_ok)
    lowest_fpr_index = min(
        recall_indices, key=lambda i: (fpr[i], -macro_recall[i], -recall[i], -thresholds[i])
    ) if len(recall_indices) else None
    selected = best(passed)
    return {
        "distinct_score_cutoffs": int(len(thresholds)),
        "passing_cutoffs": int(np.sum(passed)),
        "selected": describe(selected),
        "best_source_mean_recall_under_fpr_constraints": describe(best(fpr_ok)),
        "lowest_overall_fpr_with_recall_at_least_80pct": describe(lowest_fpr_index),
        "source_ham_constraints_applied": [
            s for s, values in source_arrays.items() if values[0] >= 100],
        "empirical_point_estimates_only": True,
    }


def save_scores(path, rows, scores, threshold=None, folds=None):
    fields = ["id", "source_id", "label", "score"]
    if threshold is not None:
        fields += ["threshold", "predicted_label"]
    if folds is not None:
        fields += ["cv_fold"]
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for index, (row, score) in enumerate(zip(rows, scores)):
            values = [row["id"], row["source_id"], row["label"], float(score)]
            if threshold is not None:
                values += [float(threshold), int(score >= threshold)]
            if folds is not None:
                values += [int(folds[index])]
            writer.writerow(values)
