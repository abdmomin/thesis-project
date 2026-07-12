"""
Decision-threshold calibration for the WDC train/test prior mismatch (Issue 4).

WDC Products' train/valid splits are ~20% positive but the test split is ~11%.
The model, early stopping, and the default 0.5 / argmax decision rule are all
calibrated to the 20% prior, which caps test F1. This script reports what
happens under three operating points, all chosen WITHOUT peeking at test labels
except the last (oracle) reference:

  1. default        — argmax / 0.5 (current reporting).
  2. valid_max_f1   — threshold maximizing F1 on the validation set. Because
                      validation mirrors the train prior, this is not expected
                      to fully recover test F1 (the review's point).
  3. prior_matched  — threshold on validation set so the predicted positive rate
                      equals the known test prior π_test. Uses only the test
                      *prior* (a documented benchmark property), not test labels.
  4. test_max_f1    — ORACLE upper bound: threshold maximizing F1 directly on
                      test. Reported only to bound the achievable gain; never a
                      selection rule.

Requires per-pair predictions for valid and test (evaluate.py --save_preds).
Missing prediction files are generated automatically (inference only).

Usage:
    python src/baseline/select_threshold.py --dataset wdc-products --run baseline_cw
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "experiments" / "results"

from src.baseline.evaluate import evaluate


def _load_scores(run: str, dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (true_labels, match_scores); generate the preds file if missing."""
    path = RESULTS / f"{run}_{dataset}_{split}_preds.jsonl"
    if not path.exists():
        print(f"  [info] {path.name} missing — running evaluate(split={split}) to generate it")
        evaluate(dataset=dataset, split=split, run_name=run, save_preds=True)
    true, score = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            true.append(int(r["true_label"]))
            score.append(float(r["score"]))
    return np.array(true), np.array(score)


def _metrics_at(true: np.ndarray, score: np.ndarray, thr: float) -> dict:
    pred = (score >= thr).astype(int)
    return {
        "threshold": round(float(thr), 4),
        "precision": round(float(precision_score(true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(true, pred, zero_division=0)), 4),
    }


def _best_f1_threshold(true: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([score, [0.5]]))
    best_thr, best_f1 = 0.5, -1.0
    for thr in candidates:
        f1 = f1_score(true, (score >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def select(dataset: str, run: str) -> dict:
    valid_true, valid_score = _load_scores(run, dataset, "valid")
    test_true, test_score = _load_scores(run, dataset, "test")

    pi_test = float(np.mean(test_true))
    pi_valid = float(np.mean(valid_true))

    thr_valid = _best_f1_threshold(valid_true, valid_score)
    thr_prior = float(np.quantile(valid_score, 1 - pi_test))  # predict π_test fraction positive
    thr_oracle = _best_f1_threshold(test_true, test_score)

    report = {
        "dataset": dataset,
        "run": run,
        "prior_valid": round(pi_valid, 4),
        "prior_test": round(pi_test, 4),
        "operating_points": {
            "default": _metrics_at(test_true, test_score, 0.5),
            "valid_max_f1": {"chosen_on": "valid", "thr": round(thr_valid, 4), **_metrics_at(test_true, test_score, thr_valid)},
            "prior_matched": {"chosen_on": "valid@π_test", "thr": round(thr_prior, 4), **_metrics_at(test_true, test_score, thr_prior)},
            "test_max_f1_oracle": {"chosen_on": "test (oracle)", "thr": round(thr_oracle, 4), **_metrics_at(test_true, test_score, thr_oracle)},
        },
    }

    print(f"\n{'='*70}\n  Threshold calibration — {run} on {dataset}\n{'='*70}")
    print(f"  valid prior = {pi_valid:.3f}   test prior = {pi_test:.3f}")
    print(f"\n  {'operating point':<22} {'thr':>6} {'P':>8} {'R':>8} {'F1':>8}")
    print("  " + "-" * 56)
    default_f1 = report["operating_points"]["default"]["f1"]
    for name, op in report["operating_points"].items():
        thr = op["threshold"]
        delta = op["f1"] - default_f1
        tag = f"  (Δ {delta:+.4f})" if name != "default" else ""
        print(f"  {name:<22} {thr:>6.3f} {op['precision']:>8.4f} {op['recall']:>8.4f} {op['f1']:>8.4f}{tag}")

    out_path = RESULTS / f"threshold_calibration_{run}_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {out_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Calibrate decision threshold (Issue 4)")
    ap.add_argument("--dataset", required=True, choices=["wdc-products", "dblp-scholar"])
    ap.add_argument("--run", required=True, help="Run name (e.g. baseline_cw)")
    args = ap.parse_args()
    select(args.dataset, args.run)


if __name__ == "__main__":
    main()
