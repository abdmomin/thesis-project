"""
Quantify LLM-teacher label quality against ground truth (Task 6 requirement / review #5).

The thesis is framed as teacher/student distillation (Claude = teacher labeler, roberta-base =
student). DistillER's whole analysis hinges on teacher label noise, and the supervisor explicitly
required a label-error check. For WDC Products we have `cluster_id` ground truth for *every*
entity, so we can compute the Claude teacher's exact accuracy / precision / recall / F1 and its
label-noise rate on all seed + active-learning labels — zero API calls.

(DBLP-Scholar has no per-entity cluster ground truth, so teacher quality there can only be
spot-checked qualitatively; the WDC table is the quantitative one.)

Usage:
    python src/analysis/teacher_label_quality.py
"""

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw" / "wdc-products"
PROCESSED = ROOT / "data" / "processed" / "wdc-products"
RESULTS = ROOT / "experiments" / "results"


def _cluster_map() -> dict[int, str]:
    """entity id -> cluster_id ground truth (same cluster == true match)."""
    cl = {}
    for gz in ("train_raw.json.gz", "valid_raw.json.gz", "test_raw.json.gz"):
        path = RAW / gz
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                cl[int(r["id_left"])] = r["cluster_id_left"]
                cl[int(r["id_right"])] = r["cluster_id_right"]
    return cl


def evaluate_teacher() -> dict:
    cl = _cluster_map()
    recs = []
    for fn in ("seed_labeled.jsonl", "al_labeled.jsonl"):
        p = PROCESSED / fn
        if p.exists():
            recs += [json.loads(l) for l in open(p)]

    tp = fp = fn_ = tn = unmapped = 0
    by_source = {}
    for r in recs:
        a, b = int(r["id1"]), int(r["id2"])
        if a not in cl or b not in cl:
            unmapped += 1
            continue
        true = 1 if cl[a] == cl[b] else 0
        pred = int(r["label"])
        cell = ("tp" if (pred, true) == (1, 1) else "fp" if (pred, true) == (1, 0)
                else "fn" if (pred, true) == (0, 1) else "tn")
        if cell == "tp": tp += 1
        elif cell == "fp": fp += 1
        elif cell == "fn": fn_ += 1
        else: tn += 1
        s = r.get("source", "?").split("_iter")[0]  # group al_iterN together
        by_source.setdefault(s, {"n": 0, "err": 0})
        by_source[s]["n"] += 1
        by_source[s]["err"] += int(pred != true)

    n = tp + fp + fn_ + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn_) if tp + fn_ else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    report = {
        "dataset": "wdc-products",
        "teacher_model": "claude-sonnet-4-6",
        "n_labeled": n,
        "unmapped": unmapped,
        "confusion": {"tp": tp, "fp": fp, "fn": fn_, "tn": tn},
        "accuracy": round((tp + tn) / n, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "true_positive_rate_in_pool": round((tp + fn_) / n, 4),
        "label_noise_rate": round((fp + fn_) / n, 4),
        "by_source": {s: {"n": v["n"], "noise": round(v["err"] / v["n"], 4)} for s, v in by_source.items()},
    }

    print(f"WDC Claude teacher vs cluster_id ground truth (n={n} labeled pairs):")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn_} TN={tn}")
    print(f"  Teacher accuracy = {report['accuracy']}  precision = {report['precision']}  "
          f"recall = {report['recall']}  F1 = {report['f1']}")
    print(f"  True positives in labeled pool: {tp + fn_} ({100*report['true_positive_rate_in_pool']:.1f}%)")
    print(f"  Overall label-noise rate: {100*report['label_noise_rate']:.1f}%  (FP={fp} false matches, FN={fn_} missed matches)")
    print("  Noise by stage:")
    for s, v in report["by_source"].items():
        print(f"    {s:10s}: n={v['n']:>4}  noise={100*v['noise']:.1f}%")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "teacher_label_quality_wdc.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {out}")
    return report


if __name__ == "__main__":
    evaluate_teacher()
