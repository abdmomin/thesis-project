"""
Aggregate multi-seed runs into mean ± std and a seed-ensemble prediction file.

Companion to run_multiseed.py. For each base run, it:
  1. reports mean ± std of precision / recall / F1 across seeds (the headline
     deliverable: effects must exceed run-to-run noise to be credible);
  2. writes a seed-ensemble prediction file by averaging the per-pair match
     score across seeds and thresholding at 0.5, saved as
     "<run>_seedmean_<dataset>_test_preds.jsonl" so significance.py can run the
     paired McNemar / bootstrap-F1 tests on noise-averaged predictions.

Usage:
    python src/analysis/aggregate_seeds.py --dataset wdc-products \
        --runs baseline_cw string_aug_cw llm_aug_cw web_aug_cw --seeds 13 42 87

    # then, e.g.:
    python src/analysis/significance.py --dataset wdc-products \
        --baseline baseline_cw_seedmean --treatment string_aug_cw_seedmean
"""

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "experiments" / "results"

METRIC_KEYS = ["precision", "recall", "f1"]


def _load_metric_json(run: str, seed: int, dataset: str) -> dict | None:
    path = RESULTS / f"{run}_s{seed}_{dataset}_test.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["metrics"]


def _load_seed_preds(run: str, seed: int, dataset: str) -> list[dict] | None:
    path = RESULTS / f"{run}_s{seed}_{dataset}_test_preds.jsonl"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate_run(run: str, dataset: str, seeds: list[int]) -> dict:
    # --- metrics mean ± std ---
    per_seed = {k: [] for k in METRIC_KEYS}
    found = []
    for s in seeds:
        m = _load_metric_json(run, s, dataset)
        if m is None:
            print(f"  [warn] missing metrics for {run}_s{s}_{dataset}")
            continue
        found.append(s)
        for k in METRIC_KEYS:
            per_seed[k].append(m[k])

    summary = {"run": run, "dataset": dataset, "seeds": found}
    for k in METRIC_KEYS:
        vals = per_seed[k]
        summary[k] = {
            "mean": round(statistics.mean(vals), 4) if vals else None,
            # sample std (÷ n-1) to match how Aaron/DistillER report ± over runs
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "values": [round(v, 4) for v in vals],
        }

    # --- seed-ensemble predictions (average score, threshold 0.5) ---
    pred_sets = [(_load_seed_preds(run, s, dataset)) for s in seeds]
    pred_sets = [p for p in pred_sets if p is not None]
    if pred_sets:
        n = len(pred_sets[0])
        if all(len(p) == n for p in pred_sets):
            ens_path = RESULTS / f"{run}_seedmean_{dataset}_test_preds.jsonl"
            with open(ens_path, "w") as f:
                for i in range(n):
                    mean_score = sum(p[i]["score"] for p in pred_sets) / len(pred_sets)
                    rec = pred_sets[0][i]
                    f.write(
                        json.dumps(
                            {
                                "left": rec["left"],
                                "right": rec["right"],
                                "true_label": rec["true_label"],
                                "pred_label": int(mean_score >= 0.5),
                                "score": round(mean_score, 6),
                            }
                        )
                        + "\n"
                    )
            summary["seedmean_preds"] = str(ens_path.relative_to(ROOT))
            print(f"  wrote seed-ensemble preds → {ens_path.name}")
        else:
            print(
                f"  [warn] seed pred files for {run} differ in length — skipping ensemble"
            )

    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate multi-seed runs (mean ± std + ensemble preds)"
    )
    ap.add_argument(
        "--dataset", required=True, choices=["wdc-products", "dblp-scholar"]
    )
    ap.add_argument(
        "--runs", nargs="+", required=True, help="Base run names (no seed suffix)"
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 87])
    args = ap.parse_args()

    summaries = []
    print(
        f"\n{'='*70}\n  Seed aggregation — {args.dataset}  (seeds {args.seeds})\n{'='*70}"
    )
    for run in args.runs:
        print(f"\n{run}:")
        summaries.append(aggregate_run(run, args.dataset, args.seeds))

    print(f"\n{'Run':<18} {'Precision':>16} {'Recall':>16} {'F1 (mean ± std)':>18}")
    print("-" * 72)
    for s in summaries:
        if s["f1"]["mean"] is None:
            print(f"{s['run']:<18}  (no results found)")
            continue
        p, r, f = s["precision"], s["recall"], s["f1"]
        print(
            f"{s['run']:<18} {p['mean']:>7.4f} ± {p['std']:<6.4f} {r['mean']:>7.4f} ± {r['std']:<6.4f} {f['mean']:>7.4f} ± {f['std']:<6.4f}"
        )

    out_path = RESULTS / f"seed_aggregate_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
