"""
Task 6 — Three-Layer Analysis Computation Script.

Computes all three layers of evaluation for all 8 run/dataset combos and writes:
  - experiments/results/task6_dataset_characteristics.json  (Layer 1 + Layer 2)
  - experiments/results/task6_error_analysis.json           (Layer 3)

Usage:
    python src/analysis/task6_compute.py
"""

import json
import sys
import time
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.analysis.common import (
    ATTR_PATTERNS,
    classify_error,
    corner_case_stats,
    load_pairs,
    pair_sim,
    value_tokens,
)

RESULTS_DIR = ROOT / "experiments" / "results"
PROCESSED = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUNS_WDC = ["baseline_cw", "string_aug_cw", "llm_aug_cw", "web_aug_cw"]
RUNS_DBLP = ["baseline", "string_aug", "llm_aug", "web_aug"]

TRAIN_FILES = [
    "train.txt",
    "train_aug_string.txt",
    "train_aug_llm.txt",
    "train_aug_web.txt",
]
STRATEGY_LABELS = ["baseline", "string_aug", "llm_aug", "web_aug"]

LLM_MODEL = "claude-sonnet-4-6"
MAX_EXAMPLES_PER_CLASS = 3
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Layer 1 — compile metrics from existing result JSONs
# ---------------------------------------------------------------------------


def load_layer1() -> dict:
    layer1 = {}
    for dataset, runs in [("wdc-products", RUNS_WDC), ("dblp-scholar", RUNS_DBLP)]:
        layer1[dataset] = {}
        for run in runs:
            path = RESULTS_DIR / f"{run}_{dataset}_test.json"
            if path.exists():
                with open(path) as f:
                    layer1[dataset][run] = json.load(f)
            else:
                print(f"  [warn] missing: {path}")
    return layer1


# ---------------------------------------------------------------------------
# Layer 2 — dataset characteristics
# ---------------------------------------------------------------------------


def compute_layer2() -> dict:
    layer2 = {}
    for dataset in ["wdc-products", "dblp-scholar"]:
        data_dir = PROCESSED / dataset
        layer2[dataset] = {}

        # Per-train-file stats
        for fname, label in zip(TRAIN_FILES, STRATEGY_LABELS):
            fpath = data_dir / fname
            if not fpath.exists():
                print(f"  [skip] {fpath} not found")
                continue
            print(f"  Computing dataset chars: {dataset}/{fname} ...")
            pairs = load_pairs(fpath)
            n = len(pairs)
            n_pos = sum(1 for _, _, lb in pairs if lb == 1)
            n_neg = n - n_pos

            cc = corner_case_stats(pairs)

            # Attribute coverage (WDC only)
            attr_cov = {}
            if dataset == "wdc-products":
                for attr, pat in ATTR_PATTERNS.items():
                    hits = sum(1 for l, r, _ in pairs if pat.search(l) or pat.search(r))
                    attr_cov[attr] = {"hits": hits, "pct": round(100 * hits / n, 1)}

            layer2[dataset][label] = {
                "file": fname,
                "n": n,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "pct_pos": round(100 * n_pos / n, 1),
                **cc,
                "attr_coverage": attr_cov,
            }

        # Pairwise overlap between added-pair sets (string_aug, llm_aug, web_aug)
        base_pairs_path = data_dir / "train.txt"
        base_set = set()
        if base_pairs_path.exists():
            for l, r, lb in load_pairs(base_pairs_path):
                base_set.add((l, r, lb))

        added_sets: dict[str, set] = {}
        for fname, label in zip(TRAIN_FILES[1:], STRATEGY_LABELS[1:]):
            fpath = data_dir / fname
            if not fpath.exists():
                continue
            aug_set = set()
            for l, r, lb in load_pairs(fpath):
                aug_set.add((l, r, lb))
            added_sets[label] = aug_set - base_set

        overlap = {}
        labels_aug = list(added_sets.keys())
        for i in range(len(labels_aug)):
            for j in range(i + 1, len(labels_aug)):
                a, b = labels_aug[i], labels_aug[j]
                sa, sb = added_sets[a], added_sets[b]
                inter = len(sa & sb)
                union = len(sa | sb)
                jac = round(inter / union, 4) if union else 0.0
                overlap[f"{a}_vs_{b}"] = {
                    "intersection": inter,
                    "union": union,
                    "jaccard": jac,
                }
        layer2[dataset]["_overlap"] = overlap
        layer2[dataset]["_added_pair_counts"] = {
            k: len(v) for k, v in added_sets.items()
        }

    return layer2


# ---------------------------------------------------------------------------
# Layer 3 — qualitative error analysis
# ---------------------------------------------------------------------------


def _pick_representatives(errors: list[dict], n: int, seed: int) -> list[dict]:
    """Pick up to n representative examples by highest confidence mismatch."""
    # Sort by distance of score from 0.5 (most confident wrong predictions first)
    sorted_errs = sorted(errors, key=lambda x: abs(x["score"] - 0.5), reverse=True)
    return sorted_errs[:n]


def _llm_explain(client: anthropic.Anthropic, example: dict) -> str:
    """Return a 1-sentence explanation of why the model likely made this error."""
    true_label_str = "match" if example["true_label"] == 1 else "non-match"
    pred_label_str = "match" if example["pred_label"] == 1 else "non-match"
    prompt = (
        "You are analysing errors of a RoBERTa-based entity matching model.\n\n"
        f"LEFT:  {example['left'][:600]}\n\n"
        f"RIGHT: {example['right'][:600]}\n\n"
        f"True label: {true_label_str}   |   Model predicted: {pred_label_str}   "
        f"|   Token-Jaccard similarity: {example['sim']:.3f}\n\n"
        "In ONE sentence, explain the most likely reason this model prediction is wrong. "
        "Focus on the entity content (attributes, values, or missing information), not the model architecture."
    )
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=LLM_MODEL,
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            if attempt == 2:
                return f"[explanation failed: {type(e).__name__}]"
            time.sleep(2)
    return "[explanation failed]"


def compute_layer3() -> dict:
    client = anthropic.Anthropic()
    layer3 = {}

    run_dataset_pairs = [("wdc-products", run) for run in RUNS_WDC] + [
        ("dblp-scholar", run) for run in RUNS_DBLP
    ]

    for dataset, run in run_dataset_pairs:
        preds_path = RESULTS_DIR / f"{run}_{dataset}_test_preds.jsonl"
        if not preds_path.exists():
            print(f"  [skip] {preds_path} not found")
            continue

        print(f"\n  Analysing errors: {run} / {dataset}")
        preds = []
        with open(preds_path) as f:
            for line in f:
                preds.append(json.loads(line))

        # Classify every misclassified pair
        error_records: list[dict] = []
        for rec in preds:
            if rec["true_label"] == rec["pred_label"]:
                continue
            sim = pair_sim(rec["left"], rec["right"])
            err_class = classify_error(
                rec["left"], rec["right"], rec["true_label"], rec["pred_label"], sim
            )
            error_records.append(
                {**rec, "sim": round(sim, 4), "error_class": err_class}
            )

        # Count per class
        class_counts: dict[str, int] = {}
        class_errors: dict[str, list] = {}
        for err in error_records:
            cls = err["error_class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
            class_errors.setdefault(cls, []).append(err)

        n_errors = len(error_records)
        n_total = len(preds)
        print(f"    {n_errors}/{n_total} misclassified, classes: {class_counts}")

        # Pick representatives and get LLM explanations
        class_examples: dict[str, list] = {}
        for cls, errs in class_errors.items():
            reps = _pick_representatives(errs, MAX_EXAMPLES_PER_CLASS, RNG_SEED)
            examples_with_explanation = []
            for rep in reps:
                print(
                    f"    LLM explain: {cls} example (score={rep['score']:.3f}, sim={rep['sim']:.3f}) ..."
                )
                explanation = _llm_explain(client, rep)
                examples_with_explanation.append(
                    {
                        "left": rep["left"],
                        "right": rep["right"],
                        "true_label": rep["true_label"],
                        "pred_label": rep["pred_label"],
                        "score": rep["score"],
                        "sim": rep["sim"],
                        "llm_explanation": explanation,
                    }
                )
                time.sleep(0.3)
            class_examples[cls] = examples_with_explanation

        layer3[f"{run}_{dataset}"] = {
            "run": run,
            "dataset": dataset,
            "n_total": n_total,
            "n_errors": n_errors,
            "error_rate": round(n_errors / n_total, 4),
            "class_counts": class_counts,
            "class_pct": {
                cls: round(100 * cnt / n_errors, 1) if n_errors else 0.0
                for cls, cnt in class_counts.items()
            },
            "representative_examples": class_examples,
        }

    return layer3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=== Task 6: Three-Layer Analysis ===\n")

    print("--- Layer 1: compiling metrics ---")
    layer1 = load_layer1()

    print("\n--- Layer 2: dataset characteristics ---")
    layer2 = compute_layer2()

    # Save Layer 1 + 2
    out_chars = RESULTS_DIR / "task6_dataset_characteristics.json"
    with open(out_chars, "w") as f:
        json.dump({"layer1": layer1, "layer2": layer2}, f, indent=2)
    print(f"\n  Saved -> {out_chars}")

    print("\n--- Layer 3: error analysis + LLM explanations ---")
    layer3 = compute_layer3()

    out_errors = RESULTS_DIR / "task6_error_analysis.json"
    with open(out_errors, "w") as f:
        json.dump(layer3, f, indent=2)
    print(f"\n  Saved -> {out_errors}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
