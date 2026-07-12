"""
Evaluate a trained entity matching model on the test set.

Loads the best checkpoint produced by train_baseline.py, runs inference on
the test split, and writes precision / recall / F1 + confusion matrix to
experiments/results/<run>_<dataset>_<split>.json.

Usage:
    python src/baseline/evaluate.py --dataset wdc-products
    python src/baseline/evaluate.py --dataset dblp-scholar
    python src/baseline/evaluate.py --dataset wdc-products --split valid
    python src/baseline/evaluate.py --dataset wdc-products --run baseline_cw
    python src/baseline/evaluate.py --dataset wdc-products --run baseline_cw --save_preds
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.metrics import compute_metrics, full_report, confusion

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "experiments" / "models"
RESULTS = ROOT / "experiments" / "results"

MAX_LENGTH = 256


def load_pairs(path: Path) -> tuple[list[str], list[str], list[int]]:
    lefts, rights, labels = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            lefts.append(parts[0])
            rights.append(parts[1])
            labels.append(int(parts[2]))
    return lefts, rights, labels


def build_hf_dataset(
    lefts: list[str],
    rights: list[str],
    labels: list[int],
    tokenizer,
    max_length: int = MAX_LENGTH,
) -> Dataset:
    encodings = tokenizer(
        lefts,
        rights,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )
    encodings["labels"] = labels
    return Dataset.from_dict(encodings)


def evaluate(
    dataset: str,
    split: str = "test",
    run_name: str = "baseline",
    save_preds: bool = False,
    threshold: float | None = None,
):
    model_dir = MODELS / f"{run_name}_{dataset}" / "best"
    data_file = PROCESSED / dataset / f"{split}.txt"

    if not model_dir.exists():
        sys.exit(
            f"[error] Model not found at {model_dir}. "
            "Run src/baseline/train_baseline.py first."
        )
    if not data_file.exists():
        sys.exit(
            f"[error] {data_file} not found. "
            "Run src/data_prep/preprocess.py first."
        )

    print(f"\n=== Evaluating {run_name} on {dataset} [{split}] ===")

    lefts, rights, labels = load_pairs(data_file)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    ds = build_hf_dataset(lefts, rights, labels, tokenizer)

    eval_args = TrainingArguments(
        output_dir=str(MODELS / "tmp_eval"),
        per_device_eval_batch_size=64,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=eval_args)

    output = trainer.predict(ds)
    logits = output.predictions

    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    scores = (exp / exp.sum(axis=-1, keepdims=True))[:, 1].tolist()

    if threshold is None:
        # Default: argmax over logits (equivalent to a 0.5 cut on the softmax).
        preds = np.argmax(logits, axis=-1).tolist()
    else:
        # Operating-point calibration (Issue 4): cut the match-score at `threshold`.
        preds = [int(s >= threshold) for s in scores]

    metrics = compute_metrics(labels, preds)
    cm = confusion(labels, preds)

    print(f"\n  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"\n{full_report(labels, preds)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    result = {
        "run": run_name,
        "dataset": dataset,
        "split": split,
        "model": str(model_dir),
        "threshold": threshold if threshold is not None else "argmax",
        "metrics": metrics,
        "confusion_matrix": cm,
        "n_pairs": len(labels),
        "n_matches": sum(labels),
        "n_non_matches": len(labels) - sum(labels),
    }
    # Tag thresholded runs so they don't overwrite the canonical argmax results.
    thr_tag = "" if threshold is None else f"_thr{threshold:.2f}"
    out_file = RESULTS / f"{run_name}_{dataset}_{split}{thr_tag}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {out_file}")

    if save_preds:
        preds_file = RESULTS / f"{run_name}_{dataset}_{split}{thr_tag}_preds.jsonl"
        with open(preds_file, "w") as f:
            for left, right, true_label, pred_label, score in zip(lefts, rights, labels, preds, scores):
                f.write(json.dumps({
                    "left": left,
                    "right": right,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "score": round(score, 6),
                }) + "\n")
        print(f"  Predictions saved to {preds_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate entity matching model")
    parser.add_argument("--dataset", required=True, choices=["wdc-products", "dblp-scholar"])
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument(
        "--run",
        default="baseline",
        help="Model run name (matches the prefix used in train_baseline.py)",
    )
    parser.add_argument(
        "--save_preds",
        action="store_true",
        help="Save per-pair predictions to <run>_<dataset>_<split>_preds.jsonl",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Match-score decision threshold (default: argmax / 0.5). "
        "Use a calibrated value from select_threshold.py to address the WDC "
        "train/test prior mismatch (Issue 4).",
    )
    args = parser.parse_args()

    evaluate(
        dataset=args.dataset,
        split=args.split,
        run_name=args.run,
        save_preds=args.save_preds,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
