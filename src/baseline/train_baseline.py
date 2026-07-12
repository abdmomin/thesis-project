"""
Fine-tune a RoBERTa-base model as a binary entity matching classifier.

This is the baseline model (Task 2). It is trained only on the original seed
training data without any augmentation, establishing a performance floor that
all augmentation strategies (Tasks 3–5) are compared against.

Architecture mirrors Ditto:
  - Model  : roberta-base
  - Input  : [CLS] left_entity [SEP] right_entity [SEP]
  - Output : binary classification (0 = non-match, 1 = match)
  - Loss   : cross-entropy (optionally class-weighted)

Usage:
    python src/baseline/train_baseline.py --dataset wdc-products
    python src/baseline/train_baseline.py --dataset dblp-scholar --epochs 15
    python src/baseline/train_baseline.py --dataset wdc-products --class_weight balanced --run_name baseline_cw
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.metrics import compute_metrics

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "experiments" / "models"

MODEL_NAME = "roberta-base"
MAX_LENGTH = 256  # max tokens per pair


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pairs(path: Path) -> tuple[list[str], list[str], list[int]]:
    """Read a Ditto-format .txt file into (lefts, rights, labels)."""
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
    path: Path,
    tokenizer,
    max_length: int = MAX_LENGTH,
) -> Dataset:
    lefts, rights, labels = load_pairs(path)

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


# ---------------------------------------------------------------------------
# HuggingFace compute_metrics callback
# ---------------------------------------------------------------------------


def make_compute_metrics_fn():
    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1).tolist()
        labels = labels.tolist()
        return compute_metrics(labels, preds)

    return _compute


# ---------------------------------------------------------------------------
# Class-weighted Trainer
# ---------------------------------------------------------------------------


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class weights to the cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss = torch.nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Inverse-frequency weights: w_c = n_total / (n_classes * n_c)."""
    counts = [labels.count(0), labels.count(1)]
    n = len(labels)
    weights = [n / (2 * c) if c > 0 else 1.0 for c in counts]
    print(f"  Class weights: non-match={weights[0]:.3f}, match={weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    dataset: str,
    epochs: int = 10,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 5e-5,
    seed: int = 42,
    class_weight: str = "none",
    run_name: str = "baseline",
    train_file: Path | None = None,
):
    data_dir = PROCESSED / dataset
    if train_file is None:
        train_file = data_dir / "train.txt"
    valid_file = data_dir / "valid.txt"

    if not train_file.exists():
        sys.exit(
            f"[error] {train_file} not found. Run src/data_prep/preprocess.py first."
        )

    model_out = MODELS / f"{run_name}_{dataset}"
    model_out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Training {run_name} on {dataset} ===")
    print(f"  Model        : {MODEL_NAME}")
    print(f"  Train file   : {train_file}")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch size   : {batch_size}  (grad_accum={grad_accum}, effective={batch_size*grad_accum})")
    print(f"  LR           : {lr}")
    print(f"  Class weight : {class_weight}")
    print(f"  Output dir   : {model_out}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    print("  Tokenizing training set...")
    train_ds = build_hf_dataset(train_file, tokenizer)

    eval_ds = None
    if valid_file.exists():
        print("  Tokenizing validation set...")
        eval_ds = build_hf_dataset(valid_file, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(model_out),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch" if eval_ds else "epoch",
        load_best_model_at_end=True if eval_ds else False,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if eval_ds else []

    if class_weight == "balanced":
        _, _, train_labels = load_pairs(train_file)
        weights = compute_class_weights(train_labels)
        trainer = WeightedTrainer(
            class_weights=weights,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=make_compute_metrics_fn(),
            callbacks=callbacks,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=make_compute_metrics_fn(),
            callbacks=callbacks,
        )

    trainer.train()

    # Save final best model + tokenizer
    trainer.save_model(str(model_out / "best"))
    tokenizer.save_pretrained(str(model_out / "best"))
    print(f"\n  Best model saved to {model_out / 'best'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train RoBERTa baseline for entity matching"
    )
    parser.add_argument(
        "--dataset", required=True, choices=["wdc-products", "dblp-scholar"]
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class_weight",
        choices=["none", "balanced"],
        default="none",
        help="Use 'balanced' to apply inverse-frequency class weights to the loss",
    )
    parser.add_argument(
        "--run_name",
        default="baseline",
        help="Name prefix for the saved model directory (default: baseline)",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        help="Override the default train.txt path (for augmented training sets)",
    )
    args = parser.parse_args()

    train(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        seed=args.seed,
        class_weight=args.class_weight,
        run_name=args.run_name,
        train_file=Path(args.train_file) if args.train_file else None,
    )


if __name__ == "__main__":
    main()
