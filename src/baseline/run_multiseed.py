"""
Train + evaluate one configuration across multiple random seeds.

Addresses methodology_review.md Issue 3: every model was trained once at
seed=42, so reported deltas of ±0.005–0.012 F1 could not be distinguished from
run-to-run noise. This runner repeats a configuration over several seeds so the
downstream metrics can be reported as mean ± std (see aggregate_seeds.py).

It is a thin orchestrator over the existing train()/evaluate() entrypoints — no
training logic is duplicated. Each seed is saved under run name
"<run>_s<seed>", so evaluate.py / aggregate_seeds.py resolve them normally.

This is the *heavy* step (≈3× training per config). Run it on the regenerated,
entity-disjoint augmented data from notebooks 04/07 (Issue 1 fix).

Examples (run after the leakage fix; 8 configs total):
    # WDC Products (class-weighted)
    python src/baseline/run_multiseed.py --dataset wdc-products --run baseline_cw   --class_weight balanced
    python src/baseline/run_multiseed.py --dataset wdc-products --run string_aug_cw --class_weight balanced --train_file data/processed/wdc-products/train_aug_string.txt
    python src/baseline/run_multiseed.py --dataset wdc-products --run llm_aug_cw     --class_weight balanced --train_file data/processed/wdc-products/train_aug_llm.txt
    python src/baseline/run_multiseed.py --dataset wdc-products --run web_aug_cw     --class_weight balanced --train_file data/processed/wdc-products/train_aug_web.txt

    # DBLP-Scholar (no class weighting)
    python src/baseline/run_multiseed.py --dataset dblp-scholar --run baseline
    python src/baseline/run_multiseed.py --dataset dblp-scholar --run string_aug --train_file data/processed/dblp-scholar/train_aug_string.txt
    python src/baseline/run_multiseed.py --dataset dblp-scholar --run llm_aug    --train_file data/processed/dblp-scholar/train_aug_llm.txt
    python src/baseline/run_multiseed.py --dataset dblp-scholar --run web_aug    --train_file data/processed/dblp-scholar/train_aug_web.txt
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.baseline.train_baseline import train
from src.baseline.evaluate import evaluate

DEFAULT_SEEDS = [13, 42, 87]


def run_multiseed(
    dataset: str,
    run: str,
    seeds: list[int],
    class_weight: str = "none",
    train_file: str | None = None,
    epochs: int = 10,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 2e-5,   # final protocol (avoids the imbalanced-set collapse seen at 5e-5)
):
    tf = Path(train_file) if train_file else None
    for seed in seeds:
        run_name = f"{run}_s{seed}"
        print(f"\n{'#'*70}\n# {run_name} on {dataset}  (seed={seed})\n{'#'*70}")
        train(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            lr=lr,
            seed=seed,
            class_weight=class_weight,
            run_name=run_name,
            train_file=tf,
        )
        evaluate(dataset=dataset, split="test", run_name=run_name, save_preds=True)

    print(f"\nDone. Aggregate with:\n  python src/analysis/aggregate_seeds.py --dataset {dataset} --run {run} --seeds {' '.join(map(str, seeds))}")


def main():
    ap = argparse.ArgumentParser(description="Train+evaluate a config across multiple seeds")
    ap.add_argument("--dataset", required=True, choices=["wdc-products", "dblp-scholar"])
    ap.add_argument("--run", required=True, help="Base run name (seed suffix added automatically)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--class_weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    run_multiseed(
        dataset=args.dataset,
        run=args.run,
        seeds=args.seeds,
        class_weight=args.class_weight,
        train_file=args.train_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
