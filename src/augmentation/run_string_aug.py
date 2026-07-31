"""
CLI for string-based data augmentation (Task 3).

Generates an augmented training file using Ditto-style string operations
(token deletion, span shuffle, attribute drop) and writes it to
data/processed/<dataset>/train_aug_string.txt.

Usage:
    python src/augmentation/run_string_aug.py --dataset wdc-products
    python src/augmentation/run_string_aug.py --dataset dblp-scholar --multiplier 2
    python src/augmentation/run_string_aug.py --dataset wdc-products --op del --multiplier 1

Then retrain:
    python src/baseline/train_baseline.py \\
        --dataset wdc-products \\
        --class_weight balanced \\
        --run_name string_aug_cw \\
        --train_file data/processed/wdc-products/train_aug_string.txt
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.augmentation.string_augment import augment_dataset

PROCESSED = ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(
        description="Generate string-augmented training data (Task 3)"
    )
    parser.add_argument(
        "--dataset", required=True, choices=["wdc-products", "dblp-scholar"]
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=1,
        help="Number of augmented copies per original pair (default: 1)",
    )
    parser.add_argument(
        "--op",
        choices=["del", "swap", "drop_col", "all"],
        default="all",
        help="Augmentation op to apply (default: all — RandAugment-style)",
    )
    parser.add_argument(
        "--augment_matches_only",
        action="store_true",
        help="Only augment match pairs (label=1); leave non-matches as-is",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = PROCESSED / args.dataset
    input_file = data_dir / "train.txt"
    output_file = data_dir / "train_aug_string.txt"

    if not input_file.exists():
        sys.exit(f"[error] {input_file} not found. Run preprocess.py first.")

    print(f"\n=== String Augmentation: {args.dataset} ===")
    print(f"  Op         : {args.op}")
    print(f"  Multiplier : {args.multiplier}x")
    print(f"  Matches only: {args.augment_matches_only}")
    print(f"  Input      : {input_file}")
    print(f"  Output     : {output_file}")

    n = augment_dataset(
        input_file=input_file,
        output_file=output_file,
        multiplier=args.multiplier,
        op=args.op,
        augment_matches_only=args.augment_matches_only,
        seed=args.seed,
    )

    print(f"\n  Written {n:,} pairs to {output_file}")
    print("Done.")


if __name__ == "__main__":
    main()
