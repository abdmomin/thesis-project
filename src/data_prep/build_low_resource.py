"""
Build low-resource ablation training sets (Task 7 / B6).

Supervisor request: reduce the base training set (to 50% and 25%) and combine the smaller
base with the LLM-augmented pairs, to test whether the full base "already had sufficient
coverage" — i.e. whether LLM augmentation helps *more* when the base is small (Aaron's
Figure-4 / DistillER-D8 low-resource regime, where machine-labeled data matters most).

For each fraction f in {0.5, 0.25} it writes, per dataset:
    train_sub{f}.txt        — stratified f-subset of train.txt (base at reduced budget)
    train_sub{f}_llm.txt    — that subset + the full llm_aug *added* slice (seed+AL pairs)

The subset is drawn with a FIXED seed so it is identical across the 3 training seeds
(we vary model init, not the data). Class ratio is preserved (stratified by label).

Compare, at each budget: F1(train_sub{f}) vs F1(train_sub{f}_llm). If the augmentation gap is
positive at small budgets but ~0 at full budget, the base did NOT have sufficient coverage and
LLM augmentation helps in the low-resource regime.

Usage:
    python src/data_prep/build_low_resource.py --dataset wdc-products
    python src/data_prep/build_low_resource.py --dataset all
"""

import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = ROOT / "data" / "processed"

FRACTIONS = [0.5, 0.25]
SUBSET_SEED = 42  # fixed: the data subset is constant across the 3 model-init seeds


def _read(path: Path) -> list[str]:
    return [l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip()]


def _pos_rate(lines: list[str]) -> float:
    return 100 * sum(1 for l in lines if l.endswith("\t1")) / len(lines) if lines else 0.0


def _stratified_subset(lines: list[str], frac: float, rng: random.Random) -> list[str]:
    pos = [l for l in lines if l.endswith("\t1")]
    neg = [l for l in lines if l.endswith("\t0")]
    k_pos, k_neg = round(len(pos) * frac), round(len(neg) * frac)
    return rng.sample(pos, k_pos) + rng.sample(neg, k_neg)


def build(dataset: str):
    d = PROCESSED / dataset
    base = _read(d / "train.txt")
    base_set = set(base)
    llm_slice = [l for l in _read(d / "train_aug_llm.txt") if l not in base_set]  # seed+AL added pairs
    print(f"  base={len(base)} ({_pos_rate(base):.1f}% pos)   llm added slice={len(llm_slice)} ({_pos_rate(llm_slice):.1f}% pos)")

    for f in FRACTIONS:
        rng = random.Random(SUBSET_SEED)
        sub = _stratified_subset(base, f, rng)
        tag = f"sub{int(f*100)}"
        # base subset alone
        p_sub = d / f"train_{tag}.txt"
        with open(p_sub, "w", encoding="utf-8") as fh:
            fh.write("\n".join(sub) + "\n")
        # subset + full llm added slice
        combined = sub + llm_slice
        p_llm = d / f"train_{tag}_llm.txt"
        with open(p_llm, "w", encoding="utf-8") as fh:
            fh.write("\n".join(combined) + "\n")
        print(f"    {tag:6s}: base {len(sub):>5} ({_pos_rate(sub):.1f}% pos) → {p_sub.name}"
              f"   |   +llm {len(combined):>5} ({_pos_rate(combined):.1f}% pos) → {p_llm.name}")


def main():
    ap = argparse.ArgumentParser(description="Build low-resource ablation sets (B6)")
    ap.add_argument("--dataset", choices=["wdc-products", "dblp-scholar", "all"], default="all")
    args = ap.parse_args()
    datasets = ["wdc-products", "dblp-scholar"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print(f"\n=== {ds} ===")
        build(ds)


if __name__ == "__main__":
    main()
