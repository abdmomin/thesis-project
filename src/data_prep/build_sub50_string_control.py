"""
Build the sub50 + string-slice CONTROL set (Task 7, WDC only).

Purpose — isolate label signal from data volume at the 50% budget.
The key positive result is:
    sub50            F1 = 0.606
    sub50 + llm      F1 = 0.626   (+0.020, McNemar p=0.0002)
Is the +0.020 from the LLM's genuine new label signal, or just from adding 2,100 more training
pairs (volume)? String augmentation transforms EXISTING pairs and keeps their label, so it adds
volume with NO new label information. It is the perfect control.

Design (matched to the llm slice so the ONLY difference is information content):
  - added-slice SIZE      = 2,100 pairs (identical to the llm slice)
  - added-slice BALANCE   = 217 pos / 1,883 neg (identical to the llm slice → identical class
                            weights under `_cw`)
  - added-slice SOURCE    = the sub50 base ONLY (1,250 pairs) — string-augmented copies, so the
                            control uses exactly the same labeled information as sub50, inflated to
                            the same volume as sub50_llm. No leakage from the withheld 50%.

Output: data/processed/wdc-products/train_sub50_string.txt   (3,350 pairs, 13.9% pos)

Interpretation after training (3 seeds, run_multiseed → sub50_string_cw):
  - sub50_string gain < +0.019  → LLM label signal beats pure volume  → headline STRENGTHENS
  - sub50_string gain ≈ +0.020  → it was volume, not signal → reframe to "volume below the
                                   coverage point"

Usage:
    python src/data_prep/build_sub50_string_control.py
"""

import random
from pathlib import Path

from src.augmentation.string_augment import augment_record

ROOT = Path(__file__).resolve().parent.parent.parent
WDC = ROOT / "data" / "processed" / "wdc-products"
SEED = 42  # fixed: the control slice is constant across the 3 model-init training seeds


def _read(path: Path) -> list[tuple[str, str, int]]:
    rows = []
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 3:
            rows.append((parts[0], parts[1], int(parts[2])))
    return rows


def _slice_composition(base: set, aug_file: Path) -> tuple[int, int]:
    """Return (n_pos, n_neg) of the pairs ADDED beyond the base in aug_file."""
    pos = neg = 0
    for l, r, y in _read(aug_file):
        if (l, r, y) not in base:
            if y == 1:
                pos += 1
            else:
                neg += 1
    return pos, neg


def _augment_n(
    sources: list[tuple[str, str, int]], n: int, rng: random.Random
) -> list[tuple[str, str, int]]:
    """Produce exactly n string-augmented copies drawn (with repetition if needed) from sources.
    Each copy augments both sides with op='all' (RandAugment-style, same as train_aug_string).
    """
    out = []
    i = 0
    order = sources[:]
    rng.shuffle(order)
    while len(out) < n:
        if i >= len(order):
            rng.shuffle(order)
            i = 0
        left, right, label = order[i]
        i += 1
        aug_l = augment_record(left, op="all")
        aug_r = augment_record(right, op="all")
        out.append((aug_l, aug_r, label))
    return out[:n]


def main():
    base = _read(WDC / "train_sub50.txt")
    base_set = set(base)
    # Match the llm slice exactly (size + class balance) so the only variable is label info.
    n_pos, n_neg = _slice_composition(base_set, WDC / "train_sub50_llm.txt")
    print(
        f"sub50 base: {len(base)} pairs "
        f"({sum(1 for *_, y in base if y==1)} pos / {sum(1 for *_, y in base if y==0)} neg)"
    )
    print(f"llm slice to match: {n_pos+n_neg} pairs ({n_pos} pos / {n_neg} neg)")

    pos_src = [row for row in base if row[2] == 1]
    neg_src = [row for row in base if row[2] == 0]

    # Set the module-level RNG deterministically for augment_record's internal ops.
    random.seed(SEED)
    rng = random.Random(SEED)
    aug_pos = _augment_n(pos_src, n_pos, rng)
    aug_neg = _augment_n(neg_src, n_neg, rng)

    combined = base + aug_pos + aug_neg
    out = WDC / "train_sub50_string.txt"
    with open(out, "w", encoding="utf-8") as fh:
        for l, r, y in combined:
            fh.write(f"{l}\t{r}\t{y}\n")

    tot = len(combined)
    tp = sum(1 for *_, y in combined if y == 1)
    print(f"\nwrote {out.name}: {tot} pairs ({tp} pos, {100*tp/tot:.1f}%)")
    print(
        f"  parallels train_sub50_llm.txt (3350 pairs, 13.9% pos) — matched size & balance"
    )


if __name__ == "__main__":
    main()
