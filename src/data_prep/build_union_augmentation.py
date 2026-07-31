"""
Build union / combined augmentation training sets (Task 7 / B4 + B8).

The augmentation strategies barely overlap (Layer 2c, Jaccard ≈ 0), so
train the student on the UNION of their added pairs (and combinations) to test whether
complementary augmentation + more data improves downstream F1.

Each `train_aug_<strategy>.txt` = base `train.txt` + that strategy's *added* slice. This script
extracts each added slice (pairs beyond base), deduplicates across strategies, and writes the
requested unions on top of the same base:

    train_aug_union_all.txt          base + string + llm + web
    train_aug_union_string_llm.txt   base + string + llm
    train_aug_union_string_web.txt   base + string + web
    train_aug_union_llm_web.txt      base + llm + web

Usage:
    python src/data_prep/build_union_augmentation.py --dataset wdc-products
    python src/data_prep/build_union_augmentation.py --dataset all
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = ROOT / "data" / "processed"

STRATEGIES = {
    "string": "train_aug_string.txt",
    "llm": "train_aug_llm.txt",
    "web": "train_aug_web.txt",
}

COMBOS = {
    "all": ["string", "llm", "web"],
    "string_llm": ["string", "llm"],
    "string_web": ["string", "web"],
    "llm_web": ["llm", "web"],
}


def _read(path: Path) -> list[str]:
    return [l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip()]


def _pos_rate(lines: list[str]) -> float:
    pos = sum(1 for l in lines if l.endswith("\t1"))
    return 100 * pos / len(lines) if lines else 0.0


def build(dataset: str):
    d = PROCESSED / dataset
    base = _read(d / "train.txt")
    base_set = set(base)

    # Extract each strategy's added slice (pairs beyond base)
    added = {}
    for name, fn in STRATEGIES.items():
        path = d / fn
        if not path.exists():
            print(f"  [skip] {fn} missing")
            continue
        added[name] = [l for l in _read(path) if l not in base_set]
        print(
            f"  {name:6s} added slice: {len(added[name]):>6} pairs ({_pos_rate(added[name]):.1f}% pos)"
        )

    print(f"\n  {dataset}: base={len(base)} ({_pos_rate(base):.1f}% pos)")
    for combo, parts in COMBOS.items():
        if not all(p in added for p in parts):
            continue
        seen = set()
        union_added = []
        for p in parts:
            for l in added[p]:
                if l not in seen:
                    seen.add(l)
                    union_added.append(l)
        out_lines = base + union_added
        out_path = d / f"train_aug_union_{combo}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        print(
            f"    union_{combo:11s}: {len(out_lines):>6} pairs ({_pos_rate(out_lines):.1f}% pos)  → {out_path.name}"
        )


def main():
    ap = argparse.ArgumentParser(
        description="Build union augmentation training sets (B4/B8)"
    )
    ap.add_argument(
        "--dataset", choices=["wdc-products", "dblp-scholar", "all"], default="all"
    )
    args = ap.parse_args()
    datasets = (
        ["wdc-products", "dblp-scholar"] if args.dataset == "all" else [args.dataset]
    )
    for ds in datasets:
        print(f"\n=== {ds} ===")
        build(ds)


if __name__ == "__main__":
    main()
