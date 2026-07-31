"""
Verify entity-disjointness of splits and augmented training sets.

Usage:
    python src/analysis/check_split_disjointness.py
    python src/analysis/check_split_disjointness.py --dataset dblp-scholar
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data_prep import entity_splits as es

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "experiments" / "results"

DATASETS = ["wdc-products", "dblp-scholar"]


# ---------------------------------------------------------------------------
# Split-level overlap
# ---------------------------------------------------------------------------


def split_overlap(dataset: str) -> dict:
    side = es._side_ids(dataset)
    train_l, train_r = side["train"]["left"], side["train"]["right"]
    test_l, test_r = side["test"]["left"], side["test"]["right"]
    valid_l, valid_r = side["valid"]["left"], side["valid"]["right"]

    def pct(a, b):
        return round(100 * len(a & b) / len(a), 1) if a else 0.0

    pairs = es.split_pairs(dataset)
    return {
        "shared_id_space": es.SHARED_ID_SPACE[dataset],
        "test_left_entities": len(test_l),
        "test_left_seen_in_train": len(test_l & train_l),
        "pct_test_left_seen_in_train": pct(test_l, train_l),
        "test_right_entities": len(test_r),
        "test_right_seen_in_train": len(test_r & train_r),
        "pct_test_right_seen_in_train": pct(test_r, train_r),
        "valid_left_seen_in_train": len(valid_l & train_l),
        "valid_right_seen_in_train": len(valid_r & train_r),
        "shared_pairs_train_test": len(pairs["train"] & pairs["test"]),
        "shared_pairs_train_valid": len(pairs["train"] & pairs["valid"]),
        "shared_pairs_test_valid": len(pairs["test"] & pairs["valid"]),
    }


# ---------------------------------------------------------------------------
# Augmented-file leakage
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _text_id_maps(dataset: str) -> tuple[dict, dict]:
    """Build {text: id} maps for left/right from the entity tables."""
    import csv

    def load(path: Path) -> dict:
        m = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    m[row["text"]] = int(row["id"])
        return m

    d = PROCESSED / dataset
    return load(d / "entities_left.csv"), load(d / "entities_right.csv")


def _count_touching(
    id_pairs: list[tuple[int | None, int | None]], s1: set, s2: set
) -> int:
    """Count pairs where id1∈s1 or id2∈s2."""
    n = 0
    for a, b in id_pairs:
        if (a is not None and a in s1) or (b is not None and b in s2):
            n += 1
    return n


def _leak_counts(
    dataset: str, id_pairs: list[tuple[int | None, int | None]], splits
) -> int:
    """Count pairs touching an entity from the given splits (context for Issue 5)."""
    ex1, ex2 = es.excluded_id_sets(dataset, splits=splits)
    return _count_touching(id_pairs, ex1, ex2)


def _heldout_counts(dataset: str, id_pairs: list[tuple[int | None, int | None]]) -> int:
    """
    Count pairs touching a genuinely held-out (non-train) entity. This is the
    canonical pass/fail metric: 0 means augmentation introduced no entity outside
    the train set, regardless of the benchmark's own (non-)disjointness.
    """
    held1, held2 = es.heldout_entity_ids(dataset)
    return _count_touching(id_pairs, held1, held2)


def augmented_leakage(dataset: str) -> dict:
    d = PROCESSED / dataset
    report = {}

    # --- LLM: seed + AL labeled pairs (the added pairs beyond train.txt) ---
    llm_records = _read_jsonl(d / "seed_labeled.jsonl") + _read_jsonl(
        d / "al_labeled.jsonl"
    )
    llm_ids = [(int(r["id1"]), int(r["id2"])) for r in llm_records]
    report["llm_aug"] = {
        "added_pairs": len(llm_ids),
        "touch_heldout": _heldout_counts(dataset, llm_ids),
        "touch_valid_or_test": _leak_counts(dataset, llm_ids, ("valid", "test")),
        "touch_test": _leak_counts(dataset, llm_ids, ("test",)),
    }

    # --- Web: relevant query entities (id, side) ---
    web_records = [r for r in _read_jsonl(d / "web_labeled.jsonl") if r.get("relevant")]
    web_ids: list[tuple[int | None, int | None]] = []
    for r in web_records:
        if r.get("side") == "left":
            web_ids.append((int(r["id"]), None))
        else:
            web_ids.append((None, int(r["id"])))
    report["web_aug"] = {
        "added_pairs": len(web_ids),
        "touch_heldout": _heldout_counts(dataset, web_ids),
        "touch_valid_or_test": _leak_counts(dataset, web_ids, ("valid", "test")),
        "touch_test": _leak_counts(dataset, web_ids, ("test",)),
    }

    # --- Train base (string aug inherits these entities, adds none of its own) ---
    left_map, right_map = _text_id_maps(dataset)
    base_ids: list[tuple[int | None, int | None]] = []
    train_path = d / "train.txt"
    if train_path.exists() and left_map and right_map:
        with open(train_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                base_ids.append((left_map.get(parts[0]), right_map.get(parts[1])))
        report["train_base"] = {
            "pairs": len(base_ids),
            "unmapped": sum(1 for a, b in base_ids if a is None or b is None),
            "touch_heldout": _heldout_counts(dataset, base_ids),
            "touch_valid_or_test": _leak_counts(dataset, base_ids, ("valid", "test")),
            "touch_test": _leak_counts(dataset, base_ids, ("test",)),
            "note": "string_aug only duplicates these entities; it adds none beyond train.txt",
        }
    else:
        report["train_base"] = {
            "note": "entities_{left,right}.csv not found — cannot map train base"
        }

    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def run(datasets: list[str]) -> dict:
    out = {}
    for ds in datasets:
        print(f"\n{'='*64}\n  {ds}\n{'='*64}")

        ov = split_overlap(ds)
        out.setdefault(ds, {})["split_overlap"] = ov
        print("Split overlap (Issue 5):")
        print(f"  shared id space            : {ov['shared_id_space']}")
        print(
            f"  test-left seen in train    : {ov['test_left_seen_in_train']}/{ov['test_left_entities']} ({ov['pct_test_left_seen_in_train']}%)"
        )
        print(
            f"  test-right seen in train   : {ov['test_right_seen_in_train']}/{ov['test_right_entities']} ({ov['pct_test_right_seen_in_train']}%)"
        )
        print(f"  shared exact pairs tr/te   : {ov['shared_pairs_train_test']}")
        print(f"  shared exact pairs tr/va   : {ov['shared_pairs_train_valid']}")

        leak = augmented_leakage(ds)
        out[ds]["augmented_leakage"] = leak
        print("\nAugmented-set leakage (Issue 1):")
        print("  PASS/FAIL = 'held-out' (entity NOT in train) must be 0;")
        print(
            "  'valid|test' is Issue-5 context (nonzero is expected & accepted on DBLP)."
        )
        all_clean = True
        for strat in ("train_base", "llm_aug", "web_aug"):
            r = leak.get(strat, {})
            denom = r.get("added_pairs") or r.get("pairs")
            if denom is None:
                continue
            held = r.get("touch_heldout", 0)
            vt = r.get("touch_valid_or_test", 0)
            if held > 0:
                all_clean = False
            flag = "✗ LEAK" if held > 0 else "✓"
            print(
                f"  {strat:11s}: held-out={held}  {flag}   |  valid|test={vt}/{denom}"
            )
        print(
            f"  => {'CLEAN — no held-out entities introduced' if all_clean else 'LEAKAGE PRESENT — regenerate'}"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "split_disjointness.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Check split / augmented-set entity disjointness"
    )
    ap.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    args = ap.parse_args()
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    run(datasets)


if __name__ == "__main__":
    main()
