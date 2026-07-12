"""
Convert raw datasets into Ditto's COL/VAL format for entity matching.

Ditto serializes each record as:
    COL <attr_name> VAL <attr_value> COL <attr_name> VAL <attr_value> ...

Training/validation/test files are tab-separated with 3 columns:
    <left_entity>   <right_entity>   <label>

where label is 0 (non-match) or 1 (match).

Usage:
    python src/data_prep/preprocess.py --dataset wdc-products
    python src/data_prep/preprocess.py --dataset dblp-scholar
    python src/data_prep/preprocess.py --dataset all
"""

import argparse
import gzip
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Core serialization
# ---------------------------------------------------------------------------

def serialize_record(row: dict, columns: list[str]) -> str:
    """Serialize a record dict into Ditto's COL/VAL string format."""
    parts = []
    for col in columns:
        val = row.get(col, "")
        if val is None or (isinstance(val, float) and pd.isna(val)):
            val = ""
        val = str(val).strip()
        parts.append(f"COL {col} VAL {val}")
    return " ".join(parts)


def write_pairs(pairs: list[tuple[str, str, int]], out_path: Path):
    """Write (left, right, label) tuples to a tab-separated file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for left, right, label in pairs:
            f.write(f"{left}\t{right}\t{label}\n")
    print(f"  Wrote {len(pairs)} pairs → {out_path.name}")


# ---------------------------------------------------------------------------
# WDC Products
# ---------------------------------------------------------------------------

# Attributes used for serialization (both left and right)
WDC_COLS = ["brand", "title", "description", "price", "priceCurrency"]


def _wdc_left(record: dict) -> str:
    row = {col: record.get(f"{col}_left", "") for col in WDC_COLS}
    return serialize_record(row, WDC_COLS)


def _wdc_right(record: dict) -> str:
    row = {col: record.get(f"{col}_right", "") for col in WDC_COLS}
    return serialize_record(row, WDC_COLS)


def _read_wdc_gz(path: Path) -> list[tuple[str, str, int]]:
    pairs = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label = int(record.get("label", 0))
            pairs.append((_wdc_left(record), _wdc_right(record), label))
    return pairs


def preprocess_wdc_products():
    raw_dir = RAW / "wdc-products"
    out_dir = PROCESSED / "wdc-products"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Preprocessing WDC Products ===")

    split_files = {
        "train": raw_dir / "train_raw.json.gz",
        "valid": raw_dir / "valid_raw.json.gz",
        "test":  raw_dir / "test_raw.json.gz",
    }

    for split, gz_path in split_files.items():
        if not gz_path.exists():
            print(f"  [skip] {gz_path.name} not found — run download_datasets.py first")
            continue
        pairs = _read_wdc_gz(gz_path)
        write_pairs(pairs, out_dir / f"{split}.txt")
        _print_stats(pairs, split)

    print("WDC Products preprocessing complete.")


# ---------------------------------------------------------------------------
# DBLP-Scholar
# ---------------------------------------------------------------------------

DBLP_COLS = ["title", "authors", "venue", "year"]


def preprocess_dblp_scholar():
    raw_dir = RAW / "dblp-scholar"
    out_dir = PROCESSED / "dblp-scholar"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Preprocessing DBLP-Scholar ===")

    table_a_path = _find_file(raw_dir, "tableA.csv")
    table_b_path = _find_file(raw_dir, "tableB.csv")
    if table_a_path is None or table_b_path is None:
        print("  [error] tableA.csv or tableB.csv not found. Run download_datasets.py first.")
        return

    table_a = pd.read_csv(table_a_path)
    table_b = pd.read_csv(table_b_path)

    # Normalize column names to lowercase
    table_a.columns = [c.lower() for c in table_a.columns]
    table_b.columns = [c.lower() for c in table_b.columns]

    cols_a = [c for c in DBLP_COLS if c in table_a.columns]
    cols_b = [c for c in DBLP_COLS if c in table_b.columns]

    table_a = table_a.set_index("id")
    table_b = table_b.set_index("id")

    for split in ("train", "valid", "test"):
        split_path = _find_file(raw_dir, f"{split}.csv")
        if split_path is None:
            print(f"  [skip] {split}.csv not found")
            continue

        split_df = pd.read_csv(split_path)
        split_df.columns = [c.lower() for c in split_df.columns]

        pairs = []
        for _, row in split_df.iterrows():
            id_a = row["ltable_id"]
            id_b = row["rtable_id"]
            label = int(row["label"])

            rec_a = table_a.loc[id_a].to_dict() if id_a in table_a.index else {}
            rec_b = table_b.loc[id_b].to_dict() if id_b in table_b.index else {}

            left = serialize_record(rec_a, cols_a)
            right = serialize_record(rec_b, cols_b)
            pairs.append((left, right, label))

        write_pairs(pairs, out_dir / f"{split}.txt")
        _print_stats(pairs, split)

    print("DBLP-Scholar preprocessing complete.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_file(directory: Path, name: str) -> Path | None:
    for p in directory.rglob(name):
        return p
    return None


def _print_stats(pairs: list[tuple], split: str):
    total = len(pairs)
    matches = sum(1 for _, _, lbl in pairs if lbl == 1)
    non_matches = total - matches
    pct = 100 * matches / total if total else 0
    print(f"    {split}: {total:,} pairs  ({matches:,} matches / {non_matches:,} non-matches, {pct:.1f}% positive)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess entity matching datasets to Ditto format")
    parser.add_argument(
        "--dataset",
        choices=["wdc-products", "dblp-scholar", "all"],
        default="all",
        help="Which dataset to preprocess",
    )
    args = parser.parse_args()

    if args.dataset in ("wdc-products", "all"):
        preprocess_wdc_products()
    if args.dataset in ("dblp-scholar", "all"):
        preprocess_dblp_scholar()

    print("\nDone.")


if __name__ == "__main__":
    main()
