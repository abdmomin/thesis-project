"""
Download benchmark datasets for entity matching experiments.

Datasets:
  - WDC Products (small, 80% corner cases) from data.dws.informatik.uni-mannheim.de
  - DBLP-Scholar from the Magellan data repository (via anhaidgroup/deepmatcher)

Usage:
    python src/data_prep/download_datasets.py --dataset all
    python src/data_prep/download_datasets.py --dataset wdc-products
    python src/data_prep/download_datasets.py --dataset dblp-scholar
"""

import argparse
import io
import gzip
import shutil
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Dataset URLs
# ---------------------------------------------------------------------------

# WDC Products 80% CC — one zip contains train/valid/gs files
WDC_80PAIR_URL = (
    "https://data.dws.informatik.uni-mannheim.de"
    "/largescaleproductcorpus/data/wdc-products/80pair.zip"
)

# DBLP-Scholar from Magellan / DeepMatcher repository (Structured version)
DBLP_SCHOLAR_URL = (
    "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-GoogleScholar/dblp_scholar_exp_data.zip"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_bytes(url: str, desc: str = "") -> bytes:
    print(f"  Downloading {desc or url} ...")
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    chunks = []
    total = 0
    for chunk in r.iter_content(65536):
        chunks.append(chunk)
        total += len(chunk)
    data = b"".join(chunks)
    print(f"  Downloaded {total / 1e6:.1f} MB")
    return data


def save_if_missing(path: Path, data: bytes):
    if path.exists():
        print(f"  [skip] {path.name} already exists")
    else:
        path.write_bytes(data)
        print(f"  Saved {path.name}")


def extract_gz_bytes(data: bytes, out_path: Path):
    """Decompress gzip bytes to a file."""
    if out_path.exists():
        print(f"  [skip] {out_path.name} already decompressed")
        return
    with gzip.open(io.BytesIO(data)) as f_in:
        out_path.write_bytes(f_in.read())
    print(f"  Decompressed to {out_path.name}")


# ---------------------------------------------------------------------------
# WDC Products (small, 80% CC)
# ---------------------------------------------------------------------------

# Files to extract from 80pair.zip and their output names
WDC_FILES = {
    "wdcproducts80cc20rnd000un_train_small.json.gz": "train_raw.json.gz",
    "wdcproducts80cc20rnd000un_valid_small.json.gz": "valid_raw.json.gz",
    "wdcproducts80cc20rnd000un_gs.json.gz":           "test_raw.json.gz",   # gold standard
}


def download_wdc_products():
    dest = RAW / "wdc-products"
    dest.mkdir(parents=True, exist_ok=True)

    print("\n=== WDC Products (small, 80% CC) ===")

    # Check if already done
    if all((dest / out_name).exists() for out_name in WDC_FILES.values()):
        print("  [skip] All WDC files already present")
        return

    zip_data = download_bytes(WDC_80PAIR_URL, "80pair.zip")
    zf = zipfile.ZipFile(io.BytesIO(zip_data))

    for zip_name, out_name in WDC_FILES.items():
        out_path = dest / out_name
        if out_path.exists():
            print(f"  [skip] {out_name} already exists")
            continue
        gz_data = zf.read(zip_name)
        out_path.write_bytes(gz_data)
        print(f"  Extracted {zip_name} → {out_name}")

    print("WDC Products download complete.")
    print(f"  Files: {[p.name for p in dest.iterdir()]}")


# ---------------------------------------------------------------------------
# DBLP-Scholar
# ---------------------------------------------------------------------------

def download_dblp_scholar():
    dest = RAW / "dblp-scholar"
    dest.mkdir(parents=True, exist_ok=True)

    print("\n=== DBLP-Scholar ===")

    # Check if already done
    if (dest / "tableA.csv").exists() and (dest / "tableB.csv").exists():
        print("  [skip] DBLP-Scholar files already present")
        return

    zip_data = download_bytes(DBLP_SCHOLAR_URL, "DBLP-GoogleScholar.zip")
    zf = zipfile.ZipFile(io.BytesIO(zip_data))

    for name in zf.namelist():
        # Flatten the directory structure
        fname = Path(name).name
        if not fname:
            continue
        out_path = dest / fname
        if out_path.exists():
            print(f"  [skip] {fname}")
            continue
        out_path.write_bytes(zf.read(name))
        print(f"  Extracted {fname}")

    print("DBLP-Scholar download complete.")
    print(f"  Files: {[p.name for p in sorted(dest.iterdir())]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download entity matching datasets")
    parser.add_argument(
        "--dataset",
        choices=["wdc-products", "dblp-scholar", "all"],
        default="all",
        help="Which dataset(s) to download",
    )
    args = parser.parse_args()

    if args.dataset in ("wdc-products", "all"):
        download_wdc_products()
    if args.dataset in ("dblp-scholar", "all"):
        download_dblp_scholar()

    print("\nDone.")


if __name__ == "__main__":
    main()
