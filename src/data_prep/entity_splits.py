"""
Single source of truth for entity-to-split membership.

Training-quality require entity-disjoint splits:
no entity that appears in valid/test may be used to build augmented
training pairs. The LLM and web pipelines originally blocked
over the full entity universe and only excluded exact known pairs, so they
leaked valid/test entities into training. This module exposes the entity-ID
sets needed to enforce and verify disjointness.

Two id-space conventions matter:
  - WDC Products has a SINGLE shared id space: the same entity id can appear on
    the left of one pair and the right of another (self-pairs exist). So an
    excluded entity must be filtered from BOTH sides of a candidate pair.
  - DBLP-Scholar has TWO separate id spaces (tableA = left, tableB = right);
    both number from 0, so ids must be matched side-specifically or a left id
    would wrongly exclude an unrelated right entity.

Use `excluded_id_sets()` to get the correct (id1_exclude, id2_exclude) pair for
filtering a candidate pool regardless of dataset.
"""

import gzip
import json
from pathlib import Path

import pandas as pd


def _find_raw() -> Path:
    """
    Locate the data/raw directory. An env var DATA_RAW overrides everything.
    """
    import os

    if os.environ.get("DATA_RAW"):
        return Path(os.environ["DATA_RAW"]).resolve()
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "data" / "raw",  # repo: src/data_prep/..
        Path.cwd() / "data" / "raw",  # Colab flat: cwd == /content
        here.parent / "data" / "raw",  # file sits next to data/
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to the repo location; downstream existence checks raise clearly.
    return candidates[0]


RAW = _find_raw()

# Whether left/right entity ids share one universe (so an excluded entity must
# be removed from both sides of a candidate pair).
SHARED_ID_SPACE = {
    "wdc-products": True,
    "dblp-scholar": False,
}

SPLITS = ("train", "valid", "test")


# ---------------------------------------------------------------------------
# Per-side entity ids by split
# ---------------------------------------------------------------------------


def _wdc_side_ids() -> dict[str, dict[str, set[int]]]:
    """Return {split: {"left": set, "right": set}} of entity ids for WDC."""
    raw_dir = RAW / "wdc-products"
    files = {
        "train": raw_dir / "train_raw.json.gz",
        "valid": raw_dir / "valid_raw.json.gz",
        "test": raw_dir / "test_raw.json.gz",
    }
    out: dict[str, dict[str, set[int]]] = {}
    for split, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"WDC raw split missing: {path}. entity_splits needs data/raw/ to "
                "compute split membership — without it the entity-disjoint filter "
                "would silently no-op. Ensure data/raw is present in this environment."
            )
        left, right = set(), set()
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                left.add(int(rec["id_left"]))
                right.add(int(rec["id_right"]))
        out[split] = {"left": left, "right": right}
    return out


def _dblp_side_ids() -> dict[str, dict[str, set[int]]]:
    """Return {split: {"left": set, "right": set}} of entity ids for DBLP."""
    raw_dir = RAW / "dblp-scholar"
    out: dict[str, dict[str, set[int]]] = {}
    for split in SPLITS:
        path = raw_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"DBLP raw split missing: {path}. entity_splits needs data/raw/ to "
                "compute split membership — without it the entity-disjoint filter "
                "would silently no-op. Ensure data/raw is present in this environment."
            )
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        left = set(int(x) for x in df["ltable_id"].tolist())
        right = set(int(x) for x in df["rtable_id"].tolist())
        out[split] = {"left": left, "right": right}
    return out


def _side_ids(dataset: str) -> dict[str, dict[str, set[int]]]:
    if dataset == "wdc-products":
        return _wdc_side_ids()
    if dataset == "dblp-scholar":
        return _dblp_side_ids()
    raise ValueError(f"unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_entity_ids(dataset: str) -> tuple[set[int], set[int]]:
    """(left_ids, right_ids) appearing in the train split."""
    s = _side_ids(dataset)["train"]
    return s["left"], s["right"]


def valid_test_entity_ids(dataset: str) -> tuple[set[int], set[int]]:
    """(left_ids, right_ids) appearing in valid OR test."""
    s = _side_ids(dataset)
    left = s["valid"]["left"] | s["test"]["left"]
    right = s["valid"]["right"] | s["test"]["right"]
    return left, right


def train_keep_sets(dataset: str) -> tuple[set[int], set[int]]:
    """
    Return (keep_id1, keep_id2): the TRAIN entity-id sets a candidate pool may use.

    This is the policy filter for augmentation: keep only pairs where both sides are
    train entities. For WDC (entity-disjoint benchmark) this is equivalent to
    excluding valid/test, so augmentation is fully test-disjoint. For DBLP (whose
    standard split is NOT entity-disjoint) train entities also appear in
    valid/test, so this makes augmentation "as (non-)disjoint as the baseline" — it
    introduces no entity the baseline train set didn't already use. Verify with
    `heldout_entity_ids` (must touch 0 held-out entities).
    """
    s = _side_ids(dataset)
    left, right = s["train"]["left"], s["train"]["right"]
    if SHARED_ID_SPACE[dataset]:
        both = left | right
        return both, both
    return left, right


def heldout_entity_ids(dataset: str) -> tuple[set[int], set[int]]:
    """
    Return (held_id1, held_id2): entities that appear in valid/test but NOT in train.

    These are the only genuinely held-out entities — touching one in an augmented
    pair is real leakage regardless of dataset. (For WDC this equals all valid/test
    entities since the split is disjoint; for DBLP it is the small minority of
    valid/test entities absent from train.) The canonical pass/fail check.
    """
    s = _side_ids(dataset)
    tr_l, tr_r = s["train"]["left"], s["train"]["right"]
    vt_l = s["valid"]["left"] | s["test"]["left"]
    vt_r = s["valid"]["right"] | s["test"]["right"]
    if SHARED_ID_SPACE[dataset]:
        tr, vt = tr_l | tr_r, vt_l | vt_r
        held = vt - tr
        return held, held
    return vt_l - tr_l, vt_r - tr_r


def excluded_id_sets(
    dataset: str, splits: tuple[str, ...] = ("valid", "test")
) -> tuple[set[int], set[int]]:
    """
    Return (exclude_id1, exclude_id2): the entity-id sets to remove from the
    left (id1) and right (id2) of a candidate pool so the pool is disjoint from
    `splits`. Handles shared vs separate id spaces automatically.
    """
    s = _side_ids(dataset)
    left = set().union(*(s[sp]["left"] for sp in splits))
    right = set().union(*(s[sp]["right"] for sp in splits))
    if SHARED_ID_SPACE[dataset]:
        both = left | right
        return both, both
    return left, right


def entity_membership(dataset: str) -> dict[tuple[str, int], set[str]]:
    """
    Map (side, id) -> set of splits it appears in. `side` is "left"/"right";
    for WDC (shared space) an id may legitimately appear on both sides.
    Useful for reporting cross-split entity overlap.
    """
    s = _side_ids(dataset)
    membership: dict[tuple[str, int], set[str]] = {}
    for split in SPLITS:
        for side in ("left", "right"):
            for eid in s[split][side]:
                membership.setdefault((side, eid), set()).add(split)
    return membership


def split_pairs(dataset: str) -> dict[str, set[tuple[int, int]]]:
    """Return {split: set of (left_id, right_id)} for shared-pair analysis."""
    out: dict[str, set[tuple[int, int]]] = {}
    if dataset == "wdc-products":
        raw_dir = RAW / "wdc-products"
        files = {
            "train": raw_dir / "train_raw.json.gz",
            "valid": raw_dir / "valid_raw.json.gz",
            "test": raw_dir / "test_raw.json.gz",
        }
        for split, path in files.items():
            pairs = set()
            if path.exists():
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        pairs.add((int(rec["id_left"]), int(rec["id_right"])))
            out[split] = pairs
    elif dataset == "dblp-scholar":
        raw_dir = RAW / "dblp-scholar"
        for split in SPLITS:
            pairs = set()
            path = raw_dir / f"{split}.csv"
            if path.exists():
                df = pd.read_csv(path)
                df.columns = [c.lower() for c in df.columns]
                pairs = set(
                    (int(a), int(b)) for a, b in zip(df["ltable_id"], df["rtable_id"])
                )
            out[split] = pairs
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return out
