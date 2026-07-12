"""
Reusable analysis utilities for Task 6 (and future runs).

Functions here are the canonical implementations shared across all augmentation
strategies and datasets. Extracted from notebooks 02/03/07 (canonical: notebook 07).
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Token-Jaccard similarity
# ---------------------------------------------------------------------------

_COL_VAL_RE = re.compile(r"COL\s+\S+\s+VAL\s*")


def value_tokens(col_val_str: str) -> set[str]:
    """Strip COL/VAL markers and return lowercased value token set."""
    text = _COL_VAL_RE.sub(" ", col_val_str)
    return set(text.lower().split())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_sim(left: str, right: str) -> float:
    return jaccard(value_tokens(left), value_tokens(right))


# ---------------------------------------------------------------------------
# WDC decisive-attribute patterns (WDC Products only)
# ---------------------------------------------------------------------------

ATTR_PATTERNS: dict[str, re.Pattern] = {
    "color": re.compile(
        r"\b(black|white|red|silver|blue|gold|grey|gray|green|yellow|pink)\b", re.I
    ),
    "memory/storage": re.compile(r"\d+\s*(gb|tb|mb)", re.I),
    "size": re.compile(r'\d+\s*(inch|in\b|"|cm|mm|\bl\b|xl\b|xs\b|sm\b)', re.I),
    "bundle": re.compile(r"\b(bundle|kit|set|pack)\b", re.I),
    "edition": re.compile(r"\b(edition|version|\bse\b|\ble\b|\bpro\b|\bplus\b)\b", re.I),
    "model number": re.compile(r"\b[A-Z]{1,4}\d{3,}\b"),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pairs(path: Path | str) -> list[tuple[str, str, int]]:
    """Read a Ditto COL/VAL .txt file into (left, right, label) tuples."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                pairs.append((parts[0], parts[1], int(parts[2])))
    return pairs


# ---------------------------------------------------------------------------
# Corner-case classification
# ---------------------------------------------------------------------------

# Thresholds kept identical to notebooks 02/03/07 for cross-strategy comparability.
HARD_POS_SIM_THRESHOLD = 0.3   # label=1 AND sim < this → hard positive
HARD_NEG_SIM_THRESHOLD = 0.4   # label=0 AND sim > this → hard negative


def corner_case_stats(pairs: list[tuple[str, str, int]]) -> dict:
    """
    Classify pairs into hard positives, hard negatives, and easy cases.

    Returns a dict with counts and percentages:
      hard_pos, hard_neg, total_corner, n, pct_hard_pos, pct_hard_neg, pct_corner
    """
    hard_pos = hard_neg = 0
    for left, right, label in pairs:
        sim = pair_sim(left, right)
        if label == 1 and sim < HARD_POS_SIM_THRESHOLD:
            hard_pos += 1
        elif label == 0 and sim > HARD_NEG_SIM_THRESHOLD:
            hard_neg += 1
    n = len(pairs)
    corner = hard_pos + hard_neg
    return {
        "n": n,
        "hard_pos": hard_pos,
        "hard_neg": hard_neg,
        "total_corner": corner,
        "pct_hard_pos": round(100 * hard_pos / n, 1) if n else 0.0,
        "pct_hard_neg": round(100 * hard_neg / n, 1) if n else 0.0,
        "pct_corner": round(100 * corner / n, 1) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Error classification for qualitative analysis (Layer 3)
# ---------------------------------------------------------------------------

# Min token count below which a side is considered "noisy/incomplete"
_MIN_TOKENS_FOR_COMPLETE = 6


def classify_error(
    left: str,
    right: str,
    true_label: int,
    pred_label: int,
    sim: float,
) -> str:
    """
    Assign a qualitative error class to a misclassified pair.

    Classes (matching CLAUDE.md error taxonomy):
      high_sim_non_match  — model predicted match, true non-match, high similarity
      low_sim_match       — model predicted non-match, true match, low similarity
      noisy_incomplete    — one side has very few tokens (incomplete/noisy description)
      ambiguous_variant   — everything else (borderline similarity, variant differences)
    """
    toks_l = value_tokens(left)
    toks_r = value_tokens(right)
    if len(toks_l) < _MIN_TOKENS_FOR_COMPLETE or len(toks_r) < _MIN_TOKENS_FOR_COMPLETE:
        return "noisy_incomplete"
    if pred_label == 1 and true_label == 0 and sim > 0.5:
        return "high_sim_non_match"
    if pred_label == 0 and true_label == 1 and sim < 0.2:
        return "low_sim_match"
    return "ambiguous_variant"
