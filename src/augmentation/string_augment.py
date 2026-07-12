"""
String-based data augmentation for entity matching (Task 3).

Ports the relevant operators from Ditto's augment.py into a clean, standalone
module that operates directly on COL/VAL serialized strings.

Supported operations:
  - 'del'      : delete a random span of 1–2 value tokens
  - 'swap'     : shuffle a random span of 2–4 value tokens
  - 'drop_col' : drop a random short attribute (COL ... VAL ...) entirely
  - 'all'      : apply 3 randomly chosen ops (RandAugment-style, as in Ditto)

All ops preserve COL/VAL header tokens — only value tokens ('O' label) are
modified, keeping the serialization format valid.

Usage (as a library):
    from src.augmentation.string_augment import augment_record, augment_dataset
"""

import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Token labelling
# ---------------------------------------------------------------------------

def _label_tokens(tokens: list[str]) -> list[str]:
    """Assign labels: 'HD' for COL/VAL headers, 'O' for value tokens."""
    labels = []
    for t in tokens:
        if t in ("COL", "VAL"):
            labels.append("HD")
        else:
            labels.append("O")
    return labels


def _sample_span(tokens: list[str], labels: list[str], span_len: int) -> tuple[int, int]:
    """Return (start, end) of a contiguous span of 'O' tokens, or (-1,-1)."""
    candidates = [
        (i, i + span_len - 1)
        for i in range(len(tokens))
        if i + span_len - 1 < len(tokens)
        and all(labels[i + k] == "O" for k in range(span_len))
    ]
    if not candidates:
        return -1, -1
    return random.choice(candidates)


# ---------------------------------------------------------------------------
# Individual ops (operate on token+label lists)
# ---------------------------------------------------------------------------

def _op_del(tokens: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
    span_len = random.randint(1, 2)
    p1, p2 = _sample_span(tokens, labels, span_len)
    if p1 < 0:
        return tokens, labels
    return tokens[:p1] + tokens[p2 + 1:], labels[:p1] + labels[p2 + 1:]


def _op_swap(tokens: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
    span_len = random.randint(2, 4)
    p1, p2 = _sample_span(tokens, labels, span_len)
    if p1 < 0:
        return tokens, labels
    sub = tokens[p1:p2 + 1]
    random.shuffle(sub)
    return tokens[:p1] + sub + tokens[p2 + 1:], labels[:]


def _op_drop_col(tokens: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
    """Drop one short attribute block (COL name VAL value...) at random."""
    col_starts = [i for i, t in enumerate(tokens) if t == "COL"]
    if len(col_starts) < 2:  # keep at least one attribute
        return tokens, labels

    # Compute end of each attribute block
    col_ends = []
    for i, start in enumerate(col_starts):
        end = col_starts[i + 1] - 1 if i + 1 < len(col_starts) else len(tokens) - 1
        col_ends.append(end)

    col_lens = [e - s + 1 for s, e in zip(col_starts, col_ends)]
    # Prefer dropping shorter attributes (less information loss)
    candidates = [i for i, le in enumerate(col_lens) if le <= 8]
    if not candidates:
        candidates = list(range(len(col_starts)))

    idx = random.choice(candidates)
    s, e = col_starts[idx], col_ends[idx]
    return tokens[:s] + tokens[e + 1:], labels[:s] + labels[e + 1:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_OPS = ["del", "swap", "drop_col"]


def augment_record(serialized: str, op: str = "all", seed: int | None = None) -> str:
    """
    Apply one augmentation operation to a COL/VAL serialized entity string.

    Args:
        serialized: A single entity string, e.g. "COL title VAL foo COL year VAL 2020"
        op: One of 'del', 'swap', 'drop_col', 'all'
        seed: Optional random seed for reproducibility

    Returns:
        Augmented string (same format).
    """
    if seed is not None:
        random.seed(seed)

    tokens = serialized.split(" ")
    labels = _label_tokens(tokens)

    if op == "all":
        for chosen_op in random.choices(_OPS, k=3):
            tokens, labels = _apply_op(tokens, labels, chosen_op)
    else:
        tokens, labels = _apply_op(tokens, labels, op)

    return " ".join(tokens)


def _apply_op(
    tokens: list[str], labels: list[str], op: str
) -> tuple[list[str], list[str]]:
    if op == "del":
        return _op_del(tokens, labels)
    elif op == "swap":
        return _op_swap(tokens, labels)
    elif op == "drop_col":
        return _op_drop_col(tokens, labels)
    return tokens, labels


def augment_dataset(
    input_file: Path,
    output_file: Path,
    multiplier: int = 1,
    op: str = "all",
    augment_matches_only: bool = False,
    seed: int = 42,
) -> int:
    """
    Read a Ditto-format train.txt, augment each pair, and write to output_file.

    The output contains the original pairs followed by `multiplier` augmented
    copies of each pair (both left and right entities are independently augmented).

    Args:
        input_file:          Source .txt file (tab-separated COL/VAL format)
        output_file:         Destination .txt file
        multiplier:          Number of augmented copies per original pair
        op:                  Augmentation op ('del', 'swap', 'drop_col', 'all')
        augment_matches_only: If True, only augment match pairs (label=1)
        seed:                Base random seed

    Returns:
        Total number of pairs written (original + augmented).
    """
    random.seed(seed)

    pairs: list[tuple[str, str, int]] = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            pairs.append((parts[0], parts[1], int(parts[2])))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(output_file, "w", encoding="utf-8") as f:
        # Write originals first
        for left, right, label in pairs:
            f.write(f"{left}\t{right}\t{label}\n")
            written += 1

        # Write augmented copies
        for copy_idx in range(multiplier):
            for left, right, label in pairs:
                if augment_matches_only and label == 0:
                    continue
                aug_left = augment_record(left, op=op)
                aug_right = augment_record(right, op=op)
                f.write(f"{aug_left}\t{aug_right}\t{label}\n")
                written += 1

    return written
