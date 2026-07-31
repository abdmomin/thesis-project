"""
Mine "same-title, different-publication" hard negatives for DBLP-Scholar (Task 7 / B3).

This script explicitly adds hard negatives such as papers with the same
title first published at a workshop/conference and later in a journal — authors and title
are nearly identical, only venue and year differ. Adds them and checks whether they turn the
(near-ceiling) DBLP results positive.

Two safe sources (both yield GUARANTEED non-matches, no LLM verification needed):

  1. Conf/journal cross-pairs (within-DBLP). DBLP (tableA) has authoritative venue/year, so
     two DBLP records with the same title but different venue/year are different publications
     (a conference and its journal version). For versions v1, v2 with known Scholar matches
     M1, M2, the cross-pair (v1_DBLP, M2_Scholar) is a near-identical-title NON-match.

  2. Cross-table title near-duplicates with venue AND year both differing. A DBLP↔Scholar
     pair with high title overlap that is not a known match AND differs in both venue and
     year is very likely a different publication (a same-title pair sharing venue+year would
     be a missed match, so we require both to differ to stay safe).

Both sources are restricted to TRAIN entities (entity-disjoint policy, see entity_splits).

Structural finding this exposes: DBLP-Scholar has only ~100–150 such clean hard negatives,
because titles are near-unique per paper — this is exactly why the base training set has ~0
hard negatives (Layer 2) and why the dataset sits at the F1 ceiling.

Usage:
    python src/data_prep/mine_dblp_hard_negatives.py                 # mine + write files
    python src/data_prep/mine_dblp_hard_negatives.py --oversample 10 # replicate hard negs 10x
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data_prep.preprocess import serialize_record, DBLP_COLS

RAW = ROOT / "data" / "raw" / "dblp-scholar"
PROCESSED = ROOT / "data" / "processed" / "dblp-scholar"

OVERLAP_THRESHOLD = 0.75  # cross-table title overlap-coefficient
MIN_TITLE_TOKENS = 5


def _toks(t: str) -> set[str]:
    return set(
        w for w in re.sub(r"[^a-z0-9 ]", " ", str(t).lower()).split() if len(w) > 2
    )


def _load():
    A = pd.read_csv(RAW / "tableA.csv")
    A.columns = [c.lower() for c in A.columns]
    B = pd.read_csv(RAW / "tableB.csv")
    B.columns = [c.lower() for c in B.columns]
    known = set()
    train_pos = defaultdict(set)  # dblp_id -> {scholar match ids in train}
    train_dblp, train_schol = set(), set()
    for sp in ("train", "valid", "test"):
        d = pd.read_csv(RAW / f"{sp}.csv")
        d.columns = [c.lower() for c in d.columns]
        for a, b, l in zip(d.ltable_id, d.rtable_id, d.label):
            known.add((int(a), int(b)))
            if sp == "train":
                train_dblp.add(int(a))
                train_schol.add(int(b))
                if int(l) == 1:
                    train_pos[int(a)].add(int(b))
    return A, B, known, train_pos, train_dblp, train_schol


def _text(row, cols) -> str:
    return serialize_record(row.to_dict(), [c for c in cols if c in row.index])


def mine(overlap_threshold: float = OVERLAP_THRESHOLD) -> list[dict]:
    A, B, known, train_pos, train_dblp, train_schol = _load()
    A["tt"] = A.title.map(_toks)
    B["tt"] = B.title.map(_toks)
    Arows = {int(r.id): r for _, r in A.iterrows()}
    Brows = {int(r.id): r for _, r in B.iterrows()}
    hard = {}  # (dblp_id, scholar_id) -> source

    # --- Source 1: conf/journal cross-pairs ---
    groups = defaultdict(list)
    for _, a in A.iterrows():
        if len(a.tt) >= 4 and int(a.id) in train_dblp:
            groups[" ".join(sorted(a.tt))].append(a)
    for g in groups.values():
        if len(g) < 2:
            continue
        venues = {str(x.venue).strip().lower() for x in g}
        years = {str(x.year).strip() for x in g}
        if len(venues) <= 1 and len(years) <= 1:
            continue  # same venue+year → not a distinct publication
        for vi in g:
            for vj in g:
                if int(vi.id) == int(vj.id):
                    continue
                for sch in train_pos[int(vj.id)]:
                    if sch not in train_pos[int(vi.id)]:
                        hard[(int(vi.id), sch)] = "conf_journal"

    # --- Source 2: cross-table title near-dup, venue AND year both differ ---
    inv = defaultdict(list)
    for bid, ts in zip(B.id, B.tt):
        for w in ts:
            inv[w].append(int(bid))
    common = {w for w, l in inv.items() if len(l) > 2000}
    for _, a in A.iterrows():
        if len(a.tt) < MIN_TITLE_TOKENS or int(a.id) not in train_dblp:
            continue
        cand = set()
        for w in a.tt:
            if w not in common:
                cand.update(inv[w])
        for bid in cand:
            if bid not in train_schol or (int(a.id), bid) in known:
                continue
            b = Brows[bid]
            m = min(len(a.tt), len(b.tt))
            if m < MIN_TITLE_TOKENS:
                continue
            if len(a.tt & b.tt) / m < overlap_threshold:
                continue
            av, bv = str(a.venue).strip().lower(), str(b.venue).strip().lower()
            ay, by = str(a.year).strip(), str(b.year).strip()
            venue_known = av not in ("", "nan") and bv not in ("", "nan")
            year_known = ay not in ("", "nan") and by not in ("", "nan")
            if venue_known and year_known and av != bv and ay != by:
                hard.setdefault((int(a.id), bid), "title_dup_venue_year")

    # Serialize
    cols = [c for c in DBLP_COLS if c in A.columns]
    out = []
    for (aid, bid), src in hard.items():
        out.append(
            {
                "id1": aid,
                "id2": bid,
                "source": src,
                "label": 0,
                "left_text": _text(Arows[aid], cols),
                "right_text": _text(Brows[bid], cols),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Mine DBLP-Scholar same-title hard negatives (B3)"
    )
    ap.add_argument("--overlap-threshold", type=float, default=OVERLAP_THRESHOLD)
    ap.add_argument(
        "--oversample", type=int, default=1, help="Replicate each hard negative N times"
    )
    args = ap.parse_args()

    hard = mine(args.overlap_threshold)
    by_src = defaultdict(int)
    for h in hard:
        by_src[h["source"]] += 1

    print(f"Mined {len(hard)} clean hard negatives:")
    for s, n in by_src.items():
        print(f"  {s}: {n}")

    hn_path = PROCESSED / "hard_negatives_dblp.jsonl"
    with open(hn_path, "w") as f:
        for h in hard:
            f.write(json.dumps(h) + "\n")
    print(f"Saved provenance → {hn_path}")

    # Build augmented training file
    train_lines = [
        l for l in open(PROCESSED / "train.txt", encoding="utf-8") if l.strip()
    ]
    out_path = PROCESSED / "train_aug_dblp_hardneg.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for l in train_lines:
            f.write(l if l.endswith("\n") else l + "\n")
        for _ in range(args.oversample):
            for h in hard:
                f.write(f"{h['left_text']}\t{h['right_text']}\t0\n")
    total = len(train_lines) + len(hard) * args.oversample
    print(
        f"Saved {out_path.name}: {len(train_lines)} base + {len(hard)*args.oversample} hard negs (x{args.oversample}) = {total} pairs"
    )

    print("\nExamples:")
    for h in hard[:4]:
        print(f"  [{h['source']}] {h['left_text'][:70]}")
        print(f"                 {h['right_text'][:70]}")


if __name__ == "__main__":
    main()
