"""
Paired significance testing for model-vs-model comparisons on a shared test set.

Replaces the unpaired two-proportion z-test on accuracy used in earlier Task 6
reporting (methodology_review.md, Issue 2). Because every model is evaluated on
the *same* test pairs, comparisons are paired:

  - McNemar's test (exact binomial + continuity-corrected χ²) on the discordant
    predictions — the correct paired test for two classifiers on one test set.
  - Bootstrap confidence interval on F1 (and on the paired F1 difference), since
    accuracy is a weak proxy for F1 under class imbalance (WDC test is 11% pos).
  - Cohen's h effect size on accuracy (CLAUDE.md requirement), so statistical
    significance is always reported next to practical significance.

Reads the per-pair prediction files written by evaluate.py --save_preds:
    experiments/results/<run>_<dataset>_test_preds.jsonl
with fields: true_label, pred_label, score.

Usage:
    # one comparison
    python src/analysis/significance.py --dataset wdc-products \
        --baseline baseline_cw --treatment string_aug_cw

    # reproduce the full Task 6 comparison table
    python src/analysis/significance.py --all
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "experiments" / "results"

# Canonical run names per dataset (WDC uses class-weighted variants).
STANDARD_RUNS = {
    "wdc-products": {
        "baseline": "baseline_cw",
        "treatments": ["string_aug_cw", "llm_aug_cw", "web_aug_cw"],
    },
    "dblp-scholar": {
        "baseline": "baseline",
        "treatments": ["string_aug", "llm_aug", "web_aug"],
    },
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_preds(run: str, dataset: str, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
    """Return (true_labels, pred_labels) from a *_preds.jsonl file."""
    path = RESULTS / f"{run}_{dataset}_{split}_preds.jsonl"
    if not path.exists():
        sys.exit(f"[error] predictions not found: {path}")
    true, pred = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            true.append(int(r["true_label"]))
            pred.append(int(r["pred_label"]))
    return np.array(true), np.array(pred)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def mcnemar(true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """
    Paired McNemar test on correctness of A vs B over identical test items.

    n10 = # items A correct & B wrong; n01 = # items A wrong & B correct.
    Returns continuity-corrected χ² p-value and the exact two-sided binomial
    p-value (preferred when n10+n01 is small).
    """
    a_correct = pred_a == true
    b_correct = pred_b == true
    n10 = int(np.sum(a_correct & ~b_correct))
    n01 = int(np.sum(~a_correct & b_correct))
    disc = n10 + n01

    if disc == 0:
        return {"n10": 0, "n01": 0, "discordant": 0, "chi2": 0.0, "p_chi2": 1.0, "p_exact": 1.0}

    chi2 = (abs(n10 - n01) - 1) ** 2 / disc
    p_chi2 = float(stats.chi2.sf(chi2, df=1))
    # Exact two-sided binomial on the smaller discordant count.
    p_exact = float(stats.binomtest(min(n10, n01), n=disc, p=0.5, alternative="two-sided").pvalue)
    return {
        "n10": n10,
        "n01": n01,
        "discordant": disc,
        "chi2": round(chi2, 4),
        "p_chi2": p_chi2,
        "p_exact": p_exact,
    }


def f1_of(true: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(true, pred, zero_division=0))


def accuracy_of(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(true == pred))


def bootstrap_f1_ci(
    true: np.ndarray, pred: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Percentile bootstrap CI on F1 for a single run."""
    rng = np.random.default_rng(seed)
    n = len(true)
    f1s = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        f1s[i] = f1_score(true[idx], pred[idx], zero_division=0)
    lo, hi = np.percentile(f1s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"f1": round(f1_of(true, pred), 4), "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4)}


def bootstrap_f1_diff_ci(
    true: np.ndarray,
    pred_base: np.ndarray,
    pred_treat: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    Paired bootstrap CI on F1(treatment) - F1(baseline). Same resampled indices
    are applied to both runs to preserve pairing. Two-sided p approximated from
    the bootstrap distribution's mass on the opposite side of zero.
    """
    rng = np.random.default_rng(seed)
    n = len(true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        t = true[idx]
        diffs[i] = f1_score(t, pred_treat[idx], zero_division=0) - f1_score(
            t, pred_base[idx], zero_division=0
        )
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = f1_of(true, pred_treat) - f1_of(true, pred_base)
    frac_le0 = float(np.mean(diffs <= 0))
    frac_ge0 = float(np.mean(diffs >= 0))
    p_boot = min(1.0, 2 * min(frac_le0, frac_ge0))
    return {
        "f1_diff": round(point, 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "p_boot": p_boot,
    }


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size between two proportions."""
    phi = lambda p: 2 * math.asin(math.sqrt(max(0.0, min(1.0, p))))
    return abs(phi(p1) - phi(p2))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def compare(
    dataset: str, baseline: str, treatment: str, n_boot: int = 10000, seed: int = 0, quiet: bool = False
) -> dict:
    true_b, pred_b = load_preds(baseline, dataset)
    true_t, pred_t = load_preds(treatment, dataset)
    if not np.array_equal(true_b, true_t):
        sys.exit(
            f"[error] test labels differ between {baseline} and {treatment} — not the same test set."
        )
    true = true_b

    mc = mcnemar(true, pred_b, pred_t)
    f1_base = bootstrap_f1_ci(true, pred_b, n_boot, seed=seed)
    f1_treat = bootstrap_f1_ci(true, pred_t, n_boot, seed=seed)
    diff = bootstrap_f1_diff_ci(true, pred_b, pred_t, n_boot, seed=seed)
    h = cohens_h(accuracy_of(true, pred_t), accuracy_of(true, pred_b))

    direction = "better" if diff["f1_diff"] > 0 else "worse"
    sig = mc["p_exact"] < 0.05
    verdict = f"{'SIGNIFICANT' if sig else 'n.s.'} ({direction})"

    result = {
        "dataset": dataset,
        "baseline": baseline,
        "treatment": treatment,
        "f1_baseline": f1_base,
        "f1_treatment": f1_treat,
        "f1_diff": diff,
        "mcnemar": mc,
        "cohens_h_accuracy": round(h, 4),
        "significant": sig,
        "verdict": verdict,
    }

    if not quiet:
        print(f"\n{'='*64}\n  {dataset}: {treatment} vs {baseline}\n{'='*64}")
        print(f"  F1 baseline   : {f1_base['f1']:.4f}  [{f1_base['ci_lo']:.4f}, {f1_base['ci_hi']:.4f}]")
        print(f"  F1 treatment  : {f1_treat['f1']:.4f}  [{f1_treat['ci_lo']:.4f}, {f1_treat['ci_hi']:.4f}]")
        print(f"  F1 difference : {diff['f1_diff']:+.4f}  [{diff['ci_lo']:+.4f}, {diff['ci_hi']:+.4f}]  (bootstrap p={diff['p_boot']:.4f})")
        print(f"  McNemar       : n10={mc['n10']} n01={mc['n01']}  χ²(cc)={mc['chi2']}  p_exact={mc['p_exact']:.4f}  p_χ²={mc['p_chi2']:.4f}")
        print(f"  Cohen's h     : {h:.4f} (on accuracy)")
        print(f"  Verdict       : {verdict}  [α=0.05, exact McNemar]")
    return result


def run_all(n_boot: int = 10000, seed: int = 0, seedmean: bool = True) -> list[dict]:
    # FINAL protocol = 3-seed seed-ensemble predictions (run aggregate_seeds.py first).
    # seedmean=True appends "_seedmean" to every run name; set False to reproduce the
    # (superseded) single-seed table.
    sfx = "_seedmean" if seedmean else ""
    rows = []
    for dataset, cfg in STANDARD_RUNS.items():
        for treat in cfg["treatments"]:
            rows.append(compare(dataset, cfg["baseline"] + sfx, treat + sfx,
                                 n_boot=n_boot, seed=seed, quiet=True))

    print(f"\n{'Dataset':<13} {'Comparison':<28} {'ΔF1':>8} {'McNemar p':>11}  Verdict")
    print("-" * 78)
    for r in rows:
        comp = f"{r['treatment']} vs {r['baseline']}"
        print(f"{r['dataset']:<13} {comp:<28} {r['f1_diff']['f1_diff']:>+8.4f} {r['mcnemar']['p_exact']:>11.4f}  {r['verdict']}")

    out_path = RESULTS / "significance_tests.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved → {out_path}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Paired significance tests (McNemar + bootstrap F1)")
    ap.add_argument("--dataset", choices=list(STANDARD_RUNS.keys()))
    ap.add_argument("--baseline")
    ap.add_argument("--treatment")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="Run the standard Task 6 comparison table (3-seed seed-ensemble by default)")
    ap.add_argument("--single-seed", action="store_true", help="With --all: use single-seed run names instead of _seedmean (reproduces the superseded table)")
    args = ap.parse_args()

    if args.all:
        run_all(n_boot=args.n_boot, seed=args.seed, seedmean=not args.single_seed)
    else:
        if not (args.dataset and args.baseline and args.treatment):
            ap.error("provide --dataset --baseline --treatment, or use --all")
        compare(args.dataset, args.baseline, args.treatment, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
