# Progress Report — Task 6 Complete

**To:** Prof. Christian Bizer  
**From:** Abdullah Al Momin  
**Date:** 2026-06-25  
**Subject:** Entity Matching Augmentation — Task 6 Results + Task 7 Direction

---

Dear Professor Bizer,

Task 6 (evaluation & analysis) is complete. All numbers below are **final**, computed on
**entity-disjoint** data after a methodology audit (summarised in §0) and verified with paired
significance testing. Below are the roberta-base results for all four strategies, the DistillER
comparison, the key finding, and the proposed Task 7 direction.

---

## 0. Methodology corrections applied (audit, resolved)

An internal code/methodology audit found and fixed six issues; all results below already reflect
the fixes (details + code in `thesis-docs/task6_findings.md`, "Methodology Corrections"):

1. **Entity leakage fixed.** The LLM/web candidate pools were built over the full entity
   universe, so 35–78% of the *added* `llm_aug`/`web_aug` pairs included held-out valid/test
   entities. The pipelines now use **train-only** entities; `check_split_disjointness.py`
   confirms **0** held-out entities introduced. `llm_aug`/`web_aug` were regenerated and retrained.
2. **Paired McNemar test** (+ bootstrap-F1 CI + Cohen's h) replaces the earlier unpaired
   accuracy z-test, which was the wrong test for two models on the same test set.
3. **Multi-seed done (3 seeds, lr=2e-5).** Each config retrained over seeds 13/42/87 and reported
   as mean ± std. This overturned two single-seed WDC conclusions (string_aug "significant" and
   llm_aug "significant worse" were seed artifacts) — see §1. Only lr changed (5e-5→2e-5) to avoid
   a training-instability collapse on the imbalanced `llm_aug` set.
4. **WDC ceiling is largely a prior-shift artifact** (train/valid 20% positive vs test 11%); no
   selectable threshold recovers more than ~+0.018 F1.
5. **DBLP split is not entity-disjoint** (93% of test entities appear in train) — the 0.958
   baseline is partly memorisation; accepted and documented.
6. **Corner-case target (40–50%) not met** — best 31.7% (WDC) / 4.4% (DBLP).

---

## 1. Downstream Results — All Strategies (final: 3-seed mean ± std, lr=2e-5)

F1 = mean ± std over seeds 13/42/87; Δ = vs baseline; McNemar on the seed-ensemble predictions.

**WDC Products** (test n=4,500 · 500 matches · 4,000 non-matches)

F1 = mean ± sample std (3 seeds). McNemar on seed-averaged (ensemble) predictions.

| Run | F1 (mean ± std) | ΔF1 | beats base | McNemar p | Verdict |
|-----|-----------------|-----|-----------|-----------|---------|
| `baseline_cw` | 0.659 ± 0.015 | — | — | — | — |
| `string_aug_cw` | **0.675 ± 0.016** | +0.016 | 2/3 | 0.244 | n.s. (best mean) |
| `llm_aug_cw` | 0.661 ± 0.020 | +0.002 | 2/3 | 0.356 | n.s. (tie) |
| `web_aug_cw` | 0.658 ± 0.015 | −0.001 | 2/3 | 0.396 | n.s. (tie) |

**DBLP-Scholar** (test n=5,742 · 1,070 matches · 4,672 non-matches)

| Run | F1 (mean ± std) | ΔF1 | beats base | McNemar p | Verdict |
|-----|-----------------|-----|-----------|-----------|---------|
| `baseline` | 0.9559 ± 0.0019 | — | — | — | — |
| `web_aug` | 0.9567 ± 0.0005 | +0.001 | 2/3 | 0.935 | n.s. (tie) |
| `llm_aug` | 0.9566 ± 0.0006 | +0.001 | 2/3 | 0.728 | n.s. (tie) |
| `string_aug` | 0.9483 ± 0.0020 | −0.008 | 0/3 | 0.0079 | worse (tiny, nominal) |

**Summary (seed-robust):** Under 3-seed evaluation, **no augmentation strategy robustly beats the
baseline on WDC** — `string_aug` has the best mean (+0.016) but is within the seed-noise band and
not significant (it loses on 1 of 3 seeds); `llm_aug` and `web_aug` tie the baseline. On DBLP
(near-ceiling), only `string_aug` reliably (but negligibly) hurts; `llm_aug`/`web_aug` tie.
**Important:** the single-seed results I sent earlier overstated the effects — WDC `string_aug`
"significant +0.035" and WDC `llm_aug` "significant −0.027" were **seed artifacts** that
disappear under proper multi-seed testing (the WDC baseline is itself high-variance across seeds).

---

## 2. Key finding — the LLM augmentation's value was a leakage artifact

The most important result of the audit: once leakage is removed, the entity-disjoint LLM
candidate pool is **intrinsically positive-starved**. Of the 4,722 WDC train-only candidate
pairs only **266 (5.6%) are true matches** (verified against `cluster_id`); DBLP's pool yields
**~0** discoverable new positives. On a closed benchmark the true matches are already in the
labeled set and are excluded as known pairs, so blocking surfaces almost no *new* positives.

The previous `llm_aug` result (25% positive) was therefore driven by the leaked valid/test
entities. Clean, `llm_aug` is only 15.6% positive and (over 3 seeds) **ties** the baseline — it
adds no lift. This means the supervisor's 25%-positive / 40–50% corner-case targets are
**structurally unreachable** for entity-disjoint LLM blocking-augmentation on these benchmarks —
a finding in itself. We accepted and documented this rather than re-introduce leakage.

---

## 3. DistillER comparison (Zeakis et al., 2026)

- **DBLP-Scholar — same direction.** DistillER's D9 shows RoBERTa F1 0.89 (GT) vs 0.86 (LLM
  labels); we see baseline 0.958 with all augmentation ≤ baseline. LLM-labeled data does not help
  RoBERTa on DBLP-Scholar in either study (near-ceiling, no headroom).
- **Product datasets are hardest.** DistillER's D2/D3/D8 are their hardest (high token overlap
  between non-matching products); our WDC ceiling (~0.61–0.68) vs DBLP (~0.95) mirrors this, and
  our dominant WDC error class is exactly the shared-token / subtle-variant confusion they describe.
- **Data ratio (nuanced).** DistillER's 3:1 ratio target matches a labeled baseline. We find the
  honest entity-disjoint LLM pool **cannot reach** that ratio (only 5.6% positives exist), and the
  web slice overshoots it (81% positive). Ratio control is exactly the Task 7 lever (§4).

---

## 4. Proposed Task 7 direction — rebalanced web augmentation (WDC)

**Why web, not string or LLM:** over 3 seeds **no strategy robustly beats the WDC baseline**, so
Task 7 aims to *find* one. `string_aug` has the best mean but is n.s. and offers no LLM/web
research question; `llm_aug` is positive-starved and a dead end on a closed benchmark; **`web_aug`
is the only strategy that introduces genuinely new positive entities** (via retrieval). Its added
slice is badly skewed (**81% positive / 68% corner**), so rebalancing **plus injecting generated
hard negatives** is the clearest lever to turn a tie into a robust gain.

**Experiment (`web_v2`, notebook 07b + resampling):**
1. Generate hard negatives — ask an LLM for similar-but-different products, retrieve their offers
   via Tavily, pair with the original (per your suggestion).
2. Downsample web positives so the added slice is ≈1:3 positive; also produce a hard-neg-heavy
   variant. → `train_aug_web_v2_{balanced,hardneg}.txt`.
3. Retrain (class-weighted) **over 3 seeds** and compare to `baseline_cw` and `string_aug_cw`.

**Hypothesis:** removing the positive skew and adding hard negatives cuts the false-positive surge
and converts the current baseline-tie into a robust gain (exceeding `string_aug_cw`'s best mean
0.675) while improving corner-case coverage.

**Prerequisite:** run the 3-seed confirmation of the current four configs first (the web_aug gain
is currently within noise). I would value your feedback on this direction before running Task 7.

Full analysis, error examples, and the DistillER detail are in `thesis-docs/task6_findings.md`.

Best regards,  
Abdullah
