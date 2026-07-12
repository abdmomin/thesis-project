# Task 6 — Evaluation Findings

**Author:** Abdullah Al Momin  
**Date:** 2026-06-17  
**Datasets:** WDC Products (small / 80% CC) · DBLP-Scholar (citation matching)  
**Model:** `roberta-base` fine-tuned as binary sequence-pair classifier (Ditto architecture)  

Reusable computation code: `src/analysis/common.py` · `src/analysis/task6_compute.py`  
Analysis notebook: `notebooks/08_task6_analysis.ipynb`  
Raw JSON outputs: `experiments/results/task6_dataset_characteristics.json` · `task6_error_analysis.json`

---

## Methodology Corrections (2026-06-21)

An independent audit (`thesis-docs/methodology_review.md`) identified six issues.
This section records the corrections; downstream tables below are annotated where a
claim was superseded. Supporting code: `src/analysis/significance.py`,
`src/analysis/check_split_disjointness.py`, `src/baseline/select_threshold.py`,
`src/baseline/run_multiseed.py`. Outputs: `experiments/results/significance_tests.json`,
`split_disjointness.json`, `threshold_calibration_*.json`.

### C1 — Entity leakage fixed in the LLM/web pipelines (was critical)
The candidate pool (notebook 04) and web sampling (notebook 07) blocked over the
**full** entity universe, so 35–78% of the *added* `llm_aug`/`web_aug` pairs paired in
held-out valid/test entities (verified with `check_split_disjointness.py`):

| Added pairs touching a **test** entity | WDC Products | DBLP-Scholar |
|---|---|---|
| `train_aug_llm.txt` (seed+AL) | 1,235 / 2,100 | 1,645 / 2,100 |
| `train_aug_web.txt` (relevant) | 280 / 799 | 271 / 699 |

Both notebooks now apply a **train-only entity filter** (`src/data_prep/entity_splits.py`,
`train_keep_sets`): the candidate pool keeps only pairs whose entities are both train
entities. On WDC (disjoint benchmark) this is fully test-disjoint; on DBLP (non-disjoint
benchmark, C5) it is "as disjoint as the baseline". **RESOLVED (2026-06-25):** `llm_aug`/`web_aug`
were regenerated on the clean pool and retrained; `check_split_disjointness.py` confirms
**held-out (non-train) entities = 0** for both on both datasets. All Layer 1/2/3 tables below
now show the **final clean numbers**. As predicted, `llm_aug` dropped on WDC. `baseline` and
`string_aug` were unaffected.

**C1a — Positive yield collapses once leakage is removed (key finding).** After the fix, the
clean LLM candidate pool is intrinsically almost all negative: of the 4,722 WDC train-only
candidate pairs only **266 (5.6%) are true matches** (verified against WDC `cluster_id`
ground truth), and the DBLP seed step found **0** discoverable positives in its top-100. This
is structural: on a closed benchmark essentially every true match is already in the labeled
set and is excluded as a "known pair", so blocking surfaces almost no *new* positives. The
original `llm_aug`'s ~25% positive rate was therefore largely a **leakage artifact** — the
held-out valid/test entities supplied the positives. Consequence: the supervisor's 25%-positive
/ 40–50% corner-case targets (C6) are **structurally unreachable** for entity-disjoint LLM
blocking-augmentation on these datasets. Per decision (2026-06), we **accept and document**
this rather than re-introduce leakage or oversample; the regenerated `llm_aug` set is honestly
negative-heavy and is expected to provide little or no lift. **The DBLP AL slice is 0/2,085 =
0.0% positive** (the headline 16.6% is entirely the base set) — the positive-scarcity finding at
its starkest.

**C1b — Teacher label quality (supervisor requirement; `teacher_label_quality.py`).** In the
distillation framing the Claude model is the *teacher*. On WDC we have `cluster_id` ground truth
for every candidate, so the teacher's labels can be scored exactly (zero API). Over all 2,100
seed+AL labels:

| Teacher (Claude) vs `cluster_id` | value |
|---|---|
| accuracy | 0.971 |
| precision / recall / F1 (match class) | 0.866 / 0.858 / **0.862** |
| overall label-noise rate | **2.9%** (29 false matches, 31 missed matches) |
| true positives in labeled pool | 219 / 2,100 (10.4%) |
| noise by stage | seed 0.0% · active-learning 3.0% |

The teacher is a **good but imperfect** labeler (F1 0.862, ~3% noise, concentrated on the hard
AL-selected pairs). Crucially the teacher's *labeling* F1 (0.862) far exceeds the student's
*downstream* F1 (~0.66 on WDC), so the augmentation bottleneck is **not teacher noise** — it is
the positive scarcity (only 10.4% of labeled pairs are true matches) and the intrinsic task
difficulty. This mirrors DistillER's teacher-noise analysis and supports the augmentation-vs-
replacement framing. (DBLP has no per-entity ground truth → qualitative spot-check only.)

### C2 — Significance test corrected: unpaired z-test → paired McNemar
Both models in any comparison are evaluated on the *same* test pairs, so the paired
**McNemar test** is correct, not the unpaired two-proportion z-test on accuracy. F1 is also
reported with a bootstrap CI (10k resamples) and the F1-difference CI; Cohen's *h* (on
accuracy) is the effect size. Recomputed on the current predictions:

**FINAL — 3-seed results (mean ± std over seeds 13/42/87; lr=2e-5).** These supersede the earlier
single-seed (5e-5) numbers wholesale (`seed_aggregate_*.json`; McNemar on the seed-ensemble preds):

| Comparison | F1 baseline | F1 treatment | McNemar p | beats base | Verdict |
|---|---|---|---|---|---|
| WDC `string_aug_cw` vs `baseline_cw` | 0.659 ± 0.015 | **0.675 ± 0.016** | 0.244 | 2/3 seeds | n.s. (best mean) |
| WDC `llm_aug_cw` vs `baseline_cw` | 0.659 ± 0.015 | 0.661 ± 0.020 | 0.356 | 2/3 | n.s. (tie) |
| WDC `web_aug_cw` vs `baseline_cw` | 0.659 ± 0.015 | 0.658 ± 0.015 | 0.396 | 2/3 | n.s. (tie) |
| DBLP `string_aug` vs `baseline` | 0.9559 ± 0.0019 | 0.9483 ± 0.0020 | 0.0079 | **0/3** | worse (tiny, nominal) |
| DBLP `llm_aug` vs `baseline` | 0.9559 ± 0.0019 | 0.9566 ± 0.0006 | 0.728 | 2/3 | n.s. (tie) |
| DBLP `web_aug` vs `baseline` | 0.9559 ± 0.0019 | 0.9567 ± 0.0005 | 0.935 | 2/3 | n.s. (tie) |

**Robust conclusions (seed-aware):**
1. **On WDC, no augmentation strategy shows a detectable effect at this power.** `string_aug` has
   the best mean (+0.016) but loses on 1/3 seeds and is **not significant** (McNemar p=0.244); its
   effect sits inside the ±0.015–0.020 seed-noise band. `llm_aug` and `web_aug` are
   indistinguishable from baseline. (Phrase as "no *detectable* effect at n=3", not "no effect".)
2. **On DBLP, only `string_aug` reliably (but negligibly) hurts** — worse on **all 3 seeds**
   (−0.008, McNemar p=0.0079 nominal, h≈0.03); `llm_aug`/`web_aug` are indistinguishable.
3. The data-level findings (leakage fix, positive scarcity C1a) are unaffected — they are about
   the training data, not the F1.

### C3 — Multi-seed variance — RESOLVED, and it overturned single-seed conclusions
Each config was retrained over **3 seeds (13, 42, 87)** at lr=2e-5 (`run_multiseed.py` +
`aggregate_seeds.py`; McNemar re-run on seed-ensemble preds). Three single-seed conclusions did
**not** survive seed averaging — a strong justification for the multi-seed requirement:

| Single-seed (5e-5) claim | 3-seed reality |
|---|---|
| WDC `string_aug` +0.035, **significant** (p<0.001) | +0.016, **n.s.** (p=0.244) — best mean but within noise |
| WDC `llm_aug` −0.027, **significantly worse** (p=0.0004) | **+0.002, tie** (p=0.356) — the drop was a seed-42 artifact; **retracted** |
| WDC `web_aug` +0.011 borderline | −0.001, tie |
| DBLP `string_aug` significantly worse | **holds** (worse on all 3 seeds, p=0.0079) |
| DBLP `llm_aug` significantly worse | **tie** (p=0.728) — also a single-seed artifact |

The WDC baseline itself is high-variance (per-seed 0.677 / 0.649 / 0.653), which is exactly why
single-seed deltas were unreliable there.

> **Confound to acknowledge:** the single-seed runs were lr=5e-5 and the 3-seed runs lr=2e-5
> (changed to avoid a training-instability collapse on the imbalanced `llm_aug` set). So "the WDC
> `llm_aug` drop was a seed-42 artifact" is partly a **hyperparameter** effect — seed 42 at 2e-5
> is not the same run as seed 42 at 5e-5. The multi-seed conclusion (no detectable WDC effect)
> holds under either lr; but attribute the retraction to *seed + lr together*, not seed alone.

### C4 — WDC threshold / prior mismatch
WDC train/valid are ~20% positive but test is ~11.1%; the fixed 0.5/argmax rule is calibrated
to the wrong prior. Calibration on `baseline_cw` (`select_threshold.py`):

| Operating point | threshold | F1 | ΔF1 |
|---|---|---|---|
| default (argmax) | 0.500 | 0.641 | — |
| max-F1 on validation | 0.444 | 0.632 | −0.009 |
| matched to test prior (π=0.11) | 0.986 | 0.565 | −0.076 |
| max-F1 on test (oracle bound) | 0.843 | 0.659 | **+0.018** |

The oracle threshold is far above 0.5 (0.843), confirming the model over-predicts positives
at the default cut because of the prior shift. But **neither validation-tuned nor prior-matched
selection recovers it** — validation mirrors train, not test — so the achievable, *selectable*
gain is negligible. The WDC ≈0.64 ceiling is therefore substantially a **benchmark prior-shift
artifact of the WDC "small" split**, not purely model limitation; the oracle bound caps the
upside of threshold tuning at ~+0.018 F1.

### C5 — DBLP-Scholar splits are not entity-disjoint (benchmark artifact)
The standard DeepMatcher/Magellan DBLP-Scholar split is pair-random, not entity-disjoint
(`check_split_disjointness.py`): **93.0%** of test-left entities and **63.5%** of test-right
entities also appear in train. On top of that, while there are **0 *id-level* shared pairs**,
**186 train∩test pairs (3.2% of the test set) are *textually identical*** (plus 160 train∩valid,
64 test∩valid) — the DBLP/Scholar tables contain duplicate records under different ids, so the
serialized COL/VAL strings collide. That is genuine pair-level memorization on top of the 93%
entity overlap. **State it as "0 id-level shared pairs; 186 text-identical pairs (3.2% of test)."**
The near-ceiling baseline F1 of 0.956 is therefore partly memorization, and "no augmentation
beats baseline on DBLP" is partly an artifact of an already partly-seen test set. This is the
conventional benchmark split and is **not** re-split, but it is acknowledged when interpreting
DBLP results. (WDC base splits are verified perfectly entity-disjoint: 0 overlap, 0 text-identical.)

### C6 — Corner-case design target not met
The supervisor target is **40–50% corner cases**. On the clean data the best achieved was
**31.7%** (WDC `web_aug`) and **4.4%** (DBLP `string_aug`) — see Layer 2. No strategy reached
the target on either dataset; DBLP is structurally incapable of it (only 3–5% borderline pairs
exist), and on WDC the entity-disjoint LLM pool is positive-starved (C1a), so the
hard-positive supply is capped. This gap between design requirement and outcome is stated
plainly rather than left implicit, and informs the Task 7 direction.

---

## Layer 1 — Standard Metrics

**FINAL numbers: 3-seed mean ± sample std (seeds 13/42/87, lr=2e-5), clean entity-disjoint
data.** Std is the sample estimator (÷ n−1), matching how Aaron/DistillER report ± over runs.
These supersede the earlier single-seed (5e-5) numbers. Significance is the paired McNemar test on
the **seed-averaged (ensemble) predictions** — i.e. it tests the score-averaged ensemble model,
the pragmatic choice given per-seed tests at n=3 are underpowered (it is not a test of the
mean-F1 difference). ~9 comparisons were run at α=0.05; under Bonferroni (α≈0.0056) only
`web_v2_hn` (p=0.0024) strictly clears, so the borderline results below are nominal. Baselines:
`baseline_cw` (WDC), `baseline` (DBLP).

### WDC Products (test n=4,500 · 500 matches · 4,000 non-matches)

| Run | Precision | Recall | F1 (mean ± std) | ΔF1 | McNemar p | Verdict |
|-----|-----------|--------|-----------------|-----|-----------|---------|
| `baseline_cw` *(baseline)* | 0.618 ± 0.056 | 0.715 ± 0.070 | 0.659 ± 0.015 | — | — | — |
| `string_aug_cw` | 0.618 ± 0.051 | 0.751 ± 0.068 | **0.675 ± 0.016** | +0.016 | 0.244 | n.s. (best mean) |
| `llm_aug_cw` | 0.587 ± 0.031 | 0.759 ± 0.050 | 0.661 ± 0.020 | +0.002 | 0.356 | n.s. (tie) |
| `web_aug_cw` | 0.589 ± 0.043 | 0.750 ± 0.035 | 0.658 ± 0.015 | −0.001 | 0.396 | n.s. (tie) |

**No augmentation strategy robustly beats the baseline on WDC.** `string_aug_cw` has the best
mean (+0.016) but is within the seed-noise band (loses on 1/3 seeds; McNemar n.s.). `llm_aug_cw`
and `web_aug_cw` tie the baseline. The single-seed "string wins / llm loses" story was a seed
artifact (see C3).

### DBLP-Scholar (test n=5,742 · 1,070 matches · 4,672 non-matches)

| Run | Precision | Recall | F1 (mean ± std) | ΔF1 | McNemar p | Verdict |
|-----|-----------|--------|-----------------|-----|-----------|---------|
| `baseline` *(baseline)* | 0.952 ± 0.004 | 0.960 ± 0.001 | **0.9559 ± 0.0019** | — | — | — |
| `web_aug` | 0.948 ± 0.006 | 0.966 ± 0.007 | 0.9567 ± 0.0005 | +0.001 | 0.935 | n.s. (tie) |
| `llm_aug` | 0.953 ± 0.006 | 0.961 ± 0.006 | 0.9566 ± 0.0006 | +0.001 | 0.728 | n.s. (tie) |
| `string_aug` | 0.927 ± 0.007 | 0.971 ± 0.005 | 0.9483 ± 0.0020 | −0.008 | 0.0079 | ❌ worse (tiny, nominal) |

**Baseline unbeaten; `string_aug` reliably (but negligibly) hurts** — worse on all 3 seeds
(−0.008, McNemar p=0.0079 — *nominal*: borderline under Bonferroni; the 0/3-seed consistency is
the stronger evidence; h≈0.03). `llm_aug`/`web_aug` are statistically indistinguishable from
baseline. DBLP is at the architecture ceiling (~0.956) with no headroom, and its split is not
entity-disjoint (C5), so the baseline is partly inflated by memorization — consistent with
DistillER's finding that LLM-labeled data does not help RoBERTa on DBLP-Scholar.

---

## Layer 2 — Augmented Dataset Characteristics

### 2a. Positive ratio and corner-case proportion

Corner cases: hard positive (label=1, token-Jaccard sim < 0.3) or hard negative
(label=0, sim > 0.4). Targets from supervisor feedback: ~25% positive, 40–50% corner cases.

> **Threshold justification (review #8):** the cutoffs are token-Jaccard on the COL/VAL value
> tokens (`common.py: pair_sim`) and are deliberately **asymmetric** — a *match* below 0.3 overlap
> is genuinely hard (few shared tokens yet the same entity), whereas a *non-match* needs higher
> overlap (>0.4) to count as hard (many shared tokens yet different entities); the gap (0.3–0.4)
> is a neutral band excluded from both. The C6 "target not met" conclusion is not sensitive to
> the exact cutoffs — the best corner-case rate (31.7% WDC) is far below the 40–50% target under
> any reasonable choice, and DBLP's ~3–5% reflects a genuine absence of borderline pairs
> (confirmed independently by the 92-pair hard-negative ceiling in T7a).

> **Target not met (C6):** on the clean data the best corner-case proportion achieved was
> **31.7%** (WDC `web_aug`) and **4.4%** (DBLP `string_aug`) — below the 40–50% target on both
> datasets. DBLP is structurally incapable of reaching it (only 3–5% borderline pairs exist); on
> WDC the entity-disjoint LLM pool is positive-starved (C1a) so the hard-positive supply is
> capped. This gap motivates the Task 7 direction.

#### WDC Products (clean, entity-disjoint)

| Strategy (file) | N | % Positive | Hard Pos | Hard Neg | % Corner |
|---|---|---|---|---|---|
| baseline (`train.txt`) | 2,500 | 20.0% | 435 | 65 | 20.0% |
| string\_aug (`train_aug_string.txt`) | 5,000 | 20.0% | 882 | 123 | 20.1% |
| llm\_aug (`train_aug_llm.txt`) | 4,600 | 15.6% | 595 | 211 | 17.5% |
| web\_aug (`train_aug_web.txt`) | 3,303 | 34.8% | 980 | 66 | 31.7% |

**Note on `llm_aug`:** positive rate fell to 15.6% (from the leaked 25.1%) and corner-cases to
17.5% (from 27.5%) — the missing positives were the leaked valid/test matches (C1a). It now
carries *more* hard negatives (211) than the baseline, from active-learning hard-negative mining.

**Note on `web_aug`:** the combined file's 34.8% positive / 31.7% corner are diluted by the
`train.txt` base; the *added web slice alone* (803 pairs) is still **81% positive / 68% corner**,
because Tavily queries built from `{brand} {title}` inherently surface the same product on other
retailer sites. This skew persists after the leakage fix — so `web_aug_cw`'s small positive ΔF1
(+0.011, within noise) should not be read as the skew being "solved".

#### DBLP-Scholar (clean)

| Strategy (file) | N | % Positive | Hard Pos | Hard Neg | % Corner |
|---|---|---|---|---|---|
| baseline (`train.txt`) | 17,223 | 18.6% | 200 | 344 | 3.2% |
| string\_aug (`train_aug_string.txt`) | 34,446 | 18.6% | 935 | 567 | 4.4% |
| llm\_aug (`train_aug_llm.txt`) | 19,308 | 16.6% | 200 | 385 | 3.0% |
| web\_aug (`train_aug_web.txt`) | 17,965 | 21.4% | 337 | 351 | 3.8% |

**Observation:** DBLP-Scholar has a structurally low corner-case proportion (3–5%) across
all strategies. Academic paper descriptions tend to be either clearly overlapping (same
paper) or clearly distinct (different papers), with few genuinely borderline pairs —
unlike e-commerce products where variant-level differences create abundant hard cases.

### 2b. Decisive-attribute coverage (WDC Products only)

Percentage of pairs (either left or right entity) matching each attribute pattern:

| Strategy | color | memory/storage | size | bundle | edition | model number |
|---|---|---|---|---|---|---|
| baseline | 36.4% | 40.5% | 36.8% | 13.2% | 20.7% | 35.0% |
| string\_aug | 35.5% | 39.9% | 36.4% | 13.0% | 20.3% | 34.1% |
| llm\_aug | 33.9% | 33.9% | 37.3% | 13.8% | 20.9% | 29.6% |
| web\_aug | 34.9% | 37.0% | 36.3% | 12.8% | 19.4% | 32.1% |

All strategies broadly maintain similar attribute coverage to the baseline. `llm_aug` shows
slightly lower model-number coverage (29.6% vs ~34-35%) and memory/storage (33.9%); no
strategy materially expands coverage of underrepresented attributes (bundle: ~13% in all).

### 2c. Overlap between augmentation strategies (added pairs only)

Pairwise Jaccard similarity of the pairs *added* beyond `train.txt` by each strategy:

| Pair | Jaccard | Intersection | Union |
|---|---|---|---|
| WDC: string\_aug ∩ llm\_aug | 0.0 | 0 | 4,600 |
| WDC: string\_aug ∩ web\_aug | 0.0 | 0 | 3,298 |
| WDC: llm\_aug ∩ web\_aug | 0.0 | 0 | 2,898 |
| DBLP: string\_aug ∩ llm\_aug | 0.0 | 0 | 19,282 |
| DBLP: string\_aug ∩ web\_aug | 0.0 | 0 | 17,898 |
| DBLP: llm\_aug ∩ web\_aug | 0.0 | 0 | 2,734 |

**Zero overlap** across all pairs and datasets. Each strategy explores a completely disjoint
region of pair space: string augmentation creates modified variants of existing pairs, LLM
active learning selects from a blocking-derived candidate pool, and web retrieval sources
pairs from external websites. The three strategies are fully complementary by construction.

---

## Layer 3 — Qualitative Error Analysis

> **Note:** the counts below are from the **single-seed (seed-42, 5e-5) models**, so the totals
> reflect one run each (e.g. the seed-42 `llm_aug` was a low outlier — see C3). Read the error
> *classes and mechanisms* as illustrative of failure modes, not as the headline result; the
> headline is the 3-seed F1 in Layer 1. Similarity metric = **token-level Jaccard** (`pair_sim`).

Error classification uses similarity-based heuristics (see `src/analysis/common.py`):
- **ambiguous\_variant**: catches most errors — borderline or moderate similarity, model
  decision threshold crossed in either direction
- **low\_sim\_match** (FN): label=1 but sim < 0.2 — model failed to predict match despite
  low token overlap
- **high\_sim\_non\_match** (FP): label=0 but sim > 0.5 — model over-predicted match for
  high-similarity pair that is actually a non-match (e.g., same product, different variant)
- **noisy\_incomplete**: one entity has < 6 tokens after stripping COL/VAL markers

### 3a. Error counts and class breakdown

#### WDC Products

| Run | Total errors | ambiguous\_variant | low\_sim\_match | high\_sim\_non\_match | noisy\_incomplete |
|---|---|---|---|---|---|
| `baseline_cw` | 421 (9.4%) | 299 (71%) | 103 (24%) | 16 (4%) | 3 (1%) |
| `string_aug_cw` | **337 (7.5%)** | 208 (62%) | 122 (36%) | 5 (1%) | 2 (1%) |
| `web_aug_cw` | 384 (8.5%) | 256 (67%) | 117 (30%) | 9 (2%) | 2 (1%) |
| `llm_aug_cw` | **485 (10.8%)** | **380 (78%)** | 93 (19%) | 11 (2%) | 1 (0%) |

**Pattern (single-seed, illustrative):** in the seed-42 run `string_aug_cw` had the fewest errors
(337, lowest `high_sim_non_match` FP rate) and `llm_aug_cw` the most (485, `ambiguous_variant`
FPs surging to 380 — over-predicting "match" at moderate similarity). This illustrates the
*failure mode* of a negative-heavy AL set, but note the seed-42 `llm_aug` was a low outlier: over
3 seeds `llm_aug_cw` **ties** the baseline (Layer 1 / C3), so this large error count is not
representative of its typical performance.

#### DBLP-Scholar

| Run | Total errors | ambiguous\_variant | high\_sim\_non\_match | noisy\_incomplete | low\_sim\_match |
|---|---|---|---|---|---|
| `baseline` | **91 (1.6%)** | 75 (82%) | 7 (8%) | 6 (7%) | 3 (3%) |
| `llm_aug` | 103 (1.8%) | 86 (83%) | 11 (11%) | 4 (4%) | 2 (2%) |
| `web_aug` | 107 (1.9%) | 94 (88%) | 10 (9%) | 1 (1%) | 2 (2%) |
| `string_aug` | 117 (2.0%) | 101 (86%) | 10 (9%) | 3 (3%) | 3 (3%) |

DBLP errors are uniformly dominated by `ambiguous_variant` (~82–87%), with a consistent
minority of `high_sim_non_match` cases (~7–12%; papers sharing title tokens but at different
venues or years). The error distribution is essentially invariant across all four strategies —
consistent with the very small F1 effects, though note that two of those effects are
statistically significant *regressions* under the paired test (C2), not ties.

### 3b. Representative examples with LLM explanations (selected)

**WDC `string_aug_cw` — high\_sim\_non\_match (best run; 5 total)**

> LEFT:  `COL brand VAL Corsair COL title VAL CORSAIR Crystal 570X RGB Tempered Glass, Premium
> ATX Mid Tower Case, White ...`  
> RIGHT: `COL brand VAL Corsair COL title VAL CORSAIR Crystal 570X RGB Tempered Glass, Premium
> ATX Mid Tower Case, Black ...`  
> true=NON-MATCH · pred=MATCH · sim=0.697  
> *LLM:* "The model likely confused these as a match because their descriptions are nearly identical
> except for the color variant (White vs. Black), a subtle difference that token-Jaccard overlap
> fails to penalize."

**WDC `llm_aug_cw` — ambiguous\_variant (model over-predicts match; 380 total — most of any run)**

> LEFT:  `COL brand VAL Kingston COL title VAL Kingston A2000 NVMe PCIe SSD 500GB ...`  
> RIGHT: `COL brand VAL Kingston COL title VAL Kingston A2000 NVMe PCIe SSD 1000GB ...`  
> true=NON-MATCH · pred=MATCH · sim=0.518  
> *LLM:* "The model predicted match because both describe the same product line (Kingston A2000 NVMe)
> with high token overlap, but the capacity difference (500GB vs. 1TB) makes them distinct products
> — a decisive attribute the model failed to weight sufficiently."

**DBLP `baseline` — high\_sim\_non\_match (consistent across strategies; 7–12 per run)**

> LEFT:  `COL title VAL Querying and Mining Data Streams: You Only Get One Look COL authors VAL J.
> Gehrke, F. Korn, S. Muthukrishnan COL venue VAL SIGMOD COL year VAL 2002`  
> RIGHT: `COL title VAL Querying and Mining Data Streams: You Only Get One Look COL authors VAL J.
> Gehrke, F. Korn, S. Muthukrishnan COL venue VAL SIGMOD Record COL year VAL 2003`  
> true=NON-MATCH · pred=MATCH · sim=0.636  
> *LLM:* "The model predicted match because the title and authors are identical, but these are
> distinct publications — a conference paper (SIGMOD 2002) and its subsequent journal version
> (SIGMOD Record 2003), a distinction only resolvable via year/venue combination."

Full error examples for all 8 runs are in `experiments/results/task6_error_analysis.json`
and browsable via `notebooks/08_task6_analysis.ipynb` (cell 3b, configurable `SHOW_RUN`).

---

## DistillER Comparison (Zeakis et al., 2026)

**Reference:** "DistillER: Knowledge Distillation for Low-Resource Entity Resolution"
(Zeakis et al., 2026). Paper: `thesis-docs/2602.05452v1.pdf`.

DistillER studies LLM-based knowledge distillation for entity resolution, using RoBERTa as
one of the "student" models trained on noisy LLM-teacher labels (Qwen-2.5:32b). Their
results offer three direct comparison points.

### Comparison point 1 — DBLP-Scholar: same direction, different scale

DistillER's closest dataset to this thesis's DBLP-Scholar is **D9** (DBLP/Scholar;
2,476 DBLP entities, 61,350 Scholar entities, trained on ~10% of entities):

| Setting | DistillER D9 RoBERTa | This thesis DBLP-Scholar |
|---|---|---|
| Ground-truth labels | F1 = 0.89 | F1 = **0.958** (baseline) |
| LLM-labeled data | F1 = 0.86 (−0.03) | F1 ≤ 0.953 (all augment. ≤ baseline) |

Both show the same **direction**: LLM-labeled/augmented data does not improve over
ground-truth labels on DBLP-Scholar. This thesis's absolute F1 is higher because it trains
on the full split (not 10% of entities), but the trend is consistent — DBLP-Scholar is near
the ceiling for RoBERTa-base, and any augmentation approach adds noise rather than signal.

### Comparison point 2 — Product datasets: consistently the hardest

DistillER's product datasets (D2 Abt-Buy, D3 Amazon-Google, D8 Walmart-Amazon) are their
hardest across all strategies. From the paper (Table 8):

| Dataset | RoBERTa (GT labels) | RoBERTa (LLM labels) |
|---|---|---|
| D2 (Abt-Buy) | 0.78 | 0.75 |
| D3 (Amazon-Google) | 0.40 | 0.41 |
| D8 (Walmart-Amazon) | 0.44 | 0.57 |
| **Mean (all 8 datasets)** | **0.66** | **0.69** |

The paper explicitly attributes product difficulty to "common tokens that might disorientate
the teacher model in detecting true matches. For example, an Ethernet cable described as
'Cat6 Ethernet cable 5m' while another entry appears as 'Cat6 Ethernet cable 15m,' where
the shared tokens overwhelm the subtle but crucial difference."

This is directly mirrored in this thesis's WDC Products results: the best 3-seed mean any strategy
achieved was 0.675 (`string_aug_cw`), far below DBLP-Scholar's baseline of 0.956. The
dominant error class in WDC — `high_sim_non_match` and borderline `ambiguous_variant` — is
exactly the "shared-token / subtle-difference" failure pattern DistillER identifies.

DistillER's D8 (Walmart-Amazon) shows a notable exception: LLM labels give +0.13 F1 vs
ground truth (0.44 → 0.57). This is the one product dataset where LLM supervision helped
substantially, likely because their 10%-entity low-resource regime is so sparse that LLM
labels provide meaningful new signal. In this thesis's WDC Products setting (full train
split, 2,500 pairs base), the same saturation effect was not present — the model already had
sufficient coverage of common patterns.

### Comparison point 3 — Data selection ratio validates supervisor's target

DistillER's data selection methodology (Section 4.1) uses ranking-based blocking and
explicitly sets a **3:1 positive-to-negative ratio** target (their wording: "approximate
the 3-to-1 positive-to-negative ratio"), achieving comparable performance to a labeled
baseline. Their finding: "Ranking emerges as the most reliable unsupervised selection method,
maintaining a balanced class distribution" — directly consistent with the supervisor's 1:3
(equivalent) ratio requirement for Tasks 4 and 5.

On the clean, 3-seed data this comparison is nuanced. Once leakage is removed, the
entity-disjoint `llm_aug` pool is positive-starved (15.6% WDC / 16.6% DBLP, C1a) — so the
Steiner & Bizer ratio target cannot be met *honestly* on a closed benchmark (a finding in
itself) — and `llm_aug` **ties** the baseline on both datasets (no lift). `web_aug` is the only
strategy that injects genuinely new positive entities (via retrieval), but its added slice is
still skewed (81% positive / 68% corner) and it too ties baseline. DistillER's ratio-aware
selection remains the relevant lever — rebalancing + hard-negative injection is the Task 7
direction below.

### Summary of DistillER comparison

| Dimension | DistillER finding | This thesis — consistent? |
|---|---|---|
| DBLP-Scholar: LLM labels vs GT | LLM labels slightly worse (F1 −0.03) | ✅ Same direction: all augment ≤ baseline |
| Product datasets: hardest | D2/D3/D8 hardest, high token overlap | ✅ WDC Products F1 0.63–0.68 vs DBLP 0.95+ |
| LLM labels can match GT | Yes, avg +0.03 over 8 datasets | ⚠️ Clean, 3-seed `llm_aug` **ties** baseline on both datasets (no lift; positive-starved pool, C1a) |
| Data ratio matters | 3:1 pos:neg approximates labeled baseline | ⚠️ Clean entity-disjoint pool can't reach 25% honestly (C1a); web slice still skewed → Task 7 |
| Best strategy for future | SFT on LLM-generated *explanations* for LLM students (Llama) | ⚠️ Out of scope for roberta-base classifier; noted as future work |

> **Note on absolute-number comparability:** DistillER trains on ~10% of entities per dataset;
> this thesis uses full train splits. Absolute F1 numbers are not directly comparable —
> only trends and directional effects are.

---

## Task 7 Proposal — Rebalanced Web Augmentation for WDC Products

Based on the **clean** Layer 1–3 analysis and the DistillER comparison, the proposed Task 7
direction is: **rebalance the WDC Products web augmentation slice toward a 1:3 positive ratio.**

### Why web augmentation (not string or LLM)
Under 3-seed evaluation **no strategy robustly beats the WDC baseline** (Layer 1), so Task 7's
goal is to find one that *does*:
- `string_aug_cw` has the best mean (+0.016) but it is n.s. and is already a maxed-out simple
  baseline — no LLM/web research question in it.
- `llm_aug_cw` ties baseline and is positive-starved after the leakage fix (15.6% positive) — on
  a closed benchmark it cannot supply new positives honestly (C1a), a dead end without changing
  the data source.
- `web_aug` ties baseline too, but it is the **only strategy that introduces genuinely new
  positive entities** (via retrieval) — directly addressing the positive-scarcity problem. Its
  added slice is badly skewed (**81% positive / 68% corner**), so the model over-predicts at
  moderate similarity. Fixing the skew **and injecting generated hard negatives** (B7) is the
  clearest lever to turn a tie into a robust gain.

### Proposed experiment — `web_aug_cw_v2` (WDC Products)
1. From the relevant `web_labeled.jsonl` slice (~650 positives + ~153 negatives), **downsample
   the positives** so the added slice is ≈1:3 positive (e.g. keep ~3× the negatives ≈ 460 pos,
   stratified by similarity percentile to retain a spread of hard positives).
2. Merge with `train.txt` → target ≈25–30% positive combined; write `train_aug_web_v2.txt`.
3. Retrain `web_aug_cw_v2` with `--class_weight balanced`, **over 3 seeds** (13, 42, 87), and
   evaluate vs `baseline_cw` and `string_aug_cw` with `significance.py`.

**Hypothesis:** reducing the web-slice positive skew (and adding generated hard negatives, B7 /
notebook 07b) cuts the `ambiguous_variant` FP surge and converts the current baseline-tie into a
*robust* gain, ideally exceeding `string_aug_cw`'s best-mean 0.675, while raising corner-case
coverage. Evaluated over **3 seeds** with McNemar, since single-seed effects on WDC are unreliable.

---

## Task 7 Results (executed)

### T7a — DBLP hard-negative mining (B3) — null
`mine_dblp_hard_negatives.py` yielded **92** clean same-title-different-paper hard negatives (42
conf/journal cross-pairs + 50 title-dups w/ venue&year both differing). `dblp_hardneg` (single
seed): F1 **0.9573** vs baseline 0.9579 (Δ−0.0006, McNemar p=1.000) — **no effect**. Expected:
92 is the *structural ceiling* (DBLP titles are near-unique), which is itself the finding —
DBLP-Scholar barely contains this hard-negative type, explaining the ~0 hard negatives (Layer 2)
and the F1 ceiling. Not worth 3 seeds.

### T7b — WDC generated hard negatives (B7, notebook 07b) — significantly worse
The LLM generated **1,290** verified hard negatives (similar-but-different products via Tavily;
142 correctly dropped as accidental same-product — the negatives are high quality). Two web-v2
variants (3 seeds):

| Run | F1 (mean ± std) | ΔF1 | McNemar p | Verdict |
|---|---|---|---|---|
| `web_v2_bal_cw` (≈1:3) | 0.643 ± 0.019 | −0.016 | 0.0098 | worse (nominal) |
| `web_v2_hn_cw` (hard-neg-heavy) | 0.631 ± 0.036 | −0.029 | 0.0024 | **significant worse** |

**It backfired — the opposite of the hypothesis.** Error analysis (seed-ensemble): false
positives *increased* (baseline 216 → bal 259 → hn 283), driven by `ambiguous_variant`. Adding
hard *negatives* made the model over-predict *more*.

**Diagnosis / finding:** the hard negatives are `(clean benchmark entity, web-extracted offer)`
pairs, but the test set is `(clean, clean)`. The web-extracted text is a **different
distribution**, so it does not transfer, and the class-weighted loss over the retained web
positives + off-distribution negatives pushes over-prediction. **The problem with web
augmentation is the web-vs-benchmark text mismatch, not the positive ratio** — rebalancing and
injecting even high-quality hard negatives does not fix it, it worsens it. Combined with
`llm_aug` (clean hard negatives) merely *tying* baseline, hard-negative injection does not help
WDC in either form.

### Task 7 interim conclusion
Neither targeted intervention improved results: DBLP hard negatives are structurally scarce
(null), WDC web hard negatives are distribution-mismatched (worse). This reinforces the central
finding — on these benchmarks, with leakage removed and multi-seed evaluation, augmentation does
not robustly help. Remaining experiments: **union/combined training sets** (does complementarity
help?) and **low-resource ablation** (does augmentation help when the base is small — Aaron's
Figure-4 / DistillER-D8 regime, the most likely place for a positive result).
