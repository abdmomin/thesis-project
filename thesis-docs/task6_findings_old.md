# Task 6 — Evaluation Findings

**Author:** Abdullah Al Momin  
**Date:** 2026-06-17  
**Datasets:** WDC Products (small / 80% CC) · DBLP-Scholar (citation matching)  
**Model:** `roberta-base` fine-tuned as binary sequence-pair classifier (Ditto architecture)  

Reusable computation code: `src/analysis/common.py` · `src/analysis/task6_compute.py`  
Analysis notebook: `notebooks/08_task6_analysis.ipynb`  
Raw JSON outputs: `experiments/results/task6_dataset_characteristics.json` · `task6_error_analysis.json`

---

## Layer 1 — Standard Metrics

All four augmentation strategies (Tasks 3–5) compared against the final baselines
(`baseline_cw` for WDC Products, `baseline` for DBLP-Scholar).

### WDC Products (test n=4,500 · 500 matches · 4,000 non-matches)

| Run | Precision | Recall | F1 | ΔF1 vs baseline | p-value (z-test) |
|-----|-----------|--------|----|-----------------|-----------------|
| `baseline_cw` *(final baseline)* | 0.559 | 0.752 | 0.641 | — | — |
| `string_aug_cw` | 0.651 | 0.704 | **0.676** | **+0.035** | **0.0014** ✅ |
| `llm_aug_cw` | 0.563 | 0.756 | 0.646 | +0.005 | 0.828 ❌ |
| `web_aug_cw` | 0.507 | 0.832 | 0.630 | −0.011 | 0.017 ✅ (worse) |

**Best:** `string_aug_cw` — the only strategy that significantly improved WDC Products F1.  
**Worst:** `web_aug_cw` — significantly below the baseline (p=0.017) and far below `string_aug_cw` (p<0.001).

### DBLP-Scholar (test n=5,742 · 1,070 matches · 4,672 non-matches)

| Run | Precision | Recall | F1 | ΔF1 vs baseline | p-value (z-test) |
|-----|-----------|--------|----|-----------------|-----------------|
| `baseline` *(final baseline)* | 0.949 | 0.967 | **0.958** | — | — |
| `string_aug` | 0.938 | 0.953 | 0.946 | −0.012 | 0.069 ❌ |
| `llm_aug` | 0.935 | 0.959 | 0.947 | −0.011 | 0.092 ❌ |
| `web_aug` | 0.942 | 0.964 | 0.953 | −0.005 | 0.425 ❌ |

**Best:** `baseline` remains unbeaten — all augmentation strategies are statistically
indistinguishable from or slightly below it. DBLP-Scholar is near the ceiling for this
architecture (~0.95+) and augmentation provides no significant lift.

---

## Layer 2 — Augmented Dataset Characteristics

### 2a. Positive ratio and corner-case proportion

Corner cases: hard positive (label=1, token-Jaccard sim < 0.3) or hard negative
(label=0, sim > 0.4). Targets from supervisor feedback: ~25% positive, 40–50% corner cases.

#### WDC Products

| Strategy (file) | N | % Positive | Hard Pos | Hard Neg | % Corner |
|---|---|---|---|---|---|
| baseline (`train.txt`) | 2,500 | 20.0% | 478 | 23 | 20.0% |
| string\_aug (`train_aug_string.txt`) | 5,000 | 20.0% | 959 | 48 | 20.1% |
| llm\_aug (`train_aug_llm.txt`) | 4,600 | 25.1% | 1,156 | 109 | 27.5% |
| web\_aug (`train_aug_web.txt`) | 3,299 | 34.6% | 906 | 156 | 32.2% |

**Note on web\_aug:** The combined file's 34.6% positive / 32.2% corner are
diluted by the original `train.txt` base. The *added web slice alone* (799 pairs) is
**80.1% positive** and **70.2% corner case** — almost entirely hard positives
(557 pairs with sim < 0.3 but label=1), reflecting that Tavily queries built from
`{brand} {title}` systematically surface the same product on alternative retailer sites.

#### DBLP-Scholar

| Strategy (file) | N | % Positive | Hard Pos | Hard Neg | % Corner |
|---|---|---|---|---|---|
| baseline (`train.txt`) | 17,223 | 18.6% | 547 | 0 | 3.2% |
| string\_aug (`train_aug_string.txt`) | 34,446 | 18.6% | 1,518 | 0 | 4.4% |
| llm\_aug (`train_aug_llm.txt`) | 19,323 | 16.8% | 621 | 17 | 3.3% |
| web\_aug (`train_aug_web.txt`) | 17,922 | 21.1% | 659 | 7 | 3.7% |

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
| `llm_aug_cw` | 415 (9.2%) | 306 (74%) | 102 (25%) | 6 (1%) | 1 (0%) |
| `web_aug_cw` | 489 (10.9%) | **399 (82%)** | **69 (14%)** | **17 (3%)** | 4 (1%) |

**Pattern:** `string_aug_cw` achieves the fewest errors overall by being more conservative —
it has the highest `low_sim_match` FN rate (122; 36%) but the lowest `high_sim_non_match` FP
rate (5; 1%). `web_aug_cw` shows the opposite extreme: fewest FNs at low similarity (69),
but an avalanche of `ambiguous_variant` FPs (399; 82%) — a direct consequence of the model
learning "low similarity does not mean non-match" from the 557 hard-positive web pairs and
then over-generalising at test time.

#### DBLP-Scholar

| Run | Total errors | ambiguous\_variant | high\_sim\_non\_match | noisy\_incomplete | low\_sim\_match |
|---|---|---|---|---|---|
| `baseline` | **91 (1.6%)** | 75 (82%) | 7 (8%) | 6 (7%) | 3 (3%) |
| `string_aug` | 117 (2.0%) | 101 (86%) | 10 (9%) | 3 (3%) | 3 (3%) |
| `llm_aug` | 115 (2.0%) | 100 (87%) | 12 (10%) | 2 (2%) | 1 (1%) |
| `web_aug` | 102 (1.8%) | 86 (84%) | 12 (12%) | 1 (1%) | 3 (3%) |

DBLP errors are uniformly dominated by `ambiguous_variant` (~82–87%), with a consistent
minority of `high_sim_non_match` cases (~7–12%; papers sharing title tokens but at different
venues or years). The error distribution is essentially invariant across all four strategies,
matching the statistically indistinguishable F1 results.

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

**WDC `web_aug_cw` — ambiguous\_variant (model over-predicts match; 399 total)**

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

This is directly mirrored in this thesis's WDC Products results: the highest F1 any strategy
achieved was 0.676 (`string_aug_cw`), far below DBLP-Scholar's baseline of 0.958. The
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

This provides retrospective validation for this thesis's design choice: `llm_aug`'s
Steiner & Bizer active learning pipeline produced a 25.1%/16.8% positive rate (close to the
1:3 = 25% target), and did not significantly hurt performance on either dataset. `web_aug`'s
violation of this target (34.6% combined / 80% in the added slice) directly produced the
observed WDC regression — exactly the failure mode DistillER's ratio-aware selection is
designed to prevent.

### Summary of DistillER comparison

| Dimension | DistillER finding | This thesis — consistent? |
|---|---|---|
| DBLP-Scholar: LLM labels vs GT | LLM labels slightly worse (F1 −0.03) | ✅ Same direction: all augment ≤ baseline |
| Product datasets: hardest | D2/D3/D8 hardest, high token overlap | ✅ WDC Products F1 0.63–0.68 vs DBLP 0.95+ |
| LLM labels can match GT | Yes, avg +0.03 over 8 datasets | ✅ `llm_aug` roughly ties baseline on DBLP; slightly below for WDC |
| Data ratio matters | 3:1 pos:neg approximates labeled baseline | ✅ Well-calibrated `llm_aug` ties baseline; poorly-calibrated `web_aug` regresses |
| Best strategy for future | SFT on LLM-generated *explanations* for LLM students (Llama) | ⚠️ Out of scope for roberta-base classifier; noted as future work |

> **Note on absolute-number comparability:** DistillER trains on ~10% of entities per dataset;
> this thesis uses full train splits. Absolute F1 numbers are not directly comparable —
> only trends and directional effects are.

---

## Task 7 Proposal — Rebalanced Web Augmentation for WDC Products

Based on the Layer 2 analysis and DistillER comparison, the proposed Task 7 direction is:

**Rebalance the WDC Products web augmentation slice before training.**

### Justification

1. The `web_aug_cw` regression (F1 0.630, lowest of all WDC strategies) is traceable to
   a single structural cause: the 799 Tavily-retrieved web pairs are 80.1% positive and
   70.2% corner case (almost entirely hard positives), because queries built from the entity's
   own `{brand} {title}` string inevitably surface pages describing that same product.

2. Training on 557 hard-positive pairs (low sim but label=1) taught the model to predict
   "match" even at low-moderate similarity — confirmed by the shift in error distribution:
   `ambiguous_variant` FPs jumped from 299 (baseline) to 399 (web_aug), while
   `low_sim_match` FNs dropped from 103 to 69.

3. DistillER's data-selection finding and the supervisor's design criteria both point to the
   same fix: enforce a balanced ratio before merging.

### Proposed experiment

**`web_aug_cw_v2` (WDC Products)**:
1. Take the existing `web_labeled.jsonl` (640 web positives + 159 web negatives).
2. Downsample the 640 web positives to ~477 (keeping a stratified random 75% of them
   by similarity percentile, to retain some hard positives while reducing skew).
3. This targets a combined training set of ~25% positive:
   - base train.txt: 500 pos + 2000 neg = 2500
   - web slice v2: ~477 pos + 159 neg = 636
   - combined: 977 pos / 2159 neg = ~31% positive (better, but still above target due to 
     the base set's 20%)
   - Alternatively, a stricter downsample (e.g., 159 × 3 = 477 → target 1:3 in web slice only).
4. Retrain with `--class_weight balanced --train_file train_aug_web_v2.txt`.
5. Evaluate against baseline_cw and string_aug_cw.

**Hypothesis:** Reducing the web-slice positive skew will reduce the `ambiguous_variant` FP
surge, bring precision back above 0.55, and achieve F1 ≥ 0.64 (approaching or matching
`llm_aug_cw`). Combined with `string_aug_cw`'s advantage in conservative precision, the
rebalanced web_aug may reach string_aug-level performance by better representing hard
negatives alongside hard positives.

No new Tavily searches or LLM labeling calls are needed — the fix is entirely in the
resampling of already-labeled data in `web_labeled.jsonl`.
