# Methodology Review & Diagnosis — Tasks 1–6

**Reviewer:** Independent code & methodology audit
**Date:** 2026-06-21
**Scope:** Full pipeline through Task 6 — data prep, baseline, string/LLM/web augmentation, three-layer analysis, statistical testing, DistillER comparison.
**Method:** Read all source (`src/`), all notebooks (`notebooks/`), Task 6 docs, and **independently recomputed** every reported F1 from the saved prediction files plus the significance tests and leakage checks described below. Nothing in the repository was modified.

---

## Verdict

The engineering is solid and reproducible. Every reported F1 was recomputed from the saved `*_preds.jsonl` files and matches the values in `task6_findings.md` exactly. The code is clean, modular, well-documented; the string-aug operators are faithful to Ditto; the active-learning loop follows the Steiner & Bizer recipe; and the Task 6 three-layer analysis is genuinely thorough.

However, there are **six issues**, two of them serious enough to affect the thesis's stated conclusions. They are ranked below by impact. The most important — **entity leakage in the LLM and web augmentation pipelines** — violates the supervisor's explicit entity-disjoint requirement and contaminates the `llm_aug` / `web_aug` results. The second — **use of an unpaired z-test instead of the paired McNemar test** — flips two of the reported DBLP-Scholar conclusions from "not significant" to "significantly worse."

---

## Issue 1 — Entity leakage in Task 4 (LLM) and Task 5 (web) — **critical**

### What happens
The embedding-blocking candidate pool (`04_task4_step1_2_blocking.ipynb`) is built over the **full entity universe**, and the only exclusion applied is exact known *pairs* (`known_pairs_excluded` in `candidates_pool_summary.json`) — **not** held-out *entities*. The WDC entity tables contain 2,896 left / 2,891 right entities, but `train.txt` has only 1,000 unique entities; the extra entities are the valid and test records. As a result, the augmented training sets pair held-out test entities (with LLM-assigned labels) into training data.

### Evidence (recomputed)
Count of *added* pairs (beyond `train.txt`) that include at least one **test-set entity**:

| Augmented file | WDC Products | DBLP-Scholar |
|---|---|---|
| `train_aug_llm.txt` | 1,235 / 2,100 (**59%**) | 1,646 / 2,100 (**78%**) |
| `train_aug_web.txt` | 280 / 799 (**35%**) | 271 / 699 (**39%**) |
| `train_aug_string.txt` | 0 | 45 |

No *exact* test pairs leak (so this is not direct label leakage of the test set), but the model nonetheless sees test-side entity text during training in new pairings. WDC Products' original splits are perfectly entity-disjoint (verified: 0 train/test entity overlap); the LLM and web pipelines **broke** that disjointness.

### Why it matters
CLAUDE.md's training-quality criteria explicitly require "entity-disjoint splits: verify that no entity from train appears in valid/test." That holds for the baseline and string-aug runs but is violated for `llm_aug` and `web_aug` — precisely the strategies whose value the thesis is trying to assess. Their absolute numbers and any comparison that relies on them are contaminated.

### Mitigating note
`string_aug` is clean, and it still beats `llm_aug` on WDC Products *despite* `llm_aug` carrying a leakage advantage — so that specific conclusion ("simple string augmentation beats the LLM active-learning pipeline on WDC") is robust, arguably strengthened.

### Recommended fix
Filter the candidate pool to **train-only** entity IDs before blocking (exclude any entity appearing in valid/test), then regenerate `train_aug_llm.txt` / `train_aug_web.txt` and rerun those four model trainings. Expect `llm_aug` / `web_aug` numbers to **drop** after the fix — that is the correct, honest result and it makes the `string_aug` win cleaner.

---

## Issue 2 — Wrong significance test; two DBLP conclusions flip — **serious**

### What happens
CLAUDE.md and `task6_findings.md` compare two models with an **unpaired two-proportion z-test on accuracy**. Both models are evaluated on the *same* test set, so the correct test is **McNemar's test** (paired, on the discordant predictions). The unpaired test ignores the pairing and is underpowered.

### Evidence (recomputed via McNemar with continuity correction, from the prediction files)

| Comparison | Reported z-test p | McNemar p | Conclusion change |
|---|---|---|---|
| WDC string_aug vs baseline | 0.0014 ✅ | 0.0000 ✅ | same (better) |
| WDC llm_aug vs baseline | 0.828 ❌ | 0.798 ❌ | same (n.s.) |
| WDC web_aug vs baseline | 0.017 ✅ worse | 0.0004 ✅ worse | same (worse) |
| **DBLP string_aug vs baseline** | 0.069 ❌ | **0.0041 ✅ worse** | **flips → significantly worse** |
| **DBLP llm_aug vs baseline** | 0.092 ❌ | **0.0053 ✅ worse** | **flips → significantly worse** |
| DBLP web_aug vs baseline | 0.425 ❌ | 0.208 ❌ | same (n.s.) |

The report's claim that DBLP-Scholar augmentation is "statistically indistinguishable from baseline" is incorrect under the proper test: `string_aug` and `llm_aug` both **significantly degrade** DBLP-Scholar F1.

### Additional concern
Accuracy is a weak proxy for F1 under the 11% test-positive rate (WDC). Report a **bootstrap confidence interval directly on F1** (resample test pairs with replacement, ~10k iterations) alongside McNemar, rather than z-testing accuracy. Continue reporting Cohen's h for effect size as CLAUDE.md already prescribes.

---

## Issue 3 — Single seed; no variance estimate — **serious for marginal effects**

Every model is trained once with `seed=42` (`train_baseline.py`). For roberta-base on a few-thousand-pair training set, run-to-run F1 variance of ±0.01–0.02 is normal — the **same magnitude** as several reported effects (`llm_aug` +0.005 on WDC; DBLP deltas of −0.005 to −0.012). Without repeated seeds these effects cannot be distinguished from initialization/ordering noise.

**Recommendation:** retrain each configuration with ≥3 seeds (5 preferred), report mean ± std, and run significance tests on the seed-averaged predictions or via a paired test across seeds. This is the single most common examiner objection to a thesis of this design and is inexpensive to address.

---

## Issue 4 — WDC Products train/test prior mismatch is the real F1 ceiling

Verified positive rates: **train 20.0%, valid 20.0%, test 11.1%**. Model selection, early stopping (on validation F1), and the fixed 0.5 `argmax` decision threshold are all calibrated to a 20% prior, while the test set is 11%. Tuning the threshold on validation cannot fix this because validation mirrors train, not test.

This prior shift — not only "products are intrinsically hard" — is a primary driver of the ~0.64 WDC ceiling. **Recommendation:** report a decision threshold calibrated to the test (or a held-out, test-matched) prior, or explicitly frame the gap as a benchmark artifact of the WDC "small" split rather than pure model limitation.

---

## Issue 5 — DBLP-Scholar benchmark splits are not entity-disjoint

The standard DeepMatcher/Magellan DBLP-Scholar splits used here are pair-level random, not entity-disjoint. Verified: **72% of test entities also appear in train**, and there are **186 exact (left,right) pairs shared between train and test** (plus 160 train/valid, 64 test/valid). The near-ceiling baseline F1 of 0.958 is therefore partly memorization, and the finding that "augmentation cannot beat baseline on DBLP" is partly an artifact of an already partly-seen test set. This is the conventional benchmark split and need not be re-split, but it should be **explicitly acknowledged** when interpreting DBLP results.

---

## Issue 6 — The corner-case design target was not met

The supervisor's criterion is **40–50% corner cases**. The project's own Layer 2 table shows the best achieved was **27.5%** (WDC `llm_aug`) and **3.3%** (DBLP `llm_aug`); no strategy reached the target on either dataset. Structurally, DBLP-Scholar offers very few borderline pairs (3–5%), so the target may be unattainable there — but this gap between design requirement and outcome should be stated plainly rather than left implicit.

---

## Minor notes

- **Decision threshold fixed at 0.5 everywhere.** Class weighting shifts the effective boundary, but no explicit threshold selection on validation is performed for the final RoBERTa classifier. (CLAUDE.md's "model selection: best classifier + threshold" refers to the AL ensemble, not the downstream model.) A validation-tuned threshold is a cheap potential gain on WDC.
- **Web query design (`{brand} {title}`) guarantees positive skew.** This was correctly diagnosed in Task 6 and motivates the Task 7 rebalancing proposal — good. Run that proposal *after* the leakage fix, not before, to avoid tuning on contaminated data.
- **"Downstream evaluation on hard subsets"** (a supervisor criterion) is partially addressed via error-class breakdowns but not as a separate hard-subset F1 metric. Consider reporting F1 restricted to corner-case test pairs.
- **Reproducibility positive:** all reported F1 values reproduce exactly from the saved predictions — good practice worth keeping.

---

## Priority action list

1. **Fix the candidate-pool filter** (exclude valid/test entities), regenerate `train_aug_llm` / `train_aug_web`, retrain and re-evaluate. *(Correctness — not optional.)*
2. **Switch to McNemar + bootstrap-F1** for all model-vs-model comparisons; correct the DBLP "indistinguishable" claim in `task6_findings.md` and the progress report.
3. **Add ≥3 seeds per configuration**, report mean ± std.
4. **Calibrate / report the WDC decision threshold** to the test prior; frame the WDC ceiling accordingly.
5. **Acknowledge DBLP split non-disjointness** in the results discussion.
6. **State the corner-case target gap** explicitly.

Items 1–3 are needed before the Task 6 conclusions and the progress report to the supervisor can be considered sound. The Task 7 rebalancing direction is well-reasoned and should follow item 1.
