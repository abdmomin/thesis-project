# Comparison to Steiner & Bizer (Aaron's paper) — Task 7 / B9

**Paper:** Aaron Steiner & Christian Bizer, *"Labeling Training Data for Entity Matching Using
Large Language Models"*, arXiv 2606.28823 (Jun 2026). `thesis-docs/2606.28823v1.pdf`.

> This is the same **"Steiner & Bizer 2026"** whose active-learning pair-selection recipe this
> thesis's Task 4 (`llm_aug`) implements. His **"Active learning (ML)"** variant (feature-ensemble
> committee over similarity features) == this thesis's `llm_aug` pipeline.

---

## 1. The central methodological difference — augmentation vs replacement

| | Aaron (Steiner & Bizer) | This thesis |
|---|---|---|
| **Goal** | *Replace* the human benchmark labels with LLM-teacher labels ("fitness for use": does a student trained on machine-labeled data match one trained on the benchmark?) | *Augment* — add **new** LLM-labeled pairs on top of the existing human `train.txt` |
| **Candidate pool** | **Keeps** the benchmark positive pairs — his pools recover **99.56–99.92%** of the original positives | **Excludes** all known (train/valid/test) pairs → only *new* pairs remain |
| **Consequence** | Pool has a healthy positive rate; LLM relabels the real matches | Entity-disjoint pool is **positive-starved** (WDC 5.6% true matches, DBLP ~0%) — see the positive-scarcity finding |
| **Student** | Ditto (RoBERTa), XGBoost, Qwen3 SLMs | roberta-base |
| **Teacher** | GPT-5.2 (main), Qwen 3.6 Plus, Kimi K2.6 | Claude (`claude-sonnet-4-6`) |

**This is the thesis's novel angle and the explanation for the supervisor's surprise** that
`llm_aug` F1 "hardly changes": under the *augmentation* framing on a closed benchmark, blocking
surfaces almost no genuinely new positives, so the LLM pairs are near-all-negative and provide
little signal (and, on clean entity-disjoint data, `llm_aug` slightly *hurts* WDC).

## 2. Datasets & setup — what is and isn't comparable

| | This thesis | Aaron | Comparable? |
|---|---|---|---|
| WDC Products version | 80%-CC hard, **unseen** test (4,500 pairs; 75% of negatives `is_hard_negative`) | Same 80%-CC hard, unseen-entity test | ✅ same benchmark + test |
| WDC **training split** | **small (2,500)** | **large (~16–20k)** | ⚠️ different size — the main driver of the WDC gap |
| DBLP-Scholar | full benchmark train (17,223), test 5,742 | benchmark train, published test split | ✅ directly comparable |
| Metric | F1 (mean ± std, 3 seeds) | F1 (mean ± std, 3 seeds {42,52,62}) | ✅ same |

## 3. Head-to-head (RoBERTa / Ditto student, F1)

Aaron's numbers from his Table 1 (Ditto student, GPT-5.2 teacher). This thesis's numbers are the
final clean, entity-disjoint results. **`llm_aug` ↔ Aaron's "AL (ML)"** is the most direct
method match (both feature-ensemble active learning).

This thesis numbers are **3-seed mean ± std** (lr=2e-5). `llm_aug` ↔ Aaron **"AL (ML)"**.

| Dataset | Metric | This thesis (3-seed) | Aaron (Ditto) | Note |
|---|---|---|---|---|
| **WDC Products** | benchmark / baseline | 0.659 ± 0.015 (small, 2.5k) | **0.719** (large, ~16k) | gap = training size, not test |
| | best augmentation | 0.675 ± 0.016 (`string_aug`, n.s.) | 0.722 (AL-Ditto) / 0.707 (AL-ML) | ours: no strategy robustly beats baseline |
| | LLM-AL method | 0.661 ± 0.020 (`llm_aug`, tie) | 0.707 (AL-ML) | Aaron keeps positives (replacement); ours doesn't |
| **DBLP-Scholar** | benchmark / baseline | **0.9559 ± 0.0019** | 0.956 | ✅ essentially identical |
| | LLM-AL method | 0.9566 ± 0.0006 (`llm_aug`, tie) | 0.938 (AL-ML) | ours slightly higher (full train + augment) |
| | Δ of aug vs benchmark | −0.008 to +0.001 (all n.s. except string) | −1.78 F1 (AL-Ditto) | both: LLM-labeled data ≤ benchmark on DBLP |

## 4. Reconciling the supervisor's remark ("Aaron's WDC is lower than yours")

The remark and the raw numbers only look contradictory until the **training-budget axis** is fixed:

- **At Aaron's budget (~16k labels):** his WDC benchmark = 0.719 > our small-train 0.659 — simply
  ~8× more training data. (His Figure 4 shows WDC only plateaus at ~16k; it *crosses* 0.719 around
  15k labeled pairs.)
- **At our budget (~2,500 labels):** Aaron's machine-labeled WDC sits ≈0.55–0.62 (his curve rises
  from ~0.50 at 1k). So at a *matched* low budget, **our benchmark-trained WDC (0.659) is
  competitive with / above Aaron's machine-labeled result** — which is exactly the supervisor's
  point that "his WDC results are lower than yours and he requires much more training data."

Both statements are true; the comparison must be **budget-aware**. DBLP needs no such caveat
(same full train → 0.958 vs 0.956, indistinguishable).

## 5. Implications for the thesis & Task 7

- The **augmentation-vs-replacement** distinction is the framing to lead with in the write-up and
  the direct contrast to Aaron.
- The supervisor's **low-resource ablation (B6)** and **scale-up (B8)** experiments map exactly
  onto Aaron's Figure-4 budget axis: shrinking/growing our WDC training budget lets us place this
  thesis's curve against his, and test whether augmentation helps more at small budgets.
- **Consistency to highlight:** on DBLP-Scholar both works find LLM-labeled data ≤ the human
  benchmark near the ceiling; on the product benchmark both find it the harder domain (shared-token
  variants) — the same qualitative picture as DistillER (Zeakis et al.).

## 6. Open items before finalizing the table
- (Optional) re-run the WDC baseline at Aaron's Ditto config (batch 64, 50 epochs) and/or the large
  training split to produce a like-for-like row, if the supervisor wants matched-config numbers.
- Fill the AL-ML column exactly from Aaron's Table 1 (WDC 70.65±0.42, DBLP-Scholar 93.81±0.38).
- Add 3-seed mean±std for this thesis's rows once the multiseed runs finish.
