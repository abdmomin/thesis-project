# Automatic Training Set Expansion for Entity Matching Using LLM-Based Labeling and Web Retrieval

Master's thesis project, Data and Web Science Group, University of Mannheim.

- **Author:** Abdullah Momin
- **Supervisor:** Prof. Dr. Christian Bizer
- **Submitted:** 31st July 2026
- **Thesis PDF:** [`thesis-docs/thesis-paper/thesis.pdf`](thesis-docs/thesis-paper/thesis.pdf)

## What this project is about

Transformer-based entity matchers (Ditto-style) need labeled record pairs, and labeling is the expensive part. This thesis asks whether an **existing human-labeled training set can be expanded automatically**, without further human labeling, and whether the expansion improves a `roberta-base` student model. Three expansion strategies are compared on two benchmarks:

1. **String-based augmentation** – Ditto-style label-preserving perturbations (span deletion, span shuffle, attribute drop) of existing pairs.
2. **LLM-based augmentation** – embedding-based blocking builds a candidate pool of *new* pairs, committee-based active learning selects the most uncertain ones, and an LLM teacher (Claude) labels them (pipeline follows Steiner & Bizer).
3. **Web retrieval augmentation** – Tavily web search retrieves new offers/records for training entities, and the LLM teacher extracts and labels the resulting pairs.

**Benchmarks:** WDC Products (80% corner cases, unseen test set, small training split) and DBLP-Scholar.

## Key findings

- **At the full labeling budget, no strategy beats the baseline** (3 seeds, McNemar on seed-ensemble predictions). WDC Products: string +0.016 F1 (p=0.244, n.s.), LLM +0.002, web −0.001. DBLP-Scholar: only string augmentation has a significant effect, and it is a small *regression* (−0.008, p=0.0079).
- **Why: positive scarcity.** Once expansion is honestly entity-disjoint, the candidate pool of a closed benchmark contains almost no undiscovered matches (5.6% on WDC Products, none on DBLP-Scholar). The teacher is not the problem (97.1% label accuracy, F1 0.862 against cluster ground truth).
- **The picture flips in the low-resource regime.** With the base training set halved, the same LLM-labeled additions give a significant gain (0.606 → 0.626 F1, p=0.0002, Bonferroni-surviving), and a size- and class-balance-matched string control gains nothing – the benefit is teacher-label signal, not data volume. At a quarter budget the additions rescue a training set that otherwise collapses.
- **Web-specific failure mode:** retrieved records solve the content problem but introduce a text-style mismatch; rebalancing the web slice and adding verified generated hard negatives made results significantly *worse* (−0.029, p=0.0024).

| Dataset | Run | F1 (3-seed mean ± std) |
|---|---|---|
| WDC Products | baseline (class-weighted) | 0.659 ± 0.015 |
| WDC Products | string_aug | **0.675 ± 0.016** (n.s.) |
| WDC Products | llm_aug | 0.661 ± 0.020 |
| WDC Products | web_aug | 0.658 ± 0.015 |
| WDC Products | union (string+llm+web) | 0.677 ± 0.015 (n.s. vs string) |
| DBLP-Scholar | baseline | **0.9559 ± 0.0019** |
| DBLP-Scholar | string_aug | 0.9483 ± 0.0020 (worse, p=0.0079) |
| DBLP-Scholar | llm_aug / web_aug | 0.9566 / 0.9567 (ties) |

## Repository structure

```
├── data/
│   ├── raw/                      # downloaded benchmarks (gitignored)
│   └── processed/<dataset>/      # Ditto-format splits + all augmented training files
├── experiments/
│   ├── models/                   # trained checkpoints (gitignored)
│   └── results/                  # metrics JSONs, per-pair predictions, aggregates
├── notebooks/                    # LLM/web augmentation pipelines + analysis + figures
├── src/
│   ├── data_prep/                # download, preprocess, entity splits, low-resource
│   │                             #   subsets, union sets, DBLP hard-negative mining
│   ├── augmentation/             # Ditto-style string augmentation
│   ├── baseline/                 # train, evaluate, multi-seed runner, threshold selection
│   ├── analysis/                 # significance tests, seed aggregation, three-layer
│   │                             #   evaluation, teacher label quality, leakage checks
│   └── utils/                    # metrics (positive-class P/R/F1)
├── thesis-docs/thesis-paper/     # LaTeX source, figures, compiled thesis.pdf
└── requirements.txt
```

## Setup

Python 3.12 recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API keys (only needed to re-run the LLM/web labeling pipelines; training and evaluation work without them) go into a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=...   # LLM teacher (Claude)
TAVILY_API_KEY=...      # web search (Task 5 / 7)
```

## Reproducing the experiments

### 1. Data

```bash
python src/data_prep/download_datasets.py --dataset all
python src/data_prep/preprocess.py --dataset all
```

This produces the Ditto-format splits `data/processed/<dataset>/{train,valid,test}.txt`. Format: one pair per line, `serialized_left \t serialized_right \t label`, records serialized as `COL <attr> VAL <value> ...`.

### 2. Augmented training sets

All augmented files are committed under `data/processed/<dataset>/`, so this step is optional.

- **String augmentation:** `python src/augmentation/run_string_aug.py` → `train_aug_string.txt`
- **LLM augmentation:** notebooks `04` (blocking + pool filtering), `05` (seed labeling), `06` (active learning loop) → `train_aug_llm.txt`. Requires `ANTHROPIC_API_KEY`.
- **Web augmentation:** notebooks `07_task5_step1` – `step4` → `train_aug_web.txt`. Requires both API keys.
- **Second iteration (Task 7):** `build_union_augmentation.py`, `build_low_resource.py`, `build_sub50_string_control.py`, `mine_dblp_hard_negatives.py` (all in `src/data_prep/`), and notebook `07b` for the web-v2 hard-negative variants.

All pipelines restrict augmentation to **training-split entities only** (`src/data_prep/entity_splits.py`). Verify with:

```bash
python src/analysis/check_split_disjointness.py
```

The pass criterion is zero added pairs touching a held-out entity.

### 3. Training and evaluation

Single run (defaults are MPS-friendly: batch 8, grad-accum 4):

```bash
python src/baseline/train_baseline.py --dataset wdc-products
python src/baseline/evaluate.py --dataset wdc-products --run baseline_cw --save_preds
```

Final protocol – 3 seeds (13/42/87), lr 2e-5, class weighting on WDC Products:

```bash
python src/baseline/run_multiseed.py --dataset wdc-products --run <run_name>
```

### 4. Aggregation and significance

```bash
# mean ± std over seeds + seed-ensemble predictions
python src/analysis/aggregate_seeds.py --dataset wdc-products --runs baseline_cw string_aug_cw llm_aug_cw web_aug_cw ...

# paired McNemar (exact) + bootstrap F1 CIs + Cohen's h, on seed-ensemble predictions
python src/analysis/significance.py --all
```

### 5. Analysis (three-layer evaluation)

```bash
python src/analysis/task6_compute.py            # layer 1-3: metrics, training set characteristics, error analysis
python src/analysis/teacher_label_quality.py    # teacher audit against WDC cluster ground truth
python src/baseline/select_threshold.py         # WDC threshold/prior-shift analysis
```

Notebook `08` explores the analysis outputs; notebook `09` regenerates every figure in the thesis.

## Where each thesis number lives

| Thesis table/figure | Backing file(s) in `experiments/results/` |
|---|---|
| Main results, per-seed values (Tables 6.1–6.3, C.2) | `<run>_s<seed>_<dataset>_test.json`, `seed_aggregate_<dataset>.json` |
| Significance (McNemar/bootstrap columns) | `significance_tests.json`, `*_seedmean_*_test_preds.jsonl` |
| Training set characteristics (Table 6.4, C.3) | `task6_dataset_characteristics.json` |
| Error analysis (Section 6.6, Appendix D) | `task6_error_analysis.json`, `*_preds.jsonl` |
| Teacher label quality (Section 6.5) | `teacher_label_quality_wdc.json` |
| Threshold calibration (Table C.1) | `threshold_calibration_baseline_cw_wdc-products.json` |
| Split hygiene / leakage check (Table 3.2) | `split_disjointness.json` |

Per-pair predictions and scores of every run are stored, so all reported metrics, significance tests, and error analyses can be recomputed **without retraining**. The only non-deterministic component is the hosted LLM teacher; all teacher outputs (labels and reasoning) were persisted at labeling time under `data/processed/<dataset>/`.

## Notes

- The original [Ditto codebase](https://github.com/megagonlabs/ditto) was used as an architectural reference only (a local copy is gitignored); the thesis re-implements the equivalent architecture with current Hugging Face APIs.
- Model checkpoints and raw benchmark downloads are gitignored; everything needed to re-train from scratch is scripted.

## References

The two closest related works, discussed and compared against in detail in the thesis:

- Steiner & Bizer: *Labeling Training Data for Entity Matching Using Large Language Models* (arXiv:2606.28823)
- Zeakis et al.: *DistillER: Knowledge Distillation in Entity Resolution with Large Language Models* (arXiv:2602.05452)
