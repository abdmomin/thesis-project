
## Issues found

**1. DBLP "no direct pair leakage" claim is wrong at the text level — fix before writing.** `task6_findings.md` C5 says "0 shared exact pairs, so no direct pair leakage," while `methodology_review.md` said 186 shared pairs. Both are half right: id-level shared pairs = 0, but I verified **186 train∩test pairs are textually identical** (plus 160 train∩valid, 64 test∩valid) because the DBLP/Scholar tables contain duplicate records under different ids. That's 3.2% of test pairs seen verbatim in training — genuine pair-level memorization on top of the 93% entity overlap. An examiner who checks will find this; state it as "0 id-level, 186 text-identical" in the thesis.

**2. You report population std, not sample std.** `aggregate_seeds.py` uses `pstdev` (÷n). With n=3 this understates spread ~22% (WDC baseline ±0.0122 → ±0.0149 sample). Aaron and DistillER almost certainly report sample std, and your "comparable = ±1 std overlap" convention depends on it. Switch to sample std or at minimum state the estimator.

**3. Describe the significance test precisely.** McNemar on seed-*ensemble* predictions tests the score-averaged ensemble model, not the distribution of per-seed results. It's a defensible pragmatic choice (per-seed tests with n=3 are underpowered), but the thesis should say "McNemar on seed-averaged predictions," not imply it tests the mean-F1 difference. Relatedly: you ran ~10 tests at α=0.05 with no multiple-comparison note. Your significant results (p=0.0079, 0.0098, 0.0024) survive Bonferroni, so conclusions don't change — but add the sentence.

**4. The lr change is confounded with the seed change.** Single-seed runs were lr=5e-5, final 3-seed runs lr=2e-5. So "the WDC llm_aug drop was a seed-42 artifact" is partly a hyperparameter artifact — seed 42 at 2e-5 isn't the same run. The multi-seed conclusion stands either way, but acknowledge the confound in one line rather than attributing everything to seed noise. Also, `run_multiseed.py` still defaults to lr=5e-5 — a footgun if anyone reruns without the flag; the docs say 2e-5 is the final protocol.

**5. Teacher label noise was never quantified — this is your cheapest remaining win.** The supervisor explicitly required a label-noise check, and you only did qualitative spot-checks. But on WDC you have `cluster_id` ground truth for every candidate pair (you already used it to count the 266 true matches). You can compute exact Claude-teacher accuracy/precision/recall on all seed + AL labels with a few lines of code and zero API calls. In a thesis framed around teacher/student distillation, a "teacher quality" table is almost mandatory — DistillER's whole analysis hinges on teacher noise.

**6. `significance.py --all` reproduces the wrong table.** `STANDARD_RUNS` points at single-seed run names, so the documented "full Task 6 table" command regenerates the superseded single-seed results, not the final seedmean ones. Reproducibility trap for future-you.

**7. Report the added-slice positive rate for DBLP llm_aug.** The AL slice is 41/2,100 ≈ 2% positive; the headline 16.6% is diluted by the base set. The 2% figure is actually your positive-scarcity finding at its starkest — lead with it.

**8. Corner-case thresholds are asymmetric and unjustified.** Hard positive = sim < 0.3, hard negative = sim > 0.4 (token-Jaccard). The C6 "target not met" conclusion depends on these cutoffs; add a one-paragraph justification or a small sensitivity check.

## Things you could have done differently (honest hindsight)

The positive-scarcity problem was discoverable before running the full AL pipeline — a 20-line check of the clean pool's label composition against `cluster_id` would have shown 5.6% positives upfront. It cost you a full regenerate-retrain cycle. That said, you converted it into the thesis's central finding (augmentation vs replacement), which is the right salvage.

The strongest missing experiment is a **matched replacement arm**: relabel the existing 2,500 WDC train pairs with your Claude teacher and train on those (Aaron's setting, your infrastructure, ~2.5k API calls, one 3-seed run). Right now the augmentation-vs-replacement contrast rests entirely on cross-paper comparison with different budgets and configs; one in-house run would make it airtight. If you have API budget and 2 days, this is worth more than the union runs.

DBLP-Scholar was a weak second dataset for this research question — ceiling at 0.956, non-disjoint split, structurally no hard negatives. You've documented all three facts well, but a mid-difficulty dataset (Walmart-Amazon, Abt-Buy) would have given augmentation headroom to actually show an effect. Not fixable now; frame it as a lesson-learned in the discussion.

Finally, be careful with claim strength: with 3 seeds and seed-noise of ±0.012–0.016, the honest WDC conclusion is "no effect detectable at this power," not "no effect."

## Where you stand vs the deadline

19 days left, and I found no thesis draft in the repo. Remaining supervisor items: B4/B8 union training (sets are built, zero training runs done), B6 low-resource ablation (not started), plus writing. My prioritization: **B6 first** — it's the one remaining experiment where a positive result is plausible (Aaron's Figure-4 / DistillER-D8 low-resource regime, and the supervisor asked for it explicitly). For unions, run `union_all` only, 3 seeds, both datasets, and skip the pairwise combos unless it wins — 4 combos × 3 seeds × 2 datasets is ~2 training-days you don't have. Then freeze experiments and write. The experimental record is already coherent and defensible; the biggest risk to this thesis is no longer methodology, it's time.

Short answer: only three need action, and two of those are under an hour of work. Everything else is a writing fix.

**Must be changed (not explainable away):**

1. **The C5 "no direct pair leakage" claim (#1).** You can't ignore it because as written it's factually false — 186 test pairs are text-identical to train pairs. The fix costs nothing: no retraining, no code. Just correct the claim wherever it appears (thesis, task6_findings if you reuse it) to "0 id-level shared pairs; 186 text-identical pairs due to duplicate records (~3.2% of test)". Ignoring it risks an examiner finding a false statement, which taints trust in everything else.

2. **pstdev → sample std (#2).** These numbers go directly into your comparison tables, and your "comparable = ±1 std overlap" convention against Aaron/DistillER assumes the same estimator they use. Reporting a knowingly ~22% understated std isn't something you can footnote around. It's a one-word change in `aggregate_seeds.py` (`pstdev` → `stdev`) plus rerunning the aggregation script — the means, McNemar p-values, and all conclusions are untouched, only the ± values widen slightly.

3. **Teacher label-noise quantification (#5).** This is an explicit supervisor requirement ("spot-check for label errors before training"), so skipping it is a gamble on your grader. And it's the cheapest table in the thesis: WDC cluster_id ground truth is already on disk, zero API calls, a short script. I'd treat it as mandatory.

**Handled purely in the writing (no code, no reruns):**

The test description (#3 — call it "McNemar on seed-averaged predictions" + one sentence on multiple comparisons), the lr/seed confound (#4 — one acknowledging sentence), the DBLP added-slice 2% positive rate (#7 — actually strengthens your story), the corner-case thresholds (#8 — a justification paragraph; sensitivity check only if you have time), the DBLP dataset-choice limitation, and the "no detectable effect at this power" claim-strength phrasing.

**Safe to ignore entirely:**

The `significance.py --all` footgun (#6) and the `run_multiseed.py` lr default — they're reproducibility hygiene for future reruns, not correctness of anything reported. Fix them only if you touch those scripts again. The replacement-arm experiment stays optional: valuable, but the thesis stands without it.

So the only thing resembling real work is the teacher-noise script, and it's small. Items 1 and 2 are minutes each. Everything else folds into writing you have to do anyway.