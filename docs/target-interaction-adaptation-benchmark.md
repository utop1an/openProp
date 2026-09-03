# Multiplicity-controlled typed interaction adaptation

Date: 2026-08-26

## Research question

The local adaptation stress test exposed a structural failure: a subject-only or
scene-only gate can improve aggregate performance under an XOR shift while
damaging stable contexts. This experiment asks whether target calibration can
select among global, typed main-effect, and sparse pairwise repairs without
using target-test labels, generator change indicators, or a test-selected axis.

The intended contribution is not unrestricted target fine-tuning. It is a
fail-closed adaptation rule that makes its representational scope and
multiplicity cost explicit. All evidence remains synthetic mechanism
validation, not real or semi-real grounding evidence.

## Method

### Fixed typed candidate family

The candidate family is declared before evaluation and contains 12 groups:

- one global group;
- three `subject_type` groups;
- two `scene` groups;
- six `subject_type x scene` groups.

These partitions preserve typed values and do not embed heterogeneous fields in
one latent space. The generator-defined changed-context set is absent from the
method and is used only for held-out subgroup reporting.

### Discovery, confirmation, and family-wise control

Calibration group IDs are hash-ranked within each complete five-field context.
One third of every context is used to fit candidate affine log-risk adapters;
the other two thirds are identity-disjoint confirmation data. A candidate must
have a negative discovery slope and positive confirmation NLL gain.

For a fixed discovery-fitted candidate, the exponential survival likelihood
ratio on confirmation data is a predictive e-value under the source-model null
(with non-informative censoring). The reciprocal e-value is compared with
`0.05 / 12 = 0.004167`. Bonferroni therefore controls the probability of any
false candidate activation at 0.05 when the frozen source null is correctly
specified. Because the source model is estimated rather than known, the
benchmark also reports empirical in-distribution activation instead of treating
the nominal guarantee as sufficient evidence.

### Hierarchical heterogeneity veto

A pooled parent repair is unsafe when its finer typed cells have opposing risk
directions. Therefore, a significant global or main-effect group is vetoed when
any available refinement contains both negative and nonnegative discovery
slopes. Among significant, non-vetoed partitions, the method selects the largest
confirmation NLL gain. Only accepted groups in that partition are refitted on
the complete calibration set; every other group returns the frozen source
prediction exactly.

No threshold, candidate partition, split seed, or optimizer setting is chosen
on target test.

## Frozen evaluation protocol

The experiment uses ten seeds, all 18 typed contexts, and the five paired
mechanisms from the local stress benchmark. Each condition has 32 target records
per context. The maximum outcome-independent calibration pool contains 16 per
context, leaving a fixed 288-record, group-disjoint test for every calibration
size. Nested calibration sizes are 2, 4, 8, and 16 per context (36, 72, 144, and
288 total labels). Both clean calibration and deterministic 20% event-status
flips are evaluated against the same clean target test.

Baselines are the frozen source model, global sign gate, subject gate, scene
gate, and target-only per-context MLE. NLL, C-index, and integrated Brier score
use the same horizons as earlier studies. Uncertainty uses 20,000 deterministic
seed-cluster bootstrap resamples and two-sided exact sign tests. Nested sizes
and noise variants are not treated as independent trials.

Run:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_target_interaction_adaptation.py
```

The command writes `artifacts/target_interaction_adaptation_results.json` with
all 400 seed-condition-size-noise runs, candidate diagnostics, subgroup metrics,
aggregate summaries, and paired inference.

## XOR result

The source model has NLL/C-index/IBS 1.983/0.504/0.272, changed-context C-index
0.269, and stable-context C-index 0.747. The table shows the hierarchical method
as calibration support increases. `Exact` counts seeds whose activated typed
cells exactly equal the three changed XOR cells. `FP` and `FN` count cells
summed across ten seeds.

| k | Labels | Noise | NLL | C-index | IBS | Changed C | Stable C | Exact | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36 | 0% | 1.904 | 0.524 | 0.264 | 0.314 | 0.747 | 0/10 | 0 | 27 |
| 2 | 36 | 20% | 1.920 | 0.524 | 0.267 | 0.308 | 0.749 | 0/10 | 1 | 28 |
| 4 | 72 | 0% | 1.800 | 0.580 | 0.236 | 0.425 | 0.742 | 1/10 | 1 | 20 |
| 4 | 72 | 20% | 1.794 | 0.563 | 0.241 | 0.395 | 0.750 | 0/10 | 1 | 22 |
| 8 | 144 | 0% | 1.570 | 0.689 | 0.181 | 0.618 | 0.747 | 3/10 | 0 | 9 |
| 8 | 144 | 20% | 1.604 | 0.680 | 0.181 | 0.603 | 0.745 | 2/10 | 1 | 11 |
| 16 | 288 | 0% | 1.500 | 0.732 | 0.157 | 0.705 | 0.747 | 8/10 | 0 | 2 |
| 16 | 288 | 20% | 1.606 | 0.696 | 0.171 | 0.631 | 0.747 | 3/10 | 0 | 7 |

At `k=16` without label noise, the hierarchy improves NLL by 0.483 (95% CI
[0.409, 0.550]), C-index by 0.228 [0.209, 0.247], and IBS by 0.115 [0.107,
0.123] relative to source. All three metrics improve in 10/10 seeds (exact sign
p=0.001953). Stable-context predictions are exactly unchanged in all ten seeds.

The strongest single-axis comparison is the subject gate, whose C-index is
0.622 and whose stable-context C-index falls to 0.691. The hierarchy improves
overall C-index by 0.111 [0.084, 0.138] and stable-context C-index by 0.056
[0.042, 0.072], with 10/10 wins. NLL and IBS gains are 0.202 [0.138, 0.270]
and 0.047 [0.029, 0.069]. Thus the interaction mechanism resolves the specific
subgroup-harm failure that motivated it.

The target-only per-context model reaches C-index 0.741 and NLL 1.476. Its small
mean advantage over the hierarchy is not established for clean calibration:
hierarchy-minus-target C-index is -0.009 [-0.024, 0.004], and NLL reduction is
-0.024 [-0.095, 0.017]. Under 20% label flips, target-only is stronger on mean
NLL and IBS. The hierarchy's advantage is preservation of non-activated source
groups, not universal dominance over a flexible target estimator.

## Noise and power are the main limitation

With 20% calibration-label flips at `k=16`, the hierarchy still improves source
NLL/C-index/IBS by 0.377/0.192/0.100 with 10/10 paired wins and keeps stable
C-index exactly unchanged. However, exact XOR recovery falls from 8/10 to 3/10
because seven true cells are missed across seeds. The method is conservative:
label noise primarily reduces power, but does not justify claiming robustness to
arbitrary annotation or observation error.

## Local-shift and in-distribution audit

At `k=16`, the hierarchy discovers simple main effects without being told the
correct axis:

| Shift | Noise | Source NLL/C/IBS | Hierarchy NLL/C/IBS | Correct-axis gate NLL/C/IBS | Stable C: source/hierarchy | Exact cells |
|---|---:|---|---|---|---|---:|
| cup only | 0% | 1.789/0.583/0.233 | 1.434/0.750/0.143 | 1.433/0.749/0.143 | 0.758/0.758 | 10/10 |
| cup only | 20% | 1.789/0.583/0.233 | 1.437/0.749/0.144 | 1.436/0.750/0.143 | 0.758/0.758 | 10/10 |
| busy only | 0% | 1.838/0.528/0.253 | 1.341/0.698/0.175 | 1.306/0.704/0.171 | 0.680/0.680 | 9/10 |
| busy only | 20% | 1.838/0.528/0.253 | 1.442/0.667/0.188 | 1.310/0.702/0.170 | 0.680/0.680 | 5/10 |

For both clean local shifts, hierarchy-versus-source improvements have 10/10
wins on NLL, C-index, and IBS. The method matches the oracle-declared subject
axis but loses power relative to the oracle-declared scene axis under label
noise. This is the price of axis discovery plus multiplicity control.

Across the 40 clean in-distribution seed-size decisions, the hierarchy never
activates. Across 40 noisy decisions it activates once, at `k=8`, changing one
typed cell. These decisions are nested and must not be presented as 80
independent Bernoulli trials. The observed false activation also prevents a
claim of empirical zero false positives, despite nominal family-wise control.

## Ablation interpretation

- Removing pairwise candidates reproduces the stable-subgroup harm under XOR.
- Removing the heterogeneity veto permits a scene parent to win confirmation
  likelihood while changing the opposed `cup,busy` cell in the fixed unit-test
  fixture.
- Removing independent confirmation yields the high-power simple sign gates,
  but those gates false-activate under low support and have no family-wise
  control across searched groups.
- Requiring independent confirmation creates a visible power curve: exact XOR
  recovery is 0/10, 1/10, 3/10, and 8/10 at clean `k=2,4,8,16`.

## Completed source-null follow-up

The frozen-source-null assumption is now tested directly in the
[source-misspecification benchmark](source-misspecification-adaptation-benchmark.md).
That follow-up preserves this benchmark's reversal-only path and adds a separate,
opt-in general predictive scope with global-first and parent-child closure. It
repairs typed scene and joint source permutations that global affine calibration
cannot, while exposing incomplete exact recovery, occasional
over-refinement of global calibration errors, and one noisy correct-source false
activation. The follow-up narrows the source-null risk; it does not remove it.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Calibration-only typed interactions can repair the benchmark XOR failure without harming stable contexts. | At clean `k=16`, C-index rises 0.504 to 0.732, stable C remains 0.747, and hierarchy beats both single axes in 10/10 C-index comparisons. | Supported for the paired synthetic generator at 288 labels. |
| The method automatically discovers simple local axes. | At clean `k=16`, cup cells are exact in 10/10 seeds and busy cells in 9/10; hierarchy nearly matches oracle-declared gates. | Supported for the two tested typed axes. |
| Multiplicity control eliminates false activation. | One noisy ID decision activates one cell. The theoretical bound also assumes the frozen source null is correctly specified. | Contradicted as an empirical zero-error claim. |
| The method is robust to noisy target labels. | Mean gains remain at 20% flips, but exact XOR recovery falls to 3/10 and target-only has better NLL/IBS. | Partial support only; noise mainly exposes low power. |
| OpenProp now has real-world safe adaptation evidence. | All mechanisms and labels are synthetic. | Unsupported and must not be claimed. |

## Reviewer-style self-audit

- **Contribution:** the nontrivial result is controlled typed interaction
  selection that fixes an observed subgroup-safety failure, not another
  unrestricted target fine-tuner.
- **Clarity:** candidate count, split, e-value threshold, veto, selection, and
  refit boundary are specified; a formal notation section is still needed in
  the manuscript.
- **Experimental strength:** the hierarchy decisively beats single-axis gates on
  XOR, but does not beat target-only MLE and is evaluated only synthetically.
- **Evaluation completeness:** source misspecification now has a paired stress
  test, but unseen typed values, higher-order structures, non-affine local errors,
  and semi-real histories remain necessary.
- **Method soundness:** test labels and generator change indicators are excluded;
  the main risks are source-null misspecification, reduced power, and the
  restricted pairwise candidate family.
