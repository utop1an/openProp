# Controlled target adaptation under source-model misspecification

Date: 2026-08-26

## Research question

The typed interaction gate's predictive e-value is valid under a frozen,
correctly specified source-model null. This benchmark asks what happens when the
deployed source model is wrong while target records remain in distribution. It
separates three failures that should not be conflated: global calibration error,
typed value permutation, and insufficient target-calibration power.

The result is a source-misspecification stress test and a controlled extension
of the gate. It remains synthetic mechanism evidence. It does not establish
safe adaptation on real observation histories.

## Fixed protocol

All seven conditions share the same source training data, target calibration
and test rows, event draws, and censoring draws within a seed. Only the deployed
source risk model changes:

- `correct_source` leaves the fitted source model unchanged;
- `rate_x2` doubles every source hazard;
- `risk_compressed` and `risk_expanded` apply strictly monotone powers to
  positive source risk, preserving every source ordering;
- `subject_cycle` permutes the three typed subject values;
- `scene_swap` swaps the two typed scene values;
- `subject_scene_permutation` applies both typed permutations.

The typed permutation wrapper uses explicit one-to-one value maps and fails
closed on unseen values. Timestamps, provenance, and outcomes are not rewritten.
The correct source model is retained as an evaluation-only reference.

Each condition uses ten seeds and all 18 typed contexts. A maximum
outcome-independent calibration pool contains 16 records per context. Nested
sizes of 2, 4, 8, and 16 per context correspond to 36, 72, 144, and 288 labels;
the same 288-record, group-disjoint target test is used at every size. Clean
calibration and deterministic 20% event-status flips are evaluated against the
same clean test. The full matrix contains 560 runs.

Run:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_source_misspecification_adaptation.py
```

The command writes
`artifacts/source_misspecification_adaptation_results.json`. The artifact
contains all runs, activation decisions, aggregate metrics, and paired
seed-cluster bootstrap inference. Nested sizes and noise variants are not
treated as independent trials.

## Methods

The benchmark compares the correct-source reference, deployed source,
unrestricted global log-risk affine calibration, the original reversal-only
interaction gate, the controlled general gate, and target-only per-context MLE.
All fitted methods receive exactly the same target calibration records.

The controlled general gate extends candidate eligibility from negative slopes
to any discovery-fitted affine correction with positive confirmation gain. It
retains identity-disjoint discovery and confirmation, the fixed 12-group typed
family, Bonferroni predictive e-values, and the child-sign heterogeneity veto.
Two additional closure rules limit unnecessary specialization:

1. a local partition cannot activate unless the global candidate first has
   predictive evidence against the deployed source;
2. a supported parent is preferred unless a discovery-fitted child partition
   also predicts confirmation outcomes better than that parent at the same
   multiplicity-controlled threshold.

The original `reversal_only` scope remains the default API behavior. The broader
`any_predictive_gain` scope must be requested explicitly.

## Main results at 288 calibration labels

The table reports mean NLL, C-index, and integrated Brier score over ten paired
seeds. Lower NLL/IBS and higher C-index are better.

| Source condition | Noise | Deployed source | Global affine | Reversal only | Controlled general | Target only |
|---|---:|---:|---:|---:|---:|---:|
| correct | 0% | 1.410/0.754/0.147 | 1.406/0.754/0.149 | 1.410/0.754/0.147 | 1.410/0.754/0.147 | 1.438/0.744/0.154 |
| rate x2 | 0% | 1.582/0.754/0.155 | 1.406/0.754/0.149 | 1.582/0.754/0.155 | 1.423/0.749/0.149 | 1.438/0.744/0.154 |
| compressed risk | 0% | 1.460/0.754/0.153 | 1.406/0.754/0.149 | 1.460/0.754/0.153 | 1.420/0.754/0.150 | 1.438/0.744/0.154 |
| expanded risk | 0% | 1.605/0.754/0.165 | 1.406/0.754/0.149 | 1.605/0.754/0.165 | 1.409/0.753/0.150 | 1.438/0.744/0.154 |
| scene swap | 0% | 2.034/0.480/0.302 | 1.635/0.515/0.213 | 1.753/0.549/0.234 | **1.482/0.733/0.158** | 1.438/0.744/0.154 |
| subject cycle | 0% | 1.516/0.705/0.173 | 1.492/0.705/0.170 | 1.516/0.705/0.173 | 1.448/0.738/0.155 | **1.438/0.744/0.154** |
| subject x scene | 0% | 2.126/0.435/0.325 | 1.623/0.565/0.209 | 1.783/0.528/0.248 | **1.558/0.692/0.180** | 1.438/0.744/0.154 |

For scene swap, the controlled general gate improves deployed-source NLL by
0.552 (95% CI [0.432, 0.650]) and C-index by 0.253 [0.219, 0.286], with 10/10
paired wins. It improves over unrestricted global affine by 0.152
[0.046, 0.234] NLL with 8/10 wins and by 0.219 [0.192, 0.247] C-index with
10/10 wins.

For the joint permutation, it improves deployed-source NLL by 0.568
[0.436, 0.680] and C-index by 0.257 [0.212, 0.297], with 10/10 wins. Relative to
global affine, the NLL gain is 0.065 [-0.036, 0.143] with 8/10 wins, while the
C-index gain is 0.127 [0.090, 0.164] with 10/10 wins. These results establish the value
of typed local structure when a single global calibration cannot restore risk
ordering.

The upgraded group-level veto and dense predictive closure increase subject-cycle
activation to 9/10 clean runs. Mean NLL/C-index/IBS reach 1.448/0.738/0.155,
close to target-only's 1.438/0.744/0.154. Controlled-general versus target-only
differences are -0.010 [-0.043, 0.024] NLL improvement and -0.006
[-0.018, 0.005] C-index improvement; both intervals cross zero. This is improved
power, not evidence of dominance over target-only estimation.

## Calibration errors and ranking preservation

Strictly monotone source errors preserve C-index by construction. The original
reversal-only gate correctly refuses all such repairs, but therefore leaves NLL
miscalibration unchanged. The controlled general gate reduces mean NLL by
0.159, 0.041, and 0.196 for rate scaling, compressed risk, and expanded risk.
Its mean C-index changes by -0.005, 0.000, and -0.001, respectively. The small
losses arise from occasional over-refinement after global evidence, not from the
global affine correction itself.

Unrestricted global affine is stronger for these purely global errors. At 20%
label noise, controlled general is worse than global affine by 0.033 NLL
[0.023, 0.042] for compressed risk. The controlled method is therefore not a
replacement for unrestricted calibration when global misspecification is known
in advance; its purpose is typed structure search with an explicit fail-closed
boundary.

## Activation and structural recovery

At clean 288-label support, correct source remains inactive in 10/10 seeds. At
20% calibration noise, it remains inactive in 9/10 and selects a global repair
once. Noise-corrupted calibration is not distributed according to the source
null, so the nominal source-null guarantee does not cover this case.

Clean scene swap selects the scene partition in 10/10 seeds. The joint
permutation selects the pairwise partition in 6/10 and the scene partition in
4/10. Rate scaling and expanded
risk each choose the intended global partition in 8/10, with two over-refined
selections. Compressed risk activates globally in 7/10 and stays inactive in
3/10. Subject cycle selects subject in 9/10 and stays inactive once. Thus
predictive performance is stronger than exact structural
recovery; the two must not be reported as equivalent.

## Noise stress result

At 20% label flips, scene swap still improves deployed-source NLL by 0.539
[0.427, 0.633] and C-index by 0.253 [0.216, 0.289], with 10/10 wins. The joint
permutation improves NLL by 0.577 [0.469, 0.665] and C-index by 0.253
[0.215, 0.287], also with 10/10 wins. However, controlled general is worse than
target-only on the joint permutation, and correct-source false activation is no
longer zero. The supported claim is resilience of mean repair gains in these
two mechanisms, not arbitrary noisy-label safety.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Typed controlled adaptation repairs non-global source misspecification that a global affine model cannot. | Scene improves over global affine on NLL in 8/10 and C-index in 10/10; joint improves NLL in 8/10 with an interval crossing zero and C-index in 10/10 with its interval excluding zero. | Supported most strongly for risk ordering in the paired synthetic mechanisms at 288 labels. |
| The generalized gate preserves a correct deployed source. | Clean correct-source condition is inactive in 10/10 seeds at 288 labels and has identical metrics. | Supported for this clean synthetic setting, not universally. |
| The method identifies the true misspecification structure. | Scene is exact in 10/10 and subject cycle in 9/10, but joint pairwise is 6/10 and monotone conditions sometimes over-refine. | Contradicted as a universal exact-recovery claim. |
| The method dominates unrestricted global calibration. | It decisively restores ranking for typed permutations, but joint NLL uncertainty crosses zero and purely global calibration errors favor the unrestricted model. | Unsupported as a uniform metric-dominance claim. |
| The method dominates target-only calibration. | Scene swap and subject cycle are statistically tied with target-only, while the joint permutation favors target-only. | Unsupported. |
| The source-null guarantee survives calibration-label corruption. | One of ten noisy correct-source decisions activates. | Unsupported. |
| OpenProp now has real-world safe adaptation evidence. | Every mechanism and outcome in this benchmark is synthetic. | Unsupported and must not be claimed. |

## Reviewer-style self-audit

- **Contribution — needs stronger external evidence:** the benchmark exposes a
  previously unstated source-null assumption and introduces a closed typed
  repair hierarchy, but it is still a synthetic method study.
- **Writing clarity — pass for reproducibility:** source variants, split,
  candidate family, closure rules, baselines, metrics, and artifact are explicit.
- **Experimental strength — mixed:** typed permutations show large paired gains
  over a strong global calibration baseline on ranking; joint NLL evidence and
  exact structure recovery remain weaker.
- **Evaluation completeness — needs new experiment:** unseen typed values and
  higher-order interactions are now covered by the linked open-world benchmark;
  non-affine local errors, broader noise processes, and official longitudinal
  histories remain untested.
- **Method soundness — needs new experiment:** predictive e-values rely on a
  correctly specified source null, while the general gate's local descent also
  relies on finite-sample parent-child comparisons. External validation must
  test whether the added complexity has positive net value.

## Claim boundary and next experiment

This benchmark supports a narrow conclusion: with enough clean calibration,
closed typed structure search can repair synthetic source permutations that
global calibration cannot, while clean correct-source behavior remains
fail-closed in the tested seeds. It does not support real-world effectiveness,
universal false-positive control, exact mechanism identification, or dominance
over target-only estimation.

The unseen-value and higher-order mechanism experiment is now completed in the
[open-world higher-order benchmark](open-world-higher-order-adaptation-benchmark.md),
including a calibration-support fallback. The next synthetic step is non-affine
local error. The main submission blocker remains an official or independently
audited longitudinal benchmark with natural observation histories and held-out
current truth.

## 2026-08-26 non-affine follow-up

The previously listed non-affine next step is now complete. The
[local non-affine stress benchmark](non-affine-adaptation-stress-benchmark.md)
adds 240 paired runs over local saturation, folded ordering, and a smooth typed
interaction bump. It shows low power for mild saturation, weak folded-order
recovery, and noisy false activation: both BIC and non-BIC typed gates activate
in 5/10 correct-source controls with 20% calibration label flips. The next
method step is therefore robust, complexity-controlled nonlinear calibration.
The main external-evidence blocker remains unchanged.

