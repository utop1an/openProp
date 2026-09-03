# Open-world value support and higher-order target adaptation

## Research question

This benchmark asks whether calibration-only typed adaptation remains controlled
when target data contain values absent from source training and when the target
mechanism requires a genuine three-way interaction. It is a synthetic mechanism
validation, not evidence of real-world open-world grounding.

The experiment addresses four questions separately:

1. Can a target-calibrated novel value be repaired without changing stable
   source-supported values?
2. Does a test-only novel value fail closed instead of inheriting an unrelated
   repair from a coarser partition?
3. Can an unknown sparse hierarchy distinguish pairwise structure from a true
   three-way mechanism?
4. How do sample size, calibration-label noise, and a discovery complexity
   screen change structural recovery and predictive performance?

## Protocol

Each typed context has five values: property, subject type, relation, context
object, and scene. Source training contains 27 contexts formed by three known
subjects, three objects, and three scenes. The target adds nine `bottle`
contexts and nine `plate` contexts:

- `source_seen`: all 27 source contexts also occur in target calibration;
- `target_calibrated_novel`: `bottle` is absent from source training but its
  nine contexts occur in target calibration;
- `target_uncalibrated_novel`: `plate` is absent from both source training and
  target calibration, and its nine contexts occur only in target test.

Novel values remain typed strings. They are never rewritten as `unknown`.
Target calibration is predeclared before outcome inspection. The maximum pool
contains 48 examples from each of 36 eligible contexts; nested subsets contain
12, 24, or 48 examples per context, or 432, 864, and 1,728 labels in total.
Every setting uses the same 2,592-example test within a seed. Test includes 48
held-out examples from every calibration-eligible context and all 96 examples
from each test-only `plate` context.

Ten fixed seeds are paired across five mechanisms:

| Condition | Changed target mechanism | Intended role |
|---|---|---|
| `open_world_control` | none | clean false-activation control |
| `calibrated_novel_subject_reversal` | all `bottle` contexts | novel-value adaptation |
| `uncalibrated_novel_subject_reversal` | all `plate` contexts | identifiability negative control |
| `pairwise_subject_scene_xor` | known subject-by-scene cells | pairwise structural control |
| `three_way_subject_object_scene_latin` | known cells satisfying a 3-by-3-by-3 Latin rule | genuine three-way test |

The Latin rule changes a context iff the subject, object, and scene indices sum
to zero modulo three. Unlike binary parity under an affine local adapter, this
mechanism cannot be represented exactly by any pairwise partition in the
declared family.

For each calibration size, adapters are fitted on an identity-disjoint discovery
third and tested for predictive gain on the confirmation two-thirds. The full
hierarchy searches global, three main-effect, three pairwise, and one three-way
partition. Its 80 candidate groups share a Bonferroni family. Generalized cells
may fit an affine log-risk map or an intercept-only hazard when the cell has one
source-risk level. A discovery BIC/MDL screen, confirmation likelihood-ratio
e-values, child-sign heterogeneity veto, and parent-child predictive closure are
all frozen before target test evaluation.

Prediction has an additional support boundary: if any typed feature value was
absent from target calibration, the hierarchy returns the deployed source model
before partition routing. This prevents a selected global or object-by-scene
repair from silently extrapolating to `plate`. It is conservative: a real shift
confined to `plate` remains unrecoverable without calibration evidence.

The compared methods are the deployed source model, unrestricted global affine
calibration, pairwise hierarchy, full three-way hierarchy, full hierarchy without
the BIC/MDL screen, a declared-three-way candidate ablation, target-only
per-context maximum likelihood, and a target-hazard oracle. Metrics are held-out
negative log-likelihood (NLL), concordance index (C-index), and integrated Brier
score (IBS). Twenty-percent deterministic event-label flips are applied only to
maximum-size calibration, never to test. Paired uncertainty uses 20,000
seed-cluster bootstrap resamples and two-sided exact sign tests.

Run the frozen experiment with:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_open_world_adaptation.py
```

The command writes `artifacts/open_world_adaptation_results.json`. The artifact
contains all 200 protocol runs, support and changed/stable slices, activation
diagnostics, aggregate metrics, paired deltas, bootstrap intervals, and sign
tests. All 20 protocol cells contain exactly ten seeds.

## Maximum-support clean results

The table reports mean NLL / C-index / IBS at 48 calibration examples per
eligible context. Lower NLL and IBS and higher C-index are better.

| Condition | Source | Pairwise hierarchy | Three-way hierarchy | Target-only | Oracle |
|---|---:|---:|---:|---:|---:|
| Control | 1.437 / 0.738 / 0.156 | 1.437 / 0.738 / 0.156 | 1.437 / 0.738 / 0.156 | 1.503 / 0.681 / 0.174 | 1.435 / 0.738 / 0.155 |
| Calibrated novel value | 1.610 / 0.647 / 0.198 | 1.460 / 0.741 / 0.152 | 1.460 / 0.741 / 0.152 | 1.527 / 0.689 / 0.167 | 1.458 / 0.743 / 0.150 |
| Uncalibrated novel value | 1.783 / 0.564 / 0.239 | 1.783 / 0.564 / 0.239 | 1.783 / 0.564 / 0.239 | 1.594 / 0.666 / 0.183 | 1.484 / 0.744 / 0.147 |
| Pairwise XOR | 1.650 / 0.632 / 0.207 | 1.442 / 0.741 / 0.154 | 1.442 / 0.741 / 0.154 | 1.506 / 0.688 / 0.170 | 1.439 / 0.743 / 0.152 |
| Three-way Latin | 1.617 / 0.632 / 0.206 | 1.490 / 0.721 / 0.163 | 1.447 / 0.743 / 0.155 | 1.512 / 0.691 / 0.169 | 1.443 / 0.745 / 0.151 |

The hierarchy is inactive in all 10 clean control seeds and exactly matches the
source metrics. It selects subject main effect in 10/10 calibrated-novel seeds,
pairwise subject-by-scene in 10/10 XOR seeds, and the declared three-way
partition in 10/10 Latin seeds. It is inactive in all 10 uncalibrated-novel
seeds, as required by identifiability.

On calibrated novel values, the three-way hierarchy improves over source by
0.150 NLL, 0.095 C-index, and 0.047 IBS. All three comparisons win in 10/10
seeds; their 95% intervals are [0.145, 0.155], [0.092, 0.097], and
[0.046, 0.048]. The method also beats target-only per-context fitting in 10/10
seeds, with gains of 0.067 NLL, 0.052 C-index, and 0.016 IBS.

On the three-way Latin shift, the full hierarchy improves over the pairwise
hierarchy by 0.042 NLL [0.037, 0.047], 0.023 C-index [0.019, 0.027], and 0.009
IBS [0.008, 0.010]. Every comparison wins in 10/10 seeds and the exact sign-test
`p` value is 0.00195. Relative to target-only fitting, its gains are 0.064 NLL,
0.052 C-index, and 0.014 IBS, again with 10/10 wins.

## Structural and subgroup audit

The subgroup result distinguishes useful localization from an aggregate gain.
Under the Latin shift, changed-cell NLL / C-index / IBS are:

- source: 2.543 / 0.259 / 0.426;
- pairwise hierarchy: 1.623 / 0.664 / 0.156;
- three-way hierarchy: 1.522 / 0.735 / 0.143.

On stable cells, the source and three-way hierarchy are exactly
1.432 / 0.736 / 0.158. The pairwise hierarchy is worse at
1.463 / 0.718 / 0.165. Thus the three-way gain is not purchased by changing
stable contexts; the incomplete pairwise representation does make that trade.

The support audit is also exact. Across all 200 runs and all four hierarchical
variants, the maximum absolute difference from source on the test-only `plate`
support slice is 0.0 for every reported metric. In the uncalibrated-novel
negative control, this safety rule preserves poor source performance on the
changed `plate` cells: 2.479 NLL, 0.279 C-index, and 0.393 IBS, compared with the
oracle's 1.583 / 0.723 / 0.135. The result proves fail-closed routing, not
recovery without evidence.

## Sample efficiency and noise

Three-way recovery is support limited. On the Latin condition, the full hierarchy
selects the true partition in 1/10 seeds at 12 examples per context, 8/10 at 24,
and 10/10 at 48. Its predictive advantage over the pairwise hierarchy is:

| Calibration per context | NLL gain | C-index gain | IBS gain | Paired wins |
|---:|---:|---:|---:|---:|
| 12 | 0.003 [0.000, 0.010] | 0.004 [0.000, 0.012] | 0.002 [0.000, 0.005] | 1/10 |
| 24 | 0.041 [0.026, 0.054] | 0.023 [0.014, 0.032] | 0.008 [0.005, 0.012] | 8/10 |
| 48 | 0.042 [0.037, 0.047] | 0.023 [0.019, 0.027] | 0.009 [0.008, 0.010] | 10/10 |

With 20% calibration-label flips at maximum support, the three-way hierarchy
recovers the Latin partition in 9/10 seeds. It still improves over pairwise by
0.054 NLL [0.033, 0.083], 0.034 C-index [0.020, 0.053], and 0.013 IBS
[0.007, 0.021], with nine wins and no losses. Noise is not harmless: on the
control, only one seed remains inactive, six select global calibration, and
three select a local main effect. Mean control NLL worsens from 1.437 to 1.444
and C-index from 0.738 to 0.736, although IBS improves slightly from 0.156 to
0.155. No arbitrary-noise safety claim is supported.

The BIC/MDL screen changes 23 of 200 structural decisions relative to the
unscreened hierarchy, but its effect is not uniformly beneficial. At Latin
support 24 it improves mean NLL/C-index/IBS by 0.011/0.008/0.002; at support 12
it is slightly worse, and under noisy control the unscreened variant has lower
NLL. The screen is therefore retained as an explicit complexity preference, not
claimed as a general robustness solution.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Target-calibrated novel typed values can be repaired without changing stable support. | `bottle` selects subject in 10/10 at support 48; all metrics beat source and target-only in 10/10; stable cells are unchanged. | Supported for the paired synthetic affine reversal. |
| Calibration-unseen typed values fail closed. | Test-only support metrics match source exactly in every one of 200 runs for all hierarchical variants. | Supported as a routing invariant, not as accurate prediction. |
| An unknown hierarchy can recover a true three-way mechanism. | Latin selects the three-way partition in 10/10 at support 48 and beats pairwise on all metrics in 10/10. | Supported for the fixed 3-by-3-by-3 synthetic family at adequate support. |
| Three-way structure is few-shot identifiable. | Only 1/10 exact recovery and intervals touching zero at support 12. | Contradicted for the smallest tested calibration set. |
| BIC/MDL screening provides general noise robustness. | Effects change sign across sample sizes and noisy conditions. | Unsupported. |
| The method adapts a shift confined to a calibration-unseen value. | The hierarchy must return source on `plate`; the oracle gap remains large. | Contradicted by design and by the identifiability negative control. |
| The benchmark demonstrates real-world open-world grounding. | All mechanisms and outcomes are synthetic. | Unsupported. |

## Reviewer-style self-review

- **Contribution — pass for mechanism scope.** The experiment adds a concrete
  support-aware routing rule and demonstrates a predictive distinction between
  pairwise and genuine three-way typed shifts.
- **Writing clarity — pass for this benchmark.** Values, support partitions,
  candidate family, splits, metrics, inference, and negative controls are
  explicit and reproducible.
- **Experimental strength — needs new experiment.** Ten paired seeds and strong
  internal baselines support the mechanism, but no official longitudinal main
  result or recent external method comparison exists.
- **Evaluation completeness — needs new experiment.** Higher-order structure,
  unseen values, calibration size, label noise, and complexity screening are
  covered. Non-affine local errors, irregular observation processes, and
  semi-real histories remain open.
- **Method soundness — pass within stated assumptions.** Selection uses only
  identity-disjoint calibration, and support-external values fail closed.
  Candidate growth, label-noise sensitivity, and inability to repair unsupported
  shifts are explicit limitations.

The next decisive step is not a larger synthetic candidate family. It is an
official or independently audited longitudinal benchmark with observation
histories, changing typed state, natural missingness, language queries, and
evaluation-only current truth.
