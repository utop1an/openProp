# Local target-adaptation stress benchmark

Date: 2026-08-26

## Question and claim boundary

The exact-affine target-adaptation benchmark asks whether one global sign flip
can be repaired. This stress test asks a harder question: can calibration-only
gates localize a reversal to one declared typed factor without changing stable
contexts, and what fails under label noise or an interaction that no single
factor can represent?

This is synthetic mechanism validation. Generator-defined changed contexts are
used only to partition held-out metrics; they are never available to a fitted
model or to method selection.

## Frozen protocol

The benchmark uses the same ten seeds as the exact-affine experiment and all 18
combinations of three subject types, three tasks, and two scenes. Source
training has 24 records per context. Target calibration has 2, 4, or 8 records
per context (36, 72, or 144 labels total), selected by an outcome-independent
hash of group identity. Every calibration size is evaluated on the same 288
group-disjoint target-test records per condition.

Five paired mechanisms share group identities, features, latent event draws,
and censoring draws:

- in distribution;
- global risk reversal;
- reversal only for `subject_type=cup` (6 of 18 contexts);
- reversal only for `scene=busy` (9 of 18 contexts);
- reversal when `subject_type=cup XOR scene=busy` (9 of 18 contexts).

The noisy setting deterministically flips 20% of calibration event-status
labels while leaving times, features, group identities, and the target test
unchanged. This is an artificial annotation-noise stressor, not a realistic
model of observation error.

Methods include the unchanged source model, the global sign gate, subject- and
scene-grouped gates, a target-only per-context MLE, and a conservative
confirmation ablation. A grouped gate fits an affine log-risk adapter within
each value of one predeclared typed feature and activates only groups with a
negative calibration slope. The confirmation ablation additionally requires
negative slopes in two identity-disjoint calibration halves. Both typed axes
are always reported; no axis is selected on target test.

Run:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_target_adaptation_stress.py
```

The command writes `artifacts/target_adaptation_stress_results.json`, including
all 300 seed-condition-size-noise runs, subgroup metrics, activation logs,
20,000-resample seed-cluster bootstrap intervals, and exact sign tests.

## Correct-axis local repair

Means below are across ten seeds. `Changed C` and `Stable C` expose whether the
method repairs the intended contexts without perturbing the others.

| Shift | Noise | k | Labels | Method | NLL | C-index | IBS | Changed C | Stable C |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| cup only | 0% | 2 | 36 | source | 1.794 | 0.576 | 0.245 | 0.258 | 0.755 |
| cup only | 0% | 2 | 36 | subject gate | 1.455 | 0.751 | 0.149 | 0.742 | 0.755 |
| cup only | 0% | 2 | 36 | target only | 1.696 | 0.694 | 0.202 | 0.692 | 0.689 |
| cup only | 20% | 8 | 144 | subject gate | 1.450 | 0.750 | 0.149 | 0.742 | 0.755 |
| cup only | 20% | 8 | 144 | target only | 1.498 | 0.721 | 0.161 | 0.678 | 0.729 |
| busy only | 0% | 2 | 36 | source | 1.830 | 0.521 | 0.266 | 0.300 | 0.690 |
| busy only | 0% | 2 | 36 | scene gate | 1.344 | 0.675 | 0.186 | 0.700 | 0.610 |
| busy only | 0% | 2 | 36 | target only | 1.627 | 0.625 | 0.235 | 0.632 | 0.593 |
| busy only | 20% | 8 | 144 | scene gate | 1.323 | 0.704 | 0.175 | 0.700 | 0.690 |
| busy only | 20% | 8 | 144 | target only | 1.389 | 0.651 | 0.189 | 0.645 | 0.635 |

At `k=8`, the subject gate improves NLL by 0.355, C-index by 0.176, and IBS by
0.098 without noise. The seed-cluster 95% intervals are [0.324, 0.389], [0.165,
0.187], and [0.094, 0.103], respectively. The scene gate improves NLL by 0.514,
C-index by 0.184, and IBS by 0.091, with intervals [0.401, 0.627], [0.166,
0.201], and [0.081, 0.101]. All six comparisons improve in 10/10 seeds
(two-sided exact sign-test p=0.001953 each). With 20% calibration-label flips,
the corresponding NLL/C-index gains remain 0.344/0.174 and 0.507/0.183, again
with 10/10 wins.

The subject gate leaves all stable-subgroup predictions exactly equal to the
source model at every reported size. The scene gate does so at `k=8`, but at
`k=2` it falsely activates a stable scene in 2/10 seeds, reducing mean stable
C-index by 0.080. The apparent `k=2` overall gain therefore combines true local
repair with an unsafe false activation and should not support a safety claim.

## Activation audit and confirmation ablation

Entries are seeds with any false activation under the unchanged mechanism, or
seeds detecting the true affected group under the corresponding local shift.

| k | Noise | Subject false: simple / confirmed | Scene false: simple / confirmed | Cup detected: simple / confirmed | Busy detected: simple / confirmed |
|---:|---:|---:|---:|---:|---:|
| 2 | 0% | 0/10 / 0/10 | 2/10 / 1/10 | 10/10 / 9/10 | 10/10 / 8/10 |
| 2 | 20% | 0/10 / 3/10 | 2/10 / 1/10 | 10/10 / 10/10 | 10/10 / 7/10 |
| 4 | 0% | 0/10 / 1/10 | 1/10 / 1/10 | 10/10 / 10/10 | 10/10 / 10/10 |
| 4 | 20% | 1/10 / 1/10 | 1/10 / 0/10 | 10/10 / 10/10 | 10/10 / 10/10 |
| 8 | 0% | 0/10 / 0/10 | 0/10 / 0/10 | 10/10 / 10/10 | 10/10 / 10/10 |
| 8 | 20% | 0/10 / 0/10 | 0/10 / 0/10 | 10/10 / 9/10 | 10/10 / 10/10 |

Identity-split confirmation is not uniformly safer: it removes one scene false
activation at `k=2` but introduces subject false activations under noisy labels
and misses real shifts. It is retained as a preregistered-style ablation, not
promoted as the primary method. The simple gate is empirically reliable here
only once each typed group has enough calibration support (`k=8`).

## Interaction failure is the important boundary

No single subject or scene partition represents the XOR shift. At `k=8`
without label noise, the subject gate raises overall C-index from 0.498 to
0.632, but changed-context C-index rises from 0.265 to only 0.567 while stable
C-index falls from 0.757 to 0.680. The stable-subgroup C-index harm is -0.077,
with seed-cluster 95% interval [-0.107, -0.047], and occurs in 9/10 seeds. With
20% noise, overall C-index is 0.580 and stable C-index remains harmed at 0.696.
The target-only per-context model reaches 0.738/0.726 overall C-index without/
with noise, confirming that the limitation is the single-axis representation,
not absence of target signal.

This result rejects a universal-safe-adaptation claim. The next method must
represent sparse typed interactions while controlling family-wise false
activation, preferably with hierarchical shrinkage or held-out calibration
evidence rather than choosing interactions on target test.

That next step is now implemented in the
[multiplicity-controlled typed interaction benchmark](target-interaction-adaptation-benchmark.md).
It repairs the XOR stable-subgroup failure at higher calibration support, while
showing a clear confirmation-power cost and one noisy in-distribution false
activation.

## What is and is not supported

Supported under this generator:

- a predeclared correct typed axis can localize a simple partial reversal;
- with 144 total calibration labels, local gates repair changed contexts while
  leaving stable contexts exactly unchanged in all tested seeds;
- the gains survive 20% artificial calibration event-status corruption;
- subgroup audits expose false activation that aggregate metrics hide.

Not supported:

- safe adaptation at every sample size;
- automatic discovery of the correct type axis or interaction;
- unseen factor values, nonlinear time dynamics, informative censoring,
  recurrent transitions, or realistic observation/annotation noise;
- real or semi-real few-shot adaptation or grounding performance.

