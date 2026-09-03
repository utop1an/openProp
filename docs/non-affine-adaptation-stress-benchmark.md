# Local non-affine target-adaptation stress benchmark

Date: 2026-08-26

## Reviewer question

Earlier OpenProp adaptation experiments use global log-risk calibration, typed
reversals, and discrete value permutations. Those mechanisms do not establish
that a sparse typed affine gate can repair a source model whose error is
nonlinear *within* a typed region. This benchmark asks two separate questions:

1. can the current gate detect and repair local non-affine source error; and
2. when it cannot, does it at least preserve unaffected contexts and a correct
   source?

The answer is mixed and materially narrows the method claim. The gate repairs
part of a smooth interaction-local calibration error, but it does not reliably
recover local saturation or a folded risk ordering. Under calibration-label
noise it also false-activates too often to support a broad safety claim.

## Predeclared paired protocol

The executable protocol is
`scripts/evaluate_non_affine_adaptation_stress.py`. It uses ten fixed seeds,
18 typed contexts, 32 target examples per context, nested calibration sizes of
4, 8, and 16 examples per context, and calibration-only event-label flip rates
of 0 and 0.2. Each condition shares source training, target calibration rows,
fixed group-disjoint target test rows, latent event draws, censoring, horizons,
and optimizer settings. The artifact contains 240 unique factorial runs and 288
test examples per run.

The target mechanism remains the undistorted fitted source. Only the deployed
source prediction is changed:

- `correct_source_control`: no distortion;
- `local_subject_saturation`: monotone `tanh` compression in log risk for
  `subject=cup`;
- `local_scene_fold`: an absolute-log fold around 0.12 events/hour inside
  `scene=busy`, producing a nonmonotone ordering below the pivot; and
- `local_subject_scene_bump`: a smooth Gaussian log-risk bump only for
  `subject=cup, scene=busy`.

The affected subset is determined by deployed-versus-source predictions before
looking at target outcomes. The stable subset is its complement. This prevents
evaluation loss from defining the subgroup. All three warps leave stable
predictions unchanged to numerical tolerance; this holds in 240/240 runs.

Methods are the correct-source reference, deployed source, unrestricted global
affine calibration, the controlled typed gate, its BIC-screened variant, and
target-only per-context MLE. Both controlled gates use identity-disjoint
discovery and confirmation, familywise alpha 0.05, and the same global,
subject, scene, and subject-by-scene candidates.

Run:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_non_affine_adaptation_stress.py
```

The frozen output is
`artifacts/non_affine_adaptation_stress_results.json`. Paired uncertainty uses
20,000 seed-cluster bootstrap resamples and exact two-sided sign tests. Synthetic
results validate mechanisms and failure modes, not real-world effectiveness.

## Maximum-support clean results

All-case means at 16 calibration examples per typed context:

| Condition | Method | NLL | C-index | IBS |
|---|---|---:|---:|---:|
| Correct control | Deployed source | 1.418 | 0.745 | 0.151 |
| Correct control | Controlled typed | 1.420 | 0.745 | 0.153 |
| Subject saturation | Deployed source | 1.427 | 0.741 | 0.152 |
| Subject saturation | Controlled typed | 1.427 | 0.741 | 0.152 |
| Scene fold | Deployed source | 1.475 | 0.721 | 0.157 |
| Scene fold | Global affine | 1.460 | 0.721 | 0.163 |
| Scene fold | Controlled typed | 1.464 | 0.724 | 0.157 |
| Scene fold | Controlled typed + BIC | 1.460 | 0.725 | 0.160 |
| Scene fold | Target per-context | 1.440 | 0.738 | 0.158 |
| Subject-scene bump | Deployed source | 1.495 | 0.734 | 0.156 |
| Subject-scene bump | Global affine | 1.449 | 0.734 | 0.159 |
| Subject-scene bump | Controlled typed | 1.446 | 0.736 | 0.158 |
| Subject-scene bump | Controlled typed + BIC | 1.444 | 0.736 | 0.157 |
| Subject-scene bump | Target per-context | 1.440 | 0.738 | 0.158 |

The correct-source reference is 1.418/0.745/0.151 for every distorted
condition. The distortions therefore leave recoverable error rather than merely
creating harmless parameter changes.

## Local repair versus stable-region harm

For the scene fold, the correct source improves affected-subset C-index by
0.192 [0.101, 0.279] over the deployed source. The controlled gate recovers only
0.031 [0.000, 0.093], activating in 2/10 clean maximum-support seeds. Its
BIC-screened variant recovers 0.025 [-0.008, 0.069] and activates in 5/10 with
inconsistent global, subject, and interaction structures. Target per-context
recovers 0.174 [0.081, 0.266] but worsens stable-subset NLL by 0.022
[0.003, 0.044]. No method simultaneously recovers the fold and gives a strong
stable-safety guarantee.

For the smooth subject-scene bump, global affine, controlled typed, BIC typed,
and target-only improve all-case NLL over deployed by 0.046, 0.048, 0.050, and
0.054 respectively; all bootstrap intervals exclude zero. However, controlled
typed affected-subset C-index improvement is exactly 0.000 in all ten seeds.
The all-case NLL gain is calibration repair, not evidence of ordering recovery.
Selected typed structures are also unstable: clean maximum-support runs choose
global in five seeds, subject in three, and remain inactive in two.

Subject saturation is a lower-severity calibration error: the correct-source
NLL advantage is 0.009 [0.006, 0.013] overall and 0.028 [0.017, 0.038] on the
affected subset. Both typed gates remain inactive in 10/10 clean
maximum-support seeds. This is conservative but has low power for mild local
nonlinearity.

## Noise and false activation

On the clean correct-source control at maximum support, both typed gates are
inactive in 9/10 seeds and false-activate globally once. With 20% calibration
label flips, each is inactive in only 5/10: the unscreened gate selects global
four times and subject once; the BIC gate selects global four times and scene
once. Corresponding control NLL deltas are negative, so these are harmful false
activations, not benign structural refinements.

BIC screening changes some selected partitions but does not reduce the total
noisy-control activation count. It is therefore not a sufficient robustness
mechanism for this label-noise process.

## Claim-evidence boundary

| Claim | Evidence | Status |
|---|---|---|
| Typed affine adaptation handles arbitrary local source misspecification. | It misses mild saturation, recovers little of the folded affected ordering, and does not improve bump affected-subset C-index. | Contradicted. |
| Controlled adaptation can improve smooth local calibration error. | On the bump, typed+BIC improves all-case NLL by 0.050 [0.026, 0.075] with 7/10 wins. | Supported only for synthetic NLL calibration at high support. |
| BIC screening prevents noisy false activation. | Both controlled variants activate in 5/10 noisy correct-source controls. | Contradicted for deterministic 20% event-label flips. |
| Target-only fitting is a uniformly safe fallback. | It repairs fold ranking but significantly harms stable NLL and correct-source controls. | Contradicted. |
| Stable deployed contexts are genuinely unchanged by the generator. | Maximum deployed-versus-source hazard difference is at most `1e-12` in all 240 runs. | Supported as a construction invariant. |

## Method implication

This benchmark does not justify adding an unrestricted flexible calibrator. The
failure is structured: local nonlinearity needs a richer within-group family,
while noisy controls require stronger robustness than ordinary BIC plus a
likelihood-ratio confirmation split. The next method iteration should compare a
small predeclared nonlinear basis against affine repair using robust
calibration-only confirmation, preserve unsupported values by construction, and
retain the stable-subset and correct-source audits used here.

## Candidate follow-up decision

The proposed nonlinear follow-up was executed and rejected. Quadratic and hinge
bases add no reliable benefit over matched sparse affine routing. Sparse
coverage closure reduces fresh-seed noisy-control activation from 4/10 to 0/10
but does not improve fold ranking over the previous BIC gate and slightly
worsens bump NLL. The frozen confirmation records `accepted: false`. See the
[sparse adaptation candidate audit](sparse-adaptation-candidate-audit.md) for
SHA-bound development artifacts, fresh seeds, acceptance criteria, and the
complete claim boundary.
