# Sparse nonlinear and coverage-closure candidate audit

Date: 2026-08-26

## Decision

Neither candidate is promoted into the main OpenProp method.

- The affine/quadratic/hinge basis expansion is rejected because the nonlinear
  terms add no reliable benefit over an otherwise sparse affine control and
  increase noisy-control activation.
- The sparse-coverage affine closure is rejected by its fresh-seed acceptance
  protocol. It materially reduces false activation, but does not improve fold
  recovery over the previous BIC gate and slightly worsens bump NLL.

The code and artifacts remain as reproducible negative ablations. Paper claims
must not describe either candidate as an accepted improvement.

## Development-stage nonlinear candidate

`SparseNonlinearTypedGate` excludes the global partition, considers subject,
scene, and subject-by-scene routing, and caps the fraction of calibrated typed
contexts that may be adapted. Within each group, discovery-only BIC chooses
among affine, quadratic, and one-knot hinge log-risk bases. The chosen family
must then pass an identity-disjoint confirmation likelihood-ratio e-value with
Bonferroni correction. Unsupported typed values route to the frozen source.

The ablation is SHA-bound to the original 240-run non-affine artifact:

- base: `1a892b543a01db515b01cbe71f0418a627e6b5736fd4f264a4d3c9999e12303b`;
- sparse nonlinear: `ddb56de6d2fce1bbfe1d0ff9f2ed1a68648f518da8ec70c70408159c3bf60d42`;
- sparse affine necessity control:
  `73d9b93d39d26a3c8ed89e3d51710a990a44e48a84bf8b26af7858fa1205efad`.

At maximum clean support, sparse nonlinear versus deployed source improves:

- fold all-case NLL by 0.020 [0.005, 0.037];
- fold affected C-index by 0.075 [0.006, 0.160]; and
- bump all-case NLL by 0.061 [0.035, 0.086].

However, basis-selection diagnostics show that almost all accepted groups are
affine. On the clean bump, quadratic is selected in only one seed; no hinge is
selected. Relative to the matched sparse-affine control, nonlinear clean bump
NLL gain is only 0.003 [-0.006, 0.011] and affected C-index gain is exactly
zero. For the fold, nonlinear-versus-affine affected C-index is 0.017
[0.000, 0.041] with only 2 wins and 8 ties. No broad nonlinear-family benefit
is established.

The safety direction is worse: on maximum-support noisy correct-source controls,
sparse nonlinear activates in 2/10 seeds while sparse affine activates in 1/10.
This violates the intended requirement that added flexibility must not increase
false activation. The nonlinear candidate is therefore rejected.

## Fresh-seed sparse-coverage confirmation

The useful development signal was not nonlinearity but sparse closure: exclude
global calibration and reject a typed candidate covering more than half of the
calibrated contexts. This was implemented independently as
`SparseCoverageAffineGate` and evaluated on ten fresh seeds:

`181, 191, 211, 223, 239, 251, 263, 277, 293, 307`.

These seeds are absent from the earlier adaptation artifacts. The runner freezes
maximum support (16 examples per each of 18 contexts), clean and 20% label-flip
conditions, optimizer settings, paired rows, and the following acceptance
criteria before inspecting confirmation outcomes:

1. zero clean correct-source activations;
2. at most two noisy correct-source activations;
3. strictly fewer noisy activations than the previous controlled BIC gate;
4. at least +0.02 mean fold affected C-index over the previous gate;
5. a nonnegative bootstrap lower bound for that fold improvement; and
6. bump all-case NLL no more than 0.01 worse at the lower confidence bound.

The frozen artifact is
`artifacts/sparse_coverage_affine_confirmation.json`, SHA-256
`9f2ce3f157535e48e350a4857cfa96257b6bf6ce304f75d6dc5b0e7466777d89`.
Its own decision field records `accepted: false`.

## Confirmation results

| Criterion | Result | Pass |
|---|---:|---|
| Clean correct-source activation | 0/10 | yes |
| Noisy correct-source activation | sparse 0/10; previous 4/10 | yes |
| Fold affected C-index versus previous | 0.000 [0.000, 0.000] | **no** |
| Bump all-case NLL versus previous | -0.003 [-0.005, -0.001] | allowed by tolerance, but worse |

Against the unadapted deployed source, sparse closure does recover some signal:
fold affected C-index improves by 0.046 [0.011, 0.086], and bump all-case NLL
improves by 0.033 [0.011, 0.058]. But the previous controlled BIC gate obtains
the same fold ranking recovery on the fresh seeds. Sparse closure also worsens
fold stable-subset NLL by 0.009 [0.002, 0.018]. It is a stronger false-activation
guard, not a better repair model.

The confirmation runner's legacy comparator serializes its global partition as
an empty string in per-run diagnostics; activation counts treat this as active.
This presentation quirk does not affect predictions, paired metrics, criteria,
or the recorded decision.

## Reviewer-facing interpretation

The experiment prevents two tempting but unsupported narratives:

- extra nonlinear bases did not solve local non-affine adaptation; and
- a safer sparse closure did not produce a superior accuracy-safety frontier
  under the frozen acceptance rule.

The remaining technical problem is identifiable rather than vague. Calibration
label flips resemble a genuine broad hazard shift, so a likelihood-only gate
cannot guarantee both adaptation power and noise rejection without additional
assumptions or evidence. The next candidate should explicitly model label noise
or obtain repeated/clean calibration evidence; it should not add another flexible
risk basis to the same confirmation mechanism.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_sparse_nonlinear_ablation.py
python scripts/evaluate_sparse_affine_control.py
python scripts/confirm_sparse_coverage_affine.py
```

All results are synthetic method and failure-mode evidence. They do not support
real-world adaptation, arbitrary-noise robustness, or general nonlinear repair.
