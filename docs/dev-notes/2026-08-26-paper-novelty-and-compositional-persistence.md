# Paper novelty and compositional persistence upgrade

Date: 2026-08-26

## Context

This work session assessed OpenProp as an academic paper, considered whether it
fits ICLR, identified the main novelty gaps, and implemented the first
novelty-oriented experiment upgrade.

The project is topically compatible with ICLR areas such as structured
prediction, probabilistic methods, uncertainty quantification, hybrid AI,
robotics and datasets. The pre-upgrade evidence was not competitive for an ICLR
main-track paper because it showed mechanism plumbing rather than a new learning
result: the original temporal benchmark gave both fixed and learned decay
perfect grounding accuracy, and the learned model used fixed-policy fallback for
untrained properties and event retention.

## Paper Positioning

The preferred paper story is:

> OpenProp studies current-entity grounding from heterogeneous, incomplete and
> stale evidence by separating semantic query interpretation from typed,
> uncertainty-aware and temporally calibrated entity scoring.

The strongest future claim should concern compositional, decision-useful state
persistence rather than a general extensible software framework.

Three contribution axes were selected:

1. typed compositional persistence across unseen context combinations;
2. separation of latent state dynamics from observation availability;
3. joint survival calibration and downstream grounding quality.

The first axis has an implemented mechanism-validation experiment. The second
and third remain follow-up method work.

## Implemented Upgrade

### Compositional OOD data protocol

`src/openprop/compositional_persistence.py` introduces a factorised synthetic
location-persistence process over subject type, relation predicate, context
object and scene. The default protocol contains 18 contexts:

- 12 training contexts with 960 episodes;
- 3 validation contexts with 240 episodes;
- 3 test contexts with 240 episodes.

Every individual validation/test feature value appears in training, but each
complete validation/test tuple is held out. Entity group identifiers are
disjoint. Grounding cases keep `current_truth` separate from matcher inputs and
use strict score differences so candidate order cannot create the reported gain.

### Statistical baselines

`src/openprop/statistical_persistence.py` adds:

- global exponential maximum-likelihood persistence;
- smoothed per-context exponential maximum likelihood;
- explicit global backoff for unseen context tuples.
- L2-regularized factorized log-linear exponential persistence with interval
  censoring and validation-only calibration;

These baselines share the same auditable event-retention boundary as existing
models so comparisons isolate time-retention estimation.

### Evaluation and orchestration

`src/openprop/advanced_survival_evaluation.py` adds Harrell's C-index and a
trapezoidal mean Brier score over requested horizons while preserving the rule
that censoring before a horizon yields unknown truth.

Reproducible entry points:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_compositional_persistence.py
python scripts/evaluate_compositional_multiseed.py
python -m unittest discover -s tests -v
```

Results are written to:

- `artifacts/compositional_persistence_results.json`;
- `artifacts/compositional_persistence_multiseed_results.json`.

## Verified Results

Five seeds were run: 31, 41, 53, 67 and 79.

| Model | Test NLL lower is better | C-index higher is better | Integrated Brier lower is better | Grounding Top-1 higher is better |
|---|---:|---:|---:|---:|
| Global exponential | 1.795 +/- 0.033 | 0.500 +/- 0.000 | 0.243 +/- 0.003 | 0.000 +/- 0.000 |
| Factorized exponential | 1.239 +/- 0.069 | 0.749 +/- 0.014 | 0.079 +/- 0.010 | 1.000 +/- 0.000 |
| Neural compositional | 1.266 +/- 0.061 | 0.723 +/- 0.026 | 0.084 +/- 0.009 | 0.972 +/- 0.056 |

The factorized model outperformed the neural model on all four aggregate
metrics and achieved Top-1 1.000 for every seed. Neural grounding Top-1 ranged
from 0.861 to 1.000. Thus this benchmark supports typed factorization, not
neural necessity. No decay, fixed four-hour decay, global exponential and
per-context exponential scored zero in the single-seed comparison.

The full suite now contains 75 passing tests, including factorized OOD,
interval-censoring, Weibull, piecewise, Cox, split-horizon, latent-mechanism,
observation-process, synthetic-oracle, and EM coverage.

## Supported And Unsupported Claims

Supported as synthetic mechanism validation:

- familiar typed factors can be reused on held-out combinations;
- exact-context lookup and global decay cannot solve this split;
- improved survival estimates can change the final grounding decision.

Not supported:

- real-world state-transition probabilities or grounding accuracy;
- generalisation to wholly unseen property/type values;
- robustness to noisy language parsing;
- learned event effects;
- freedom from observation-policy bias;
- robustness to arbitrary latent-mechanism change;
- a main-track ICLR-level empirical claim.

## Next Tasks

1. Run the prepared TEACh feasibility audit when official data access is restored
   or an author-maintained release is identified.
2. Evaluate noisy language-to-`QueryFrame` parsing with failures retained in
   the denominator.
3. Extend the completed leakage-safe calibration experiment beyond exact global
   affine reversal to partial/local changes, noisy labels, and unseen contexts.
4. Extend the completed training-only observation estimator to false positives,
   irregular coverage, recurrent transitions, and conflicting sources.
5. Add a grounding-aware objective only if semi-real evidence shows that the
   simpler factorized model is insufficient.

## Operational Notes

The workspace exposed no usable Git repository metadata during this session, so
no Git diff or commit was produced. Existing v1 APIs were left intact; the new
experiment path is additive.
