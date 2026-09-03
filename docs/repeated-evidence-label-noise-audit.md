# Repeated calibration evidence under event-label noise

Date: 2026-08-26

## Decision

Repeated evidence is retained as an identifiable noise protocol and negative
adaptation ablation, but it is not promoted into the main OpenProp method.

Five independent annotations of each calibration identity reduce synthetic
event-status error from 20.0% to 5.5% by majority vote and to 0.84% after a
posterior-confidence abstention rule. The latter removes all noisy correct-source
activations and preserves useful likelihood gains. It does not, however, give a
reliable ranking-safety guarantee: one development seed passes a calibration
concordance guard and then reverses the held-out affected-subset ordering. The
candidate therefore fails before fresh-seed confirmation.

## Research question

The preceding sparse-candidate audit showed that a likelihood-only gate cannot
distinguish a broad true hazard shift from calibration event-label flips. This
experiment asks whether the ambiguity becomes identifiable when the calibration
protocol supplies repeated, conditionally independent status evidence.

The additional assumption is explicit rather than hidden: every calibration
identity has an odd number of annotations with a common symmetric flip
probability below one half. The method does not claim robustness to correlated,
instance-dependent, adversarial, or annotator-specific errors.

## Method and evidence boundary

`RepeatedEventEvidence` stores typed context, duration, identity, annotator IDs,
and status labels, but never stores the latent event status. Repeated-label
provenance remains outside the ordinary property dictionary.

For two conditionally independent annotators with symmetric error rate
`epsilon`, their disagreement probability is
`d = 2 epsilon (1 - epsilon)`. The protocol estimates `epsilon` from pairwise
calibration disagreement using the identifiable lower root
`(1 - sqrt(1 - 2d)) / 2`. Given an equal latent-class prior, repeated votes yield
a posterior event probability. A hard consensus uses its more likely status;
the confident variant abstains when posterior confidence is below 0.9. Abstained
records are missing evidence, not negative events.

The downstream adapter is unchanged: every decoded calibration set is passed to
the same multiplicity-controlled BIC typed gate. A development-only outer
ablation additionally requires candidate calibration C-index to be no worse
than the frozen source. Target test rows and hidden clean labels are never
available to decoding, fitting, or selection. Hidden clean labels appear only
in the named oracle and the evaluation-only decoder-error audit.

## Development protocol

- ten pre-existing development seeds: 31, 41, 53, 67, 79, 97, 109, 127, 149,
  and 173;
- 18 typed contexts and 32 generated rows per context;
- 15 calibration identities per context and an identity-disjoint fixed test;
- clean and 20% calibration-only symmetric flip conditions;
- correct-source, local saturation, local fold, and local interaction-bump
  deployment conditions;
- source training, deployment distortions, test rows, horizons, and typed-gate
  settings shared by all methods;
- paired seed bootstrap with 20,000 resamples.

Two controls separate evidence quality from annotation cost. `3 x 5` uses three
annotations on five identities and `5 x 3` uses five annotations on three
identities; both spend the same 15 labels per context as the single-label
baseline. `5 x 15` instead holds the 15 identities fixed and spends 75 labels
per context. The clean-label oracle is evaluation-only.

## Results

The table reports the noisy 20% condition. NLL and affected C-index deltas are
relative to the unadapted deployed source; positive values are better. Error is
measured against hidden clean status for audit only.

| Calibration evidence | Labels/context | Decoded error | Control activation | Fold NLL | Fold affected C | Bump NLL | Bump affected C |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single label on 15 cases | 15 | 20.00% | 2/10 | +0.015 | +0.030 | +0.047 | 0.000 |
| 3 labels on 5 cases | 15 | 9.56% | 0/10 | 0.000 | 0.000 | +0.019 | 0.000 |
| 5 labels on 3 cases | 15 | 6.67% | 0/10 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 labels on 15 cases, majority | 75 | 5.52% | 0/10 | +0.003 | 0.000 | +0.039 | 0.000 |
| 5 labels on 15 cases, confidence >= 0.9 | 75 | **0.84%** | **0/10** | +0.004 | +0.019 | +0.037 | **-0.036** |
| Clean 15-case oracle | 15 | 0.00% | 0/10 | +0.007 | +0.020 | +0.029 | 0.000 |

The confidence decoder retains 74.0% of repeated records (188--208 of 270 per
seed) and estimates mean flip probability 0.201, close to the synthetic 0.20
mechanism. Its bump all-case NLL gain is +0.037 [0.015, 0.060], and its fold
affected C-index gain is +0.019 [0.000, 0.058]. These are mechanism-level
benefits under the declared independent-noise assumption.

The equal-budget controls expose the practical cost. Repeated annotation reduces
label error, but replacing independent calibration identities with repeats
removes nearly all fold power. The same-case protocol needs five times the
annotation budget to retain useful adaptation.

## Why the candidate is rejected

Confidence abstention does not solve multi-objective selection. On the bump,
the likelihood-selected typed repair reduces NLL but lowers affected C-index by
0.036 [-0.109, 0.000]. A calibration-only concordance non-inferiority guard does
not change the aggregate result. In seed 173, the guard observes a +0.0072
calibration C-index delta and accepts the repair, but the held-out affected
C-index delta is -0.3629. The calibration statistic therefore does not certify
ranking safety across identities.

Running a fresh-seed confirmation after observing this development failure would
not rescue a prespecified candidate. No confirmation was run, and no main-method
claim is made. The reusable result is the identifiability and budget boundary:
independent repetition can estimate and suppress symmetric label noise, but
finite repeated evidence does not make downstream adaptation uniformly safe.

## Claim--evidence map

| Claim | Evidence | Status |
|---|---|---|
| Pairwise disagreement identifies homogeneous independent symmetric noise below 0.5. | The estimator follows the closed-form lower root and recovers mean 0.201 under a 0.20 generator. | Supported under the declared model. |
| Posterior abstention reduces status error and noisy control activation. | Error falls from 20.0% to 0.84%; correct-source activations fall from 2/10 to 0/10. | Supported on ten synthetic development seeds. |
| Repetition is cost-free robustness. | Equal-budget repetitions lose nearly all fold power. | Contradicted. |
| Calibration concordance guarantees held-out ranking safety. | Seed 173 passes calibration but loses 0.3629 held-out affected C-index. | Contradicted. |
| The protocol handles arbitrary annotation noise. | Only homogeneous symmetric independent flips are modeled. | Not supported. |

## Reviewer-style self-review

- **Contribution:** the identifiable repeated-evidence boundary is precise, but
  posterior voting itself is standard and is not sufficient method novelty.
- **Writing clarity:** assumptions, hidden-truth boundary, costs, and rejection
  decision are explicit and reproducible.
- **Experimental strength:** paired controls and failure analysis are strong for
  mechanism diagnosis, but all evidence here is synthetic development evidence.
- **Evaluation completeness:** correlated and annotator-specific noise, variable
  repeat counts, interval-censored labels, and real repeated evidence remain
  untested.
- **Method soundness:** the decoder is sound under its assumptions; the attempted
  downstream ranking guard does not generalize and must not be claimed as safe.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_repeated_evidence_adaptation.py
```

The development artifact is
`artifacts/repeated_evidence_adaptation_development.json`, SHA-256
`ce7247134e186684bd86a71c63359867c5e31c074f7623dfce89a1a59d89a403`.
It contains 80 paired condition/noise/seed runs, 20 calibration audits, full
method metrics, activation diagnostics, posterior estimates, annotation budgets,
and the evaluation-only decoder error rates.
