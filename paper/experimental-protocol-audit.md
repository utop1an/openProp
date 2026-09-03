# Experimental-protocol audit

Date: 2026-08-27

This document freezes the evidence logic behind manuscript Section 4. It maps
every paper claim to an executable population, comparison, analysis unit, and
scope. It does not promote pending or contradicted claims.

## Claim-to-protocol matrix

| Claim ID | Question and comparison | Population and split | Primary evidence | Status boundary |
|---|---|---|---|---|
| `C1_TYPED_COMPOSITION` | Can familiar typed factors compose on unseen complete tuples? Factorized exponential versus global, exact-context, fixed/no decay, and neural models. | Synthetic 18-context generator; complete tuples disjoint across train/validation/test, while every factor value is represented in train; five paired seeds. | `scripts/evaluate_compositional_multiseed.py` -> `artifacts/compositional_persistence_multiseed_results.json` | Synthetic mechanism validation only. |
| `N1_NEURAL_NECESSITY` | Is neural parameterization necessary? Neural and factorized models use the same split and survival targets. | Same paired rows and five seeds as C1. | Same executable evidence as C1. | Contradicted; the paper claims typed factorization, not neural novelty. |
| `C2_TYPED_COMPONENTS` | Does each typed axis improve survival calibration? Full context versus each leave-one-axis-out model. | Same held-out-tuple generator; ten paired seeds and identical rows across eight fixed conditions. | `scripts/evaluate_typed_context_ablation.py` -> `artifacts/typed_context_component_ablation.json` | NLL is primary; grounding ties cannot establish decision utility. |
| `C3_DECISION_UTILITY` | Can each axis change a rank when its decision boundary is identifiable? Full context versus the matched missing-axis model. | Forty analytic old/new-balanced cases; inspected development seeds are disjoint from ten one-shot confirmation seeds. | `scripts/evaluate_component_balanced_grounding.py` -> `artifacts/component_balanced_grounding_confirmation.json` | Controlled decision utility, not natural prevalence. |
| `C4_INTERVAL_CENSORING` | Does treating detection as exact time create schedule-conditioned persistence and ranking error? Naive detection-time versus interval-aware training, with true hazard as an evaluation-only oracle. | Equal-hazard synthetic schedules; five estimation seeds plus forty target-scene-balanced cases on ten untouched confirmation seeds. | `scripts/evaluate_observation_process.py`, `scripts/evaluate_observation_grounding.py` -> their frozen result artifacts. | Does not cover informative observation, missed detections, or real histories. |
| `C5_EXTERNAL_LANGUAGE` | Does explicit positive span evidence improve a strong supervised retrieval prior? BM25 plus evidence versus train-only BM25. | Official ALFRED train index and all 945 supported validation descriptions; valid-seen/unseen, exact-repeat, and novel-query slices remain explicit. | `scripts/evaluate_alfred_retrieval_baseline.py` -> `artifacts/alfred_retrieval_baseline.json`, `artifacts/alfred_retrieval_comparison.json`, and `artifacts/alfred_retrieval_vs_llm.json` | Language-to-frame parsing only; no entities or temporal grounding. |
| `C6_FALSE_POSITIVE_OBSERVATIONS` | Does fixing specificity at one bias persistence, and can specificity be estimated from training sequences? | Five paired seeds, 1,200 training episodes and 1,000 independent exact-time test rows per false-positive rate; latent transition and test draws are shared across conditions. | `scripts/evaluate_false_positive_observation.py` -> `artifacts/false_positive_observation_results.json` | Synthetic mechanism validation only; benefit is supported at rate 0.10, while simultaneous NLL intervals at 0.02 and 0.05 cross zero. |
| `C7_RECURRENT_OBSERVATIONS` | Can training-only observation logs identify reversible state dynamics, and do they repair current-state prediction when return transitions occur? | Five paired seeds; return rates 0.00/0.15/0.30/0.45; 600 logged episodes and 2,000 independent exact-state rows per condition. Shared random streams pair conditions; zero return is excluded from the three-comparison primary family. | `scripts/evaluate_recurrent_observation.py` -> `artifacts/recurrent_observation_results.json` | Synthetic matched-model validation only; all three nonzero simultaneous NLL intervals exclude zero, while irregular timing is isolated separately and multi-valued/source-specific processes remain untested. |
| `C8_IRREGULAR_OBSERVATIONS` | Does replacing bursty elapsed intervals by their mean bias reversible state estimation and current-state prediction? | Five paired seeds; 600 episodes, 16 opportunities, and exactly 12 h follow-up per contrast; 20,000 independent exact-state rows. Gap contrasts 0.00/0.50/0.75/0.90 share random streams and burst positions; zero contrast is excluded from the primary family. | `scripts/evaluate_irregular_observation.py` -> `artifacts/irregular_observation_results.json` | Synthetic exogenous burst-timing validation only; all three nonzero simultaneous NLL intervals exclude zero. Informative timing, timestamp error, multi-valued states, and natural prevalence remain untested. |
| `N2_REAL_WORLD_GROUNDING` | Does the complete pipeline improve official longitudinal grounding? Full typed model versus property-only and simpler decay baselines. | Predeclared floorplan-disjoint TEACh Layer B/C protocol; official archive, gates, and independent rich-frame labels are not currently available. | Pending status: `scripts/audit_teach_access.py` -> `artifacts/teach_access_audit.json`; future evidence: `scripts/prepare_teach_manifest.py`, `scripts/audit_teach_dataset.py`, `scripts/evaluate_teach_layer_b.py`, and `scripts/evaluate_teach_layer_c.py`. | Pending external evidence and a submission blocker. The access artifact is infrastructure evidence only. |
| `C9_SOURCE_RELIABILITY` | Does tying heterogeneous source emissions flatten current-state evidence? Source-specific versus tied-emission recurrent EM. | Five paired seeds; 400 episodes and 20,000 filtered-current-state rows per severity; two sources; 24 regular opportunities; source-average emissions fixed across severities 0.00/0.33/0.67/1.00. Evaluation state paths are paired and truth is withheld from the filter; zero conflict is excluded from the primary family. | `scripts/evaluate_source_reliability.py` -> `artifacts/source_reliability_results.json` | Synthetic matched-model validation only; all three nonzero simultaneous NLL intervals exclude zero. Correlated/adversarial sources, source churn, multi-valued states, and natural prevalence remain untested. |
| `N3_GENERAL_ADAPTATION_SAFETY` | Do repeated noisy labels certify generally safe target adaptation? | Synthetic development-only nonlinear/noisy stress audit; no fresh confirmation. | `scripts/evaluate_repeated_evidence_adaptation.py` -> `artifacts/repeated_evidence_adaptation_development.json` | Contradicted as a general safety claim; retained only as a boundary result. |

## Shared anti-leakage contract

- Persistence effects are fitted on training entities. Validation data may
  select a fixed half-life, calibrate one hazard multiplier, or fix horizons;
  test outcomes never choose a model or threshold.
- Candidate observations never contain `current_truth`. Analytic target labels
  live only on evaluation cases, and candidate order is reversed in invariant
  tests.
- Development and confirmation seeds are disjoint wherever case construction
  was informed by development behavior. Failed seeds and parse failures remain
  in the declared denominator.
- ALFRED retrieval uses train frames only. Validation labels are used only for
  evaluation; the oracle-at-five row is an analysis upper bound, not a method.
- TEACh partitions whole floorplans. Target-action snapshots and final state are
  excluded from matcher observations; unobserved targets and tied cases are
  counted before identifiable-subset reporting.

## Metrics and inferential units

- Survival quality uses censoring-aware mean NLL as the primary calibration
  metric, with C-index for ordering and integrated Brier score for horizon
  calibration. Cox is not assigned a continuous-event likelihood it does not
  define.
- Grounding reports all-case Top-1 where the protocol permits it, plus MRR and
  predeclared coverage/slice metrics. Controlled confirmation deltas are paired
  by seed.
- Language parsing reports property F1, canonical typed-value recall, and exact
  frame accuracy. Confidence intervals resample task IDs, stratified by task
  type, so multiple descriptions from one trajectory are not treated as
  independent.
- Reported standard deviations are population standard deviations over the
  frozen seed set. Paired intervals use 20,000 deterministic bootstrap
  resamples at the seed or task-cluster level. The three primary component NLL
  intervals and three axis-isolated decision intervals use shared paired-seed
  resamples and family-wise simultaneous maximum-deviation critical values.

## Reverse outline of manuscript Section 4

1. Questions: one claim per experimental block, with TEACh explicitly pending.
2. Controlled typed persistence: held-out-combination design, fixed ablations,
   analytic confirmation, and train/validation/test ownership.
3. Observation process: equal latent hazard, schedule-only intervention, and
   downstream balanced confirmation.
4. External language: train-only retrieval, exact-overlap audit, task-clustered
   uncertainty, and language-only boundary.
5. Metrics/baselines: fair-row comparisons, primary metrics, pairing units, and
   failure denominators.
6. Official longitudinal protocol: executable gates and the exact evidence that
   remains absent.

## Reviewer-facing open risks

- The only external executed result is language parsing; it cannot establish
  temporal or visual grounding.
- Synthetic generators align with important parts of the proposed
  factorization. Misspecification and adaptation audits constrain claims but do
  not replace official longitudinal evaluation.
- Secondary component metrics remain comparison-wise diagnostics; primary NLL and decision families are simultaneous.
- Official TEACh access, gate execution, and three independent rich-frame
  annotations remain necessary before the paper can support its central
  external-validity claim.
