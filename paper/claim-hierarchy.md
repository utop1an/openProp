# OpenProp claim hierarchy

## One-sentence thesis

Language-conditioned current-entity grounding from long-lived memory is a
typed decision under incomplete and stale evidence, so semantic query parsing,
value-family comparison, observation-aware persistence, and deterministic
decision scoring must remain explicit and separately testable.

This is the stable paper thesis. It does not depend on a neural architecture
winning, and it does not imply real-world effectiveness before the official
longitudinal benchmark is complete.

## Claim ladder

| Level | Claim ID | Paper-eligible wording | Evidence boundary | Forbidden extension |
|---|---|---|---|---|
| Primary | `C1_TYPED_COMPOSITION` | Typed factorization reuses familiar context values on held-out combinations and changes controlled grounding decisions. | Five-seed synthetic factorized generator. | General open-world or real-world transfer. |
| Primary | `C2_TYPED_COMPONENTS` | Subject, relation, and scene independently improve held-out survival calibration. | Ten paired synthetic seeds; NLL is primary. | Every component is necessary for every natural grounding case. |
| Primary | `C3_DECISION_UTILITY` | Each typed factor can change entity rank when competing evidence is analytically balanced. | Ten untouched confirmation seeds and 40 controlled cases. | Naturalistic prevalence or end-to-end effectiveness. |
| Secondary | `C4_INTERVAL_CENSORING` | Interval-aware learning suppresses inspection-frequency bias in hazard estimation and controlled entity ranking. | Five-seed estimation study plus untouched ten-seed, 40-case analytic grounding confirmation. | Natural effect prevalence, arbitrary missingness, or real observation-process robustness. |
| Secondary | `C5_EXTERNAL_LANGUAGE` | Positive lexical evidence improves a train-only BM25 typed-frame baseline on independently authored ALFRED language. | Language-to-frame parsing only. | Visual, temporal, or end-to-end grounding. |
| Secondary | `C6_FALSE_POSITIVE_OBSERVATIONS` | Training-only specificity estimation removes hazard bias when false-positive violations are large enough to identify. | Five paired synthetic seeds; only the 0.10 NLL interval excludes zero. | Universal benefit, arbitrary sensor noise, or multi-source robustness. |
| Secondary | `C7_RECURRENT_OBSERVATIONS` | A training-only reversible binary CTMC repairs current-state prediction when return transitions violate the irreversible model. | Five paired synthetic seeds; 5/5 NLL wins and simultaneous intervals above zero at return rates 0.15, 0.30, and 0.45/h; near-tie at zero return. | Real-world recurrence, multi-valued dynamics, arbitrary observation noise, or universal dominance. |
| Secondary | `C8_IRREGULAR_OBSERVATIONS` | Exact elapsed intervals prevent bursty observation timing from becoming a false transition-rate signal. | Five paired synthetic seeds; 5/5 NLL wins and simultaneous intervals above zero at gap contrasts 0.50, 0.75, and 0.90; near-tie on the regular grid. | Natural timing prevalence, informative opportunity timing, timestamp error, or source-specific robustness. |
| Secondary | `C9_SOURCE_RELIABILITY` | Source-specific emissions prevent conflicting reliability profiles from flattening filtered current-state evidence. | Five paired synthetic seeds; fixed source-average emissions; 5/5 NLL wins and simultaneous intervals above zero at conflict severities 0.33, 0.67, and 1.00; near-tie at zero conflict. | Natural source prevalence, correlated/adversarial sources, source churn, or real-world robustness. |
| Boundary | `N1_NEURAL_NECESSITY` | The factorized statistical model is the strongest current persistence model. | Neural necessity is contradicted by current results. | Neural architectural novelty. |
| Blocker | `N2_REAL_WORLD_GROUNDING` | Semi-real longitudinal effectiveness is not yet established. | Official TEACh result pending. | Any real-world or semi-real performance claim. |
| Boundary | `N3_GENERAL_ADAPTATION_SAFETY` | Adaptation failures identify conditions requiring independent evidence. | Frozen nonlinear and noisy audits. | General repair or arbitrary-noise safety. |

The executable source of truth is [claims.json](claims.json). Any prose that
changes a claim's scope must first change the manifest and pass the verifier.

## Narrative order

1. **Task:** resolve a current entity from language when memory contains typed,
   partial, and differently aged evidence.
2. **Failure of flattening:** a single similarity space cannot express numeric
   tolerance, relational argument identity, semantic similarity, missingness,
   provenance, and temporal validity with one shared meaning.
3. **OpenProp boundary:** semantic parsing proposes typed constraints;
   deterministic comparators and explicit confidence, freshness, match, and
   coverage terms make the decision.
4. **Persistence problem:** state confidence depends on typed context and on
   what was observed when; unobserved change times are interval- or
   right-censored rather than negative labels.
5. **Evidence:** controlled experiments isolate compositional generalization,
   factor necessity, decision utility, observation-frequency bias, and a false-
   positive identifiability boundary. ALFRED
   tests only the language-to-frame boundary.
6. **External gate:** TEACh must test the integrated longitudinal claim with
   observation histories and evaluation-only current truth.

## Locked terminology

- **Current truth:** evaluation-only final state; never matcher input.
- **Unknown:** missing evidence, not negative evidence or a zero match.
- **Typed context:** named factors such as subject, relation, and scene whose
  values retain their identities.
- **Persistence:** probability that a previously observed state remains valid
  at a query time.
- **Observation history:** timestamped, provenance-bearing evidence stored
  outside the ordinary property dictionary.
- **Mechanism validation:** controlled synthetic evidence that establishes a
  causal or statistical behavior, not real-world effectiveness.
- **External language evidence:** evaluation of query-to-frame parsing only.

## Submission gates

The paper cannot be called submission-ready until all critical gates pass:

1. Run the frozen TEACh archive-to-manifest, Layer A/B audit, and Layer C manual
   labeling protocol on official data.
2. Report the complete gate denominators, floorplan-disjoint split, baseline
   comparisons, uncertainty, and failure slices.
3. Replace the abstract's explicit pending-result sentence with the verified
   official result, without expanding its scope beyond the benchmark.
4. Retain the completed task/pipeline figure and seven evidence-locked controlled
   result, component-ablation, decision-utility, observation-estimation, observation-grounding, external-language, and claim-boundary tables. Add
   the official TEACh result table only after its
   gate passes, with dataset, denominators, metrics, uncertainty, and claim
   boundaries in the caption.
5. Rebuild `paper/reproducibility_manifest.json` from a clean Git checkout, then
   run `python scripts/verify_reproducibility_manifest.py --require-runtime-match
   --require-git-revision`, `python scripts/verify_paper_claims.py`,
   `python scripts/build_paper_tables.py --check`, and the full test suite. A
   content-addressed snapshot without a clean revision binding does not satisfy
   this release gate.

