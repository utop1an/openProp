# Closing-section audit

Date: 2026-08-26

This document records the outline, paragraph roles, claim evidence, and
reviewer-facing self-review for manuscript Sections 7--9. It is an internal
writing audit, not submission evidence.

## Compact outline

1. Interpret the controlled evidence as support for an explicit typed decision
   boundary, not a universal persistence model.
2. Use the neural, adaptation, and observation-process negative results to
   identify what actually carries the current contribution.
3. Separate data scope, model assumptions, upstream-system dependence, and
   broader-impact risks.
4. Close with the strongest verified mechanisms and name official longitudinal
   evaluation as the next decisive test.

## Reverse outline and paragraph roles

| Section and paragraph | Role | One-sentence message |
|---|---|---|
| Discussion 1 | Opening/interpretation | OpenProp's supported contribution is an auditable current-evidence decision boundary. |
| Discussion 2 | Mechanistic insight | Typed factorization, rather than neural capacity, explains the controlled compositional result. |
| Discussion 3 | Observation insight | Observation timing is part of the grounding problem because detection-time bias can change a rank. |
| Discussion 4 | Negative evidence | Adaptation failures and language-only results constrain, rather than broaden, the claim. |
| Limitations 1 | Data scope | Synthetic and templated-language evidence does not establish naturalistic longitudinal performance. |
| Limitations 2 | Model scope | The main model assumes a factorized exponential hazard and the executed observation studies omit important real-process complications. |
| Limitations 3 | System boundary | Candidate recall, identity, perception, mapping, and action remain upstream. |
| Limitations 4 | Broader impact | Freshness scoring may reproduce unequal observation coverage, so provenance, abstention, and restrictions on sensitive properties are necessary. |
| Conclusion 1 | Restatement | Current-entity grounding is a typed decision under incomplete and stale evidence. |
| Conclusion 2 | Evidence | Controlled studies validate typed composition, axis-level decision utility, and interval-aware ranking; ALFRED validates only the parsing boundary. |
| Conclusion 3 | Final boundary/direction | Official floorplan-disjoint longitudinal evaluation is required before an integrated effectiveness claim. |

## Claim-evidence map

Claim: typed factorization can reuse familiar values on held-out complete
combinations. | Evidence: `C1_TYPED_COMPOSITION` in `paper/claims.json`. |
Status: supported as synthetic mechanism validation.

Claim: each typed axis can contribute to calibration and alter a controlled
rank. | Evidence: `C2_TYPED_COMPONENTS` and `C3_DECISION_UTILITY`. | Status:
supported on the declared generator and analytic confirmation only.

Claim: interval-aware likelihood can remove inspection-schedule-induced ranking
bias. | Evidence: `C4_INTERVAL_CENSORING`. | Status: supported under the frozen
equal-hazard synthetic intervention.

Claim: specificity estimation can remove high false-positive persistence bias.
| Evidence: `C6_FALSE_POSITIVE_OBSERVATIONS`. | Status: supported at the 0.10
synthetic violation; simultaneous NLL intervals at 0.02 and 0.05 retain the
finite-sample boundary.

Claim: a reversible binary CTMC can repair current-state prediction under
matched recurrent dynamics. | Evidence: `C7_RECURRENT_OBSERVATIONS`. | Status:
supported synthetically at all three nonzero return rates; zero-return control retained.

Claim: exact elapsed intervals prevent bursty timing from biasing reversible
state estimation. | Evidence: `C8_IRREGULAR_OBSERVATIONS`. | Status: supported
synthetically at all three nonzero contrasts; regular-grid control retained.

Claim: positive span evidence improves supervised typed-frame retrieval. |
Evidence: `C5_EXTERNAL_LANGUAGE`. | Status: supported for ALFRED
language-to-frame parsing only.

Claim: neural persistence is necessary. | Evidence: `N1_NEURAL_NECESSITY`. |
Status: contradicted and excluded from the conclusion.

Claim: OpenProp improves semi-real longitudinal grounding. | Evidence:
`N2_REAL_WORLD_GROUNDING`. | Status: pending; the conclusion must state the
missing test, not imply its outcome.

Claim: target adaptation is generally safe. | Evidence:
`N3_GENERAL_ADAPTATION_SAFETY`. | Status: contradicted and retained as a design
boundary.

## Five-dimension self-review

| Dimension | Judgment | Evidence or required restraint |
|---|---|---|
| Contribution | Pass for the bounded draft | The close centers the typed, auditable grounding boundary and observation-process insight rather than neural novelty. |
| Writing clarity | Pass | Each paragraph has one declared role, uses locked terminology, and separates interpretation from limitations. |
| Experimental strength | Needs new experiment | Controlled effects and external parsing are reproducible, but official longitudinal grounding is absent. |
| Evaluation completeness | Needs new experiment | TEACh Layer B/C, independent rich-frame labels, and recent system-level comparisons remain required. |
| Method soundness | Needs official-data audit | Recurrent and exogenous bursty timing mechanisms are controlled; informative timing, correlated sources, identity errors, and support coverage must still be measured rather than assumed away. |

## Submission-prose boundary

The former numbered reproducibility checklist is removed from the manuscript
ending. Its requirements remain in `paper/reproducibility.md`,
`paper/reproducibility_manifest.json`, `paper/claims.json`, and the submission
form. This prevents internal release mechanics from displacing the scientific
takeaway while preserving stricter executable gates.

Forbidden closing claims remain:

- real-world or semi-real longitudinal effectiveness;
- neural architectural novelty;
- general adaptation or arbitrary-noise safety;
- natural prevalence of analytic boundary cases;
- visual or temporal grounding evidence from ALFRED frame parsing.
