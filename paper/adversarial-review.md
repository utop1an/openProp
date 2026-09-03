# Adversarial pre-submission review

## Overall recommendation: reject in current form

The system is unusually careful about typed evidence, missingness, censoring,
and leakage, but the current paper-level evidence is dominated by controlled
generators. The core formulation could support a strong submission; the present
record does not yet establish that it matters on longitudinal embodied data.
This is a solvable evidence gap, not a reason to inflate the claims.

## Five-dimension review

| Dimension | Judgment | Strongest evidence | Main rejection reason | Required repair |
|---|---|---|---|---|
| Contribution | Needs revision | A coherent typed decision boundary links language, comparison, persistence, and coverage. | The main learned model is a simple factorization and beats the neural model on a generator built from the same factorization; novelty may look like benchmark alignment. | Lead with the task/evaluation boundary, then show value on official longitudinal data and at least one mismatch condition. |
| Soundness | Promising | Hidden truth is evaluation-only; censoring, provenance, disjoint splits, confirmation seeds, reversible state, exact irregular-time, and source-specific reliability stresses are explicit. | Synthetic assumptions remain strong; recurrence, burst timing, and independent-source reliability are matched-model results, while informative timing, correlated/adversarial sources, timestamp error, and multi-valued dynamics remain untested. | Audit these assumptions on TEACh and report unidentifiable cases rather than imputing them. |
| Experimental strength | Insufficient | Five- and ten-seed paired experiments with uncertainty isolate mechanisms. | There is no integrated semi-real temporal grounding result or common-benchmark comparison. | Execute the frozen TEACh experiment with static/no-decay, fixed-decay, factorized, and relevant learned baselines. |
| Writing and positioning | Complete for the current evidence, empirically incomplete | The teaser and pipeline figures lead into formal Methods and Experimental Protocol sections; a claim-audited Discussion, four-part Limitations/Broader Impact section, and bounded Conclusion now close the scientific story. Internal release mechanics live outside the numbered submission narrative. | The conclusion must still change after an official TEACh table demonstrates or falsifies the integrated claim. | Add the verified TEACh outcome and failure slices without weakening current boundaries. |
| Reproducibility | Strong foundation | Thirteen claim artifacts and 309 exact metric/protocol assertions are machine checked. Reproducibility schema v2 binds the observed runtime, eleven experiment entries with thirteen experiment outputs, one separately classified non-performance TEACh access audit, two vector figures, and a non-mutating table check; the deterministic suite is green. The access audit pins the official downloader commit and records four 403 probes without pretending to be dataset evidence. | The desktop snapshot cannot bind a clean Git revision, and the official TEACh feasibility/performance artifacts do not exist. | Rebuild from a clean revision with the release gate enabled; bind the TEACh run and official table to both manifests. |

## Major questions a skeptical reviewer will ask

1. **Is the compositional result circular?** The generator factorizes subject,
   relation, and scene, and the winning model uses the same structure. Add a
   declared misspecified or partially interacting official-data analysis; do
   not frame the current result as universal compositional generalization.
2. **Why is this a grounding paper rather than a survival-analysis benchmark?**
   The component-balanced and inspection-confounded confirmations now show that
   typed persistence and interval semantics alter controlled candidate rankings,
   but both are analytic. TEACh must show that calibration differences alter real candidate
   rankings under historical evidence.
3. **Where is the end-to-end language result?** The frozen Layer C runner now
   separates official target-type and predicted frames, uses strictly pre-action
   candidates, and retains all coverage/parse failures. A three-annotator,
   target/candidate/model-blind rich-frame resolver with exact spans and a frozen
   agreement gate now closes the annotation-design gap. It does not close the
   evidence gap: official labels and execution remain absent.
4. **What is learned beyond fixed decay?** The existing temporal benchmark ties
   learned and fixed decay. The official experiment must include per-property
   fixed decay, globally learned decay, typed factorization, and a no-temporal
   baseline under the same cases.
5. **How does this differ from dynamic scene memories?** Position OpenProp as an
   auditable current-entity decision layer over historical evidence, not as a
   replacement for perception or mapping. A direct conceptual diagram should
   show where a dynamic map supplies observations and where OpenProp begins.
6. **Are negative results diluting the paper?** Keep only those that protect the
   main interpretation: neural non-necessity, observation-process bias, and the
   boundary of adaptation safety. Move the broader adaptation matrix to an
   appendix or follow-up paper unless TEACh directly motivates it.

## Protocol-specific review after the Section 4 freeze

| Dimension | Judgment | Evidence and unresolved issue |
|---|---|---|
| Contribution | Needs new experiment | The protocol now prevents one mechanism block from impersonating integrated grounding, but protocol quality is not itself external evidence. |
| Writing clarity | Pass for the current evidence | Questions, populations, split ownership, interventions, metrics, cluster units, and denominators are explicit and mapped to executable artifacts. |
| Experimental strength | Needs new experiment | Controlled effects are large and paired; the only executed external block ends at typed-frame prediction. |
| Evaluation completeness | Needs new experiment | Component and observation interventions are well isolated, but official longitudinal candidates and recent system-level comparisons remain absent. |
| Method soundness | Needs revision after official data | Primary comparison families use shared paired-seed simultaneous intervals. Observation EM exposes false positives, reversible state, and exact elapsed intervals while retaining zero/low-violation controls. Informative timing, source reliability, multi-valued state, and natural observability still require official histories. |

The Section 4 reverse outline now follows one message per block: evaluation
separation; typed persistence and decisions; observation-process intervention;
external language; fair baselines and inference; and the pending official gate.
No paragraph reports a TEACh outcome or promotes the ALFRED result into temporal
grounding.

## Reverse outline of the current introduction

| Paragraph | Job | Verdict |
|---|---|---|
| 1 | Establish current-entity grounding from heterogeneous historical memory. | Keep. Concrete and task-first. |
| 2 | Decompose typed comparison, persistence, and observation-process challenges. | Keep; support with one compact schematic. |
| 3 | Locate the gap relative to open-vocabulary and dynamic memories. | Revised after the 2026-08-26 audit: keep the complementary evidence-scoring boundary and avoid priority over the broader phrase “dynamic grounding.” |
| 4 | Give the OpenProp design and deny neural novelty. | Keep; add the scoring equation reference. |
| 5 | Preview evidence and state the external-validity blocker. | Replace the blocker sentence with verified TEACh evidence before submission. |

## Ranked action list

1. **Critical:** run and independently audit the official TEACh longitudinal
   protocol; freeze its artifacts and add their checks to `claims.json`.
2. **Critical:** collect three independent labels with the frozen blind Layer C
   templates, pass majority resolution and the 0.80 agreement gate, then execute
   type-only oracle, rich oracle, and predicted-frame reports with complete
   coverage and rejection denominators.
3. **High:** retain the completed task/pipeline figure and seven evidence-locked
   controlled, ablation, decision, observation-estimation, observation-grounding, external-language, and
   boundary tables; append the official table only after the TEACh
   result fixes its rows and claim order.
4. **High:** the official runner now freezes no-time, fixed decay, global,
   exact-context-backoff, property-only, nested typed-factor, and full
   factorized persistence rows on identical cases. Execute and report the
   matrix; require a gain over property-only before attributing value to typed
   context.
5. **Medium:** test the main conclusion under context interactions or other
   misspecification present in official histories.
6. **Medium:** compress the adaptation program into a boundary analysis; do not
   let it displace the central grounding story.

## Acceptance condition for the next review

The next review should begin only after a machine-verified TEACh claim exists.
If the official gate fails because current truth, history coverage, or aligned
language cases are insufficient, the paper must change its benchmark or narrow
its thesis rather than treating a failed gate as missing paperwork.

