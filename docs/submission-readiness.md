# Submission readiness audit

Date: 2026-08-27
Target: a competitive ICLR submission, not only a working research prototype.

## Current verdict

OpenProp has a coherent problem boundary and reproducible mechanism experiments,
but it is not submission-ready. Official ALFRED human descriptions now provide
a narrow external language audit, not temporal grounding evidence. The decisive
limitation remains external validity: there is no semi-real longitudinal
performance result with observation histories and independently held current truth.

## Reviewer-style assessment

| Dimension | Status | Evidence | Required resolution |
|---|---|---|---|
| Contribution | Needs revision | Typed property matching, compositional persistence, observation-process handling, and calibration-only detection of typed-risk reversal form a coherent pipeline, but a strong factorized statistical model still outperforms the neural parameterization on the current generator. | Center the new task and typed, observation-aware modeling boundary plus safe adaptation; do not claim neural architectural novelty without a benchmark that requires it. |
| Writing clarity | Needs revision | An evidence-locked claim hierarchy, source-grounded 2023--2026 related-work audit, claim-bound first-page teaser, detailed vector pipeline figure, and seven controlled, secondary, and boundary tables now exist. The teaser exposes the inspection-confounded failure, rank repair, evaluation-only truth, and synthetic scope without implying upstream perception. Thirteen claim artifacts and 309 metric/protocol assertions are executable; reproducibility schema v2 separately binds eleven experiments and the non-performance TEACh access audit. The official TEACh table and official-result narrative remain absent. | Keep prose and both figures bound to the executable manifests; add the official table and final result order only after the TEACh gate fixes the external claim. |
| Experimental strength | Needs new experiment | Five-seed temporal mechanism experiments, two untouched ten-seed analytic grounding confirmations, multiple adaptation studies, and external language audits are reproducible. Interval-aware training now improves inspection-confounded Top-1 by 0.450 [0.350, 0.500] and removes a 0.900 target-scene gap. Non-affine, sparse, and repeated-evidence stress tests reject overbroad robustness claims. A train-only BM25 typed-frame baseline is stronger than both local LLM parsers. TEACh now has high-precision alignment, a cryptographically bound manual gate, a strictly pre-action evaluator, and a target/candidate/model-blind rich-frame annotation resolver, but no official labels or result exist. | Run the gate and both Layer B/C evaluators on official longitudinal data; collect three independent Layer C labels and require the frozen 0.80 agreement gate before reporting richer referential grounding. |
| Evaluation completeness | Needs new experiment | Static, interference, temporal, survival, grounding, factorized log-linear, Weibull, piecewise, Cox, duration shift, latent-mechanism shift, reversible binary state, bursty irregular timing, source-specific reliability, exact/local/pairwise/three-way target adaptation, source misspecification, novel typed values, calibration noise, complexity screening, changed/stable and support-stratified safety, joint observation-state, EM-identifiability, external ALFRED parsing, retrieval, exact-overlap controls, and a three-layer TEACh readiness protocol exist. | Add official longitudinal evaluation; test correlated and changing sources, ablate any grounding-aware loss, and add broader learned language baselines and recent external methods. |
| Method soundness | Needs new experiment | Hidden truth is separated, groups and test streams are disjoint, and censoring is conservative. Specificity estimation improves only the largest tested false-positive violation. Reversible, exact-irregular, and source-specific CTMC fits each win 5/5 seeds across all three nonzero primary conditions and retain near-tie controls. Repeated evidence retains its equal-budget power boundary, and the rejected concordance guard remains visible. | Verify invariants and alignment precision on official data; test informative timing, correlated/adversarial sources, source churn, and multi-valued recurrence. |

The official-data execution path now begins with strict all-session discovery,
game/replay timestamp pairing, and content-hashed manifest generation. This
closes a manual-selection reproducibility gap, but it does not change the absence
of an official semi-real result.
The current access snapshot additionally proves that all four required official
HEAD probes return 403 and that issue #37 has no maintainer replacement. It
makes the blocker auditable, but it is infrastructure evidence only and leaves
`release_ready=false`.


## Current claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Relevance weighting can resist explicitly irrelevant attributes. | Synthetic interference Top-1 is 1.000 for weighted constraints and 0.033 for equal weights. | Supported only as mechanism validation. |
| Temporal confidence can change the final entity decision. | Synthetic temporal grounding Top-1 is 1.000 with decay and 0.250 without decay. | Supported only as mechanism validation. |
| Typed context factors can generalize to held-out combinations. | The factorized exponential model achieves NLL 1.239 +/- 0.069, C-index 0.749 +/- 0.014, IBS 0.079 +/- 0.010 and Top-1 1.000 +/- 0.000; it outperforms the neural model on every aggregate metric. | Supported under the factorized synthetic generator; neural necessity is contradicted. |
| Subject, relation, and scene each contribute to compositional survival calibration. | Removing scene, relation, or subject worsens paired test NLL by 0.402 [0.364, 0.441], 0.156 [0.137, 0.176], and 0.052 [0.038, 0.066], respectively, with 10/10 wins. Brackets are family-wise simultaneous 95% intervals across the three predeclared NLL comparisons. | Survival-component necessity is supported only on the synthetic generator; downstream necessity for every factor is contradicted by the original grounding cases. |
| Subject, relation, and scene persistence can each change a controlled grounding decision. | On ten untouched confirmation seeds, full-context probe Top-1 is 0.973/0.993/1.000 and exceeds no-subject/no-relation/no-scene by 0.347 [0.329, 0.364], 0.327 [0.313, 0.340], and 0.500 [0.500, 0.500], with 10/10 wins. Brackets are family-wise simultaneous 95% intervals across the three probes. | Supported only for analytically balanced cases from the synthetic factorized generator; not naturalistic or semi-real evidence. |
| Interval-aware learning reduces inspection-frequency bias. | Five-seed hazard MAE falls from 0.0559 +/- 0.0044 to 0.0058 +/- 0.0044; the false schedule gap falls from 0.0745 to 0.0038. | Supported under synthetic exponential dynamics. |
| Inspection-frequency bias can change a controlled entity decision. | On ten untouched seeds and 40 target-scene-balanced cases, interval-aware Top-1 is 1.000 versus 0.550 +/- 0.150 for detected-time training; paired gain is 0.450 [0.350, 0.500] with 9/1/0 wins/ties/losses. Worst-scene Top-1 rises from 0.100 to 1.000 and the target-scene gap falls from 0.900 to 0.000. | Supported only as synthetic analytic decision evidence; not natural prevalence or real-world grounding. |
| Non-exponential persistence can be learned and evaluated without misusing exponential likelihoods. | Weibull shape recovery is 0.610 +/- 0.021 for true 0.6 and 1.631 +/- 0.066 for true 1.6; paired NLL improves by 0.123 and 0.099 with 5/5 wins, while true shape 1.0 is unchanged. | Supported as synthetic model-misspecification validation. |
| Typed survival models tolerate moderate follow-up duration shift when latent dynamics are stationary. | With train/validation/test horizons 6/12/24 h and evaluation through 18 h, Weibull paired NLL penalties are 0.003 for shape 0.6 and 0.006 for shape 1.6; all conditions use identical test rows. | Supported only for synthetic non-informative duration-support shift. |
| Observation-aware persistence can estimate inspection and detection parameters from training logs when the process is identifiable. | All 20 factorial EM runs converge; mean parameter errors are below 0.009 and the estimated-model NLL gap to the logged-parameter upper bound is below 0.0002. The 50-episode stress setting exposes unstable hazard recovery despite numerical convergence. | Supported only for the synthetic irreversible process with a regular grid and known initial state. |
| Training-only specificity estimation can remove a substantial false-positive bias. | At false-positive rate 0.10, estimated specificity reduces hazard MAE from 0.0284 to 0.0061 and fixed-minus-estimated exact-test NLL is 0.00615 [0.00343, 0.00887] with 5/5 wins. At 0.02 and 0.05, simultaneous intervals cross zero. | Supported only as a high-violation synthetic mechanism result; lower-noise benefit is not established. |
| A reversible binary observation process can repair an irreversible current-state model when returns occur. | At return rates 0.15/0.30/0.45 per hour, irreversible-minus-reversible exact-state NLL is 0.06038/0.08099/0.08263; all simultaneous 95% intervals exclude zero and each condition wins 5/5 seeds. Zero-return NLL differs by 0.00005. | Supported only for the matched synthetic binary CTMC, regular opportunities, known initial state, and homogeneous detector. |
| Exact elapsed intervals prevent bursty observation timing from becoming a false rate signal. | At gap contrasts 0.50/0.75/0.90, mean-grid-minus-exact current-state NLL is 0.00391/0.01446/0.03140; all simultaneous intervals exclude zero and each condition wins 5/5 seeds. The regular-grid control differs by 0.00001. | Supported only for exogenous two-gap synthetic schedules with fixed follow-up, binary CTMC dynamics, known initial state, and homogeneous detector. |
| Cox is competitive when typed risk ordering is stationary. | In distribution, Cox C-index is 0.751 and IBS is 0.080, with same-test oracle regrets 0.000 and 0.001. Cox continuous event-time NLL is deliberately not reported. | Supported only for the paired synthetic benchmark. |
| Source-trained typed models are not robust to arbitrary factor reversal. | Under typed factor reversal, all four models have C-index about 0.18 versus oracle 0.82; C-index regret is 0.637-0.638 and IBS regret is 0.533-0.555. | Supported as synthetic failure analysis, not a robustness guarantee. |
| OpenProp improves real-world open-world grounding. | No real or semi-real longitudinal benchmark currently supports this statement. | Unsupported; must not appear as a result claim. |
| A small target calibration set can repair the benchmark's exact typed-factor reversal without test leakage. | Across ten seeds, six labels improve NLL/C-index/IBS from 3.506/0.183/0.627 to 1.191/0.817/0.129 with 10/10 paired wins; the gate activates in 50/50 reversal and 0/200 non-reversal decisions. | Supported only for the synthetic global affine reversal with known target contexts. |
| A declared typed axis can localize a partial reversal without changing stable groups. | At 144 labels over 18 contexts, subject and scene gates improve C-index from 0.576 to 0.752 and 0.521 to 0.705 with 10/10 wins; stable groups are exactly unchanged, including under 20% label flips. At 36 labels the scene gate false-activates in 2/10 seeds; under XOR, stable C-index falls from 0.757 to 0.680. | Supported only for simple synthetic local shifts at adequate calibration support. The XOR result contradicts universal safety and motivates interaction-aware control. |
| Multiplicity-controlled typed interactions can repair the XOR subgroup failure. | At 288 clean labels, hierarchy NLL/C-index/IBS is 1.500/0.732/0.157 versus source 1.983/0.504/0.272 and subject gate 1.701/0.622/0.204; stable C stays 0.747. All hierarchy-versus-subject metrics improve in 10/10 seeds. Exact XOR recovery is 8/10 clean and 3/10 with 20% label flips; one noisy ID decision false-activates. | Supported only for the paired synthetic generator at high calibration support. Zero-error, arbitrary-noise, source-misspecification, and real-world safety claims remain unsupported. |
| Parser tolerance is not semantic grounding repair. | On the 40-case, six-template gemma3:4b audit, tolerance raises parse success from 0.750 to 1.000 but leaves all-case Top-1 at 0.375; value recall is 0.625. | Supported as a single-run execution-path and failure analysis, not a model-quality claim. |
| Typed schema repair is guarded and can recover field-permutation errors without grounding leakage. | Development Top-1 rises 0.375 to 0.750. On 30 unseen bilingual paraphrases, gemma rises 0.675 to 0.700 with no rank regressions, while llama3.2 triggers no repairs and remains 0.725. Across 80 cases the paired gain is 0.013, cluster-bootstrap 95% CI [0.000, 0.039], sign p=1.000. | Guarded transfer is supported; a general accuracy improvement is not statistically established. |
| Current structured parsing transfers cleanly to independently authored human task language. | On a frozen 40-case ALFRED valid-unseen sample, repaired strict canonical-value recall is 0.296 for gemma3:4b and 0.171 for llama3.2; exact frames are 1/40 and 0/40. Across all 945 supported descriptions, exact PDDL object labels occur in 0.762 and 60 cases contain an explicit conflicting object label. | Contradicted as a clean transfer claim. This is external parser evidence only, and strict canonical matching is confounded by aliases and verified label conflicts. |
| A fail-closed typed ontology can improve external canonical values without validation-label access. | Train PDDL vocabulary plus fixed schema semantics raises value recall from 0.296 to 0.508 and 0.171 to 0.400; paired gains are 20/0 and 23/0 cases, aggregate 95% CI [0.160, 0.281]. Atomic and leave-one-out ablations localize gains to complementary relation, state, and type components. | Supported on the frozen 40-case ALFRED protocol; not evidence of complete parsing, visual grounding, or temporal reasoning. |
| Controlled typed adaptation can repair source misspecification that global calibration cannot. | At 288 clean labels, scene and joint typed permutations improve controlled-general versus global-affine C-index by 0.219 [0.192, 0.247] and 0.127 [0.090, 0.164], with 10/10 wins. NLL wins are 8/10 for each and the joint interval crosses zero. Correct-source clean runs remain inactive in 10/10; subject-cycle activates in 9/10, while noisy correct-source activation is 1/10. | Supported only for the paired synthetic typed permutations; uniform metric dominance, exact recovery, arbitrary-noise safety, target-only dominance, and real-world effectiveness are unsupported. |
| Support-aware higher-order adaptation preserves unsupported values and separates pairwise from three-way shifts. | All hierarchical variants match source exactly on the test-only value slice in 200/200 runs. At maximum clean support, calibrated novel values select subject in 10/10; Latin selects three-way in 10/10 and beats pairwise by 0.042 NLL [0.037, 0.047] and 0.023 C-index [0.019, 0.027]. At minimum support Latin exact recovery is 1/10, and noisy controls often activate. | Supported as synthetic mechanism validation. Recovery of unsupported shifts, few-shot high-order identification, general noise robustness, and real-world effectiveness are unsupported. |
| Explicit positive query evidence can repair property selection without treating absence as negative. | On a pre-frozen task-disjoint confirmation sample, two-model mean property-F1 delta is +0.196 [0.163, 0.231] and exact-frame delta is +0.238 [0.138, 0.350], with 49/0 F1 improvements. Across all 945 validation descriptions, evidence-only selection precision is about 0.999 and recall is 0.711-0.746. Absence gating is rejected after development regressions. | Supported for ALFRED language-to-frame parsing; broader models, languages, repeated generations, and end-to-end grounding remain untested. |
| Positive span evidence improves a strong train-only retrieval baseline. | On full valid-seen/unseen, BM25 plus evidence reaches F1 0.989/0.986, value recall 0.880/0.896, and exact frames 0.710/0.736. Against BM25, task-clustered 95% CIs exclude zero on all three metrics. Novel-query subsets retain the gains. | Supported for supervised ALFRED language-to-frame parsing; not temporal, visual, or open-world grounding evidence. |
| Independent repeated evidence can identify and suppress homogeneous symmetric calibration noise. | At 20% flips, the disagreement estimator recovers mean 0.201; five-label confidence abstention reduces status error to 0.84% and correct-source activation from 2/10 to 0/10. Equal-budget variants lose fold power, and bump affected C-index falls by 0.036. | Supported only as a synthetic identifiability and cost boundary; general robust adaptation and ranking safety are contradicted. |

## Priority order

1. Execute the frozen TEACh feasibility and experiment path on official data,
   beginning with the fail-closed archive-to-manifest command so no session can
   be dropped manually; run the frozen
   no-decay/fixed/global/exact-context/property-only/nested/full
   model matrix, and require the full typed model to beat property-only before
   attributing a gain to context rather than property-specific decay. The
   report hashes the manifest and feasibility audit and exposes feature/context
   support and global backoff. Complete three target/candidate/model-blind Layer C
   annotation files, pass deterministic majority resolution and the frozen 0.80
   pairwise agreement gate, then execute the strictly pre-action type-oracle,
   rich-oracle, and predicted-frame comparison. Report same-type ambiguity,
   target-unobserved coverage, and every gate denominator before any semi-real
   performance claim.
2. Test evidence-constrained selection across additional learned model families,
   repeated generations, and multilingual/paraphrastic language while retaining
   every failure; BM25 retrieval is now the minimum language baseline.
3. Treat non-affine and label-noise adaptation as mapped failure boundaries;
   revisit them only with correlated-source evidence or a genuinely independent
   ranking confirmation set, not another same-calibration heuristic.
4. Typed-context and component-balanced grounding ablations are complete. Add
   matched ablations for interval censoring, validation calibration, event
   retention, and any future grounding-aware loss.
5. The main claim hierarchy, evidence-locked working manuscript, limitations,
   computational snapshot, reproducibility checklist, and adversarial review are
   complete at the content-hash level. Rebuild the manifest from a clean Git
   revision before release. After TEACh fixes the external claim, retain the
   completed pipeline figure and current evidence-locked tables, then add the
   official table and finalize the camera-ready narrative around the completed teaser.

## 2026-08-26 non-affine stress update

The earlier priority to test non-affine local error is complete. The frozen
240-run audit changes the readiness judgment:

- **Experimental strength remains incomplete.** On a smooth interaction bump,
  controlled typed adaptation improves all-case NLL by 0.048
  [0.029, 0.067], but affected-subset C-index gain is exactly zero.
- **Method soundness now needs revision.** On the folded scene error, the gate
  recovers only 0.031 [0.000, 0.093] affected C-index versus a 0.192
  [0.101, 0.279] correct-source gap.
- **Noise safety is contradicted.** With 20% calibration label flips, both BIC
  and non-BIC typed variants activate in 5/10 correct-source controls.

Consequently, typed affine repair must not be presented as a general local
misspecification or noise-safe solution. Priority 3 is superseded by: develop a
small robust nonlinear typed calibrator and require it to improve the frozen
fold/bump audit without increasing noisy-control activation. The complete
protocol and claim boundary are in the
[local non-affine benchmark](non-affine-adaptation-stress-benchmark.md).


This audit should be revised whenever a verified experiment changes the
claim-evidence boundary.

## 2026-08-26 sparse-candidate confirmation update

The first proposed response to the non-affine failure has been tested and must
not be promoted. Nonlinear basis expansion is unnecessary in almost every
selected group and increases noisy-control activation versus sparse affine.
Sparse coverage closure then passes its fresh-seed safety criteria—0/10 noisy
correct-source activations versus 4/10 for the previous gate—but fails the
predeclared efficacy criterion: fold affected C-index improvement over the
previous gate is exactly 0.000. It also worsens bump all-case NLL by 0.003
[0.001, 0.005]. The artifact records `accepted: false`.

Method priority is therefore revised again: model calibration-label corruption
or require repeated/clean target evidence before adding more functional
flexibility. The rejected-candidate evidence is documented in the
[sparse adaptation candidate audit](sparse-adaptation-candidate-audit.md).

## 2026-08-26 repeated-evidence update

The explicit label-noise priority is complete as a development audit. Under the
declared homogeneous independent symmetric-noise model, pairwise disagreement
estimates the 0.20 flip mechanism at mean 0.201. Five annotations on the same
15 identities plus posterior-confidence abstention reduce evaluation-only label
error to 0.84% and correct-source false activation from 2/10 to 0/10.

This does not clear the method gate. Equal-budget repetition trades away too
many independent identities and loses fold power. Spending five times the label
budget preserves bump NLL and some fold C-index benefit, but bump affected
C-index declines by 0.036 [-0.109, 0.000]. A calibration C-index guard accepts
the seed with a -0.3629 held-out ranking delta. It therefore cannot be described
as a ranking-safety certificate, and no fresh confirmation was run.

The next decisive work returns to external validity: run the already frozen
TEACh path on official data. Repeated evidence remains a precise assumption and
cost ablation, not a new main method. See the
[repeated-evidence label-noise audit](repeated-evidence-label-noise-audit.md).
