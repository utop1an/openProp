# Method-to-implementation audit

Date: 2026-08-27

This audit freezes the executable meaning of Sections 2--3 of the manuscript.
It is a consistency artifact, not additional experimental evidence.

| Manuscript object | Executable representation | Enforced boundary |
|---|---|---|
| Entity set `E` and observation `o_ik` | `Entity` and `Observation` in `src/openprop/models.py` | Values remain typed; confidence is in `[0,1]`; an observed state requires a value. |
| Schema `D_k` | `PropertyDefinition` and `PropertyRegistry` in `src/openprop/property_registry.py` | Exact or alias resolution precedes fuzzy resolution; schema growth is explicit and opt-in. |
| Query frame `Q` | `QueryFrame`, `PropertyConstraint`, and `MentionBasedSelector` in `src/openprop/selectors.py` | Resolution confidence multiplies relevance; unresolved properties do not enter scoring. |
| Typed match `m_ik` | `ComparatorRegistry` in `src/openprop/comparators.py` | One comparator is selected by value family or an explicit property override; scores must be finite and in `[0,1]`. |
| Freshness `f_ik(T)` | `PersistenceModel` in `src/openprop/persistence.py` and learned models in `src/openprop/statistical_persistence.py` | The matcher sees only a survival/freshness value and cannot inspect evaluation truth. |
| Effective evidence mass `a_ik` | `effective_weight` in `src/openprop/matcher.py` | Relevance, schema-resolution confidence, observation confidence, and freshness are multiplicative. |
| Match `M_i`, coverage `C_i`, rank score `s_i` | `match_score`, `coverage`, and `score` in `src/openprop/matcher.py` | Missing evidence does not enter the match numerator or denominator; ties use lexical entity ID. |
| Change and censoring records | `ObservationHistory` conversion in `src/openprop/observation_history.py` and `PersistenceTrainingExample` in `src/openprop/persistence_data.py` | Detected changes can be interval-censored; unchanged follow-up is right-censoring, not a negative label. |
| Observation-process parameters `q_0,q_1,s,f` | Emission likelihoods in `src/openprop/informative_observation.py` and forward-backward estimation in `src/openprop/observation_em.py` | Missing, negative, and positive outcomes form a normalized typed emission model; `f` estimation is explicit opt-in and defaults to perfect specificity for backward compatibility. |
| Semantic parser boundary | Strict response parsing in `src/openprop/llm.py` | Parsing ends at a typed frame; candidates and `current_truth` are not parser inputs. |

## Deliberate semantics and limitations

- Unknown and not-applicable observations both contribute zero evidence mass in
  the current ranker, but their distinct states remain visible in the
  per-property audit. Unknown is never converted into a mismatch. A future task
  utility may assign separate abstention costs without changing stored states.
- The main learned persistence model is factorized exponential survival, not a
  neural contribution. It composes coefficients for familiar factor values on
  unseen complete tuples. A factor value absent from training contributes its
  default zero effect; this support boundary must be reported rather than hidden.
- Fixed and learned persistence share event-retention handling. The learned
  survival term replaces time decay, while post-observation events multiply it.
- Comparator quality is modular. The dependency-free semantic comparator is a
  token-overlap baseline; claims concern the typed scoring boundary rather than
  a new semantic encoder.
- The observation-process extension estimates one homogeneous false-positive
  rate. Its optional reversible binary CTMC estimates forward and return rates
  from training observation sequences and reaches the matcher only through a
  property-specific Boolean adapter. A source-aware variant assigns provenance-specific
  inspection, sensitivity, and false-positive parameters while sharing dynamics.
  It does not model correlated detectors, source churn, informative opportunity timing, or
  multi-valued recurrent states. The irregular-time variant stores one positive
  elapsed interval per outcome and computes its exact CTMC matrix; its
  generalized M-step accepts only likelihood-preserving rate updates.
- `current_truth` is evaluation-only and is absent from entity, observation,
  query-frame, matcher, and persistence prediction interfaces.

## Review checklist

1. Every score equation in the paper can be reconstructed from the matcher.
2. Every survival likelihood distinguishes exact, interval-censored, and
   right-censored records.
3. Train entities fit effects, validation data selects/calibrates, and test data
   is used once for reporting.
4. Language-only results are not described as temporal, visual, or integrated
   grounding evidence.
5. Synthetic benchmarks are described only as mechanism validation.
6. Observation-process claims state whether specificity is fixed or estimated
   and retain the simultaneous low-false-positive power boundary.
7. Recurrent-process claims use independent exact-state test rows, exclude the
   zero-return compatibility control from the primary family, and remain synthetic.
8. Irregular-timing claims hold total follow-up fixed, pair burst positions, and
   exclude the zero-contrast regular-grid control from the primary family.
