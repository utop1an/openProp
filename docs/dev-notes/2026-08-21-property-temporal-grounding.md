# Property-guided temporal grounding foundation

Date: 2026-08-21

## Context

The project began from an open-world entity-matching proposal: represent each entity with an extensible property set, then let language select and weight the properties relevant to a referring expression. A property has a name, description, type, value, confidence, source, and—when it is an observation—time metadata. Different value families keep different comparison semantics.

The discussion then added three requirements:

1. unknown properties must remain missing evidence rather than mismatches;
2. irrelevant properties must not dominate matching;
3. state evidence must become less reliable over time, with context and intervening events affecting persistence.

The working project name is OpenProp. Earlier naming discussion also considered PropGround; OpenProp was retained in the repository because the extensible property dictionary is central to the current implementation.

## Implemented foundation

- Extensible `PropertyRegistry` with aliases and conservative schema growth.
- Typed semantic, categorical, numeric, vector, relation, entity-reference, and temporal values.
- Structured relations that preserve predicate and argument identity.
- Explicit `observed`, `unknown`, and `not_applicable` states.
- Relevance-weighted matching with separate match and evidence-coverage scores.
- Strict LLM query parsing plus local Ollama support.
- Static bilingual and irrelevant-attribute benchmarks.
- Fixed temporal decay with explicit `as_of` replay and event invalidation.
- Replaceable `PersistenceModel` boundary.
- Context-conditioned neural exponential hazard model with right-censored survival training.
- Observation-history JSONL ingestion, entity-grouped train/validation/test splits, validation calibration, multi-horizon metrics, and model save/load.
- End-to-end temporal entity-grounding benchmark with observations separated from held-out current truth.

## Verified results

### Learned persistence

The synthetic persistence experiment used 600 episodes. The contextual model learned a higher hazard for `on(cup, table)` than for `inside(cup, cabinet)`. In the complete grouped pipeline:

```text
train / validation / test: 420 / 90 / 90
training loss:              2.4846 -> 1.2736
validation hazard scale:    0.7695
validation NLL:             1.2624 -> 1.2424
test NLL:                   1.3757
```

The global calibration factor improved overall validation likelihood and short-horizon calibration, but not every longer horizon. This motivates time-dependent calibration rather than test-set retuning.

### Temporal grounding

The 40-case diagnostic benchmark contains stale locations, missing observations, invalidating events, irrelevant attributes, bilingual queries, and static controls:

| Strategy | Top-1 | MRR |
|---|---:|---:|
| No temporal decay | 0.250 | 0.625 |
| Fixed temporal decay | 1.000 | 1.000 |
| Learned contextual decay | 1.000 | 1.000 |

Candidate-order reversal is tested so the gain is not caused by stable sorting of tied candidates. All 37 automated tests pass.

## Research interpretation

The current evidence supports two narrow conclusions:

- context-conditioned survival learning can recover deliberately different synthetic transition rates;
- temporal confidence can improve the final entity-ranking decision under controlled stale and incomplete evidence.

It does not yet support a claim that learned persistence outperforms a fixed policy, because both solve the current temporal benchmark. It also does not establish real-world persistence probabilities, robustness to noisy language parsing, or freedom from observation-policy bias.

## Follow-ups

The next academic priority is a benchmark that distinguishes learned contextual persistence from fixed half-lives. It should include held-out entity/context combinations, different relations with different dynamics, longer and irregular histories, ambiguous or unobserved events, temporal distribution shift, and noisy language-to-property parsing.

After that:

1. add simple statistical survival baselines such as per-context exponential, Weibull, and Cox models;
2. report C-index, integrated Brier score, calibration curves, and final grounding accuracy;
3. model observation availability separately from state persistence;
4. collect or annotate real longitudinal histories;
5. compare learned relevance and LLM property selection against equal/all-property baselines.

## Validation commands

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python examples/train_persistence_pipeline.py
python scripts/evaluate_temporal_grounding.py --learned-model artifacts/contextual_persistence.pt
```

