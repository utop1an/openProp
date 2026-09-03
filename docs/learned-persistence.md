# Learned state persistence

OpenProp separates state observations from persistence prediction. Timestamps,
sources, confidence, and events are observation history; they are not ordinary
matching properties. `EntityMatcher` consumes a replaceable `PersistenceModel`:

```text
Observation + entity/relation context + event history + as_of
                              |
                              v
                     PersistenceModel
                              |
                              v
                     FreshnessResult
                              |
                              v
              effective confidence and coverage
```

`ExponentialPersistenceModel` preserves the fixed half-life baseline.
`NeuralPersistenceModel` uses categorical embeddings and a configurable-depth
MLP to predict a context-conditioned transition hazard.

## Training record

```python
PersistenceTrainingExample(
    property_name="location",
    subject_type="cup",
    state_predicate="on",
    context_object="table",
    scene="kitchen",
    duration_seconds=7200,
    event_observed=True,
)
```

`event_observed=True` means the state ended at the recorded duration. `False`
means observation stopped while the state had not yet been seen to change. The
latter is right-censored data, not a positive “still true forever” label.

The model minimizes exponential survival negative log-likelihood:

```text
event:     hazard * duration - log(hazard)
censored:  hazard * duration
```

The predicted persistence at time `t` is:

```text
P(still true at t | context) = exp(-hazard(context) * t)
```

## Synthetic end-to-end experiment

The included generator intentionally uses different underlying rates:

- `on(cup, table)`: transition hazard `0.50/hour`;
- `inside(cup, cabinet)`: transition hazard `0.05/hour`.

Run:

```powershell
python -m pip install -e ".[ml]"
$env:PYTHONPATH = "src"
python examples/train_contextual_persistence.py
```

Verified training output:

```text
examples: 600 (events=355, censored=245)
loss: 2.4810 -> 1.2801
on(table) hazard/hour:       0.5478
inside(cabinet) hazard/hour: 0.0554
on(table), after 5h:         0.0646
inside(cabinet), after 5h:   0.7582
```

Inject the trained model with:

```python
matcher = EntityMatcher(
    registry,
    comparators,
    selector,
    persistence_model=training.model,
)
```

## Research boundary

This proves the training and inference path can recover context-dependent state
persistence. It does not establish realistic table or cabinet probabilities.
The observation-history pipeline now provides JSONL ingestion, entity-grouped
train/validation/test splits, held-out calibration metrics, and model
serialization. See [the pipeline note](observation-history-pipeline.md).

The current neural head still assumes a constant hazard after conditioning on
context. A global validation-derived hazard multiplier improves overall
validation likelihood but does not calibrate every time horizon equally. Later
models should compare piecewise hazards or temporal models over event histories,
and real deployment still requires observation-policy bias analysis, learned
event effects, and uncertainty intervals.
