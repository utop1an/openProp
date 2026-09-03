# Temporal evidence

Some properties describe stable identity (`type`, `material`), while others are
stateful (`location`, `cleanliness`, `temperature`). OpenProp applies temporal
decay only when a property explicitly defines a `TemporalPolicy` and an
observation has a timestamp.

## Model

```text
effective confidence
  = observation confidence
  * time retention
  * event retention

time retention = 2 ^ (-age / half_life)
```

`minimum_freshness` can place a floor on time retention. Events after the
observation multiply event retention. An uncertain event interpolates between
no effect and its configured full effect:

```text
event factor = 1 - event_confidence * (1 - configured_retention)
```

Events before the observation and events after the requested `as_of` time are
ignored.

## Example policies

```python
location = PropertyDefinition(
    "location",
    "current spatial relation",
    ValueType.RELATION,
    temporal_policy=TemporalPolicy(half_life_seconds=2 * HOUR),
)

cleanliness = PropertyDefinition(
    "cleanliness",
    "whether clothing is clean",
    ValueType.CATEGORICAL,
    temporal_policy=TemporalPolicy(
        half_life_seconds=7 * DAY,
        event_retention={"worn": 0.1},
    ),
)
```

Observed results from the deterministic example:

```text
cup on table, observed 5 hours ago     freshness=0.177
clean, observed 3 days ago             freshness=0.743
clean 3 days ago, then worn            freshness=0.074
```

Run it with:

```powershell
$env:PYTHONPATH = "src"
python examples/temporal_states.py
```

## Interpretation

Freshness is currently an interpretable evidence-persistence prior, not a
calibrated real-world probability. It lowers evidence weight and coverage while
leaving semantic similarity unchanged. A `worn` event weakens evidence that a
garment remains clean; it does not automatically assert `dirty`. An explicit
state-transition model should create a new dirty observation when the domain
supports that inference.

Half-lives and event-retention values must eventually be learned or calibrated
per property and environment. Stable properties should omit `TemporalPolicy`
rather than receiving arbitrary long half-lives.
