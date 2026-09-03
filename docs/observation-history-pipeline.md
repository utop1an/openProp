# Observation-history training pipeline

This experiment turns timestamped state episodes into a reproducible survival-learning workflow:

```text
observation history JSONL
        -> censored/event training episodes
        -> entity-grouped train/validation/test split
        -> contextual hazard model
        -> validation calibration
        -> held-out survival metrics
        -> saved, reloadable model
```

## JSONL contract

Each line describes one observed state episode. For example:

```json
{"record_id":"r-001","entity_id":"cup-17","property_name":"location","subject_type":"cup","state_predicate":"on","context_object":"table","scene":"kitchen","observed_at":"2026-08-20T00:00:00+00:00","last_confirmed_at":"2026-08-20T01:00:00+00:00","followup_at":"2026-08-20T02:00:00+00:00","state_changed":true,"source":"camera-1","observation_confidence":0.96}
```

- `observed_at` is when the state was established.
- `followup_at` is when a change was detected or observation ended.
- `state_changed=true` is a detected transition event.
- `state_changed=false` is right censoring: the state had not been seen to change by `followup_at`.
- Optional `last_confirmed_at` says the state was still known to hold at that
  inspection. A later detected change is interval censored between this time and
  `followup_at`; omitting it preserves exact-event compatibility.
- `entity_id` is the split group, preventing episodes from the same entity appearing in multiple partitions.
- `source` and `observation_confidence` remain available for later observation-bias and uncertainty models; the current hazard model does not consume them.

See `examples/observation_history.sample.jsonl` for directly editable records.

## Run the complete experiment

```powershell
python -m pip install -e ".[ml]"
$env:PYTHONPATH = "src"
python examples/train_persistence_pipeline.py
```

The script creates:

- `artifacts/observation_history.jsonl`: the complete generated history;
- `artifacts/contextual_persistence.pt`: the calibrated, reloadable model.

The split is deterministic, seeded, and grouped by entity. The default ratios are 70% training, 15% validation, and 15% testing.

## Verified result

The current local run used 600 episodes:

```text
split: train=420 validation=90 test=90
training loss: 2.4846 -> 1.2736
validation hazard scale: 0.7695
validation NLL: 1.2624 -> 1.2424 after calibration
test NLL: 1.3757
```

Held-out test calibration:

| Horizon | Evaluable episodes | Brier score | ECE |
|---:|---:|---:|---:|
| 1 hour | 88 | 0.1637 | 0.0640 |
| 5 hours | 72 | 0.1186 | 0.1265 |
| 12 hours | 60 | 0.0894 | 0.1539 |

An episode censored before a requested horizon is excluded from that horizon's Brier score and ECE because its true state at the horizon is unknown. It still contributes correctly to survival likelihood.

## Calibration finding

The validation-derived global hazard multiplier improved validation negative log-likelihood and 1-hour ECE. It did not improve every longer horizon: 5-hour and 12-hour validation ECE became worse. This is evidence that a single multiplier is too restrictive for full time-dependent calibration, not a reason to tune against the test set.

Interval-censored training now prevents inspection time from being treated as an
exact transition time. The next model iteration should address informative
inspection policies, missed detections, and source reliability, then compare
piecewise hazards or horizon-specific calibration on validation data.

