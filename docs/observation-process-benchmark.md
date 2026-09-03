# Observation-process benchmark

This benchmark isolates a failure mode in learned state persistence: a transition
is often discovered at an inspection time, not observed at its true occurrence
time. Treating detection as an exact event makes sparsely inspected states look
artificially persistent.

## Data contract

OpenProp now distinguishes three survival-record types:

- an exact event records a known transition time;
- a right-censored record says the state remained confirmed through follow-up;
- an interval-censored event says the transition occurred after
  `last_confirmed_at` and no later than `followup_at`.

`ObservationHistoryRecord.last_confirmed_at` is observation-history metadata.
It is converted to `PersistenceTrainingExample.interval_start_seconds`; it is
not stored in the ordinary property dictionary or exposed as current truth.
Exponential statistical and neural models use the probability mass between the
two inspection boundaries. Horizon metrics omit an interval that straddles the
evaluation horizon because its binary state at that horizon is unknown.

## Controlled protocol

Both synthetic groups have the same latent exponential transition hazard,
`0.25/hour`, and a 12-hour administrative follow-up. The only difference is
inspection frequency: every 0.5 hours or every 4 hours.

For each of five seeds, the experiment generates 600 training episodes per
inspection schedule and 400 independent exact-time test episodes per schedule.
It compares:

- `naive-detection-time`: the first positive inspection is treated as the exact
  transition time;
- `interval-aware`: the transition is learned from the last-negative to
  first-positive inspection interval.

Both use the same per-context exponential estimator. The inspection schedule is
included as a context label only to measure how much false schedule dependence
each protocol learns.

## Verified five-seed result

Values are population mean and standard deviation across seeds 101, 211, 307,
401, and 503.

| Training semantics | Hazard MAE vs. 0.25 | False schedule gap | Exact-test NLL |
|---|---:|---:|---:|
| Naive detection time | 0.0559 +/- 0.0044 | 0.0745 +/- 0.0058 | 2.3351 +/- 0.0188 |
| Interval-aware | 0.0058 +/- 0.0044 | 0.0038 +/- 0.0036 | 2.2914 +/- 0.0270 |

The interval-aware estimator reduces mean hazard error by about 90% and removes
most of the spurious difference between frequent and sparse inspection. Its
lower likelihood loss on independent exact-time samples shows that the gain is
not only agreement with the generator parameter.

## Run

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_observation_process.py
```

The command writes `artifacts/observation_process_results.json`, including all
per-seed hazards and aggregate statistics.

## Supported interpretation and limits

This experiment supports a narrow mechanism claim: when transition times are
known only between inspections, respecting those intervals substantially
reduces observation-frequency bias under synthetic exponential dynamics.

It is not real-world evidence. This first benchmark assumes regular inspection,
non-informative administrative censoring, exact inspection outcomes, one event
per episode, and a correctly specified exponential model. The follow-up
[informative-observation benchmark](informative-observation-benchmark.md)
relaxes inspection independence and perfect sensitivity with a joint likelihood.
Conflicting sources, estimated rather than logged observation parameters, and
semi-real validation remain unresolved.
