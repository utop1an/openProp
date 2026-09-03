# Temporal entity-grounding benchmark

This benchmark evaluates the decision OpenProp ultimately needs to make: selecting the correct current entity from stale, incomplete, and distracting observations.

It is deliberately separate from the survival-only experiment. A persistence model can have good likelihood while still making the wrong entity decision; this benchmark measures Top-1 accuracy and rank directly.

## Case contract

Every JSONL case contains two distinct layers:

- `entities`: only the observations and events available to the matcher;
- `current_truth`: held-out current state used to construct and audit the target label.

The query, gold property constraints, query time (`as_of`), target entity, and scenario tags are also recorded. Current truth is never inserted into an entity's property dictionary during matching.

The 40-case `temporal-grounding-v1` dataset contains ten bilingual repetitions of four scenario families:

| Scenario | Purpose |
|---|---|
| `stale-location` | A previously correct spatial relation was later invalidated. |
| `missing-observation` | The correct entity has an unknown color, while a stale distractor looks complete. |
| `event-invalidated` | An old clean-clothing observation is invalidated by a later wearing event. |
| `static-control` | Stable type, color, and material evidence must not be needlessly decayed. |

All cases contain irrelevant observed attributes. Temporal challenge cases include explicit timestamps, and applicable cases include intermediate events. Candidate order is not used to create the performance gap: stale distractors have a strict no-time score advantage over slightly uncertain current observations.

## Run

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_temporal_grounding.py --learned-model artifacts/contextual_persistence.pt
```

The command writes:

- `artifacts/temporal_grounding_benchmark.jsonl`;
- `artifacts/temporal_grounding_results.json`.

## Verified result

| Strategy | Top-1 | Top-3 | MRR |
|---|---:|---:|---:|
| No temporal decay | 0.250 | 1.000 | 0.625 |
| Fixed policy decay | 1.000 | 1.000 | 1.000 |
| Learned contextual decay | 1.000 | 1.000 | 1.000 |

Both temporal methods solve all three temporal challenge families and retain perfect accuracy on the static control family.

## Interpretation and limits

This result establishes that temporal evidence can change the final grounding decision under controlled missingness and distractors. It does **not** establish that learned decay is better than a fixed policy: both score 1.000 on this intentionally diagnostic dataset.

The dataset is synthetic, uses gold query frames, and currently treats the target label as given by the generator's hidden state. It therefore supports mechanism validation, not a real-world performance claim. The next experimental expansion should introduce noisy natural-language parsing, richer trajectories, ambiguous or unobserved events, held-out context combinations, and real annotated histories. Fixed and learned temporal methods must then be compared without tuning on the test set.

