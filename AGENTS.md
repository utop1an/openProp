# Agent Instructions

## Project Overview

OpenProp researches language-conditioned, property-guided entity grounding in an open world. Properties are heterogeneous and extensible; query parsing selects relevant properties, typed comparators score them, and a persistence model discounts stale state evidence.

## Working Rules

- Preserve typed values. Do not force numeric, relational, identity, and semantic values into one embedding representation.
- Treat `unknown` as missing evidence, not negative evidence.
- Keep timestamps, provenance, events, and observation histories outside the ordinary property dictionary.
- Keep semantic query parsing separate from deterministic entity scoring.
- Never expose temporal benchmark `current_truth` to the matcher; it is evaluation-only.
- Report synthetic results as mechanism validation, not real-world evidence.

## Common Commands

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python scripts/evaluate.py --dataset interference --strategy gold-weighted
python examples/train_persistence_pipeline.py
python scripts/evaluate_temporal_grounding.py --learned-model artifacts/contextual_persistence.pt
```

Install PyTorch support with `python -m pip install -e ".[ml]"`. Ollama evaluation is optional and should not be required by deterministic tests.

## Architecture Notes

- `PropertyRegistry` owns schema resolution and controlled growth.
- `ComparatorRegistry` owns value-family-specific comparison.
- `EntityMatcher` combines relevance, confidence, freshness, match, and coverage.
- `PersistenceModel` is replaceable; fixed and learned implementations share the same matching boundary.
- Observation-history training uses right-censored survival records and entity-grouped splits.

## Known Pitfalls

- Do not tune calibration or decision thresholds against the test split.
- Censoring before an evaluation horizon means the horizon label is unknown, not negative.
- Check candidate-order invariance when synthetic candidates otherwise tie.
- The current temporal grounding benchmark does not distinguish learned decay from fixed decay; both score perfectly.
- AI2-THOR metadata visibility does not guarantee an instance-detection box. For oracle-box VLM inputs, use visible objects with valid 2D anchors; record omitted visible objects as missing candidate evidence.
- AI2-THOR actions can advance unrelated movable objects through physics settling. Audit all derived changes and do not equate every changed entity with the intended intervention target.

## Memory Maintenance

Use `docs/dev-notes/README.md` as the progress index and `docs/decisions/` for durable architectural choices. Update focused experiment documents when verified metrics change.
