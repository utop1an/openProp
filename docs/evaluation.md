# Evaluation

`core-v1` is the first OpenProp benchmark. It contains 30 bilingual referring
expressions across six scenes and covers semantic, categorical, numeric,
relation, material, owner, and multi-property constraints.

## Strategies

- `gold-weighted` uses annotated properties and relevance weights. It is an
  upper-bound and matcher/data integrity check.
- `gold-equal` uses the same annotated properties with all relevance values set
  to one. It isolates the effect of weighting; it is not yet an all-property or
  whole-entity embedding baseline.
- `llm-weighted` asks a local Ollama model to parse and weight properties before
  running the same matcher.

## Metrics

- Top-1 entity accuracy and Top-3 recall;
- mean reciprocal rank (MRR);
- macro property-selection precision, recall, and F1;
- target evidence coverage;
- parsing/matching failures and mean end-to-end latency.

LLM failures remain in the metric denominator. This prevents malformed or empty
outputs from disappearing from reported accuracy.

## Run

```powershell
$env:PYTHONPATH = "src"

python scripts/evaluate.py --strategy gold-weighted
python scripts/evaluate.py --strategy gold-equal

$env:OLLAMA_MODEL = "gemma3:4b"
python scripts/evaluate.py --strategy llm-weighted --limit 5
```

Remove `--limit` for all 30 local-model calls. `--ollama-host` overrides the
default `http://127.0.0.1:11434` endpoint.

## Current boundary

The first benchmark is deliberately controlled and all gold strategies should
score 1.0. It validates plumbing but is not evidence that relevance weighting
improves grounding. The next dataset revision should add distractors where
irrelevant, noisy, or missing properties make equal weighting worse, followed
by a whole-entity embedding baseline and confidence calibration plots.
## Initial smoke result

A five-case `gemma3:4b` run on 2026-08-21 completed all cases after enabling
per-constraint tolerance in the evaluator:

```text
top-1 accuracy:      0.600
top-3 recall:        1.000
MRR:                 0.800
property F1:         0.893
mean latency:        3.012 seconds
```

Strict parsing remains the default for application use. Benchmark LLM parsing
skips individual semantically invalid constraints and retains them as validation
evidence, so one malformed unused property does not invalidate the whole query.
This smoke result is model- and run-dependent, not a final benchmark claim.
