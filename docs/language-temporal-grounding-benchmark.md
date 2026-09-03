# Language-to-temporal-grounding benchmark

Date: 2026-08-26

## Research question

This experiment measures whether a language model can convert a bilingual
referring expression into typed constraints that preserve the final temporal
entity decision. Unlike parser-only evaluation, every request failure,
validation failure, wrong value, relation-argument error, and grounding error
remains in the primary Top-1 denominator.

Current truth remains evaluation-only. The language model receives the query
and property dictionary, while the matcher receives only parsed constraints,
entity observations, provenance, timestamps, and events.

## Fair strict/tolerant protocol

Each distinct query is sent to the local model once. The raw structured
response is then replayed through:

- `gold`: gold query frame with fixed temporal persistence;
- `llm-strict`: reject the complete frame when any selected constraint is
  semantically invalid;
- `llm-tolerant`: discard invalid individual constraints but retain diagnostics.
- `llm-schema-repaired`: apply typed schema repair, then the same tolerant
  validator.

Strict and tolerant parsing therefore see identical model output. Conditional
Top-1 over successfully parsed cases is reported only as a diagnostic; Top-1
over all cases is the primary result.

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_language_temporal_grounding.py --model gemma3:4b
python scripts/evaluate_language_temporal_grounding.py --responses-input artifacts/language_temporal_grounding_results.json --output artifacts/language_temporal_grounding_repaired_results.json
```

The command stores raw responses and per-case results in
`artifacts/language_temporal_grounding_results.json`.

## Initial local-model result

The run contains 40 temporal grounding cases but only six distinct bilingual
query templates, so it is an execution-path audit rather than a publication
benchmark.

| Strategy | Parse success | Top-1 all cases | Top-1 parsed only | Property F1 | Value recall | Relevance MAE |
|---|---:|---:|---:|---:|---:|---:|
| Gold | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| gemma3:4b strict | 0.750 | 0.375 | 0.500 | 0.690 | 0.458 | 0.590 |
| gemma3:4b tolerant | 1.000 | 0.375 | 0.375 | 0.940 | 0.625 | 0.457 |
| gemma3:4b schema-repaired | 1.000 | 0.750 | 0.750 | 0.940 | 0.792 | 0.457 |

Tolerance salvages all ten strict validation failures, but it does not improve
Top-1. This separates executable structured output from correct grounding:
discarding malformed constraints cannot repair validly encoded but semantically
wrong constraints.

## Schema-grounded repair

The repair uses only the registered value type, allowed relation predicates,
argument roles, and the frozen raw response. It never inspects candidates,
scores, target labels, current truth, or test outcomes. When a relation output
places an allowed predicate in the generic scalar field, the rule moves that
token to `predicate` and moves the old predicate to the sole registered
argument role. When the predicate is already allowed, a non-predicate scalar is
moved to that argument role. Every change is logged per case.

On the frozen responses, repair fires in 20/40 cases and doubles all-case Top-1
from 0.375 to 0.750. Relation-value correction raises constraint value recall
from 0.625 to 0.792 without changing property F1 or relevance MAE. Remaining
errors are informative: Chinese cleanliness queries still replace state with
material, and five missing-observation cases remain wrong because the model's
color relevance outweighs location relevance. The method therefore repairs a
specific typed inconsistency rather than using test feedback to rewrite all
mistakes.

## Unseen-paraphrase cross-model validation

The repair rule was frozen before generating a disjoint evaluation set of 30
manually written bilingual queries: ten relation, ten cleanliness, and ten
static type/color/material paraphrases. The 40 underlying grounding cases,
gold constraints, observations, events, targets, and hidden truth are unchanged.
None of the new query strings appears in the six-template development audit.

```powershell
python scripts/evaluate_language_temporal_grounding.py --query-set paraphrase --model gemma3:4b --output artifacts/language_temporal_paraphrase_gemma3_4b.json
python scripts/evaluate_language_temporal_grounding.py --query-set paraphrase --model llama3.2:latest --output artifacts/language_temporal_paraphrase_llama3_2.json
python scripts/analyze_language_schema_repair.py artifacts/language_temporal_paraphrase_gemma3_4b.json artifacts/language_temporal_paraphrase_llama3_2.json
```

| Model and strategy | Parse success | Top-1 | Property F1 | Value recall | Repair cases |
|---|---:|---:|---:|---:|---:|
| gemma3:4b strict | 1.000 | 0.675 | 0.876 | 0.642 | 0 |
| gemma3:4b tolerant | 1.000 | 0.675 | 0.876 | 0.642 | 0 |
| gemma3:4b repaired | 1.000 | 0.700 | 0.876 | 0.658 | 4 |
| llama3.2 strict | 0.500 | 0.400 | 0.298 | 0.308 | 0 |
| llama3.2 tolerant | 1.000 | 0.725 | 0.695 | 0.558 | 0 |
| llama3.2 repaired | 1.000 | 0.725 | 0.695 | 0.558 | 0 |

For gemma3:4b, repair improves one Top-1 decision and a second rank without any
rank regression. For llama3.2, the relation inconsistency never occurs, so the
repair correctly remains inactive and produces identical results. Across both
models, the paired Top-1 change is +0.013 over 80 cases. A 10,000-sample
model-query-cluster bootstrap gives a 95% interval of [0.000, 0.039], while the
two-sided paired sign test is p=1.000 because only one Top-1 pair is discordant.
This is evidence of guarded behavior and limited transfer, not statistically
established general improvement.

## Failure anatomy

The saved responses expose three recurring error classes:

1. Relation fields are permuted. For `the red cup on the table`, the model emits
   predicate `table`, scalar `on`, and relation argument `cup`, causing every
   stale-location case to select the stale candidate.
2. Mentioned state is replaced by a plausible unmentioned attribute. For the
   Chinese clean-shirt query, the model selects `material=cotton` instead of
   `cleanliness=clean`.
3. Type and material values are swapped in some red ceramic cup responses.

These failures show why a schema repair must remain narrower than general
semantic correction. Missing properties and plausible but wrong values require
additional query-grounded evidence; candidate- or label-guided rewriting would
violate the evaluation boundary.

## Limits and next gate

The environment and targets remain synthetic, the paraphrases are manually
written, and each local model has only one deterministic response per query.
The cross-model result does not establish real-world grounding or a general
accuracy gain. Before main-paper use, add independent language authors or a
source dataset, more model families and repeated generations, and a semi-real
longitudinal environment with independently audited current truth.
