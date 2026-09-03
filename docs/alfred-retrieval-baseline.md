# Train-only retrieval and positive-evidence fusion on ALFRED

Date: 2026-08-26

## Research question

The earlier ALFRED audit compared evidence-constrained selection only with two
small generative parsers. This experiment asks a harder question: does the
method still help against a strong supervised retrieval baseline that can reuse
typed frames from ALFRED train descriptions?

This is a language-to-frame experiment. It contains no candidate entities,
visual observations, persistence estimates, matcher decisions, or temporal
grounding evidence.

## Leakage-safe protocol

The retriever indexes 11,974 supported human descriptions from ALFRED train and
their typed frames derived from train PDDL parameters. Validation data is never
used to fit the vocabulary, document frequencies, parameters, or tie breaks.
BM25 uses fixed conventional parameters `k1=1.2` and `b=0.75`; ties are resolved
by lexical train case ID. Train and validation have zero task-ID overlap.

The evaluation exposes an important language-overlap fact rather than hiding
it: 234 of 945 validation cases, representing 189 distinct descriptions, have
an exact text match in train. The report therefore separates exact matches from
novel descriptions.

The proposed deterministic hybrid follows two rules that do not use validation
labels:

1. If the complete query occurred in train, retain the top retrieved typed
   frame unchanged.
2. Otherwise, add missing properties and replace conflicting values only when
   the existing selector locates explicit positive evidence in the query.

Failure to recognize a state cue never removes a retrieved state property.
This preserves the project invariant that missing evidence is unknown, not
negative. Every changed value remains typed, including relational locations,
and every change is attributable to a query span.

`BM25 oracle@5` chooses the best of five retrieved frames using gold labels. It
is reported only as an evaluation-time retrieval coverage upper bound and is
not an executable method.

## Full validation results

| Split | Method | Property F1 | Value recall | Exact frame |
|---|---|---:|---:|---:|
| valid-seen | Evidence only | 0.788 | 0.652 | 0.345 |
| valid-seen | BM25 top-1 | 0.983 | 0.818 | 0.612 |
| valid-seen | BM25 + positive evidence | **0.989** | **0.880** | **0.710** |
| valid-seen | BM25 oracle@5, analysis only | 0.997 | 0.911 | 0.782 |
| valid-unseen | Evidence only | 0.822 | 0.697 | 0.404 |
| valid-unseen | BM25 top-1 | 0.979 | 0.802 | 0.611 |
| valid-unseen | BM25 + positive evidence | **0.986** | **0.896** | **0.736** |
| valid-unseen | BM25 oracle@5, analysis only | 0.995 | 0.894 | 0.764 |

The hybrid even exceeds the raw-frame oracle's value recall on valid-unseen
because explicit evidence can correct a value that is absent from all five
retrieved frames. It does not exceed the oracle in exact frames.

The gains do not depend only on duplicated text:

| Split and query subset | Cases | Method | Property F1 | Value recall | Exact frame |
|---|---:|---|---:|---:|---:|
| valid-seen novel | 359 | BM25 | 0.978 | 0.771 | 0.515 |
| valid-seen novel | 359 | BM25 + evidence | **0.986** | **0.855** | **0.649** |
| valid-unseen novel | 352 | BM25 | 0.973 | 0.751 | 0.514 |
| valid-unseen novel | 352 | BM25 + evidence | **0.981** | **0.873** | **0.676** |

On exact-train-query cases, the short circuit leaves BM25 unchanged: exact
frame accuracy is 0.883 on 128 valid-seen cases and 0.934 on 106 valid-unseen
cases. These are not perfect because identical language can map to different
canonical task parameters.

## Paired uncertainty

Confidence intervals use 20,000 paired bootstrap samples, resample whole task
IDs, and stratify by task type. This avoids treating multiple descriptions of
one trajectory as independent.

| Split | Metric, hybrid minus BM25 | Delta | 95% CI | Wins / ties / losses |
|---|---|---:|---:|---:|
| frozen confirmation | Property F1 | +0.014 | [0.000, 0.029] | 3 / 37 / 0 |
| frozen confirmation | Value recall | +0.146 | [0.083, 0.212] | 13 / 27 / 0 |
| frozen confirmation | Exact frame | +0.125 | [0.050, 0.225] | 5 / 35 / 0 |
| valid-seen | Property F1 | +0.006 | [0.003, 0.009] | 15 / 472 / 0 |
| valid-seen | Value recall | +0.062 | [0.046, 0.079] | 90 / 380 / 17 |
| valid-seen | Exact frame | +0.099 | [0.064, 0.134] | 61 / 413 / 13 |
| valid-unseen | Property F1 | +0.006 | [0.003, 0.010] | 16 / 441 / 1 |
| valid-unseen | Value recall | +0.094 | [0.062, 0.124] | 117 / 321 / 20 |
| valid-unseen | Exact frame | +0.124 | [0.071, 0.177] | 74 / 367 / 17 |

The F1 interval on the 40-case confirmation sample touches zero because only
three cases change. The value and exact-frame improvements are larger and their
intervals exclude zero. Both complete validation splits support all three
directions.

## Comparison with frozen generative parsers

On the same pre-frozen 40-case confirmation set, the retrieval hybrid reaches
property F1 0.984, value recall 0.896, and exact-frame accuracy 0.675. Paired
task-stratified comparisons against the existing evidence-fused generations are:

| LLM parser | F1 delta | Value-recall delta | Exact-frame delta |
|---|---:|---:|---:|
| gemma3:4b | +0.219 [0.105, 0.341] | +0.388 [0.296, 0.479] | +0.450 [0.300, 0.600] |
| llama3.2 | +0.241 [0.142, 0.350] | +0.317 [0.221, 0.412] | +0.400 [0.250, 0.550] |

This overturns any implication that the two local LLMs were strong baselines.
For this narrow ALFRED frame task, supervised sparse retrieval plus auditable
positive evidence is substantially stronger and cheaper.

## Limitations and claim boundary

- ALFRED language is templated and has substantial exact train-validation text
  overlap; the novel-query split reduces but does not eliminate template reuse.
- Retrieval uses train PDDL frame supervision and is not a zero-shot parser.
- The supported adapter excludes multi-entity and nested-container goals.
- State and destination evidence still uses fixed English lexical rules.
- The result validates a language-to-typed-frame mechanism, not open-world,
  visual, real-world, or temporal grounding.
- The benchmark still cannot distinguish learned from fixed temporal decay.

The defensible contribution is an evidence boundary: retrieval supplies a
high-recall typed prior, while explicit query spans safely correct selected
values without converting missing cues into negative evidence. It is not a
claim of a novel retrieval algorithm.

## Reproducibility

    python scripts/evaluate_alfred_retrieval_baseline.py --root artifacts/external/alfred/json_2.1.0
    python scripts/analyze_alfred_retrieval_comparison.py
    python scripts/analyze_alfred_retrieval_vs_llm.py

Primary artifacts:

- `artifacts/alfred_retrieval_baseline.json`
- `artifacts/alfred_retrieval_comparison.json`
- `artifacts/alfred_retrieval_vs_llm.json`

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Positive evidence improves a strong train-only retrieval baseline. | On both full splits, task-clustered paired CIs exclude zero for F1, value recall, and exact frames. | Supported for the scoped ALFRED language-to-frame task. |
| The gain generalizes beyond exact train-query repeats. | On 711 novel descriptions, value recall rises by 0.084 and 0.122 across the two splits, while exact frames rise by 0.134 and 0.162. | Supported within ALFRED's templated language distribution. |
| Small local generative parsers are competitive baselines for this task. | The retrieval hybrid exceeds both frozen parsers by 0.219-0.241 F1 and 0.317-0.388 value recall on the same cases. | Contradicted. |
| Retrieval fusion establishes temporal or visual grounding. | No observations, candidates, persistence outputs, or matcher decisions are present. | Unsupported and must not be claimed. |
