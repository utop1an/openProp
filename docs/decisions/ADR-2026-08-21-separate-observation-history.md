# ADR 2026-08-21: Separate observation history from the property schema

## Status

Accepted

## Context

Entity properties include both stable descriptions, such as type and material, and state observations, such as a cup being on a table or clothing being clean. State reliability changes with elapsed time, surrounding context, observation source, and intervening events. Encoding decay itself as another property would mix observed facts with inference about whether those facts still hold.

## Decision

Keep property values and observation history as separate layers.

- The property dictionary defines semantic meaning, value type, comparator, and optional temporal policy.
- An observation records value, state, confidence, provenance, and timestamp.
- Entity events remain timestamped history rather than ordinary matching properties.
- A replaceable `PersistenceModel` converts observation age, context, and events into freshness at query time.
- Matching multiplies property relevance by effective observation confidence while preserving the original value and comparison result.
- Benchmark `current_truth` is held out from matcher inputs and used only for labeling and auditing.

## Consequences

Different persistence models can be compared without changing the property schema or matcher contract. Historical evidence remains auditable, missing values remain distinct from mismatches, and stable properties need no artificial decay record.

The system must maintain reliable timestamps and event provenance. Learned persistence also requires censored-data handling and protection against observation-policy bias. Current truth must never be copied into observable entity properties during evaluation.

