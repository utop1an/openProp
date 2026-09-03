# Irrelevant-attribute interference

`core-v1-interference` derives 30 stress cases from `core-v1`. Each query keeps
its original target expression and appends an explicitly unrelated background
record containing all observed properties of another entity in the scene.

Example:

```text
桌上的红色杯子。
无关背景记录（不要用于目标指代）:
entity=cabinet_red_cup, type=cup, color=red, material=plastic,
location=inside(object=cabinet), size=12
```

The benchmark stores two views:

- clean gold constraints used for property-selection metrics;
- distractor constraints used only by scoring ablations.

`gold-weighted` assigns each distractor weight `0.03`. `gold-equal` promotes the
exact same constraints to weight `1.0`. This isolates weighting while holding
the compared values constant.

## Deterministic result

```text
                         gold-weighted   gold-equal
Top-1 accuracy               1.000          0.033
Top-3 recall                 1.000          1.000
MRR                          1.000          0.506
```

Run the comparison:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate.py --dataset interference --strategy gold-weighted
python scripts/evaluate.py --dataset interference --strategy gold-equal
```

This is a deliberately strong synthetic stress test, not an estimate of natural
query frequency or real-world accuracy. A later benchmark should provide mild,
medium, and adversarial interference levels and report performance as a curve.
## Initial Ollama smoke result

On 2026-08-21, `gemma3:4b` produced the following five-case result:

```text
completed:           5 / 5
Top-1 accuracy:      0.400
Top-3 recall:        0.800
MRR:                 0.650
property F1:         0.893
mean latency:        3.577 seconds
```

The clean five-case smoke test had Top-1 `0.600` with the same property F1
`0.893`. This suggests the model often retained the right property names but
bound one or more values to the unrelated entity. A constraint-value and
relation-binding metric is therefore the next required diagnostic.
