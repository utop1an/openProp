# LLM integration

The LLM performs semantic query planning, not entity scoring. It receives the
current property dictionary and a referring expression, then returns a strict
JSON object containing only relevant, weighted property constraints. OpenProp
validates that object and continues through deterministic comparators.

```text
"桌上的红色杯子"
        |
        v
LLMQueryParser + property dictionary
        |
        v
QueryFrame(type, color, location)
        |
        v
MentionBasedSelector -> EntityMatcher
```

## OpenAI setup

Install the optional SDK and provide credentials through the environment:

```powershell
python -m pip install -e ".[openai]"
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "your-structured-output-model"
$env:PYTHONPATH = "src"
python examples/llm_red_cup.py
```

The model is intentionally not hard-coded. `OPENAI_MODEL` must name a model
available to the caller that supports Structured Outputs.

## Safety and schema policy

- The query is serialized as data and the instruction explicitly rejects
  instructions embedded in it.
- Responses use strict JSON Schema and are validated again locally.
- API requests set `store=False`.
- Unknown properties are reported in `ParsedQuery.ignored_properties` and are
  not added by default.
- `allow_property_creation=True` enables dictionary growth explicitly. New
  definitions are still passed through registry resolution to reduce duplicates.
- Production systems should place human review, audit logs, or a stronger
  semantic resolver in front of permanent schema changes.

## Provider-neutral testing

`LLMQueryParser` depends on the small `JSONLLMClient` protocol. Tests and other
providers can implement `generate_json(...)` without importing the OpenAI SDK.
This also allows query parsing behavior to be evaluated from recorded responses
without making network requests.
