# Local Ollama testing

OpenProp includes a dependency-free `OllamaClient` for the local
`POST /api/chat` endpoint. It sends `QUERY_FRAME_SCHEMA` in Ollama's `format`
field, disables streaming, and uses temperature zero for repeatable extraction.
The same schema is also included in the prompt to improve small-model adherence.

## Run

Make sure Ollama is running and the selected model is installed:

```powershell
ollama list
$env:PYTHONPATH = "src"
$env:OLLAMA_MODEL = "gemma3:4b"
python examples/ollama_red_cup.py
```

Optional server override:

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
```

The example defaults to `gemma3:4b`. On the development machine it produced:

```text
type: cup                    relevance=0.70
color: red                  relevance=0.80
location: on(object=table)  relevance=0.60

cup_red   score=1.000
cup_blue  score=0.619
```

`llama3.2:latest` followed the JSON schema but assigned zero relevance to every
property in the same test. This is a model-quality failure that the local
validator correctly rejects. Local model comparisons should therefore measure
both schema validity and semantic constraint accuracy.

## Relation metadata

Relation properties can provide expected argument roles to the LLM:

```python
PropertyDefinition(
    "location",
    "spatial relation between an entity and another entity",
    ValueType.RELATION,
    metadata={"argument_roles": ["object"]},
)
```

This makes `on(object=table)` consistent with stored entity relations and avoids
role drift such as `entity=table`.
