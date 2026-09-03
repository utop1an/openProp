from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .llm import LLMError


class OllamaClient:
    """Dependency-free client for Ollama's local structured chat API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        schema_dict = dict(schema)
        grounded_input = (
            f"{input_text}\n\nReturn only JSON matching this schema "
            f"({schema_name}):\n{json.dumps(schema_dict, ensure_ascii=False)}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": grounded_input},
            ],
            "format": schema_dict,
            "stream": False,
            "think": False,
            "options": {"temperature": self.temperature},
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise LLMError(f"Ollama HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise LLMError(f"cannot reach Ollama at {self.base_url}: {error.reason}") from error
        except (TimeoutError, json.JSONDecodeError) as error:
            raise LLMError(f"invalid or timed-out Ollama response: {error}") from error

        try:
            content = response_data["message"]["content"]
        except (KeyError, TypeError) as error:
            raise LLMError("Ollama response did not contain message.content") from error
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Ollama response contained no output text")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as error:
            raise LLMError("Ollama message content was not valid JSON") from error
        if not isinstance(result, Mapping):
            raise LLMError("Ollama structured response must be a JSON object")
        return result
