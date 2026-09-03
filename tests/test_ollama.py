import io
import json
import unittest
from unittest.mock import patch

from openprop.llm import LLMError
from openprop.ollama import OllamaClient


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class OllamaClientTests(unittest.TestCase):
    @patch("openprop.ollama.urlopen")
    def test_sends_schema_non_streaming_and_temperature_zero(self, mock_urlopen):
        output = {"constraints": []}
        mock_urlopen.return_value = FakeHTTPResponse(
            {"message": {"role": "assistant", "content": json.dumps(output)}}
        )
        client = OllamaClient(model="local-model")
        result = client.generate_json(
            instructions="instructions",
            input_text="input",
            schema_name="query_frame",
            schema={"type": "object"},
        )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(result, output)
        self.assertEqual(payload["format"], {"type": "object"})
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["temperature"], 0.0)
        self.assertIn("query_frame", payload["messages"][1]["content"])

    @patch("openprop.ollama.urlopen")
    def test_rejects_non_json_message_content(self, mock_urlopen):
        mock_urlopen.return_value = FakeHTTPResponse(
            {"message": {"role": "assistant", "content": "not json"}}
        )
        with self.assertRaisesRegex(LLMError, "not valid JSON"):
            OllamaClient(model="local-model").generate_json(
                instructions="instructions",
                input_text="input",
                schema_name="query_frame",
                schema={"type": "object"},
            )

    def test_rejects_invalid_base_url(self):
        with self.assertRaises(ValueError):
            OllamaClient(model="local-model", base_url="localhost:11434")


if __name__ == "__main__":
    unittest.main()
