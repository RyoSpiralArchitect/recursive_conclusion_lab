"""Provider HTTP failures must not echo prompts or response bodies."""

import sys
import traceback
from types import SimpleNamespace
import unittest
from unittest import mock

from recursive_conclusion_lab import post_json


class HttpErrorRedactionTests(unittest.TestCase):
    def call_with_response(self, response):
        requests = SimpleNamespace(post=lambda *args, **kwargs: response)
        with mock.patch.dict(sys.modules, {"requests": requests}):
            return post_json(
                url="https://provider.example/api?key=sk-url-secret",
                headers={"Authorization": "Bearer sk-header-secret"},
                payload={"prompt": "sk-prompt-secret"},
                timeout_seconds=10,
            )

    def test_http_error_omits_request_and_response_content(self):
        response = SimpleNamespace(status_code=401, text="sk-response-secret")
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
            self.call_with_response(response)
        error = str(caught.exception)
        for secret in ("sk-url-secret", "sk-header-secret", "sk-prompt-secret", "sk-response-secret"):
            self.assertNotIn(secret, error)

    def test_non_json_error_omits_response_content(self):
        def fail_json():
            raise ValueError("sk-decoder-secret")

        response = SimpleNamespace(status_code=200, text="sk-response-secret", json=fail_json)
        with self.assertRaisesRegex(RuntimeError, "Non-JSON response") as caught:
            self.call_with_response(response)
        error = str(caught.exception)
        self.assertNotIn("sk-response-secret", error)
        self.assertNotIn("sk-decoder-secret", error)
        formatted = "".join(traceback.format_exception(caught.exception))
        self.assertNotIn("sk-decoder-secret", formatted)


if __name__ == "__main__":
    unittest.main()
