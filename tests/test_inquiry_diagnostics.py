# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rejected workpad diagnostics preserve the existing fail-closed behavior."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from inquiry_state import InquiryState, InquiryValidationError
import recursive_conclusion_lab as lab

VALID = {"focus": "Explore", "hypotheses": [], "open_questions": [], "next_step": "Draft an option", "revision_note": ""}


class DiagnosticAdapter(lab.DummyAdapter):
    def __init__(self, text):
        super().__init__("dummy-v1")
        self.text = text
        self.seen_reply_messages = []

    def generate(self, **kwargs):
        result = super().generate(**kwargs)
        if kwargs["system"] == lab.INQUIRY_PROBE_SYSTEM:
            result.text = self.text
        else:
            self.seen_reply_messages.append((kwargs["system"], kwargs["messages"]))
        return result


class InquiryDiagnosticsTests(unittest.TestCase):
    def test_stable_field_codes_without_value_disclosure(self):
        cases = [
            ({**VALID, "extra": "SECRET"}, "fields", "state"),
            ({**VALID, "focus": 42}, "text_type", "focus"),
            ({**VALID, "focus": "SECRET" * 100}, "text_length", "focus"),
            ({**VALID, "next_step": " "}, "blank_text", "next_step"),
            ({**VALID, "hypotheses": "SECRET"}, "list_type", "hypotheses"),
            ({**VALID, "hypotheses": ["SECRET"] * 4}, "list_length", "hypotheses"),
            ({**VALID, "hypotheses": [None]}, "text_type", "hypotheses[0]"),
        ]
        for value, code, field in cases:
            with self.subTest(code=code):
                with self.assertRaises(InquiryValidationError) as error:
                    InquiryState.from_dict(value)
                self.assertEqual((error.exception.code, error.exception.field), (code, field))
                self.assertNotIn("SECRET", str(error.exception))

    def test_diagnostics_are_bounded_and_never_injected(self):
        for text, stage, code in [("NOT_JSON_CANARY" * 500, "json_extraction", "missing_or_null_json"),
                                  (json.dumps({**VALID, "hypotheses": ["x"] * 4}), "schema", "list_length"),
                                  ("null", "json_extraction", "missing_or_null_json")]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "events.jsonl"
                adapter = DiagnosticAdapter(text)
                session = lab.RecursiveConclusionSession(adapter=adapter, log_path=path,
                    config=lab.ExperimentConfig(conclusion_mode=lab.ConclusionMode.HOLD, conclusion_every=1, memory_every=0))
                session.inquiry_state = InquiryState.from_dict(VALID)
                session.inquiry_state_turn = 0
                result = session.user_turn("Change the premise.")
                payload = next(json.loads(s)["payload"] for s in path.read_text().splitlines() if json.loads(s)["event_type"] == "inquiry_update")
                diagnostic = payload["validation"]
                self.assertEqual((diagnostic["stage"], diagnostic["code"]), (stage, code))
                self.assertEqual(diagnostic["response_text"], text[:4096])
                self.assertEqual(diagnostic["response_text_sha256"], hashlib.sha256(text.encode()).hexdigest())
                self.assertEqual(diagnostic["response_text_truncated"], len(text) > 4096)
                self.assertIsNone(result["inquiry_state"])
                self.assertIsNone(result["inquiry_state_turn"])
                self.assertNotIn("NOT_JSON_CANARY", str(adapter.seen_reply_messages))


if __name__ == "__main__":
    unittest.main()
