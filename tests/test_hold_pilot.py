# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded execution and failure evidence; no live calls or quality claims."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import hold_dialogue_pilot as pilot
import recursive_conclusion_lab as lab


class InvalidFirstWorkpad(lab.DummyAdapter):
    def __init__(self):
        super().__init__("dummy-v1")
        self.first = True

    def generate(self, **kwargs):
        result = super().generate(**kwargs)
        if kwargs["system"] == lab.INQUIRY_PROBE_SYSTEM and self.first:
            self.first = False
            result.text = "REJECTED_WORKPAD_CANARY"
        return result


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / "run"

    def run_pilot(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return pilot.run_pilot(self.output)

    def test_offline_complete_receipts_and_no_overwrite(self):
        with mock.patch.object(lab, "post_json", side_effect=AssertionError("No network")):
            result = self.run_pilot()
        self.assertEqual((result["generations"], result["valid_workpads"], result["complete_pairs"], result["complete_primary_pairs"]), (48, 12, 6, 2))
        manifest = pilot.read_json(self.output / "run_manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["automatic_retries"], 0)
        before = (self.output / "run_manifest.json").read_bytes()
        with mock.patch.object(lab, "build_adapter") as build:
            with self.assertRaises(ValueError):
                self.run_pilot()
        build.assert_not_called()
        self.assertEqual((self.output / "run_manifest.json").read_bytes(), before)

    def test_invalid_workpads_preserve_full_review_and_exclude_primary_pairs(self):
        with mock.patch.object(lab, "build_adapter", side_effect=lambda *a, **k: InvalidFirstWorkpad()):
            result = self.run_pilot()
        self.assertEqual((result["valid_workpads"], result["all_pairs"], result["complete_pairs"], result["complete_primary_pairs"]), (10, 6, 2, 0))
        for folder, count in [("review", 6), ("complete_review", 2)]:
            self.assertEqual(pilot.read_json(self.output / folder / "manifest.json")["item_count"], count)
            self.assertNotIn("REJECTED_WORKPAD_CANARY", (self.output / folder / "booklet.md").read_text())
        for path in (self.output / "compare").glob("arm_hold_workpad*.jsonl"):
            events = [json.loads(s) for s in path.read_text().splitlines()]
            first = next(e["payload"] for e in events if e["event_type"] == "assistant_reply")
            self.assertIsNone(first["inquiry_state"])

    def test_truncated_reply_excluded_even_when_provider_returns(self):
        generate = lab.DummyAdapter.generate
        count = 0

        def truncate(adapter, **kwargs):
            nonlocal count
            count += 1
            response = generate(adapter, **kwargs)
            if count == 1:
                response.finish_reason = "length"
            return response

        with mock.patch.object(lab.DummyAdapter, "generate", truncate):
            result = self.run_pilot()
        self.assertEqual(result["complete_pairs"], 4)
        self.assertEqual(result["complete_primary_pairs"], 2)
        self.assertEqual(result["dialogues"][0]["exclusions"][0]["finish_reason"], "length")

    def test_all_invalid_outputs_keep_full_review_without_empty_subset(self):
        generate = lab.DummyAdapter.generate

        def unknown(adapter, **kwargs):
            response = generate(adapter, **kwargs)
            response.finish_reason = None
            return response

        with mock.patch.object(lab.DummyAdapter, "generate", unknown):
            result = self.run_pilot()
        self.assertEqual(result["complete_pairs"], 0)
        self.assertTrue((self.output / "review" / "READY.json").exists())
        self.assertFalse((self.output / "complete_review").exists())

    def test_later_invalid_arm_fails_without_artifacts_or_adapter(self):
        original = pilot.read_json

        def read(path):
            value = original(path)
            if path.name == "compare_matrix_hold_dialogue.json":
                value["arms"][2]["args"].update(observer_provider="dummy")
            return value

        with mock.patch.object(pilot, "read_json", read), mock.patch.object(lab, "build_adapter") as build:
            with self.assertRaises(ValueError):
                self.run_pilot()
        build.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_missing_live_credential_fails_before_artifacts(self):
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(lab, "build_adapter") as build:
            with self.assertRaises(ValueError):
                pilot.run_pilot(self.output, "mistral", "mistral-small-2603")
        build.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_unknown_usage_is_not_reported_as_zero(self):
        self.assertEqual(pilot.token_total([{}, {"prompt_tokens": 4}], ("prompt_tokens",)), (None, 1))
        self.assertEqual(pilot.token_total([{"promptTokenCount": 0}, {"promptTokenCount": 7}], ("prompt_tokens", "promptTokenCount")), (7, 0))

    def test_setup_write_failure_records_incomplete_without_provider_call(self):
        write = pilot.write_json

        def fail(path, value):
            if path.name == "human_eval.json":
                raise OSError("fixture disk failure")
            write(path, value)

        with mock.patch.object(pilot, "write_json", fail), mock.patch.object(lab, "build_adapter") as build:
            with self.assertRaises(OSError):
                self.run_pilot()
        build.assert_not_called()
        manifest = pilot.read_json(self.output / "run_manifest.json")
        self.assertEqual((manifest["status"], manifest["actual_generation_calls"]), ("incomplete", 0))

    def test_failed_call_has_write_ahead_receipt_and_no_secret_or_retry(self):
        def fail(*args, **kwargs):
            records = pilot.read_json(self.output / "generation_receipts.json")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "started")
            raise RuntimeError("SECRET_EXCEPTION_CANARY")

        with mock.patch.object(lab.DummyAdapter, "generate", side_effect=fail) as generate:
            with self.assertRaises(RuntimeError):
                self.run_pilot()
        self.assertEqual(generate.call_count, 1)
        manifest = pilot.read_json(self.output / "run_manifest.json")
        self.assertEqual(manifest["status"], "incomplete")
        self.assertEqual(manifest["actual_generation_calls"], 1)
        receipt = pilot.read_json(self.output / "generation_receipts.json")[0]
        self.assertEqual(receipt["status"], "failed_or_unknown")
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertNotIn("SECRET_EXCEPTION_CANARY", path.read_text())

    def test_call_limit_stops_before_49th_generation(self):
        def excessive(session, text):
            for _ in range(49):
                session.adapter.generate(system="fixture", messages=[], config=session.config.reply_config)

        with mock.patch.object(lab.RecursiveConclusionSession, "user_turn", excessive), mock.patch.object(lab.DummyAdapter, "generate", wraps=lab.DummyAdapter("dummy-v1").generate) as generate:
            with self.assertRaisesRegex(RuntimeError, "budget"):
                self.run_pilot()
        self.assertEqual(generate.call_count, 48)
        self.assertEqual(pilot.read_json(self.output / "run_manifest.json")["actual_generation_calls"], 48)


if __name__ == "__main__":
    unittest.main()
