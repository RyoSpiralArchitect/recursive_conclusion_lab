# SPDX-License-Identifier: AGPL-3.0-or-later
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hold_action_comparison as experiment
import hold_dialogue_pilot as pilot
import recursive_conclusion_lab as lab


class ActionComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "run"

    def run_experiment(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return experiment.run_comparison(self.out)

    def test_balanced_design_prompt_isolation_receipts_and_no_overwrite(self):
        with patch.object(lab, "post_json", side_effect=AssertionError("No network")):
            metrics = self.run_experiment()
        self.assertEqual(
            (
                metrics["generations"],
                metrics["valid_workpads"],
                metrics["complete_pairs"],
            ),
            (144, 72, 6),
        )
        calls = pilot.read_json(self.out / "generation_receipts.json")
        for arm in experiment.ARMS:
            selected = [
                c for c in calls if c["arm"] == arm and c["kind"] == "inquiry_update"
            ]
            self.assertEqual(len(selected), 36)
            self.assertEqual(
                {c["prompt_sha256"] for c in selected},
                {hashlib.sha256(experiment.PROMPTS[arm].encode()).hexdigest()},
            )
        reply_hashes = {
            c["prompt_sha256"] for c in calls if c["kind"] == "assistant_reply"
        }
        self.assertEqual(len(reply_hashes), 1)
        self.assertEqual(lab.INQUIRY_PROBE_SYSTEM, experiment.CONTROL)
        order = pilot.read_json(self.out / "run_manifest.json")["execution_order"]
        self.assertEqual(
            [order[i]["arm"] for i in range(0, 12, 2)],
            ["current", "action", "current", "action", "current", "action"],
        )
        public = (self.out / "review/eval_items.jsonl").read_text()
        self.assertNotIn("inquiry_state", public)
        self.assertNotIn(experiment.CANDIDATE, public)
        with patch.object(lab, "build_adapter") as build:
            with self.assertRaises(ValueError):
                self.run_experiment()
        build.assert_not_called()

    def test_late_invalid_script_has_no_artifacts_or_calls(self):
        original = lab.load_script

        def load(path):
            system, turns = original(path)
            return system, turns[:-1] if "newsletter" in path.name else turns

        with (
            patch.object(lab, "load_script", load),
            patch.object(lab, "build_adapter") as build,
        ):
            with self.assertRaises(ValueError):
                self.run_experiment()
        self.assertFalse(self.out.exists())
        build.assert_not_called()

    def test_invalid_memos_keep_all_pairs_and_clear_state(self):
        original = lab.DummyAdapter.generate

        def invalid(adapter, **kwargs):
            response = original(adapter, **kwargs)
            if kwargs["system"] == experiment.CANDIDATE:
                response.text = "REJECTED_CANARY"
            return response

        with patch.object(lab.DummyAdapter, "generate", invalid):
            metrics = self.run_experiment()
        self.assertEqual(
            (
                metrics["valid_workpads"],
                metrics["all_pairs"],
                metrics["complete_pairs"],
            ),
            (36, 6, 0),
        )
        self.assertFalse((self.out / "complete_review").exists())
        self.assertNotIn(
            "REJECTED_CANARY", (self.out / "review/booklet.md").read_text()
        )

    def test_write_ahead_failure_no_retry_and_prompt_restored(self):
        def fail(*a, **k):
            calls = pilot.read_json(self.out / "generation_receipts.json")
            self.assertEqual(calls[-1]["status"], "started")
            raise RuntimeError("SECRET_CANARY")

        with patch.object(lab.DummyAdapter, "generate", side_effect=fail) as generate:
            with self.assertRaises(RuntimeError):
                self.run_experiment()
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(
            pilot.read_json(self.out / "run_manifest.json")["status"], "incomplete"
        )
        self.assertEqual(lab.INQUIRY_PROBE_SYSTEM, experiment.CONTROL)
        self.assertNotIn(
            "SECRET_CANARY", (self.out / "generation_receipts.json").read_text()
        )

    def test_budget_enforced_before_next_dispatch(self):
        with patch.object(experiment, "CALL_LIMIT", 1):
            with self.assertRaisesRegex(RuntimeError, "budget"):
                self.run_experiment()
        self.assertEqual(len(pilot.read_json(self.out / "generation_receipts.json")), 1)

    def test_missing_credential_fails_before_reservation(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                experiment.run_comparison(self.out, live=True)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
