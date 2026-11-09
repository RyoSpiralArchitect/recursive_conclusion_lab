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

    def test_failure_metadata_keeps_only_allowlisted_http_status(self):
        self.assertEqual(
            experiment.failure_metadata(
                RuntimeError(
                    "HTTP 429 from provider endpoint; response details omitted"
                )
            )["http_status"],
            429,
        )
        self.assertIsNone(
            experiment.failure_metadata(RuntimeError("SECRET_CANARY"))["http_status"]
        )
        self.assertNotIn(
            "SECRET_CANARY",
            str(experiment.failure_metadata(RuntimeError("SECRET_CANARY"))),
        )

    def test_openai_selection_requires_its_own_key(self):
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "fixture"}, clear=True):
            with self.assertRaisesRegex(ValueError, "credential"):
                experiment.run_comparison(
                    self.out, live=True, selected_provider="openai"
                )
        self.assertFalse(self.out.exists())

    def test_unsupported_provider_has_no_artifacts(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            experiment.run_comparison(self.out, selected_provider="unknown")
        self.assertFalse(self.out.exists())

    def test_missing_credential_fails_before_reservation(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                experiment.run_comparison(self.out, live=True)
        self.assertFalse(self.out.exists())

    def test_new_suites_keep_budget_and_isolate_scenarios(self):
        for suite in ("scope", "purpose"):
            with self.subTest(suite=suite):
                out = self.out / suite
                with (
                    patch.object(
                        lab, "post_json", side_effect=AssertionError("No network")
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    metrics = experiment.run_comparison(out, suite=suite)
                self.assertEqual(
                    (
                        metrics["generations"],
                        metrics["valid_workpads"],
                        metrics["all_pairs"],
                        metrics["complete_pairs"],
                    ),
                    (144, 72, 6, 6),
                )
                manifest = pilot.read_json(out / "run_manifest.json")
                self.assertEqual(manifest["suite"], suite)
                self.assertEqual(
                    {d["scenario"] for d in metrics["dialogues"]},
                    set(experiment.SUITES[suite][0]),
                )
                self.assertEqual(
                    [r["arm"] for r in manifest["execution_order"][::2]],
                    ["current", "action", "current", "action", "current", "action"],
                )
                self.assertIn(
                    "purpose_and_scope", (out / "human_eval.json").read_text()
                )

    def test_new_suite_late_script_drift_has_no_artifacts_or_calls(self):
        original = lab.load_script

        def drift(path):
            system, turns = original(path)
            if "quiet_reading" in path.name:
                turns[-1] = "A different final request."
            return system, turns

        with (
            patch.object(lab, "load_script", drift),
            patch.object(lab, "build_adapter") as build,
        ):
            with self.assertRaisesRegex(ValueError, "frozen fact ledger"):
                experiment.run_comparison(self.out, suite="purpose")
        self.assertFalse(self.out.exists())
        build.assert_not_called()

    def test_unknown_suite_has_no_artifacts_or_calls(self):
        with patch.object(lab, "build_adapter") as build:
            with self.assertRaisesRegex(ValueError, "Unsupported experiment suite"):
                experiment.run_comparison(self.out, suite="unknown")
        self.assertFalse(self.out.exists())
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
