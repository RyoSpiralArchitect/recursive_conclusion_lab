# SPDX-License-Identifier: AGPL-3.0-or-later
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hold_source_comparison as experiment
import hold_dialogue_pilot as pilot
import recursive_conclusion_lab as lab


class SourceExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "run"

    def run_experiment(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return experiment.run_comparison(self.out, **kwargs)

    def test_both_suites_call_savings_source_provenance_and_review(self):
        for suite in experiment.SUITES:
            self.out = self.out.parent / suite
            with patch.object(
                lab, "post_json", side_effect=AssertionError("No network")
            ):
                metrics = self.run_experiment(suite=suite)
            self.assertEqual(
                (
                    metrics["generations"],
                    metrics["replies"],
                    metrics["workpads"],
                    metrics["complete_pairs"],
                ),
                (288, 216, 72, 36),
            )
            calls = pilot.read_json(self.out / "generation_receipts.json")
            self.assertEqual(
                {c["arm"] for c in calls if c["kind"] == "inquiry_update"}, {"workpad"}
            )
            self.assertEqual(
                len(
                    {
                        c["prompt_sha256"]
                        for c in calls
                        if c["kind"] == "assistant_reply"
                    }
                ),
                1,
            )
            self.assertEqual(
                [metrics["by_arm"][a]["calls"] for a in experiment.ARMS], [144, 72, 72]
            )
            self.assertNotIn(
                "source_user_history",
                (self.out / "review/eval_items.jsonl").read_text(),
            )
            self.assertEqual(lab.INQUIRY_REPLY_SYSTEM, experiment.prior.CONTROL)
            order = pilot.read_json(self.out / "run_manifest.json")["execution_order"]
            for scene in {row["scenario"] for row in order}:
                group = [row for row in order if row["scenario"] == scene]
                for position in range(3):
                    self.assertEqual(
                        sorted(row["arm"] for row in group[position::3]),
                        sorted(experiment.ARMS),
                    )
            with patch.object(lab, "build_adapter") as build:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    self.run_experiment(suite=suite)
            build.assert_not_called()

    def test_late_invalid_script_has_no_calls_or_artifacts(self):
        original = lab.load_script

        def drift(path):
            system, turns = original(path)
            if "purpose_revision" in path.name:
                turns[-1] = "Changed after freezing."
            return system, turns

        with (
            patch.object(lab, "load_script", drift),
            patch.object(lab, "build_adapter") as build,
        ):
            with self.assertRaisesRegex(ValueError, "frozen facts"):
                self.run_experiment(suite="transfer")
        build.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_failed_call_write_ahead_no_retry_and_prompt_restored(self):
        def fail(*args, **kwargs):
            self.assertEqual(
                pilot.read_json(self.out / "generation_receipts.json")[-1]["status"],
                "started",
            )
            raise RuntimeError("SECRET_CANARY")

        with patch.object(lab.DummyAdapter, "generate", side_effect=fail) as generate:
            with self.assertRaises(RuntimeError):
                self.run_experiment()
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(lab.INQUIRY_REPLY_SYSTEM, experiment.prior.CONTROL)
        self.assertNotIn(
            "SECRET_CANARY", (self.out / "generation_receipts.json").read_text()
        )
        self.assertEqual(
            pilot.read_json(self.out / "run_manifest.json")["status"], "incomplete"
        )

    def test_budget_is_enforced_before_dispatch(self):
        with patch.object(experiment, "CALL_LIMIT", 1):
            with self.assertRaisesRegex(RuntimeError, "budget"):
                self.run_experiment()
        self.assertEqual(len(pilot.read_json(self.out / "generation_receipts.json")), 1)

    def test_missing_key_has_no_artifacts(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Credential"):
                self.run_experiment(live=True)
        self.assertFalse(self.out.exists())

    def test_invalid_workpads_do_not_exclude_no_workpad_pairs(self):
        original = lab.DummyAdapter.generate

        def invalid(adapter, **kwargs):
            response = original(adapter, **kwargs)
            if kwargs["system"] == experiment.PROBE:
                response.text = "REJECTED_CANARY"
            else:
                self.assertNotIn("REJECTED_CANARY", str(kwargs["messages"]))
            return response

        with patch.object(lab.DummyAdapter, "generate", invalid):
            metrics = self.run_experiment()
        self.assertEqual(
            (
                metrics["valid_workpads"],
                metrics["all_pairs"],
                metrics["complete_pairs"],
            ),
            (0, 36, 12),
        )


if __name__ == "__main__":
    unittest.main()
