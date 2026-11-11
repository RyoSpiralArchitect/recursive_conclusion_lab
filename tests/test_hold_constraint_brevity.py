# SPDX-License-Identifier: AGPL-3.0-or-later
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hold_constraint_brevity as experiment
import hold_dialogue_pilot as pilot
import recursive_conclusion_lab as lab


class ConstraintBrevityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name) / "run"

    def run_experiment(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return experiment.run_comparison(self.out, **kwargs)

    def test_factor_isolation_balanced_order_accounting_and_review(self):
        orders = []
        for suite in experiment.SUITES:
            self.out = self.out.parent / suite
            with patch.object(
                lab, "post_json", side_effect=AssertionError("No network")
            ):
                metrics = self.run_experiment(suite=suite)
            self.assertEqual(
                (
                    metrics["generations"],
                    metrics["valid_workpads"],
                    metrics["all_pairs"],
                    metrics["complete_pairs"],
                ),
                (192, 96, 16, 16),
            )
            calls = pilot.read_json(self.out / "generation_receipts.json")
            self.assertEqual(
                {c["prompt_sha256"] for c in calls if c["kind"] == "inquiry_update"},
                {experiment.sha(experiment.PROBE)},
            )
            for arm in experiment.ARMS:
                replies = [
                    c
                    for c in calls
                    if c["arm"] == arm and c["kind"] == "assistant_reply"
                ]
                self.assertEqual(len(replies), 24)
                self.assertEqual(len({c["prompt_sha256"] for c in replies}), 1)
            self.assertEqual(
                len(
                    {
                        c["prompt_sha256"]
                        for c in calls
                        if c["kind"] == "assistant_reply"
                    }
                ),
                4,
            )
            orders.extend(
                pilot.read_json(self.out / "run_manifest.json")["execution_order"]
            )
            public = (self.out / "review/eval_items.jsonl").read_text()
            self.assertNotIn("inquiry_state", public)
            self.assertNotIn(experiment.CONSTRAINTS, public)
            key = pilot.read_json(self.out / "review/blind_key.json")
            self.assertEqual(
                {
                    tuple(k["arm_pair_sorted"])
                    for name, k in key.items()
                    if not name.startswith("_")
                },
                {tuple(sorted(p)) for p in experiment.CONTRASTS},
            )
            self.assertEqual(lab.INQUIRY_REPLY_SYSTEM, experiment.CONTROL)
            self.assertEqual(lab.INQUIRY_PROBE_SYSTEM, experiment.PROBE)
            with patch.object(lab, "build_adapter") as build:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    self.run_experiment(suite=suite)
            build.assert_not_called()
        for position in range(4):
            self.assertEqual(
                sorted(o["arm"] for o in orders[position::4]),
                sorted(list(experiment.ARMS) * 2),
            )
        self.assertEqual(
            experiment.PROMPTS["both"],
            experiment.CONTROL + experiment.CONSTRAINTS + experiment.BREVITY,
        )

    def test_late_script_drift_fails_before_artifacts_or_calls(self):
        original = lab.load_script

        def drift(path):
            system, turns = original(path)
            if "craft_pickup" in path.name:
                turns[-1] += " Changed."
            return system, turns

        with (
            patch.object(lab, "load_script", drift),
            patch.object(lab, "build_adapter") as build,
        ):
            with self.assertRaisesRegex(ValueError, "frozen facts"):
                self.run_experiment()
        self.assertFalse(self.out.exists())
        build.assert_not_called()

    def test_invalid_optional_pairing_precedes_reservation_and_catalog(self):
        for invalid in (
            {"observer_provider": "openai"},
            {"embedding_provider": "mistral", "embedding_model": "bad"},
        ):
            with (
                self.subTest(invalid=invalid),
                patch.dict(experiment.SETTINGS, invalid),
            ):
                # Keep freeze matched to test downstream adapter prevalidation.
                original = pilot.read_json

                def frozen(path):
                    value = original(path)
                    if path.name == Path(experiment.FROZEN).name:
                        value["settings"] = experiment.SETTINGS.copy()
                    return value

                with (
                    patch.object(pilot, "read_json", frozen),
                    patch.object(lab, "build_adapter") as build,
                ):
                    with self.assertRaises(ValueError):
                        self.run_experiment()
                self.assertFalse(self.out.exists())
                build.assert_not_called()

    def test_missing_key_no_reservation_or_dispatch(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(lab, "build_adapter") as build,
        ):
            with self.assertRaisesRegex(ValueError, "credential"):
                self.run_experiment(live=True)
        self.assertFalse(self.out.exists())
        build.assert_not_called()

    def test_write_ahead_no_retry_and_prompt_restoration(self):
        def fail(*args, **kwargs):
            self.assertEqual(
                pilot.read_json(self.out / "generation_receipts.json")[-1]["status"],
                "started",
            )
            raise RuntimeError("SECRET_CANARY")

        with patch.object(lab.DummyAdapter, "generate", fail) as generate:
            with self.assertRaises(RuntimeError):
                self.run_experiment()
        self.assertIs(generate, fail)
        calls = pilot.read_json(self.out / "generation_receipts.json")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["status"], "failed_or_unknown")
        self.assertNotIn("SECRET_CANARY", str(calls))
        self.assertEqual(
            pilot.read_json(self.out / "run_manifest.json")["status"], "incomplete"
        )
        self.assertEqual(lab.INQUIRY_REPLY_SYSTEM, experiment.CONTROL)

    def test_budget_stops_before_next_dispatch(self):
        with patch.object(experiment, "CALL_LIMIT", 1):
            with self.assertRaisesRegex(RuntimeError, "budget"):
                self.run_experiment()
        self.assertEqual(len(pilot.read_json(self.out / "generation_receipts.json")), 1)

    def test_invalid_workpads_kept_in_full_set_cleared_before_reply(self):
        original = lab.DummyAdapter.generate

        def invalid(adapter, **kwargs):
            if kwargs["system"] != experiment.PROBE:
                self.assertNotIn("REJECTED_CANARY", json.dumps(kwargs, default=str))
            response = original(adapter, **kwargs)
            if (
                kwargs["system"] == experiment.PROBE
                and lab.INQUIRY_REPLY_SYSTEM == experiment.PROMPTS["both"]
            ):
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
            (72, 16, 8),
        )
        self.assertNotIn(
            "REJECTED_CANARY", (self.out / "review/booklet.md").read_text()
        )

    def test_prompt_drift_rejected_before_artifacts(self):
        with (
            patch.dict(experiment.PROMPTS, {"brief": "drift"}),
            patch.object(lab, "build_adapter") as build,
        ):
            with self.assertRaisesRegex(ValueError, "frozen design"):
                self.run_experiment()
        self.assertFalse(self.out.exists())
        build.assert_not_called()

    def test_abnormal_short_reply_does_not_count_as_complete(self):
        original = lab.DummyAdapter.generate

        def truncated(adapter, **kwargs):
            response = original(adapter, **kwargs)
            if kwargs["system"] != experiment.PROBE:
                response.text, response.finish_reason = "Short but unfinished", "length"
            return response

        with patch.object(lab.DummyAdapter, "generate", truncated):
            metrics = self.run_experiment()
        self.assertEqual(metrics["complete_pairs"], 0)
        for arm in pilot.read_json(self.out / "completion_metrics.json")[
            "by_arm"
        ].values():
            self.assertEqual(arm["within_180_words"], 24)
            self.assertEqual(arm["short_complete"], 0)


if __name__ == "__main__":
    unittest.main()
