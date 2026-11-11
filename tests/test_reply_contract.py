# SPDX-License-Identifier: AGPL-3.0-or-later
import contextlib
import copy
import dataclasses
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import analyze_runs
import recursive_conclusion_lab as lab
from playtest_server import (
    experiment_config_from_dict,
    restore_session_state,
    serialize_session_state,
)
from reply_contract import (
    WORD_COUNTER,
    budget_instruction,
    count_words,
    reply_contract,
    summarize_contracts,
)


class ReplyContractTests(unittest.TestCase):
    def session(self, limit=0, log=None):
        return lab.RecursiveConclusionSession(
            adapter=lab.DummyAdapter("dummy-v1"),
            log_path=log,
            config=lab.ExperimentConfig(
                conclusion_mode=lab.ConclusionMode.HOLD,
                conclusion_every=0,
                memory_every=0,
                hold_reply_max_words=limit,
                hold_user_replay=True,
            ),
        )

    def test_counter_keeps_markdown_and_unicode_whitespace_definition(self):
        text = "## Choice\n- two-word\u00a0idea; keep_going."
        self.assertEqual(count_words(text), 6)
        self.assertEqual(reply_contract(text, 6)["status"], "within_budget")
        self.assertEqual(reply_contract(text, 5)["excess_words"], 1)
        self.assertEqual(reply_contract("  \n", 5)["status"], "empty")
        self.assertEqual(reply_contract(text)["status"], "not_configured")
        for bad in (True, False, -1, 1.5, "180", None):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                reply_contract(text, bad)

    def test_overlength_reply_retains_final_constraint_without_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "trace.jsonl"
            session = self.session(5, log)
            reply = "A small trial remains provisional. No returned tool may remain overnight."
            captured = []

            def generate(**kwargs):
                captured.append(copy.deepcopy(kwargs))
                return lab.ProviderResponse(
                    "dummy", "dummy-v1", reply, {}, finish_reason="stop"
                )

            with (
                patch.object(session.adapter, "generate", generate),
                patch.object(
                    lab, "post_json", side_effect=AssertionError("No network")
                ),
            ):
                result = session.user_turn(
                    "Reopen this; all returned tools must leave before closing."
                )
            self.assertEqual(len(captured), 1)
            self.assertIn(budget_instruction(5), captured[0]["system"])
            self.assertEqual(result["assistant"], reply)
            self.assertEqual(session.history[-1].content, reply)
            self.assertEqual(result["reply_contract"]["status"], "exceeded")
            self.assertNotIn("reply_contract", str(session._reply_messages()))
            event = next(
                json.loads(x)
                for x in log.read_text().splitlines()
                if json.loads(x)["event_type"] == "assistant_reply"
            )
            self.assertEqual(
                event["payload"]["reply_contract"], result["reply_contract"]
            )
            metrics = analyze_runs.summarize_log(log, {})
            self.assertEqual(metrics["avg_reply_words"], len(re.findall(r"\w+", reply)))
            self.assertEqual(metrics["avg_reply_whitespace_words"], len(reply.split()))
            self.assertEqual(metrics["reply_whitespace_word_counter"], WORD_COUNTER)
            self.assertEqual(metrics["reply_budget_exceeded_count"], 1)

    def test_default_preserves_prompt_and_no_contract_is_inferred(self):
        session = self.session()
        self.assertEqual(session._build_system_prompt(), lab.INQUIRY_REPLY_SYSTEM)
        result = session.user_turn("Explore before choosing.")
        self.assertNotIn("reply_contract", result)
        self.assertEqual(
            summarize_contracts([result])["reply_budget_configured_count"], 0
        )

    def test_finish_and_length_are_separate_and_bad_diagnostics_are_not_success(self):
        row = dict(
            assistant="A short reply",
            finish_reason="length",
            reply_contract=reply_contract("A short reply", 10),
        )
        self.assertEqual(summarize_contracts([row])["reply_budget_within_count"], 1)
        self.assertNotIn("normal_finish", row["reply_contract"])
        for change in ({"words": 0}, {"word_counter": "unknown"}, {"max_words": True}):
            broken = {**row, "reply_contract": {**row["reply_contract"], **change}}
            self.assertEqual(
                summarize_contracts([broken])["reply_budget_invalid_diagnostic_count"],
                1,
            )
            self.assertEqual(
                summarize_contracts([broken])["reply_budget_configured_count"], 0
            )

    def test_persistence_retains_target_and_explicit_revision_in_current_user_turn(
        self,
    ):
        session = self.session(180)
        session.user_turn("Original goal: learning. No overnight returns.")
        state = serialize_session_state(session)
        config = experiment_config_from_dict(
            json.loads(json.dumps(dataclasses.asdict(session.config)))
        )
        restored = lab.RecursiveConclusionSession(
            adapter=lab.DummyAdapter("dummy-v1"), config=config
        )
        restore_session_state(restored, state)
        captured = []
        original = restored.adapter.generate

        def generate(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return original(**kwargs)

        revision = "I explicitly change the goal to completed repairs; voluntary participation remains."
        with patch.object(restored.adapter, "generate", generate):
            result = restored.user_turn(revision)
        self.assertEqual(restored.config.hold_reply_max_words, 180)
        self.assertEqual(captured[0]["messages"][-1].content, revision)
        self.assertNotIn("Original goal", captured[0]["system"])
        self.assertEqual(result["reply_contract"]["max_words"], 180)
        self.assertEqual(experiment_config_from_dict({}).hold_reply_max_words, 0)
        with self.assertRaises(ValueError):
            lab.validate_hold_configuration(
                experiment_config_from_dict({"hold_reply_max_words": "180"})
            )

    def test_invalid_single_and_late_matrix_targets_fail_before_artifacts_or_calls(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "script.json"
            script.write_text('["Explore."]')
            for run in (lab.run_compare, lab.execute_compare):
                for mode, limit in (("hold", -1), ("observe", 180)):
                    args = lab.build_parser().parse_args(
                        [
                            "compare",
                            "--script",
                            str(script),
                            "--providers",
                            "dummy=dummy-v1",
                            "--out-dir",
                            str(root / "out"),
                            "--conclusion-mode",
                            mode,
                            "--hold-reply-max-words",
                            str(limit),
                        ]
                    )
                    with (
                        patch.object(lab, "build_adapter") as build,
                        self.assertRaises(ValueError),
                    ):
                        run(args)
                    build.assert_not_called()
                    self.assertFalse((root / "out").exists())
            data = dict(
                script=str(script),
                providers=["dummy=dummy-v1"],
                out_dir=str(root / "matrix"),
                args=dict(conclusion_mode="hold", memory_every=0),
                arms=[
                    dict(name="good", args=dict(hold_reply_max_words=180)),
                    dict(name="bad", args=dict(hold_reply_max_words=-1)),
                ],
            )
            with (
                patch.object(lab, "build_adapter") as build,
                self.assertRaises(ValueError),
            ):
                lab.run_compare_matrix_from_config_data(data)
            build.assert_not_called()
            self.assertFalse((root / "matrix").exists())
            for bad in (True, False, "180", 1.5):
                data["arms"][1]["args"]["hold_reply_max_words"] = bad
                with (
                    patch.object(lab, "build_adapter") as build,
                    self.assertRaises(ValueError),
                ):
                    lab.run_compare_matrix_from_config_data(data)
                build.assert_not_called()
                self.assertFalse((root / "matrix").exists())

    def test_compare_summary_carries_contract_without_exposing_it_in_transcript(self):
        import build_human_eval_set as review

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "script.json"
            script.write_text('["Explore."]')
            args = lab.build_parser().parse_args(
                [
                    "compare",
                    "--script",
                    str(script),
                    "--providers",
                    "dummy=dummy-v1",
                    "--out-dir",
                    str(root / "out"),
                    "--conclusion-mode",
                    "hold",
                    "--conclusion-every",
                    "0",
                    "--memory-every",
                    "0",
                    "--hold-reply-max-words",
                    "10",
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                execution = lab.execute_compare(args)
            self.assertEqual(execution.rows[0]["reply_contract"]["max_words"], 10)
            self.assertNotIn("reply_contract", review.render_transcript(execution.rows))


if __name__ == "__main__":
    unittest.main()
