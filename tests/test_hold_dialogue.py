# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline contracts for a revisable inquiry workpad, not model-quality evidence."""

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from inquiry_state import INQUIRY_PROBE_SYSTEM, InquiryState
from build_human_eval_set import build_items
import recursive_conclusion_lab as rcl
from playtest_server import (
    CreateSessionRequest,
    SessionManager,
    restore_session_state,
    serialize_session_state,
)


FIRST = {
    "focus": "How could a shared room serve the neighborhood?",
    "hypotheses": [
        "A repair workshop could help.",
        "A quiet reading space could help.",
    ],
    "open_questions": ["What noise and access limits apply?"],
    "next_step": "Clarify the building's constraints before choosing.",
    "revision_note": "",
}
REVISED = {
    "focus": "What quiet activity could work with limited access?",
    "hypotheses": ["A reading space could fit."],
    "open_questions": ["Which opening hours are possible?"],
    "next_step": "Explore a small reading-space trial.",
    "revision_note": "Removed the workshop after the user ruled out noise.",
}


class WorkpadAdapter(rcl.DummyAdapter):
    def __init__(self, outputs):
        super().__init__(model="dummy-v1")
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, *, system, messages, config):
        self.calls.append((system, copy.deepcopy(messages)))
        if system == INQUIRY_PROBE_SYSTEM:
            value = next(self.outputs)
            text = json.dumps(value) if not isinstance(value, str) else value
        else:
            text = "A fixture reply; this test does not evaluate language quality."
        return rcl.ProviderResponse(
            provider="dummy", model=self.model, text=text, raw={"dummy": True}
        )


class HoldDialogueTests(unittest.TestCase):
    def test_example_matrix_feeds_blind_review_without_workpad_metadata(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            config = json.loads(
                (repo / "templates/compare_matrix_hold_dialogue.json").read_text()
            )
            config["script"] = str(repo / config["script"])
            config["out_dir"] = tmp
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rcl.run_compare_matrix_from_config_data(config), 0)
            review = json.loads(
                (repo / "templates/human_eval_hold_dialogue.json").read_text()
            )
            review["scenarios"][0]["compare_out_dir"] = tmp
            items, _ = build_items(review)
            self.assertEqual(len(items), 6)
            serialized = json.dumps(items)
            self.assertNotIn("inquiry_state", serialized)
            self.assertNotIn("hold_workpad", serialized)
            self.assertNotIn("conclusion_mode", serialized)
            rows = json.loads(
                (Path(tmp) / "summary__hold_workpad__run_001.json").read_text()
            )
            self.assertEqual(
                [row["inquiry_state_turn"] for row in rows], list(range(1, 7))
            )

    def session(self, outputs, **overrides):
        config = rcl.ExperimentConfig(
            conclusion_mode=rcl.ConclusionMode.HOLD,
            conclusion_every=1,
            memory_every=0,
            **overrides,
        )
        adapter = WorkpadAdapter(outputs)
        return rcl.RecursiveConclusionSession(adapter=adapter, config=config), adapter

    def test_latest_user_turn_revises_workpad_without_conclusion_target(self):
        session, adapter = self.session([FIRST, REVISED])
        first = session.user_turn("Let's explore uses for the room; do not choose yet.")
        self.assertEqual(first["inquiry_state"], FIRST)
        second = session.user_turn("New premise: no noise is allowed.")

        self.assertEqual(second["inquiry_state"], REVISED)
        self.assertEqual(second["inquiry_state_turn"], 2)
        self.assertIn("no noise is allowed", adapter.calls[2][1][0].content)
        self.assertEqual(
            json.loads(adapter.calls[2][1][0].content)["previous_workpad"], FIRST
        )
        self.assertNotIn("A repair workshop could help.", second["system_prompt"])
        self.assertIn("When the user explicitly asks", second["system_prompt"])
        self.assertIsNone(second["conclusion_probe"])
        self.assertEqual(session.conclusion_hypotheses, [])
        self.assertEqual(session.delayed_mentions, [])
        self.assertEqual(session.deferred_intents, [])
        self.assertNotIn("Provisional end-state hypothesis", second["system_prompt"])
        self.assertEqual(len(adapter.calls), 4)

    def test_invalid_revision_clears_stale_workpad_and_continues_dialogue(self):
        session, _ = self.session([FIRST, "not a workpad"])
        session.user_turn("Explore the room's possible uses.")
        result = session.user_turn("The old premise no longer applies.")
        self.assertIsNone(result["inquiry_state"])
        self.assertIsNone(result["inquiry_state_turn"])
        self.assertNotIn("A repair workshop could help.", result["system_prompt"])
        self.assertTrue(result["assistant"])

    def test_workpad_schema_rejects_unbounded_or_finalized_state(self):
        invalid = [
            {**FIRST, "conclusion": "Choose the workshop."},
            {**FIRST, "hypotheses": ["x"] * 4},
            {**FIRST, "open_questions": ["x"] * 5},
            {**FIRST, "focus": "x" * 401},
            {**FIRST, "next_step": ""},
            {**FIRST, "hypotheses": "not a list"},
        ]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    InquiryState.from_dict(value)
        self.assertEqual(InquiryState.from_dict(FIRST).to_dict(), FIRST)

    def test_prompt_only_hold_makes_no_workpad_call(self):
        adapter = WorkpadAdapter([])
        session = rcl.RecursiveConclusionSession(
            adapter=adapter,
            config=rcl.ExperimentConfig(
                conclusion_mode=rcl.ConclusionMode.HOLD,
                conclusion_every=0,
                memory_every=0,
            ),
        )
        result = session.user_turn("Explore before deciding.")
        self.assertEqual(len(adapter.calls), 1)
        self.assertIsNone(result["inquiry_state"])
        self.assertIn("making useful progress", result["system_prompt"])

    def test_incompatible_controllers_fail_before_reservation_in_every_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "script.json"
            script.write_text(json.dumps(["Explore."]), encoding="utf-8")
            for index, run in enumerate((rcl.run_compare, rcl.execute_compare)):
                out = root / str(index)
                args = rcl.build_parser().parse_args(
                    [
                        "compare",
                        "--script",
                        str(script),
                        "--providers",
                        "dummy=dummy-v1",
                        "--out-dir",
                        str(out),
                        "--conclusion-mode",
                        "hold",
                        "--delayed-mention-every",
                        "1",
                    ]
                )
                with mock.patch.object(rcl.DummyAdapter, "generate") as generate:
                    with self.assertRaisesRegex(ValueError, "hold"):
                        run(args)
                generate.assert_not_called()
                self.assertFalse(out.exists())
            out = root / "matrix"
            config = {
                "script": str(script),
                "providers": ["dummy=dummy-v1"],
                "out_dir": str(out),
                "arms": [
                    {"name": "valid", "args": {}},
                    {
                        "name": "invalid",
                        "args": {
                            "conclusion_mode": "hold",
                            "delayed_mention_mode": "soft_fire",
                        },
                    },
                ],
            }
            with mock.patch.object(rcl.DummyAdapter, "generate") as generate:
                with self.assertRaisesRegex(ValueError, "hold"):
                    rcl.run_compare_matrix_from_config_data(config)
            generate.assert_not_called()
            self.assertFalse(out.exists())

    def test_saved_hold_session_resumes_and_failed_reply_rolls_back_workpad(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = SessionManager(root)
            detail = manager.create_session(
                CreateSessionRequest(
                    provider="dummy", model="dummy-v1", arm_preset="hold"
                )
            )
            session_id = detail["session_id"]
            self.assertEqual(detail["config"]["semantic_judge_backend"], "off")
            manager.append_turn(session_id, "Explore without deciding.")
            record = manager.get_record(session_id)
            before = serialize_session_state(record.session)
            log_before = record.log_path.read_bytes()
            resumed = SessionManager(root).get_record(session_id)
            self.assertEqual(serialize_session_state(resumed.session), before)

            def fail_reply(**kwargs):
                if kwargs["system"] == INQUIRY_PROBE_SYSTEM:
                    return rcl.ProviderResponse(
                        provider="dummy",
                        model="dummy-v1",
                        text=json.dumps(REVISED),
                        raw={},
                    )
                raise RuntimeError("simulated reply failure")

            with mock.patch.object(
                record.session, "_probe_memory_capsule", return_value=None
            ):
                with mock.patch.object(
                    record.session.adapter, "generate", side_effect=fail_reply
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "simulated reply failure"
                    ):
                        manager.append_turn(session_id, "The premise changed.")
            self.assertEqual(serialize_session_state(record.session), before)
            self.assertEqual(record.log_path.read_bytes(), log_before)
            resumed_manager = SessionManager(root)
            self.assertEqual(
                serialize_session_state(resumed_manager.get_record(session_id).session),
                before,
            )
            self.assertEqual(
                resumed_manager.session_detail(session_id)["pending_user_text"],
                "The premise changed.",
            )

    def test_legacy_snapshot_without_workpad_restores(self):
        session, _ = self.session([FIRST])
        state = serialize_session_state(session)
        state.pop("inquiry_state")
        state.pop("inquiry_state_turn")
        restore_session_state(session, state)
        self.assertIsNone(session.inquiry_state)
        self.assertIsNone(session.inquiry_state_turn)


if __name__ == "__main__":
    unittest.main()
