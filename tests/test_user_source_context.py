# SPDX-License-Identifier: AGPL-3.0-or-later
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import recursive_conclusion_lab as lab
from playtest_server import (
    experiment_config_from_dict,
    serialize_session_state,
    restore_session_state,
)
from user_source_context import MAX_SOURCE_CHARS, MAX_SOURCE_TURNS, source_user_history


class UserSourceTests(unittest.TestCase):
    def session(self, replay=True):
        return lab.RecursiveConclusionSession(
            adapter=lab.DummyAdapter("dummy-v1"),
            config=lab.ExperimentConfig(
                conclusion_mode=lab.ConclusionMode.HOLD,
                conclusion_every=0,
                memory_every=0,
                hold_user_replay=replay,
            ),
        )

    def test_original_source_returns_without_extra_generation_or_system_promotion(self):
        session = self.session()
        captured = []
        original = session.adapter.generate

        def generate(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return original(**kwargs)

        with patch.object(session.adapter, "generate", generate):
            for i in range(1, 7):
                session.user_turn(f"USER_SOURCE_{i}: keep open.")
        self.assertEqual(len(captured), 6)
        payload = json.loads(
            captured[-1]["messages"][1].content.split("\n\nEarlier assistant reply:")[0]
        )
        source = payload["source_user_history"]
        self.assertEqual(
            source["messages"],
            [
                {"turn": 1, "text": "USER_SOURCE_1: keep open."},
                {"turn": 2, "text": "USER_SOURCE_2: keep open."},
            ],
        )
        self.assertTrue(source["complete_outside_window"])
        self.assertNotIn("USER_SOURCE", captured[-1]["system"])
        self.assertEqual(
            captured[-1]["messages"][-1].content, "USER_SOURCE_6: keep open."
        )
        self.assertEqual(len(session.history), 12)
        self.assertFalse(
            any("source_user_history" in m.content for m in session.history)
        )
        self.assertNotIn(
            "USER_SOURCE_1", str([m.content for m in session._recent_window()])
        )

    def test_revision_is_not_classified_or_removed_and_assistant_cannot_create_source(
        self,
    ):
        history = [
            lab.ChatMessage("user", "Original goal; do not store returns."),
            lab.ChatMessage("assistant", "ASSISTANT_CANARY: storage allowed."),
            lab.ChatMessage(
                "user", "I now explicitly allow checked returns overnight."
            ),
            lab.ChatMessage("assistant", "A proposal."),
            lab.ChatMessage("user", "Continue."),
        ]
        before = copy.deepcopy(history)
        source = source_user_history(history, 1)
        self.assertEqual([m["turn"] for m in source["messages"]], [1, 2])
        self.assertIn("now explicitly allow", source["messages"][1]["text"])
        self.assertNotIn("ASSISTANT_CANARY", json.dumps(source))
        self.assertEqual(history, before)
        self.assertIsNone(source_user_history(history, 0))

    def test_bound_is_verbatim_and_omission_is_explicit(self):
        history = [lab.ChatMessage("user", "x" * (MAX_SOURCE_CHARS + 1))]
        history += [lab.ChatMessage("user", str(i)) for i in range(30)]
        result = source_user_history(history, 1)
        self.assertLessEqual(result["text_characters"], MAX_SOURCE_CHARS)
        self.assertLessEqual(len(result["messages"]), MAX_SOURCE_TURNS)
        self.assertFalse(result["complete_outside_window"])
        self.assertEqual(result["first_omitted_turn"], 1)
        self.assertFalse(any("x" in m["text"] for m in result["messages"]))
        self.assertEqual(result["omitted_count"] + len(result["messages"]), 30)
        small_origin = [lab.ChatMessage("user", "origin"), *history[1:]]
        self.assertEqual(
            source_user_history(small_origin, 1)["messages"][0]["text"], "origin"
        )

    def test_resume_reconstructs_same_source_without_a_second_mutable_ledger(self):
        session = self.session()
        for i in range(6):
            session.user_turn(f"Turn {i}")
        before = session._reply_messages()
        snapshot = serialize_session_state(session)
        resumed = self.session()
        restore_session_state(resumed, snapshot)
        self.assertEqual(resumed._reply_messages(), before)
        # Standard persisted config values are serialized by the server.
        import dataclasses

        data = json.loads(json.dumps(dataclasses.asdict(session.config)))
        restored = experiment_config_from_dict(data)
        self.assertTrue(restored.hold_user_replay)
        data.pop("hold_user_replay")
        self.assertFalse(experiment_config_from_dict(data).hold_user_replay)

    def test_disabled_mode_keeps_legacy_messages(self):
        session = self.session(False)
        for i in range(6):
            result = session.user_turn(f"Turn {i}")
        self.assertNotIn("source_user_history", result)
        self.assertEqual(session._reply_messages(), session._recent_window())

    def test_nonhold_replay_fails_before_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "script.json"
            script.write_text('["A question"]')
            for run in (lab.run_compare, lab.execute_compare):
                args = lab.build_parser().parse_args(
                    [
                        "compare",
                        "--script",
                        str(script),
                        "--providers",
                        "dummy=dummy-v1",
                        "--out-dir",
                        str(root / "out"),
                        "--hold-user-replay",
                    ]
                )
                with patch.object(lab, "build_adapter") as build:
                    with self.assertRaisesRegex(ValueError, "requires"):
                        run(args)
                build.assert_not_called()
                self.assertFalse((root / "out").exists())
            with self.assertRaisesRegex(ValueError, "boolean"):
                lab.validate_hold_configuration(
                    lab.ExperimentConfig(hold_user_replay="false")
                )


if __name__ == "__main__":
    unittest.main()
