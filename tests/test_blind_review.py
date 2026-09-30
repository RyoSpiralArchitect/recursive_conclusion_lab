# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest import mock

from blind_review import (
    BlindReviewManager,
    ReviewConflictError,
    ReviewSetCatalog,
    ReviewValidationError,
    json_digest,
)
from build_human_eval_set import build_items, write_outputs


ROOT_DIR = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT_DIR / "examples" / "human_eval_sets"
QUESTION_IDS = {
    "overall_better",
    "leakage_control_better",
    "staged_release_better",
    "earned_final_better",
}


def response_payload(choice: str = "A") -> dict[str, object]:
    return {
        "answers": {question_id: choice for question_id in QUESTION_IDS},
        "confidence": "medium",
        "evidence": "The final answer follows from constraints introduced earlier.",
        "counterevidence": "The other transcript is more concise.",
        "abstain": False,
    }


class ReviewSetCatalogTests(unittest.TestCase):
    def test_public_bundle_does_not_expose_private_fields(self) -> None:
        review_set = ReviewSetCatalog(FIXTURE_ROOT).load("earned_conclusion_demo")
        public_json = json.dumps(
            {"summary": review_set.summary(), "items": review_set.items},
            ensure_ascii=False,
        )
        for forbidden in (
            "arm_pair",
            "label_to_arm",
            "compare_out_dir",
            "provider",
            "model",
            "blind_key",
            "private_config",
        ):
            self.assertNotIn(forbidden, public_json)

    def test_builder_writes_reviewer_safe_public_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compare_dir = root / "compare"
            compare_dir.mkdir()
            for arm, assistant in (
                ("static", "Answer now."),
                ("adaptive", "Ask, then answer."),
            ):
                (compare_dir / f"summary__{arm}__run_000.json").write_text(
                    json.dumps(
                        [
                            {
                                "turn": 1,
                                "user": "What should I choose?",
                                "assistant": assistant,
                                "provider": "hidden-provider",
                                "model": "hidden-model",
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
            config = {
                "title": "Builder Test",
                "rubric_version": "test_v1",
                "seed": 11,
                "out_dir": str(root / "packets"),
                "scenarios": [
                    {
                        "name": "choice",
                        "display_name": "Choice",
                        "compare_out_dir": str(compare_dir),
                        "arms": ["static", "adaptive"],
                    }
                ],
                "questions": [
                    {
                        "id": "earned",
                        "prompt": "Which feels earned?",
                        "choices": ["A", "B", "Tie"],
                    }
                ],
            }
            items, blind_key = build_items(config)
            write_outputs(
                out_dir=root / "packets",
                items=items,
                blind_key=blind_key,
                config=config,
            )
            public_text = (root / "packets" / "manifest.json").read_text(
                encoding="utf-8"
            ) + (root / "packets" / "eval_items.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("static", public_text)
            self.assertNotIn("adaptive", public_text)
            self.assertNotIn("hidden-provider", public_text)
            self.assertNotIn("hidden-model", public_text)
            private_key = json.loads(
                (root / "packets" / "blind_key.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                private_key["_meta"]["schema"], "rcl.blind_pairwise_key.v2"
            )
            private_item_ids = [key for key in private_key if not key.startswith("_")]
            self.assertEqual(len(private_item_ids), 1)
            self.assertEqual(
                private_key[private_item_ids[0]]["arm_pair_sorted"],
                ["adaptive", "static"],
            )
            self.assertIsNone(re.fullmatch(r"item_\d{3}", private_item_ids[0]))
            self.assertTrue((root / "packets" / "READY.json").exists())
            generated_set = ReviewSetCatalog(root / "packets").load("packets")
            self.assertEqual(
                private_key["_meta"]["review_bundle_digest"],
                generated_set.review_bundle_digest,
            )
            self.assertEqual(len(private_key["_meta"]["source_digest"]), 64)

    def test_catalog_ignores_a_packet_without_readiness_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(
                FIXTURE_ROOT / "earned_conclusion_demo", root / "incomplete"
            )
            (root / "incomplete" / "READY.json").unlink()
            self.assertEqual(ReviewSetCatalog(root).list_sets(), [])

    def test_builder_rejects_divergent_user_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compare_dir = Path(temporary)
            for arm, user in (
                ("left", "Question one"),
                ("right", "Different question"),
            ):
                (compare_dir / f"summary__{arm}__run_000.json").write_text(
                    json.dumps([{"turn": 1, "user": user, "assistant": "Answer"}]),
                    encoding="utf-8",
                )
            with self.assertRaises(SystemExit):
                build_items(
                    {
                        "scenarios": [
                            {
                                "name": "divergent",
                                "compare_out_dir": str(compare_dir),
                                "arms": ["left", "right"],
                            }
                        ]
                    }
                )

    def test_builder_rejects_non_pairwise_choices(self) -> None:
        with self.assertRaises(SystemExit):
            build_items(
                {
                    "scenarios": [{}],
                    "questions": [
                        {
                            "id": "incomplete",
                            "prompt": "Which response?",
                            "choices": ["A", "B"],
                        }
                    ],
                }
            )

    def test_builder_rejects_duplicate_question_and_arm_ids(self) -> None:
        with self.assertRaises(SystemExit):
            build_items(
                {
                    "scenarios": [{}],
                    "questions": [
                        {
                            "id": "same",
                            "prompt": "First?",
                            "choices": ["A", "B", "Tie"],
                        },
                        {
                            "id": "same",
                            "prompt": "Second?",
                            "choices": ["A", "B", "Tie"],
                        },
                    ],
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            compare_dir = Path(temporary)
            (compare_dir / "summary__same__run_000.json").write_text(
                json.dumps([{"turn": 1, "user": "Question", "assistant": "Answer"}]),
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                build_items(
                    {
                        "scenarios": [
                            {
                                "name": "duplicate-arms",
                                "compare_out_dir": str(compare_dir),
                                "arms": ["same", "same"],
                            }
                        ]
                    }
                )


class BlindReviewManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.sessions_dir = Path(self.temporary.name) / "sessions"
        self.manager = BlindReviewManager(
            eval_sets_dir=FIXTURE_ROOT,
            sessions_dir=self.sessions_dir,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create_session(self, rater_id: str = "reviewer-01") -> dict[str, object]:
        return self.manager.create_session(
            eval_set_id="earned_conclusion_demo",
            rater_id=rater_id,
        )

    def test_assignment_is_stable_and_private(self) -> None:
        first = self.create_session()
        second = self.create_session()
        self.assertEqual(first["items"], second["items"])
        serialized = json.dumps(first, ensure_ascii=False)
        self.assertNotIn("presentation", serialized)
        self.assertNotIn("label_to_arm", serialized)
        self.assertNotIn("adaptive_kind_aware", serialized)

    def test_rebuilt_packet_or_key_does_not_reuse_stale_session(self) -> None:
        eval_sets_dir = Path(self.temporary.name) / "eval_sets"
        packet_dir = eval_sets_dir / "earned_conclusion_demo"
        shutil.copytree(FIXTURE_ROOT / "earned_conclusion_demo", packet_dir)
        manager = BlindReviewManager(
            eval_sets_dir=eval_sets_dir,
            sessions_dir=self.sessions_dir,
        )
        first = manager.create_session(
            eval_set_id="earned_conclusion_demo", rater_id="same-rater"
        )
        original_packet = manager.catalog.load("earned_conclusion_demo")

        manifest_path = packet_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["title"] = "Rebuilt review packet"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        new_bundle_digest = json_digest(
            {
                "manifest": {
                    **original_packet.manifest,
                    "title": manifest["title"],
                },
                "items": original_packet.items,
            }
        )
        ready_path = packet_dir / "READY.json"
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        ready["review_bundle_digest"] = new_bundle_digest
        ready_path.write_text(json.dumps(ready), encoding="utf-8")
        key_path = packet_dir / "blind_key.json"
        key = json.loads(key_path.read_text(encoding="utf-8"))
        key["_meta"]["review_bundle_digest"] = new_bundle_digest
        key_path.write_text(json.dumps(key), encoding="utf-8")

        second = manager.create_session(
            eval_set_id="earned_conclusion_demo", rater_id="same-rater"
        )
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertEqual(second["review_bundle_digest"], new_bundle_digest)
        with self.assertRaises(ReviewConflictError):
            manager.session_detail(str(first["session_id"]))

        key["_meta"]["seed"] += 1
        key_path.write_text(json.dumps(key), encoding="utf-8")
        third = manager.create_session(
            eval_set_id="earned_conclusion_demo", rater_id="same-rater"
        )
        self.assertNotEqual(second["session_id"], third["session_id"])
        repeated = manager.create_session(
            eval_set_id="earned_conclusion_demo", rater_id="same-rater"
        )
        self.assertEqual(third["session_id"], repeated["session_id"])
        self.assertEqual(len(list(self.sessions_dir.glob("*/session.json"))), 3)

    def test_submission_is_idempotent_and_corrections_supersede(self) -> None:
        session = self.create_session()
        session_id = str(session["session_id"])
        item_id = str(session["items"][0]["item_id"])
        payload = response_payload()
        first = self.manager.submit_judgment(
            session_id=session_id,
            item_id=item_id,
            submission_id="submit-001",
            **payload,
        )
        event_path = self.sessions_dir / session_id / "judgments.jsonl"
        line_count = len(event_path.read_text(encoding="utf-8").splitlines())
        repeated = self.manager.submit_judgment(
            session_id=session_id,
            item_id=item_id,
            submission_id="submit-001",
            **payload,
        )
        self.assertEqual(first["responses"], repeated["responses"])
        self.assertEqual(
            line_count,
            len(event_path.read_text(encoding="utf-8").splitlines()),
        )
        changed = response_payload("B")
        corrected = self.manager.submit_judgment(
            session_id=session_id,
            item_id=item_id,
            submission_id="submit-002",
            **changed,
        )
        self.assertEqual(corrected["responses"][item_id]["revision"], 2)
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        judgments = [
            event for event in events if event["event_type"] == "judgment_submitted"
        ]
        self.assertEqual(judgments[1]["supersedes"], judgments[0]["event_id"])
        with self.assertRaises(ReviewConflictError):
            self.manager.submit_judgment(
                session_id=session_id,
                item_id=item_id,
                submission_id="submit-002",
                **payload,
            )

    def test_seal_requires_completion_then_unblinds_once(self) -> None:
        session = self.create_session()
        session_id = str(session["session_id"])
        with self.assertRaises(ReviewValidationError):
            self.manager.seal_session(session_id)
        for index, item in enumerate(session["items"]):
            item_id = str(item["item_id"])
            if index == 1:
                payload = {
                    "answers": {},
                    "confidence": "low",
                    "evidence": "The visible dialogue does not distinguish the two timings.",
                    "counterevidence": "",
                    "abstain": True,
                }
            else:
                payload = response_payload("A")
            self.manager.submit_judgment(
                session_id=session_id,
                item_id=item_id,
                submission_id=f"submit-{index}",
                **payload,
            )
        sealed = self.manager.seal_session(session_id)
        self.assertIsNotNone(sealed["sealed_at"])
        self.assertNotIn("results", sealed)
        results = json.loads(
            (self.sessions_dir / session_id / "unblinded_results.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(results["judgments"]), 3)
        for summary in results["question_summaries"].values():
            self.assertEqual(sum(summary["wins"].values()), 2)
            self.assertEqual(summary["abstentions"], 1)
        receipt = sealed["seal_receipt"]
        for judgment in results["judgments"]:
            self.assertEqual(
                set(judgment["presentation_display_to_packet"]), {"A", "B"}
            )
            self.assertEqual(set(judgment["packet_label_to_arm"]), {"A", "B"})
        self.assertEqual(len(receipt["receipt_digest"]), 64)
        self.assertEqual(len(receipt["results_digest"]), 64)
        with self.assertRaises(ReviewConflictError):
            self.manager.submit_judgment(
                session_id=session_id,
                item_id=str(session["items"][0]["item_id"]),
                submission_id="after-seal",
                **response_payload(),
            )
        reloaded = BlindReviewManager(
            eval_sets_dir=FIXTURE_ROOT,
            sessions_dir=self.sessions_dir,
        ).session_detail(session_id)
        self.assertEqual(
            reloaded["seal_receipt"]["receipt_digest"],
            receipt["receipt_digest"],
        )
        self.assertNotIn("results", reloaded)

    def test_session_assignment_tampering_is_rejected(self) -> None:
        for field, replacement in (
            ("item_order", []),
            ("presentation", {}),
        ):
            with self.subTest(field=field):
                session = self.create_session(f"tamper-{field}")
                session_path = (
                    self.sessions_dir / str(session["session_id"]) / "session.json"
                )
                stored = json.loads(session_path.read_text(encoding="utf-8"))
                stored[field] = replacement
                session_path.write_text(json.dumps(stored), encoding="utf-8")
                with self.assertRaises(ReviewConflictError):
                    self.manager.session_detail(str(session["session_id"]))

    def test_torn_final_event_is_ignored_during_recovery(self) -> None:
        session = self.create_session("torn-tail")
        session_id = str(session["session_id"])
        event_path = self.sessions_dir / session_id / "judgments.jsonl"
        with event_path.open("a", encoding="utf-8") as stream:
            stream.write('{"event_type":"judgment_submitted"')
        recovered = self.manager.session_detail(session_id)
        self.assertEqual(recovered["completed_count"], 0)
        item_id = str(recovered["items"][0]["item_id"])
        saved = self.manager.submit_judgment(
            session_id=session_id,
            item_id=item_id,
            submission_id="after-torn-tail",
            **response_payload(),
        )
        self.assertEqual(saved["completed_count"], 1)
        reloaded = BlindReviewManager(
            eval_sets_dir=FIXTURE_ROOT,
            sessions_dir=self.sessions_dir,
        ).session_detail(session_id)
        self.assertEqual(reloaded["completed_count"], 1)
        recovered_events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        recovery = [
            event
            for event in recovered_events
            if event["event_type"] == "log_tail_recovered"
        ]
        self.assertEqual(len(recovery), 1)
        quarantine = event_path.parent / recovery[0]["payload"]["quarantine_file"]
        self.assertTrue(quarantine.exists())
        self.assertEqual(
            recovery[0]["payload"]["sha256"],
            hashlib.sha256(quarantine.read_bytes()).hexdigest(),
        )

    def test_seal_repairs_a_torn_tail_before_hashing_the_ledger(self) -> None:
        session = self.create_session("torn-before-seal")
        session_id = str(session["session_id"])
        for index, item in enumerate(session["items"]):
            self.manager.submit_judgment(
                session_id=session_id,
                item_id=str(item["item_id"]),
                submission_id=f"complete-before-torn-{index}",
                **response_payload(),
            )
        event_path = self.sessions_dir / session_id / "judgments.jsonl"
        with event_path.open("a", encoding="utf-8") as stream:
            stream.write('{"event_type":"interrupted-seal-boundary"')
        sealed = self.manager.seal_session(session_id)
        self.assertIsNotNone(sealed["sealed_at"])
        self.assertEqual(
            self.manager.session_detail(session_id)["seal_receipt"]["receipt_digest"],
            sealed["seal_receipt"]["receipt_digest"],
        )
        event_types = [
            json.loads(line)["event_type"]
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(event_types[-2:], ["log_tail_recovered", "review_sealed"])

    def test_create_can_retry_after_snapshot_write_failure(self) -> None:
        with mock.patch.object(
            self.manager,
            "_save_session",
            side_effect=OSError("injected snapshot failure"),
        ):
            with self.assertRaises(OSError):
                self.create_session("retry-after-failure")
        recovered = self.create_session("retry-after-failure")
        self.assertEqual(recovered["rater_id"], "retry-after-failure")
        self.assertEqual(recovered["completed_count"], 0)

    def test_sealed_artifact_tampering_is_rejected(self) -> None:
        for artifact_name in ("seal_receipt.json", "unblinded_results.json"):
            with self.subTest(artifact=artifact_name):
                session = self.create_session(f"artifact-{artifact_name}")
                session_id = str(session["session_id"])
                for index, item in enumerate(session["items"]):
                    self.manager.submit_judgment(
                        session_id=session_id,
                        item_id=str(item["item_id"]),
                        submission_id=f"artifact-{artifact_name}-{index}",
                        **response_payload(),
                    )
                self.manager.seal_session(session_id)
                artifact_path = self.sessions_dir / session_id / artifact_name
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                artifact["rater_id"] = "tampered-rater"
                artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
                with self.assertRaises(ReviewConflictError):
                    self.manager.session_detail(session_id)

    def test_catalog_rejects_incomplete_or_degenerate_blind_key(self) -> None:
        eval_sets_dir = Path(self.temporary.name) / "bad_key_sets"
        fixture_copy = eval_sets_dir / "earned_conclusion_demo"
        shutil.copytree(FIXTURE_ROOT / "earned_conclusion_demo", fixture_copy)
        key_path = fixture_copy / "blind_key.json"
        key = json.loads(key_path.read_text(encoding="utf-8"))
        key["item_001"]["label_to_arm"] = {"A": "same", "B": "same"}
        key_path.write_text(json.dumps(key), encoding="utf-8")
        with self.assertRaises(ReviewValidationError):
            ReviewSetCatalog(eval_sets_dir).load("earned_conclusion_demo")

    def test_seal_rejects_a_blind_key_changed_after_session_creation(self) -> None:
        eval_sets_dir = Path(self.temporary.name) / "eval_sets"
        fixture_copy = eval_sets_dir / "earned_conclusion_demo"
        shutil.copytree(FIXTURE_ROOT / "earned_conclusion_demo", fixture_copy)
        manager = BlindReviewManager(
            eval_sets_dir=eval_sets_dir,
            sessions_dir=Path(self.temporary.name) / "tamper_sessions",
        )
        session = manager.create_session(
            eval_set_id="earned_conclusion_demo",
            rater_id="tamper-test",
        )
        for index, item in enumerate(session["items"]):
            manager.submit_judgment(
                session_id=session["session_id"],
                item_id=item["item_id"],
                submission_id=f"tamper-submit-{index}",
                **response_payload(),
            )
        key_path = fixture_copy / "blind_key.json"
        key = json.loads(key_path.read_text(encoding="utf-8"))
        key["item_001"]["label_to_arm"]["A"] = "tampered-arm"
        key_path.write_text(json.dumps(key), encoding="utf-8")
        with self.assertRaises(ReviewConflictError):
            manager.seal_session(session["session_id"])

    def test_abstain_needs_a_reason_and_cannot_carry_answers(self) -> None:
        session = self.create_session()
        session_id = str(session["session_id"])
        item_id = str(session["items"][0]["item_id"])
        with self.assertRaises(ReviewValidationError):
            self.manager.submit_judgment(
                session_id=session_id,
                item_id=item_id,
                submission_id="bad-abstain",
                answers={"overall_better": "A"},
                confidence="low",
                evidence="Cannot judge.",
                abstain=True,
            )
        with self.assertRaises(ReviewValidationError):
            self.manager.submit_judgment(
                session_id=session_id,
                item_id=item_id,
                submission_id="empty-reason",
                answers={},
                confidence="low",
                evidence="",
                abstain=True,
            )


if __name__ == "__main__":
    unittest.main()
