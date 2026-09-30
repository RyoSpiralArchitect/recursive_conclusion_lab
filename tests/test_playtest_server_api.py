# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from playtest_server import build_app


ROOT_DIR = Path(__file__).resolve().parents[1]
FIXTURE_SET_DIR = ROOT_DIR / "examples" / "human_eval_sets" / "earned_conclusion_demo"


class PlaytestServerApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.playtest_sessions_dir = self.root / "playtest_sessions"
        self.eval_sets_dir = self.root / "eval_sets"
        self.review_sessions_dir = self.root / "review_sessions"
        shutil.copytree(FIXTURE_SET_DIR, self.eval_sets_dir / "demo")
        self.client = TestClient(
            build_app(
                sessions_dir=self.playtest_sessions_dir,
                allowed_origins=[],
                eval_sets_dir=self.eval_sets_dir,
                review_sessions_dir=self.review_sessions_dir,
            )
        )

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def create_review_session(
        self, rater_id: str = "api-reviewer"
    ) -> dict[str, object]:
        response = self.client.post(
            "/api/review/sessions",
            json={"eval_set_id": "demo", "rater_id": rater_id},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    @staticmethod
    def judgment_payload(
        item: dict[str, object], choice: str = "A"
    ) -> dict[str, object]:
        questions = item.get("questions") or []
        return {
            "answers": {str(question["id"]): choice for question in questions},
            "confidence": "medium",
            "evidence": "The selected response earns its conclusion from the visible context.",
            "counterevidence": "The other response is shorter.",
            "abstain": False,
        }

    def test_legacy_playtest_dummy_flow_and_restart(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"ok": True, "workspace_mode": "full"})

        options = self.client.get("/api/options")
        self.assertEqual(options.status_code, 200)
        self.assertIn("dummy", options.json()["providers"])
        self.assertEqual(
            options.json()["providers"],
            ["openai", "anthropic", "mistral", "gemini", "hf", "dummy"],
        )
        gpt56_profile = next(
            profile
            for profile in options.json()["model_profiles"]
            if profile["id"] == "openai.gpt-5.6.v1"
        )
        self.assertEqual(
            gpt56_profile["models"],
            ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
        )

        created = self.client.post(
            "/api/sessions",
            json={
                "title": "Offline API smoke",
                "provider": "dummy",
                "model": "dummy",
                "observer_provider": "dummy",
                "observer_model": "dummy",
                "embedding_provider": "dummy",
                "embedding_model": "hash-128",
                "script_id": "free_chat",
                "arm_preset": "static",
                "semantic_judge_backend": "both",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        session_id = created.json()["session_id"]

        notes = self.client.put(
            f"/api/sessions/{session_id}/notes",
            json={"notes": "Observe whether the answer commits too early."},
        )
        self.assertEqual(notes.status_code, 200, notes.text)
        self.assertEqual(
            notes.json()["notes"],
            "Observe whether the answer commits too early.",
        )

        turn = self.client.post(
            f"/api/sessions/{session_id}/turn",
            json={"user_text": "Help me choose between the two options."},
        )
        self.assertEqual(turn.status_code, 200, turn.text)
        self.assertEqual(turn.json()["turn_index"], 1)
        self.assertEqual(turn.json()["last_error"], "")
        self.assertGreaterEqual(len(turn.json()["history"]), 2)

        restarted_client = TestClient(
            build_app(
                sessions_dir=self.playtest_sessions_dir,
                allowed_origins=[],
                eval_sets_dir=self.eval_sets_dir,
                review_sessions_dir=self.review_sessions_dir,
            )
        )
        try:
            restored = restarted_client.get(f"/api/sessions/{session_id}")
            self.assertEqual(restored.status_code, 200, restored.text)
            self.assertEqual(restored.json()["turn_index"], 1)
            self.assertEqual(
                restored.json()["notes"],
                "Observe whether the answer commits too early.",
            )
        finally:
            restarted_client.close()

    def test_gpt56_generation_config_survives_keyless_restart(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            created = self.client.post(
                "/api/sessions",
                json={
                    "title": "GPT-5.6 offline restart",
                    "provider": "openai",
                    "model": "gpt-5.6-terra",
                    "observer_provider": "openai",
                    "observer_model": "gpt-5.6-luna",
                    "script_id": "free_chat",
                    "arm_preset": "static",
                    "semantic_judge_backend": "off",
                    "reasoning_effort": "medium",
                    "reasoning_mode": "pro",
                    "reasoning_context": "all_turns",
                    "text_verbosity": "high",
                    "probe_reasoning_effort": "none",
                    "probe_reasoning_mode": "standard",
                    "probe_reasoning_context": "current_turn",
                    "probe_text_verbosity": "low",
                },
            )
            self.assertEqual(created.status_code, 200, created.text)
            created_payload = created.json()
            session_id = created_payload["session_id"]
            saved_payload = json.loads(
                (self.playtest_sessions_dir / session_id / "session.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved_payload["version"], 2)

            reply_config = created_payload["config"]["reply_config"]
            self.assertEqual(reply_config["temperature"], 0.2)
            self.assertEqual(reply_config["reasoning_effort"], "medium")
            self.assertEqual(reply_config["reasoning_mode"], "pro")
            self.assertEqual(reply_config["reasoning_context"], "all_turns")
            self.assertEqual(reply_config["text_verbosity"], "high")
            self.assertEqual(reply_config["model_profile_id"], "openai.gpt-5.6.v1")
            self.assertEqual(reply_config["model_profile_version"], 1)

            probe_config = created_payload["config"]["probe_config"]
            self.assertEqual(probe_config["temperature"], 0.0)
            self.assertEqual(probe_config["reasoning_effort"], "none")
            self.assertEqual(probe_config["reasoning_mode"], "standard")
            self.assertEqual(probe_config["reasoning_context"], "current_turn")
            self.assertEqual(probe_config["text_verbosity"], "low")
            self.assertEqual(probe_config["model_profile_id"], "openai.gpt-5.6.v1")
            self.assertEqual(probe_config["model_profile_version"], 1)

            observer_config = created_payload["config"]["observer_config"]
            self.assertEqual(observer_config, probe_config)

            restarted_client = TestClient(
                build_app(
                    sessions_dir=self.playtest_sessions_dir,
                    allowed_origins=[],
                    eval_sets_dir=self.eval_sets_dir,
                    review_sessions_dir=self.review_sessions_dir,
                )
            )
            try:
                restored = restarted_client.get(f"/api/sessions/{session_id}")
                self.assertEqual(restored.status_code, 200, restored.text)
                restored_payload = restored.json()
                self.assertEqual(restored_payload["provider"], "openai")
                self.assertEqual(restored_payload["model"], "gpt-5.6-terra")
                self.assertEqual(
                    restored_payload["config"]["reply_config"], reply_config
                )
                self.assertEqual(
                    restored_payload["config"]["probe_config"], probe_config
                )
                self.assertEqual(
                    restored_payload["config"]["observer_config"], observer_config
                )
            finally:
                restarted_client.close()

    def test_cross_provider_observer_gets_its_own_auto_profile(self) -> None:
        created = self.client.post(
            "/api/sessions",
            json={
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "observer_provider": "anthropic",
                "observer_model": "claude-offline",
                "script_id": "free_chat",
                "arm_preset": "static",
                "semantic_judge_backend": "off",
                "probe_reasoning_effort": "medium",
                "probe_reasoning_mode": "pro",
                "probe_reasoning_context": "all_turns",
                "probe_text_verbosity": "high",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        config = created.json()["config"]
        self.assertEqual(config["probe_config"]["reasoning_effort"], "medium")
        observer_config = config["observer_config"]
        self.assertEqual(observer_config["model_profile_id"], "anthropic.default")
        self.assertIsNone(observer_config["reasoning_effort"])
        self.assertIsNone(observer_config["reasoning_mode"])
        self.assertIsNone(observer_config["reasoning_context"])
        self.assertIsNone(observer_config["text_verbosity"])

    def test_legacy_session_is_migrated_to_durable_profile_pins(self) -> None:
        created = self.client.post(
            "/api/sessions",
            json={
                "provider": "openai",
                "model": "gpt-4.1-mini-2025-04-14",
                "script_id": "free_chat",
                "arm_preset": "static",
                "semantic_judge_backend": "off",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        session_id = created.json()["session_id"]
        session_json = self.playtest_sessions_dir / session_id / "session.json"
        legacy = json.loads(session_json.read_text(encoding="utf-8"))
        legacy["version"] = 1
        legacy["config"].pop("observer_config", None)
        for label in ("reply_config", "probe_config"):
            legacy["config"][label].pop("model_profile_id", None)
            legacy["config"][label].pop("model_profile_version", None)
            legacy["config"][label]["reasoning_effort"] = None
            legacy["config"][label]["reasoning_mode"] = None
            legacy["config"][label]["reasoning_context"] = None
        session_json.write_text(
            json.dumps(legacy, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        restarted_client = TestClient(
            build_app(
                sessions_dir=self.playtest_sessions_dir,
                allowed_origins=[],
                eval_sets_dir=self.eval_sets_dir,
                review_sessions_dir=self.review_sessions_dir,
            )
        )
        try:
            restored = restarted_client.get(f"/api/sessions/{session_id}")
            self.assertEqual(restored.status_code, 200, restored.text)
        finally:
            restarted_client.close()

        migrated = json.loads(session_json.read_text(encoding="utf-8"))
        self.assertEqual(migrated["version"], 2)
        for label in ("reply_config", "probe_config", "observer_config"):
            self.assertEqual(
                migrated["config"][label]["model_profile_id"],
                "openai.default",
            )
            self.assertEqual(
                migrated["config"][label]["model_profile_version"],
                1,
            )

    def test_unpinned_legacy_exact_profile_is_not_inferred(self) -> None:
        created = self.client.post(
            "/api/sessions",
            json={
                "provider": "openai",
                "model": "gpt-5.6-terra",
                "script_id": "free_chat",
                "arm_preset": "static",
                "semantic_judge_backend": "off",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        session_id = created.json()["session_id"]
        session_json = self.playtest_sessions_dir / session_id / "session.json"
        legacy = json.loads(session_json.read_text(encoding="utf-8"))
        legacy["version"] = 1
        legacy["config"].pop("observer_config", None)
        for label in ("reply_config", "probe_config"):
            legacy["config"][label].pop("model_profile_id", None)
            legacy["config"][label].pop("model_profile_version", None)
            legacy["config"][label]["reasoning_effort"] = None
            legacy["config"][label]["reasoning_mode"] = None
            legacy["config"][label]["reasoning_context"] = None
        session_json.write_text(
            json.dumps(legacy, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        restarted_client = TestClient(
            build_app(
                sessions_dir=self.playtest_sessions_dir,
                allowed_origins=[],
                eval_sets_dir=self.eval_sets_dir,
                review_sessions_dir=self.review_sessions_dir,
            )
        )
        try:
            listing = restarted_client.get("/api/sessions")
            self.assertEqual(listing.status_code, 200, listing.text)
            self.assertEqual(listing.json()["sessions"], [])
            self.assertEqual(
                listing.json()["load_errors"][0]["code"],
                "legacy_profile_ambiguous",
            )
            self.assertNotIn("gpt-5.6-terra", listing.text)
        finally:
            restarted_client.close()

        self.assertEqual(
            json.loads(session_json.read_text(encoding="utf-8"))["version"],
            1,
        )

    def test_already_pinned_v1_exact_profile_migrates(self) -> None:
        created = self.client.post(
            "/api/sessions",
            json={
                "provider": "openai",
                "model": "gpt-5.6-luna",
                "script_id": "free_chat",
                "arm_preset": "static",
                "semantic_judge_backend": "off",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        session_id = created.json()["session_id"]
        session_json = self.playtest_sessions_dir / session_id / "session.json"
        pinned_v1 = json.loads(session_json.read_text(encoding="utf-8"))
        pinned_v1["version"] = 1
        session_json.write_text(
            json.dumps(pinned_v1, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        restarted_client = TestClient(
            build_app(
                sessions_dir=self.playtest_sessions_dir,
                allowed_origins=[],
                eval_sets_dir=self.eval_sets_dir,
                review_sessions_dir=self.review_sessions_dir,
            )
        )
        try:
            restored = restarted_client.get(f"/api/sessions/{session_id}")
            self.assertEqual(restored.status_code, 200, restored.text)
        finally:
            restarted_client.close()
        self.assertEqual(
            json.loads(session_json.read_text(encoding="utf-8"))["version"],
            2,
        )

    def test_profile_version_mismatch_is_fail_closed_and_reported(self) -> None:
        created = self.client.post(
            "/api/sessions",
            json={
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "script_id": "free_chat",
                "arm_preset": "static",
                "semantic_judge_backend": "off",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        session_id = created.json()["session_id"]
        session_json = self.playtest_sessions_dir / session_id / "session.json"
        payload = json.loads(session_json.read_text(encoding="utf-8"))
        payload["config"]["reply_config"]["model_profile_version"] = 999
        session_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        restarted_client = TestClient(
            build_app(
                sessions_dir=self.playtest_sessions_dir,
                allowed_origins=[],
                eval_sets_dir=self.eval_sets_dir,
                review_sessions_dir=self.review_sessions_dir,
            )
        )
        try:
            listing = restarted_client.get("/api/sessions")
            self.assertEqual(listing.status_code, 200, listing.text)
            self.assertEqual(listing.json()["sessions"], [])
            self.assertEqual(listing.json()["load_error_count"], 1)
            self.assertEqual(listing.json()["load_errors"][0]["session_id"], session_id)
            self.assertEqual(
                listing.json()["load_errors"][0]["code"],
                "model_profile_mismatch",
            )
            self.assertNotIn("999", listing.text)
            self.assertEqual(
                restarted_client.get(f"/api/sessions/{session_id}").status_code,
                404,
            )
        finally:
            restarted_client.close()

    def test_blind_review_complete_flow_and_idempotent_seal(self) -> None:
        sets_response = self.client.get("/api/review/sets")
        self.assertEqual(sets_response.status_code, 200, sets_response.text)
        self.assertEqual(sets_response.json()["sets"][0]["eval_set_id"], "demo")

        created = self.create_review_session()
        session_id = str(created["session_id"])

        fetched = self.client.get(f"/api/review/sessions/{session_id}")
        self.assertEqual(fetched.status_code, 200, fetched.text)
        self.assertEqual(fetched.json()["completed_count"], 0)

        current = fetched.json()
        for index, item in enumerate(current["items"]):
            submitted = self.client.put(
                f"/api/review/sessions/{session_id}/items/{item['item_id']}",
                json={
                    "submission_id": f"submission-{index}",
                    **self.judgment_payload(item),
                },
            )
            self.assertEqual(submitted.status_code, 200, submitted.text)
            self.assertEqual(submitted.json()["completed_count"], index + 1)

        sealed = self.client.post(f"/api/review/sessions/{session_id}/seal")
        self.assertEqual(sealed.status_code, 200, sealed.text)
        sealed_payload = sealed.json()
        self.assertIsNotNone(sealed_payload["sealed_at"])
        self.assertNotIn("results", sealed_payload)
        result_path = self.review_sessions_dir / session_id / "unblinded_results.json"
        self.assertTrue(result_path.exists())
        receipt_digest = sealed_payload["seal_receipt"]["receipt_digest"]

        repeated = self.client.post(f"/api/review/sessions/{session_id}/seal")
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assertEqual(repeated.json()["sealed_at"], sealed_payload["sealed_at"])
        self.assertEqual(
            repeated.json()["seal_receipt"]["receipt_digest"],
            receipt_digest,
        )

    def test_review_route_status_mappings(self) -> None:
        missing = self.client.get("/api/review/sessions/review-does-not-exist")
        self.assertEqual(missing.status_code, 404, missing.text)

        unknown_set = self.client.post(
            "/api/review/sessions",
            json={"eval_set_id": "does-not-exist", "rater_id": "api-reviewer"},
        )
        self.assertEqual(unknown_set.status_code, 404, unknown_set.text)

        invalid_body = self.client.post(
            "/api/review/sessions",
            json={"eval_set_id": "demo"},
        )
        self.assertEqual(invalid_body.status_code, 422, invalid_body.text)

        created = self.create_review_session("status-reviewer")
        session_id = str(created["session_id"])
        item = created["items"][0]
        item_id = str(item["item_id"])

        incomplete_seal = self.client.post(f"/api/review/sessions/{session_id}/seal")
        self.assertEqual(incomplete_seal.status_code, 400, incomplete_seal.text)

        missing_field = self.client.put(
            f"/api/review/sessions/{session_id}/items/{item_id}",
            json={
                "submission_id": "missing-evidence",
                "answers": self.judgment_payload(item)["answers"],
                "confidence": "medium",
            },
        )
        self.assertEqual(missing_field.status_code, 422, missing_field.text)

        first_payload = {
            "submission_id": "reused-submission-id",
            **self.judgment_payload(item, "A"),
        }
        first = self.client.put(
            f"/api/review/sessions/{session_id}/items/{item_id}",
            json=first_payload,
        )
        self.assertEqual(first.status_code, 200, first.text)

        conflicting_payload = {
            "submission_id": "reused-submission-id",
            **self.judgment_payload(item, "B"),
        }
        conflict = self.client.put(
            f"/api/review/sessions/{session_id}/items/{item_id}",
            json=conflicting_payload,
        )
        self.assertEqual(conflict.status_code, 409, conflict.text)

    def test_review_only_mode_closes_playtest_api_boundary(self) -> None:
        review_client = TestClient(
            build_app(
                sessions_dir=self.root / "isolated_playtest_sessions",
                allowed_origins=[],
                eval_sets_dir=self.eval_sets_dir,
                review_sessions_dir=self.root / "isolated_review_sessions",
                workspace_mode="review",
            )
        )
        try:
            self.assertEqual(
                review_client.get("/api/health").json()["workspace_mode"],
                "review",
            )
            self.assertEqual(review_client.get("/api/options").status_code, 404)
            self.assertEqual(review_client.get("/api/sessions").status_code, 404)
            self.assertEqual(review_client.get("/openapi.json").status_code, 404)
            self.assertEqual(review_client.get("/docs").status_code, 404)
            self.assertEqual(review_client.get("/api/review/sets").status_code, 200)
        finally:
            review_client.close()


if __name__ == "__main__":
    unittest.main()
