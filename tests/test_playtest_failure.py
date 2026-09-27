import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from playtest_server import (
    CreateSessionRequest,
    SessionManager,
    TurnRequest,
    build_app,
    failed_event_receipts,
    generation_config_from_dict,
    json_ready,
    serialize_session_state,
)


class PlaytestFailureTests(unittest.TestCase):
    def create_manager_and_session(self, sessions_dir: Path):
        manager = SessionManager(sessions_dir)
        detail = manager.create_session(
            CreateSessionRequest(
                provider="dummy",
                model="dummy-v1",
                script_id="free_chat",
                arm_preset="static",
                semantic_judge_backend="llm",
            )
        )
        record = manager.get_record(detail["session_id"])
        config = record.session.config
        config.memory_every = 1
        config.conclusion_every = 0
        config.delayed_mention_every = 0
        config.latent_convergence_every = 0
        config.deferred_intent_every = 0
        return manager, record

    def fail_after_memory_probe(self, record):
        adapter = record.session.adapter
        original_generate = adapter.generate
        calls = 0

        def generate_then_fail(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated reply failure sk-test-secret")
            return original_generate(**kwargs)

        adapter.generate = generate_then_fail
        return original_generate

    def test_failed_turn_restores_state_and_log_but_keeps_draft(self):
        with TemporaryDirectory() as temp_dir:
            sessions_dir = Path(temp_dir)
            manager, record = self.create_manager_and_session(sessions_dir)
            session_id = record.session_id
            original_generate = self.fail_after_memory_probe(record)
            before = serialize_session_state(record.session)
            first_attempt = "first attempt"

            with self.assertRaisesRegex(RuntimeError, "simulated reply failure"):
                manager.append_turn(session_id, first_attempt)

            self.assertEqual(serialize_session_state(record.session), before)
            self.assertEqual(record.pending_user_text, first_attempt)
            self.assertEqual(record.last_error, "RuntimeError: turn failed")
            self.assertFalse(record.log_path.exists())
            self.assertNotIn(
                "sk-test-secret",
                manager._session_json_path(session_id).read_text(encoding="utf-8"),
            )
            audit_path = record.storage_dir / "failed_attempts.jsonl"
            audit_raw = audit_path.read_text(encoding="utf-8")
            self.assertNotIn("sk-test-secret", audit_raw)
            audit = json.loads(audit_raw.splitlines()[0])
            self.assertEqual(audit["attempt_turn_index"], 1)
            self.assertEqual(audit["error_type"], "RuntimeError")
            self.assertGreaterEqual(audit["discarded_event_count"], 1)
            probe_receipt = next(
                event
                for event in audit["discarded_events"]
                if event.get("event_type") == "memory_capsule"
            )
            self.assertEqual(probe_receipt["event_type"], "memory_capsule")
            self.assertTrue(probe_receipt["payload"]["request_id"].startswith("dummy-"))
            self.assertEqual(
                probe_receipt["payload"]["usage"],
                {"input_tokens": 0, "output_tokens": 0},
            )
            self.assertNotIn("capsule", probe_receipt["payload"])

            reloaded = SessionManager(sessions_dir)
            saved = reloaded.get_record(session_id)
            self.assertEqual(serialize_session_state(saved.session), before)
            self.assertEqual(saved.pending_user_text, first_attempt)

            record.session.adapter.generate = original_generate
            manager.append_turn(session_id, first_attempt)
            self.assertEqual(record.session.turn_index, 1)
            self.assertEqual([item.role for item in record.session.history], ["user", "assistant"])
            self.assertEqual(record.pending_user_text, "")
            previous_log = record.log_path.read_bytes()

            original_generate = self.fail_after_memory_probe(record)
            before = serialize_session_state(record.session)
            with self.assertRaisesRegex(RuntimeError, "simulated reply failure"):
                manager.append_turn(session_id, "second attempt")

            self.assertEqual(serialize_session_state(record.session), before)
            self.assertEqual(record.log_path.read_bytes(), previous_log)
            self.assertEqual(record.pending_user_text, "second attempt")
            audits = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(audits), 2)
            self.assertEqual(audits[1]["attempt_turn_index"], 2)
            self.assertGreaterEqual(audits[1]["discarded_event_count"], 1)
            record.session.adapter.generate = original_generate

    def test_resume_preserves_explicit_zero_config_values(self):
        with TemporaryDirectory() as temp_dir:
            sessions_dir = Path(temp_dir)
            manager, record = self.create_manager_and_session(sessions_dir)
            config = record.session.config
            config.memory_every = 0
            config.memory_capsule_limit = 0
            config.memory_word_budget = 0
            config.delayed_mention_min_nonconclusion_items = 0
            config.delayed_mention_fire_prob = 0.0
            config.delayed_mention_fire_max_items = 0
            config.delayed_mention_leak_threshold = 0.0
            config.deferred_intent_offset = 0
            config.deferred_intent_grace = 0
            config.deferred_intent_plan_max_new = 0
            config.reply_config.temperature = 0.0
            manager._save_record(record)

            saved_config = json.loads(
                manager._session_json_path(record.session_id).read_text(encoding="utf-8")
            )["config"]
            resumed = SessionManager(sessions_dir).get_record(record.session_id).session.config
            self.assertEqual(json_ready(resumed), saved_config)

    def test_resumed_zero_memory_limits_skip_capsule_generation(self):
        for zero_field in ("memory_capsule_limit", "memory_word_budget"):
            with self.subTest(zero_field=zero_field), TemporaryDirectory() as temp_dir:
                sessions_dir = Path(temp_dir)
                manager, record = self.create_manager_and_session(sessions_dir)
                setattr(record.session.config, zero_field, 0)
                manager._save_record(record)

                resumed_manager = SessionManager(sessions_dir)
                resumed_record = resumed_manager.get_record(record.session_id)
                self.assertEqual(getattr(resumed_record.session.config, zero_field), 0)
                result = resumed_manager.append_turn(record.session_id, "one turn")
                self.assertEqual(result["turn_index"], 1)
                self.assertEqual(resumed_record.session.memory_capsules, [])
                events = [
                    json.loads(line)
                    for line in resumed_record.log_path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertNotIn("memory_capsule", [event["event_type"] for event in events])

    def test_generation_config_rejects_nonpositive_provider_limits(self):
        self.assertEqual(generation_config_from_dict({"temperature": 0.0}).temperature, 0.0)
        for field in ("max_tokens", "timeout_seconds"):
            for value in (0, -1):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, f"{field} must be positive"):
                        generation_config_from_dict({field: value})

    def test_failed_event_receipt_excludes_body_and_secret_fields(self):
        raw_event = json.dumps(
            {
                "timestamp": 1.0,
                "turn_index": 1,
                "event_type": "memory_capsule",
                "payload": {
                    "capsule": "user included sk-test-secret",
                    "request_id": "sk-test-secret",
                    "usage": {
                        "input_tokens": 5,
                        "api_key": "sk-test-secret",
                        "api_token_count": 12345,
                    },
                },
            }
        ).encode("utf-8")

        receipt = failed_event_receipts(raw_event)[0]
        self.assertEqual(receipt["payload"], {"usage": {"input_tokens": 5}})
        self.assertNotIn("sk-test-secret", json.dumps(receipt))

    def test_turn_endpoint_hides_adapter_error_text(self):
        with TemporaryDirectory() as temp_dir:
            app = build_app(sessions_dir=Path(temp_dir), allowed_origins=[])
            endpoint = next(
                route.endpoint
                for route in app.routes
                if getattr(route, "path", None) == "/api/sessions/{session_id}/turn"
                and "POST" in getattr(route, "methods", set())
            )
            cases = (
                (
                    RuntimeError(
                        "HTTP 429 from https://provider.example; Request payload: sk-test-secret"
                    ),
                    "RuntimeError: provider HTTP 429",
                ),
                (ValueError("invalid provider response sk-test-secret"), "ValueError: turn failed"),
            )
            for error, expected_detail in cases:
                with self.subTest(error_type=type(error).__name__):
                    with patch.object(SessionManager, "append_turn", side_effect=error):
                        with self.assertRaises(HTTPException) as caught:
                            endpoint("example", TurnRequest(user_text="hello"))
                    public_error = caught.exception
                    self.assertEqual(public_error.status_code, 500)
                    self.assertEqual(public_error.detail, expected_detail)
                    self.assertTrue(public_error.__suppress_context__)
                    self.assertNotIn("sk-test-secret", str(public_error.detail))


if __name__ == "__main__":
    unittest.main()
