"""Regression checks for compare artifact isolation using the local dummy adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import recursive_conclusion_lab as rcl


class CompareOutputIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.script = self.root / "script.json"
        self.script.write_text(
            json.dumps({"system": "Answer briefly.", "turns": ["one", "two", "three"]}),
            encoding="utf-8",
        )

    def compare_args(self, out_dir: Path, *providers: str):
        return rcl.build_parser().parse_args(
            [
                "compare",
                "--script",
                str(self.script),
                "--providers",
                *providers,
                "--out-dir",
                str(out_dir),
                "--memory-every",
                "0",
                "--conclusion-every",
                "2",
                "--deferred-intent-every",
                "0",
                "--delayed-mention-every",
                "0",
            ]
        )

    @staticmethod
    def assistant_reply_count(path: Path) -> int:
        return sum(
            json.loads(line).get("event_type") == "assistant_reply"
            for line in path.read_text(encoding="utf-8").splitlines()
        )

    def test_compare_rejects_rerun_without_changing_old_artifacts(self) -> None:
        out_dir = self.root / "compare"
        args = self.compare_args(out_dir, "dummy=dummy-v1")
        self.assertEqual(rcl.run_compare(args), 0)

        log = out_dir / "dummy__dummy-v1.jsonl"
        summary = out_dir / "summary.json"
        self.assertEqual(self.assistant_reply_count(log), 3)
        self.assertEqual(len(json.loads(summary.read_text(encoding="utf-8"))), 3)
        before = {path: path.read_bytes() for path in (log, summary)}

        with mock.patch.object(
            rcl.DummyAdapter, "generate", side_effect=AssertionError("provider was called")
        ):
            with self.assertRaises(rcl.OutputCollisionError):
                rcl.run_compare(args)

        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_sanitized_model_collision_fails_before_provider_call(self) -> None:
        out_dir = self.root / "model_collision"
        args = self.compare_args(out_dir, "dummy=a/b", "dummy=a_b")

        with mock.patch.object(
            rcl.DummyAdapter, "generate", side_effect=AssertionError("provider was called")
        ):
            with self.assertRaises(rcl.OutputCollisionError):
                rcl.run_compare(args)

        self.assertEqual(list(out_dir.rglob("*.json*")), [])

    def test_unsupported_provider_does_not_reserve_outside_output_dir(self) -> None:
        out_dir = self.root / "provider_collision"
        args = self.compare_args(out_dir, "../outside=dummy-v1")

        with self.assertRaisesRegex(ValueError, "Unsupported provider"):
            rcl.run_compare(args)

        self.assertFalse(out_dir.exists())
        self.assertEqual(list(self.root.glob("outside*")), [])

    def test_invalid_optional_adapters_leave_single_compare_unreserved(self) -> None:
        cases = (
            ({"observer_provider": "dummy"}, "--observer-provider and --observer-model"),
            ({"observer_model": "dummy-v1"}, "--observer-provider and --observer-model"),
            (
                {"observer_provider": "unsupported", "observer_model": "model"},
                "Unsupported provider",
            ),
            (
                {"observer_provider": "dummy=bad", "observer_model": "model"},
                "Unsupported provider",
            ),
            ({"embedding_provider": "dummy"}, "--embedding-provider and --embedding-model"),
            ({"embedding_model": "dummy-v1"}, "--embedding-provider and --embedding-model"),
            (
                {"embedding_provider": "unsupported", "embedding_model": "model"},
                "Unsupported embedding provider",
            ),
            ({"semantic_judge_backend": "embedding"}, "semantic_judge_backend requires"),
        )
        for index, (overrides, error) in enumerate(cases):
            with self.subTest(overrides=overrides):
                out_dir = self.root / f"invalid_single_{index}"
                args = self.compare_args(out_dir, "dummy=dummy-v1")
                for name, value in overrides.items():
                    setattr(args, name, value)
                with mock.patch.object(rcl.DummyAdapter, "generate") as generate:
                    with self.assertRaisesRegex(ValueError, error):
                        rcl.run_compare(args)
                generate.assert_not_called()
                self.assertFalse(out_dir.exists())

    def test_direct_compare_preflights_before_reserving_logs(self) -> None:
        out_dir = self.root / "invalid_direct"
        args = self.compare_args(out_dir, "dummy=dummy-v1")
        args.embedding_provider = "dummy"

        with mock.patch.object(rcl.DummyAdapter, "generate") as generate:
            with self.assertRaisesRegex(ValueError, "--embedding-provider and --embedding-model"):
                rcl.execute_compare(args)

        generate.assert_not_called()
        self.assertFalse(out_dir.exists())

    def matrix_config(self, out_dir: Path, arm_names: tuple[str, str]) -> dict:
        return {
            "script": str(self.script),
            "providers": ["dummy=dummy-v1"],
            "out_dir": str(out_dir),
            "repeats": 2,
            "seed": 7,
            "args": {
                "memory_every": 0,
                "conclusion_every": 2,
                "deferred_intent_every": 0,
                "delayed_mention_every": 0,
            },
            "arms": [
                {"name": arm_names[0], "args": {"conclusion_mode": "observe"}},
                {"name": arm_names[1], "args": {"conclusion_mode": "soft_steer"}},
            ],
        }

    def test_sanitized_arm_collision_fails_before_provider_call(self) -> None:
        out_dir = self.root / "arm_collision"
        config = self.matrix_config(out_dir, ("arm/a", "arm_a"))

        with mock.patch.object(
            rcl.DummyAdapter, "generate", side_effect=AssertionError("provider was called")
        ):
            with self.assertRaises(rcl.OutputCollisionError):
                rcl.run_compare_matrix_from_config_data(config)

        self.assertEqual(list(out_dir.rglob("*.json*")), [])

    def test_invalid_later_matrix_arm_leaves_all_outputs_unreserved(self) -> None:
        cases = (
            ({"observer_provider": "dummy"}, "--observer-provider and --observer-model"),
            (
                {"observer_provider": "unsupported", "observer_model": "model"},
                "Unsupported provider",
            ),
            (
                {"observer_provider": "dummy=bad", "observer_model": "model"},
                "Unsupported provider",
            ),
            ({"embedding_provider": "dummy"}, "--embedding-provider and --embedding-model"),
            (
                {"embedding_provider": "unsupported", "embedding_model": "model"},
                "Unsupported embedding provider",
            ),
            ({"semantic_judge_backend": "both"}, "semantic_judge_backend requires"),
        )
        for index, (overrides, error) in enumerate(cases):
            with self.subTest(overrides=overrides):
                out_dir = self.root / f"invalid_matrix_{index}"
                config = self.matrix_config(out_dir, ("valid", "invalid"))
                config["arms"][1]["args"].update(overrides)
                with mock.patch.object(rcl.DummyAdapter, "generate") as generate:
                    with self.assertRaisesRegex(ValueError, error):
                        rcl.run_compare_matrix_from_config_data(config)
                generate.assert_not_called()
                self.assertFalse(out_dir.exists())

    def test_valid_optional_adapters_are_constructed_during_execution(self) -> None:
        out_dir = self.root / "optional_adapters"
        args = self.compare_args(out_dir, "dummy=dummy-v1")
        args.observer_provider = "dummy"
        args.observer_model = "observer-v1"
        args.embedding_provider = "dummy"
        args.embedding_model = "embedding-v1"
        args.semantic_judge_backend = "both"
        args.latent_convergence_every = 1

        self.assertEqual(rcl.run_compare(args), 0)

        rows = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 3)
        self.assertIn(
            "independent_observer",
            [row["latent_convergence_judge_source"] for row in rows],
        )
        self.assertIn(
            "dummy",
            [row["embedding_convergence_judge_provider"] for row in rows],
        )

    def test_fresh_matrix_writes_one_run_per_arm_and_repeat(self) -> None:
        out_dir = self.root / "matrix"
        config = self.matrix_config(out_dir, ("observe", "soft_steer"))
        self.assertEqual(rcl.run_compare_matrix_from_config_data(config), 0)

        logs = sorted(out_dir.glob("*.jsonl"))
        self.assertEqual(len(logs), 4)
        self.assertEqual([self.assistant_reply_count(path) for path in logs], [3] * 4)
        self.assertEqual(
            len(json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))), 12
        )
        self.assertEqual(
            len(json.loads((out_dir / "analysis_runs.json").read_text(encoding="utf-8"))), 4
        )
        self.assertEqual(
            len(json.loads((out_dir / "analysis_aggregate.json").read_text(encoding="utf-8"))), 2
        )


if __name__ == "__main__":
    unittest.main()
