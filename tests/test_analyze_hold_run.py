# SPDX-License-Identifier: AGPL-3.0-or-later
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from analyze_hold_run import FROZEN, analyze_run, request_messages, write_report
from inquiry_state import InquiryState
from user_source_context import source_user_history

VALID = dict(
    focus="Explore",
    hypotheses=[],
    open_questions=[],
    next_step="Try one option",
    revision_note="",
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def bind_messages(receipt, messages):
    receipt.update(
        messages_sha256=digest(json.dumps(messages, ensure_ascii=False)),
        message_characters=sum(len(m["content"]) for m in messages),
    )


def fixture(root, *, interrupted=False, started=False):
    root.mkdir()
    source = root / "inputs/fixture.txt"
    source.parent.mkdir()
    source.write_text("frozen input")
    script_path = "protocol_scripts/hold_test_case.json"
    users = ["User 1", "User 2"]
    write(root / "inputs" / script_path, {"system": "Test system", "turns": users})
    script_hash = digest((root / "inputs" / script_path).read_text())
    prompts = {"probe": "Probe system", "reply": "Reply system"}
    for name, prompt in prompts.items():
        (root / f"{name}.txt").write_text(prompt)
    write(
        root / "inputs" / FROZEN,
        dict(
            script_sha256={"case": script_hash},
            settings=dict(
                window=2, max_tokens=900, probe_max_tokens=600, temperature=0.2
            ),
            **{k + "_sha256": digest(v) for k, v in prompts.items()},
        ),
    )
    sources = {
        "fixture.txt": digest(source.read_text()),
        script_path: script_hash,
        FROZEN: digest((root / "inputs" / FROZEN).read_text()),
    }
    arms = ("no_workpad", "source_replay", "workpad")
    plan = [dict(scenario="case", arm=a, repeat=1) for a in arms]
    calls, timings = [], []
    for arm in arms:
        events, history, state, state_turn = [], [], None, None
        log = f"case/arm_{arm}___openai__gpt-5.6-luna__run_001.jsonl"
        for turn in (1, 2):
            history.append(dict(role="user", content=users[turn - 1]))
            source_context = (
                source_user_history([SimpleNamespace(**m) for m in history], 2)
                if arm == "source_replay"
                else None
            )
            fail = interrupted and arm == "workpad" and turn == 2
            for kind in (
                ("inquiry_update", "assistant_reply")
                if arm == "workpad"
                else ("assistant_reply",)
            ):
                probe = kind == "inquiry_update"
                c = dict(
                    call=len(calls) + 1,
                    scenario="case",
                    arm=arm,
                    run=1,
                    dialogue_log=log,
                    turn=turn,
                    kind=kind,
                    status="returned",
                    ok=True,
                    elapsed_seconds=0.04,
                    usage=dict(input_tokens=10, output_tokens=2),
                    finish_reason="completed",
                    request_id=f"id_{len(calls)}",
                    prompt_sha256=digest(
                        prompts["probe"] if probe else "Test system\n\nReply system"
                    ),
                    temperature=0.0 if probe else 0.2,
                    output_token_limit=600 if probe else 900,
                )
                bind_messages(
                    c,
                    request_messages(
                        history[-2:],
                        probe=probe,
                        state=state,
                        state_turn=state_turn,
                        source=source_context,
                    ),
                )
                if fail:
                    c.update(
                        status="started" if started else "failed_or_unknown",
                        ok=None if started else False,
                    )
                    for k in ("usage", "finish_reason", "request_id"):
                        c.pop(k)
                    if started:
                        c.pop("elapsed_seconds")
                    calls.append(c)
                    break
                calls.append(c)
                p = {k: c[k] for k in ("usage", "finish_reason", "request_id")}
                if kind == "inquiry_update":
                    p.update(state=copy.deepcopy(VALID), error=None)
                    state, state_turn = InquiryState.from_dict(VALID), turn
                else:
                    p["user"] = users[turn - 1]
                    p["assistant"] = (
                        "## Next\n- stay open"
                        if arm == "no_workpad" and turn == 1
                        else "Keep the condition."
                    )
                    if arm == "source_replay":
                        p["source_user_history"] = source_context
                    history.append(dict(role="assistant", content=p["assistant"]))
                events.append(
                    dict(
                        provider="openai",
                        model="gpt-5.6-luna",
                        event_type=kind,
                        turn_index=turn,
                        payload=p,
                    )
                )
            timings.append(
                dict(
                    scenario="case",
                    arm=arm,
                    run=1,
                    dialogue_log=log,
                    turn=turn,
                    ok=not fail,
                    elapsed_seconds=0.1,
                )
            )
        target = root / "compare" / log
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    write(root / "generation_receipts.json", calls)
    write(root / "turn_timings.json", timings)
    write(
        root / "run_manifest.json",
        dict(
            status="incomplete" if interrupted else "complete",
            provider="openai",
            model="gpt-5.6-luna",
            source_commit="fixture",
            execution_order=plan,
            expected_replies=6,
            generation_call_limit=8,
            actual_generation_calls=len(calls),
            actual_turns=len(timings),
            source_files_sha256=sources,
        ),
    )
    return calls


class HoldRunAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.out = Path(self.temp.name) / "new/report.json"

    def test_complete_run_accounts_calls_pairs_and_counter_disagreement(self):
        fixture(self.root)
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = write_report(self.root, self.out, 4)
        self.assertEqual(
            (result["attempted_calls"], result["returned_calls"], result["replies"]),
            (8, 8, 6),
        )
        self.assertEqual((result["available_pairs"], result["complete_pairs"]), (3, 3))
        self.assertEqual(
            result["by_arm"]["no_workpad"]["counter_decision_disagreements"], 1
        )
        self.assertEqual(result["by_arm"]["workpad"]["valid_workpads"], 2)
        self.assertIsNone(result["by_arm"]["source_replay"]["valid_workpads"])
        self.assertFalse(result["analysis_limit_is_model_configuration"])
        self.assertEqual(
            before,
            {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()},
        )

    def test_interrupted_or_started_request_is_unknown_not_missing_zero_cost(self):
        for started in (False, True):
            with self.subTest(started=started):
                root = self.root.with_name("started" if started else "failed")
                fixture(root, interrupted=True, started=started)
                result = analyze_run(root, 4)
                self.assertEqual(
                    (
                        result["attempted_calls"],
                        result["returned_calls"],
                        result["replies"],
                    ),
                    (7, 6, 5),
                )
                self.assertEqual(
                    result["dialogue_statuses"], {"finished": 2, "partial": 1}
                )
                self.assertEqual(
                    (result["available_pairs"], result["complete_pairs"]), (1, 1)
                )
                self.assertEqual(result["planned_pairs"], 3)
                arm = result["by_arm"]["workpad"]
                self.assertEqual(
                    (arm["returned_input_tokens"], arm["unknown_outcome_calls"]),
                    (20, 1),
                )

    def test_not_started_dialogues_do_not_become_finished_from_empty_reservations(self):
        fixture(self.root, interrupted=True)
        manifest = json.loads((self.root / "run_manifest.json").read_text())
        for arm in ("workpad", "no_workpad", "source_replay"):
            manifest["execution_order"].append(dict(scenario="case", arm=arm, repeat=2))
            (
                self.root
                / "compare/case"
                / f"arm_{arm}___openai__gpt-5.6-luna__run_002.jsonl"
            ).touch()
        manifest["expected_replies"] = 12
        manifest["generation_call_limit"] = 16
        write(self.root / "run_manifest.json", manifest)
        result = analyze_run(self.root)
        self.assertEqual(
            result["dialogue_statuses"], {"finished": 2, "partial": 1, "not_started": 3}
        )
        self.assertEqual((result["planned_pairs"], result["available_pairs"]), (6, 1))

    def test_stopped_before_first_call_has_no_invented_usage_or_success(self):
        fixture(self.root)
        manifest = json.loads((self.root / "run_manifest.json").read_text())
        manifest.update(status="incomplete", actual_generation_calls=0, actual_turns=0)
        write(self.root / "run_manifest.json", manifest)
        for name in ("generation_receipts.json", "turn_timings.json"):
            (self.root / name).unlink()
        for log in (self.root / "compare").rglob("*.jsonl"):
            log.write_text("")
        result = analyze_run(self.root)
        self.assertEqual(result["dialogue_statuses"], {"not_started": 3})
        self.assertEqual(
            (result["replies"], result["available_pairs"], result["complete_pairs"]),
            (0, 0, 0),
        )
        self.assertIsNone(result["by_arm"]["workpad"]["median_completed_turn_seconds"])

    def test_abnormal_finish_and_invalid_workpad_retain_all_pairs(self):
        fixture(self.root)
        calls_path = self.root / "generation_receipts.json"
        calls = json.loads(calls_path.read_text())
        target = self.root / "compare" / calls[-1]["dialogue_log"]
        events = [json.loads(x) for x in target.read_text().splitlines()]
        # The last workpad is explicitly rejected; its reply has unknown finish.
        events[-2]["payload"]["state"] = None
        events[-1]["payload"]["finish_reason"] = None
        calls[-1]["finish_reason"] = None
        bind_messages(
            calls[-1],
            [
                dict(role="assistant", content="Keep the condition."),
                dict(role="user", content="User 2"),
            ],
        )
        target.write_text("\n".join(json.dumps(x) for x in events) + "\n")
        write(calls_path, calls)
        result = analyze_run(self.root)
        self.assertEqual((result["available_pairs"], result["complete_pairs"]), (3, 1))
        self.assertEqual(result["by_arm"]["workpad"]["normal_finish"], 1)
        self.assertEqual(result["by_arm"]["workpad"]["valid_workpads"], 1)

    def test_corruption_is_rejected_before_output_directory_creation(self):
        fixture(self.root)
        path = self.root / "generation_receipts.json"
        original = path.read_text()
        for mutate in (
            lambda cs: cs.__setitem__(1, copy.deepcopy(cs[0])),
            lambda cs: cs[0].update(usage={"input_tokens": 999}),
            lambda cs: cs[0].update(dialogue_log="../../elsewhere.jsonl"),
            lambda cs: cs[0].update(turn=2),
            lambda cs: cs[0].update(messages_sha256="bad"),
            lambda cs: cs[0].update(output_token_limit=901),
        ):
            calls = json.loads(original)
            mutate(calls)
            write(path, calls)
            with self.assertRaises(ValueError):
                write_report(self.root, self.out)
            self.assertFalse(self.out.parent.exists())
        path.write_text(original)
        (self.root / "inputs/fixture.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            write_report(self.root, self.out)
        self.assertFalse(self.out.parent.exists())

    def test_global_call_order_rejects_swapped_dialogues(self):
        calls = fixture(self.root)
        calls = calls[2:4] + calls[:2] + calls[4:]
        for index, receipt in enumerate(calls, 1):
            receipt["call"] = index
        write(self.root / "generation_receipts.json", calls)
        with self.assertRaisesRegex(ValueError, "global execution order"):
            write_report(self.root, self.out)
        self.assertFalse(self.out.parent.exists())

    def test_replay_payload_must_match_actual_treatment(self):
        calls = fixture(self.root)
        for receipt, replacement in ((calls[0], {}), (calls[2], {}), (calls[3], None)):
            target = self.root / "compare" / receipt["dialogue_log"]
            original = target.read_text()
            events = [json.loads(line) for line in original.splitlines()]
            events[receipt["turn"] - 1]["payload"]["source_user_history"] = replacement
            target.write_text("\n".join(json.dumps(e) for e in events) + "\n")
            with self.assertRaisesRegex(ValueError, "source replay"):
                write_report(self.root, self.out)
            self.assertFalse(self.out.parent.exists())
            target.write_text(original)

    def test_trailing_return_without_event_preserves_usage_and_partial_status(self):
        calls = fixture(self.root)
        target = self.root / "compare" / calls[-1]["dialogue_log"]
        target.write_text("\n".join(target.read_text().splitlines()[:-1]) + "\n")
        manifest_path = self.root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "incomplete"
        write(manifest_path, manifest)
        timings_path = self.root / "turn_timings.json"
        timings = json.loads(timings_path.read_text())
        timings[-1]["ok"] = False
        write(timings_path, timings)
        result = analyze_run(self.root)
        self.assertEqual(result["returned_without_local_event"], [8])
        self.assertEqual((result["returned_calls"], result["replies"]), (8, 5))
        self.assertEqual(result["by_arm"]["workpad"]["returned_input_tokens"], 40)
        self.assertEqual(result["dialogue_statuses"], {"finished": 2, "partial": 1})
        self.assertEqual(result["complete_pairs"], 1)

    def test_existing_outputs_evidence_mutation_and_running_snapshots_are_rejected(
        self,
    ):
        fixture(self.root)
        with self.assertRaisesRegex(ValueError, "outside"):
            write_report(self.root, self.root / "report.json")
        self.out.parent.mkdir()
        self.out.write_text("existing")
        with self.assertRaisesRegex(ValueError, "exists"):
            write_report(self.root, self.out)
        self.assertEqual(self.out.read_text(), "existing")
        manifest = json.loads((self.root / "run_manifest.json").read_text())
        manifest["status"] = "running"
        write(self.root / "run_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "stopped"):
            analyze_run(self.root)


if __name__ == "__main__":
    unittest.main()
