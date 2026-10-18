#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the bounded, six-dialogue hold pilot using repository-owned templates.

Adapted from hold_dialogue_pilot.zip (2026-10-18). This single-process CLI
instruments the existing matrix runner; it does not alter prompts or retry calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch

import build_human_eval_set as review
import recursive_conclusion_lab as lab

REPO = Path(__file__).resolve().parent
ARMS = ("baseline", "hold_prompt", "hold_workpad")
CALL_LIMIT = 48
KEY_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
             "mistral": "MISTRAL_API_KEY", "gemini": "GEMINI_API_KEY", "hf": "HF_TOKEN"}
NORMAL_FINISH = {"dummy": {"stop"}, "mistral": {"stop"}, "hf": {"stop"},
                 "openai": {"completed"}, "anthropic": {"end_turn"}, "gemini": {"STOP"}}
SOURCE_FILES = (
    "hold_dialogue_pilot.py", "recursive_conclusion_lab.py", "inquiry_state.py",
    "analyze_runs.py", "build_human_eval_set.py",
    "protocol_scripts/hold_open_then_revise.json",
    "templates/compare_matrix_hold_dialogue.json", "templates/human_eval_hold_dialogue.json",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    """Replace a complete snapshot atomically within this reserved run directory."""
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_total(usages: list, names: tuple[str, ...]):
    """Unknown usage stays unknown, including providers with other field names."""
    values = []
    for usage in usages:
        value = next((usage[name] for name in names if name in usage), None)
        values.append(value if type(value) is int and value >= 0 else None)
    missing = sum(value is None for value in values)
    return (None if missing else sum(values)), missing


def preflight(output: Path, provider: str, model: str):
    if output.exists():
        raise ValueError("Output already exists")
    if provider not in {"dummy", *KEY_NAMES} or not model.strip():
        raise ValueError("Unsupported provider or empty model")
    if provider != "dummy":
        if not os.environ.get(KEY_NAMES[provider], "").strip():
            raise ValueError("Selected provider credential is unavailable")
        if importlib.util.find_spec("requests") is None:
            raise ValueError("requests is required for live execution")
    matrix = read_json(REPO / "templates/compare_matrix_hold_dialogue.json")
    matrix.update(script=str(REPO / "protocol_scripts/hold_open_then_revise.json"),
                  providers=[f"{provider}={model}"], out_dir=str(output / "compare"))
    # Match the ZIP's frozen pilot settings, rather than general CLI defaults.
    matrix["args"].update(temperature=0.2, max_tokens=900, probe_max_tokens=600, timeout=60)
    if matrix["repeats"] != 2 or tuple(a["name"] for a in matrix["arms"]) != ARMS:
        raise ValueError("Pilot design changed; review the declared call budget")
    if len(lab.load_script(Path(matrix["script"]))[1]) != 6:
        raise ValueError("Pilot requires six scripted turns")
    for index, arm in enumerate(matrix["arms"]):
        args = lab.build_compare_args_from_config(matrix, args_override=arm["args"])
        lab.validate_optional_adapter_options(args)
        config = lab.make_experiment_config_from_args(args)
        if (config.memory_every or config.delayed_mention_every or config.deferred_intent_every
                or config.latent_convergence_every or config.semantic_judge_backend != "off"
                or config.conclusion_every != (1 if index == 2 else 0)
                or config.conclusion_mode != (lab.ConclusionMode.OBSERVE if index == 0 else lab.ConclusionMode.HOLD)):
            raise ValueError("Pilot controllers differ from the frozen design")
    rubric = read_json(REPO / "templates/human_eval_hold_dialogue.json")
    rubric["out_dir"] = str(output / "review")
    if len(rubric["scenarios"]) != 1 or tuple(rubric["scenarios"][0]["arms"]) != ARMS:
        raise ValueError("Pilot review design changed")
    rubric["scenarios"][0]["compare_out_dir"] = matrix["out_dir"]
    source = {name: digest(REPO / name) for name in SOURCE_FILES}
    revision = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout != ""
    return matrix, rubric, source, revision, dirty


def summarize(output: Path, provider: str, calls: list, timings: list, items: list, key: dict):
    """Keep all-dialogue evidence separate from the complete-treatment subset."""
    finished = NORMAL_FINISH[provider]
    dialogues, accounting, seen = [], {}, set()
    for path in sorted((output / "compare").glob("arm_*.jsonl")):
        arm = path.name.split("___")[0].removeprefix("arm_")
        run = int(path.stem.rsplit("__run_", 1)[1])
        events = [json.loads(s) for s in path.read_text().splitlines()]
        replies = [e for e in events if e["event_type"] == "assistant_reply"]
        updates = [e for e in events if e["event_type"] == "inquiry_update"]
        if len(replies) != 6 or len(updates) != (6 if arm == "hold_workpad" else 0):
            raise ValueError("Unexpected dialogue event count")
        failures = []
        for event in replies + updates:
            p = event["payload"]
            signature = (path.name, event["turn_index"], event["event_type"])
            if signature in seen:
                raise ValueError("Duplicate generation event")
            seen.add(signature)
            accounting[signature] = p
            if (p.get("finish_reason") not in finished or p.get("error")
                    or (event["event_type"] == "inquiry_update" and not p.get("state"))):
                failures.append({"turn": event["turn_index"], "kind": event["event_type"],
                                 "finish_reason": p.get("finish_reason"), "error": p.get("error")})
        dialogues.append({"arm": arm, "run": run, "log": path.name,
                          "valid_workpads": sum(bool(e["payload"].get("state")) and not e["payload"].get("error") for e in updates),
                          "complete_treatment": not failures, "exclusions": failures})
    if len(dialogues) != 6 or len(calls) != CALL_LIMIT or len(timings) != 36:
        raise ValueError("Unexpected pilot size")
    call_keys = set()
    for call in calls:
        signature = (call["dialogue_log"], call["turn"], call["kind"])
        if signature in call_keys or not call.get("ok"):
            raise ValueError("Duplicate or unsuccessful call")
        call_keys.add(signature)
        p = accounting[signature]
        if any(call.get(field) != p.get(field) for field in ("usage", "finish_reason", "request_id")):
            raise ValueError("Call/event receipt mismatch")
    if call_keys != seen or len({(t["dialogue_log"], t["turn"]) for t in timings}) != 36 or not all(t["ok"] for t in timings):
        raise ValueError("Turn/call reconciliation failed")
    by_arm = []
    for arm in ARMS:
        selected = [c for c in calls if c["dialogue_log"].startswith(f"arm_{arm}___")]
        elapsed = [t["elapsed_seconds"] for t in timings if t["dialogue_log"].startswith(f"arm_{arm}___")]
        usages = [c.get("usage") or {} for c in selected]
        inputs, missing_inputs = token_total(usages, ("input_tokens", "prompt_tokens", "promptTokenCount"))
        outputs, missing_outputs = token_total(usages, ("output_tokens", "completion_tokens", "candidatesTokenCount"))
        by_arm.append({"arm": arm, "calls": len(selected), "reply_calls": 12,
                       "workpad_calls": len(selected) - 12,
                       "input_tokens": inputs, "output_tokens": outputs,
                       "missing_input_usage_calls": missing_inputs, "missing_output_usage_calls": missing_outputs,
                       "turn_mean_seconds": statistics.mean(elapsed),
                       "turn_median_seconds": statistics.median(elapsed),
                       "finish_reasons": dict(Counter(str(c.get("finish_reason")) for c in selected))})
    complete = {(d["arm"], d["run"]) for d in dialogues if d["complete_treatment"]}
    eligible = [i for i in items if all((arm, i["run_index"]) in complete for arm in key[i["item_id"]]["arm_pair_sorted"])]
    primary = [i for i in eligible if key[i["item_id"]]["arm_pair_sorted"] == ["hold_prompt", "hold_workpad"]]
    metrics = {"generations": len(calls), "replies": 36, "workpads": 12,
               "valid_workpads": sum(d["valid_workpads"] for d in dialogues),
               "all_pairs": len(items), "complete_pairs": len(eligible), "complete_primary_pairs": len(primary),
               "by_arm": by_arm, "dialogues": dialogues,
               "included_item_ids": [i["item_id"] for i in eligible],
               "quality_judgments": "not collected; completion and schema validity are not quality scores"}
    return metrics, eligible


def run_pilot(output: Path, provider: str = "dummy", model: str = "dummy-v1") -> dict:
    output = output.expanduser().resolve()
    matrix, rubric, source, revision, dirty = preflight(output, provider, model)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "mode": "offline_dummy" if provider == "dummy" else "live",
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "provider": provider, "model": model,
                "source_commit": revision, "source_dirty": dirty, "source_files_sha256": source,
                "provider_call_limit": CALL_LIMIT, "expected_reply_calls": 36, "expected_workpad_calls": 12,
                "automatic_retries": 0, "quality_judgments": "not collected",
                "subset_policy": "Exclude a dialogue from complete-treatment pairs if any generation has a non-normal/unknown finish or any workpad is invalid; retain all-dialogue evidence."}
    calls, timings, active = [], [], {}
    build_adapter, user_turn = lab.build_adapter, lab.RecursiveConclusionSession.user_turn
    write_json(output / "run_manifest.json", manifest)

    def measured_adapter(*args, **kwargs):
        adapter = build_adapter(*args, **kwargs)
        generate = adapter.generate

        def measured_generate(**kwargs):
            is_workpad = kwargs["system"] == lab.INQUIRY_PROBE_SYSTEM
            limit = 600 if is_workpad else 900
            if not active or len(calls) >= CALL_LIMIT or kwargs["config"].max_tokens != limit:
                raise RuntimeError("Pilot call budget or generation context violated")
            entry = {"call": len(calls) + 1, **active, "kind": "inquiry_update" if is_workpad else "assistant_reply",
                     "output_token_limit": limit, "status": "started", "ok": None}
            calls.append(entry)
            # Persist intent before the external call. An interrupted started call
            # has an unknown provider outcome and must never be retried implicitly.
            write_json(output / "generation_receipts.json", calls)
            started = time.perf_counter()
            try:
                response = generate(**kwargs)
                entry.update(status="returned", ok=True, usage=response.usage,
                             finish_reason=response.finish_reason, request_id=response.request_id,
                             response_model=(response.adapter_metadata or {}).get("response_model"))
                return response
            except BaseException as exc:
                entry.update(status="failed_or_unknown", ok=False, error_type=type(exc).__name__)
                raise
            finally:
                entry["elapsed_seconds"] = time.perf_counter() - started
                write_json(output / "generation_receipts.json", calls)
        adapter.generate = measured_generate
        return adapter

    def measured_turn(session, text):
        entry = {"dialogue_log": session.log_path.name, "turn": session.turn_index + 1, "ok": False}
        active.update(dialogue_log=entry["dialogue_log"], turn=entry["turn"])
        started = time.perf_counter()
        try:
            result = user_turn(session, text)
            entry["ok"] = True
            return result
        finally:
            entry["elapsed_seconds"] = time.perf_counter() - started
            timings.append(entry)
            active.clear()
            write_json(output / "turn_timings.json", timings)

    started = time.perf_counter()
    try:
        write_json(output / "matrix.json", matrix)
        write_json(output / "human_eval.json", rubric)
        with patch.object(lab, "build_adapter", measured_adapter), patch.object(lab.RecursiveConclusionSession, "user_turn", measured_turn):
            lab.run_compare_matrix_from_config_data(matrix)
        items, key = review.build_items(rubric)
        if len(items) != 6:
            raise ValueError("Unexpected review pair count")
        metrics, eligible = summarize(output, provider, calls, timings, items, key)
        review.write_outputs(out_dir=output / "review", items=items, blind_key=key, config=rubric)
        if eligible:
            subset_key = {"_build_meta": key["_build_meta"], **{i["item_id"]: key[i["item_id"]] for i in eligible}}
            subset_config = {**rubric, "out_dir": str(output / "complete_review"),
                             "subset_policy": manifest["subset_policy"], "included_item_ids": metrics["included_item_ids"]}
            review.write_outputs(out_dir=output / "complete_review", items=eligible, blind_key=subset_key, config=subset_config)
        write_json(output / "metrics.json", metrics)
        manifest.update(status="complete", complete_pairs=len(eligible))
    except BaseException as exc:
        manifest.update(status="incomplete", error_type=type(exc).__name__)
        raise
    finally:
        manifest.update(elapsed_seconds=time.perf_counter() - started,
                        actual_generation_calls=len(calls), actual_turns=len(timings))
        write_json(output / "run_manifest.json", manifest)
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="New run directory; never overwritten")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--provider-model", help="One provider=model; use an existing environment credential")
    args = parser.parse_args(argv)
    provider, model = "dummy", "dummy-v1"
    if not args.offline:
        provider, sep, model = args.provider_model.partition("=")
        if not sep or provider not in KEY_NAMES or not model.strip():
            parser.error("Expected a supported provider=model")
    try:
        metrics = run_pilot(args.out, provider, model)
    except Exception as exc:
        print(f"Stopped ({type(exc).__name__}). Preserve partial outputs; no automatic retry.")
        return 1
    print(f"Completed {metrics['replies']} replies and {metrics['workpads']} workpad calls; "
          f"{metrics['valid_workpads']} valid workpads, {metrics['complete_primary_pairs']} complete primary pairs.")
    print("Dummy results are wiring checks only." if args.offline else "Human quality review remains pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
