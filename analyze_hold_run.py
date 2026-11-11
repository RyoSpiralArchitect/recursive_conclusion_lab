#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline, receipt-backed readout of complete or interrupted source-replay runs.

Reads the three-arm hold_source_comparison format. Never imports an adapter,
resumes a run, or changes its files. A word limit is an analysis criterion,
not an inferred model instruction. Reject inconsistent evidence before output.
"""

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

from inquiry_state import InquiryState
from reply_contract import WORD_COUNTER, count_words, reply_contract, validate_max_words

ARMS = ("workpad", "no_workpad", "source_replay")
NORMAL_FINISH = {"openai": {"completed"}, "dummy": {"stop"}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, relative):
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), "Evidence path escapes run")
    return path


def identity(row, repeat_key="run"):
    scene, arm, run = row["scenario"], row["arm"], row[repeat_key]
    require(isinstance(scene, str) and bool(scene), "Invalid scenario")
    require(arm in ARMS and type(run) is int and run > 0, "Invalid arm/repetition")
    return scene, arm, run


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def usage_total(calls, aliases):
    total, missing = 0, 0
    for c in calls:
        usage = c.get("usage") or {}
        value = next((usage[k] for k in aliases if k in usage), None)
        if numeric(value):
            total += value
        else:
            missing += 1
    return total, missing


def distribution(rows):
    words = [r["words"] for r in rows]
    return dict(
        replies=len(rows),
        normal_finish=sum(r["normal_finish"] for r in rows),
        short_normal=sum(r["short_normal"] for r in rows),
        median_words=statistics.median(words) if words else None,
        mean_excess_words=statistics.mean(r["excess_words"] for r in rows)
        if rows
        else None,
        heading_replies=sum(r["heading_lines"] > 0 for r in rows),
        mean_list_lines=statistics.mean(r["list_lines"] for r in rows)
        if rows
        else None,
        counter_decision_disagreements=sum(
            r["counter_decision_disagreement"] for r in rows
        ),
    )


def analyze_run(root, max_words=180):
    validate_max_words(max_words)
    require(max_words > 0, "Analysis max_words must be positive")
    root = Path(root).resolve()
    require(
        all(
            (root / name).resolve().is_relative_to(root)
            for name in ("inputs", "compare")
        ),
        "Evidence directory escapes run",
    )
    hashes = {}

    def read(relative, jsonl=False):
        p = inside(root, relative)
        data = p.read_bytes()
        hashes[str(p.relative_to(root))] = hashlib.sha256(data).hexdigest()
        return (
            [json.loads(line) for line in data.splitlines() if line.strip()]
            if jsonl
            else json.loads(data)
        )

    manifest = read("run_manifest.json")
    require(
        manifest["status"] in ("complete", "incomplete"),
        "Run must have stopped before analysis",
    )
    provider = manifest["provider"]
    require(provider in NORMAL_FINISH, "Unsupported source-replay provider")
    normal = NORMAL_FINISH[provider]
    plan = [identity(p, "repeat") for p in manifest["execution_order"]]
    require(plan and len(set(plan)) == len(plan), "Duplicate or empty execution plan")
    expected = manifest["expected_replies"]
    require(
        type(expected) is int and expected > 0 and expected % len(plan) == 0,
        "Invalid expected reply count",
    )
    turns = expected // len(plan)
    for source, digest in manifest["source_files_sha256"].items():
        p = inside(root / "inputs", source)
        require(
            hashlib.sha256(p.read_bytes()).hexdigest() == digest,
            "Frozen input hash mismatch",
        )
        hashes[str(p.relative_to(root))] = digest
    calls = (
        []
        if manifest.get("actual_generation_calls") == 0
        and not (root / "generation_receipts.json").exists()
        else read("generation_receipts.json")
    )
    timings = (
        []
        if manifest.get("actual_turns") == 0
        and not (root / "turn_timings.json").exists()
        else read("turn_timings.json")
    )
    require(
        len(calls) <= manifest["generation_call_limit"], "Generation ceiling exceeded"
    )
    require(
        manifest.get("actual_generation_calls") == len(calls),
        "Manifest call count mismatch",
    )
    require(
        manifest.get("actual_turns") == len(timings), "Manifest turn count mismatch"
    )
    signatures, paths, used_logs = set(), {}, set()
    for index, c in enumerate(calls, 1):
        key = identity(c)
        require(key in plan and c["call"] == index, "Unplanned or unordered receipt")
        require(
            type(c["turn"]) is int and 1 <= c["turn"] <= turns, "Invalid receipt turn"
        )
        require(
            c["kind"] in ("assistant_reply", "inquiry_update"), "Unexpected call kind"
        )
        require(
            c["kind"] != "inquiry_update" or key[1] == "workpad",
            "Unexpected auxiliary generation",
        )
        status = c["status"]
        require(
            status in ("returned", "failed_or_unknown", "started"),
            "Invalid receipt status",
        )
        require(
            c.get("ok")
            is (
                True
                if status == "returned"
                else False
                if status == "failed_or_unknown"
                else None
            ),
            "Inconsistent receipt outcome",
        )
        if status != "started":
            require(numeric(c.get("elapsed_seconds")), "Invalid call duration")
        sig = (key, c["turn"], c["kind"])
        require(sig not in signatures, "Duplicate generation receipt")
        signatures.add(sig)
        relative = c["dialogue_log"]
        path = inside(root / "compare", relative)
        require(path.suffix == ".jsonl", "Invalid dialogue path")
        require(paths.get(key, relative) == relative, "Dialogue changed log paths")
        paths[key] = relative
    require(len(set(paths.values())) == len(paths), "Log reused for multiple dialogues")
    unknown = [i for i, c in enumerate(calls) if c["status"] != "returned"]
    require(
        not unknown or unknown == [len(calls) - 1],
        "Run continued after an unknown outcome",
    )
    timing_signatures = set()
    for t in timings:
        sig = (identity(t), t["turn"])
        require(
            sig[0] in paths and paths[sig[0]] == t["dialogue_log"], "Unplanned timing"
        )
        require(
            type(t["ok"]) is bool and numeric(t["elapsed_seconds"]),
            "Invalid timing outcome",
        )
        require(sig not in timing_signatures, "Duplicate turn timing")
        require(
            any(k == sig[0] and turn == sig[1] for k, turn, _ in signatures),
            "Timing has no call",
        )
        timing_signatures.add(sig)

    rows, dialogues = [], []
    for key in plan:
        scene, arm, run = key
        cs = [c for c in calls if identity(c) == key]
        schedule = [
            (t, kind)
            for t in range(1, turns + 1)
            for kind in (
                ("inquiry_update", "assistant_reply")
                if arm == "workpad"
                else ("assistant_reply",)
            )
        ]
        require(
            [(c["turn"], c["kind"]) for c in cs] == schedule[: len(cs)],
            "Dialogue calls differ from planned order",
        )
        ts = [t for t in timings if identity(t) == key]
        events = read("compare/" + paths[key], jsonl=True) if cs else []
        if cs:
            used_logs.add(inside(root / "compare", paths[key]).resolve())
        relevant = [
            v
            for v in events
            if v["event_type"] in ("assistant_reply", "inquiry_update")
        ]
        by_event = {(v["turn_index"], v["event_type"]): v for v in relevant}
        require(len(by_event) == len(relevant), "Duplicate generation event")
        returned = [c for c in cs if c["status"] == "returned"]
        require(
            [(c["turn"], c["kind"]) for c in returned] == list(by_event),
            "Receipt/event sequence mismatch",
        )
        exclusions, reply_turns, valid_memos = [], [], 0
        for c in returned:
            event = by_event[(c["turn"], c["kind"])]
            p = event["payload"]
            require(
                event["provider"] == provider and event["model"] == manifest["model"],
                "Event model mismatch",
            )
            require(
                all(
                    c.get(k) == p.get(k)
                    for k in ("usage", "finish_reason", "request_id")
                ),
                "Receipt/event metadata mismatch",
            )
            good_finish = p.get("finish_reason") in normal
            if c["kind"] == "inquiry_update":
                valid = False
                if p.get("state") is not None:
                    InquiryState.from_dict(p["state"])
                    valid = True
                valid_memos += valid
                if not valid or not good_finish:
                    exclusions.append(
                        dict(
                            turn=c["turn"],
                            kind=c["kind"],
                            reason="invalid_or_incomplete_workpad",
                        )
                    )
                continue
            text = p["assistant"]
            require(isinstance(text, str), "Reply text must be a string")
            diagnostic = reply_contract(text, max_words)
            legacy = len(re.findall(r"\w+", text))
            reply_turns.append(c["turn"])
            if not good_finish or not text.strip():
                exclusions.append(
                    dict(
                        turn=c["turn"], kind=c["kind"], reason="empty_or_abnormal_reply"
                    )
                )
            rows.append(
                dict(
                    scenario=scene,
                    arm=arm,
                    run=run,
                    turn=c["turn"],
                    words=count_words(text),
                    legacy_regex_words=legacy,
                    excess_words=diagnostic["excess_words"],
                    normal_finish=good_finish,
                    finish_reason=p.get("finish_reason"),
                    short_normal=good_finish
                    and diagnostic["status"] == "within_budget",
                    counter_decision_disagreement=(
                        diagnostic["status"] == "within_budget"
                    )
                    != (bool(text.strip()) and legacy <= max_words),
                    heading_lines=len(re.findall(r"^#{1,6}\s", text, re.M)),
                    list_lines=len(re.findall(r"^\s*(?:[-*]|\d+[.)])\s", text, re.M)),
                    reply_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
        require(
            reply_turns == list(range(1, len(reply_turns) + 1)),
            "Reply turns are not a contiguous prefix",
        )
        require(
            {t["turn"] for t in ts if t["ok"]} == set(reply_turns),
            "Reply/timing outcome mismatch",
        )
        finished = len(reply_turns) == turns
        expected_calls = turns * (2 if arm == "workpad" else 1)
        if finished:
            require(
                len(returned) == expected_calls,
                "Finished dialogue has missing expected calls",
            )
        if not finished or any(c["status"] != "returned" for c in cs):
            exclusions.append(dict(reason="incomplete_dialogue"))
        dialogues.append(
            dict(
                scenario=scene,
                arm=arm,
                run=run,
                status="finished" if finished else "partial" if cs else "not_started",
                replies=len(reply_turns),
                expected_workpads=turns if arm == "workpad" else 0,
                returned_workpads=sum(c["kind"] == "inquiry_update" for c in returned),
                valid_workpads=valid_memos if arm == "workpad" else None,
                complete_treatment=finished and not exclusions,
                exclusions=exclusions,
            )
        )
    for log in (root / "compare").rglob("*.jsonl"):
        require(log.resolve().is_relative_to(root / "compare"), "Log path escapes run")
        require(
            not log.stat().st_size or log.resolve() in used_logs,
            "Unreceipted dialogue log",
        )
    if manifest["status"] == "complete":
        require(
            all(d["status"] == "finished" for d in dialogues)
            and all(c["status"] == "returned" for c in calls),
            "Complete manifest has incomplete execution",
        )
    by_arm = {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        ds = [d for d in dialogues if d["arm"] == arm]
        cs = [c for c in calls if c["arm"] == arm]
        returned = [c for c in cs if c["status"] == "returned"]
        times = [t["elapsed_seconds"] for t in timings if t["arm"] == arm and t["ok"]]
        tokens_in, missing_in = usage_total(returned, ("input_tokens", "prompt_tokens"))
        tokens_out, missing_out = usage_total(
            returned, ("output_tokens", "completion_tokens")
        )
        by_arm[arm] = dict(
            **distribution(rs),
            attempted_calls=len(cs),
            returned_calls=len(returned),
            unknown_outcome_calls=len(cs) - len(returned),
            returned_input_tokens=tokens_in,
            returned_output_tokens=tokens_out,
            returned_missing_input_usage=missing_in,
            returned_missing_output_usage=missing_out,
            median_completed_turn_seconds=statistics.median(times) if times else None,
            returned_workpads=sum(d["returned_workpads"] for d in ds),
            valid_workpads=sum(d["valid_workpads"] or 0 for d in ds)
            if arm == "workpad"
            else None,
            by_turn={
                str(t): distribution([r for r in rs if r["turn"] == t])
                for t in range(1, turns + 1)
            },
        )
    pairs, blocks = [], []
    for scene, run in sorted({(d["scenario"], d["run"]) for d in dialogues}):
        ds = {
            d["arm"]: d for d in dialogues if (d["scenario"], d["run"]) == (scene, run)
        }
        require(set(ds) == set(ARMS), "Plan needs three arms per case/repetition")
        blocks.append(
            dict(
                scenario=scene,
                run=run,
                finished=all(d["status"] == "finished" for d in ds.values()),
                complete=all(d["complete_treatment"] for d in ds.values()),
            )
        )
        for a, b in itertools.combinations(ARMS, 2):
            pairs.append(
                dict(
                    scenario=scene,
                    run=run,
                    arms=[a, b],
                    available=all(ds[x]["status"] == "finished" for x in (a, b)),
                    complete=all(ds[x]["complete_treatment"] for x in (a, b)),
                )
            )
    return dict(
        schema="rcl.source_run_analysis.v1",
        run_status=manifest["status"],
        source_commit=manifest["source_commit"],
        analysis_word_counter=WORD_COUNTER,
        analysis_max_words=max_words,
        analysis_limit_is_model_configuration=False,
        semantic_quality="not_scored",
        call_statuses=dict(Counter(c["status"] for c in calls)),
        attempted_calls=len(calls),
        returned_calls=sum(c["status"] == "returned" for c in calls),
        replies=len(rows),
        dialogue_statuses=dict(Counter(d["status"] for d in dialogues)),
        planned_dialogues=len(plan),
        planned_pairs=len(pairs),
        available_pairs=sum(p["available"] for p in pairs),
        complete_pairs=sum(p["complete"] for p in pairs),
        by_arm=by_arm,
        rows=rows,
        dialogues=dialogues,
        pairs=pairs,
        blocks=blocks,
        evidence_sha256=hashes,
    )


def write_report(root, output, max_words=180):
    root, output = Path(root).resolve(), Path(output).absolute()
    require(
        not output.resolve().is_relative_to(root),
        "Output must be outside the evidence run",
    )
    require(not output.exists() and not output.is_symlink(), "Output already exists")
    result = analyze_run(root, max_words)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-words", type=int, default=180)
    args = parser.parse_args()
    result = write_report(args.run, args.out, args.max_words)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "run_status",
                    "attempted_calls",
                    "returned_calls",
                    "replies",
                    "dialogue_statuses",
                    "available_pairs",
                    "complete_pairs",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
