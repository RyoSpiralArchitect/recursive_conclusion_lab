#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare workpad cost, its removal, and bounded verbatim user replay."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch

import build_human_eval_set as review
import hold_action_comparison as old
import hold_constraint_brevity as prior
import hold_dialogue_pilot as pilot
import recursive_conclusion_lab as lab

REPO = Path(__file__).resolve().parent
ARMS = ("workpad", "no_workpad", "source_replay")
CASES = prior.CASES + (
    ("camera_returns", "hold_source"),
    ("workshop_learning", "hold_source"),
    ("storage_revision", "hold_source"),
    ("purpose_revision", "hold_source"),
)
SUITES = {"regression": (0, 1, 2, 3), "transfer": (4, 5, 6, 7)}
SETTINGS = prior.SETTINGS.copy()
REPLY = prior.CONTROL + prior.BREVITY
PROBE = lab.INQUIRY_PROBE_SYSTEM
CALL_LIMIT = 288
DESIGN = "docs/experiments/hold-source-replay.md"
FROZEN = "docs/experiments/data/hold_source_frozen.json"
SOURCE = (
    "hold_source_comparison.py",
    "user_source_context.py",
    "recursive_conclusion_lab.py",
    "reply_contract.py",
    "inquiry_state.py",
    "playtest_server.py",
    "hold_dialogue_pilot.py",
    "hold_action_comparison.py",
    "hold_constraint_brevity.py",
    "build_human_eval_set.py",
    "analyze_runs.py",
    DESIGN,
    FROZEN,
    *(
        f"docs/experiments/data/{prefix}_facts.json"
        for prefix in ("hold_scope", "hold_purpose", "hold_source")
    ),
    *(f"protocol_scripts/{prefix}_{name}.json" for name, prefix in CASES),
)


def preflight(output, live, suite):
    if suite not in SUITES:
        raise ValueError("Unsupported suite")
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists")
    frozen = pilot.read_json(REPO / FROZEN)
    if (
        frozen["reply_sha256"] != prior.sha(REPLY)
        or frozen["probe_sha256"] != prior.sha(PROBE)
        or frozen["settings"] != SETTINGS
    ):
        raise ValueError("Frozen prompt or settings changed")
    provider, model = ("openai", "gpt-5.6-luna") if live else ("dummy", "dummy-v1")
    if live and not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ValueError("Credential unavailable")
    dirty = bool(
        subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=REPO)
    )
    tracked = set(
        subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).splitlines()
    )
    if live and (dirty or not set(SOURCE).issubset(tracked)):
        raise ValueError("Commit frozen sources before live execution")
    sources = {s: pilot.digest(REPO / s) for s in SOURCE}
    planned = []
    for repeat in (1, 2, 3):
        for index in SUITES[suite]:
            name, prefix = CASES[index]
            script = REPO / f"protocol_scripts/{prefix}_{name}.json"
            system, turns = lab.load_script(script)
            facts = pilot.read_json(REPO / f"docs/experiments/data/{prefix}_facts.json")
            if (
                len(turns) != 6
                or frozen["script_sha256"][name] != pilot.digest(script)
                or facts["scenarios"][name]["user_turns_exact"]
                != {str(i): t for i, t in enumerate(turns, 1)}
            ):
                raise ValueError("Script differs from frozen facts")
            shift = (index + repeat - 1) % len(ARMS)
            for arm in ARMS[shift:] + ARMS[:shift]:
                settings = dict(
                    SETTINGS,
                    conclusion_every=int(arm == "workpad"),
                    hold_user_replay=arm == "source_replay",
                )
                args = lab.build_compare_args_from_config(
                    dict(
                        script=str(script),
                        providers=[f"{provider}={model}"],
                        out_dir=str(output / "compare" / name),
                        args=settings,
                    ),
                    arm_name=arm,
                )
                args.run_index, args.run_name, args.random_seed = (
                    repeat,
                    f"run_{repeat:03d}",
                    6 + repeat,
                )
                lab.parse_provider_specs(args.providers)
                lab.validate_optional_adapter_options(args)
                lab.make_experiment_config_from_args(args, base_system=system)
                planned.append(
                    (name, arm, args, "\n\n".join((lab.compact_text(system), REPLY)))
                )
    catalog = None
    if live:
        import requests

        response = requests.get(
            old.CATALOG_URLS[provider],
            headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
            timeout=30,
        )
        if response.status_code != 200:
            raise ValueError(
                f"Model catalog HTTP {response.status_code}; details omitted"
            )
        row = next(
            (r for r in response.json().get("data", []) if r.get("id") == model), None
        )
        if row is None:
            raise ValueError("Exact model unavailable; no replacement")
        catalog = {"id": row["id"]}
    return provider, model, planned, sources, dirty, catalog


def summarize(output, provider, planned, calls, timings, items, key):
    expected, rows, dialogues = {}, [], []
    normal = pilot.NORMAL_FINISH[provider]
    for scenario, arm, args, system in planned:
        _, source_turns = lab.load_script(Path(args.script))
        log = lab.build_compare_log_path(
            Path(args.out_dir),
            provider=provider,
            model=args.providers[0].split("=", 1)[1],
            arm_name=arm,
            run_name=args.run_name,
        )
        identity = str(log.relative_to(output / "compare"))
        events = [json.loads(line) for line in log.read_text().splitlines()]
        relevant = [
            e
            for e in events
            if e["event_type"] in ("inquiry_update", "assistant_reply")
        ]
        if Counter(e["event_type"] for e in relevant) != {
            "assistant_reply": 6,
            **({"inquiry_update": 6} if arm == "workpad" else {}),
        }:
            raise ValueError("Unexpected dialogue call composition")
        failures, valid = [], 0
        for e in relevant:
            p, turn = e["payload"], e["turn_index"]
            signature = (identity, turn, e["event_type"])
            if signature in expected:
                raise ValueError("Duplicate event")
            expected[signature] = p
            is_reply = e["event_type"] == "assistant_reply"
            good = p.get("finish_reason") in normal and (
                bool(p.get("assistant", "").strip())
                if is_reply
                else bool(p.get("state"))
            )
            if not good:
                failures.append(
                    dict(
                        turn=turn,
                        kind=e["event_type"],
                        finish=p.get("finish_reason"),
                        error=p.get("error"),
                    )
                )
            if not is_reply:
                valid += bool(p.get("state"))
            else:
                source = p.get("source_user_history")
                if (
                    arm == "source_replay"
                    and turn >= 5
                    and (
                        not source
                        or source["omitted_count"]
                        or source["messages"]
                        != [
                            {"turn": i, "text": source_turns[i - 1]}
                            for i in range(1, turn - 3)
                        ]
                    )
                ):
                    raise ValueError("Expected verbatim source coverage is missing")
                rows.append(
                    dict(
                        scenario=scenario,
                        arm=arm,
                        run=args.run_index,
                        turn=turn,
                        normal_finish=p.get("finish_reason") in normal,
                        words=len(p["assistant"].split()),
                        source_context=source,
                    )
                )
        dialogues.append(
            dict(
                scenario=scenario,
                arm=arm,
                run=args.run_index,
                log=identity,
                valid_workpads=valid,
                expected_workpads=6 if arm == "workpad" else 0,
                complete_treatment=not failures,
                exclusions=failures,
            )
        )
    signatures = set()
    for call in calls:
        sig = (call["dialogue_log"], call["turn"], call["kind"])
        if sig in signatures or not call["ok"] or call["status"] != "returned":
            raise ValueError("Duplicate or incomplete call")
        signatures.add(sig)
        if any(
            call.get(k) != expected[sig].get(k)
            for k in ("usage", "finish_reason", "request_id")
        ):
            raise ValueError("Receipt/event mismatch")
    if (
        signatures != set(expected)
        or len(calls) != 288
        or len(rows) != 216
        or len(timings) != 216
        or not all(t["ok"] for t in timings)
    ):
        raise ValueError("Incomplete accounting")
    if len({(t["dialogue_log"], t["turn"]) for t in timings}) != 216:
        raise ValueError("Duplicate turn timing")
    complete = {
        (d["scenario"], d["arm"], d["run"])
        for d in dialogues
        if d["complete_treatment"]
    }
    eligible = [
        i
        for i in items
        if all(
            (i["scenario_name"], a, i["run_index"]) in complete
            for a in key[i["item_id"]]["arm_pair_sorted"]
        )
    ]
    groups = {}
    for arm in ARMS:
        cs = [c for c in calls if c["arm"] == arm]
        ts = [t for t in timings if t["arm"] == arm]
        rs = [r for r in rows if r["arm"] == arm]
        inputs, missing_in = pilot.token_total(
            [c.get("usage") or {} for c in cs], ("input_tokens", "prompt_tokens")
        )
        outputs, missing_out = pilot.token_total(
            [c.get("usage") or {} for c in cs], ("output_tokens", "completion_tokens")
        )
        groups[arm] = dict(
            calls=len(cs),
            replies=len(rs),
            normal_finish=sum(r["normal_finish"] for r in rs),
            short_complete=sum(
                r["normal_finish"] and 0 < r["words"] <= 180 for r in rs
            ),
            median_words=statistics.median(r["words"] for r in rs),
            median_turn_seconds=statistics.median(t["elapsed_seconds"] for t in ts),
            provider_seconds=sum(c["elapsed_seconds"] for c in cs),
            local_residual_seconds=sum(t["elapsed_seconds"] for t in ts)
            - sum(c["elapsed_seconds"] for c in cs),
            input_tokens=inputs,
            output_tokens=outputs,
            missing_input_usage=missing_in,
            missing_output_usage=missing_out,
        )
    return dict(
        generations=len(calls),
        replies=len(rows),
        workpads=72,
        valid_workpads=sum(d["valid_workpads"] for d in dialogues),
        all_pairs=len(items),
        complete_pairs=len(eligible),
        by_arm=groups,
        dialogues=dialogues,
        reply_metrics=rows,
        quality_judgments="Independent judgments not collected",
    ), eligible


def run_comparison(output, live=False, suite="regression"):
    output = Path(output).expanduser().resolve()
    provider, model, planned, sources, dirty, catalog = preflight(output, live, suite)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        mode="live" if live else "offline_dummy",
        provider=provider,
        model=model,
        suite=suite,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        source_tracked_dirty=dirty,
        source_files_sha256=sources,
        model_catalog=catalog,
        generation_call_limit=CALL_LIMIT,
        expected_replies=216,
        expected_workpads=72,
        automatic_retries=0,
        execution_order=[
            dict(scenario=s, arm=a, repeat=x.run_index) for s, a, x, _ in planned
        ],
    )
    pilot.write_json(output / "run_manifest.json", manifest)
    calls, timings, active = [], [], {}
    expected_system = None
    original_build, original_turn = (
        lab.build_adapter,
        lab.RecursiveConclusionSession.user_turn,
    )

    def build(*args, **kwargs):
        adapter = original_build(*args, **kwargs)
        generate = adapter.generate

        def measured(**kwargs):
            probe = kwargs["system"] == PROBE
            cap, temp = (600, 0.0) if probe else (900, 0.2)
            if (
                not active
                or len(calls) >= CALL_LIMIT
                or kwargs["config"].max_tokens != cap
                or kwargs["config"].temperature != temp
                or (probe and active["arm"] != "workpad")
                or (not probe and kwargs["system"] != expected_system)
            ):
                raise RuntimeError("Call budget or generation context violated")
            message_data = [
                dict(role=m.role, content=m.content) for m in kwargs["messages"]
            ]
            entry = dict(
                call=len(calls) + 1,
                **active,
                kind="inquiry_update" if probe else "assistant_reply",
                prompt_sha256=prior.sha(kwargs["system"]),
                messages_sha256=prior.sha(json.dumps(message_data, ensure_ascii=False)),
                message_characters=sum(len(m["content"]) for m in message_data),
                output_token_limit=cap,
                temperature=temp,
                status="started",
                ok=None,
            )
            calls.append(entry)
            pilot.write_json(output / "generation_receipts.json", calls)
            start = time.perf_counter()
            try:
                response = generate(**kwargs)
                entry.update(
                    status="returned",
                    ok=True,
                    usage=response.usage,
                    finish_reason=response.finish_reason,
                    request_id=response.request_id,
                    response_model=response.adapter_metadata.get("response_model"),
                )
                return response
            except BaseException as exc:
                entry.update(
                    status="failed_or_unknown", ok=False, **old.failure_metadata(exc)
                )
                raise
            finally:
                entry["elapsed_seconds"] = time.perf_counter() - start
                pilot.write_json(output / "generation_receipts.json", calls)

        adapter.generate = measured
        return adapter

    def turn(session, text):
        active.update(
            dialogue_log=str(session.log_path.relative_to(output / "compare")),
            turn=session.turn_index + 1,
        )
        entry = dict(**active, ok=False)
        start = time.perf_counter()
        try:
            result = original_turn(session, text)
            entry["ok"] = True
            return result
        finally:
            entry["elapsed_seconds"] = time.perf_counter() - start
            timings.append(entry)
            pilot.write_json(output / "turn_timings.json", timings)

    start = time.perf_counter()
    try:
        for name in SOURCE:
            target = output / "inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / name).read_bytes())
        (output / "reply.txt").write_text(REPLY)
        (output / "probe.txt").write_text(PROBE)
        paths = []
        for _, arm, args, _ in planned:
            paths += [
                lab.build_compare_log_path(
                    Path(args.out_dir),
                    provider=provider,
                    model=model,
                    arm_name=arm,
                    run_name=args.run_name,
                ),
                lab.build_compare_summary_path(
                    Path(args.out_dir), arm_name=arm, run_name=args.run_name
                ),
            ]
        lab.reserve_new_artifacts(paths)
        with (
            patch.object(lab, "build_adapter", build),
            patch.object(lab.RecursiveConclusionSession, "user_turn", turn),
            patch.object(lab, "INQUIRY_REPLY_SYSTEM", REPLY),
        ):
            for scenario, arm, args, expected_system in planned:
                active.clear()
                active.update(scenario=scenario, arm=arm, run=args.run_index)
                execution = lab.execute_compare(args, artifacts_reserved=True)
                lab.write_compare_summary(
                    Path(args.out_dir),
                    execution.rows,
                    arm_name=arm,
                    run_name=args.run_name,
                )
                print(
                    f"Completed {scenario} {arm} repeat {args.run_index}: {len(calls)}/288 calls",
                    flush=True,
                )
        rubric = dict(
            title="Workpad cost and verbatim user replay",
            rubric_version="hold_source_v1",
            seed=11,
            scenarios=[
                dict(
                    name=CASES[i][0],
                    display_name=CASES[i][0].replace("_", " ").title(),
                    compare_out_dir=str(output / "compare" / CASES[i][0]),
                    arms=list(ARMS),
                )
                for i in SUITES[suite]
            ],
            questions=[
                dict(
                    id="purpose_scope_revision",
                    choices=["A", "B", "Tie"],
                    prompt="Which dialogue better preserves the user's purpose and exact active constraints while accepting explicit user revisions? Do not treat a third-party preference as a user revision. Quote supporting and conflicting evidence or abstain.",
                ),
                dict(
                    id="useful_completion",
                    choices=["A", "B", "Tie"],
                    prompt="Which dialogue makes useful, concise progress with revisable conclusions and material uncertainty? Shortness and normal finish alone are insufficient. Read the full history and quote evidence or abstain.",
                ),
            ],
        )
        pilot.write_json(output / "human_eval.json", rubric)
        items, key = review.build_items(rubric)
        if len(items) != 36:
            raise ValueError("Expected 36 pairs")
        metrics, eligible = summarize(
            output, provider, planned, calls, timings, items, key
        )
        review.write_outputs(
            out_dir=output / "review", items=items, blind_key=key, config=rubric
        )
        if eligible:
            subset_key = {
                "_build_meta": key["_build_meta"],
                **{i["item_id"]: key[i["item_id"]] for i in eligible},
            }
            review.write_outputs(
                out_dir=output / "complete_review",
                items=eligible,
                blind_key=subset_key,
                config={
                    **rubric,
                    "subset_policy": "Whole pair excluded if any abnormal finish, empty reply, or invalid expected workpad; no-workpad arms do not require memos.",
                },
            )
        if any(pilot.digest(REPO / s) != h for s, h in sources.items()):
            raise ValueError("Source changed during execution")
        pilot.write_json(output / "metrics.json", metrics)
        manifest.update(status="complete", complete_pairs=len(eligible))
    except BaseException as exc:
        manifest.update(status="incomplete", error_type=type(exc).__name__)
        raise
    finally:
        manifest.update(
            actual_generation_calls=len(calls),
            actual_turns=len(timings),
            elapsed_seconds=time.perf_counter() - start,
        )
        pilot.write_json(output / "run_manifest.json", manifest)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--suite", required=True, choices=tuple(SUITES))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--live", action="store_true")
    args = parser.parse_args()
    try:
        result = run_comparison(args.out, args.live, args.suite)
    except Exception as exc:
        print(f"Stopped ({type(exc).__name__}); preserve receipts, no automatic retry.")
        return 1
    print(
        f"Complete: {result['generations']} generations; {result['complete_pairs']}/36 complete pairs. Independent judgments pending."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
