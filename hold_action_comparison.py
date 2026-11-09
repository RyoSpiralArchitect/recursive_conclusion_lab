#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fixed, bounded comparison of two inquiry prompts; no default prompt changes."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import build_human_eval_set as review
import hold_dialogue_pilot as pilot
import recursive_conclusion_lab as lab

REPO = Path(__file__).resolve().parent
SCENARIOS = ("museum", "seeds", "newsletter")
ARMS = ("current", "action")
ORDER = (
    (0, 1, ARMS),
    (1, 1, ARMS[::-1]),
    (2, 1, ARMS),
    (0, 2, ARMS[::-1]),
    (1, 2, ARMS),
    (2, 2, ARMS[::-1]),
)
CALL_LIMIT = 144
CONTROL = lab.INQUIRY_PROBE_SYSTEM
CANDIDATE = CONTROL.replace(
    "next_step (one useful conversational move)",
    "next_step (one concrete action for the assistant's current reply)",
).replace(
    "Each string must be at most 400 characters.",
    "Use next_step for the assistant's concrete action on the current user request; "
    "keep unresolved facts that require outside confirmation in open_questions. "
    "When the user asks for a trial or a provisional recommendation, propose one "
    "with explicit assumptions instead of making that request depend on another question. "
    "Each string must be at most 400 characters.",
)
PROMPTS = dict(zip(ARMS, (CONTROL, CANDIDATE)))
SOURCE = (
    "hold_action_comparison.py",
    "hold_dialogue_pilot.py",
    "recursive_conclusion_lab.py",
    "inquiry_state.py",
    "build_human_eval_set.py",
    "analyze_runs.py",
    "docs/experiments/data/hold_action_facts.json",
    *(f"protocol_scripts/hold_action_{s}.json" for s in SCENARIOS),
)
SETTINGS = dict(
    window=8,
    memory_every=0,
    conclusion_mode="hold",
    conclusion_every=1,
    delayed_mention_every=0,
    deferred_intent_every=0,
    latent_convergence_every=0,
    semantic_judge_backend="off",
    temperature=0.2,
    max_tokens=900,
    probe_max_tokens=600,
    timeout=60,
)


def failure_metadata(exc):
    """Keep only the known transport status template, never arbitrary exception text."""
    match = re.fullmatch(
        r"HTTP ([1-5][0-9]{2}) from provider endpoint; response details omitted",
        str(exc),
    )
    return {
        "error_type": type(exc).__name__,
        "http_status": int(match[1]) if match else None,
    }


LIVE_MODELS = {"mistral": "mistral-small-2603", "openai": "gpt-5.6-luna"}
CATALOG_URLS = {
    "mistral": "https://api.mistral.ai/v1/models",
    "openai": "https://api.openai.com/v1/models",
}


def preflight(output, live, selected_provider="mistral"):
    if selected_provider not in LIVE_MODELS:
        raise ValueError("Unsupported experiment provider")
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists")
    if (
        hashlib.sha256(CONTROL.encode()).hexdigest()
        != "610fd917ee57533a29b3b1f976e1750daf10d7253dccd2158090ccbe88168332"
    ):
        raise ValueError("Control prompt changed; revise the experiment explicitly")
    if (
        hashlib.sha256(CANDIDATE.encode()).hexdigest()
        != "972de4a5ae216f3df0871f5fecd9d04e25b8af53b0bf6164386813cc858168db"
    ):
        raise ValueError("Candidate differs from the frozen proposal")
    provider, model = (
        (selected_provider, LIVE_MODELS[selected_provider])
        if live
        else ("dummy", "dummy-v1")
    )
    if live and not os.environ.get(pilot.KEY_NAMES[provider], "").strip():
        raise ValueError("Selected provider credential unavailable")
    dirty = bool(
        subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=REPO)
    )
    tracked = set(
        subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).splitlines()
    )
    if live and (dirty or not set(SOURCE).issubset(tracked)):
        raise ValueError("Commit experiment sources before live execution")
    sources = {s: pilot.digest(REPO / s) for s in SOURCE}
    facts = pilot.read_json(REPO / "docs/experiments/data/hold_action_facts.json")
    if set(facts["scenarios"]) != set(SCENARIOS):
        raise ValueError("Fact ledger does not match scenarios")
    planned = []
    for index, repeat, arms in ORDER:
        scenario = SCENARIOS[index]
        script = REPO / f"protocol_scripts/hold_action_{scenario}.json"
        system, turns = lab.load_script(script)
        if len(turns) != 6 or any(not t.strip() for t in turns):
            raise ValueError("Each scenario requires six nonempty turns")
        for arm in arms:
            config = dict(
                script=str(script),
                providers=[f"{provider}={model}"],
                out_dir=str(output / "compare" / scenario),
                args=SETTINGS,
            )
            args = lab.build_compare_args_from_config(config, arm_name=arm)
            args.run_index, args.run_name, args.random_seed = (
                repeat,
                f"run_{repeat:03d}",
                7 + repeat - 1,
            )
            lab.validate_optional_adapter_options(args)
            lab.make_experiment_config_from_args(args, base_system=system)
            planned.append((scenario, arm, args))
    catalog = None
    if live:
        import requests

        response = requests.get(
            CATALOG_URLS[provider],
            headers={
                "Authorization": "Bearer " + os.environ[pilot.KEY_NAMES[provider]]
            },
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
        catalog = {k: row[k] for k in ("id", "name", "capabilities") if k in row}
    return provider, model, planned, sources, dirty, catalog


def summarize(output, provider, calls, timings, items, key):
    accounting, dialogues = {}, []
    for path in sorted((output / "compare").glob("*/*.jsonl")):
        events = [json.loads(s) for s in path.read_text().splitlines()]
        relevant = [
            e
            for e in events
            if e["event_type"] in ("inquiry_update", "assistant_reply")
        ]
        if Counter(e["event_type"] for e in relevant) != {
            "inquiry_update": 6,
            "assistant_reply": 6,
        }:
            raise ValueError("Unexpected dialogue size")
        identity = str(path.relative_to(output / "compare"))
        failures = []
        for e in relevant:
            signature = (identity, e["turn_index"], e["event_type"])
            if signature in accounting:
                raise ValueError("Duplicate generation event")
            p = e["payload"]
            accounting[signature] = p
            if (
                p.get("finish_reason") not in pilot.NORMAL_FINISH[provider]
                or p.get("error")
                or (e["event_type"] == "inquiry_update" and not p.get("state"))
                or (
                    e["event_type"] == "assistant_reply"
                    and not p.get("assistant", "").strip()
                )
            ):
                failures.append(
                    dict(
                        turn=e["turn_index"],
                        kind=e["event_type"],
                        finish_reason=p.get("finish_reason"),
                        error=p.get("error"),
                    )
                )
        dialogues.append(
            dict(
                scenario=path.parent.name,
                arm=path.name.split("___")[0][4:],
                run=int(path.stem.rsplit("__run_", 1)[1]),
                log=identity,
                valid_workpads=sum(
                    bool(e["payload"].get("state"))
                    for e in relevant
                    if e["event_type"] == "inquiry_update"
                ),
                complete_treatment=not failures,
                exclusions=failures,
            )
        )
    signatures = set()
    for call in calls:
        signature = (call["dialogue_log"], call["turn"], call["kind"])
        if signature in signatures or not call.get("ok"):
            raise ValueError("Duplicate or failed call")
        signatures.add(signature)
        if any(
            call.get(k) != accounting[signature].get(k)
            for k in ("usage", "finish_reason", "request_id")
        ):
            raise ValueError("Receipt/event mismatch")
    if (
        signatures != set(accounting)
        or len(calls) != 144
        or len(timings) != 72
        or not all(t["ok"] for t in timings)
    ):
        raise ValueError("Incomplete accounting")
    if len({(t["dialogue_log"], t["turn"]) for t in timings}) != 72:
        raise ValueError("Duplicate turn")
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
    groups = []
    for arm in ARMS:
        selected = [c for c in calls if c["arm"] == arm]
        usages = [c.get("usage") or {} for c in selected]
        inputs, missing_in = pilot.token_total(
            usages, ("input_tokens", "prompt_tokens")
        )
        outputs, missing_out = pilot.token_total(
            usages, ("output_tokens", "completion_tokens")
        )
        groups.append(
            dict(
                arm=arm,
                calls=len(selected),
                input_tokens=inputs,
                output_tokens=outputs,
                missing_input_usage_calls=missing_in,
                missing_output_usage_calls=missing_out,
                finish_reasons=dict(Counter(c.get("finish_reason") for c in selected)),
            )
        )
    return dict(
        generations=len(calls),
        replies=72,
        workpads=72,
        valid_workpads=sum(d["valid_workpads"] for d in dialogues),
        all_pairs=len(items),
        complete_pairs=len(eligible),
        by_arm=groups,
        dialogues=dialogues,
        quality_judgments="not collected; primary comparison unresolved",
    ), eligible


def run_comparison(output, live=False, selected_provider="mistral"):
    output = Path(output).expanduser().resolve()
    provider, model, planned, sources, dirty, catalog = preflight(
        output, live, selected_provider
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        mode="live" if live else "offline_dummy",
        provider=provider,
        model=model,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        source_tracked_dirty=dirty,
        source_files_sha256=sources,
        provider_call_limit=CALL_LIMIT,
        expected_reply_calls=72,
        expected_workpad_calls=72,
        automatic_retries=0,
        settings=SETTINGS,
        model_catalog=catalog,
        quality_judgments="not collected",
        subset_policy="Exclude a whole dialogue if any generation has abnormal/unknown finish or any workpad is invalid; keep all pairs.",
        execution_order=[
            dict(scenario=s, arm=a, repeat=x.run_index) for s, a, x in planned
        ],
    )
    calls, timings, active = [], [], {}
    original_build, original_turn = (
        lab.build_adapter,
        lab.RecursiveConclusionSession.user_turn,
    )
    pilot.write_json(output / "run_manifest.json", manifest)

    def build(*args, **kwargs):
        adapter = original_build(*args, **kwargs)
        generate = adapter.generate

        def measured(**kwargs):
            is_probe = kwargs["system"] in PROMPTS.values()
            limit = 600 if is_probe else 900
            if (
                not active
                or len(calls) >= CALL_LIMIT
                or kwargs["config"].max_tokens != limit
            ):
                raise RuntimeError("Call budget or generation context violated")
            entry = dict(
                call=len(calls) + 1,
                **active,
                kind="inquiry_update" if is_probe else "assistant_reply",
                prompt_sha256=hashlib.sha256(
                    (kwargs["system"] or "").encode()
                ).hexdigest(),
                output_token_limit=limit,
                temperature=kwargs["config"].temperature,
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
                    status="failed_or_unknown", ok=False, **failure_metadata(exc)
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
        for arm, prompt in PROMPTS.items():
            (output / f"probe_{arm}.txt").write_text(prompt)
        paths = []
        for scenario, arm, args in planned:
            out = Path(args.out_dir)
            paths += [
                lab.build_compare_log_path(
                    out,
                    provider=provider,
                    model=model,
                    arm_name=arm,
                    run_name=args.run_name,
                ),
                lab.build_compare_summary_path(
                    out, arm_name=arm, run_name=args.run_name
                ),
            ]
        lab.reserve_new_artifacts(paths)
        with (
            patch.object(lab, "build_adapter", build),
            patch.object(lab.RecursiveConclusionSession, "user_turn", turn),
        ):
            for scenario, arm, args in planned:
                active.clear()
                active.update(scenario=scenario, arm=arm, run=args.run_index)
                with patch.object(lab, "INQUIRY_PROBE_SYSTEM", PROMPTS[arm]):
                    execution = lab.execute_compare(args, artifacts_reserved=True)
                lab.write_compare_summary(
                    Path(args.out_dir),
                    execution.rows,
                    arm_name=arm,
                    run_name=args.run_name,
                )
                print(
                    f"Completed {scenario} {arm} repeat {args.run_index}: {len(calls)}/144 calls",
                    flush=True,
                )
        rubric = dict(
            title="Current reply action comparison",
            rubric_version="hold_action_v1",
            seed=7,
            out_dir=str(output / "review"),
            scenarios=[
                dict(
                    name=s,
                    display_name=s.title(),
                    compare_out_dir=str(output / "compare" / s),
                    arms=list(ARMS),
                )
                for s in SCENARIOS
            ],
            questions=[
                dict(
                    id="current_request",
                    prompt="Using the full history, which dialogue better performs the requested trial design, provisional recommendation, and constraint revision at turns 4–6? Quote evidence and counterevidence; do not require a particular activity.",
                    choices=["A", "B", "Tie"],
                )
            ],
        )
        pilot.write_json(output / "human_eval.json", rubric)
        items, key = review.build_items(rubric)
        if len(items) != 6:
            raise ValueError("Expected six paired comparisons")
        metrics, eligible = summarize(output, provider, calls, timings, items, key)
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
                config={**rubric, "subset_policy": manifest["subset_policy"]},
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument(
        "--live",
        action="store_true",
        help="Use the selected provider existing environment credential and fixed model",
    )
    parser.add_argument("--provider", choices=tuple(LIVE_MODELS), default="mistral")
    args = parser.parse_args()
    try:
        metrics = run_comparison(args.out, args.live, args.provider)
    except Exception as exc:
        print(f"Stopped ({type(exc).__name__}); preserve receipts, no automatic retry.")
        return 1
    print(
        f"Complete: {metrics['generations']} generations; {metrics['valid_workpads']}/72 valid workpads; {metrics['complete_pairs']}/6 complete pairs. Human judgments pending."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
