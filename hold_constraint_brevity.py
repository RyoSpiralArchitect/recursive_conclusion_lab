#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded 2x2 reply experiment; product defaults and workpad prompt stay fixed."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch

import build_human_eval_set as review
import hold_action_comparison as previous
import hold_dialogue_pilot as pilot
import recursive_conclusion_lab as lab

REPO = Path(__file__).resolve().parent
ARMS = ("baseline", "constraints", "brief", "both")
CONSTRAINTS = (
    "\n- Carry forward the user's stated purpose until the user explicitly changes it. "
    "A third party's proposed metric is a proposal, not an adopted replacement goal. "
    "Preserve each user constraint's exact scope: actor, action, object, place, and time. "
    "Do not add exceptions merely because an item is checked, an action seems helpful, "
    "or permission might be obtainable; do not broaden a restriction to unrelated options. "
    "When a proposal conflicts with a constraint, revise the proposal or state the "
    "unresolved feasibility. Only an explicit user revision changes the constraint. "
    "Before replying, check that the proposed action fits the still-active constraints "
    "and serves the original purpose; show only the useful answer, not this check."
)
BREVITY = (
    "\n- Complete this reply in at most 180 English words, preferably 90-150. "
    "Lead with the direct answer or concrete provisional move, keep only the reasoning "
    "needed to use it, and retain material uncertainty or a condition that could change it. "
    "Use at most one necessary follow-up question. Omit repeated context, long menus, "
    "and closing offers to do more. Finish the answer within this length; brevity must "
    "not hide a feasibility problem or force a final conclusion."
)
CONTROL = lab.INQUIRY_REPLY_SYSTEM
PROBE = lab.INQUIRY_PROBE_SYSTEM
PROMPTS = {
    "baseline": CONTROL,
    "constraints": CONTROL + CONSTRAINTS,
    "brief": CONTROL + BREVITY,
    "both": CONTROL + CONSTRAINTS + BREVITY,
}
CASES = (
    ("tool_returns", "hold_scope"),
    ("craft_pickup", "hold_scope"),
    ("gallery_access", "hold_scope"),
    ("repair_learning", "hold_purpose"),
)
SUITES = {"scope": (0, 1), "anchor": (2, 3)}
# Each row occurs twice across the two suites, balancing every arm/position.
ROWS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
CONTRASTS = (
    ("baseline", "constraints"),
    ("baseline", "brief"),
    ("constraints", "both"),
    ("brief", "both"),
)
CALL_LIMIT = 192  # Per suite; two suites, at most 384 generations in total.
SETTINGS = previous.SETTINGS.copy()
DESIGN = "docs/experiments/hold-constraint-brevity.md"
FROZEN = "docs/experiments/data/hold_constraint_brevity_frozen.json"
SOURCE = (
    "hold_constraint_brevity.py",
    "hold_action_comparison.py",
    "hold_dialogue_pilot.py",
    "recursive_conclusion_lab.py",
    "reply_contract.py",
    "inquiry_state.py",
    "build_human_eval_set.py",
    "analyze_runs.py",
    DESIGN,
    FROZEN,
    "docs/experiments/data/hold_scope_facts.json",
    "docs/experiments/data/hold_purpose_facts.json",
    *(f"protocol_scripts/{prefix}_{name}.json" for name, prefix in CASES),
)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def preflight(output, live, suite):
    if suite not in SUITES:
        raise ValueError("Unsupported suite")
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists")
    frozen = pilot.read_json(REPO / FROZEN)
    if (
        frozen["reply_sha256"] != {a: sha(p) for a, p in PROMPTS.items()}
        or frozen["probe_sha256"] != sha(PROBE)
        or frozen["settings"] != SETTINGS
    ):
        raise ValueError("Prompt or settings differ from the frozen design")
    provider, model = ("openai", "gpt-5.6-luna") if live else ("dummy", "dummy-v1")
    if live and not os.environ.get("OPENAI_API_KEY", "").strip():
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
    planned = []
    for repeat in (1, 2):
        for index in SUITES[suite]:
            scenario, prefix = CASES[index]
            script = REPO / f"protocol_scripts/{prefix}_{scenario}.json"
            system, turns = lab.load_script(script)
            facts = pilot.read_json(REPO / f"docs/experiments/data/{prefix}_facts.json")
            if (
                len(turns) != 6
                or facts["scenarios"][scenario]["user_turns_exact"]
                != {str(i): text for i, text in enumerate(turns, 1)}
                or pilot.digest(script) != frozen["script_sha256"][scenario]
            ):
                raise ValueError("Script differs from frozen facts")
            for position in ROWS[(index + 2 * (repeat - 1)) % 4]:
                arm = ARMS[position]
                args = lab.build_compare_args_from_config(
                    dict(
                        script=str(script),
                        providers=[f"{provider}={model}"],
                        out_dir=str(output / "compare" / scenario),
                        args=SETTINGS,
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
                expected_system = "\n\n".join(
                    (lab.compact_text(system), PROMPTS[arm])
                ).strip()
                planned.append((scenario, arm, args, expected_system))
    catalog = None
    if live:
        import requests

        response = requests.get(
            previous.CATALOG_URLS[provider],
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
        catalog = {k: row[k] for k in ("id", "name", "capabilities") if k in row}
    return provider, model, planned, sources, dirty, catalog


def completion_metrics(output, provider):
    """Mechanical observations only; length is not a semantic quality score."""
    rows = []
    for path in sorted((output / "compare").glob("*/*.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event_type"] != "assistant_reply":
                continue
            p = event["payload"]
            text = p.get("assistant", "")
            words = len(text.split())
            rows.append(
                dict(
                    scenario=path.parent.name,
                    arm=path.name.split("___")[0][4:],
                    run=int(path.stem.rsplit("__run_", 1)[1]),
                    turn=event["turn_index"],
                    words=words,
                    normal_finish=p.get("finish_reason")
                    in pilot.NORMAL_FINISH[provider],
                    nonempty=bool(text.strip()),
                )
            )
    by_arm = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        by_arm[arm] = dict(
            replies=len(selected),
            median_words=statistics.median(r["words"] for r in selected),
            max_words=max(r["words"] for r in selected),
            normal_finish=sum(r["normal_finish"] for r in selected),
            within_180_words=sum(r["words"] <= 180 and r["nonempty"] for r in selected),
            short_complete=sum(
                r["normal_finish"] and r["nonempty"] and r["words"] <= 180
                for r in selected
            ),
            truncated_by_turn=dict(
                Counter(r["turn"] for r in selected if not r["normal_finish"])
            ),
        )
    return dict(
        definition="Whitespace-delimited words, nonempty, normal provider finish; not semantic adequacy.",
        by_arm=by_arm,
        turns=rows,
    )


def run_comparison(output, live=False, suite="scope"):
    output = Path(output).expanduser().resolve()
    provider, model, planned, sources, dirty, catalog = preflight(output, live, suite)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        mode="live" if live else "offline_dummy",
        suite=suite,
        provider=provider,
        model=model,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        source_tracked_dirty=dirty,
        source_files_sha256=sources,
        provider_call_limit=CALL_LIMIT,
        expected_reply_calls=96,
        expected_workpad_calls=96,
        automatic_retries=0,
        settings=SETTINGS,
        model_catalog=catalog,
        probe_sha256=sha(PROBE),
        reply_instruction_sha256={a: sha(p) for a, p in PROMPTS.items()},
        quality_judgments="not collected",
        subset_policy="Exclude a whole dialogue if any generation has abnormal/unknown finish or any workpad is invalid; keep all planned pairs.",
        execution_order=[
            dict(scenario=s, arm=a, repeat=x.run_index) for s, a, x, _ in planned
        ],
        contrasts=CONTRASTS,
    )
    calls, timings, active = [], [], {}
    original_build, original_turn = (
        lab.build_adapter,
        lab.RecursiveConclusionSession.user_turn,
    )
    pilot.write_json(output / "run_manifest.json", manifest)
    expected_reply_system = None

    def build(*args, **kwargs):
        adapter = original_build(*args, **kwargs)
        generate = adapter.generate

        def measured(**kwargs):
            system = kwargs["system"]
            is_probe = system == PROBE
            limit, temperature = (600, 0.0) if is_probe else (900, 0.2)
            if (
                not active
                or len(calls) >= CALL_LIMIT
                or kwargs["config"].max_tokens != limit
                or kwargs["config"].temperature != temperature
                or (not is_probe and system != expected_reply_system)
            ):
                raise RuntimeError("Call budget or generation context violated")
            entry = dict(
                call=len(calls) + 1,
                **active,
                kind="inquiry_update" if is_probe else "assistant_reply",
                prompt_sha256=sha(system),
                output_token_limit=limit,
                temperature=temperature,
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
                    status="failed_or_unknown",
                    ok=False,
                    **previous.failure_metadata(exc),
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
        (output / "probe.txt").write_text(PROBE)
        for arm, prompt in PROMPTS.items():
            (output / f"reply_{arm}.txt").write_text(prompt)
        paths = []
        for scenario, arm, args, _ in planned:
            out = Path(args.out_dir)
            paths.extend(
                (
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
                )
            )
        lab.reserve_new_artifacts(paths)
        with (
            patch.object(lab, "build_adapter", build),
            patch.object(lab.RecursiveConclusionSession, "user_turn", turn),
        ):
            for scenario, arm, args, expected_reply_system in planned:
                active.clear()
                active.update(scenario=scenario, arm=arm, run=args.run_index)
                with patch.object(lab, "INQUIRY_REPLY_SYSTEM", PROMPTS[arm]):
                    execution = lab.execute_compare(args, artifacts_reserved=True)
                lab.write_compare_summary(
                    Path(args.out_dir),
                    execution.rows,
                    arm_name=arm,
                    run_name=args.run_name,
                )
                print(
                    f"Completed {scenario} {arm} repeat {args.run_index}: {len(calls)}/192 calls",
                    flush=True,
                )
        rubric = dict(
            title="Purpose, constraint scope and concise completion",
            rubric_version="hold_constraint_brevity_v1",
            seed=7,
            out_dir=str(output / "review"),
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
                dict(id=key, prompt=prompt, choices=["A", "B", "Tie"])
                for key, prompt in (
                    (
                        "scope",
                        "Which dialogue better preserves the exact scope of active constraints, without unauthorized exceptions or broadening? Distinguish an explicitly conditional suggestion from treating permission as established. Quote evidence and counterevidence or abstain.",
                    ),
                    (
                        "purpose",
                        "Which dialogue better serves the original user purpose unless the user explicitly revises it? A third-party metric is a suggestion. Quote evidence and counterevidence or abstain.",
                    ),
                    (
                        "useful_completion",
                        "Which dialogue answers the request concisely and completely while keeping material uncertainty and conclusions revisable? Shortness alone is insufficient. Read turns 4-6 and their earlier context; quote evidence and counterevidence or abstain.",
                    ),
                )
            ],
        )
        pilot.write_json(output / "human_eval.json", rubric)
        items, key = review.build_items(rubric)
        edges = {tuple(sorted(pair)) for pair in CONTRASTS}
        items = [
            i for i in items if tuple(key[i["item_id"]]["arm_pair_sorted"]) in edges
        ]
        key = {
            "_build_meta": key["_build_meta"],
            **{i["item_id"]: key[i["item_id"]] for i in items},
        }
        if len(items) != 16:
            raise ValueError("Expected 16 predeclared paired contrasts")
        metrics, eligible = previous.summarize(
            output, provider, calls, timings, items, key, arms=ARMS, expected_calls=192
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
                config={**rubric, "subset_policy": manifest["subset_policy"]},
            )
        if any(pilot.digest(REPO / s) != h for s, h in sources.items()):
            raise ValueError("Source changed during execution")
        pilot.write_json(output / "metrics.json", metrics)
        pilot.write_json(
            output / "completion_metrics.json", completion_metrics(output, provider)
        )
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
        f"Complete: {result['generations']} generations; {result['valid_workpads']}/96 valid workpads. Independent judgments pending."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
