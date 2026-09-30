#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import secrets
import textwrap
from pathlib import Path
from typing import Any


DEFAULT_QUESTIONS = [
    {
        "id": "overall_better",
        "prompt": "Which transcript better balances delayed release and final articulation overall?",
        "choices": ["A", "B", "Tie"],
    },
    {
        "id": "leakage_control_better",
        "prompt": "Which transcript better avoids premature leakage without becoming inert?",
        "choices": ["A", "B", "Tie"],
    },
    {
        "id": "staged_release_better",
        "prompt": "Which transcript handles shortlist/support/final-packet staging more naturally?",
        "choices": ["A", "B", "Tie"],
    },
    {
        "id": "earned_final_better",
        "prompt": "Which final answer feels more earned by the earlier turns?",
        "choices": ["A", "B", "Tie"],
    },
]


def compact_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def render_transcript(rows: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for row in rows:
        turn = row.get("turn")
        user = str(row.get("user") or "").strip()
        assistant = str(row.get("assistant") or "").strip()
        chunks.append(
            textwrap.dedent(
                f"""\
                Turn {turn}
                User: {user}
                Assistant: {assistant}
                """
            ).strip()
        )
    return "\n\n".join(chunks).strip()


def render_packet_markdown(
    *,
    item_id: str,
    scenario_display_name: str,
    run_index: int,
    transcript_a: str,
    transcript_b: str,
    questions: list[dict[str, Any]],
) -> str:
    rubric_lines = []
    for question in questions:
        prompt = compact_text(str(question.get("prompt") or ""))
        rubric_lines.append(f"- {prompt} (`A` / `B` / `Tie`)")
    rubric = "\n".join(rubric_lines)
    return (
        textwrap.dedent(
            f"""\
        # {item_id}

        Scenario: {scenario_display_name}
        Run: {run_index}

        ## Instructions

        Compare transcript `A` and transcript `B` for the same scenario and run.
        Judge timing and articulation only from the visible dialogue. Do not infer model identity.
        The A/B order is randomized independently for each item.

        Record:
        {rubric}
        - Confidence: `low` / `medium` / `high`
        - Abstain only when the visible dialogue is insufficient to judge.
        - Short note: one or two sentences on why.

        ## Transcript A

        {transcript_a}

        ## Transcript B

        {transcript_b}
        """
        ).strip()
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a blind pairwise human-eval packet set from compare-matrix summaries."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON config describing scenarios, compare output dirs, and arms.",
    )
    return parser


def load_config(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise SystemExit("Config must be a JSON object.")
    return data


def coerce_questions(raw_questions: Any) -> list[dict[str, Any]]:
    questions = raw_questions or DEFAULT_QUESTIONS
    if not isinstance(questions, list) or not questions:
        raise SystemExit("Questions must be a non-empty list.")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise SystemExit(f"Question {index} must be an object.")
        question_id = compact_text(str(question.get("id") or ""))
        prompt = compact_text(str(question.get("prompt") or ""))
        if not question_id or not prompt:
            raise SystemExit(
                f"Question {index} must include non-empty 'id' and 'prompt'."
            )
        if question_id in seen_ids:
            raise SystemExit(f"Question {index} duplicates id '{question_id}'.")
        seen_ids.add(question_id)
        choices = question.get("choices") or ["A", "B", "Tie"]
        if not isinstance(choices, list) or not choices:
            raise SystemExit(f"Question {index} choices must be a non-empty list.")
        normalized_choices = [compact_text(str(choice)) for choice in choices]
        if set(normalized_choices) != {"A", "B", "Tie"} or len(normalized_choices) != 3:
            raise SystemExit(
                f"Question {index} choices must contain exactly 'A', 'B', and 'Tie'."
            )
        normalized.append(
            {"id": question_id, "prompt": prompt, "choices": normalized_choices}
        )
    return normalized


def load_summary(
    compare_out_dir: Path,
    arm: str,
    run_index: int,
) -> tuple[list[dict[str, Any]], Path, str]:
    path = compare_out_dir / f"summary__{arm}__run_{run_index:03d}.json"
    if not path.exists():
        raise SystemExit(f"Missing summary file: {path}")
    try:
        source_bytes = path.read_bytes()
        rows = json.loads(source_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read summary JSON: {path}") from exc
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SystemExit(f"Summary must be a list of objects: {path}")
    previous_turn = 0
    for index, row in enumerate(rows, start=1):
        turn = row.get("turn")
        user = row.get("user")
        assistant = row.get("assistant")
        if isinstance(turn, bool) or not isinstance(turn, int) or turn <= previous_turn:
            raise SystemExit(
                f"Summary row {index} needs a strictly increasing integer turn: {path}"
            )
        if not isinstance(user, str) or not user.strip():
            raise SystemExit(f"Summary row {index} has an empty user turn: {path}")
        if not isinstance(assistant, str) or not assistant.strip():
            raise SystemExit(f"Summary row {index} has an empty assistant turn: {path}")
        previous_turn = turn
    return rows, path, hashlib.sha256(source_bytes).hexdigest()


def scenario_runs(
    compare_out_dir: Path, arms: list[str], explicit_runs: list[int] | None
) -> list[int]:
    if explicit_runs:
        return sorted({int(x) for x in explicit_runs})
    runs: set[int] = set()
    for arm in arms:
        for path in compare_out_dir.glob(f"summary__{arm}__run_*.json"):
            stem = path.stem
            marker = "__run_"
            if marker not in stem:
                continue
            try:
                runs.add(int(stem.split(marker, 1)[1]))
            except ValueError:
                continue
    return sorted(runs)


def build_items(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(config.get("seed") or 7)
    rng = random.Random(seed)
    scenarios = config.get("scenarios") or []
    if not isinstance(scenarios, list) or not scenarios:
        raise SystemExit("Config must include a non-empty 'scenarios' list.")
    questions = coerce_questions(config.get("questions"))
    item_id_salt = compact_text(
        str(config.get("item_id_salt") or "")
    ) or secrets.token_hex(16)

    items: list[dict[str, Any]] = []
    blind_key: dict[str, Any] = {
        "_build_meta": {"item_id_salt": item_id_salt},
    }
    for scenario_index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise SystemExit("Each scenario must be a JSON object.")
        scenario_name = compact_text(str(scenario.get("name") or ""))
        if not scenario_name:
            raise SystemExit("Scenario missing name.")
        display_name = compact_text(str(scenario.get("display_name") or scenario_name))
        compare_out_dir = Path(str(scenario.get("compare_out_dir") or ""))
        if not compare_out_dir.is_absolute():
            compare_out_dir = (Path.cwd() / compare_out_dir).resolve()
        arms = [
            compact_text(str(x))
            for x in (scenario.get("arms") or [])
            if compact_text(str(x))
        ]
        if len(arms) < 2:
            raise SystemExit(f"Scenario '{scenario_name}' needs at least two arms.")
        if len(arms) != len(set(arms)):
            raise SystemExit(f"Scenario '{scenario_name}' contains duplicate arm ids.")
        run_indexes = scenario_runs(compare_out_dir, arms, scenario.get("runs"))
        if not run_indexes:
            raise SystemExit(
                f"Scenario '{scenario_name}' has no runs in {compare_out_dir}"
            )

        transcripts_by_run_arm: dict[tuple[int, str], dict[str, Any]] = {}
        for run_index in run_indexes:
            for arm in arms:
                rows, source_path, source_sha256 = load_summary(
                    compare_out_dir, arm, run_index
                )
                transcript = render_transcript(rows)
                if not transcript:
                    raise SystemExit(
                        f"Empty rendered transcript for scenario '{scenario_name}', arm '{arm}', run {run_index}."
                    )
                first = rows[0] if rows else {}
                transcripts_by_run_arm[(run_index, arm)] = {
                    "rows": rows,
                    "transcript": transcript,
                    "provider": first.get("provider"),
                    "model": first.get("model"),
                    "source_path": display_path(source_path),
                    "source_sha256": source_sha256,
                }

            reference_rows = transcripts_by_run_arm[(run_index, arms[0])]["rows"]
            reference_turns = [
                (int(row["turn"]), str(row["user"]).strip()) for row in reference_rows
            ]
            for arm in arms[1:]:
                candidate_rows = transcripts_by_run_arm[(run_index, arm)]["rows"]
                candidate_turns = [
                    (int(row["turn"]), str(row["user"]).strip())
                    for row in candidate_rows
                ]
                if candidate_turns != reference_turns:
                    raise SystemExit(
                        f"Scenario '{scenario_name}' run {run_index} has divergent user turns "
                        f"between arms '{arms[0]}' and '{arm}'."
                    )

        for run_index in run_indexes:
            for arm_left, arm_right in itertools.combinations(arms, 2):
                pair = [arm_left, arm_right]
                rng.shuffle(pair)
                label_map = {"A": pair[0], "B": pair[1]}
                private_identity = canonical_json(
                    {
                        "scenario_index": scenario_index,
                        "scenario_name": scenario_name,
                        "run_index": run_index,
                        "arm_pair_sorted": sorted((arm_left, arm_right)),
                    }
                )
                item_id = (
                    "item_"
                    + hashlib.sha256(
                        f"{item_id_salt}:{private_identity}".encode("utf-8")
                    ).hexdigest()[:16]
                )
                if item_id in blind_key:
                    raise SystemExit(
                        "Opaque item id collision; choose a new item_id_salt."
                    )
                transcript_a = transcripts_by_run_arm[(run_index, label_map["A"])][
                    "transcript"
                ]
                transcript_b = transcripts_by_run_arm[(run_index, label_map["B"])][
                    "transcript"
                ]
                items.append(
                    {
                        "item_id": item_id,
                        "scenario_name": scenario_name,
                        "scenario_display_name": display_name,
                        "run_index": run_index,
                        "labels": {
                            "A": {"transcript": transcript_a},
                            "B": {"transcript": transcript_b},
                        },
                        "questions": questions,
                    }
                )
                blind_key[item_id] = {
                    "scenario_name": scenario_name,
                    "scenario_display_name": display_name,
                    "run_index": run_index,
                    "compare_out_dir": str(compare_out_dir),
                    "label_to_arm": label_map,
                    "arm_pair_sorted": sorted([arm_left, arm_right]),
                    "labels": {
                        label: {
                            "arm": arm,
                            "source_path": transcripts_by_run_arm[(run_index, arm)][
                                "source_path"
                            ],
                            "source_sha256": transcripts_by_run_arm[(run_index, arm)][
                                "source_sha256"
                            ],
                        }
                        for label, arm in label_map.items()
                    },
                }
    rng.shuffle(items)
    return items, blind_key


def write_outputs(
    *,
    out_dir: Path,
    items: list[dict[str, Any]],
    blind_key: dict[str, Any],
    config: dict[str, Any],
) -> None:
    ensure_dir(out_dir)
    ready_path = out_dir / "READY.json"
    ready_path.unlink(missing_ok=True)
    packets_dir = out_dir / "packets"
    ensure_dir(packets_dir)
    for stale_packet in packets_dir.glob("item_*.md"):
        stale_packet.unlink()

    scenario_counts: dict[str, int] = {}
    for item in items:
        scenario_counts[item["scenario_name"]] = (
            scenario_counts.get(item["scenario_name"], 0) + 1
        )

    questions = coerce_questions(config.get("questions"))
    rubric_version = compact_text(
        str(config.get("rubric_version") or "staged_release_pairwise_v1")
    )
    manifest = {
        "schema": "rcl.blind_pairwise_eval_set.v2",
        "title": compact_text(
            str(config.get("title") or "Staged Release Pairwise Review")
        ),
        "rubric_version": rubric_version,
        "item_count": len(items),
        "questions": questions,
        "scenario_item_counts": scenario_counts,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    blinded_items_path = out_dir / "eval_items.jsonl"
    with blinded_items_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    source_records = sorted(
        {
            (str(label["source_path"]), str(label["source_sha256"]))
            for item in blind_key.values()
            for label in dict(item.get("labels") or {}).values()
        }
    )
    private_key = {
        "_meta": {
            "schema": "rcl.blind_pairwise_key.v2",
            "seed": int(config.get("seed") or 7),
            "rubric_version": rubric_version,
            "review_bundle_digest": json_digest({"manifest": manifest, "items": items}),
            "source_digest": json_digest(source_records),
            "private_config": config,
        },
        **blind_key,
    }
    blind_key_path = out_dir / "blind_key.json"
    blind_key_path.write_text(
        json.dumps(private_key, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    booklet_sections = [
        "# Blind Human Eval Set",
        "",
        "Use the matching `answer_sheet.csv` to record pairwise judgments.",
        "Each item compares the same scenario and run across two blinded arms.",
        "",
    ]
    for item in items:
        packet_text = render_packet_markdown(
            item_id=item["item_id"],
            scenario_display_name=item["scenario_display_name"],
            run_index=int(item["run_index"]),
            transcript_a=item["labels"]["A"]["transcript"],
            transcript_b=item["labels"]["B"]["transcript"],
            questions=item["questions"],
        )
        packet_path = packets_dir / f"{item['item_id']}.md"
        packet_path.write_text(packet_text, encoding="utf-8")
        booklet_sections.append(packet_text.rstrip())
        booklet_sections.append("")
    (out_dir / "booklet.md").write_text(
        "\n".join(booklet_sections).rstrip() + "\n", encoding="utf-8"
    )

    answer_sheet_path = out_dir / "answer_sheet.csv"
    fieldnames = ["item_id", "scenario_name", "run_index"]
    fieldnames.extend(str(question.get("id")) for question in questions)
    fieldnames.extend(["confidence", "abstain", "evidence", "counterevidence", "notes"])
    with answer_sheet_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "item_id": item["item_id"],
                    "scenario_name": item["scenario_name"],
                    "run_index": item["run_index"],
                }
            )

    ready = {
        "schema": "rcl.blind_pairwise_ready.v1",
        "review_bundle_digest": private_key["_meta"]["review_bundle_digest"],
        "source_digest": private_key["_meta"]["source_digest"],
    }
    temporary_ready_path = out_dir / f".READY.{secrets.token_hex(4)}.tmp"
    temporary_ready_path.write_text(
        json.dumps(ready, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_ready_path.replace(ready_path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    out_dir = Path(str(config.get("out_dir") or "human_eval_sets/blind_eval")).resolve()
    items, blind_key = build_items(config)
    write_outputs(out_dir=out_dir, items=items, blind_key=blind_key, config=config)
    print(f"Wrote {out_dir / 'manifest.json'}")
    print(f"Wrote {out_dir / 'eval_items.jsonl'}")
    print(f"Wrote {out_dir / 'booklet.md'}")
    print(f"Wrote {out_dir / 'answer_sheet.csv'}")
    print(f"Wrote {out_dir / 'blind_key.json'}")
    print(f"Published {out_dir / 'READY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
