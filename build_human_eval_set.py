#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    return textwrap.dedent(
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
        - Short note: one or two sentences on why.

        ## Transcript A

        {transcript_a}

        ## Transcript B

        {transcript_b}
        """
    ).strip() + "\n"


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
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise SystemExit(f"Question {index} must be an object.")
        question_id = compact_text(str(question.get("id") or ""))
        prompt = compact_text(str(question.get("prompt") or ""))
        if not question_id or not prompt:
            raise SystemExit(f"Question {index} must include non-empty 'id' and 'prompt'.")
        choices = question.get("choices") or ["A", "B", "Tie"]
        if not isinstance(choices, list) or not choices:
            raise SystemExit(f"Question {index} choices must be a non-empty list.")
        normalized.append({"id": question_id, "prompt": prompt, "choices": [str(x) for x in choices]})
    return normalized


def load_summary_rows(compare_out_dir: Path, arm: str, run_index: int) -> list[dict[str, Any]]:
    path = compare_out_dir / f"summary__{arm}__run_{run_index:03d}.json"
    if not path.exists():
        raise SystemExit(f"Missing summary file: {path}")
    rows = load_json(path)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SystemExit(f"Summary must be a list of objects: {path}")
    return rows


def scenario_runs(compare_out_dir: Path, arms: list[str], explicit_runs: list[int] | None) -> list[int]:
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

    items: list[dict[str, Any]] = []
    blind_key: dict[str, Any] = {}
    item_counter = 1
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise SystemExit("Each scenario must be a JSON object.")
        scenario_name = compact_text(str(scenario.get("name") or ""))
        if not scenario_name:
            raise SystemExit("Scenario missing name.")
        display_name = compact_text(str(scenario.get("display_name") or scenario_name))
        compare_out_dir = Path(str(scenario.get("compare_out_dir") or ""))
        if not compare_out_dir.is_absolute():
            compare_out_dir = (Path.cwd() / compare_out_dir).resolve()
        arms = [compact_text(str(x)) for x in (scenario.get("arms") or []) if compact_text(str(x))]
        if len(arms) < 2:
            raise SystemExit(f"Scenario '{scenario_name}' needs at least two arms.")
        run_indexes = scenario_runs(compare_out_dir, arms, scenario.get("runs"))
        if not run_indexes:
            raise SystemExit(f"Scenario '{scenario_name}' has no runs in {compare_out_dir}")

        transcripts_by_run_arm: dict[tuple[int, str], dict[str, Any]] = {}
        for run_index in run_indexes:
            for arm in arms:
                rows = load_summary_rows(compare_out_dir, arm, run_index)
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
                }

        for run_index in run_indexes:
            for arm_left, arm_right in itertools.combinations(arms, 2):
                pair = [arm_left, arm_right]
                rng.shuffle(pair)
                label_map = {"A": pair[0], "B": pair[1]}
                item_id = f"item_{item_counter:03d}"
                item_counter += 1
                transcript_a = transcripts_by_run_arm[(run_index, label_map["A"])]["transcript"]
                transcript_b = transcripts_by_run_arm[(run_index, label_map["B"])]["transcript"]
                items.append(
                    {
                        "item_id": item_id,
                        "scenario_name": scenario_name,
                        "scenario_display_name": display_name,
                        "run_index": run_index,
                        "arm_pair": sorted([arm_left, arm_right]),
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
    packets_dir = out_dir / "packets"
    ensure_dir(packets_dir)

    scenario_counts: dict[str, int] = {}
    for item in items:
        scenario_counts[item["scenario_name"]] = scenario_counts.get(item["scenario_name"], 0) + 1

    manifest = {
        "seed": int(config.get("seed") or 7),
        "item_count": len(items),
        "questions": coerce_questions(config.get("questions")),
        "scenarios": config.get("scenarios") or [],
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

    (out_dir / "blind_key.json").write_text(
        json.dumps(blind_key, ensure_ascii=False, indent=2) + "\n",
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
    (out_dir / "booklet.md").write_text("\n".join(booklet_sections).rstrip() + "\n", encoding="utf-8")

    answer_sheet_path = out_dir / "answer_sheet.csv"
    questions = coerce_questions(config.get("questions"))
    fieldnames = ["item_id", "scenario_name", "run_index"]
    fieldnames.extend(str(question.get("id")) for question in questions)
    fieldnames.append("notes")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
