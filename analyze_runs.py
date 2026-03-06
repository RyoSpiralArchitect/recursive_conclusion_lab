#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


DEFERRED_REALIZATION_THRESHOLD = 0.10
CONCLUSION_MENTION_THRESHOLD = 0.10


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def lexical_overlap(a: str, b: str) -> float:
    toks_a = {t.lower() for t in re.findall(r"\w+", a)}
    toks_b = {t.lower() for t in re.findall(r"\w+", b)}
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def mean_or_none(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return statistics.fmean(vals)


def count_strings(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = compact_text(str(value))
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def parse_confidence(text: str) -> float | None:
    m = re.search(r"CONFIDENCE:\s*([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    if 0.0 <= value <= 1.0:
        return value
    return None


def extract_conclusion_line(text: str) -> str:
    m = re.search(r"(?im)^\s*conclusion\s*:\s*(.+?)\s*$", text or "")
    if not m:
        return ""
    return m.group(1).strip()


def keyword_coverage(keywords: list[str], text: str) -> float | None:
    if not keywords:
        return None
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits / len(keywords)


def keyword_hits(keywords: list[str], text: str) -> int | None:
    if not keywords:
        return None
    lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in lower)


def load_evaluation_spec(script_path: Path | None) -> dict[str, Any]:
    if script_path is None:
        return {}
    data = json.loads(script_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    evaluation = data.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        return {}
    return evaluation


def coerce_usage_num(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def summarize_log(path: Path, evaluation: dict[str, Any]) -> dict[str, Any]:
    events = read_jsonl(path)
    provider = None
    model = None

    assistant_rows: list[dict[str, Any]] = []
    assistant_text_by_turn: dict[int, str] = {}
    probes: list[str] = []
    probe_events: list[tuple[int, str]] = []
    probe_plan_events: list[dict[str, Any]] = []
    delayed_mention_plans: list[dict[str, Any]] = []
    delayed_mention_actions: list[dict[str, Any]] = []
    delayed_mention_plan_error_count = 0
    capsules: list[str] = []
    planned_intents: list[dict[str, Any]] = []
    intent_status_by_id: dict[str, str] = {}
    inband_prune_events: list[dict[str, Any]] = []
    plan_probe_call_count = 0

    for row in events:
        provider = provider or row.get("provider")
        model = model or row.get("model")
        et = row.get("event_type")
        payload = row.get("payload") or {}
        if et == "assistant_reply":
            assistant_rows.append(payload)
            turn_idx = row.get("turn_index")
            if isinstance(turn_idx, int):
                assistant_text_by_turn[turn_idx] = str(payload.get("assistant") or "")
        elif et == "conclusion_probe":
            hypothesis = str(payload.get("hypothesis", "") or "")
            probes.append(hypothesis)
            turn_idx = row.get("turn_index")
            if isinstance(turn_idx, int):
                raw_line = str(payload.get("conclusion_line", "") or "")
                line_text = extract_conclusion_line(raw_line) or extract_conclusion_line(hypothesis)
                if line_text:
                    probe_events.append((turn_idx, line_text))
                    raw_keywords = payload.get("keywords") or []
                    keywords: list[str] = []
                    if isinstance(raw_keywords, list):
                        keywords = [str(x) for x in raw_keywords if x]
                    elif isinstance(raw_keywords, str):
                        keywords = [x.strip() for x in re.split(r"[;,|\n]\s*", raw_keywords) if x.strip()]
                    probe_plan_events.append(
                        {
                            "turn": turn_idx,
                            "conclusion_line": line_text,
                            "keywords": keywords,
                            "mention_delay_min_turns": payload.get("mention_delay_min_turns"),
                            "mention_delay_max_turns": payload.get("mention_delay_max_turns"),
                            "mention_likelihood": payload.get("mention_likelihood"),
                            "delay_strategy": payload.get("delay_strategy"),
                            "delay_signals": payload.get("delay_signals") or [],
                        }
                    )
        elif et == "memory_capsule":
            capsules.append(payload.get("capsule", ""))
        elif et == "deferred_intent_plan":
            plan_probe_call_count += 1
            if payload.get("created"):
                if isinstance(payload.get("intent"), dict):
                    planned_intents.append(payload["intent"])
                created_items = payload.get("created_intents") or []
                if isinstance(created_items, list):
                    for item in created_items:
                        if isinstance(item, dict):
                            planned_intents.append(item)
        elif et == "deferred_intent_plan_error":
            plan_probe_call_count += 1
        elif et == "deferred_intent_decision":
            for decision in payload.get("decisions") or []:
                if isinstance(decision, dict) and decision.get("intent_id"):
                    intent_status_by_id[str(decision["intent_id"])] = str(decision.get("status_after", ""))
        elif et == "inband_state_prune":
            if isinstance(payload, dict):
                inband_prune_events.append(payload)
        elif et == "delayed_mention_plan":
            if isinstance(payload, dict):
                if isinstance(payload.get("item"), dict):
                    delayed_mention_plans.append(payload["item"])
                created_items = payload.get("created_items") or []
                if isinstance(created_items, list):
                    for item in created_items:
                        if isinstance(item, dict):
                            delayed_mention_plans.append(item)
        elif et == "delayed_mention_plan_error":
            delayed_mention_plan_error_count += 1
        elif et == "delayed_mention_action":
            if isinstance(payload, dict):
                delayed_mention_actions.append(payload)

    final_assistant = assistant_rows[-1].get("assistant", "") if assistant_rows else ""
    final_user = assistant_rows[-1].get("user", "") if assistant_rows else ""
    final_probe = probes[-1] if probes else ""

    probe_confidences = [c for c in (parse_confidence(p) for p in probes) if c is not None]
    probe_stabilities = [
        lexical_overlap(a, b) for a, b in zip(probes[:-1], probes[1:]) if a and b
    ]
    per_turn_overlap = [
        float(r.get("probe_reply_overlap", 0.0) or 0.0)
        for r in assistant_rows
        if r.get("conclusion_probe")
    ]

    conclusion_line_probe_count = len(probe_events)
    conclusion_line_mention_count = 0
    conclusion_line_immediate_count = 0
    conclusion_line_mention_delays: list[int] = []
    assistant_turns_sorted = sorted(assistant_text_by_turn.keys())
    for probe_turn, conclusion_line in probe_events:
        for turn_idx in assistant_turns_sorted:
            if turn_idx < probe_turn:
                continue
            if (
                lexical_overlap(conclusion_line, assistant_text_by_turn.get(turn_idx, ""))
                >= CONCLUSION_MENTION_THRESHOLD
            ):
                conclusion_line_mention_count += 1
                delay = turn_idx - probe_turn
                conclusion_line_mention_delays.append(delay)
                if delay == 0:
                    conclusion_line_immediate_count += 1
                break

    conclusion_any_mention_count = 0
    conclusion_any_mention_delays: list[int] = []
    conclusion_plan_window_defined_count = 0
    conclusion_plan_within_window_count = 0
    conclusion_plan_early_count = 0
    conclusion_plan_late_count = 0
    conclusion_plan_missing_count = 0
    planned_conclusion_delay_mins: list[int] = []
    planned_conclusion_delay_maxs: list[int] = []
    planned_conclusion_window_widths: list[int] = []
    conclusion_delay_strategies: list[str] = []
    conclusion_delay_signals: list[str] = []
    conclusion_mention_likelihoods: list[float] = []

    for plan in probe_plan_events:
        probe_turn = plan.get("turn")
        if not isinstance(probe_turn, int):
            continue
        conclusion_line = str(plan.get("conclusion_line") or "")
        keywords = plan.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(x) for x in keywords if x]

        delay_min = plan.get("mention_delay_min_turns")
        delay_max = plan.get("mention_delay_max_turns")
        has_window = isinstance(delay_min, int) and isinstance(delay_max, int) and delay_max >= delay_min
        if has_window:
            conclusion_plan_window_defined_count += 1
            planned_conclusion_delay_mins.append(delay_min)
            planned_conclusion_delay_maxs.append(delay_max)
            planned_conclusion_window_widths.append(delay_max - delay_min)

        strat = plan.get("delay_strategy")
        if strat:
            conclusion_delay_strategies.append(str(strat))
        sigs = plan.get("delay_signals") or []
        if isinstance(sigs, list):
            conclusion_delay_signals.extend(str(x) for x in sigs if x)
        likelihood = plan.get("mention_likelihood")
        if isinstance(likelihood, (int, float)):
            conclusion_mention_likelihoods.append(float(likelihood))

        mention_delay: int | None = None
        for turn_idx in assistant_turns_sorted:
            if turn_idx < probe_turn:
                continue
            text = assistant_text_by_turn.get(turn_idx, "")
            line_ok = (
                bool(conclusion_line)
                and lexical_overlap(conclusion_line, text) >= CONCLUSION_MENTION_THRESHOLD
            )
            kw_ok = False
            if keywords:
                hits = keyword_hits(keywords, text) or 0
                required_hits = max(2, (len(keywords) + 1) // 2)
                kw_ok = hits >= required_hits
            if line_ok or kw_ok:
                mention_delay = turn_idx - probe_turn
                conclusion_any_mention_count += 1
                conclusion_any_mention_delays.append(mention_delay)
                if has_window:
                    if mention_delay < delay_min:
                        conclusion_plan_early_count += 1
                    elif mention_delay > delay_max:
                        conclusion_plan_late_count += 1
                    else:
                        conclusion_plan_within_window_count += 1
                break

        if mention_delay is None and has_window:
            conclusion_plan_missing_count += 1

    conclusion_line_mention_rate = (
        conclusion_line_mention_count / conclusion_line_probe_count
        if conclusion_line_probe_count
        else None
    )
    conclusion_line_immediate_mention_rate = (
        conclusion_line_immediate_count / conclusion_line_probe_count
        if conclusion_line_probe_count
        else None
    )
    conclusion_any_mention_rate = (
        conclusion_any_mention_count / conclusion_line_probe_count
        if conclusion_line_probe_count
        else None
    )
    conclusion_plan_within_window_rate = (
        conclusion_plan_within_window_count / conclusion_plan_window_defined_count
        if conclusion_plan_window_defined_count
        else None
    )
    conclusion_plan_early_rate = (
        conclusion_plan_early_count / conclusion_plan_window_defined_count
        if conclusion_plan_window_defined_count
        else None
    )
    conclusion_plan_late_rate = (
        conclusion_plan_late_count / conclusion_plan_window_defined_count
        if conclusion_plan_window_defined_count
        else None
    )
    conclusion_plan_missing_rate = (
        conclusion_plan_missing_count / conclusion_plan_window_defined_count
        if conclusion_plan_window_defined_count
        else None
    )
    reply_word_counts = [len(re.findall(r"\w+", r.get("assistant", ""))) for r in assistant_rows]
    last_turn_alignment = lexical_overlap(final_user, final_assistant)
    final_probe_reply_overlap = lexical_overlap(final_probe, final_assistant)

    input_tokens = []
    output_tokens = []
    for r in assistant_rows:
        usage = r.get("usage") or {}
        if isinstance(usage, dict):
            input_tokens.append(
                coerce_usage_num(
                    usage.get("input_tokens")
                    or usage.get("prompt_tokens")
                    or usage.get("inputTokenCount")
                )
            )
            output_tokens.append(
                coerce_usage_num(
                    usage.get("output_tokens")
                    or usage.get("completion_tokens")
                    or usage.get("candidatesTokenCount")
                )
            )

    fire_overlaps: list[float] = []
    fire_count = 0
    realized_fire_count = 0
    injected_fire_count = 0
    cancel_count = 0
    expire_count = 0
    revise_count = 0
    premature_fire_count = 0
    stale_fire_count = 0
    fire_turn_gaps: list[int] = []
    fire_window_widths: list[int] = []
    fire_window_position_ratios: list[float] = []
    fire_at_earliest_count = 0
    fire_at_latest_count = 0
    seen_fired_ids: set[str] = set()
    seen_terminal_ids: set[str] = set()

    for row in assistant_rows:
        current_turn = None
        if isinstance(row.get("due_deferred_intents"), list) and row["due_deferred_intents"]:
            current_turn = row["due_deferred_intents"][0].get("fire_turn")
        for action in row.get("deferred_intent_actions") or []:
            if not isinstance(action, dict):
                continue
            intent_id = str(action.get("intent_id", ""))
            act = str(action.get("action", "")).lower()
            if intent_id and action.get("status_after"):
                intent_status_by_id[intent_id] = str(action.get("status_after"))
            if act == "fire":
                fire_count += 1
                if intent_id:
                    seen_fired_ids.add(intent_id)
                overlap = action.get("assistant_overlap")
                if isinstance(overlap, (int, float)):
                    fire_overlaps.append(float(overlap))
                    if float(overlap) >= DEFERRED_REALIZATION_THRESHOLD:
                        realized_fire_count += 1
                if action.get("injected"):
                    injected_fire_count += 1
                turn = action.get("created_turn")
                fire_turn = action.get("fire_turn")
                if isinstance(turn, int) and isinstance(fire_turn, int):
                    fire_turn_gaps.append(fire_turn - turn)
                earliest_turn = action.get("earliest_turn")
                latest_turn = action.get("latest_turn")
                fired_now = action.get("fire_turn")
                if isinstance(fired_now, int) and isinstance(earliest_turn, int) and fired_now < earliest_turn:
                    premature_fire_count += 1
                if isinstance(fired_now, int) and isinstance(latest_turn, int) and fired_now > latest_turn:
                    stale_fire_count += 1
            elif act == "cancel":
                cancel_count += 1
                if intent_id:
                    seen_terminal_ids.add(intent_id)
            elif act == "expire":
                expire_count += 1
                if intent_id:
                    seen_terminal_ids.add(intent_id)
            elif act == "revise":
                revise_count += 1

    # Assistant payload does not explicitly carry fire_turn, so infer from event order.
    if fire_count:
        fire_overlaps = []
        realized_fire_count = 0
        injected_fire_count = 0
        premature_fire_count = 0
        stale_fire_count = 0
        fire_turn_gaps = []
        fire_window_widths = []
        fire_window_position_ratios = []
        fire_at_earliest_count = 0
        fire_at_latest_count = 0
        fire_count = 0
        for turn_idx, row in enumerate(assistant_rows, start=1):
            for action in row.get("deferred_intent_actions") or []:
                if not isinstance(action, dict):
                    continue
                if str(action.get("action", "")).lower() != "fire":
                    continue
                fire_count += 1
                overlap = action.get("assistant_overlap")
                if isinstance(overlap, (int, float)):
                    fire_overlaps.append(float(overlap))
                    if float(overlap) >= DEFERRED_REALIZATION_THRESHOLD:
                        realized_fire_count += 1
                if action.get("injected"):
                    injected_fire_count += 1
                created_turn = action.get("created_turn")
                earliest_turn = action.get("earliest_turn")
                latest_turn = action.get("latest_turn")
                if isinstance(created_turn, int):
                    fire_turn_gaps.append(turn_idx - created_turn)
                if isinstance(earliest_turn, int) and turn_idx < earliest_turn:
                    premature_fire_count += 1
                if isinstance(latest_turn, int) and turn_idx > latest_turn:
                    stale_fire_count += 1
                if isinstance(earliest_turn, int) and isinstance(latest_turn, int):
                    width = latest_turn - earliest_turn
                    if width < 0:
                        continue
                    fire_window_widths.append(width)
                    if turn_idx == earliest_turn:
                        fire_at_earliest_count += 1
                    if turn_idx == latest_turn:
                        fire_at_latest_count += 1
                    if width > 0:
                        fire_window_position_ratios.append((turn_idx - earliest_turn) / width)
                    else:
                        fire_window_position_ratios.append(0.0)

    planned_ids = {
        str(item.get("intent_id"))
        for item in planned_intents
        if isinstance(item, dict) and item.get("intent_id")
    }
    planned_unique: list[dict[str, Any]] = []
    if planned_ids:
        by_id: dict[str, dict[str, Any]] = {}
        for item in planned_intents:
            if not isinstance(item, dict) or not item.get("intent_id"):
                continue
            by_id[str(item["intent_id"])] = item
        planned_unique = list(by_id.values())
    active_final_count = sum(1 for status in intent_status_by_id.values() if status == "active")

    plan_strategies = [
        str(item.get("plan_strategy") or "")
        for item in planned_unique
        if isinstance(item, dict)
    ]
    plan_signals: list[str] = []
    for item in planned_unique:
        signals = item.get("plan_signals") or []
        if isinstance(signals, list):
            plan_signals.extend(str(x) for x in signals if x)

    decision_strategies: list[str] = []
    decision_signals: list[str] = []
    for row in assistant_rows:
        for action in row.get("deferred_intent_actions") or []:
            if not isinstance(action, dict):
                continue
            act = str(action.get("action", "")).lower()
            if act not in {"fire", "cancel", "expire", "revise"}:
                continue
            ds = action.get("decision_strategy")
            if ds:
                decision_strategies.append(str(ds))
            sigs = action.get("decision_signals") or []
            if isinstance(sigs, list):
                decision_signals.extend(str(x) for x in sigs if x)

    plan_strategy_counts = count_strings(plan_strategies)
    plan_signal_counts = count_strings(plan_signals)
    decision_strategy_counts = count_strings(decision_strategies)
    decision_signal_counts = count_strings(decision_signals)

    deferred_intent_backend = None
    deferred_intent_timing = None
    deferred_intent_plan_policy = None
    deferred_intent_plan_budget = None
    deferred_intent_plan_probe_calls = 0
    for row in assistant_rows:
        if deferred_intent_backend is None:
            backend = row.get("deferred_intent_backend")
            if backend:
                deferred_intent_backend = str(backend)
        if deferred_intent_timing is None:
            timing = row.get("deferred_intent_timing")
            if timing:
                deferred_intent_timing = str(timing)
        if deferred_intent_plan_policy is None:
            policy = row.get("deferred_intent_plan_policy")
            if policy:
                deferred_intent_plan_policy = str(policy)
        if deferred_intent_plan_budget is None:
            budget = row.get("deferred_intent_plan_budget")
            if isinstance(budget, int):
                deferred_intent_plan_budget = budget
        probe_calls = row.get("deferred_intent_plan_probe_calls")
        if isinstance(probe_calls, int):
            deferred_intent_plan_probe_calls = max(deferred_intent_plan_probe_calls, probe_calls)

    planned_delay_mins: list[int] = []
    planned_delay_maxs: list[int] = []
    planned_window_widths: list[int] = []
    for item in planned_unique:
        created_turn = item.get("created_turn")
        earliest_turn = item.get("earliest_turn")
        latest_turn = item.get("latest_turn")
        if not (
            isinstance(created_turn, int)
            and isinstance(earliest_turn, int)
            and isinstance(latest_turn, int)
        ):
            continue
        planned_delay_mins.append(earliest_turn - created_turn)
        planned_delay_maxs.append(latest_turn - created_turn)
        planned_window_widths.append(latest_turn - earliest_turn)

    inband_state_errors = [
        str(row.get("inband_state_error"))
        for row in assistant_rows
        if row.get("inband_state_error")
    ]
    inband_state_sizes = [
        float(row.get("inband_state_chars"))
        for row in assistant_rows
        if isinstance(row.get("inband_state_chars"), (int, float))
    ]
    pruned_intents_total = 0
    for item in inband_prune_events:
        pruned_ids = item.get("pruned_intent_ids") or []
        if isinstance(pruned_ids, list):
            pruned_intents_total += sum(1 for x in pruned_ids if x)

    final_required = [str(x) for x in (evaluation.get("final_required_keywords") or [])]
    conversation_required = [str(x) for x in (evaluation.get("conversation_required_keywords") or [])]
    final_forbidden = [str(x) for x in (evaluation.get("final_forbidden_keywords") or [])]

    conversation_text = "\n".join(r.get("assistant", "") for r in assistant_rows)

    delayed_mention_by_id: dict[str, dict[str, Any]] = {}
    for item in delayed_mention_plans:
        if not isinstance(item, dict):
            continue
        item_id = item.get("item_id")
        if not item_id:
            continue
        delayed_mention_by_id[str(item_id)] = item
    delayed_mention_unique = list(delayed_mention_by_id.values())
    delayed_mention_planned_count = len(delayed_mention_unique)
    delayed_mention_planned_nonconclusion_count = sum(
        1
        for item in delayed_mention_unique
        if str(item.get("kind") or "").strip().lower() != "conclusion"
    )

    mention_actions = [
        act
        for act in delayed_mention_actions
        if isinstance(act, dict) and str(act.get("action") or "").lower() == "mention"
    ]
    expire_actions = [
        act
        for act in delayed_mention_actions
        if isinstance(act, dict) and str(act.get("action") or "").lower() == "expire"
    ]
    delayed_mention_mentioned_count = len(mention_actions)
    delayed_mention_expired_count = len(expire_actions)
    delayed_mention_mentioned_nonconclusion_count = sum(
        1
        for act in mention_actions
        if str(act.get("kind") or "").strip().lower() != "conclusion"
    )
    delayed_mention_within_window_count = sum(
        1 for act in mention_actions if act.get("within_window") is True
    )
    delayed_mention_early_count = 0
    delayed_mention_late_count = 0
    delayed_mention_delays: list[int] = []
    delayed_mention_injected_on_mention_count = 0
    for act in mention_actions:
        mt = act.get("mention_turn")
        et = act.get("earliest_turn")
        lt = act.get("latest_turn")
        if isinstance(mt, int) and isinstance(et, int) and mt < et:
            delayed_mention_early_count += 1
        if isinstance(mt, int) and isinstance(lt, int) and mt > lt:
            delayed_mention_late_count += 1
        delay = act.get("delay_turns")
        if isinstance(delay, int):
            delayed_mention_delays.append(delay)
        if act.get("injected") is True:
            delayed_mention_injected_on_mention_count += 1

    delayed_mention_mention_rate = (
        delayed_mention_mentioned_count / delayed_mention_planned_count
        if delayed_mention_planned_count
        else None
    )
    delayed_mention_nonconclusion_mention_rate = (
        delayed_mention_mentioned_nonconclusion_count / delayed_mention_planned_nonconclusion_count
        if delayed_mention_planned_nonconclusion_count
        else None
    )
    delayed_mention_within_window_rate = (
        delayed_mention_within_window_count / delayed_mention_planned_count
        if delayed_mention_planned_count
        else None
    )
    delayed_mention_injected_on_mention_rate = (
        delayed_mention_injected_on_mention_count / delayed_mention_mentioned_count
        if delayed_mention_mentioned_count
        else None
    )

    delayed_mention_strategies = [
        str(item.get("delay_strategy") or "")
        for item in delayed_mention_unique
        if isinstance(item, dict)
    ]
    delayed_mention_signals: list[str] = []
    for item in delayed_mention_unique:
        sigs = item.get("delay_signals") or []
        if isinstance(sigs, list):
            delayed_mention_signals.extend(str(x) for x in sigs if x)
    delayed_mention_strategy_counts = count_strings(delayed_mention_strategies)
    delayed_mention_signal_counts = count_strings(delayed_mention_signals)

    result = {
        "file": str(path),
        "provider": provider,
        "model": model,
        "turns": len(assistant_rows),
        "probe_count": len(probes),
        "conclusion_line_probe_count": conclusion_line_probe_count,
        "conclusion_line_mention_count": conclusion_line_mention_count,
        "conclusion_line_mention_rate": conclusion_line_mention_rate,
        "avg_conclusion_line_mention_delay_turns": mean_or_none(conclusion_line_mention_delays),
        "conclusion_line_immediate_mention_rate": conclusion_line_immediate_mention_rate,
        "conclusion_any_mention_count": conclusion_any_mention_count,
        "conclusion_any_mention_rate": conclusion_any_mention_rate,
        "avg_conclusion_any_mention_delay_turns": mean_or_none(conclusion_any_mention_delays),
        "conclusion_plan_window_defined_count": conclusion_plan_window_defined_count,
        "conclusion_plan_within_window_rate": conclusion_plan_within_window_rate,
        "conclusion_plan_early_rate": conclusion_plan_early_rate,
        "conclusion_plan_late_rate": conclusion_plan_late_rate,
        "conclusion_plan_missing_rate": conclusion_plan_missing_rate,
        "avg_planned_conclusion_delay_min_turns": mean_or_none(planned_conclusion_delay_mins),
        "avg_planned_conclusion_delay_max_turns": mean_or_none(planned_conclusion_delay_maxs),
        "avg_planned_conclusion_window_width_turns": mean_or_none(
            planned_conclusion_window_widths
        ),
        "avg_conclusion_mention_likelihood": mean_or_none(conclusion_mention_likelihoods),
        "delayed_mention_plan_error_count": delayed_mention_plan_error_count,
        "delayed_mention_planned_count": delayed_mention_planned_count,
        "delayed_mention_planned_nonconclusion_count": delayed_mention_planned_nonconclusion_count,
        "delayed_mention_mentioned_count": delayed_mention_mentioned_count,
        "delayed_mention_expired_count": delayed_mention_expired_count,
        "delayed_mention_mention_rate": delayed_mention_mention_rate,
        "delayed_mention_nonconclusion_mention_rate": delayed_mention_nonconclusion_mention_rate,
        "delayed_mention_within_window_rate": delayed_mention_within_window_rate,
        "avg_delayed_mention_delay_turns": mean_or_none(delayed_mention_delays),
        "delayed_mention_early_count": delayed_mention_early_count,
        "delayed_mention_late_count": delayed_mention_late_count,
        "delayed_mention_injected_on_mention_rate": delayed_mention_injected_on_mention_rate,
        "delayed_mention_strategy_counts": delayed_mention_strategy_counts or None,
        "delayed_mention_signal_counts": delayed_mention_signal_counts or None,
        "memory_capsule_count": len(capsules),
        "deferred_intent_backend": deferred_intent_backend,
        "deferred_intent_timing": deferred_intent_timing,
        "deferred_intent_plan_policy": deferred_intent_plan_policy,
        "deferred_intent_plan_budget": deferred_intent_plan_budget,
        "deferred_intent_plan_probe_calls": deferred_intent_plan_probe_calls,
        "deferred_intent_plan_probe_call_count": plan_probe_call_count,
        "deferred_intent_plan_count": len(planned_ids),
        "deferred_intent_fire_count": fire_count,
        "deferred_intent_cancel_count": cancel_count,
        "deferred_intent_expire_count": expire_count,
        "deferred_intent_revise_count": revise_count,
        "deferred_intent_active_final_count": active_final_count,
        "deferred_intent_injected_fire_count": injected_fire_count,
        "deferred_intent_realization_rate": (
            realized_fire_count / fire_count if fire_count > 0 else None
        ),
        "avg_deferred_intent_reply_overlap": mean_or_none(fire_overlaps),
        "avg_deferred_intent_fire_gap": mean_or_none(fire_turn_gaps),
        "avg_planned_delay_min_turns": mean_or_none(planned_delay_mins),
        "avg_planned_delay_max_turns": mean_or_none(planned_delay_maxs),
        "avg_planned_window_width_turns": mean_or_none(planned_window_widths),
        "avg_fire_window_width_turns": mean_or_none(fire_window_widths),
        "avg_fire_window_position_ratio": mean_or_none(fire_window_position_ratios),
        "fire_at_earliest_rate": (fire_at_earliest_count / fire_count) if fire_count else None,
        "fire_at_latest_rate": (fire_at_latest_count / fire_count) if fire_count else None,
        "deferred_intent_premature_fire_count": premature_fire_count,
        "deferred_intent_stale_fire_count": stale_fire_count,
        "avg_probe_reply_overlap": mean_or_none(per_turn_overlap),
        "avg_probe_confidence": mean_or_none(probe_confidences),
        "avg_conclusion_stability": mean_or_none(probe_stabilities),
        "last_turn_alignment": last_turn_alignment,
        "final_probe_reply_overlap": final_probe_reply_overlap,
        "avg_reply_words": mean_or_none(reply_word_counts),
        "avg_input_tokens": mean_or_none([x for x in input_tokens if x > 0]),
        "avg_output_tokens": mean_or_none([x for x in output_tokens if x > 0]),
        "inband_state_error_count": len(inband_state_errors),
        "avg_inband_state_chars": mean_or_none(inband_state_sizes),
        "max_inband_state_chars": max(inband_state_sizes) if inband_state_sizes else None,
        "inband_state_prune_count": len(inband_prune_events),
        "inband_state_pruned_intents_total": pruned_intents_total,
        "plan_strategy_counts": plan_strategy_counts or None,
        "plan_signal_counts": plan_signal_counts or None,
        "decision_strategy_counts": decision_strategy_counts or None,
        "decision_signal_counts": decision_signal_counts or None,
        "conclusion_delay_strategy_counts": count_strings(conclusion_delay_strategies) or None,
        "conclusion_delay_signal_counts": count_strings(conclusion_delay_signals) or None,
        "final_required_keyword_coverage": keyword_coverage(final_required, final_assistant),
        "conversation_required_keyword_coverage": keyword_coverage(conversation_required, conversation_text),
        "final_forbidden_keyword_hits": keyword_hits(final_forbidden, final_assistant),
        "final_user": compact_text(final_user),
        "final_assistant": compact_text(final_assistant),
        "final_probe": compact_text(final_probe),
        "planned_intent_ids": sorted(planned_ids),
    }
    return result


def collect_logs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.logs:
        paths.extend(Path(x) for x in args.logs)
    if args.log_dir:
        paths.extend(sorted(Path(args.log_dir).glob("*.jsonl")))
    if not paths:
        raise SystemExit("No logs found. Use --logs or --log-dir.")
    deduped = []
    seen = set()
    for path in paths:
        rp = path.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(path)
    return deduped


def print_table(rows: list[dict[str, Any]]) -> None:
    cols = [
        "provider",
        "model",
        "deferred_intent_backend",
        "deferred_intent_timing",
        "turns",
        "probe_count",
        "conclusion_line_mention_rate",
        "conclusion_any_mention_rate",
        "conclusion_plan_within_window_rate",
        "delayed_mention_planned_nonconclusion_count",
        "delayed_mention_nonconclusion_mention_rate",
        "delayed_mention_within_window_rate",
        "delayed_mention_injected_on_mention_rate",
        "avg_conclusion_line_mention_delay_turns",
        "avg_conclusion_any_mention_delay_turns",
        "deferred_intent_plan_count",
        "deferred_intent_fire_count",
        "deferred_intent_realization_rate",
        "avg_planned_window_width_turns",
        "avg_fire_window_position_ratio",
        "avg_deferred_intent_reply_overlap",
        "avg_probe_reply_overlap",
        "inband_state_error_count",
        "avg_inband_state_chars",
        "last_turn_alignment",
        "final_required_keyword_coverage",
        "final_forbidden_keyword_hits",
    ]

    display_rows = []
    for row in rows:
        display = []
        for c in cols:
            value = row.get(c)
            if isinstance(value, float):
                if c.endswith("_chars"):
                    display.append(f"{value:.0f}")
                else:
                    display.append(f"{value:.3f}")
            elif value is None:
                display.append("-")
            else:
                display.append(str(value))
        display_rows.append(display)

    widths = []
    for i, c in enumerate(cols):
        widths.append(max(len(c), *(len(r[i]) for r in display_rows)) if display_rows else len(c))

    def fmt_line(items: list[str]) -> str:
        return " | ".join(item.ljust(widths[i]) for i, item in enumerate(items))

    print(fmt_line(cols))
    print("-+-".join("-" * w for w in widths))
    for row in display_rows:
        print(fmt_line(row))



def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Recursive Conclusion Lab JSONL logs.")
    parser.add_argument("--logs", nargs="*", default=[], help="Specific JSONL log files.")
    parser.add_argument("--log-dir", default="", help="Directory containing JSONL log files.")
    parser.add_argument("--script", default="", help="Optional script JSON with evaluation spec.")
    parser.add_argument("--out", default="", help="Optional path to write JSON summary.")
    args = parser.parse_args()

    logs = collect_logs(args)
    evaluation = load_evaluation_spec(Path(args.script)) if args.script else {}
    rows = [summarize_log(path, evaluation) for path in logs]

    print_table(rows)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
