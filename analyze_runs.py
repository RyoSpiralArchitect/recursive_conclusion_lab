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
LATENT_ALIGNMENT_THRESHOLD = 0.55
EMBEDDING_ALIGNMENT_THRESHOLD = 0.65
SEMANTIC_JUDGE_DISAGREEMENT_THRESHOLD = 0.20
AGGREGATE_SELECTED_METRICS = [
    "conclusion_any_mention_rate",
    "conclusion_plan_within_window_rate",
    "conclusion_on_support_rate",
    "delayed_mention_planned_count",
    "delayed_mention_planned_nonconclusion_count",
    "delayed_mention_plan_warning_count",
    "delayed_mention_kind_diversity",
    "delayed_mention_nonconclusion_share",
    "delayed_mention_required_kind_coverage",
    "delayed_mention_min_nonconclusion_satisfied",
    "delayed_mention_min_kind_diversity_satisfied",
    "delayed_mention_nonconclusion_mention_rate",
    "delayed_mention_within_window_rate",
    "delayed_mention_on_support_rate",
    "delayed_mention_option_stage_mention_rate",
    "delayed_mention_option_stage_within_window_rate",
    "delayed_mention_option_stage_on_support_rate",
    "delayed_mention_final_risk_packet_mention_rate",
    "delayed_mention_final_risk_packet_within_window_rate",
    "delayed_mention_final_risk_packet_on_support_rate",
    "avg_suppressed_delayed_mention_count",
    "avg_conclusion_hazard_turn_prob_at_mention",
    "avg_conclusion_peak_support_ratio_at_mention",
    "avg_delayed_mention_hazard_turn_prob_at_mention",
    "avg_delayed_mention_peak_support_ratio_at_mention",
    "avg_adaptive_hazard_multiplier",
    "adaptive_hazard_intervention_rate",
    "avg_adaptive_hazard_turn_prob_shift",
    "avg_adaptive_hazard_threshold_shift",
    "avg_option_stage_adaptive_hazard_multiplier",
    "avg_option_stage_adaptive_threshold_shift",
    "avg_final_risk_packet_adaptive_hazard_multiplier",
    "avg_final_risk_packet_adaptive_threshold_shift",
    "avg_adaptive_embedding_prepeak_penalty",
    "avg_adaptive_embedding_prepeak_peak_gap_factor",
    "avg_conclusion_adaptive_hazard_multiplier",
    "avg_conclusion_adaptive_hazard_turn_prob",
    "avg_latent_alignment",
    "latent_semantic_leakage_rate",
    "avg_articulation_gap_turns",
    "avg_embedding_alignment",
    "embedding_alignment_slope",
    "embedding_semantic_leakage_rate",
    "avg_embedding_articulation_gap_turns",
    "avg_semantic_judge_alignment_gap",
    "semantic_judge_disagreement_rate",
    "recovery_after_perturbation_rate",
    "time_to_recover_turns",
    "probe_recovery_after_perturbation_rate",
    "probe_time_to_recover_turns",
    "probe_to_reply_recovery_gap_turns",
    "post_perturbation_forbidden_turn_rate",
    "post_perturbation_required_keyword_coverage",
]


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


def stddev_or_none(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return None
    return statistics.stdev(vals)


def quantile_or_none(values: Iterable[float], q: float) -> float | None:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, float(q))) * (len(vals) - 1)
    lo = int(pos)
    hi = min(len(vals) - 1, lo + 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def sequence_slope(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return None
    x_mean = (len(vals) - 1) / 2.0
    y_mean = statistics.fmean(vals)
    denom = sum((idx - x_mean) ** 2 for idx in range(len(vals)))
    if denom <= 0.0:
        return None
    numer = sum((idx - x_mean) * (value - y_mean) for idx, value in enumerate(vals))
    return numer / denom


def count_strings(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = compact_text(str(value))
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = compact_text(str(value))
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def hazard_delays(profile: Any) -> list[int]:
    if not isinstance(profile, list):
        return []
    delays: list[int] = []
    for item in profile:
        if not isinstance(item, dict):
            continue
        delay = coerce_int(item.get("delay_turns"))
        prob = item.get("prob")
        if delay is None:
            continue
        if isinstance(prob, (int, float)) and float(prob) <= 0.0:
            continue
        delays.append(delay)
    return sorted(set(delays))


def hazard_probability_for_delay(profile: Any, delay: int) -> float | None:
    if not isinstance(profile, list):
        return None
    for item in profile:
        if not isinstance(item, dict):
            continue
        item_delay = coerce_int(item.get("delay_turns"))
        if item_delay != delay:
            continue
        prob = item.get("prob")
        if isinstance(prob, (int, float)):
            return float(prob)
        return None
    return 0.0


def hazard_peak_probability(profile: Any) -> float | None:
    if not isinstance(profile, list):
        return None
    best: float | None = None
    for item in profile:
        if not isinstance(item, dict):
            continue
        prob = item.get("prob")
        if not isinstance(prob, (int, float)):
            continue
        value = float(prob)
        if value <= 0.0:
            continue
        if best is None or value > best:
            best = value
    return best


def parse_run_metadata(path: Path) -> tuple[str | None, int | None]:
    stem = path.stem
    match = re.search(r"__run_(\d+)$", stem)
    if not match:
        return None, None
    run_name = f"run_{match.group(1)}"
    try:
        run_index = int(match.group(1))
    except ValueError:
        run_index = None
    return run_name, run_index


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


def normalize_perturbation_spec(evaluation: dict[str, Any]) -> dict[str, Any]:
    raw = evaluation.get("perturbation") or {}
    if not isinstance(raw, dict):
        return {}
    required_keywords = [
        str(x)
        for x in (
            raw.get("required_keywords")
            or raw.get("recovery_required_keywords")
            or evaluation.get("final_required_keywords")
            or []
        )
        if x
    ]
    forbidden_keywords = [
        str(x)
        for x in (
            raw.get("forbidden_keywords")
            or raw.get("obsolete_keywords")
            or raw.get("recovery_forbidden_keywords")
            or evaluation.get("final_forbidden_keywords")
            or []
        )
        if x
    ]
    required_min_hits = coerce_int(
        raw.get("required_min_hits") or raw.get("recovery_min_hits")
    )
    if required_min_hits is None and required_keywords:
        required_min_hits = max(1, (len(required_keywords) + 1) // 2)
    probe_required_min_hits = coerce_int(
        raw.get("probe_required_min_hits") or raw.get("probe_recovery_min_hits")
    )
    if probe_required_min_hits is None and required_keywords:
        probe_required_min_hits = required_min_hits
    turn = coerce_int(raw.get("turn") or raw.get("turn_index") or raw.get("start_turn"))
    return {
        "label": compact_text(str(raw.get("label") or raw.get("kind") or "")) or None,
        "turn": turn,
        "required_keywords": required_keywords,
        "forbidden_keywords": forbidden_keywords,
        "required_min_hits": required_min_hits,
        "probe_required_min_hits": probe_required_min_hits,
    }


def normalize_delayed_mention_eval(evaluation: dict[str, Any]) -> dict[str, Any]:
    required_kinds = [
        compact_text(str(x)).lower()
        for x in (evaluation.get("delayed_mention_required_kinds") or [])
        if compact_text(str(x))
    ]
    required_kinds = list(dict.fromkeys(required_kinds))
    min_nonconclusion_items = coerce_int(
        evaluation.get("delayed_mention_min_nonconclusion_items")
    )
    min_kind_diversity = coerce_int(evaluation.get("delayed_mention_min_kind_diversity"))
    return {
        "required_kinds": required_kinds,
        "min_nonconclusion_items": max(0, min_nonconclusion_items or 0),
        "min_kind_diversity": max(0, min_kind_diversity or 0),
    }


def classify_delayed_mention_stage_role(item: dict[str, Any]) -> str:
    release_stage_role = compact_text(str(item.get("release_stage_role") or ""))
    if release_stage_role:
        return release_stage_role.lower()

    kind_norm = compact_text(str(item.get("kind") or "")).lower()
    blob_parts = [
        kind_norm,
        compact_text(str(item.get("text") or "")).lower(),
        compact_text(str(item.get("delay_strategy") or "")).lower(),
    ]
    sigs = item.get("delay_signals") or []
    if isinstance(sigs, list):
        blob_parts.extend(compact_text(str(x)).lower() for x in sigs if compact_text(str(x)))
    blob = " ".join(part for part in blob_parts if part)

    final_packet_hints = (
        "fallback",
        "backup",
        "contingency",
        "sync fail",
        "fails",
        "graceful failure",
        "migration",
        "risk",
        "rollback",
        "lock-in",
        "caveat",
    )
    option_stage_hints = (
        "shortlist",
        "candidate",
        "runner-up",
        "runner up",
        "alternative",
        "front-runner",
        "winner",
        "short list",
    )
    support_stage_hints = (
        "criteria",
        "checklist",
        "constraint",
        "requirement",
        "priority",
        "decision",
        "export",
        "markdown",
        "offline",
    )

    if kind_norm == "conclusion":
        return "conclusion"
    if kind_norm in {"caveat", "migration_risk"}:
        return "final_risk_packet"
    if any(hint in blob for hint in final_packet_hints):
        return "final_risk_packet"
    if kind_norm == "option":
        return "option_stage"
    if kind_norm in {"constraint", "definition"}:
        return "support_stage"
    if any(hint in blob for hint in option_stage_hints):
        return "option_stage"
    if any(hint in blob for hint in support_stage_hints):
        return "support_stage"
    return "support_stage"


def coerce_usage_num(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def summarize_log(path: Path, evaluation: dict[str, Any]) -> dict[str, Any]:
    events = read_jsonl(path)
    arm = None
    run_name, run_index = parse_run_metadata(path)
    perturbation = normalize_perturbation_spec(evaluation)
    delayed_mention_eval = normalize_delayed_mention_eval(evaluation)
    provider = None
    model = None
    if "___" in path.stem:
        arm_part, _ = path.stem.split("___", 1)
        if arm_part.startswith("arm_"):
            arm = arm_part[4:]

    assistant_rows: list[dict[str, Any]] = []
    assistant_text_by_turn: dict[int, str] = {}
    probes: list[str] = []
    probe_events: list[tuple[int, str]] = []
    probe_plan_events: list[dict[str, Any]] = []
    latent_trace_events: list[dict[str, Any]] = []
    embedding_trace_events: list[dict[str, Any]] = []
    delayed_mention_plans: list[dict[str, Any]] = []
    delayed_mention_actions: list[dict[str, Any]] = []
    delayed_mention_plan_error_count = 0
    delayed_mention_plan_warning_count = 0
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
                            "mention_hazard_profile": payload.get("mention_hazard_profile") or [],
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
        elif et == "latent_convergence_trace":
            if isinstance(payload, dict):
                latent_trace_events.append(
                    {
                        **payload,
                        "turn": row.get("turn_index"),
                    }
                )
        elif et == "embedding_convergence_trace":
            if isinstance(payload, dict):
                embedding_trace_events.append(
                    {
                        **payload,
                        "turn": row.get("turn_index"),
                    }
                )
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
        elif et == "delayed_mention_plan_warning":
            delayed_mention_plan_warning_count += 1
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
    planned_conclusion_hazard_support_widths: list[int] = []
    conclusion_hazard_plan_count = 0
    conclusion_hazard_turn_probs_at_mention: list[float] = []
    conclusion_peak_support_ratios_at_mention: list[float] = []
    conclusion_hazard_mention_count = 0
    conclusion_hazard_on_support_count = 0
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
        hazard_profile = plan.get("mention_hazard_profile") or []
        delays = hazard_delays(hazard_profile)
        if delays:
            conclusion_hazard_plan_count += 1
            planned_conclusion_hazard_support_widths.append(max(delays) - min(delays))
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
                if delays:
                    conclusion_hazard_mention_count += 1
                    hazard_turn_prob = hazard_probability_for_delay(hazard_profile, mention_delay)
                    if isinstance(hazard_turn_prob, (int, float)):
                        conclusion_hazard_turn_probs_at_mention.append(float(hazard_turn_prob))
                        peak_prob = hazard_peak_probability(hazard_profile)
                        if isinstance(peak_prob, (int, float)) and float(peak_prob) > 0.0:
                            conclusion_peak_support_ratios_at_mention.append(
                                min(1.0, float(hazard_turn_prob) / float(peak_prob))
                            )
                        if float(hazard_turn_prob) > 0.0:
                            conclusion_hazard_on_support_count += 1
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
    hazard_turn_probs_at_fire: list[float] = []
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
                hazard_turn_prob = action.get("hazard_turn_prob")
                if isinstance(hazard_turn_prob, (int, float)):
                    hazard_turn_probs_at_fire.append(float(hazard_turn_prob))
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
        hazard_turn_probs_at_fire = []
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
                hazard_turn_prob = action.get("hazard_turn_prob")
                if isinstance(hazard_turn_prob, (int, float)):
                    hazard_turn_probs_at_fire.append(float(hazard_turn_prob))
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
    planned_hazard_support_widths: list[int] = []
    hazard_plan_count = 0
    for item in planned_unique:
        created_turn = item.get("created_turn")
        earliest_turn = item.get("earliest_turn")
        latest_turn = item.get("latest_turn")
        hazard_profile = item.get("hazard_profile") or []
        if isinstance(hazard_profile, list) and hazard_profile:
            hazard_plan_count += 1
            delays = []
            for point in hazard_profile:
                if not isinstance(point, dict):
                    continue
                delay = point.get("delay_turns")
                if isinstance(delay, int):
                    delays.append(delay)
            if delays:
                planned_hazard_support_widths.append(max(delays) - min(delays))
        if not (
            isinstance(created_turn, int)
            and isinstance(earliest_turn, int)
            and isinstance(latest_turn, int)
        ):
            continue
        planned_delay_mins.append(earliest_turn - created_turn)
        planned_delay_maxs.append(latest_turn - created_turn)
        planned_window_widths.append(latest_turn - earliest_turn)

    latent_alignments: list[float] = []
    latent_readiness_values: list[float] = []
    latent_leakage_risks: list[float] = []
    latent_stage_counts: list[str] = []
    latent_signal_counts: list[str] = []
    latent_judge_source = None
    latent_judge_provider = None
    latent_judge_model = None
    latent_prewindow_trace_count = 0
    latent_semantic_leakage_count = 0
    first_semantic_turn_by_probe: dict[int, int] = {}
    llm_alignment_by_turn: dict[int, float] = {}
    for event in latent_trace_events:
        if latent_judge_source is None and event.get("judge_source"):
            latent_judge_source = str(event.get("judge_source"))
        if latent_judge_provider is None and event.get("judge_provider"):
            latent_judge_provider = str(event.get("judge_provider"))
        if latent_judge_model is None and event.get("judge_model"):
            latent_judge_model = str(event.get("judge_model"))
        alignment = event.get("alignment")
        readiness = event.get("articulation_readiness")
        leakage_risk = event.get("leakage_risk")
        if isinstance(alignment, (int, float)):
            latent_alignments.append(float(alignment))
        if isinstance(readiness, (int, float)):
            latent_readiness_values.append(float(readiness))
        if isinstance(leakage_risk, (int, float)):
            latent_leakage_risks.append(float(leakage_risk))
        stage = event.get("trajectory_stage")
        if stage:
            latent_stage_counts.append(str(stage))
        signals = event.get("shift_signals") or []
        if isinstance(signals, list):
            latent_signal_counts.extend(str(x) for x in signals if x)

        turn = event.get("turn")
        planned_earliest_turn = event.get("planned_earliest_turn")
        explicit_present = event.get("explicit_mention_present")
        if isinstance(turn, int) and isinstance(alignment, (int, float)):
            llm_alignment_by_turn.setdefault(turn, float(alignment))
        if isinstance(turn, int) and isinstance(planned_earliest_turn, int) and turn < planned_earliest_turn:
            latent_prewindow_trace_count += 1
            if (
                isinstance(alignment, (int, float))
                and float(alignment) >= LATENT_ALIGNMENT_THRESHOLD
                and explicit_present is not True
            ):
                latent_semantic_leakage_count += 1

        probe_turn = event.get("conclusion_probe_turn")
        if (
            isinstance(probe_turn, int)
            and isinstance(turn, int)
            and isinstance(alignment, (int, float))
            and float(alignment) >= LATENT_ALIGNMENT_THRESHOLD
            and explicit_present is not True
            and turn >= probe_turn
        ):
            first_semantic_turn_by_probe.setdefault(probe_turn, turn)

    embedding_alignments: list[float] = []
    embedding_line_alignments: list[float] = []
    embedding_keyword_alignments: list[float] = []
    embedding_judge_source = None
    embedding_judge_provider = None
    embedding_judge_model = None
    embedding_prewindow_trace_count = 0
    embedding_semantic_leakage_count = 0
    first_embedding_semantic_turn_by_probe: dict[int, int] = {}
    embedding_alignment_by_turn: dict[int, float] = {}
    for event in embedding_trace_events:
        if embedding_judge_source is None and event.get("judge_source"):
            embedding_judge_source = str(event.get("judge_source"))
        if embedding_judge_provider is None and event.get("judge_provider"):
            embedding_judge_provider = str(event.get("judge_provider"))
        if embedding_judge_model is None and event.get("judge_model"):
            embedding_judge_model = str(event.get("judge_model"))
        alignment = event.get("alignment")
        line_alignment = event.get("line_alignment")
        keyword_alignment = event.get("keyword_alignment")
        if isinstance(alignment, (int, float)):
            embedding_alignments.append(float(alignment))
        if isinstance(line_alignment, (int, float)):
            embedding_line_alignments.append(float(line_alignment))
        if isinstance(keyword_alignment, (int, float)):
            embedding_keyword_alignments.append(float(keyword_alignment))

        turn = event.get("turn")
        planned_earliest_turn = event.get("planned_earliest_turn")
        explicit_present = event.get("explicit_mention_present")
        if isinstance(turn, int) and isinstance(alignment, (int, float)):
            embedding_alignment_by_turn.setdefault(turn, float(alignment))
        if isinstance(turn, int) and isinstance(planned_earliest_turn, int) and turn < planned_earliest_turn:
            embedding_prewindow_trace_count += 1
            if (
                isinstance(alignment, (int, float))
                and float(alignment) >= EMBEDDING_ALIGNMENT_THRESHOLD
                and explicit_present is not True
            ):
                embedding_semantic_leakage_count += 1

        probe_turn = event.get("conclusion_probe_turn")
        if (
            isinstance(probe_turn, int)
            and isinstance(turn, int)
            and isinstance(alignment, (int, float))
            and float(alignment) >= EMBEDDING_ALIGNMENT_THRESHOLD
            and explicit_present is not True
            and turn >= probe_turn
        ):
            first_embedding_semantic_turn_by_probe.setdefault(probe_turn, turn)

    first_explicit_turn_by_probe: dict[int, int] = {}
    for turn_idx, row in enumerate(assistant_rows, start=1):
        probe_turn = row.get("latest_conclusion_probe_turn")
        if not isinstance(probe_turn, int):
            continue
        if row.get("latest_conclusion_mentioned_any") is True:
            first_explicit_turn_by_probe.setdefault(probe_turn, turn_idx)

    articulation_gaps: list[int] = []
    for probe_turn, semantic_turn in first_semantic_turn_by_probe.items():
        explicit_turn = first_explicit_turn_by_probe.get(probe_turn)
        if isinstance(explicit_turn, int) and explicit_turn > semantic_turn:
            articulation_gaps.append(explicit_turn - semantic_turn)

    embedding_articulation_gaps: list[int] = []
    for probe_turn, semantic_turn in first_embedding_semantic_turn_by_probe.items():
        explicit_turn = first_explicit_turn_by_probe.get(probe_turn)
        if isinstance(explicit_turn, int) and explicit_turn > semantic_turn:
            embedding_articulation_gaps.append(explicit_turn - semantic_turn)

    semantic_judge_alignment_gaps: list[float] = []
    semantic_judge_disagreement_count = 0
    for turn, llm_alignment in llm_alignment_by_turn.items():
        embedding_alignment = embedding_alignment_by_turn.get(turn)
        if not isinstance(embedding_alignment, float):
            continue
        gap = abs(llm_alignment - embedding_alignment)
        semantic_judge_alignment_gaps.append(gap)
        if gap >= SEMANTIC_JUDGE_DISAGREEMENT_THRESHOLD:
            semantic_judge_disagreement_count += 1

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

    perturbation_turn = perturbation.get("turn")
    perturbation_required_keywords = perturbation.get("required_keywords") or []
    perturbation_forbidden_keywords = perturbation.get("forbidden_keywords") or []
    perturbation_required_min_hits = perturbation.get("required_min_hits")
    perturbation_probe_required_min_hits = perturbation.get("probe_required_min_hits")
    post_perturbation_turn_count = 0
    post_perturbation_required_keyword_coverage = None
    post_perturbation_forbidden_total_hits = 0
    post_perturbation_forbidden_turn_rate = None
    recovery_turn = None
    recovery_after_perturbation = None
    recovery_after_perturbation_rate = None
    time_to_recover_turns = None
    probe_recovery_turn = None
    probe_recovery_after_perturbation = None
    probe_recovery_after_perturbation_rate = None
    probe_time_to_recover_turns = None
    probe_to_reply_recovery_gap_turns = None
    probe_post_perturbation_count = 0
    probe_post_perturbation_forbidden_turn_rate = None

    if isinstance(perturbation_turn, int):
        post_perturbation_rows = [
            (turn_idx, row)
            for turn_idx, row in enumerate(assistant_rows, start=1)
            if turn_idx >= perturbation_turn
        ]
        post_perturbation_turn_count = len(post_perturbation_rows)
        post_perturbation_text = "\n".join(
            str(row.get("assistant") or "") for _, row in post_perturbation_rows
        )
        if perturbation_required_keywords:
            post_perturbation_required_keyword_coverage = keyword_coverage(
                perturbation_required_keywords,
                post_perturbation_text,
            )
        forbidden_turn_count = 0
        for turn_idx, row in post_perturbation_rows:
            text = str(row.get("assistant") or "")
            forbidden_hits = keyword_hits(perturbation_forbidden_keywords, text) or 0
            post_perturbation_forbidden_total_hits += forbidden_hits
            if forbidden_hits > 0:
                forbidden_turn_count += 1
            if (
                recovery_turn is None
                and perturbation_required_keywords
                and isinstance(perturbation_required_min_hits, int)
            ):
                required_hits = keyword_hits(perturbation_required_keywords, text) or 0
                if required_hits >= perturbation_required_min_hits:
                    recovery_turn = turn_idx
        if post_perturbation_turn_count > 0:
            probe_post_perturbation_forbidden_turn_rate = None
            post_perturbation_forbidden_turn_rate = (
                forbidden_turn_count / post_perturbation_turn_count
            )

        recovery_after_perturbation = recovery_turn is not None
        recovery_after_perturbation_rate = (
            1.0 if recovery_after_perturbation else 0.0
        )
        if recovery_turn is not None:
            time_to_recover_turns = recovery_turn - perturbation_turn

        probe_forbidden_turn_count = 0
        for plan in probe_plan_events:
            probe_turn = plan.get("turn")
            if not isinstance(probe_turn, int) or probe_turn < perturbation_turn:
                continue
            probe_post_perturbation_count += 1
            probe_text = " ".join(
                [
                    str(plan.get("conclusion_line") or ""),
                    *[str(x) for x in (plan.get("keywords") or []) if x],
                ]
            )
            forbidden_hits = keyword_hits(perturbation_forbidden_keywords, probe_text) or 0
            if forbidden_hits > 0:
                probe_forbidden_turn_count += 1
            if (
                probe_recovery_turn is None
                and perturbation_required_keywords
                and isinstance(perturbation_probe_required_min_hits, int)
            ):
                required_hits = keyword_hits(perturbation_required_keywords, probe_text) or 0
                if required_hits >= perturbation_probe_required_min_hits:
                    probe_recovery_turn = probe_turn
        if probe_post_perturbation_count > 0:
            probe_post_perturbation_forbidden_turn_rate = (
                probe_forbidden_turn_count / probe_post_perturbation_count
            )
        probe_recovery_after_perturbation = probe_recovery_turn is not None
        probe_recovery_after_perturbation_rate = (
            1.0 if probe_recovery_after_perturbation else 0.0
        )
        if probe_recovery_turn is not None:
            probe_time_to_recover_turns = probe_recovery_turn - perturbation_turn
        if (
            isinstance(probe_recovery_turn, int)
            and isinstance(recovery_turn, int)
            and recovery_turn >= probe_recovery_turn
        ):
            probe_to_reply_recovery_gap_turns = recovery_turn - probe_recovery_turn

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
    delayed_mention_kinds = [
        compact_text(str(item.get("kind") or "")).lower()
        for item in delayed_mention_unique
        if compact_text(str(item.get("kind") or ""))
    ]
    delayed_mention_kind_counts = count_strings(delayed_mention_kinds)
    delayed_mention_kind_diversity = len(delayed_mention_kind_counts)
    delayed_mention_stage_role_by_id = {
        str(item.get("item_id")): classify_delayed_mention_stage_role(item)
        for item in delayed_mention_unique
        if item.get("item_id")
    }
    delayed_mention_stage_role_counts = count_strings(
        delayed_mention_stage_role_by_id.values()
    )
    delayed_mention_option_stage_planned_count = sum(
        1 for role in delayed_mention_stage_role_by_id.values() if role == "option_stage"
    )
    delayed_mention_final_risk_packet_planned_count = sum(
        1
        for role in delayed_mention_stage_role_by_id.values()
        if role == "final_risk_packet"
    )
    delayed_mention_nonconclusion_share = (
        delayed_mention_planned_nonconclusion_count / delayed_mention_planned_count
        if delayed_mention_planned_count
        else None
    )
    required_kinds = delayed_mention_eval.get("required_kinds") or []
    delayed_mention_required_kind_coverage = (
        sum(1 for kind in required_kinds if kind in delayed_mention_kind_counts)
        / len(required_kinds)
        if required_kinds
        else None
    )
    required_min_nonconclusion_items = int(
        delayed_mention_eval.get("min_nonconclusion_items") or 0
    )
    required_min_kind_diversity = int(
        delayed_mention_eval.get("min_kind_diversity") or 0
    )
    delayed_mention_min_nonconclusion_satisfied = (
        1.0
        if delayed_mention_planned_nonconclusion_count >= required_min_nonconclusion_items
        else 0.0
    ) if required_min_nonconclusion_items > 0 else None
    delayed_mention_min_kind_diversity_satisfied = (
        1.0 if delayed_mention_kind_diversity >= required_min_kind_diversity else 0.0
    ) if required_min_kind_diversity > 0 else None

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
    delayed_mention_hazard_mention_count = 0
    delayed_mention_hazard_on_support_count = 0
    delayed_mention_hazard_turn_probs_at_mention: list[float] = []
    delayed_mention_peak_support_ratios_at_mention: list[float] = []
    option_stage_mentioned_count = 0
    option_stage_within_window_count = 0
    option_stage_hazard_mention_count = 0
    option_stage_hazard_on_support_count = 0
    final_risk_packet_mentioned_count = 0
    final_risk_packet_within_window_count = 0
    final_risk_packet_hazard_mention_count = 0
    final_risk_packet_hazard_on_support_count = 0
    for act in mention_actions:
        stage_role = compact_text(str(act.get("release_stage_role") or "")).lower()
        if not stage_role:
            stage_role = delayed_mention_stage_role_by_id.get(str(act.get("item_id") or "")) or ""
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
        if stage_role == "option_stage":
            option_stage_mentioned_count += 1
            if act.get("within_window") is True:
                option_stage_within_window_count += 1
        elif stage_role == "final_risk_packet":
            final_risk_packet_mentioned_count += 1
            if act.get("within_window") is True:
                final_risk_packet_within_window_count += 1
        hazard_turn_prob = act.get("hazard_turn_prob")
        if isinstance(hazard_turn_prob, (int, float)):
            delayed_mention_hazard_mention_count += 1
            delayed_mention_hazard_turn_probs_at_mention.append(float(hazard_turn_prob))
            if stage_role == "option_stage":
                option_stage_hazard_mention_count += 1
            elif stage_role == "final_risk_packet":
                final_risk_packet_hazard_mention_count += 1
            peak_prob = act.get("hazard_peak_prob")
            if not isinstance(peak_prob, (int, float)):
                peak_prob = hazard_peak_probability(act.get("hazard_profile"))
            if isinstance(peak_prob, (int, float)) and float(peak_prob) > 0.0:
                delayed_mention_peak_support_ratios_at_mention.append(
                    min(1.0, float(hazard_turn_prob) / float(peak_prob))
                )
            if float(hazard_turn_prob) > 0.0:
                delayed_mention_hazard_on_support_count += 1
                if stage_role == "option_stage":
                    option_stage_hazard_on_support_count += 1
                elif stage_role == "final_risk_packet":
                    final_risk_packet_hazard_on_support_count += 1

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
    delayed_mention_option_stage_mention_rate = (
        option_stage_mentioned_count / delayed_mention_option_stage_planned_count
        if delayed_mention_option_stage_planned_count
        else None
    )
    delayed_mention_option_stage_within_window_rate = (
        option_stage_within_window_count / delayed_mention_option_stage_planned_count
        if delayed_mention_option_stage_planned_count
        else None
    )
    delayed_mention_option_stage_on_support_rate = (
        option_stage_hazard_on_support_count / option_stage_hazard_mention_count
        if option_stage_hazard_mention_count
        else None
    )
    delayed_mention_final_risk_packet_mention_rate = (
        final_risk_packet_mentioned_count / delayed_mention_final_risk_packet_planned_count
        if delayed_mention_final_risk_packet_planned_count
        else None
    )
    delayed_mention_final_risk_packet_within_window_rate = (
        final_risk_packet_within_window_count
        / delayed_mention_final_risk_packet_planned_count
        if delayed_mention_final_risk_packet_planned_count
        else None
    )
    delayed_mention_final_risk_packet_on_support_rate = (
        final_risk_packet_hazard_on_support_count
        / final_risk_packet_hazard_mention_count
        if final_risk_packet_hazard_mention_count
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

    delayed_mention_leak_policy = None
    delayed_mention_leak_threshold = None
    delayed_mention_min_nonconclusion_items_cfg = None
    delayed_mention_min_kind_diversity_cfg = None
    delayed_mention_diversity_repair = None
    adaptive_hazard_policy = None
    adaptive_hazard_profile = None
    adaptive_hazard_stage_policy = None
    adaptive_hazard_embedding_guard = None
    suppressed_delayed_mention_counts: list[int] = []
    adaptive_hazard_multipliers: list[float] = []
    adaptive_hazard_turn_prob_shifts: list[float] = []
    adaptive_hazard_thresholds: list[float] = []
    adaptive_hazard_threshold_shifts: list[float] = []
    adaptive_embedding_prepeak_penalties: list[float] = []
    adaptive_embedding_prepeak_peak_gap_factors: list[float] = []
    adaptive_hazard_intervention_count = 0
    adaptive_hazard_total_count = 0
    adaptive_hazard_reasons: list[str] = []
    option_stage_adaptive_hazard_multipliers: list[float] = []
    option_stage_adaptive_threshold_shifts: list[float] = []
    final_risk_packet_adaptive_hazard_multipliers: list[float] = []
    final_risk_packet_adaptive_threshold_shifts: list[float] = []
    conclusion_adaptive_hazard_turn_probs: list[float] = []
    conclusion_adaptive_hazard_multipliers: list[float] = []
    conclusion_adaptive_threshold_shifts: list[float] = []
    for row in assistant_rows:
        if delayed_mention_leak_policy is None:
            value = row.get("delayed_mention_leak_policy")
            if value:
                delayed_mention_leak_policy = str(value)
        if delayed_mention_leak_threshold is None:
            value = row.get("delayed_mention_leak_threshold")
            if isinstance(value, (int, float)):
                delayed_mention_leak_threshold = float(value)
        if delayed_mention_min_nonconclusion_items_cfg is None:
            value = row.get("delayed_mention_min_nonconclusion_items")
            if isinstance(value, (int, float)):
                delayed_mention_min_nonconclusion_items_cfg = int(value)
        if delayed_mention_min_kind_diversity_cfg is None:
            value = row.get("delayed_mention_min_kind_diversity")
            if isinstance(value, (int, float)):
                delayed_mention_min_kind_diversity_cfg = int(value)
        if delayed_mention_diversity_repair is None:
            value = row.get("delayed_mention_diversity_repair")
            if value:
                delayed_mention_diversity_repair = str(value)
        if adaptive_hazard_policy is None:
            value = row.get("adaptive_hazard_policy")
            if value:
                adaptive_hazard_policy = str(value)
        if adaptive_hazard_profile is None:
            value = row.get("adaptive_hazard_profile")
            if value:
                adaptive_hazard_profile = str(value)
        if adaptive_hazard_stage_policy is None:
            value = row.get("adaptive_hazard_stage_policy")
            if value:
                adaptive_hazard_stage_policy = str(value)
        if adaptive_hazard_embedding_guard is None:
            value = row.get("adaptive_hazard_embedding_guard")
            if value:
                adaptive_hazard_embedding_guard = str(value)
        suppressed = row.get("suppressed_delayed_mentions") or []
        if isinstance(suppressed, list):
            suppressed_delayed_mention_counts.append(len(suppressed))
        trace = row.get("adaptive_hazard_trace") or {}
        items = trace.get("items") if isinstance(trace, dict) else []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                adaptive_hazard_total_count += 1
                multiplier = item.get("turn_prob_multiplier")
                if isinstance(multiplier, (int, float)):
                    multiplier = float(multiplier)
                    adaptive_hazard_multipliers.append(multiplier)
                    if abs(multiplier - 1.0) >= 0.05:
                        adaptive_hazard_intervention_count += 1
                    stage_role = compact_text(
                        str(item.get("release_stage_role") or "")
                    ).lower()
                    if stage_role == "option_stage":
                        option_stage_adaptive_hazard_multipliers.append(multiplier)
                    elif stage_role == "final_risk_packet":
                        final_risk_packet_adaptive_hazard_multipliers.append(multiplier)
                base_prob = item.get("base_turn_prob")
                effective_prob = item.get("effective_turn_prob")
                if isinstance(base_prob, (int, float)) and isinstance(
                    effective_prob,
                    (int, float),
                ):
                    adaptive_hazard_turn_prob_shifts.append(
                        float(effective_prob) - float(base_prob)
                    )
                threshold = item.get("effective_threshold")
                if isinstance(threshold, (int, float)):
                    adaptive_hazard_thresholds.append(float(threshold))
                threshold_shift = item.get("threshold_shift")
                if isinstance(threshold_shift, (int, float)):
                    adaptive_hazard_threshold_shifts.append(float(threshold_shift))
                    stage_role = compact_text(
                        str(item.get("release_stage_role") or "")
                    ).lower()
                    if stage_role == "option_stage":
                        option_stage_adaptive_threshold_shifts.append(float(threshold_shift))
                    elif stage_role == "final_risk_packet":
                        final_risk_packet_adaptive_threshold_shifts.append(
                            float(threshold_shift)
                        )
                prepeak_penalty = item.get("embedding_prepeak_penalty")
                if isinstance(prepeak_penalty, (int, float)):
                    adaptive_embedding_prepeak_penalties.append(float(prepeak_penalty))
                peak_gap_factor = item.get("embedding_prepeak_peak_gap_factor")
                if isinstance(peak_gap_factor, (int, float)):
                    adaptive_embedding_prepeak_peak_gap_factors.append(float(peak_gap_factor))
                reasons = item.get("reasons") or []
                if isinstance(reasons, list):
                    adaptive_hazard_reasons.extend(str(reason) for reason in reasons if reason)
        value = row.get("latest_conclusion_plan_adaptive_hazard_turn_prob")
        if isinstance(value, (int, float)):
            conclusion_adaptive_hazard_turn_probs.append(float(value))
        value = row.get("latest_conclusion_plan_adaptive_multiplier")
        if isinstance(value, (int, float)):
            conclusion_adaptive_hazard_multipliers.append(float(value))
        value = row.get("latest_conclusion_plan_adaptive_threshold_shift")
        if isinstance(value, (int, float)):
            conclusion_adaptive_threshold_shifts.append(float(value))

    delayed_mention_hazard_plan_count = 0
    planned_delayed_mention_hazard_support_widths: list[int] = []
    for item in delayed_mention_unique:
        delays = hazard_delays(item.get("hazard_profile") or [])
        if delays:
            delayed_mention_hazard_plan_count += 1
            planned_delayed_mention_hazard_support_widths.append(max(delays) - min(delays))

    semantic_judge_backend = next(
        (
            str(row.get("semantic_judge_backend"))
            for row in assistant_rows
            if row.get("semantic_judge_backend")
        ),
        None,
    )

    result = {
        "file": str(path),
        "arm": arm,
        "run_name": run_name,
        "run_index": run_index,
        "provider": provider,
        "model": model,
        "semantic_judge_backend": semantic_judge_backend,
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
        "conclusion_hazard_plan_count": conclusion_hazard_plan_count,
        "avg_planned_conclusion_hazard_support_width_turns": mean_or_none(
            planned_conclusion_hazard_support_widths
        ),
        "avg_conclusion_hazard_turn_prob_at_mention": mean_or_none(
            conclusion_hazard_turn_probs_at_mention
        ),
        "avg_conclusion_peak_support_ratio_at_mention": mean_or_none(
            conclusion_peak_support_ratios_at_mention
        ),
        "conclusion_on_support_rate": (
            conclusion_hazard_on_support_count / conclusion_hazard_mention_count
            if conclusion_hazard_mention_count
            else None
        ),
        "avg_conclusion_mention_likelihood": mean_or_none(conclusion_mention_likelihoods),
        "delayed_mention_plan_error_count": delayed_mention_plan_error_count,
        "delayed_mention_plan_warning_count": delayed_mention_plan_warning_count,
        "delayed_mention_planned_count": delayed_mention_planned_count,
        "delayed_mention_planned_nonconclusion_count": delayed_mention_planned_nonconclusion_count,
        "delayed_mention_kind_diversity": delayed_mention_kind_diversity,
        "delayed_mention_kind_counts": delayed_mention_kind_counts or None,
        "delayed_mention_stage_role_counts": delayed_mention_stage_role_counts or None,
        "delayed_mention_option_stage_planned_count": delayed_mention_option_stage_planned_count,
        "delayed_mention_final_risk_packet_planned_count": (
            delayed_mention_final_risk_packet_planned_count
        ),
        "delayed_mention_nonconclusion_share": delayed_mention_nonconclusion_share,
        "delayed_mention_required_kinds": required_kinds or None,
        "delayed_mention_required_kind_coverage": delayed_mention_required_kind_coverage,
        "delayed_mention_required_min_nonconclusion_items": (
            required_min_nonconclusion_items or None
        ),
        "delayed_mention_required_min_kind_diversity": (
            required_min_kind_diversity or None
        ),
        "delayed_mention_min_nonconclusion_satisfied": (
            delayed_mention_min_nonconclusion_satisfied
        ),
        "delayed_mention_min_kind_diversity_satisfied": (
            delayed_mention_min_kind_diversity_satisfied
        ),
        "delayed_mention_hazard_plan_count": delayed_mention_hazard_plan_count,
        "delayed_mention_min_nonconclusion_items": delayed_mention_min_nonconclusion_items_cfg,
        "delayed_mention_min_kind_diversity": delayed_mention_min_kind_diversity_cfg,
        "delayed_mention_diversity_repair": delayed_mention_diversity_repair,
        "adaptive_hazard_policy": adaptive_hazard_policy,
        "adaptive_hazard_profile": adaptive_hazard_profile,
        "adaptive_hazard_stage_policy": adaptive_hazard_stage_policy,
        "adaptive_hazard_embedding_guard": adaptive_hazard_embedding_guard,
        "delayed_mention_leak_policy": delayed_mention_leak_policy,
        "delayed_mention_leak_threshold": delayed_mention_leak_threshold,
        "delayed_mention_mentioned_count": delayed_mention_mentioned_count,
        "delayed_mention_expired_count": delayed_mention_expired_count,
        "delayed_mention_mention_rate": delayed_mention_mention_rate,
        "delayed_mention_nonconclusion_mention_rate": delayed_mention_nonconclusion_mention_rate,
        "delayed_mention_within_window_rate": delayed_mention_within_window_rate,
        "delayed_mention_option_stage_mention_rate": (
            delayed_mention_option_stage_mention_rate
        ),
        "delayed_mention_option_stage_within_window_rate": (
            delayed_mention_option_stage_within_window_rate
        ),
        "delayed_mention_option_stage_on_support_rate": (
            delayed_mention_option_stage_on_support_rate
        ),
        "delayed_mention_final_risk_packet_mention_rate": (
            delayed_mention_final_risk_packet_mention_rate
        ),
        "delayed_mention_final_risk_packet_within_window_rate": (
            delayed_mention_final_risk_packet_within_window_rate
        ),
        "delayed_mention_final_risk_packet_on_support_rate": (
            delayed_mention_final_risk_packet_on_support_rate
        ),
        "avg_delayed_mention_delay_turns": mean_or_none(delayed_mention_delays),
        "avg_planned_delayed_mention_hazard_support_width_turns": mean_or_none(
            planned_delayed_mention_hazard_support_widths
        ),
        "avg_delayed_mention_hazard_turn_prob_at_mention": mean_or_none(
            delayed_mention_hazard_turn_probs_at_mention
        ),
        "avg_delayed_mention_peak_support_ratio_at_mention": mean_or_none(
            delayed_mention_peak_support_ratios_at_mention
        ),
        "delayed_mention_on_support_rate": (
            delayed_mention_hazard_on_support_count / delayed_mention_hazard_mention_count
            if delayed_mention_hazard_mention_count
            else None
        ),
        "avg_suppressed_delayed_mention_count": mean_or_none(
            suppressed_delayed_mention_counts
        ),
        "avg_adaptive_hazard_multiplier": mean_or_none(adaptive_hazard_multipliers),
        "adaptive_hazard_intervention_rate": (
            adaptive_hazard_intervention_count / adaptive_hazard_total_count
            if adaptive_hazard_total_count
            else None
        ),
        "avg_adaptive_hazard_turn_prob_shift": mean_or_none(
            adaptive_hazard_turn_prob_shifts
        ),
        "avg_adaptive_hazard_threshold": mean_or_none(adaptive_hazard_thresholds),
        "avg_adaptive_hazard_threshold_shift": mean_or_none(
            adaptive_hazard_threshold_shifts
        ),
        "avg_option_stage_adaptive_hazard_multiplier": mean_or_none(
            option_stage_adaptive_hazard_multipliers
        ),
        "avg_option_stage_adaptive_threshold_shift": mean_or_none(
            option_stage_adaptive_threshold_shifts
        ),
        "avg_final_risk_packet_adaptive_hazard_multiplier": mean_or_none(
            final_risk_packet_adaptive_hazard_multipliers
        ),
        "avg_final_risk_packet_adaptive_threshold_shift": mean_or_none(
            final_risk_packet_adaptive_threshold_shifts
        ),
        "avg_adaptive_embedding_prepeak_penalty": mean_or_none(
            adaptive_embedding_prepeak_penalties
        ),
        "avg_adaptive_embedding_prepeak_peak_gap_factor": mean_or_none(
            adaptive_embedding_prepeak_peak_gap_factors
        ),
        "adaptive_hazard_reason_counts": count_strings(adaptive_hazard_reasons) or None,
        "avg_conclusion_adaptive_hazard_turn_prob": mean_or_none(
            conclusion_adaptive_hazard_turn_probs
        ),
        "avg_conclusion_adaptive_hazard_multiplier": mean_or_none(
            conclusion_adaptive_hazard_multipliers
        ),
        "avg_conclusion_adaptive_threshold_shift": mean_or_none(
            conclusion_adaptive_threshold_shifts
        ),
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
        "deferred_intent_hazard_plan_count": hazard_plan_count,
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
        "avg_planned_hazard_support_width_turns": mean_or_none(planned_hazard_support_widths),
        "avg_fire_window_width_turns": mean_or_none(fire_window_widths),
        "avg_fire_window_position_ratio": mean_or_none(fire_window_position_ratios),
        "avg_hazard_turn_prob_at_fire": mean_or_none(hazard_turn_probs_at_fire),
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
        "latent_trace_count": len(latent_trace_events),
        "latent_judge_source": latent_judge_source,
        "latent_judge_provider": latent_judge_provider,
        "latent_judge_model": latent_judge_model,
        "avg_latent_alignment": mean_or_none(latent_alignments),
        "avg_latent_readiness": mean_or_none(latent_readiness_values),
        "avg_latent_leakage_risk": mean_or_none(latent_leakage_risks),
        "latent_alignment_slope": sequence_slope(latent_alignments),
        "latent_semantic_leakage_count": latent_semantic_leakage_count,
        "latent_semantic_leakage_rate": (
            latent_semantic_leakage_count / latent_prewindow_trace_count
            if latent_prewindow_trace_count
            else None
        ),
        "avg_articulation_gap_turns": mean_or_none(articulation_gaps),
        "latent_stage_counts": count_strings(latent_stage_counts) or None,
        "latent_signal_counts": count_strings(latent_signal_counts) or None,
        "embedding_trace_count": len(embedding_trace_events),
        "embedding_judge_source": embedding_judge_source,
        "embedding_judge_provider": embedding_judge_provider,
        "embedding_judge_model": embedding_judge_model,
        "avg_embedding_alignment": mean_or_none(embedding_alignments),
        "avg_embedding_line_alignment": mean_or_none(embedding_line_alignments),
        "avg_embedding_keyword_alignment": mean_or_none(embedding_keyword_alignments),
        "embedding_alignment_slope": sequence_slope(embedding_alignments),
        "embedding_semantic_leakage_count": embedding_semantic_leakage_count,
        "embedding_semantic_leakage_rate": (
            embedding_semantic_leakage_count / embedding_prewindow_trace_count
            if embedding_prewindow_trace_count
            else None
        ),
        "avg_embedding_articulation_gap_turns": mean_or_none(
            embedding_articulation_gaps
        ),
        "avg_semantic_judge_alignment_gap": mean_or_none(
            semantic_judge_alignment_gaps
        ),
        "semantic_judge_disagreement_rate": (
            semantic_judge_disagreement_count / len(semantic_judge_alignment_gaps)
            if semantic_judge_alignment_gaps
            else None
        ),
        "plan_strategy_counts": plan_strategy_counts or None,
        "plan_signal_counts": plan_signal_counts or None,
        "decision_strategy_counts": decision_strategy_counts or None,
        "decision_signal_counts": decision_signal_counts or None,
        "conclusion_delay_strategy_counts": count_strings(conclusion_delay_strategies) or None,
        "conclusion_delay_signal_counts": count_strings(conclusion_delay_signals) or None,
        "perturbation_label": perturbation.get("label"),
        "perturbation_turn": perturbation_turn,
        "post_perturbation_turn_count": post_perturbation_turn_count,
        "post_perturbation_required_keyword_coverage": post_perturbation_required_keyword_coverage,
        "post_perturbation_forbidden_total_hits": post_perturbation_forbidden_total_hits,
        "post_perturbation_forbidden_turn_rate": post_perturbation_forbidden_turn_rate,
        "recovery_after_perturbation": recovery_after_perturbation,
        "recovery_after_perturbation_rate": recovery_after_perturbation_rate,
        "time_to_recover_turns": time_to_recover_turns,
        "probe_post_perturbation_count": probe_post_perturbation_count,
        "probe_post_perturbation_forbidden_turn_rate": probe_post_perturbation_forbidden_turn_rate,
        "probe_recovery_after_perturbation": probe_recovery_after_perturbation,
        "probe_recovery_after_perturbation_rate": probe_recovery_after_perturbation_rate,
        "probe_time_to_recover_turns": probe_time_to_recover_turns,
        "probe_to_reply_recovery_gap_turns": probe_to_reply_recovery_gap_turns,
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


def aggregate_summary_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...] = ("arm", "provider", "model"),
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple("" if v is None else str(v) for v in item)):
        group_rows = grouped[key]
        aggregate: dict[str, Any] = {
            field: value for field, value in zip(group_keys, key)
        }
        aggregate["run_count"] = len(group_rows)

        constant_fields = [
            "arm_description",
            "deferred_intent_backend",
            "deferred_intent_timing",
            "deferred_intent_plan_policy",
            "delayed_mention_leak_policy",
            "delayed_mention_leak_threshold",
            "adaptive_hazard_stage_policy",
        ]
        for field in constant_fields:
            values = [row.get(field) for row in group_rows if row.get(field) is not None]
            unique_values = []
            for value in values:
                if value not in unique_values:
                    unique_values.append(value)
            if len(unique_values) == 1:
                aggregate[field] = unique_values[0]

        numeric_keys: list[str] = []
        seen_numeric: set[str] = set()
        for row in group_rows:
            for field, value in row.items():
                if field in group_keys or field in {"file", "run_name", "run_index", "random_seed"}:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if field in seen_numeric:
                    continue
                seen_numeric.add(field)
                numeric_keys.append(field)

        metrics: dict[str, dict[str, Any]] = {}
        for field in numeric_keys:
            values = [
                float(row[field])
                for row in group_rows
                if isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)
            ]
            if not values:
                continue
            metrics[field] = {
                "count": len(values),
                "mean": mean_or_none(values),
                "std": stddev_or_none(values),
                "min": min(values),
                "p25": quantile_or_none(values, 0.25),
                "p50": quantile_or_none(values, 0.50),
                "p75": quantile_or_none(values, 0.75),
                "max": max(values),
            }
            if field in AGGREGATE_SELECTED_METRICS:
                aggregate[f"{field}_mean"] = metrics[field]["mean"]
                aggregate[f"{field}_std"] = metrics[field]["std"]
                aggregate[f"{field}_p50"] = metrics[field]["p50"]
        aggregate["metrics"] = metrics
        aggregates.append(aggregate)

    return aggregates


def print_table(rows: list[dict[str, Any]]) -> None:
    cols = [
        "arm",
        "run_index",
        "provider",
        "model",
        "semantic_judge_backend",
        "latent_judge_source",
        "latent_judge_provider",
        "latent_judge_model",
        "embedding_judge_source",
        "embedding_judge_provider",
        "embedding_judge_model",
        "delayed_mention_min_nonconclusion_items",
        "delayed_mention_min_kind_diversity",
        "delayed_mention_diversity_repair",
        "adaptive_hazard_policy",
        "adaptive_hazard_profile",
        "adaptive_hazard_stage_policy",
        "adaptive_hazard_embedding_guard",
        "delayed_mention_leak_policy",
        "delayed_mention_leak_threshold",
        "deferred_intent_backend",
        "deferred_intent_timing",
        "turns",
        "probe_count",
        "conclusion_line_mention_rate",
        "conclusion_any_mention_rate",
        "conclusion_plan_within_window_rate",
        "conclusion_on_support_rate",
        "delayed_mention_plan_warning_count",
        "delayed_mention_planned_nonconclusion_count",
        "delayed_mention_kind_diversity",
        "delayed_mention_nonconclusion_share",
        "delayed_mention_required_kind_coverage",
        "delayed_mention_min_nonconclusion_satisfied",
        "delayed_mention_min_kind_diversity_satisfied",
        "delayed_mention_nonconclusion_mention_rate",
        "delayed_mention_within_window_rate",
        "delayed_mention_on_support_rate",
        "delayed_mention_option_stage_mention_rate",
        "delayed_mention_option_stage_within_window_rate",
        "delayed_mention_option_stage_on_support_rate",
        "delayed_mention_final_risk_packet_mention_rate",
        "delayed_mention_final_risk_packet_within_window_rate",
        "delayed_mention_final_risk_packet_on_support_rate",
        "avg_suppressed_delayed_mention_count",
        "avg_adaptive_hazard_multiplier",
        "adaptive_hazard_intervention_rate",
        "avg_adaptive_hazard_turn_prob_shift",
        "avg_adaptive_hazard_threshold_shift",
        "avg_option_stage_adaptive_hazard_multiplier",
        "avg_option_stage_adaptive_threshold_shift",
        "avg_final_risk_packet_adaptive_hazard_multiplier",
        "avg_final_risk_packet_adaptive_threshold_shift",
        "avg_adaptive_embedding_prepeak_penalty",
        "avg_adaptive_embedding_prepeak_peak_gap_factor",
        "avg_conclusion_adaptive_hazard_multiplier",
        "avg_conclusion_adaptive_hazard_turn_prob",
        "delayed_mention_injected_on_mention_rate",
        "avg_conclusion_line_mention_delay_turns",
        "avg_conclusion_any_mention_delay_turns",
        "avg_conclusion_hazard_turn_prob_at_mention",
        "avg_conclusion_peak_support_ratio_at_mention",
        "avg_delayed_mention_hazard_turn_prob_at_mention",
        "avg_delayed_mention_peak_support_ratio_at_mention",
        "deferred_intent_plan_count",
        "deferred_intent_fire_count",
        "deferred_intent_realization_rate",
        "avg_planned_window_width_turns",
        "avg_fire_window_position_ratio",
        "avg_deferred_intent_reply_overlap",
        "avg_hazard_turn_prob_at_fire",
        "avg_probe_reply_overlap",
        "avg_latent_alignment",
        "latent_alignment_slope",
        "latent_semantic_leakage_rate",
        "avg_articulation_gap_turns",
        "avg_embedding_alignment",
        "embedding_alignment_slope",
        "embedding_semantic_leakage_rate",
        "avg_embedding_articulation_gap_turns",
        "semantic_judge_disagreement_rate",
        "recovery_after_perturbation_rate",
        "time_to_recover_turns",
        "probe_recovery_after_perturbation_rate",
        "probe_time_to_recover_turns",
        "probe_to_reply_recovery_gap_turns",
        "post_perturbation_forbidden_turn_rate",
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
