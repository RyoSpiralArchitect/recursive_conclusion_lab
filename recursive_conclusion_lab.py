#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
recursive_conclusion_lab.py

Unified abstraction layer for multiple LLM APIs plus an experiment harness that can:
1) compress memory recursively into "memory capsules"
2) every N turns ask for the dialogue's likely end-state / terminal conclusion
3) optionally feed that conclusion back into the next turn as a soft steering hint
4) maintain "deferred utterance intents" that are planned now and softly fired later

Supported providers:
- openai    -> OpenAI Responses API
- anthropic -> Anthropic Messages API
- mistral   -> Mistral Chat Completions API
- gemini    -> Gemini generateContent API
- hf        -> Hugging Face Inference Providers (OpenAI-compatible chat-completions)
- dummy     -> local deterministic mock, useful for testing without API keys
"""
from __future__ import annotations

import abc
import argparse
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import textwrap
import time
from typing import Any, Iterable, Optional


# ----------------------------
# Core data models
# ----------------------------

@dataclass
class ChatMessage:
    role: str  # user | assistant
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported role: {self.role!r}")
        self.content = self.content.strip()


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 900
    timeout_seconds: int = 120


@dataclass
class ProviderResponse:
    provider: str
    model: str
    text: str
    raw: dict[str, Any]
    usage: Optional[dict[str, Any]] = None
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class EmbeddingResponse:
    provider: str
    model: str
    embeddings: list[list[float]]
    raw: dict[str, Any]
    usage: Optional[dict[str, Any]] = None
    request_id: Optional[str] = None


class ConclusionMode(str, Enum):
    OBSERVE = "observe"
    SOFT_STEER = "soft_steer"


class SteerStrength(str, Enum):
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"


class ConclusionSteerInjection(str, Enum):
    FULL = "full"
    CONCLUSION_LINE = "conclusion_line"


class DelayedMentionMode(str, Enum):
    OBSERVE = "observe"
    SOFT_FIRE = "soft_fire"


class DelayedMentionLeakPolicy(str, Enum):
    OFF = "off"
    ON = "on"


class DelayedMentionDiversityRepairPolicy(str, Enum):
    OFF = "off"
    ON = "on"


class AdaptiveHazardPolicy(str, Enum):
    STATIC = "static"
    ADAPTIVE = "adaptive"


class AdaptiveHazardProfile(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    EAGER = "eager"


class AdaptiveHazardEmbeddingGuard(str, Enum):
    OFF = "off"
    ON = "on"


class AdaptiveHazardStagePolicy(str, Enum):
    FLAT = "flat"
    KIND_AWARE = "kind_aware"


class SemanticJudgeBackend(str, Enum):
    OFF = "off"
    LLM = "llm"
    EMBEDDING = "embedding"
    BOTH = "both"


class DeferredIntentMode(str, Enum):
    OBSERVE = "observe"
    SOFT_FIRE = "soft_fire"
    HARD_FIRE = "hard_fire"


class DeferredIntentStrategy(str, Enum):
    FIXED = "fixed"
    TRIGGER = "trigger"
    ADAPTIVE = "adaptive"


class DeferredIntentTiming(str, Enum):
    OFFSET = "offset"
    MODEL = "model"
    HAZARD = "hazard"


class DeferredIntentPlanPolicy(str, Enum):
    PERIODIC = "periodic"
    AUTO = "auto"


class DeferredIntentBackend(str, Enum):
    EXTERNAL = "external"
    INBAND = "inband"


class DeferredIntentLatentInjection(str, Enum):
    OFF = "off"
    ACTIVE = "active"


class DeferredIntentAblation(str, Enum):
    NONE = "none"
    DELETE_PLANNED = "delete_planned"


@dataclass
class ExperimentConfig:
    base_system: str = ""
    recent_window_messages: int = 8
    memory_every: int = 3
    memory_capsule_limit: int = 4
    memory_word_budget: int = 140
    conclusion_every: int = 3
    conclusion_mode: ConclusionMode = ConclusionMode.OBSERVE
    conclusion_steer_strength: SteerStrength = SteerStrength.MEDIUM
    conclusion_steer_injection: ConclusionSteerInjection = ConclusionSteerInjection.FULL
    delayed_mention_every: int = 0
    delayed_mention_item_limit: int = 3
    delayed_mention_min_nonconclusion_items: int = 1
    delayed_mention_min_kind_diversity: int = 2
    delayed_mention_diversity_repair: DelayedMentionDiversityRepairPolicy = (
        DelayedMentionDiversityRepairPolicy.ON
    )
    delayed_mention_mode: DelayedMentionMode = DelayedMentionMode.OBSERVE
    delayed_mention_fire_prob: float = 0.35
    delayed_mention_fire_max_items: int = 2
    delayed_mention_leak_policy: DelayedMentionLeakPolicy = DelayedMentionLeakPolicy.ON
    delayed_mention_leak_threshold: float = 0.05
    adaptive_hazard_policy: AdaptiveHazardPolicy = AdaptiveHazardPolicy.ADAPTIVE
    adaptive_hazard_profile: AdaptiveHazardProfile = AdaptiveHazardProfile.BALANCED
    adaptive_hazard_stage_policy: AdaptiveHazardStagePolicy = (
        AdaptiveHazardStagePolicy.FLAT
    )
    adaptive_hazard_embedding_guard: AdaptiveHazardEmbeddingGuard = (
        AdaptiveHazardEmbeddingGuard.OFF
    )
    latent_convergence_every: int = 0
    semantic_judge_backend: SemanticJudgeBackend = SemanticJudgeBackend.LLM
    deferred_intent_every: int = 0
    deferred_intent_mode: DeferredIntentMode = DeferredIntentMode.OBSERVE
    deferred_intent_strategy: DeferredIntentStrategy = DeferredIntentStrategy.TRIGGER
    deferred_intent_timing: DeferredIntentTiming = DeferredIntentTiming.OFFSET
    deferred_intent_offset: int = 3
    deferred_intent_grace: int = 2
    deferred_intent_limit: int = 6
    deferred_intent_plan_policy: DeferredIntentPlanPolicy = DeferredIntentPlanPolicy.PERIODIC
    deferred_intent_plan_budget: int = 0
    deferred_intent_plan_max_new: int = 1
    deferred_intent_backend: DeferredIntentBackend = DeferredIntentBackend.EXTERNAL
    deferred_intent_latent_injection: DeferredIntentLatentInjection = DeferredIntentLatentInjection.OFF
    deferred_intent_ablation: DeferredIntentAblation = DeferredIntentAblation.NONE
    show_probe_outputs: bool = True
    reply_config: GenerationConfig = field(default_factory=GenerationConfig)
    probe_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(
            temperature=0.0, max_tokens=220, timeout_seconds=120
        )
    )


@dataclass
class EventRecord:
    timestamp: float
    provider: str
    model: str
    turn_index: int
    event_type: str
    payload: dict[str, Any]


@dataclass
class CompareExecutionResult:
    rows: list[dict[str, Any]]
    log_paths: list[Path]


@dataclass
class DeferredIntent:
    intent_id: str
    created_turn: int
    kind: str
    intent: str
    why_not_now: str = ""
    earliest_turn: int = 0
    latest_turn: int = 0
    hazard_profile: list[dict[str, Any]] = field(default_factory=list)
    trigger: list[str] = field(default_factory=list)
    cancel_if: list[str] = field(default_factory=list)
    confidence: float = 0.0
    priority: float = 0.0
    plan_strategy: str = ""
    plan_signals: list[str] = field(default_factory=list)
    plan_rationale: str = ""
    decision_strategy: str = ""
    decision_signals: list[str] = field(default_factory=list)
    decision_rationale: str = ""
    status: str = "active"
    revision_count: int = 0
    fire_turn: Optional[int] = None
    terminal_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def summary_line(self) -> str:
        window = f"t{self.earliest_turn}..t{self.latest_turn}"
        return f"[{self.intent_id} {self.status} {window}] {self.intent}"


@dataclass
class DelayedMentionItem:
    item_id: str
    created_turn: int
    kind: str
    text: str
    keywords: list[str] = field(default_factory=list)
    earliest_turn: int = 0
    latest_turn: int = 0
    hazard_profile: list[dict[str, Any]] = field(default_factory=list)
    likelihood: float = 0.0
    delay_strategy: str = ""
    delay_signals: list[str] = field(default_factory=list)
    delay_rationale: str = ""
    release_stage_role: str = ""
    status: str = "active"
    mention_turn: Optional[int] = None
    terminal_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def summary_line(self) -> str:
        window = f"t{self.earliest_turn}..t{self.latest_turn}"
        return f"[{self.item_id} {self.kind} {self.status} {window}] {self.text}"


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_jsonl(path: Optional[Path], record: EventRecord) -> None:
    if path is None:
        return
    ensure_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n")


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def render_messages(messages: Iterable[ChatMessage]) -> str:
    chunks: list[str] = []
    for idx, msg in enumerate(messages, start=1):
        content = strip_rcl_state(msg.content)
        chunks.append(f"{idx:02d}. {msg.role.upper()}: {content}")
    return "\n".join(chunks) if chunks else "(empty)"


def render_capsules(capsules: list[str]) -> str:
    if not capsules:
        return "(none)"
    return "\n\n".join(f"[Capsule {i+1}]\n{caps}" for i, caps in enumerate(capsules))


def render_deferred_intents(intents: Iterable[DeferredIntent]) -> str:
    items = list(intents)
    if not items:
        return "(none)"
    return "\n".join(intent.summary_line() for intent in items)


def coerce_text(value: Any) -> str:
    """Best-effort extraction from provider-specific content shapes."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = coerce_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), (str, list, dict)):
            return coerce_text(value.get("content"))
        if isinstance(value.get("parts"), list):
            parts = []
            for part in value["parts"]:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            if parts:
                return "\n".join(parts).strip()
        if isinstance(value.get("message"), (str, list, dict)):
            return coerce_text(value["message"])
    return ""


def lexical_overlap(a: str, b: str) -> float:
    toks_a = {t.lower() for t in re.findall(r"\w+", a)}
    toks_b = {t.lower() for t in re.findall(r"\w+", b)}
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def normalize_vector(values: list[float]) -> list[float]:
    norm = sum(v * v for v in values) ** 0.5
    if norm <= 0.0:
        return [0.0 for _ in values]
    return [v / norm for v in values]


def hashed_embedding_vector(text: str, *, dims: int = 128) -> list[float]:
    dims = max(8, int(dims))
    vec = [0.0] * dims
    cleaned = compact_text(text).lower()
    if not cleaned:
        return vec

    tokens = re.findall(r"\w+", cleaned)
    features = list(tokens)
    features.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
    joined = re.sub(r"\s+", " ", cleaned)
    features.extend(joined[idx : idx + 3] for idx in range(max(0, len(joined) - 2)))
    if not features:
        return vec

    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        slot = int.from_bytes(digest[:2], "big") % dims
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        weight = 1.0 + min(len(feature), 12) / 12.0
        vec[slot] += sign * weight
    return normalize_vector(vec)


def cosine_similarity(a: list[float], b: list[float]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def cosine_alignment(a: list[float], b: list[float]) -> Optional[float]:
    similarity = cosine_similarity(a, b)
    if similarity is None:
        return None
    return max(0.0, min(1.0, 0.5 * (similarity + 1.0)))


def extract_probe_line_value(text: str, key: str) -> str:
    pattern = rf"(?im)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$"
    match = re.search(pattern, text or "")
    if not match:
        return ""
    return match.group(1).strip()


def parse_probe_float(raw: str) -> Optional[float]:
    text = compact_text(raw)
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value


def parse_probe_int(raw: str) -> Optional[int]:
    text = compact_text(raw)
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def parse_probe_list(raw: str, *, max_items: int = 8, max_item_chars: int = 60) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            loaded = json.loads(text)
        except Exception:
            loaded = None
        if isinstance(loaded, list):
            items = [compact_text(str(v)) for v in loaded]
            out: list[str] = []
            seen: set[str] = set()
            for item in items:
                item = item.strip().strip('"').strip("'")
                if not item:
                    continue
                item = item[:max_item_chars]
                key = item.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
                if len(out) >= max_items:
                    break
            return out

    parts = re.split(r"[;,|\n]\s*", text)
    out = []
    seen = set()
    for part in parts:
        item = compact_text(part).strip().strip('"').strip("'")
        if not item:
            continue
        item = item[:max_item_chars]
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max_items:
            break
    return out


def keyword_hits(keywords: list[str], text: str) -> Optional[int]:
    if not keywords:
        return None
    lower = (text or "").lower()
    hits = 0
    for kw in keywords:
        kw = compact_text(kw)
        if not kw:
            continue
        if kw.lower() in lower:
            hits += 1
    return hits


def keyword_coverage(keywords: list[str], text: str) -> Optional[float]:
    if not keywords:
        return None
    hits = keyword_hits(keywords, text)
    if hits is None:
        return None
    return hits / max(1, len(keywords))


def count_strings(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = compact_text(str(value))
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def extract_conclusion_line(text: str) -> str:
    match = re.search(r"(?im)^\s*conclusion\s*:\s*(.+?)\s*$", text or "")
    if not match:
        return ""
    return f"CONCLUSION: {match.group(1).strip()}"


def extract_conclusion_probe_block(text: str) -> str:
    lines: list[str] = []
    for key in ("CONCLUSION", "CONFIDENCE", "EVIDENCE"):
        value = extract_probe_line_value(text, key)
        if value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines).strip() or (text or "").strip()


def strip_conclusion_prefix(line: str) -> str:
    return re.sub(r"(?im)^\s*conclusion\s*:\s*", "", line or "").strip()


RCL_STATE_OPEN = "<RCL_STATE>"
RCL_STATE_CLOSE = "</RCL_STATE>"


def split_rcl_state(text: str) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
    raw = text or ""
    start = raw.rfind(RCL_STATE_OPEN)
    end = raw.rfind(RCL_STATE_CLOSE)
    if start == -1 or end == -1 or end < start:
        return raw.strip(), None, None
    visible = raw[:start].rstrip()
    blob = raw[start + len(RCL_STATE_OPEN) : end].strip()
    try:
        data = json.loads(blob)
    except Exception as exc:
        return visible, None, f"Invalid RCL_STATE JSON: {exc}"
    if not isinstance(data, dict):
        return visible, None, "RCL_STATE must be a JSON object."
    return visible, data, None


def strip_rcl_state(text: str) -> str:
    visible, _, _ = split_rcl_state(text)
    return visible


INBAND_STATE_MAX_CHARS = 8000
INBAND_INTENT_TEXT_MAX_CHARS = 280
INBAND_WHY_NOT_NOW_MAX_CHARS = 160
INBAND_TERMINAL_REASON_MAX_CHARS = 160
INBAND_STRATEGY_MAX_CHARS = 48
INBAND_RATIONALE_MAX_CHARS = 200
INBAND_SIGNAL_MAX_ITEMS = 6
INBAND_SIGNAL_ITEM_MAX_CHARS = 80
INBAND_CONDITION_MAX_ITEMS = 4
INBAND_CONDITION_ITEM_MAX_CHARS = 120
INBAND_HAZARD_MAX_ITEMS = 8


def truncate_text(value: Any, max_chars: int) -> str:
    text = compact_text(str(value or ""))
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    suffix = "…"
    head = text[: max(0, max_chars - len(suffix))].rstrip()
    return (head + suffix).strip()


def truncate_conditions(items: Any) -> list[str]:
    result: list[str] = []
    for item in coerce_str_list(items)[:INBAND_CONDITION_MAX_ITEMS]:
        result.append(truncate_text(item, INBAND_CONDITION_ITEM_MAX_CHARS))
    return result


def truncate_signals(items: Any) -> list[str]:
    result: list[str] = []
    for item in coerce_str_list(items)[:INBAND_SIGNAL_MAX_ITEMS]:
        result.append(truncate_text(item, INBAND_SIGNAL_ITEM_MAX_CHARS))
    return result


def parse_probe_bool(raw: str) -> Optional[bool]:
    text = compact_text(raw).lower()
    if text in {"yes", "true", "1"}:
        return True
    if text in {"no", "false", "0"}:
        return False
    return None


def parse_hazard_profile(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        loaded = extract_json_value(value)
        if loaded is not None:
            value = loaded
    points: list[tuple[int, float]] = []
    if isinstance(value, dict):
        delay_turn_probs = value.get("delay_turn_probs")
        if isinstance(delay_turn_probs, dict):
            for raw_delay, raw_prob in delay_turn_probs.items():
                delay = coerce_int(raw_delay)
                prob = clamp01(raw_prob, default=0.0)
                if delay is None or delay < 0 or prob <= 0.0:
                    continue
                points.append((delay, prob))
        else:
            for key in ("hazard_profile", "hazard", "points"):
                nested = value.get(key)
                if isinstance(nested, list):
                    value = nested
                    break

    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            delay = coerce_int(
                item.get("delay_turns")
                or item.get("delay")
                or item.get("turn")
                or item.get("offset")
            )
            prob = clamp01(
                item.get("prob")
                or item.get("weight")
                or item.get("hazard")
                or item.get("p"),
                default=0.0,
            )
            if delay is None or delay < 0 or prob <= 0.0:
                continue
            points.append((delay, prob))

    merged: dict[int, float] = {}
    for delay, prob in points:
        merged[delay] = merged.get(delay, 0.0) + float(prob)

    total = sum(merged.values())
    if total <= 0.0:
        return []

    normalized: list[dict[str, Any]] = []
    for delay in sorted(merged):
        normalized.append(
            {
                "delay_turns": int(delay),
                "prob": float(merged[delay] / total),
            }
        )
    normalized = normalized[:INBAND_HAZARD_MAX_ITEMS]
    clipped_total = sum(clamp01(item.get("prob"), default=0.0) for item in normalized)
    if clipped_total <= 0.0:
        return []
    for item in normalized:
        item["prob"] = clamp01(item.get("prob"), default=0.0) / clipped_total
    return normalized


def default_hazard_profile(*, offset: int, grace: int) -> list[dict[str, Any]]:
    start = max(0, int(offset))
    stop = start + max(0, int(grace))
    delays = list(range(start, stop + 1))
    if not delays:
        delays = [0]
    weights = [1.0 / (idx + 1) for idx, _ in enumerate(delays)]
    total = sum(weights) or 1.0
    return [
        {
            "delay_turns": delay,
            "prob": weight / total,
        }
        for delay, weight in zip(delays, weights)
    ]


def hazard_profile_from_bounds(
    *,
    created_turn: int,
    earliest_turn: int,
    latest_turn: int,
) -> list[dict[str, Any]]:
    start = max(0, earliest_turn - created_turn)
    stop = max(start, latest_turn - created_turn)
    delays = list(range(start, stop + 1))
    prob = 1.0 / len(delays)
    return [{"delay_turns": delay, "prob": prob} for delay in delays]


def ensure_hazard_profile(
    profile: list[dict[str, Any]],
    *,
    created_turn: int,
    earliest_turn: Optional[int],
    latest_turn: Optional[int],
    offset: int,
    grace: int,
) -> list[dict[str, Any]]:
    normalized = parse_hazard_profile(profile)
    if normalized:
        return normalized
    if earliest_turn is not None:
        return hazard_profile_from_bounds(
            created_turn=created_turn,
            earliest_turn=earliest_turn,
            latest_turn=max(earliest_turn, int(latest_turn or earliest_turn)),
        )
    return default_hazard_profile(offset=offset, grace=grace)


def hazard_bounds_from_profile(
    *,
    created_turn: int,
    profile: list[dict[str, Any]],
) -> tuple[int, int]:
    normalized = parse_hazard_profile(profile)
    if not normalized:
        return created_turn + 1, created_turn + 1
    delays = [coerce_int(item.get("delay_turns")) or 0 for item in normalized]
    return created_turn + min(delays), created_turn + max(delays)


def hazard_turn_probability(
    profile: list[dict[str, Any]],
    *,
    created_turn: int,
    turn_index: int,
) -> float:
    delay = turn_index - created_turn
    if delay < 0:
        return 0.0
    for item in parse_hazard_profile(profile):
        if (coerce_int(item.get("delay_turns")) or 0) == delay:
            return clamp01(item.get("prob"), default=0.0)
    return 0.0


def hazard_peak_delay(profile: list[dict[str, Any]]) -> Optional[int]:
    normalized = parse_hazard_profile(profile)
    if not normalized:
        return None
    best = max(
        normalized,
        key=lambda item: (
            clamp01(item.get("prob"), default=0.0),
            -(coerce_int(item.get("delay_turns")) or 0),
        ),
    )
    return coerce_int(best.get("delay_turns"))


def hazard_peak_probability(profile: list[dict[str, Any]]) -> float:
    normalized = parse_hazard_profile(profile)
    if not normalized:
        return 0.0
    return max(clamp01(item.get("prob"), default=0.0) for item in normalized)


def hazard_support_size(profile: list[dict[str, Any]]) -> int:
    normalized = parse_hazard_profile(profile)
    return len(normalized)


def adaptive_hazard_profile_params(
    profile: AdaptiveHazardProfile,
) -> dict[str, float]:
    if profile == AdaptiveHazardProfile.CONSERVATIVE:
        return {
            "boost_scale": 0.50,
            "suppress_scale": 0.85,
            "gap_damping": 2.80,
            "min_multiplier": 0.30,
            "max_multiplier": 1.35,
            "threshold_raise": 0.16,
            "peak_pull_scale": 0.95,
            "peak_release_scale": 0.16,
            "peak_target_ratio": 0.78,
            "peak_release_ratio": 0.92,
            "peak_pre_peak_penalty": 0.12,
            "peak_release_bonus": 0.08,
            "threshold_peak_raise": 0.18,
            "threshold_peak_lower": 0.03,
            "embedding_prepeak_target": 0.64,
            "embedding_prepeak_scale": 0.90,
            "threshold_embedding_raise": 0.22,
        }
    if profile == AdaptiveHazardProfile.EAGER:
        return {
            "boost_scale": 0.90,
            "suppress_scale": 0.45,
            "gap_damping": 1.60,
            "min_multiplier": 0.45,
            "max_multiplier": 2.10,
            "threshold_raise": 0.08,
            "peak_pull_scale": 0.55,
            "peak_release_scale": 0.28,
            "peak_target_ratio": 0.60,
            "peak_release_ratio": 0.78,
            "peak_pre_peak_penalty": 0.08,
            "peak_release_bonus": 0.12,
            "threshold_peak_raise": 0.10,
            "threshold_peak_lower": 0.06,
            "embedding_prepeak_target": 0.78,
            "embedding_prepeak_scale": 0.40,
            "threshold_embedding_raise": 0.10,
        }
    return {
        "boost_scale": 0.70,
        "suppress_scale": 0.65,
        "gap_damping": 2.10,
        "min_multiplier": 0.35,
        "max_multiplier": 1.70,
        "threshold_raise": 0.12,
        "peak_pull_scale": 0.78,
        "peak_release_scale": 0.20,
        "peak_target_ratio": 0.68,
        "peak_release_ratio": 0.85,
        "peak_pre_peak_penalty": 0.10,
        "peak_release_bonus": 0.10,
        "threshold_peak_raise": 0.15,
        "threshold_peak_lower": 0.04,
        "embedding_prepeak_target": 0.70,
        "embedding_prepeak_scale": 0.62,
        "threshold_embedding_raise": 0.16,
    }


def classify_delayed_mention_stage_role(
    *,
    kind: str,
    text: str,
    delay_strategy: str = "",
    delay_signals: Optional[list[str]] = None,
) -> str:
    kind_norm = compact_text(kind).lower()
    blob_parts = [kind_norm, compact_text(text).lower(), compact_text(delay_strategy).lower()]
    if delay_signals:
        blob_parts.extend(compact_text(x).lower() for x in delay_signals if x)
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


def adaptive_hazard_stage_role_params(
    policy: AdaptiveHazardStagePolicy,
    stage_role: str,
) -> dict[str, float]:
    if policy != AdaptiveHazardStagePolicy.KIND_AWARE:
        return {
            "focus_scale": 1.0,
            "boost_bias": 0.0,
            "suppress_bias": 0.0,
            "peak_pull_scale": 1.0,
            "peak_release_scale": 1.0,
            "threshold_bias": 0.0,
            "prepeak_release_ratio": 1.1,
            "prepeak_release_bonus": 0.0,
            "prepeak_hold_bonus": 0.0,
        }
    if stage_role == "option_stage":
        return {
            "focus_scale": 1.12,
            "boost_bias": 0.04,
            "suppress_bias": -0.05,
            "peak_pull_scale": 0.70,
            "peak_release_scale": 1.28,
            "threshold_bias": -0.02,
            "prepeak_release_ratio": 0.58,
            "prepeak_release_bonus": 0.12,
            "prepeak_hold_bonus": 0.0,
        }
    if stage_role == "final_risk_packet":
        return {
            "focus_scale": 0.96,
            "boost_bias": -0.01,
            "suppress_bias": 0.08,
            "peak_pull_scale": 1.25,
            "peak_release_scale": 0.92,
            "threshold_bias": 0.02,
            "prepeak_release_ratio": 1.1,
            "prepeak_release_bonus": 0.0,
            "prepeak_hold_bonus": 0.09,
        }
    if stage_role == "support_stage":
        return {
            "focus_scale": 1.04,
            "boost_bias": 0.02,
            "suppress_bias": -0.02,
            "peak_pull_scale": 0.86,
            "peak_release_scale": 1.10,
            "threshold_bias": -0.008,
            "prepeak_release_ratio": 0.72,
            "prepeak_release_bonus": 0.05,
            "prepeak_hold_bonus": 0.0,
        }
    return {
        "focus_scale": 1.0,
        "boost_bias": 0.0,
        "suppress_bias": 0.0,
        "peak_pull_scale": 1.0,
        "peak_release_scale": 1.0,
        "threshold_bias": 0.0,
        "prepeak_release_ratio": 1.1,
        "prepeak_release_bonus": 0.0,
        "prepeak_hold_bonus": 0.0,
    }


def reshape_stage_role_hazard_profile(
    *,
    policy: AdaptiveHazardStagePolicy,
    stage_role: str,
    profile: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = parse_hazard_profile(profile)
    if policy != AdaptiveHazardStagePolicy.KIND_AWARE or not normalized:
        return normalized
    if stage_role in {"", "conclusion"}:
        return normalized

    peak_delay = hazard_peak_delay(normalized)
    if peak_delay is None:
        return normalized

    target_peak_delay: Optional[int] = None
    if stage_role == "support_stage":
        target_peak_delay = 3
    elif stage_role == "option_stage":
        target_peak_delay = 4
    elif stage_role == "final_risk_packet":
        target_peak_delay = 5
    if target_peak_delay is None or peak_delay >= target_peak_delay:
        return normalized

    shift = target_peak_delay - peak_delay
    reshaped: list[dict[str, Any]] = []
    for item in normalized:
        delay = coerce_int(item.get("delay_turns")) or 0
        prob = clamp01(item.get("prob"), default=0.0)
        reshaped.append({"delay_turns": delay + shift, "prob": prob})
    return parse_hazard_profile(reshaped)


def format_hazard_profile_brief(
    profile: list[dict[str, Any]],
    *,
    max_items: int = 4,
) -> str:
    normalized = parse_hazard_profile(profile)
    if not normalized:
        return ""
    parts: list[str] = []
    for item in normalized[:max(1, max_items)]:
        delay = coerce_int(item.get("delay_turns"))
        prob = clamp01(item.get("prob"), default=0.0)
        if delay is None:
            continue
        parts.append(f"{delay}:{prob:.2f}")
    if len(normalized) > max_items:
        parts.append("...")
    return ", ".join(parts)


def resolve_mention_hazard_plan(
    *,
    created_turn: int,
    delay_min: Optional[int],
    delay_max: Optional[int],
    raw_hazard_profile: Any,
    default_min: int = 1,
    default_span: int = 2,
) -> tuple[int, int, list[dict[str, Any]]]:
    if delay_min is not None:
        delay_min = max(0, int(delay_min))
    if delay_max is not None:
        delay_max = max(0, int(delay_max))
    if delay_min is not None and delay_max is not None and delay_max < delay_min:
        delay_min, delay_max = delay_max, delay_min

    resolved_min = delay_min if delay_min is not None else max(0, int(default_min))
    resolved_max = delay_max if delay_max is not None else resolved_min + max(0, int(default_span))
    resolved_max = max(resolved_min, resolved_max)

    earliest_turn = created_turn + resolved_min
    latest_turn = created_turn + resolved_max
    hazard_profile = ensure_hazard_profile(
        parse_hazard_profile(raw_hazard_profile),
        created_turn=created_turn,
        earliest_turn=earliest_turn,
        latest_turn=latest_turn,
        offset=resolved_min,
        grace=max(0, resolved_max - resolved_min),
    )
    earliest_turn, latest_turn = hazard_bounds_from_profile(
        created_turn=created_turn,
        profile=hazard_profile,
    )
    return (
        max(0, earliest_turn - created_turn),
        max(0, latest_turn - created_turn),
        hazard_profile,
    )


def deferred_intent_to_inband_dict(intent: DeferredIntent) -> dict[str, Any]:
    data = intent.to_dict()
    data["intent"] = truncate_text(data.get("intent"), INBAND_INTENT_TEXT_MAX_CHARS)
    data["why_not_now"] = truncate_text(data.get("why_not_now"), INBAND_WHY_NOT_NOW_MAX_CHARS)
    data["plan_strategy"] = truncate_text(data.get("plan_strategy"), INBAND_STRATEGY_MAX_CHARS)
    data["plan_rationale"] = truncate_text(data.get("plan_rationale"), INBAND_RATIONALE_MAX_CHARS)
    data["plan_signals"] = truncate_signals(data.get("plan_signals"))
    data["decision_strategy"] = truncate_text(
        data.get("decision_strategy"), INBAND_STRATEGY_MAX_CHARS
    )
    data["decision_rationale"] = truncate_text(
        data.get("decision_rationale"), INBAND_RATIONALE_MAX_CHARS
    )
    data["decision_signals"] = truncate_signals(data.get("decision_signals"))
    data["terminal_reason"] = truncate_text(
        data.get("terminal_reason"), INBAND_TERMINAL_REASON_MAX_CHARS
    )
    data["trigger"] = truncate_conditions(data.get("trigger"))
    data["cancel_if"] = truncate_conditions(data.get("cancel_if"))
    data["hazard_profile"] = parse_hazard_profile(data.get("hazard_profile"))
    return data


def dump_inband_state(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def build_inband_state_payload(
    intents: list[DeferredIntent],
    *,
    version: int = 1,
    max_chars: int = INBAND_STATE_MAX_CHARS,
) -> tuple[list[DeferredIntent], dict[str, Any], str]:
    carried = list(intents)

    def render(current: list[DeferredIntent]) -> tuple[dict[str, Any], str]:
        state = {"version": int(version or 1), "intents": [deferred_intent_to_inband_dict(i) for i in current]}
        return state, dump_inband_state(state)

    state, blob = render(carried)
    if len(blob) <= max_chars:
        return carried, state, blob

    # Drop terminal intents first (they don't help future planning).
    carried = [i for i in carried if i.status == "active"]
    state, blob = render(carried)
    if len(blob) <= max_chars:
        return carried, state, blob

    # Last resort: drop lowest-priority active intents until it fits (keep at least one).
    while len(blob) > max_chars and len(carried) > 1:
        drop_idx = min(
            range(len(carried)),
            key=lambda idx: (carried[idx].priority, carried[idx].created_turn),
        )
        carried.pop(drop_idx)
        state, blob = render(carried)

    return carried, state, blob


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Export your provider API key before running."
        )
    return value


def post_json(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    try:
        import requests  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: requests. Install it with `pip install requests`."
        ) from exc
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout_seconds,
    )
    if response.status_code >= 400:
        snippet = response.text[:2000]
        raise RuntimeError(
            f"HTTP {response.status_code} from {url}\n"
            f"Request payload:\n{json.dumps(payload, ensure_ascii=False)[:1500]}\n\n"
            f"Response body:\n{snippet}"
        )
    try:
        return response.json()
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Non-JSON response from {url}: {response.text[:2000]}"
        ) from exc


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = compact_text(str(item))
        if text:
            items.append(text)
    return items


def strip_code_fences(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text or "", flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return (text or "").strip()


def extract_json_value(text: str) -> Any:
    cleaned = strip_code_fences(text)
    decoder = json.JSONDecoder()
    for candidate in (cleaned,):
        try:
            return json.loads(candidate)
        except Exception:
            pass
        starts = []
        for start_char in ("{", "["):
            idx = candidate.find(start_char)
            if idx != -1:
                starts.append((idx, start_char))
        for idx, _ in sorted(starts, key=lambda item: item[0]):
            try:
                value, _ = decoder.raw_decode(candidate[idx:])
                return value
            except Exception:
                continue
    return None


def build_deferred_intent_from_plan(
    data: dict[str, Any],
    *,
    intent_id: str,
    created_turn: int,
    strategy: DeferredIntentStrategy,
    timing: DeferredIntentTiming,
    offset: int,
    grace: int,
) -> Optional[DeferredIntent]:
    create_intent = data.get("create_intent", True)
    if isinstance(create_intent, str):
        create_intent = create_intent.strip().lower() not in {"false", "0", "no", "none"}
    if not create_intent:
        return None

    intent = compact_text(str(data.get("intent", "")))
    if not intent:
        return None

    kind = compact_text(str(data.get("kind", "other"))) or "other"
    why_not_now = compact_text(str(data.get("why_not_now", "")))

    selection = data.get("selection") if isinstance(data.get("selection"), dict) else {}
    plan_strategy = compact_text(
        str(
            selection.get("strategy")
            or data.get("plan_strategy")
            or data.get("selection_strategy")
            or ""
        )
    )
    plan_rationale = compact_text(
        str(
            selection.get("rationale")
            or data.get("plan_rationale")
            or data.get("selection_rationale")
            or ""
        )
    )
    plan_signals = coerce_str_list(
        selection.get("signals") or data.get("plan_signals") or data.get("selection_signals")
    )

    if strategy == DeferredIntentStrategy.FIXED:
        trigger = []
        cancel_if = []
    else:
        trigger = coerce_str_list(data.get("trigger"))
        cancel_if = coerce_str_list(data.get("cancel_if"))

    default_earliest_turn = created_turn + max(1, offset)
    default_latest_turn = (
        default_earliest_turn
        if strategy == DeferredIntentStrategy.FIXED
        else default_earliest_turn + max(0, grace)
    )

    earliest_turn = default_earliest_turn
    latest_turn = default_latest_turn
    hazard_profile: list[dict[str, Any]] = []
    if timing == DeferredIntentTiming.MODEL:
        timing_block = data.get("timing") if isinstance(data.get("timing"), dict) else {}
        delay_min = coerce_int(
            timing_block.get("delay_min_turns")
            or timing_block.get("delay_min")
            or data.get("delay_min_turns")
            or data.get("delay_min")
        )
        delay_max = coerce_int(
            timing_block.get("delay_max_turns")
            or timing_block.get("delay_max")
            or data.get("delay_max_turns")
            or data.get("delay_max")
        )
        earliest_abs = coerce_int(timing_block.get("earliest_turn") or data.get("earliest_turn"))
        latest_abs = coerce_int(timing_block.get("latest_turn") or data.get("latest_turn"))

        if delay_min is not None:
            earliest_turn = created_turn + max(1, delay_min)
        elif earliest_abs is not None:
            earliest_turn = earliest_abs

        if earliest_turn <= created_turn:
            earliest_turn = created_turn + 1

        if delay_max is not None:
            if delay_min is not None:
                delay_max = max(delay_min, delay_max)
            latest_turn = created_turn + max(1, delay_max)
        elif latest_abs is not None:
            latest_turn = latest_abs
        else:
            latest_turn = default_latest_turn

        if latest_turn < earliest_turn:
            latest_turn = earliest_turn

        if strategy == DeferredIntentStrategy.FIXED:
            latest_turn = earliest_turn
    elif timing == DeferredIntentTiming.HAZARD:
        timing_block = data.get("timing") if isinstance(data.get("timing"), dict) else {}
        hazard_profile = ensure_hazard_profile(
            parse_hazard_profile(
                timing_block.get("hazard_profile")
                or timing_block.get("hazard")
                or data.get("hazard_profile")
                or data.get("hazard")
                or data
            ),
            created_turn=created_turn,
            earliest_turn=None,
            latest_turn=None,
            offset=offset,
            grace=grace,
        )
        earliest_turn, latest_turn = hazard_bounds_from_profile(
            created_turn=created_turn,
            profile=hazard_profile,
        )
        if strategy == DeferredIntentStrategy.FIXED:
            peak_delay = hazard_peak_delay(hazard_profile)
            if peak_delay is None:
                peak_delay = max(0, offset)
            hazard_profile = [{"delay_turns": peak_delay, "prob": 1.0}]
            earliest_turn = created_turn + peak_delay
            latest_turn = earliest_turn

    return DeferredIntent(
        intent_id=intent_id,
        created_turn=created_turn,
        kind=kind,
        intent=intent,
        why_not_now=why_not_now,
        earliest_turn=earliest_turn,
        latest_turn=max(earliest_turn, latest_turn),
        hazard_profile=hazard_profile,
        trigger=trigger,
        cancel_if=cancel_if,
        confidence=clamp01(data.get("confidence"), default=0.0),
        priority=clamp01(data.get("priority"), default=0.5),
        plan_strategy=plan_strategy,
        plan_rationale=plan_rationale,
        plan_signals=plan_signals,
    )


def apply_revised_intent(
    intent: DeferredIntent,
    data: dict[str, Any],
    *,
    current_turn: int,
    timing: DeferredIntentTiming,
    default_offset: int,
    default_grace: int,
) -> None:
    new_kind = compact_text(str(data.get("kind", "")))
    if new_kind:
        intent.kind = new_kind
    new_intent = compact_text(str(data.get("intent", "")))
    if new_intent:
        intent.intent = new_intent
    if "why_not_now" in data:
        intent.why_not_now = compact_text(str(data.get("why_not_now", "")))
    if "trigger" in data:
        intent.trigger = coerce_str_list(data.get("trigger"))
    if "cancel_if" in data:
        intent.cancel_if = coerce_str_list(data.get("cancel_if"))
    if "confidence" in data:
        intent.confidence = clamp01(data.get("confidence"), default=intent.confidence)
    if "priority" in data:
        intent.priority = clamp01(data.get("priority"), default=intent.priority)

    earliest_turn = coerce_int(data.get("earliest_turn"))
    latest_turn = coerce_int(data.get("latest_turn"))
    hazard_profile = parse_hazard_profile(
        data.get("hazard_profile")
        or data.get("hazard")
        or (data.get("timing") if isinstance(data.get("timing"), dict) else {})
    )

    if earliest_turn is None:
        earliest_turn = max(current_turn + 1, intent.earliest_turn)
    if latest_turn is None:
        latest_turn = max(earliest_turn, intent.latest_turn, current_turn + 1 + max(0, default_grace))

    if latest_turn < earliest_turn:
        latest_turn = earliest_turn

    intent.earliest_turn = earliest_turn
    intent.latest_turn = latest_turn
    if timing == DeferredIntentTiming.HAZARD:
        intent.hazard_profile = ensure_hazard_profile(
            hazard_profile or intent.hazard_profile,
            created_turn=intent.created_turn,
            earliest_turn=intent.earliest_turn,
            latest_turn=intent.latest_turn,
            offset=default_offset,
            grace=default_grace,
        )
        intent.earliest_turn, intent.latest_turn = hazard_bounds_from_profile(
            created_turn=intent.created_turn,
            profile=intent.hazard_profile,
        )
    elif hazard_profile:
        intent.hazard_profile = hazard_profile
    else:
        intent.hazard_profile = []
    intent.revision_count += 1
    intent.status = "active"
    intent.terminal_reason = ""


# ----------------------------
# Provider abstraction
# ----------------------------

class BaseAdapter(abc.ABC):
    provider_name: str

    def __init__(self, model: str) -> None:
        self.model = model

    @abc.abstractmethod
    def generate(
        self,
        *,
        system: Optional[str],
        messages: list[ChatMessage],
        config: GenerationConfig,
    ) -> ProviderResponse:
        raise NotImplementedError


class BaseEmbeddingAdapter(abc.ABC):
    provider_name: str

    def __init__(self, model: str) -> None:
        self.model = model

    @abc.abstractmethod
    def embed(
        self,
        *,
        inputs: list[str],
        config: GenerationConfig,
    ) -> EmbeddingResponse:
        raise NotImplementedError


class DummyAdapter(BaseAdapter):
    provider_name = "dummy"

    def generate(
        self,
        *,
        system: Optional[str],
        messages: list[ChatMessage],
        config: GenerationConfig,
    ) -> ProviderResponse:
        prompt = messages[-1].content if messages else ""
        lower = prompt.lower()
        system_lower = (system or "").lower()

        if "deferred-intent in-band state" in system_lower and "rcl_state" in system_lower:
            def parse_int(key: str, default: int) -> int:
                match = re.search(
                    rf"{re.escape(key)}\s*:\s*(\d+)", system or "", flags=re.IGNORECASE
                )
                if not match:
                    return default
                try:
                    return int(match.group(1))
                except Exception:
                    return default

            def parse_str(key: str, default: str) -> str:
                match = re.search(
                    rf"{re.escape(key)}\s*:\s*([a-zA-Z0-9_-]+)",
                    system or "",
                    flags=re.IGNORECASE,
                )
                if not match:
                    return default
                return compact_text(match.group(1)).lower() or default

            current_turn = parse_int("Current turn index", 0)
            next_intent_id = parse_str("next_intent_id", "di-0001")
            deferred_intent_every = parse_int("deferred_intent_every", 0)
            deferred_intent_offset = parse_int("deferred_intent_offset", 3)
            deferred_intent_grace = parse_int("deferred_intent_grace", 2)
            deferred_intent_limit = parse_int("deferred_intent_limit", 6)
            deferred_intent_plan_max_new = parse_int("deferred_intent_plan_max_new", 1)
            deferred_intent_mode = parse_str("deferred_intent_mode", DeferredIntentMode.OBSERVE.value)
            deferred_intent_strategy = parse_str("deferred_intent_strategy", DeferredIntentStrategy.TRIGGER.value)
            deferred_intent_timing = parse_str("deferred_intent_timing", DeferredIntentTiming.OFFSET.value)

            prev_state: dict[str, Any] = {"version": 1, "intents": []}
            for msg in reversed(messages):
                if msg.role != "assistant":
                    continue
                _, state, _ = split_rcl_state(msg.content)
                if isinstance(state, dict) and isinstance(state.get("intents"), list):
                    prev_state = state
                    break

            intents: list[dict[str, Any]] = []
            for item in prev_state.get("intents") or []:
                if isinstance(item, dict):
                    intents.append(dict(item))

            last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            existing_ids = {str(i.get("intent_id")) for i in intents if i.get("intent_id")}
            active_intents = [
                item
                for item in intents
                if compact_text(str(item.get("status", "active"))).lower() == "active"
            ]
            plan_capacity = max(0, deferred_intent_limit - len(active_intents))
            plan_capacity = min(plan_capacity, max(0, deferred_intent_plan_max_new))
            eligible_plan_turn = (
                deferred_intent_every > 0
                and current_turn != 0
                and current_turn % deferred_intent_every == 0
                and plan_capacity > 0
            )
            if eligible_plan_turn:
                match = re.fullmatch(r"di-(\\d+)", next_intent_id)
                base_idx = int(match.group(1)) if match else 1
                to_create = min(plan_capacity, 2)
                for idx_offset in range(to_create):
                    intent_id = f"di-{base_idx + idx_offset:04d}"
                    if intent_id in existing_ids:
                        continue
                    hazard_profile: list[dict[str, Any]] = []
                    if deferred_intent_timing == DeferredIntentTiming.HAZARD.value:
                        hazard_profile = default_hazard_profile(
                            offset=deferred_intent_offset,
                            grace=deferred_intent_grace,
                        )
                        earliest_turn, latest_turn = hazard_bounds_from_profile(
                            created_turn=current_turn,
                            profile=hazard_profile,
                        )
                    else:
                        earliest_turn = current_turn + max(1, deferred_intent_offset)
                        if deferred_intent_strategy == DeferredIntentStrategy.FIXED.value:
                            latest_turn = earliest_turn
                        else:
                            latest_turn = earliest_turn + max(0, deferred_intent_grace)

                    kind = "proposal" if idx_offset == 0 else "question"
                    intent_text = (
                        "Offer a concrete next step grounded in the user's latest constraints: "
                        f"{compact_text(last_user)[:120]}"
                        if idx_offset == 0
                        else "Ask which trace storage format to prefer (JSONL vs DB) once the user is ready."
                    )
                    intents.append(
                        {
                            "intent_id": intent_id,
                            "created_turn": current_turn,
                            "kind": kind,
                            "intent": intent_text,
                            "why_not_now": "The dialogue is still collecting constraints and timing cues.",
                            "earliest_turn": earliest_turn,
                            "latest_turn": latest_turn,
                            "hazard_profile": hazard_profile,
                            "trigger": [
                                "the user asks for a concrete next step",
                                "enough constraints have accumulated",
                            ],
                            "cancel_if": [
                                "the topic changes",
                                "the user rejects recommendations",
                            ],
                            "confidence": 0.62,
                            "priority": 0.70 - 0.05 * idx_offset,
                            "plan_strategy": "need_more_constraints",
                            "plan_signals": ["eligible_plan_turn", "conversation_incomplete"],
                            "plan_rationale": "Hold it until the user indicates readiness for concrete next steps.",
                            "status": "active",
                            "revision_count": 0,
                            "fire_turn": None,
                            "terminal_reason": "",
                        }
                    )
                    existing_ids.add(intent_id)

            fired_this_turn: list[str] = []
            for item in intents:
                if not isinstance(item, dict):
                    continue
                status = compact_text(str(item.get("status", "active"))).lower() or "active"
                if status != "active":
                    continue
                created_turn = coerce_int(item.get("created_turn")) or current_turn
                hazard_profile = parse_hazard_profile(item.get("hazard_profile"))
                earliest = coerce_int(item.get("earliest_turn"))
                latest = coerce_int(item.get("latest_turn"))
                if deferred_intent_timing == DeferredIntentTiming.HAZARD.value:
                    hazard_profile = ensure_hazard_profile(
                        hazard_profile,
                        created_turn=created_turn,
                        earliest_turn=earliest,
                        latest_turn=latest,
                        offset=deferred_intent_offset,
                        grace=deferred_intent_grace,
                    )
                    earliest, latest = hazard_bounds_from_profile(
                        created_turn=created_turn,
                        profile=hazard_profile,
                    )
                else:
                    if earliest is None:
                        earliest = created_turn + max(1, deferred_intent_offset)
                    if latest is None:
                        if deferred_intent_strategy == DeferredIntentStrategy.FIXED.value:
                            latest = earliest
                        else:
                            latest = earliest + max(0, deferred_intent_grace)
                    if latest < earliest:
                        latest = earliest
                    hazard_profile = []
                item["hazard_profile"] = hazard_profile

                if current_turn < earliest:
                    continue
                if current_turn > latest:
                    item["status"] = "expired"
                    item["terminal_reason"] = "Dummy: timing window passed."
                    item["decision_strategy"] = "window_expired"
                    item["decision_signals"] = ["past_latest_turn"]
                    item["decision_rationale"] = "The intended timing window has passed."
                    continue

                if deferred_intent_timing == DeferredIntentTiming.HAZARD.value:
                    peak_delay = hazard_peak_delay(hazard_profile)
                    if peak_delay is None:
                        peak_delay = max(0, deferred_intent_offset)
                    peak_turn = created_turn + peak_delay
                    if current_turn < peak_turn:
                        continue

                item["status"] = "fired"
                item["fire_turn"] = current_turn
                item["terminal_reason"] = "Dummy: fired when window opened."
                item["decision_strategy"] = "window_open"
                item["decision_signals"] = ["within_window"]
                item["decision_rationale"] = "The timing window is open and the utterance can be delivered now."
                fired_intent_text = compact_text(str(item.get("intent", "")))
                if fired_intent_text:
                    fired_this_turn.append(fired_intent_text)

            visible_reply = (
                "Dummy reply (in-band). I’ll respond to the latest turn concretely.\n"
                f"Grounding: {compact_text(last_user)[:180]}"
            )
            if deferred_intent_mode in {
                DeferredIntentMode.SOFT_FIRE.value,
                DeferredIntentMode.HARD_FIRE.value,
            } and fired_this_turn:
                fired_block = "\n".join(f"- {text}" for text in fired_this_turn)
                heading = (
                    "Deferred follow-up:"
                    if deferred_intent_mode == DeferredIntentMode.SOFT_FIRE.value
                    else "Deferred intents (explicit):"
                )
                visible_reply = visible_reply.rstrip() + f"\n\n{heading}\n" + fired_block

            state_out = {
                "version": int(coerce_int(prev_state.get("version")) or 1),
                "intents": intents,
            }
            output = visible_reply.rstrip() + f"\n\n{RCL_STATE_OPEN}{json.dumps(state_out, ensure_ascii=False)}{RCL_STATE_CLOSE}"

        elif "compress conversation state into a compact memory capsule" in (system or "").lower() or "existing memory capsules:" in lower and "produce one new memory capsule" in lower:
            recent_lines = [
                m.content for m in messages[-3:] if m.role in {"user", "assistant"}
            ]
            text = " ".join(recent_lines)[-260:]
            output = f"Goal remains active. Recent thread: {compact_text(text)}"
        elif "most likely eventual conclusion" in lower or "infer the likely end-state" in (system or "").lower():
            final_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            mention_hazard_line = ""
            if "mention_hazard_profile" in lower:
                mention_hazard_line = (
                    'MENTION_HAZARD_PROFILE: [{"delay_turns": 2, "prob": 0.20}, '
                    '{"delay_turns": 3, "prob": 0.50}, {"delay_turns": 4, "prob": 0.30}]\n'
                )
            output = textwrap.dedent(
                f"""\
                CONCLUSION: The dialogue is converging toward a concrete experiment design with explicit instrumentation and comparison conditions.
                CONFIDENCE: 0.74
                EVIDENCE: The recent turns emphasize recursive memory, periodic probes, and operational evaluation. Latest user turn: {final_user[:120]}
                KEYWORDS: instrumentation; baseline comparison; deferred mention; metrics; firing window
                MENTION_DELAY_MIN_TURNS: 2
                MENTION_DELAY_MAX_TURNS: 4
                {mention_hazard_line}\
                MENTION_LIKELIHOOD: 0.55
                DELAY_STRATEGY: build_context
                DELAY_SIGNALS: constraints_missing, user_not_ready, need_one_more_turn
                DELAY_RATIONALE: Hold the conclusion until after 1-2 more constraint-gathering turns.
                """
            ).strip()
        elif "latent convergence trace" in system_lower or (
            "latent_alignment" in lower and "articulation_readiness" in lower
        ):
            current_turn_match = re.search(r"current turn index:\s*(\d+)", prompt, flags=re.IGNORECASE)
            current_turn = int(current_turn_match.group(1)) if current_turn_match else 0
            earliest_match = re.search(
                r"planned earliest articulation turn:\s*(\d+)",
                prompt,
                flags=re.IGNORECASE,
            )
            earliest_turn = int(earliest_match.group(1)) if earliest_match else current_turn + 1
            latest_reply = ""
            reply_match = re.search(
                r"latest assistant reply:\s*(.+?)(?:\n\s*\n|\Z)",
                prompt,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if reply_match:
                latest_reply = compact_text(reply_match.group(1))
            explicit = "converging toward a concrete experiment design" in latest_reply.lower()
            alignment = min(0.26 + 0.12 * max(0, current_turn), 0.92)
            readiness = min(0.14 + 0.11 * max(0, current_turn), 0.90)
            leakage_risk = 0.68 if current_turn < earliest_turn else 0.24
            if explicit:
                alignment = max(alignment, 0.84)
                readiness = max(readiness, 0.95)
                leakage_risk = 0.05
            if explicit:
                stage = "explicit"
                signals = "explicit_surface,window_open"
            elif readiness >= 0.72:
                stage = "ready"
                signals = "window_open,articulation_ready"
            elif alignment >= 0.52:
                stage = "approaching"
                signals = "latent_bias,trajectory_converging"
            else:
                stage = "latent"
                signals = "holding_pattern,trajectory_forming"
            output = textwrap.dedent(
                f"""\
                LATENT_ALIGNMENT: {alignment:.2f}
                ARTICULATION_READINESS: {readiness:.2f}
                LEAKAGE_RISK: {leakage_risk:.2f}
                EXPLICIT_MENTION_PRESENT: {"yes" if explicit else "no"}
                TRAJECTORY_STAGE: {stage}
                SHIFT_SIGNALS: {signals}
                EVIDENCE: Dummy latent trace at turn {current_turn}; trajectory is being tracked without forcing explicit articulation.
                """
            ).strip()
        elif "delayed mention planner" in system_lower:
            last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            payload = {
                "items": [
                    {
                        "kind": "constraint",
                        "text": "Confirm whether we should optimize for portability (single-file) or extensibility (multi-module) before adding more components.",
                        "keywords": ["portability", "single-file", "extensibility"],
                        "mention_delay_min_turns": 1,
                        "mention_delay_max_turns": 3,
                        "mention_likelihood": 0.60,
                        "delay_strategy": "gather_constraints",
                        "delay_signals": ["needs_choice", "design_branching"],
                        "delay_rationale": f"Ask after a bit more context. Latest user: {compact_text(last_user)[:60]}",
                    },
                    {
                        "kind": "option",
                        "text": "Consider adding a probabilistic soft-fire nudge inside the predicted mention window to increase mention reliability without forcing exact wording.",
                        "keywords": ["soft-fire", "probabilistic", "mention window"],
                        "mention_delay_min_turns": 2,
                        "mention_delay_max_turns": 4,
                        "mention_likelihood": 0.52,
                        "delay_strategy": "build_context",
                        "delay_signals": ["timing_sensitive", "avoid_oversteer"],
                        "delay_rationale": "Mention it once the experiment metrics are in place.",
                    },
                ]
            }
            if "mention_hazard_profile" in lower:
                payload["items"][0]["mention_hazard_profile"] = [
                    {"delay_turns": 1, "prob": 0.45},
                    {"delay_turns": 2, "prob": 0.35},
                    {"delay_turns": 3, "prob": 0.20},
                ]
                payload["items"][1]["mention_hazard_profile"] = [
                    {"delay_turns": 2, "prob": 0.20},
                    {"delay_turns": 3, "prob": 0.55},
                    {"delay_turns": 4, "prob": 0.25},
                ]
            output = json.dumps(payload, ensure_ascii=False)
        elif "deferred utterance planner" in (system or "").lower() or ("create_intent" in lower and "why_not_now" in lower and "deferred utterance" in lower):
            latest_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            timing = None
            if "hazard_profile" in lower:
                timing = {
                    "hazard_profile": [
                        {"delay_turns": 2, "prob": 0.20},
                        {"delay_turns": 3, "prob": 0.50},
                        {"delay_turns": 4, "prob": 0.30},
                    ]
                }
            elif "delay_min_turns" in lower or "delay_max_turns" in lower:
                timing = {"delay_min_turns": 2, "delay_max_turns": 4}
            payload: dict[str, Any] = {
                "kind": "proposal",
                "intent": f"After a few more turns, offer a concise operational recommendation grounded in: {compact_text(latest_user)[:100]}",
                "why_not_now": "The conversation is still gathering constraints and timing cues.",
                "selection": {
                    "strategy": "need_more_constraints",
                    "signals": ["constraints_missing", "too_early"],
                    "rationale": "Hold a concrete recommendation until the user provides enough constraints.",
                },
                "trigger": [
                    "the user asks for a summary or concrete next step",
                    "enough constraints have accumulated",
                ],
                "cancel_if": [
                    "the topic changes",
                    "the user explicitly rejects recommendations",
                ],
                "confidence": 0.66,
                "priority": 0.72,
            }
            if timing is not None:
                payload["timing"] = timing
            output = json.dumps({"intents": [payload]}, ensure_ascii=False)
        elif "deferred utterance scheduler" in (system or "").lower() or "decisions" in lower and "intent_id" in lower and "action" in lower:
            current_turn_match = re.search(r"current turn index:\s*(\d+)", prompt, flags=re.IGNORECASE)
            current_turn = int(current_turn_match.group(1)) if current_turn_match else 0
            intents_value = extract_json_value(prompt.split("Deferred intents:", 1)[-1]) if "Deferred intents:" in prompt else None
            items = []
            if isinstance(intents_value, list):
                items = intents_value
            elif isinstance(intents_value, dict):
                items = intents_value.get("intents") or []
            decisions = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                created_turn = coerce_int(item.get("created_turn")) or current_turn
                hazard_profile = parse_hazard_profile(item.get("hazard_profile"))
                earliest = coerce_int(item.get("earliest_turn"))
                latest = coerce_int(item.get("latest_turn"))
                if hazard_profile:
                    earliest, latest = hazard_bounds_from_profile(
                        created_turn=created_turn,
                        profile=hazard_profile,
                    )
                else:
                    earliest = earliest or current_turn + 1
                    latest = latest or earliest
                peak_delay = hazard_peak_delay(hazard_profile) if hazard_profile else None
                peak_turn = created_turn + peak_delay if peak_delay is not None else None
                if current_turn < earliest:
                    action = "hold"
                    reason = "Too early. Keep the utterance in reserve."
                    decision_strategy = "too_early"
                    decision_signals = ["before_earliest_turn"]
                elif peak_turn is not None and current_turn < peak_turn:
                    action = "hold"
                    reason = "Hazard mass is still building; keep the utterance latent."
                    decision_strategy = "hazard_wait"
                    decision_signals = ["before_hazard_peak"]
                elif current_turn > latest:
                    action = "expire"
                    reason = "The intended timing window has passed."
                    decision_strategy = "window_expired"
                    decision_signals = ["past_latest_turn"]
                else:
                    action = "fire"
                    reason = "The timing window is open and the utterance can be delivered now."
                    decision_strategy = "window_open"
                    decision_signals = ["within_window"]
                decisions.append(
                    {
                        "intent_id": item.get("intent_id", "unknown"),
                        "action": action,
                        "reason": reason,
                        "decision_strategy": decision_strategy,
                        "decision_signals": decision_signals,
                        "decision_rationale": reason,
                    }
                )
            output = json.dumps({"decisions": decisions}, ensure_ascii=False)
        else:
            last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            output = (
                "Dummy reply. I would answer the latest user turn concretely, "
                f"grounding on: {last_user[:180]}"
            )
            due_deferred = []
            if system_lower and "deferred utterances are due now" in system_lower:
                for line in (system or "").splitlines():
                    match = re.match(r"\s*-\s*\[(di-[^\]]+)\]\s*(.+)\s*$", line)
                    if match:
                        due_deferred.append(match.group(2).strip())
            if due_deferred:
                block = "\n".join(f"- {text}" for text in due_deferred)
                output = output.rstrip() + "\n\nDeferred intents (simulated):\n" + block
            delayed = []
            if system_lower and "delayed mention targets are due now" in system_lower:
                for line in (system or "").splitlines():
                    match = re.match(r"\s*-\s*\[(dm-[^\]]+)\]\s*(.+)\s*$", line)
                    if match:
                        delayed.append(match.group(2).strip())
            if delayed:
                output = output.rstrip() + "\n\nDelayed mentions (simulated):\n" + "\n".join(
                    f"- {text}" for text in delayed
                )

        return ProviderResponse(
            provider=self.provider_name,
            model=self.model,
            text=output,
            raw={"dummy": True, "text": output},
            usage={"input_tokens": 0, "output_tokens": 0},
            finish_reason="stop",
            request_id=f"dummy-{random.randint(1000, 9999)}",
        )


class DummyEmbeddingAdapter(BaseEmbeddingAdapter):
    provider_name = "dummy"

    def embed(
        self,
        *,
        inputs: list[str],
        config: GenerationConfig,
    ) -> EmbeddingResponse:
        embeddings = [hashed_embedding_vector(text) for text in inputs]
        return EmbeddingResponse(
            provider=self.provider_name,
            model=self.model,
            embeddings=embeddings,
            raw={"dummy": True, "input_count": len(inputs)},
            usage={"input_tokens": 0},
            request_id=f"dummy-embed-{random.randint(1000, 9999)}",
        )


class OpenAIResponsesAdapter(BaseAdapter):
    provider_name = "openai"
    url = "https://api.openai.com/v1/responses"

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        super().__init__(model=model)
        self.api_key = api_key or require_env("OPENAI_API_KEY")

    def generate(
        self,
        *,
        system: Optional[str],
        messages: list[ChatMessage],
        config: GenerationConfig,
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": m.role, "content": m.content} for m in messages],
            "max_output_tokens": config.max_tokens,
            "temperature": config.temperature,
            "store": False,
        }
        if system:
            payload["instructions"] = system

        data = post_json(
            url=self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_seconds=config.timeout_seconds,
        )
        text_parts: list[str] = []
        for item in data.get("output", []) or []:
            if item.get("type") != "message":
                continue
            for block in item.get("content", []) or []:
                if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
        text = "\n".join(text_parts).strip()

        finish_reason = None
        try:
            finish_reason = data.get("status")
        except Exception:
            pass

        request_id = data.get("id")
        usage = data.get("usage")
        return ProviderResponse(
            provider=self.provider_name,
            model=self.model,
            text=text,
            raw=data,
            usage=usage,
            finish_reason=finish_reason,
            request_id=request_id,
        )


class OpenAIEmbeddingsAdapter(BaseEmbeddingAdapter):
    provider_name = "openai"
    url = "https://api.openai.com/v1/embeddings"

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        super().__init__(model=model)
        self.api_key = api_key or require_env("OPENAI_API_KEY")

    def embed(
        self,
        *,
        inputs: list[str],
        config: GenerationConfig,
    ) -> EmbeddingResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": inputs,
            "encoding_format": "float",
        }
        data = post_json(
            url=self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_seconds=config.timeout_seconds,
        )
        embeddings: list[list[float]] = []
        for item in data.get("data") or []:
            if not isinstance(item, dict):
                continue
            raw_embedding = item.get("embedding")
            if not isinstance(raw_embedding, list):
                continue
            vector = [
                float(x)
                for x in raw_embedding
                if isinstance(x, (int, float))
            ]
            if vector:
                embeddings.append(vector)
        if len(embeddings) != len(inputs):
            raise RuntimeError(
                f"Embedding response length mismatch: expected {len(inputs)}, got {len(embeddings)}."
            )
        return EmbeddingResponse(
            provider=self.provider_name,
            model=self.model,
            embeddings=embeddings,
            raw=data,
            usage=data.get("usage"),
            request_id=data.get("id"),
        )


class AnthropicMessagesAdapter(BaseAdapter):
    provider_name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_version: str = "2023-06-01",
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key or require_env("ANTHROPIC_API_KEY")
        self.api_version = api_version

    def generate(
        self,
        *,
        system: Optional[str],
        messages: list[ChatMessage],
        config: GenerationConfig,
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": config.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            payload["system"] = system
        payload["temperature"] = config.temperature

        data = post_json(
            url=self.url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.api_version,
                "content-type": "application/json",
            },
            payload=payload,
            timeout_seconds=config.timeout_seconds,
        )
        text_parts = []
        for block in data.get("content", []) or []:
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        text = "\n".join(text_parts).strip()
        return ProviderResponse(
            provider=self.provider_name,
            model=self.model,
            text=text,
            raw=data,
            usage=data.get("usage"),
            finish_reason=data.get("stop_reason"),
            request_id=data.get("id"),
        )


class MistralChatAdapter(BaseAdapter):
    provider_name = "mistral"
    url = "https://api.mistral.ai/v1/chat/completions"

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        super().__init__(model=model)
        self.api_key = api_key or require_env("MISTRAL_API_KEY")

    def generate(
        self,
        *,
        system: Optional[str],
        messages: list[ChatMessage],
        config: GenerationConfig,
    ) -> ProviderResponse:
        payload_messages: list[dict[str, str]] = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend({"role": m.role, "content": m.content} for m in messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": False,
        }
        data = post_json(
            url=self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_seconds=config.timeout_seconds,
        )
        choice0 = ((data.get("choices") or [{}])[0]) or {}
        message = choice0.get("message") or {}
        text = coerce_text(message.get("content"))
        return ProviderResponse(
            provider=self.provider_name,
            model=self.model,
            text=text.strip(),
            raw=data,
            usage=data.get("usage"),
            finish_reason=choice0.get("finish_reason"),
            request_id=data.get("id"),
        )


class GeminiAdapter(BaseAdapter):
    provider_name = "gemini"
    base = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        super().__init__(model=model)
        self.api_key = api_key or require_env("GEMINI_API_KEY")

    @staticmethod
    def _gemini_role(role: str) -> str:
        if role == "assistant":
            return "model"
        if role == "user":
            return "user"
        raise ValueError(role)

    def generate(
        self,
        *,
        system: Optional[str],
        messages: list[ChatMessage],
        config: GenerationConfig,
    ) -> ProviderResponse:
        contents = []
        for m in messages:
            contents.append(
                {
                    "role": self._gemini_role(m.role),
                    "parts": [{"text": m.content}],
                }
            )

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_tokens,
            },
        }
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}

        data = post_json(
            url=f"{self.base}/{self.model}:generateContent",
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_seconds=config.timeout_seconds,
        )

        text_parts: list[str] = []
        candidates = data.get("candidates") or []
        if candidates:
            content = (candidates[0] or {}).get("content") or {}
            for part in content.get("parts", []) or []:
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
        text = "\n".join(text_parts).strip()

        usage = data.get("usageMetadata")
        finish_reason = None
        if candidates:
            finish_reason = candidates[0].get("finishReason")

        return ProviderResponse(
            provider=self.provider_name,
            model=self.model,
            text=text,
            raw=data,
            usage=usage,
            finish_reason=finish_reason,
            request_id=None,
        )


class HuggingFaceRouterAdapter(BaseAdapter):
    provider_name = "hf"
    url = "https://router.huggingface.co/v1/chat/completions"

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        super().__init__(model=model)
        self.api_key = api_key or require_env("HF_TOKEN")

    def generate(
        self,
        *,
        system: Optional[str],
        messages: list[ChatMessage],
        config: GenerationConfig,
    ) -> ProviderResponse:
        payload_messages: list[dict[str, str]] = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend({"role": m.role, "content": m.content} for m in messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": False,
        }
        data = post_json(
            url=self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_seconds=config.timeout_seconds,
        )
        choice0 = ((data.get("choices") or [{}])[0]) or {}
        message = choice0.get("message") or {}
        text = coerce_text(message.get("content"))
        return ProviderResponse(
            provider=self.provider_name,
            model=self.model,
            text=text.strip(),
            raw=data,
            usage=data.get("usage"),
            finish_reason=choice0.get("finish_reason"),
            request_id=data.get("id"),
        )


def build_adapter(provider: str, model: str) -> BaseAdapter:
    provider = provider.strip().lower()
    if provider == "openai":
        return OpenAIResponsesAdapter(model=model)
    if provider == "anthropic":
        return AnthropicMessagesAdapter(model=model)
    if provider == "mistral":
        return MistralChatAdapter(model=model)
    if provider == "gemini":
        return GeminiAdapter(model=model)
    if provider in {"hf", "huggingface", "hugging_face"}:
        return HuggingFaceRouterAdapter(model=model)
    if provider == "dummy":
        return DummyAdapter(model=model)
    raise ValueError(f"Unsupported provider: {provider!r}")


def build_embedding_adapter(provider: str, model: str) -> BaseEmbeddingAdapter:
    provider = provider.strip().lower()
    if provider == "openai":
        return OpenAIEmbeddingsAdapter(model=model)
    if provider == "dummy":
        return DummyEmbeddingAdapter(model=model)
    raise ValueError(
        f"Unsupported embedding provider: {provider!r}. "
        "Supported embedding providers are: openai, dummy."
    )


def build_optional_observer_adapter(args: argparse.Namespace) -> Optional[BaseAdapter]:
    observer_provider = compact_text(str(getattr(args, "observer_provider", "") or "")).lower()
    observer_model = compact_text(str(getattr(args, "observer_model", "") or ""))
    if bool(observer_provider) != bool(observer_model):
        raise ValueError(
            "--observer-provider and --observer-model must be set together."
        )
    if not observer_provider:
        return None
    return build_adapter(observer_provider, observer_model)


def build_optional_embedding_adapter(
    args: argparse.Namespace,
) -> Optional[BaseEmbeddingAdapter]:
    embedding_provider = compact_text(
        str(getattr(args, "embedding_provider", "") or "")
    ).lower()
    embedding_model = compact_text(str(getattr(args, "embedding_model", "") or ""))
    if bool(embedding_provider) != bool(embedding_model):
        raise ValueError(
            "--embedding-provider and --embedding-model must be set together."
        )
    if not embedding_provider:
        return None
    return build_embedding_adapter(embedding_provider, embedding_model)


# ----------------------------
# Experiment engine
# ----------------------------

class RecursiveConclusionSession:
    """
    Conversation engine that keeps:
    - full history locally (for logging / evaluation)
    - only recent window + recursive memory capsules in the live prompt
    - optional provisional conclusion injection
    - optional deferred utterance intents planned ahead of time
    """

    DEFERRED_REALIZATION_THRESHOLD = 0.10
    CONCLUSION_MENTION_THRESHOLD = 0.10
    CONCLUSION_KEYWORD_MENTION_MIN_HITS = 2

    def __init__(
        self,
        *,
        adapter: BaseAdapter,
        observer_adapter: Optional[BaseAdapter] = None,
        embedding_adapter: Optional[BaseEmbeddingAdapter] = None,
        config: ExperimentConfig,
        log_path: Optional[Path] = None,
    ) -> None:
        self.adapter = adapter
        self.observer_adapter = observer_adapter or adapter
        self.embedding_adapter = embedding_adapter
        self.latent_judge_source = (
            "independent_observer" if observer_adapter is not None else "generator_adapter"
        )
        self.config = config
        self.log_path = log_path
        if self.config.semantic_judge_backend in {
            SemanticJudgeBackend.EMBEDDING,
            SemanticJudgeBackend.BOTH,
        } and self.embedding_adapter is None:
            raise ValueError(
                "semantic_judge_backend requires --embedding-provider and --embedding-model."
            )
        self.history: list[ChatMessage] = []
        self.memory_capsules: list[str] = []
        self.conclusion_hypotheses: list[str] = []
        self.latest_conclusion_probe_turn: Optional[int] = None
        self.latest_conclusion_line: str = ""
        self.latest_conclusion_keywords: list[str] = []
        self.latest_conclusion_mention_delay_min_turns: Optional[int] = None
        self.latest_conclusion_mention_delay_max_turns: Optional[int] = None
        self.latest_conclusion_mention_hazard_profile: list[dict[str, Any]] = []
        self.latest_conclusion_mention_likelihood: Optional[float] = None
        self.latest_conclusion_delay_strategy: str = ""
        self.latest_conclusion_delay_signals: list[str] = []
        self.latest_conclusion_delay_rationale: str = ""
        self.latest_latent_convergence_trace: Optional[dict[str, Any]] = None
        self.latest_embedding_convergence_trace: Optional[dict[str, Any]] = None
        self.latest_adaptive_hazard_trace: Optional[dict[str, Any]] = None
        self.delayed_mentions: list[DelayedMentionItem] = []
        self.deferred_intents: list[DeferredIntent] = []
        self.turn_index = 0
        self.next_deferred_intent_index = 1
        self.next_delayed_mention_index = 1
        self.deferred_intent_plan_probe_calls = 0
        self._deferred_intent_plan_compact_ok = False
        self._deferred_intent_scheduler_compact_ok = False

    def _recent_window(self) -> list[ChatMessage]:
        if self.config.recent_window_messages <= 0:
            return list(self.history)
        return self.history[-self.config.recent_window_messages :]

    def _active_deferred_intents(self) -> list[DeferredIntent]:
        return [intent for intent in self.deferred_intents if intent.status == "active"]

    def _active_delayed_mentions(self) -> list[DelayedMentionItem]:
        return [item for item in self.delayed_mentions if item.status == "active"]

    def _adaptive_hazard_enabled(self) -> bool:
        return self.config.adaptive_hazard_policy == AdaptiveHazardPolicy.ADAPTIVE

    def _adaptive_hazard_signal_snapshot(self) -> dict[str, Any]:
        current_conclusion_text = strip_conclusion_prefix(self.latest_conclusion_line)

        def trace_matches_current_conclusion(trace: dict[str, Any]) -> bool:
            if trace.get("conclusion_probe_turn") == self.latest_conclusion_probe_turn:
                return True
            trace_conclusion_text = compact_text(str(trace.get("conclusion_line") or ""))
            return bool(current_conclusion_text) and (
                compact_text(current_conclusion_text) == trace_conclusion_text
            )

        latent = {}
        if (
            isinstance(self.latest_latent_convergence_trace, dict)
            and trace_matches_current_conclusion(self.latest_latent_convergence_trace)
        ):
            latent = self.latest_latent_convergence_trace
        embedding = {}
        if (
            isinstance(self.latest_embedding_convergence_trace, dict)
            and trace_matches_current_conclusion(self.latest_embedding_convergence_trace)
        ):
            embedding = self.latest_embedding_convergence_trace
        latent_alignment = latent.get("alignment")
        embedding_alignment = embedding.get("alignment")
        alignments = [
            float(value)
            for value in (latent_alignment, embedding_alignment)
            if isinstance(value, (int, float))
        ]
        alignment = sum(alignments) / len(alignments) if alignments else None
        readiness = (
            float(latent.get("articulation_readiness"))
            if isinstance(latent.get("articulation_readiness"), (int, float))
            else None
        )
        leakage_risk = (
            float(latent.get("leakage_risk"))
            if isinstance(latent.get("leakage_risk"), (int, float))
            else None
        )
        judge_gap = None
        if isinstance(latent_alignment, (int, float)) and isinstance(
            embedding_alignment, (int, float)
        ):
            judge_gap = abs(float(latent_alignment) - float(embedding_alignment))
        stage = truncate_text(latent.get("trajectory_stage") or "", 24) or None
        explicit = latent.get("explicit_mention_present")
        if not isinstance(explicit, bool):
            explicit = None
        likelihood = (
            float(self.latest_conclusion_mention_likelihood)
            if isinstance(self.latest_conclusion_mention_likelihood, (int, float))
            else None
        )
        return {
            "policy": self.config.adaptive_hazard_policy.value,
            "profile": self.config.adaptive_hazard_profile.value,
            "latent_alignment": (
                float(latent_alignment)
                if isinstance(latent_alignment, (int, float))
                else None
            ),
            "embedding_alignment": (
                float(embedding_alignment)
                if isinstance(embedding_alignment, (int, float))
                else None
            ),
            "alignment": alignment,
            "readiness": readiness,
            "leakage_risk": leakage_risk,
            "judge_gap": judge_gap,
            "trajectory_stage": stage,
            "explicit_mention_present": explicit,
            "conclusion_likelihood": likelihood,
        }

    def _compute_adaptive_hazard_adjustment(
        self,
        *,
        item_kind: str,
        item_id: str,
        item_text: str,
        item_delay_strategy: str,
        item_delay_signals: list[str],
        item_release_stage_role: str,
        created_turn: int,
        hazard_profile: list[dict[str, Any]],
        base_turn_prob: float,
        base_threshold: Optional[float],
        likelihood: Optional[float],
        signal_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        peak_prob = hazard_peak_probability(hazard_profile) if hazard_profile else 0.0
        peak_delay = hazard_peak_delay(hazard_profile) if hazard_profile else None
        peak_turn = created_turn + peak_delay if peak_delay is not None else None
        peak_support_ratio: Optional[float] = None
        if peak_prob > 0.0:
            peak_support_ratio = clamp01(base_turn_prob / peak_prob, default=0.0)
        before_peak = peak_turn is not None and self.turn_index < peak_turn
        at_peak = peak_turn is not None and self.turn_index == peak_turn
        after_peak = peak_turn is not None and self.turn_index > peak_turn
        release_stage_role = (
            compact_text(item_release_stage_role) or ""
        ) or classify_delayed_mention_stage_role(
            kind=item_kind,
            text=item_text,
            delay_strategy=item_delay_strategy,
            delay_signals=item_delay_signals,
        )
        adjustment = {
            "item_id": item_id,
            "kind": item_kind,
            "text": truncate_text(item_text, 140),
            "release_stage_role": release_stage_role,
            "adaptive_hazard_stage_policy": self.config.adaptive_hazard_stage_policy.value,
            "created_turn": created_turn,
            "peak_turn": peak_turn,
            "peak_probability": peak_prob if peak_prob > 0.0 else None,
            "peak_support_ratio": peak_support_ratio,
            "before_peak": before_peak if peak_turn is not None else None,
            "at_peak": at_peak if peak_turn is not None else None,
            "after_peak": after_peak if peak_turn is not None else None,
            "base_turn_prob": float(base_turn_prob),
            "effective_turn_prob": float(base_turn_prob),
            "turn_prob_multiplier": 1.0,
            "base_threshold": base_threshold,
            "effective_threshold": base_threshold,
            "threshold_shift": 0.0 if base_threshold is not None else None,
            "reasons": [],
            "signals": signal_snapshot,
        }
        if not self._adaptive_hazard_enabled():
            return adjustment

        params = adaptive_hazard_profile_params(self.config.adaptive_hazard_profile)
        stage_role_params = adaptive_hazard_stage_role_params(
            self.config.adaptive_hazard_stage_policy,
            release_stage_role,
        )
        focus_weight = (1.0 if item_kind == "conclusion" else 0.65) * float(
            stage_role_params["focus_scale"]
        )
        alignment = signal_snapshot.get("alignment")
        embedding_alignment = signal_snapshot.get("embedding_alignment")
        readiness = signal_snapshot.get("readiness")
        leakage_risk = signal_snapshot.get("leakage_risk")
        judge_gap = signal_snapshot.get("judge_gap")
        stage = signal_snapshot.get("trajectory_stage")
        explicit = signal_snapshot.get("explicit_mention_present")
        effective_likelihood = (
            float(likelihood)
            if isinstance(likelihood, (int, float))
            else signal_snapshot.get("conclusion_likelihood")
        )

        boost_signal = 0.0
        if isinstance(alignment, (int, float)):
            boost_signal += max(0.0, float(alignment) - 0.55)
        if isinstance(readiness, (int, float)):
            boost_signal += max(0.0, float(readiness) - 0.50)
        if isinstance(effective_likelihood, (int, float)):
            boost_signal += 0.5 * max(0.0, float(effective_likelihood) - 0.50)
        if stage == "approaching":
            boost_signal += 0.05
        elif stage == "ready":
            boost_signal += 0.12
        elif stage == "explicit":
            boost_signal += 0.08
        boost_signal = max(0.0, boost_signal + float(stage_role_params["boost_bias"]))

        suppress_signal = (
            float(leakage_risk) if isinstance(leakage_risk, (int, float)) else 0.0
        )
        if stage == "latent":
            suppress_signal = max(suppress_signal, 0.18)
        elif stage == "diverged":
            suppress_signal = max(suppress_signal, 0.85)
        suppress_signal = max(
            0.0,
            suppress_signal + float(stage_role_params["suppress_bias"]),
        )

        peak_pull = 0.0
        peak_release = 0.0
        if isinstance(peak_support_ratio, (int, float)):
            peak_ratio = float(peak_support_ratio)
            peak_pull += max(0.0, params["peak_target_ratio"] - peak_ratio)
            peak_release += max(0.0, peak_ratio - params["peak_release_ratio"])
        if before_peak:
            peak_pull += params["peak_pre_peak_penalty"]
        elif at_peak:
            peak_release += params["peak_release_bonus"]
        elif after_peak and isinstance(peak_support_ratio, (int, float)):
            if float(peak_support_ratio) >= params["peak_release_ratio"]:
                peak_release += 0.5 * params["peak_release_bonus"]
        if before_peak and isinstance(peak_support_ratio, (int, float)):
            if (
                float(peak_support_ratio)
                >= float(stage_role_params["prepeak_release_ratio"])
                and stage in {"approaching", "ready", "explicit"}
            ):
                peak_release += float(stage_role_params["prepeak_release_bonus"])
            peak_pull += float(stage_role_params["prepeak_hold_bonus"])
        peak_pull *= float(stage_role_params["peak_pull_scale"])
        peak_release *= float(stage_role_params["peak_release_scale"])
        adjustment["stage_boost_bias"] = float(stage_role_params["boost_bias"])
        adjustment["stage_suppress_bias"] = float(stage_role_params["suppress_bias"])
        adjustment["stage_threshold_bias"] = float(stage_role_params["threshold_bias"])
        adjustment["stage_prepeak_release_bonus"] = float(
            stage_role_params["prepeak_release_bonus"]
        )
        adjustment["stage_prepeak_hold_bonus"] = float(
            stage_role_params["prepeak_hold_bonus"]
        )

        embedding_prepeak_penalty = 0.0
        embedding_prepeak_peak_gap_factor = None
        if (
            self.config.adaptive_hazard_embedding_guard == AdaptiveHazardEmbeddingGuard.ON
            and before_peak
            and not explicit
            and isinstance(embedding_alignment, (int, float))
        ):
            embedding_target = float(params["embedding_prepeak_target"])
            denom = max(0.05, 1.0 - embedding_target)
            embedding_prepeak_penalty = min(
                1.0,
                max(0.0, float(embedding_alignment) - embedding_target) / denom,
            )
            if isinstance(peak_support_ratio, (int, float)):
                peak_gap_factor = max(
                    0.0,
                    float(params["peak_target_ratio"]) - float(peak_support_ratio),
                ) / max(0.05, float(params["peak_target_ratio"]))
                embedding_prepeak_peak_gap_factor = clamp01(peak_gap_factor, default=0.0)
                embedding_prepeak_penalty *= embedding_prepeak_peak_gap_factor
            if stage in {"latent", "approaching"} and embedding_prepeak_penalty > 0.0:
                embedding_prepeak_penalty = min(
                    1.0,
                    embedding_prepeak_penalty + 0.12,
                )
            elif stage == "ready":
                embedding_prepeak_penalty *= 0.45
        adjustment["embedding_prepeak_penalty"] = embedding_prepeak_penalty
        adjustment["embedding_prepeak_peak_gap_factor"] = embedding_prepeak_peak_gap_factor

        confidence = 1.0
        if isinstance(judge_gap, (int, float)):
            confidence = max(
                0.35,
                1.0 - min(1.0, float(judge_gap) * params["gap_damping"]),
            )

        delta = (
            params["boost_scale"] * boost_signal * focus_weight * confidence
            + params["peak_release_scale"] * peak_release * focus_weight * confidence
            - params["suppress_scale"] * suppress_signal * focus_weight
            - params["peak_pull_scale"] * peak_pull * focus_weight
            - params["embedding_prepeak_scale"] * embedding_prepeak_penalty * focus_weight
        )
        multiplier = min(
            params["max_multiplier"],
            max(params["min_multiplier"], 1.0 + delta),
        )
        if base_turn_prob <= 0.0:
            multiplier = 1.0

        effective_turn_prob = clamp01(base_turn_prob * multiplier, default=0.0)
        adjustment["effective_turn_prob"] = effective_turn_prob
        adjustment["turn_prob_multiplier"] = multiplier

        if base_threshold is not None:
            threshold_shift = (
                params["threshold_raise"] * suppress_signal * focus_weight
                + params["threshold_peak_raise"] * peak_pull * focus_weight
                + params["threshold_embedding_raise"] * embedding_prepeak_penalty * focus_weight
                - params["threshold_peak_lower"] * peak_release * focus_weight * confidence
                + float(stage_role_params["threshold_bias"])
            )
            if explicit and not before_peak:
                threshold_shift -= 0.02 * focus_weight
            effective_threshold = clamp01(base_threshold + threshold_shift, default=base_threshold)
            adjustment["effective_threshold"] = effective_threshold
            adjustment["threshold_shift"] = effective_threshold - base_threshold

        reasons: list[str] = []
        if boost_signal >= 0.08:
            reasons.append("boost:trajectory_ready")
        elif boost_signal >= 0.03:
            reasons.append("boost:trajectory_bias")
        if suppress_signal >= 0.40:
            reasons.append("suppress:leakage_risk")
        elif suppress_signal >= 0.18:
            reasons.append("suppress:latent_hold")
        if peak_pull >= 0.08:
            reasons.append("pull:toward_peak")
        if before_peak:
            reasons.append("hold:pre_peak")
        elif peak_release >= 0.08:
            reasons.append("release:peak_support")
        if (
            before_peak
            and float(stage_role_params["prepeak_release_bonus"]) > 0.0
            and isinstance(peak_support_ratio, (int, float))
            and float(peak_support_ratio) >= float(stage_role_params["prepeak_release_ratio"])
            and stage in {"approaching", "ready", "explicit"}
        ):
            reasons.append("release:stage_prepeak")
        if float(stage_role_params["prepeak_hold_bonus"]) > 0.0 and before_peak:
            reasons.append("hold:stage_packet")
        if embedding_prepeak_penalty >= 0.08:
            reasons.append("suppress:embedding_prepeak")
        if release_stage_role:
            reasons.append(f"release_stage:{release_stage_role}")
        if stage:
            reasons.append(f"stage:{stage}")
        if isinstance(judge_gap, (int, float)) and float(judge_gap) >= 0.08:
            reasons.append("dampen:judge_gap")
        if explicit:
            reasons.append("release:explicit_seen")
        adjustment["reasons"] = reasons
        return adjustment

    def _build_adaptive_hazard_trace(
        self,
    ) -> tuple[dict[str, dict[str, Any]], Optional[dict[str, Any]]]:
        signal_snapshot = self._adaptive_hazard_signal_snapshot()
        base_threshold = (
            self._delayed_mention_leak_threshold()
            if self.config.delayed_mention_leak_policy == DelayedMentionLeakPolicy.ON
            else None
        )
        adjustments: dict[str, dict[str, Any]] = {}
        for item in self._active_delayed_mentions():
            base_turn_prob = hazard_turn_probability(
                item.hazard_profile,
                created_turn=item.created_turn,
                turn_index=self.turn_index,
            )
            adjustments[item.item_id] = self._compute_adaptive_hazard_adjustment(
                item_kind=item.kind,
                item_id=item.item_id,
                item_text=item.text,
                item_delay_strategy=item.delay_strategy,
                item_delay_signals=list(item.delay_signals),
                item_release_stage_role=item.release_stage_role,
                created_turn=item.created_turn,
                hazard_profile=item.hazard_profile,
                base_turn_prob=base_turn_prob,
                base_threshold=base_threshold,
                likelihood=item.likelihood,
                signal_snapshot=signal_snapshot,
            )

        conclusion_adjustment = None
        if self.latest_conclusion_probe_turn is not None and self.latest_conclusion_mention_hazard_profile:
            base_turn_prob = hazard_turn_probability(
                self.latest_conclusion_mention_hazard_profile,
                created_turn=self.latest_conclusion_probe_turn,
                turn_index=self.turn_index,
            )
            conclusion_adjustment = self._compute_adaptive_hazard_adjustment(
                item_kind="conclusion",
                item_id="conclusion-plan",
                item_text=strip_conclusion_prefix(self.latest_conclusion_line),
                item_delay_strategy=self.latest_conclusion_delay_strategy,
                item_delay_signals=list(self.latest_conclusion_delay_signals),
                item_release_stage_role="conclusion",
                created_turn=self.latest_conclusion_probe_turn,
                hazard_profile=self.latest_conclusion_mention_hazard_profile,
                base_turn_prob=base_turn_prob,
                base_threshold=base_threshold,
                likelihood=self.latest_conclusion_mention_likelihood,
                signal_snapshot=signal_snapshot,
            )
            conclusion_adjustment["probe_turn"] = self.latest_conclusion_probe_turn
            conclusion_adjustment["hazard_profile"] = list(
                self.latest_conclusion_mention_hazard_profile
            )

        trace = {
            "policy": self.config.adaptive_hazard_policy.value,
            "profile": self.config.adaptive_hazard_profile.value,
            "stage_policy": self.config.adaptive_hazard_stage_policy.value,
            "signals": signal_snapshot,
            "items": list(adjustments.values()),
            "latest_conclusion": conclusion_adjustment,
        }
        self.latest_adaptive_hazard_trace = trace
        self._log("adaptive_hazard_trace", trace)
        return adjustments, conclusion_adjustment

    def _delayed_mention_adjustment(
        self,
        item: DelayedMentionItem,
        adjustments: Optional[dict[str, dict[str, Any]]],
    ) -> dict[str, Any]:
        base_turn_prob = hazard_turn_probability(
            item.hazard_profile,
            created_turn=item.created_turn,
            turn_index=self.turn_index,
        )
        base_threshold = (
            self._delayed_mention_leak_threshold()
            if self.config.delayed_mention_leak_policy == DelayedMentionLeakPolicy.ON
            else None
        )
        if adjustments and isinstance(adjustments.get(item.item_id), dict):
            return adjustments[item.item_id]
        return {
            "item_id": item.item_id,
            "kind": item.kind,
            "text": truncate_text(item.text, 140),
            "release_stage_role": item.release_stage_role,
            "adaptive_hazard_stage_policy": self.config.adaptive_hazard_stage_policy.value,
            "created_turn": item.created_turn,
            "base_turn_prob": base_turn_prob,
            "effective_turn_prob": base_turn_prob,
            "turn_prob_multiplier": 1.0,
            "base_threshold": base_threshold,
            "effective_threshold": base_threshold,
            "threshold_shift": 0.0 if base_threshold is not None else None,
            "reasons": [],
            "signals": self._adaptive_hazard_signal_snapshot(),
        }

    def _due_delayed_mentions(
        self,
        *,
        adaptive_adjustments: Optional[dict[str, dict[str, Any]]] = None,
    ) -> list[DelayedMentionItem]:
        due: list[DelayedMentionItem] = []
        for item in self._active_delayed_mentions():
            if not (item.earliest_turn <= self.turn_index <= item.latest_turn):
                continue
            adjustment = self._delayed_mention_adjustment(item, adaptive_adjustments)
            if adjustment.get("effective_turn_prob", 0.0) <= 0.0:
                continue
            due.append(item)
        return due

    def _delayed_mention_leak_threshold(self) -> float:
        return clamp01(self.config.delayed_mention_leak_threshold, default=0.0)

    def _suppressed_delayed_mentions(
        self,
        *,
        injected_delayed_mentions: Optional[list[DelayedMentionItem]] = None,
        adaptive_adjustments: Optional[dict[str, dict[str, Any]]] = None,
    ) -> list[DelayedMentionItem]:
        if self.config.delayed_mention_leak_policy != DelayedMentionLeakPolicy.ON:
            return []
        injected_ids = {item.item_id for item in injected_delayed_mentions or []}
        suppressed: list[DelayedMentionItem] = []
        for item in self._active_delayed_mentions():
            if item.item_id in injected_ids:
                continue
            adjustment = self._delayed_mention_adjustment(item, adaptive_adjustments)
            turn_prob = float(adjustment.get("effective_turn_prob", 0.0) or 0.0)
            threshold = clamp01(
                adjustment.get("effective_threshold"),
                default=self._delayed_mention_leak_threshold(),
            )
            if turn_prob >= threshold:
                continue
            suppressed.append(item)
        suppressed.sort(
            key=lambda item: (
                float(
                    self._delayed_mention_adjustment(item, adaptive_adjustments).get(
                        "effective_turn_prob",
                        0.0,
                    )
                    or 0.0
                ),
                item.latest_turn,
                item.earliest_turn,
                item.item_id,
            )
        )
        return suppressed

    def _build_system_prompt(
        self,
        *,
        due_intents: Optional[list[DeferredIntent]] = None,
        injected_delayed_mentions: Optional[list[DelayedMentionItem]] = None,
        suppressed_delayed_mentions: Optional[list[DelayedMentionItem]] = None,
        adaptive_delayed_mention_adjustments: Optional[dict[str, dict[str, Any]]] = None,
    ) -> str:
        parts: list[str] = []
        base = compact_text(self.config.base_system)
        if base:
            parts.append(base)

        if self.memory_capsules:
            parts.append(
                "Recursive memory capsules loaded for context:\n"
                f"{render_capsules(self.memory_capsules)}"
            )

        if (
            self.config.conclusion_mode == ConclusionMode.SOFT_STEER
            and self.conclusion_hypotheses
        ):
            hypothesis = self.conclusion_hypotheses[-1]
            if self.config.conclusion_steer_injection == ConclusionSteerInjection.CONCLUSION_LINE:
                injected_hypothesis = extract_conclusion_line(hypothesis) or hypothesis
            else:
                injected_hypothesis = hypothesis
            if self.config.conclusion_steer_strength == SteerStrength.WEAK:
                steer_rule = (
                    "Treat it as a gentle hint. Use it only if it helps; otherwise ignore it."
                )
            elif self.config.conclusion_steer_strength == SteerStrength.STRONG:
                steer_rule = (
                    "Use it as an active steering objective for the next reply:\n"
                    "- If multiple helpful replies are possible, choose the one that most increases progress toward the hypothesis.\n"
                    "- Prefer concrete next steps, decision points, and structure that move the dialogue toward the hypothesis.\n"
                    "- If the latest user turn changes direction or contradicts the hypothesis, prioritize the current user turn and update course."
                )
            else:
                steer_rule = (
                    "Treat it as a soft hypothesis, not a hard constraint. "
                    "Revise it if the current evidence contradicts it."
                )
            parts.append(
                "Provisional end-state hypothesis for the conversation:\n"
                f"{injected_hypothesis}\n\n"
                f"{steer_rule}"
            )

        if (
            self.config.deferred_intent_backend != DeferredIntentBackend.INBAND
            and self.config.deferred_intent_latent_injection
            == DeferredIntentLatentInjection.ACTIVE
        ):
            active_intents = [
                intent
                for intent in self._active_deferred_intents()
                if intent.created_turn < self.turn_index
            ]
            if active_intents:
                parts.append(
                    "Latent deferred utterance intentions (private; do NOT reveal to the user yet):\n"
                    f"{render_deferred_intents(active_intents)}\n\n"
                    "Rules:\n"
                    "- Do not quote, paraphrase, or hint at these intentions.\n"
                    "- Keep them only as internal planning context.\n"
                    "- Only realize a deferred intention when it appears in an explicit 'due now' section."
                )

        if suppressed_delayed_mentions:
            bullets_parts: list[str] = []
            for item in suppressed_delayed_mentions:
                adjustment = self._delayed_mention_adjustment(
                    item,
                    adaptive_delayed_mention_adjustments,
                )
                p_base = clamp01(adjustment.get("base_turn_prob"), default=0.0)
                p_eff = clamp01(adjustment.get("effective_turn_prob"), default=p_base)
                threshold = clamp01(
                    adjustment.get("effective_threshold"),
                    default=self._delayed_mention_leak_threshold(),
                )
                bullets_parts.append(
                    f"- [{item.item_id}] p_base={p_base:.2f} p_eff={p_eff:.2f} threshold={threshold:.2f} | {item.text}"
                )
            bullets = "\n".join(bullets_parts)
            parts.append(
                "Delayed mention targets currently remain latent (private; do NOT surface them explicitly yet):\n"
                f"{bullets}\n\n"
                "Rules:\n"
                "- Do not quote, paraphrase, or explicitly state these targets while their current effective turn probability stays below the listed threshold.\n"
                "- They may still shape internal trajectory, question choice, and ordering.\n"
                "- Only surface them when they appear in a due-now section or their current effective turn probability clears the threshold."
            )

        if (
            self.config.deferred_intent_mode
            in {DeferredIntentMode.SOFT_FIRE, DeferredIntentMode.HARD_FIRE}
            and due_intents
        ):
            bullets = "\n".join(
                f"- [{intent.intent_id}] {intent.intent}" for intent in due_intents
            )
            if self.config.deferred_intent_mode == DeferredIntentMode.HARD_FIRE:
                parts.append(
                    "Deferred utterances are due now. Include all of them explicitly in this turn unless the latest user turn makes one impossible.\n"
                    f"{bullets}\n\n"
                    "Preserve the intent content faithfully. If there is tension with the latest user turn, state the conflict directly instead of silently dropping the deferred utterance."
                )
            else:
                parts.append(
                    "Deferred utterances are due now. Integrate them only if they still fit the current turn.\n"
                    f"{bullets}\n\n"
                    "Realize them naturally instead of quoting them verbatim. "
                    "If the latest user turn conflicts with a deferred utterance, prioritize the current evidence."
                )

        if (
            self.config.delayed_mention_mode == DelayedMentionMode.SOFT_FIRE
            and injected_delayed_mentions
        ):
            bullets = "\n".join(
                f"- [{item.item_id}] {item.text}" for item in injected_delayed_mentions
            )
            parts.append(
                "Some delayed mention targets are due now. If they fit the current turn, weave them into the reply naturally.\n"
                f"{bullets}\n\n"
                "Rules:\n"
                "- Do not quote the bullets verbatim; rephrase naturally.\n"
                "- Do not mention that you were instructed or that these are 'due'.\n"
                "- If the latest user turn conflicts, prioritize the user."
            )

        return "\n\n".join(parts).strip()

    def _log(self, event_type: str, payload: dict[str, Any]) -> None:
        save_jsonl(
            self.log_path,
            EventRecord(
                timestamp=time.time(),
                provider=self.adapter.provider_name,
                model=self.adapter.model,
                turn_index=self.turn_index,
                event_type=event_type,
                payload=payload,
            ),
        )

    def _probe_memory_capsule(self) -> Optional[str]:
        if self.config.memory_every <= 0:
            return None
        if self.turn_index == 0 or self.turn_index % self.config.memory_every != 0:
            return None

        source = textwrap.dedent(
            f"""\
            Existing memory capsules:
            {render_capsules(self.memory_capsules)}

            Recent dialogue window:
            {render_messages(self._recent_window())}

            Produce ONE new memory capsule for future turns.
            Requirements:
            - under {self.config.memory_word_budget} words
            - preserve user goals, assistant commitments, unresolved questions, constraints, and recurring motifs
            - prefer concrete facts over style
            - plain text only
            """
        ).strip()

        response = self.adapter.generate(
            system=(
                "You compress conversation state into a compact memory capsule "
                "for later recursive loading."
            ),
            messages=[ChatMessage(role="user", content=source)],
            config=self.config.probe_config,
        )
        capsule = compact_text(response.text)
        if capsule:
            self.memory_capsules.append(capsule)
            self.memory_capsules = self.memory_capsules[-self.config.memory_capsule_limit :]
            self._log(
                "memory_capsule",
                {
                    "capsule": capsule,
                    "usage": response.usage,
                    "finish_reason": response.finish_reason,
                    "request_id": response.request_id,
                },
            )
        return capsule or None

    def _probe_conclusion(self) -> Optional[str]:
        if self.config.conclusion_every <= 0:
            return None
        if self.turn_index == 0 or self.turn_index % self.config.conclusion_every != 0:
            return None

        source = textwrap.dedent(
            f"""\
            Memory capsules:
            {render_capsules(self.memory_capsules)}

            Recent dialogue window:
            {render_messages(self._recent_window())}

            Based on the trajectory so far, state the MOST LIKELY eventual conclusion
            or thesis this conversation is moving toward.

            Return exactly this format:
            CONCLUSION: <one or two sentences>
            CONFIDENCE: <0.00 to 1.00>
            EVIDENCE: <brief reason grounded in the transcript>
            KEYWORDS: <3-5 distinctive keywords/phrases separated by ;>
            MENTION_DELAY_MIN_TURNS: <integer, minimum turns to wait before the conclusion is likely to be said>
            MENTION_DELAY_MAX_TURNS: <integer >= MIN, maximum turns until it is likely to be said>
            MENTION_HAZARD_PROFILE: <JSON list like [{{"delay_turns": 1, "prob": 0.20}}, ...] or [] ; delays are relative to the current turn>
            MENTION_LIKELIHOOD: <0.00 to 1.00>
            DELAY_STRATEGY: <one of: clarify_first | gather_constraints | build_context | avoid_premature_commitment | already_ready | other>
            DELAY_SIGNALS: <2-6 short tags separated by ,>
            DELAY_RATIONALE: <<= 140 chars, no chain-of-thought>
            """
        ).strip()

        response = self.adapter.generate(
            system=(
                "You are an observer of dialogue trajectory. "
                "Do not answer the user directly. Infer the likely end-state."
            ),
            messages=[ChatMessage(role="user", content=source)],
            config=self.config.probe_config,
        )
        hypothesis = response.text.strip()
        if hypothesis:
            hypothesis_for_injection = extract_conclusion_probe_block(hypothesis)
            conclusion_line = extract_conclusion_line(hypothesis)
            keywords = parse_probe_list(
                extract_probe_line_value(hypothesis, "KEYWORDS"),
                max_items=5,
                max_item_chars=64,
            )
            mention_delay_min_turns = parse_probe_int(
                extract_probe_line_value(hypothesis, "MENTION_DELAY_MIN_TURNS")
            )
            mention_delay_max_turns = parse_probe_int(
                extract_probe_line_value(hypothesis, "MENTION_DELAY_MAX_TURNS")
            )
            mention_hazard_profile = parse_hazard_profile(
                extract_probe_line_value(hypothesis, "MENTION_HAZARD_PROFILE")
            )
            (
                mention_delay_min_turns,
                mention_delay_max_turns,
                mention_hazard_profile,
            ) = resolve_mention_hazard_plan(
                created_turn=self.turn_index,
                delay_min=mention_delay_min_turns,
                delay_max=mention_delay_max_turns,
                raw_hazard_profile=mention_hazard_profile,
            )
            mention_likelihood = parse_probe_float(
                extract_probe_line_value(hypothesis, "MENTION_LIKELIHOOD")
            )
            if mention_likelihood is not None:
                mention_likelihood = max(0.0, min(1.0, mention_likelihood))
            delay_strategy = truncate_text(
                extract_probe_line_value(hypothesis, "DELAY_STRATEGY"), 48
            )
            delay_signals = parse_probe_list(
                extract_probe_line_value(hypothesis, "DELAY_SIGNALS"),
                max_items=6,
                max_item_chars=80,
            )
            delay_rationale = truncate_text(
                extract_probe_line_value(hypothesis, "DELAY_RATIONALE"), 160
            )
            self.conclusion_hypotheses.append(hypothesis_for_injection)
            self.latest_conclusion_probe_turn = self.turn_index
            self.latest_conclusion_line = conclusion_line
            self.latest_conclusion_keywords = keywords
            self.latest_conclusion_mention_delay_min_turns = mention_delay_min_turns
            self.latest_conclusion_mention_delay_max_turns = mention_delay_max_turns
            self.latest_conclusion_mention_hazard_profile = list(mention_hazard_profile)
            self.latest_conclusion_mention_likelihood = mention_likelihood
            self.latest_conclusion_delay_strategy = delay_strategy
            self.latest_conclusion_delay_signals = delay_signals
            self.latest_conclusion_delay_rationale = delay_rationale

            conclusion_text = strip_conclusion_prefix(conclusion_line)
            conclusion_plan_item: Optional[DelayedMentionItem] = None
            if conclusion_text:
                existing = next(
                    (
                        item
                        for item in self.delayed_mentions
                        if item.status == "active" and item.kind == "conclusion"
                    ),
                    None,
                )
                if existing is None:
                    item_id = f"dm-{self.next_delayed_mention_index:04d}"
                    self.next_delayed_mention_index += 1
                    conclusion_plan_item = DelayedMentionItem(
                        item_id=item_id,
                        created_turn=self.turn_index,
                        kind="conclusion",
                        text=conclusion_text,
                        keywords=keywords,
                        earliest_turn=self.turn_index + mention_delay_min_turns,
                        latest_turn=self.turn_index + mention_delay_max_turns,
                        hazard_profile=list(mention_hazard_profile),
                        likelihood=float(mention_likelihood or 0.0),
                        delay_strategy=delay_strategy,
                        delay_signals=list(delay_signals),
                        delay_rationale=delay_rationale,
                        release_stage_role="conclusion",
                        status="active",
                    )
                    self.delayed_mentions.append(conclusion_plan_item)
                    self._log(
                        "delayed_mention_plan",
                        {
                            "source": "conclusion_probe",
                            "created": True,
                            "item": conclusion_plan_item.to_dict(),
                        },
                    )
                else:
                    existing.created_turn = self.turn_index
                    existing.text = conclusion_text
                    existing.keywords = list(keywords)
                    existing.earliest_turn = self.turn_index + mention_delay_min_turns
                    existing.latest_turn = self.turn_index + mention_delay_max_turns
                    existing.hazard_profile = list(mention_hazard_profile)
                    existing.likelihood = float(mention_likelihood or 0.0)
                    existing.delay_strategy = delay_strategy
                    existing.delay_signals = list(delay_signals)
                    existing.delay_rationale = delay_rationale
                    existing.release_stage_role = "conclusion"
                    existing.mention_turn = None
                    existing.terminal_reason = ""
                    conclusion_plan_item = existing
                    self._log(
                        "delayed_mention_plan",
                        {
                            "source": "conclusion_probe",
                            "created": False,
                            "item": conclusion_plan_item.to_dict(),
                        },
                    )
            self._log(
                "conclusion_probe",
                {
                    "hypothesis": hypothesis,
                    "hypothesis_for_injection": hypothesis_for_injection,
                    "conclusion_line": conclusion_line,
                    "keywords": keywords,
                    "mention_delay_min_turns": mention_delay_min_turns,
                    "mention_delay_max_turns": mention_delay_max_turns,
                    "mention_hazard_profile": list(mention_hazard_profile),
                    "mention_earliest_turn": (
                        self.turn_index + mention_delay_min_turns
                        if mention_delay_min_turns is not None
                        else None
                    ),
                    "mention_latest_turn": (
                        self.turn_index + mention_delay_max_turns
                        if mention_delay_max_turns is not None
                        else None
                    ),
                    "mention_likelihood": mention_likelihood,
                    "delay_strategy": delay_strategy,
                    "delay_signals": delay_signals,
                    "delay_rationale": delay_rationale,
                    "delayed_mention_item_id": (
                        conclusion_plan_item.item_id if conclusion_plan_item else None
                    ),
                    "usage": response.usage,
                    "finish_reason": response.finish_reason,
                    "request_id": response.request_id,
                },
            )
        return hypothesis or None

    def _probe_delayed_mentions(self) -> list[DelayedMentionItem]:
        if self.config.delayed_mention_every <= 0:
            return []
        if self.turn_index == 0 or self.turn_index % self.config.delayed_mention_every != 0:
            return []
        if self.config.delayed_mention_item_limit <= 0:
            return []

        limit = max(0, int(self.config.delayed_mention_item_limit))
        min_nonconclusion_items = min(
            limit,
            max(0, int(self.config.delayed_mention_min_nonconclusion_items or 0)),
        )
        min_kind_diversity = min(
            limit,
            max(1, int(self.config.delayed_mention_min_kind_diversity or 1)),
        )
        diversity_repair = (
            self.config.delayed_mention_diversity_repair
            == DelayedMentionDiversityRepairPolicy.ON
        )
        planner_probe_max_tokens = max(
            int(self.config.probe_config.max_tokens or 0),
            150 + 90 * limit,
        )

        def probe_config_with_budget(min_max_tokens: int) -> GenerationConfig:
            return GenerationConfig(
                temperature=self.config.probe_config.temperature,
                max_tokens=max(int(self.config.probe_config.max_tokens or 0), min_max_tokens),
                timeout_seconds=self.config.probe_config.timeout_seconds,
            )

        def summarize_plan(items: list[DelayedMentionItem]) -> dict[str, Any]:
            kind_counts = count_strings(item.kind for item in items)
            kind_diversity = len(kind_counts)
            nonconclusion_count = sum(1 for item in items if item.kind != "conclusion")
            return {
                "min_nonconclusion_items": min_nonconclusion_items,
                "min_kind_diversity": min_kind_diversity,
                "nonconclusion_count": nonconclusion_count,
                "kind_diversity": kind_diversity,
                "kind_counts": kind_counts or None,
                "satisfies_nonconclusion_min": (
                    nonconclusion_count >= min_nonconclusion_items
                ),
                "satisfies_kind_diversity_min": kind_diversity >= min_kind_diversity,
            }

        def build_item_from_raw(
            item_raw: dict[str, Any],
            *,
            allow_conclusion: bool = True,
        ) -> Optional[DelayedMentionItem]:
            text = compact_text(str(item_raw.get("text") or ""))
            if not text:
                return None
            kind = truncate_text(item_raw.get("kind") or "other", 24) or "other"
            kind = compact_text(kind).lower() or "other"
            if not allow_conclusion and kind == "conclusion":
                return None
            keywords = coerce_str_list(item_raw.get("keywords"))
            keywords = [truncate_text(x, 64) for x in keywords if x][:5]

            delay_min = coerce_int(item_raw.get("mention_delay_min_turns"))
            delay_max = coerce_int(item_raw.get("mention_delay_max_turns"))
            delay_min, delay_max, mention_hazard_profile = resolve_mention_hazard_plan(
                created_turn=self.turn_index,
                delay_min=delay_min,
                delay_max=delay_max,
                raw_hazard_profile=(
                    item_raw.get("mention_hazard_profile") or item_raw.get("hazard_profile")
                ),
            )

            likelihood = clamp01(item_raw.get("mention_likelihood"), default=0.0)
            delay_strategy = truncate_text(item_raw.get("delay_strategy") or "", 48)
            delay_signals = coerce_str_list(item_raw.get("delay_signals"))
            delay_signals = [truncate_text(x, 80) for x in delay_signals if x][:6]
            delay_rationale = truncate_text(item_raw.get("delay_rationale") or "", 160)
            release_stage_role = classify_delayed_mention_stage_role(
                kind=kind,
                text=text,
                delay_strategy=delay_strategy,
                delay_signals=delay_signals,
            )
            mention_hazard_profile = reshape_stage_role_hazard_profile(
                policy=self.config.adaptive_hazard_stage_policy,
                stage_role=release_stage_role,
                profile=mention_hazard_profile,
            )
            delay_min, delay_max = [
                x - self.turn_index
                for x in hazard_bounds_from_profile(
                    created_turn=self.turn_index,
                    profile=mention_hazard_profile,
                )
            ]

            item_id = f"dm-{self.next_delayed_mention_index:04d}"
            self.next_delayed_mention_index += 1
            return DelayedMentionItem(
                item_id=item_id,
                created_turn=self.turn_index,
                kind=kind,
                text=text,
                keywords=keywords,
                earliest_turn=self.turn_index + delay_min,
                latest_turn=self.turn_index + delay_max,
                hazard_profile=list(mention_hazard_profile),
                likelihood=likelihood,
                delay_strategy=delay_strategy,
                delay_signals=delay_signals,
                delay_rationale=delay_rationale,
                release_stage_role=release_stage_role,
                status="active",
            )

        def merge_diverse_items(
            primary: list[DelayedMentionItem],
            supplemental: list[DelayedMentionItem],
        ) -> list[DelayedMentionItem]:
            combined = primary + supplemental
            deduped: list[DelayedMentionItem] = []
            seen_keys: set[str] = set()
            for item in combined:
                key = compact_text(item.text).lower()
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                deduped.append(item)

            ordered = sorted(
                deduped,
                key=lambda item: (
                    item.kind == "conclusion",
                    -(item.likelihood or 0.0),
                    item.latest_turn,
                    item.item_id,
                ),
            )
            selected: list[DelayedMentionItem] = []
            selected_ids: set[str] = set()
            seen_kinds: set[str] = set()
            for item in ordered:
                if len(selected) >= limit:
                    break
                if item.kind in seen_kinds:
                    continue
                selected.append(item)
                selected_ids.add(item.item_id)
                seen_kinds.add(item.kind)
            for item in ordered:
                if len(selected) >= limit:
                    break
                if item.item_id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item.item_id)
            return selected[:limit]

        diversity_lines = [
            "- Do not spend every slot on conclusion if other delayed targets are plausible.",
            "- Separate conclusion from the supporting items that make it feel earned later.",
        ]
        if min_nonconclusion_items > 0:
            diversity_lines.append(
                f"- Return at least {min_nonconclusion_items} item(s) whose kind is not conclusion when plausible."
            )
        if min_kind_diversity > 1:
            diversity_lines.append(
                f"- Aim for at least {min_kind_diversity} distinct kinds across the returned items when plausible."
            )
        diversity_lines.append(
            "- Useful non-conclusion kinds include caveat, option, constraint, definition, and migration risk."
        )
        diversity_guidance = "\n".join(diversity_lines)
        compact_output_rules = "\n".join(
            [
                "- Keep each text to one short sentence or phrase, at most 18 words.",
                "- Use 1-3 short keywords only.",
                "- Use at most 2 hazard points per item.",
                "- Omit delay_signals and delay_rationale unless truly needed.",
                "- Prefer compact kind labels and compact delay_strategy labels.",
            ]
        )
        source = textwrap.dedent(
            f"""\
            Memory capsules:
            {render_capsules(self.memory_capsules)}

            Recent dialogue window:
            {render_messages(self._recent_window())}

            Identify up to {limit} "delayed mention targets" that the assistant is likely to bring up later,
            but should NOT mention immediately. These are not tasks; they are items that become relevant later.

            Diversity requirements:
            {diversity_guidance}

            Compact output rules:
            {compact_output_rules}

            Return STRICT JSON (no markdown, no code fences):
            {{
              "items": [
                {{
                  "kind": "caveat|definition|option|constraint|migration_risk|conclusion|other",
                  "text": "<short delayed mention target>",
                  "keywords": ["<1-3 short phrases>"],
                  "mention_hazard_profile": [{{"delay_turns": <int>=1, "prob": <0.00-1.00>}}, ...],
                  "mention_likelihood": <0.00-1.00>,
                  "delay_strategy": "<short label>"
                }}
              ]
            }}
            """
        ).strip()

        response = self.adapter.generate(
            system=(
                "Delayed mention planner. You observe dialogue trajectory and propose future mention targets. "
                "Do not answer the user directly."
            ),
            messages=[ChatMessage(role="user", content=source)],
            config=probe_config_with_budget(planner_probe_max_tokens),
        )
        raw = (response.text or "").strip()
        parsed = extract_json_value(raw)
        if not isinstance(parsed, dict):
            self._log(
                "delayed_mention_plan_error",
                {
                    "error": "Planner output was not a JSON object.",
                    "raw": raw,
                    "usage": response.usage,
                    "finish_reason": response.finish_reason,
                    "request_id": response.request_id,
                    "probe_max_tokens": planner_probe_max_tokens,
                },
            )
            return []
        items_raw = parsed.get("items")
        if not isinstance(items_raw, list):
            self._log(
                "delayed_mention_plan_error",
                {
                    "error": "JSON must contain list field 'items'.",
                    "raw": raw,
                    "parsed_type": type(parsed.get("items")).__name__,
                    "usage": response.usage,
                    "finish_reason": response.finish_reason,
                    "request_id": response.request_id,
                    "probe_max_tokens": planner_probe_max_tokens,
                },
            )
            return []

        created: list[DelayedMentionItem] = []
        for item_raw in items_raw[:limit]:
            if not isinstance(item_raw, dict):
                continue
            item = build_item_from_raw(item_raw)
            if item is not None:
                created.append(item)

        repair_payload: Optional[dict[str, Any]] = None
        plan_validation = summarize_plan(created)
        if created and diversity_repair and (
            not plan_validation["satisfies_nonconclusion_min"]
            or not plan_validation["satisfies_kind_diversity_min"]
        ):
            missing_nonconclusion = max(
                0,
                min_nonconclusion_items - int(plan_validation["nonconclusion_count"] or 0),
            )
            missing_kind_diversity = max(
                0,
                min_kind_diversity - int(plan_validation["kind_diversity"] or 0),
            )
            repair_limit = min(limit, max(1, missing_nonconclusion, missing_kind_diversity))
            repair_probe_max_tokens = max(
                int(self.config.probe_config.max_tokens or 0),
                120 + 80 * repair_limit,
            )
            used_kinds = sorted(
                str(kind)
                for kind in (plan_validation.get("kind_counts") or {}).keys()
                if kind
            )
            repair_source = textwrap.dedent(
                f"""\
                Recent dialogue window:
                {render_messages(self._recent_window())}

                Your previous delayed-mention plan under-produced non-conclusion diversity.
                Existing kinds: {", ".join(used_kinds) if used_kinds else "(none)"}
                Existing plan:
                {json.dumps([item.to_dict() for item in created], ensure_ascii=False, indent=2)}

                Return up to {repair_limit} ADDITIONAL delayed mention targets that are NOT conclusion.
                Requirements:
                - Every item must use a kind other than conclusion.
                - Prefer kinds not already used.
                - Prioritize caveat, option, constraint, definition, migration_risk, or other concrete release-time support items.
                - Do not repeat the existing items too closely.
                - Keep each text to one short sentence or phrase, at most 18 words.
                - Use 1-3 short keywords and at most 2 hazard points.
                - Omit delay_signals and delay_rationale unless truly needed.

                Return STRICT JSON (no markdown, no code fences):
                {{
                  "items": [
                    {{
                      "kind": "caveat|definition|option|constraint|migration_risk|other",
                      "text": "<short delayed mention target>",
                      "keywords": ["<1-3 short phrases>"],
                      "mention_hazard_profile": [{{"delay_turns": <int>=1, "prob": <0.00-1.00>}}, ...],
                      "mention_likelihood": <0.00-1.00>,
                      "delay_strategy": "<short label>"
                    }}
                  ]
                }}
                """
            ).strip()
            repair_response = self.adapter.generate(
                system=(
                    "Delayed mention diversity repair planner. Generate only supplemental non-conclusion "
                    "delayed mentions. Do not answer the user directly."
                ),
                messages=[ChatMessage(role="user", content=repair_source)],
                config=probe_config_with_budget(repair_probe_max_tokens),
            )
            repair_raw = (repair_response.text or "").strip()
            repair_parsed = extract_json_value(repair_raw)
            repair_created: list[DelayedMentionItem] = []
            repair_error: Optional[str] = None
            if not isinstance(repair_parsed, dict):
                repair_error = "Repair planner output was not a JSON object."
            else:
                repair_items_raw = repair_parsed.get("items")
                if not isinstance(repair_items_raw, list):
                    repair_error = "Repair JSON must contain list field 'items'."
                else:
                    for item_raw in repair_items_raw[:repair_limit]:
                        if not isinstance(item_raw, dict):
                            continue
                        item = build_item_from_raw(item_raw, allow_conclusion=False)
                        if item is not None:
                            repair_created.append(item)
            if repair_created:
                created = merge_diverse_items(created, repair_created)
            plan_validation = summarize_plan(created)
            repair_payload = {
                "requested_limit": repair_limit,
                "missing_nonconclusion": missing_nonconclusion,
                "missing_kind_diversity": missing_kind_diversity,
                "created_count": len(repair_created),
                "created_items": [item.to_dict() for item in repair_created],
                "error": repair_error,
                "raw": repair_raw,
                "usage": repair_response.usage,
                "finish_reason": repair_response.finish_reason,
                "request_id": repair_response.request_id,
                "probe_max_tokens": repair_probe_max_tokens,
                "final_plan_validation": plan_validation,
            }
            self._log("delayed_mention_diversity_repair", repair_payload)

        for item in created:
            self.delayed_mentions.append(item)

        self._log(
            "delayed_mention_plan",
            {
                "source": "delayed_mention_probe",
                "created": bool(created),
                "created_items": [item.to_dict() for item in created],
                "created_count": len(created),
                "plan_validation": plan_validation,
                "diversity_repair_policy": self.config.delayed_mention_diversity_repair.value,
                "diversity_repair_applied": repair_payload is not None,
                "raw": raw,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
                "request_id": response.request_id,
                "probe_max_tokens": planner_probe_max_tokens,
            },
        )
        if created and (
            not plan_validation["satisfies_nonconclusion_min"]
            or not plan_validation["satisfies_kind_diversity_min"]
        ):
            self._log(
                "delayed_mention_plan_warning",
                {
                    "warning": "Planner output did not satisfy delayed-mention diversity targets.",
                    "plan_validation": plan_validation,
                    "created_items": [item.to_dict() for item in created],
                },
            )
        return created

    def _probe_latent_convergence(
        self,
        *,
        assistant_text: str,
        planned_earliest_turn: Optional[int],
        planned_latest_turn: Optional[int],
        explicit_mention_present: Optional[bool],
    ) -> Optional[dict[str, Any]]:
        if self.config.latent_convergence_every <= 0:
            return None
        if self.turn_index == 0 or self.turn_index % self.config.latent_convergence_every != 0:
            return None

        conclusion_text = strip_conclusion_prefix(self.latest_conclusion_line)
        if not conclusion_text:
            return None

        source = textwrap.dedent(
            f"""\
            Current turn index: {self.turn_index}
            Latest conclusion probe turn: {self.latest_conclusion_probe_turn}
            Planned earliest articulation turn: {planned_earliest_turn if planned_earliest_turn is not None else "unknown"}
            Planned latest articulation turn: {planned_latest_turn if planned_latest_turn is not None else "unknown"}

            Latest conclusion line:
            {conclusion_text}

            Conclusion keywords:
            {"; ".join(self.latest_conclusion_keywords) if self.latest_conclusion_keywords else "(none)"}

            Latest assistant reply:
            {assistant_text}

            Recent dialogue window:
            {render_messages(self._recent_window())}

            Evaluate whether the trajectory is semantically converging toward the conclusion even if it is not explicitly stated yet.

            Return exactly this format:
            LATENT_ALIGNMENT: <0.00 to 1.00>
            ARTICULATION_READINESS: <0.00 to 1.00>
            LEAKAGE_RISK: <0.00 to 1.00>
            EXPLICIT_MENTION_PRESENT: <yes|no>
            TRAJECTORY_STAGE: <latent|approaching|ready|explicit|diverged>
            SHIFT_SIGNALS: <2-6 short tags separated by ,>
            EVIDENCE: <<= 160 chars, no chain-of-thought>
            """
        ).strip()

        response = self.observer_adapter.generate(
            system=(
                "Latent convergence trace. You observe dialogue dynamics and estimate semantic convergence "
                "toward the latest conclusion without answering the user."
            ),
            messages=[ChatMessage(role="user", content=source)],
            config=self.config.probe_config,
        )
        raw = response.text.strip()
        alignment = parse_probe_float(extract_probe_line_value(raw, "LATENT_ALIGNMENT"))
        readiness = parse_probe_float(
            extract_probe_line_value(raw, "ARTICULATION_READINESS")
        )
        leakage_risk = parse_probe_float(extract_probe_line_value(raw, "LEAKAGE_RISK"))
        explicit_from_probe = parse_probe_bool(
            extract_probe_line_value(raw, "EXPLICIT_MENTION_PRESENT")
        )
        trajectory_stage = truncate_text(
            extract_probe_line_value(raw, "TRAJECTORY_STAGE"),
            24,
        )
        shift_signals = parse_probe_list(
            extract_probe_line_value(raw, "SHIFT_SIGNALS"),
            max_items=6,
            max_item_chars=80,
        )
        evidence = truncate_text(extract_probe_line_value(raw, "EVIDENCE"), 160)
        payload = {
            "conclusion_probe_turn": self.latest_conclusion_probe_turn,
            "conclusion_line": conclusion_text,
            "planned_earliest_turn": planned_earliest_turn,
            "planned_latest_turn": planned_latest_turn,
            "judge_source": self.latent_judge_source,
            "judge_provider": self.observer_adapter.provider_name,
            "judge_model": self.observer_adapter.model,
            "alignment": alignment,
            "articulation_readiness": readiness,
            "leakage_risk": leakage_risk,
            "explicit_mention_present": (
                explicit_from_probe
                if explicit_from_probe is not None
                else explicit_mention_present
            ),
            "trajectory_stage": trajectory_stage or None,
            "shift_signals": shift_signals,
            "evidence": evidence or None,
            "raw": raw,
            "usage": response.usage,
            "finish_reason": response.finish_reason,
            "request_id": response.request_id,
        }
        self.latest_latent_convergence_trace = payload
        self._log("latent_convergence_trace", payload)
        return payload

    def _probe_embedding_convergence(
        self,
        *,
        assistant_text: str,
        planned_earliest_turn: Optional[int],
        planned_latest_turn: Optional[int],
        explicit_mention_present: Optional[bool],
    ) -> Optional[dict[str, Any]]:
        if self.config.latent_convergence_every <= 0:
            return None
        if self.turn_index == 0 or self.turn_index % self.config.latent_convergence_every != 0:
            return None
        if self.config.semantic_judge_backend not in {
            SemanticJudgeBackend.EMBEDDING,
            SemanticJudgeBackend.BOTH,
        }:
            return None
        if self.embedding_adapter is None:
            return None

        conclusion_text = strip_conclusion_prefix(self.latest_conclusion_line)
        if not conclusion_text:
            return None

        keyword_text = "; ".join(self.latest_conclusion_keywords or [])
        inputs = [assistant_text, conclusion_text]
        keyword_index: Optional[int] = None
        if keyword_text:
            keyword_index = len(inputs)
            inputs.append(keyword_text)

        response = self.embedding_adapter.embed(
            inputs=inputs,
            config=self.config.probe_config,
        )
        if len(response.embeddings) != len(inputs):
            return None

        assistant_vec = response.embeddings[0]
        conclusion_vec = response.embeddings[1]
        line_alignment = cosine_alignment(assistant_vec, conclusion_vec)
        keyword_alignment = (
            cosine_alignment(assistant_vec, response.embeddings[keyword_index])
            if keyword_index is not None
            else None
        )
        alignment_candidates = [
            value
            for value in (line_alignment, keyword_alignment)
            if isinstance(value, float)
        ]
        alignment = max(alignment_candidates) if alignment_candidates else None
        payload = {
            "conclusion_probe_turn": self.latest_conclusion_probe_turn,
            "conclusion_line": conclusion_text,
            "planned_earliest_turn": planned_earliest_turn,
            "planned_latest_turn": planned_latest_turn,
            "judge_source": "embedding",
            "judge_provider": self.embedding_adapter.provider_name,
            "judge_model": self.embedding_adapter.model,
            "alignment": alignment,
            "line_alignment": line_alignment,
            "keyword_alignment": keyword_alignment,
            "alignment_rule": "max",
            "explicit_mention_present": explicit_mention_present,
            "raw": response.raw,
            "usage": response.usage,
            "request_id": response.request_id,
        }
        self.latest_embedding_convergence_trace = payload
        self._log("embedding_convergence_trace", payload)
        return payload

    def _probe_deferred_intent_plans(self) -> list[DeferredIntent]:
        if self.turn_index == 0:
            return []

        if self.config.deferred_intent_plan_policy == DeferredIntentPlanPolicy.PERIODIC:
            if self.config.deferred_intent_every <= 0:
                return []
            if self.turn_index % self.config.deferred_intent_every != 0:
                return []
        elif self.config.deferred_intent_plan_policy == DeferredIntentPlanPolicy.AUTO:
            pass
        else:
            return []

        if (
            self.config.deferred_intent_plan_budget > 0
            and self.deferred_intent_plan_probe_calls >= self.config.deferred_intent_plan_budget
        ):
            return []

        active = self._active_deferred_intents()
        active_count = len(active)
        if active_count >= self.config.deferred_intent_limit:
            return []

        plan_capacity = max(0, self.config.deferred_intent_limit - active_count)
        plan_capacity = min(plan_capacity, max(0, int(self.config.deferred_intent_plan_max_new or 0)))
        if plan_capacity <= 0:
            return []

        if self.config.deferred_intent_strategy == DeferredIntentStrategy.FIXED:
            strategy_text = (
                f"Imagine up to {plan_capacity} future utterances that would be more appropriate later. "
                "They should be useful later but not appropriate yet."
            )
        elif self.config.deferred_intent_strategy == DeferredIntentStrategy.TRIGGER:
            strategy_text = (
                f"Imagine up to {plan_capacity} future utterances that should stay in reserve. "
                "Include timing triggers and cancel conditions."
            )
        else:
            strategy_text = (
                f"Imagine up to {plan_capacity} future utterances that should stay in reserve. "
                "Include timing triggers and cancel conditions. "
                "This plan may be revised later if the dialogue shifts."
            )

        existing = "\n".join(f"- {intent.summary_line()}" for intent in active) or "(none)"
        if self.config.deferred_intent_timing == DeferredIntentTiming.MODEL:
            timing_lines = textwrap.dedent(
                """\
                Timing:
                - Provide timing either as timing.delay_min_turns / timing.delay_max_turns (relative), or timing.earliest_turn / timing.latest_turn (absolute).
                - delay_min_turns must be >= 1. delay_max_turns must be >= delay_min_turns.
                """
            ).strip()
        elif self.config.deferred_intent_timing == DeferredIntentTiming.HAZARD:
            timing_lines = textwrap.dedent(
                """\
                Timing:
                - Provide a timing.hazard_profile list with per-delay probabilities.
                - Each item must have delay_turns >= 1 and prob > 0.0.
                - The probabilities should sum to about 1.0 and express when the utterance becomes most likely to surface.
                """
            ).strip()
        else:
            timing_lines = textwrap.dedent(
                f"""\
                Timing:
                - Do NOT provide timing fields. The harness will schedule each new intent at:
                  earliest_turn = current_turn + max(1, offset={self.config.deferred_intent_offset})
                  latest_turn = earliest_turn (+ grace={self.config.deferred_intent_grace} for trigger/adaptive)
                """
            ).strip()

        timing_schema = ""
        if self.config.deferred_intent_timing == DeferredIntentTiming.MODEL:
            timing_schema = (
                '                  "timing": {\n'
                '                    "delay_min_turns": 1,\n'
                '                    "delay_max_turns": 3\n'
                "                  },\n"
            )
        elif self.config.deferred_intent_timing == DeferredIntentTiming.HAZARD:
            timing_schema = (
                '                  "timing": {\n'
                '                    "hazard_profile": [\n'
                '                      {"delay_turns": 2, "prob": 0.20},\n'
                '                      {"delay_turns": 3, "prob": 0.45},\n'
                '                      {"delay_turns": 4, "prob": 0.35}\n'
                "                    ]\n"
                "                  },\n"
            )

        inject_schema = not self._deferred_intent_plan_compact_ok
        if inject_schema:
            schema_instructions = textwrap.dedent(
                f"""\
                Return strict JSON only.
                Use this schema:
                {{
                  "intents": [
                    {{
                      "kind": "summary|proposal|correction|question|reminder|other",
                      "intent": "what should be said later",
                      "why_not_now": "why it should be delayed",
                      "selection": {{
                        "strategy": "hold_for_user_prompt|need_more_constraints|avoid_premature|anticipate_request|topic_sensitive|other",
                        "signals": ["short tags describing cues used"],
                        "rationale": "1-2 sentences (no step-by-step reasoning)"
                      }},
                {timing_schema}\
                      "trigger": ["condition 1", "condition 2"],
                      "cancel_if": ["condition 1", "condition 2"],
                      "confidence": 0.0,
                      "priority": 0.0
                    }}
                  ]
                }}

                If no future utterance should be held, return {{"intents": []}}.
                """
            ).strip()
        else:
            timing_hint = (
                "Include timing either as timing.delay_min_turns/timing.delay_max_turns (relative), "
                "or timing.earliest_turn/timing.latest_turn (absolute)."
                if self.config.deferred_intent_timing == DeferredIntentTiming.MODEL
                else "Include timing.hazard_profile with per-delay probabilities."
                if self.config.deferred_intent_timing == DeferredIntentTiming.HAZARD
                else "Do NOT include timing fields."
            )
            schema_instructions = textwrap.dedent(
                f"""\
                Return strict JSON only as: {{"intents": [ ... ]}}.
                Each item should include: kind, intent, why_not_now, selection{{strategy,signals,rationale}}, trigger, cancel_if, confidence, priority.
                {timing_hint}
                If no intent should be held, return {{"intents": []}}.
                """
            ).strip()

        source = textwrap.dedent(
            f"""\
            Memory capsules:
            {render_capsules(self.memory_capsules)}

            Recent dialogue window:
            {render_messages(self._recent_window())}

            Current turn index: {self.turn_index}

            Existing active deferred intents:
            {existing}

            {strategy_text}

            {timing_lines}

            {schema_instructions}
            """
        ).strip()

        self.deferred_intent_plan_probe_calls += 1
        response = self.adapter.generate(
            system=(
                "You are a deferred utterance planner. Do not answer the user directly. "
                f"Propose up to {plan_capacity} future utterances to hold in reserve."
            ),
            messages=[ChatMessage(role="user", content=source)],
            config=self.config.probe_config,
        )
        parsed = extract_json_value(response.text)
        if not isinstance(parsed, (dict, list)):
            self._deferred_intent_plan_compact_ok = False
            self._log(
                "deferred_intent_plan_error",
                {
                    "raw_text": response.text,
                    "prompt_schema_injected": inject_schema,
                    "plan_policy": self.config.deferred_intent_plan_policy.value,
                    "plan_budget": self.config.deferred_intent_plan_budget,
                    "plan_probe_calls_used": self.deferred_intent_plan_probe_calls,
                    "usage": response.usage,
                    "finish_reason": response.finish_reason,
                    "request_id": response.request_id,
                },
            )
            return []

        plan_shape_ok = False
        if isinstance(parsed, list):
            plan_shape_ok = True
        elif isinstance(parsed.get("intents"), list):
            plan_shape_ok = True
        elif compact_text(str(parsed.get("intent", ""))):
            plan_shape_ok = True
        self._deferred_intent_plan_compact_ok = plan_shape_ok

        items: list[dict[str, Any]] = []
        if isinstance(parsed, list):
            items = [item for item in parsed if isinstance(item, dict)]
        elif isinstance(parsed.get("intents"), list):
            items = [item for item in parsed.get("intents") if isinstance(item, dict)]
        else:
            items = [parsed] if isinstance(parsed, dict) else []

        created: list[DeferredIntent] = []
        for item in items[:plan_capacity]:
            intent_id = f"di-{self.next_deferred_intent_index:04d}"
            planned = build_deferred_intent_from_plan(
                item,
                intent_id=intent_id,
                created_turn=self.turn_index,
                strategy=self.config.deferred_intent_strategy,
                timing=self.config.deferred_intent_timing,
                offset=self.config.deferred_intent_offset,
                grace=self.config.deferred_intent_grace,
            )
            if planned is None:
                continue
            self.next_deferred_intent_index += 1
            self.deferred_intents.append(planned)
            created.append(planned)

        self._log(
            "deferred_intent_plan",
            {
                "intent_id": created[0].intent_id if created else None,
                "created": bool(created),
                "intent": created[0].to_dict() if created else None,
                "created_intents": [intent.to_dict() for intent in created],
                "raw_plan": parsed,
                "prompt_schema_injected": inject_schema,
                "plan_policy": self.config.deferred_intent_plan_policy.value,
                "plan_budget": self.config.deferred_intent_plan_budget,
                "plan_probe_calls_used": self.deferred_intent_plan_probe_calls,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
                "request_id": response.request_id,
                "planning_capacity": plan_capacity,
            },
        )
        return created

    def _schedule_deferred_intents(self) -> tuple[list[DeferredIntent], list[dict[str, Any]]]:
        active = self._active_deferred_intents()
        if not active:
            return [], []

        scheduler_prompt_schema_injected: Optional[bool] = None
        scheduler_parse_ok: Optional[bool] = None

        if self.config.deferred_intent_strategy == DeferredIntentStrategy.FIXED:
            decisions_payload = {
                "decisions": [
                    {
                        "intent_id": intent.intent_id,
                        "action": (
                            "hold"
                            if self.turn_index < intent.earliest_turn
                            else "expire"
                            if self.turn_index > intent.latest_turn
                            else "fire"
                        ),
                        "reason": (
                            "Too early; keep it in reserve."
                            if self.turn_index < intent.earliest_turn
                            else "Timing window passed."
                            if self.turn_index > intent.latest_turn
                            else "Exact deferred timing reached."
                        ),
                    }
                    for intent in active
                ]
            }
            scheduler_usage = None
            scheduler_finish_reason = None
            scheduler_request_id = None
            scheduler_raw_text = ""
        else:
            inject_schema = not self._deferred_intent_scheduler_compact_ok
            scheduler_prompt_schema_injected = inject_schema

            intents_for_scheduler = [
                {
                    "intent_id": intent.intent_id,
                    "created_turn": intent.created_turn,
                    "earliest_turn": intent.earliest_turn,
                    "latest_turn": intent.latest_turn,
                    "hazard_profile": list(intent.hazard_profile),
                    "hazard_turn_prob": hazard_turn_probability(
                        intent.hazard_profile,
                        created_turn=intent.created_turn,
                        turn_index=self.turn_index,
                    ),
                    "kind": intent.kind,
                    "intent": intent.intent,
                    "why_not_now": intent.why_not_now,
                    "trigger": list(intent.trigger),
                    "cancel_if": list(intent.cancel_if),
                    "confidence": float(intent.confidence),
                    "priority": float(intent.priority),
                    "revision_count": int(intent.revision_count),
                }
                for intent in active
            ]
            intents_json = (
                json.dumps(intents_for_scheduler, ensure_ascii=False, indent=2)
                if inject_schema
                else json.dumps(intents_for_scheduler, ensure_ascii=False, separators=(",", ":"))
            )

            if inject_schema:
                decision_schema = textwrap.dedent(
                    """\
                    Return strict JSON only:
                    {
                      "decisions": [
                        {
                          "intent_id": "di-0001",
                          "action": "hold|fire|cancel|expire__REVISE_SUFFIX__",
                          "reason": "brief reason",
                          "decision_strategy": "trigger_match|topic_shift|window_expired|revise_wording|other",
                          "decision_signals": ["short tags describing cues used"],
                          "decision_rationale": "1-2 sentences (no step-by-step reasoning)",
                          "updated_intent": {
                            "kind": "optional",
                            "intent": "optional revised intent",
                            "why_not_now": "optional",
                            "trigger": ["optional"],
                            "cancel_if": ["optional"],
                            "confidence": 0.0,
                            "priority": 0.0,
                            "earliest_turn": 0,
                            "latest_turn": 0
                          }
                        }
                      ]
                    }
                    """
                ).strip()
            else:
                decision_schema = textwrap.dedent(
                    """\
                    Return strict JSON only as: {"decisions":[...]}.
                    Each decision must include: intent_id, action, reason, decision_strategy, decision_signals, decision_rationale.
                    If action is "revise", include updated_intent with any fields you change.
                    """
                ).strip()

            source = textwrap.dedent(
                f"""\
                Memory capsules:
                {render_capsules(self.memory_capsules)}

                Recent dialogue window:
                {render_messages(self._recent_window())}

                Current turn index: {self.turn_index}

                Deferred intents:
                {intents_json}

                Decide whether each deferred utterance should be held, fired, canceled, expired, or revised.
                Use "fire" only when the utterance would feel timely and appropriate now.
                Never fire before earliest_turn.
                Prefer "expire" if current_turn is past latest_turn and the opportunity has passed.
                Use "cancel" if the topic moved on or its assumptions broke.
                "revise" is allowed only if the core idea is still useful but the wording or timing should change.

                {decision_schema}
                """
            ).replace(
                "__REVISE_SUFFIX__",
                "|revise" if self.config.deferred_intent_strategy == DeferredIntentStrategy.ADAPTIVE else "",
            ).strip()

            response = self.adapter.generate(
                system=(
                    "You are a deferred utterance scheduler. "
                    "Do not answer the user directly. Decide timing actions for stored utterances."
                ),
                messages=[ChatMessage(role="user", content=source)],
                config=self.config.probe_config,
            )
            decisions_payload = extract_json_value(response.text)
            scheduler_parse_ok = isinstance(decisions_payload, dict) and isinstance(
                decisions_payload.get("decisions"), list
            )
            self._deferred_intent_scheduler_compact_ok = bool(scheduler_parse_ok)
            if not scheduler_parse_ok:
                decisions_payload = {"decisions": []}
            scheduler_usage = response.usage
            scheduler_finish_reason = response.finish_reason
            scheduler_request_id = response.request_id
            scheduler_raw_text = response.text

        raw_decisions = decisions_payload.get("decisions") if isinstance(decisions_payload, dict) else []
        if not isinstance(raw_decisions, list):
            raw_decisions = []
        by_id = {
            compact_text(str(item.get("intent_id", ""))): item
            for item in raw_decisions
            if isinstance(item, dict) and compact_text(str(item.get("intent_id", "")))
        }

        due_intents: list[DeferredIntent] = []
        applied: list[dict[str, Any]] = []

        for intent in active:
            item = by_id.get(intent.intent_id, {})
            action = compact_text(str(item.get("action", "hold"))).lower() or "hold"
            reason = compact_text(str(item.get("reason", "")))
            decision_strategy = compact_text(str(item.get("decision_strategy", "")))
            decision_rationale = compact_text(str(item.get("decision_rationale", "")))
            decision_signals = coerce_str_list(item.get("decision_signals"))
            allowed_actions = {"hold", "fire", "cancel", "expire"}
            if self.config.deferred_intent_strategy == DeferredIntentStrategy.ADAPTIVE:
                allowed_actions.add("revise")
            if action not in allowed_actions:
                action = "hold"
                reason = compact_text(reason + " Invalid or missing action; defaulted to hold.")

            if action == "fire" and self.turn_index < intent.earliest_turn:
                action = "hold"
                reason = compact_text(reason + " Fire blocked because earliest_turn has not been reached.")

            if self.turn_index > intent.latest_turn and action in {"hold", "fire", "cancel"}:
                if action != "cancel":
                    action = "expire"
                    reason = compact_text(reason + " Opportunity window already passed.")

            status_before = intent.status
            updated_intent_payload = item.get("updated_intent") if isinstance(item, dict) else None

            if action == "hold":
                intent.status = "active"
                status_after = intent.status
            elif action == "cancel":
                intent.status = "canceled"
                intent.terminal_reason = reason
                status_after = intent.status
            elif action == "expire":
                intent.status = "expired"
                intent.terminal_reason = reason
                status_after = intent.status
            elif action == "revise":
                if self.config.deferred_intent_strategy != DeferredIntentStrategy.ADAPTIVE or not isinstance(updated_intent_payload, dict):
                    action = "hold"
                    intent.status = "active"
                    status_after = intent.status
                    reason = compact_text(reason + " Revision unavailable; defaulted to hold.")
                    updated_intent_payload = None
                else:
                    apply_revised_intent(
                        intent,
                        updated_intent_payload,
                        current_turn=self.turn_index,
                        timing=self.config.deferred_intent_timing,
                        default_offset=self.config.deferred_intent_offset,
                        default_grace=self.config.deferred_intent_grace,
                    )
                    status_after = intent.status
            else:  # fire
                intent.status = "fired"
                intent.fire_turn = self.turn_index
                intent.terminal_reason = reason
                due_intents.append(intent)
                status_after = intent.status

            if decision_strategy or decision_signals or decision_rationale:
                intent.decision_strategy = decision_strategy
                intent.decision_signals = decision_signals
                intent.decision_rationale = decision_rationale

            applied.append(
                {
                    "intent_id": intent.intent_id,
                    "action": action,
                    "reason": reason,
                    "plan_strategy": intent.plan_strategy,
                    "plan_signals": list(intent.plan_signals),
                    "plan_rationale": intent.plan_rationale,
                    "decision_strategy": intent.decision_strategy,
                    "decision_signals": list(intent.decision_signals),
                    "decision_rationale": intent.decision_rationale,
                    "status_before": status_before,
                    "status_after": status_after,
                    "intent": intent.intent,
                    "kind": intent.kind,
                    "created_turn": intent.created_turn,
                    "earliest_turn": intent.earliest_turn,
                    "latest_turn": intent.latest_turn,
                    "hazard_profile": list(intent.hazard_profile),
                    "hazard_turn_prob": hazard_turn_probability(
                        intent.hazard_profile,
                        created_turn=intent.created_turn,
                        turn_index=self.turn_index,
                    ),
                    "priority": intent.priority,
                    "confidence": intent.confidence,
                    "revision_count": intent.revision_count,
                    "updated_intent": updated_intent_payload if isinstance(updated_intent_payload, dict) else None,
                }
            )

        self._log(
            "deferred_intent_decision",
            {
                "strategy": self.config.deferred_intent_strategy.value,
                "decisions": applied,
                "scheduler_raw": scheduler_raw_text,
                "prompt_schema_injected": scheduler_prompt_schema_injected,
                "scheduler_parse_ok": scheduler_parse_ok,
                "usage": scheduler_usage,
                "finish_reason": scheduler_finish_reason,
                "request_id": scheduler_request_id,
            },
        )
        return due_intents, applied

    def user_turn(self, user_text: str) -> dict[str, Any]:
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("user_text must not be empty")

        self.history.append(ChatMessage(role="user", content=user_text))
        self.turn_index += 1

        memory_capsule = self._probe_memory_capsule()
        conclusion = self._probe_conclusion()
        planned_delayed_mentions = self._probe_delayed_mentions()
        adaptive_delayed_mention_adjustments, adaptive_conclusion_hazard = (
            self._build_adaptive_hazard_trace()
        )
        due_delayed_mentions = self._due_delayed_mentions(
            adaptive_adjustments=adaptive_delayed_mention_adjustments
        )
        injected_delayed_mentions: list[DelayedMentionItem] = []
        delayed_mention_fire_prob = max(
            0.0, min(1.0, float(self.config.delayed_mention_fire_prob or 0.0))
        )
        delayed_mention_fire_max_items = max(0, int(self.config.delayed_mention_fire_max_items or 0))
        delayed_mention_injection_draws: list[dict[str, Any]] = []
        if (
            self.config.delayed_mention_mode == DelayedMentionMode.SOFT_FIRE
            and due_delayed_mentions
            and delayed_mention_fire_max_items > 0
            and delayed_mention_fire_prob > 0.0
        ):
            due_sorted = sorted(
                due_delayed_mentions,
                key=lambda item: (
                    -float(
                        self._delayed_mention_adjustment(
                            item,
                            adaptive_delayed_mention_adjustments,
                        ).get("effective_turn_prob", 0.0)
                        or 0.0
                    ),
                    item.latest_turn,
                    item.earliest_turn,
                    item.item_id,
                ),
            )
            for item in due_sorted:
                if len(injected_delayed_mentions) >= delayed_mention_fire_max_items:
                    break
                adjustment = self._delayed_mention_adjustment(
                    item,
                    adaptive_delayed_mention_adjustments,
                )
                hazard_turn_prob = clamp01(adjustment.get("base_turn_prob"), default=0.0)
                adaptive_hazard_turn_prob = clamp01(
                    adjustment.get("effective_turn_prob"),
                    default=hazard_turn_prob,
                )
                effective_prob = delayed_mention_fire_prob
                hazard_points = hazard_support_size(item.hazard_profile)
                if isinstance(adaptive_hazard_turn_prob, float):
                    effective_prob = min(
                        1.0,
                        delayed_mention_fire_prob
                        * max(1, hazard_points)
                        * adaptive_hazard_turn_prob,
                    )
                hazard_peak_prob = (
                    hazard_peak_probability(item.hazard_profile) if item.hazard_profile else None
                )
                draw = random.random()
                inject = draw < effective_prob
                delayed_mention_injection_draws.append(
                    {
                        "item_id": item.item_id,
                        "kind": item.kind,
                        "release_stage_role": item.release_stage_role,
                        "draw": draw,
                        "prob": effective_prob,
                        "base_prob": delayed_mention_fire_prob,
                        "hazard_profile": list(item.hazard_profile),
                        "hazard_turn_prob": hazard_turn_prob,
                        "adaptive_hazard_turn_prob": adaptive_hazard_turn_prob,
                        "adaptive_hazard_multiplier": adjustment.get("turn_prob_multiplier"),
                        "adaptive_hazard_threshold": adjustment.get("effective_threshold"),
                        "adaptive_hazard_threshold_shift": adjustment.get("threshold_shift"),
                        "adaptive_embedding_prepeak_penalty": adjustment.get(
                            "embedding_prepeak_penalty"
                        ),
                        "adaptive_embedding_prepeak_peak_gap_factor": adjustment.get(
                            "embedding_prepeak_peak_gap_factor"
                        ),
                        "adaptive_hazard_stage_policy": adjustment.get(
                            "adaptive_hazard_stage_policy"
                        ),
                        "adaptive_hazard_reasons": list(adjustment.get("reasons") or []),
                        "hazard_peak_prob": hazard_peak_prob,
                        "adaptive_peak_support_ratio": adjustment.get("peak_support_ratio"),
                        "adaptive_before_peak": adjustment.get("before_peak"),
                        "adaptive_at_peak": adjustment.get("at_peak"),
                        "hazard_support_size": hazard_points or None,
                        "inject": inject,
                    }
                )
                if inject:
                    injected_delayed_mentions.append(item)
        planned_intent: Optional[DeferredIntent] = None
        planned_intents_payload: list[dict[str, Any]] = []
        due_intents: list[DeferredIntent] = []
        deferred_actions: list[dict[str, Any]] = []
        ablation_actions: list[dict[str, Any]] = []
        suppressed_delayed_mentions = self._suppressed_delayed_mentions(
            injected_delayed_mentions=injected_delayed_mentions,
            adaptive_adjustments=adaptive_delayed_mention_adjustments,
        )
        inband_state: Optional[dict[str, Any]] = None
        inband_state_error: Optional[str] = None
        inband_state_chars: Optional[int] = None

        if self.config.deferred_intent_backend == DeferredIntentBackend.INBAND:
            prev_snapshot = {
                intent.intent_id: {
                    "status": intent.status,
                    "revision_count": intent.revision_count,
                    "intent": intent.intent,
                }
                for intent in self.deferred_intents
            }
            expected_next_intent_id = f"di-{self.next_deferred_intent_index:04d}"
            active_count = len(self._active_deferred_intents())
            plan_capacity = max(0, self.config.deferred_intent_limit - active_count)
            plan_capacity = min(plan_capacity, max(0, int(self.config.deferred_intent_plan_max_new or 0)))
            eligible_plan_turn = False
            if self.config.deferred_intent_plan_policy == DeferredIntentPlanPolicy.PERIODIC:
                eligible_plan_turn = (
                    self.config.deferred_intent_every > 0
                    and self.turn_index != 0
                    and self.turn_index % self.config.deferred_intent_every == 0
                    and plan_capacity > 0
                )
            elif self.config.deferred_intent_plan_policy == DeferredIntentPlanPolicy.AUTO:
                eligible_plan_turn = self.turn_index != 0 and plan_capacity > 0

            if (
                eligible_plan_turn
                and self.config.deferred_intent_plan_budget > 0
                and self.deferred_intent_plan_probe_calls >= self.config.deferred_intent_plan_budget
            ):
                eligible_plan_turn = False

            if eligible_plan_turn:
                self.deferred_intent_plan_probe_calls += 1

            base_prompt = self._build_system_prompt(
                due_intents=None,
                injected_delayed_mentions=injected_delayed_mentions,
                suppressed_delayed_mentions=suppressed_delayed_mentions,
                adaptive_delayed_mention_adjustments=adaptive_delayed_mention_adjustments,
            )
            prev_assistant = next(
                (m for m in reversed(self.history[:-1]) if m.role == "assistant"),
                None,
            )
            prev_state_ok = False
            if prev_assistant is not None:
                _, prev_state, prev_state_error = split_rcl_state(prev_assistant.content)
                prev_state_ok = (
                    prev_state_error is None
                    and isinstance(prev_state, dict)
                    and isinstance(prev_state.get("intents"), list)
                )
            inject_schema = not prev_state_ok

            inband_instructions = textwrap.dedent(
                f"""\
                Deferred-intent in-band state (private; never reveal to the user):
                - After the user-visible reply, append {RCL_STATE_OPEN}{{...}}{RCL_STATE_CLOSE} with strict JSON (no code fences).
                - Use the most recent {RCL_STATE_OPEN}...{RCL_STATE_CLOSE} in the conversation as the previous state. If none, start with {{"version": 1, "intents": []}}.
                - Current turn index: {self.turn_index}
                - next_intent_id: {expected_next_intent_id}
                - eligible_plan_turn_this_turn: {eligible_plan_turn}

                Config:
                - deferred_intent_every: {self.config.deferred_intent_every}
                - deferred_intent_mode: {self.config.deferred_intent_mode.value}
                - deferred_intent_strategy: {self.config.deferred_intent_strategy.value}
                - deferred_intent_timing: {self.config.deferred_intent_timing.value}
                - deferred_intent_offset: {self.config.deferred_intent_offset}
                - deferred_intent_grace: {self.config.deferred_intent_grace}
                - deferred_intent_limit: {self.config.deferred_intent_limit}
                - deferred_intent_plan_policy: {self.config.deferred_intent_plan_policy.value}
                - deferred_intent_plan_budget: {self.config.deferred_intent_plan_budget}
                - deferred_intent_plan_probe_calls_used: {self.deferred_intent_plan_probe_calls}
                - deferred_intent_plan_max_new: {self.config.deferred_intent_plan_max_new}
                - planning_capacity_this_turn: {plan_capacity}

                Rules:
                - Never mention the existence of {RCL_STATE_OPEN} or its contents.
                - Keep at most deferred_intent_limit intents in state. Prefer keeping active ones; drop older terminal ones first.
                - Consider creating up to planning_capacity_this_turn new intents only when eligible_plan_turn_this_turn is true.
                - If you create new intents, assign intent_id values starting at next_intent_id and incrementing (di-0001, di-0002, ...).
                - When you create a new intent, fill plan_strategy/plan_signals/plan_rationale (brief; no step-by-step reasoning).
                - When you change an intent's status (fire/cancel/expire/revise), fill decision_strategy/decision_signals/decision_rationale (brief; no step-by-step reasoning).
                - Update timing/status each turn:
                  - If turn_index < earliest_turn: keep status=active.
                  - If turn_index > latest_turn: set status=expired and set terminal_reason.
                  - If earliest_turn <= turn_index <= latest_turn:
                    - fixed: fire exactly at earliest_turn (latest_turn == earliest_turn).
                    - trigger/adaptive: decide fire/cancel/hold based on triggers/cancel_if and dialogue relevance.
                    - adaptive may revise wording/timing/conditions (increment revision_count) instead of firing.
                  - If deferred_intent_timing is 'hazard', include hazard_profile as [{"delay_turns": N, "prob": P}, ...].
                  - If you set status=fired, set fire_turn={self.turn_index} and set terminal_reason.
                - If deferred_intent_mode is 'soft_fire', integrate fired intents naturally into the visible reply.
                - If deferred_intent_mode is 'hard_fire', include every fired intent explicitly in the visible reply.
                - If deferred_intent_mode is 'observe', do NOT include fired intent content in the visible reply.
                """
            ).strip()
            if inject_schema:
                inband_instructions += "\n\n" + textwrap.dedent(
                    """\
                    State JSON schema:
                    {
                      "version": 1,
                      "intents": [
                        {
                          "intent_id": "di-0001",
                          "created_turn": 3,
                          "kind": "summary|proposal|correction|question|reminder|other",
                          "intent": "what should be said later",
                          "why_not_now": "...",
                          "earliest_turn": 6,
                          "latest_turn": 8,
                          "hazard_profile": [{"delay_turns": 3, "prob": 0.6}],
                          "trigger": ["..."],
                          "cancel_if": ["..."],
                          "confidence": 0.0,
                          "priority": 0.0,
                          "plan_strategy": "",
                          "plan_signals": [],
                          "plan_rationale": "",
                          "decision_strategy": "",
                          "decision_signals": [],
                          "decision_rationale": "",
                          "status": "active|fired|canceled|expired",
                          "revision_count": 0,
                          "fire_turn": null,
                          "terminal_reason": ""
                        }
                      ]
                    }
                    """
                ).strip()

            system_prompt = (f"{base_prompt}\n\n" if base_prompt else "") + inband_instructions

            messages = self._recent_window()
            if self.config.recent_window_messages == 1:
                if prev_assistant is not None:
                    messages = [prev_assistant, self.history[-1]]

            reply = self.adapter.generate(
                system=system_prompt or None,
                messages=messages,
                config=self.config.reply_config,
            )

            assistant_raw = reply.text.strip()
            assistant_text, state, state_error = split_rcl_state(assistant_raw)
            assistant_text = assistant_text.strip()
            if state is None and state_error is None:
                inband_state_error = "Missing RCL_STATE block."
            else:
                inband_state_error = state_error

            state_obj = state if isinstance(state, dict) else {}
            version = coerce_int(state_obj.get("version") or state_obj.get("v")) or 1
            intents_raw = state_obj.get("intents")
            if intents_raw is None:
                intents_raw = state_obj.get("deferred_intents")
            state_has_intents = isinstance(intents_raw, list)
            if not state_has_intents:
                intents_raw = []
                if state is not None and not inband_state_error:
                    inband_state_error = "RCL_STATE missing 'intents' list."

            parsed_intents: list[DeferredIntent] = []
            for item in intents_raw:
                if not isinstance(item, dict):
                    continue
                intent_id = compact_text(str(item.get("intent_id", "")))
                intent_text = compact_text(str(item.get("intent", "")))
                if not intent_id or not intent_text:
                    continue

                created_turn = coerce_int(item.get("created_turn")) or self.turn_index
                kind = compact_text(str(item.get("kind", "other"))) or "other"
                why_not_now = compact_text(str(item.get("why_not_now", "")))
                earliest_turn = coerce_int(item.get("earliest_turn"))
                latest_turn = coerce_int(item.get("latest_turn"))
                if earliest_turn is None:
                    earliest_turn = created_turn + max(1, self.config.deferred_intent_offset)
                if latest_turn is None:
                    if self.config.deferred_intent_strategy == DeferredIntentStrategy.FIXED:
                        latest_turn = earliest_turn
                    else:
                        latest_turn = earliest_turn + max(0, self.config.deferred_intent_grace)
                if latest_turn < earliest_turn:
                    latest_turn = earliest_turn
                hazard_profile = parse_hazard_profile(item.get("hazard_profile"))
                if self.config.deferred_intent_timing == DeferredIntentTiming.HAZARD:
                    hazard_profile = ensure_hazard_profile(
                        hazard_profile,
                        created_turn=created_turn,
                        earliest_turn=earliest_turn,
                        latest_turn=latest_turn,
                        offset=self.config.deferred_intent_offset,
                        grace=self.config.deferred_intent_grace,
                    )
                    earliest_turn, latest_turn = hazard_bounds_from_profile(
                        created_turn=created_turn,
                        profile=hazard_profile,
                    )

                status = compact_text(str(item.get("status", "active"))).lower() or "active"
                if status not in {"active", "fired", "canceled", "expired"}:
                    status = "active"

                selection = item.get("selection") if isinstance(item.get("selection"), dict) else {}
                decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}

                parsed_intents.append(
                    DeferredIntent(
                        intent_id=intent_id,
                        created_turn=created_turn,
                        kind=kind,
                        intent=intent_text,
                        why_not_now=why_not_now,
                        earliest_turn=earliest_turn,
                        latest_turn=latest_turn,
                        hazard_profile=hazard_profile,
                        trigger=coerce_str_list(item.get("trigger")),
                        cancel_if=coerce_str_list(item.get("cancel_if")),
                        confidence=clamp01(item.get("confidence"), default=0.0),
                        priority=clamp01(item.get("priority"), default=0.5),
                        plan_strategy=compact_text(
                            str(
                                selection.get("strategy")
                                or item.get("plan_strategy")
                                or item.get("selection_strategy")
                                or ""
                            )
                        ),
                        plan_rationale=compact_text(
                            str(
                                selection.get("rationale")
                                or item.get("plan_rationale")
                                or item.get("selection_rationale")
                                or ""
                            )
                        ),
                        plan_signals=coerce_str_list(
                            selection.get("signals")
                            or item.get("plan_signals")
                            or item.get("selection_signals")
                        ),
                        decision_strategy=compact_text(
                            str(
                                decision.get("strategy")
                                or item.get("decision_strategy")
                                or ""
                            )
                        ),
                        decision_rationale=compact_text(
                            str(
                                decision.get("rationale")
                                or item.get("decision_rationale")
                                or ""
                            )
                        ),
                        decision_signals=coerce_str_list(
                            decision.get("signals")
                            or item.get("decision_signals")
                        ),
                        status=status,
                        revision_count=coerce_int(item.get("revision_count")) or 0,
                        fire_turn=coerce_int(item.get("fire_turn")),
                        terminal_reason=compact_text(str(item.get("terminal_reason", ""))),
                    )
                )

            if state is None or not state_has_intents:
                parsed_intents = list(self._active_deferred_intents())

            # Deduplicate by id (keep last).
            dedup: dict[str, DeferredIntent] = {}
            for intent in parsed_intents:
                dedup[intent.intent_id] = intent
            parsed_intents = list(dedup.values())

            dropped_new_intent_ids: list[str] = []
            kept_new_ids: set[str] = set()
            new_candidates = [i for i in parsed_intents if i.intent_id not in prev_snapshot]
            if not eligible_plan_turn:
                if new_candidates:
                    dropped_new_intent_ids = sorted(i.intent_id for i in new_candidates)
                    dropped_set = set(dropped_new_intent_ids)
                    parsed_intents = [i for i in parsed_intents if i.intent_id not in dropped_set]
                new_candidates = []
            else:

                def intent_num(intent: DeferredIntent) -> int:
                    match = re.fullmatch(r"di-(\d+)", intent.intent_id)
                    return int(match.group(1)) if match else 10**9

                if len(new_candidates) > plan_capacity:
                    kept = sorted(new_candidates, key=intent_num)[:plan_capacity]
                    kept_new_ids = {i.intent_id for i in kept}
                    dropped_new_intent_ids = sorted(
                        i.intent_id for i in new_candidates if i.intent_id not in kept_new_ids
                    )
                    dropped_set = set(dropped_new_intent_ids)
                    parsed_intents = [i for i in parsed_intents if i.intent_id not in dropped_set]
                    new_candidates = kept
                else:
                    kept_new_ids = {i.intent_id for i in new_candidates}

                default_earliest_turn = self.turn_index + max(1, self.config.deferred_intent_offset)
                default_latest_turn = (
                    default_earliest_turn
                    if self.config.deferred_intent_strategy == DeferredIntentStrategy.FIXED
                    else default_earliest_turn + max(0, self.config.deferred_intent_grace)
                )

                for intent in new_candidates:
                    intent.created_turn = self.turn_index
                    intent.status = "active"
                    intent.revision_count = 0
                    intent.fire_turn = None
                    intent.terminal_reason = ""
                    intent.decision_strategy = ""
                    intent.decision_signals = []
                    intent.decision_rationale = ""
                    if self.config.deferred_intent_strategy == DeferredIntentStrategy.FIXED:
                        intent.trigger = []
                        intent.cancel_if = []
                    if self.config.deferred_intent_timing == DeferredIntentTiming.OFFSET:
                        intent.earliest_turn = default_earliest_turn
                        intent.latest_turn = default_latest_turn
                        intent.hazard_profile = []
                    elif self.config.deferred_intent_timing == DeferredIntentTiming.HAZARD:
                        intent.hazard_profile = ensure_hazard_profile(
                            intent.hazard_profile,
                            created_turn=intent.created_turn,
                            earliest_turn=intent.earliest_turn,
                            latest_turn=intent.latest_turn,
                            offset=self.config.deferred_intent_offset,
                            grace=self.config.deferred_intent_grace,
                        )
                        intent.earliest_turn, intent.latest_turn = hazard_bounds_from_profile(
                            created_turn=intent.created_turn,
                            profile=intent.hazard_profile,
                        )
                    else:
                        if intent.earliest_turn <= self.turn_index:
                            intent.earliest_turn = self.turn_index + 1
                        if intent.latest_turn < intent.earliest_turn:
                            intent.latest_turn = intent.earliest_turn
                        if self.config.deferred_intent_strategy == DeferredIntentStrategy.FIXED:
                            intent.latest_turn = intent.earliest_turn
                        intent.hazard_profile = []

            # Enforce a hard cap after applying plan policy/max-new.
            if len(parsed_intents) > self.config.deferred_intent_limit:

                def sort_key(intent: DeferredIntent) -> tuple[int, int, float]:
                    status_rank = 0 if intent.status == "active" else 1
                    return (status_rank, -intent.created_turn, -intent.priority)

                parsed_intents = sorted(parsed_intents, key=sort_key)[: self.config.deferred_intent_limit]

            planned_intents = [i for i in parsed_intents if i.intent_id in kept_new_ids]
            planned_intents = sorted(
                planned_intents,
                key=lambda intent: (intent.created_turn, intent.priority),
                reverse=True,
            )
            if planned_intents:
                planned_intent = planned_intents[0]

            if eligible_plan_turn or planned_intents or dropped_new_intent_ids:
                self._log(
                    "deferred_intent_plan",
                    {
                        "intent_id": planned_intent.intent_id if planned_intent else None,
                        "created": bool(planned_intents),
                        "intent": planned_intent.to_dict() if planned_intent else None,
                        "created_intents": [intent.to_dict() for intent in planned_intents],
                        "raw_plan": {
                            "backend": "inband",
                            "eligible": eligible_plan_turn,
                            "plan_capacity": plan_capacity,
                            "plan_policy": self.config.deferred_intent_plan_policy.value,
                            "plan_budget": self.config.deferred_intent_plan_budget,
                            "plan_probe_calls_used": self.deferred_intent_plan_probe_calls,
                            "dropped_new_intent_ids": dropped_new_intent_ids or None,
                        },
                        "plan_policy": self.config.deferred_intent_plan_policy.value,
                        "plan_budget": self.config.deferred_intent_plan_budget,
                        "plan_probe_calls_used": self.deferred_intent_plan_probe_calls,
                        "usage": None,
                        "finish_reason": None,
                        "request_id": None,
                        "planning_capacity": plan_capacity,
                    },
                )

            if self.config.deferred_intent_ablation == DeferredIntentAblation.DELETE_PLANNED:
                pruned_planned_ids: list[str] = []
                for candidate in planned_intents:
                    status_before = candidate.status
                    candidate.status = "canceled"
                    candidate.terminal_reason = (
                        "Ablation: deleted planned intent immediately (in-band backend)."
                    )
                    status_after = candidate.status
                    action = {
                        "intent_id": candidate.intent_id,
                        "action": "cancel",
                        "reason": candidate.terminal_reason,
                        "plan_strategy": candidate.plan_strategy,
                        "plan_signals": list(candidate.plan_signals),
                        "plan_rationale": candidate.plan_rationale,
                        "decision_strategy": candidate.decision_strategy,
                        "decision_signals": list(candidate.decision_signals),
                        "decision_rationale": candidate.decision_rationale,
                        "status_before": status_before,
                        "status_after": status_after,
                        "intent": candidate.intent,
                        "kind": candidate.kind,
                        "created_turn": candidate.created_turn,
                        "earliest_turn": candidate.earliest_turn,
                        "latest_turn": candidate.latest_turn,
                        "priority": candidate.priority,
                        "confidence": candidate.confidence,
                        "revision_count": candidate.revision_count,
                        "updated_intent": None,
                        "ablated": True,
                    }
                    ablation_actions.append(action)
                    pruned_planned_ids.append(candidate.intent_id)
                    self._log(
                        "deferred_intent_ablation",
                        {
                            "mode": self.config.deferred_intent_ablation.value,
                            "intent_id": candidate.intent_id,
                            "status_before": status_before,
                            "status_after": status_after,
                            "reason": candidate.terminal_reason,
                        },
                    )
                if pruned_planned_ids:
                    # Do not carry the planned intents forward in-band.
                    pruned_id_set = set(pruned_planned_ids)
                    parsed_intents = [
                        intent
                        for intent in parsed_intents
                        if intent.intent_id not in pruned_id_set
                    ]

            planned_intents_payload = [intent.to_dict() for intent in planned_intents]

            numeric_ids = []
            for intent in parsed_intents + planned_intents:
                match = re.fullmatch(r"di-(\d+)", intent.intent_id)
                if match:
                    numeric_ids.append(int(match.group(1)))
            if numeric_ids:
                self.next_deferred_intent_index = max(
                    self.next_deferred_intent_index, max(numeric_ids) + 1
                )

            # Derive decisions from status transitions.
            decisions: list[dict[str, Any]] = []
            for intent in parsed_intents:
                prev = prev_snapshot.get(intent.intent_id)
                status_before = str(prev.get("status")) if prev is not None else "active"
                action = "hold"
                updated_intent = None
                if status_before != intent.status:
                    if intent.status == "fired":
                        action = "fire"
                    elif intent.status == "canceled":
                        action = "cancel"
                    elif intent.status == "expired":
                        action = "expire"
                    else:
                        action = "revise"
                        updated_intent = intent.to_dict()
                else:
                    if prev is not None and intent.revision_count > int(prev.get("revision_count") or 0):
                        action = "revise"
                        updated_intent = intent.to_dict()

                if action == "fire":
                    due_intents.append(intent)

                if action in {"fire", "cancel", "expire"}:
                    reason = intent.terminal_reason or "In-band terminal transition."
                elif action == "revise":
                    reason = "In-band revision."
                else:
                    reason = "In-band hold."

                decisions.append(
                    {
                        "intent_id": intent.intent_id,
                        "action": action,
                        "reason": reason,
                        "plan_strategy": intent.plan_strategy,
                        "plan_signals": list(intent.plan_signals),
                        "plan_rationale": intent.plan_rationale,
                        "decision_strategy": intent.decision_strategy,
                        "decision_signals": list(intent.decision_signals),
                        "decision_rationale": intent.decision_rationale,
                        "status_before": status_before,
                        "status_after": intent.status,
                        "intent": intent.intent,
                        "kind": intent.kind,
                        "created_turn": intent.created_turn,
                        "earliest_turn": intent.earliest_turn,
                        "latest_turn": intent.latest_turn,
                        "hazard_profile": list(intent.hazard_profile),
                        "hazard_turn_prob": hazard_turn_probability(
                            intent.hazard_profile,
                            created_turn=intent.created_turn,
                            turn_index=self.turn_index,
                        ),
                        "priority": intent.priority,
                        "confidence": intent.confidence,
                        "revision_count": intent.revision_count,
                        "updated_intent": updated_intent,
                    }
                )

            if decisions:
                self._log(
                    "deferred_intent_decision",
                    {
                        "strategy": self.config.deferred_intent_strategy.value,
                        "decisions": decisions,
                        "scheduler_raw": "inband",
                        "usage": None,
                        "finish_reason": None,
                        "request_id": None,
                    },
                )

            deferred_actions = decisions + ablation_actions

            # Mirror the latest in-band state locally for analysis (best-effort).
            by_id = {intent.intent_id: intent for intent in self.deferred_intents}
            for intent in parsed_intents:
                if intent.intent_id in by_id:
                    existing = by_id[intent.intent_id]
                    existing.kind = intent.kind
                    existing.intent = intent.intent
                    existing.why_not_now = intent.why_not_now
                    existing.earliest_turn = intent.earliest_turn
                    existing.latest_turn = intent.latest_turn
                    existing.trigger = intent.trigger
                    existing.cancel_if = intent.cancel_if
                    existing.confidence = intent.confidence
                    existing.priority = intent.priority
                    existing.plan_strategy = intent.plan_strategy
                    existing.plan_signals = list(intent.plan_signals)
                    existing.plan_rationale = intent.plan_rationale
                    existing.decision_strategy = intent.decision_strategy
                    existing.decision_signals = list(intent.decision_signals)
                    existing.decision_rationale = intent.decision_rationale
                    existing.status = intent.status
                    existing.revision_count = intent.revision_count
                    existing.fire_turn = intent.fire_turn
                    existing.terminal_reason = intent.terminal_reason
                else:
                    self.deferred_intents.append(intent)

            parsed_ids = {intent.intent_id for intent in parsed_intents}
            for candidate in planned_intents:
                if candidate.intent_id in parsed_ids:
                    continue
                if candidate.intent_id in by_id:
                    continue
                self.deferred_intents.append(candidate)

            carried_intents, inband_state, state_blob = build_inband_state_payload(
                parsed_intents,
                version=version,
                max_chars=INBAND_STATE_MAX_CHARS,
            )
            inband_state_chars = len(state_blob)
            if len(carried_intents) != len(parsed_intents):
                carried_ids = {intent.intent_id for intent in carried_intents}
                pruned_ids = sorted(
                    {
                        intent.intent_id
                        for intent in parsed_intents
                        if intent.intent_id not in carried_ids
                    }
                )
                if pruned_ids:
                    for intent in self.deferred_intents:
                        if intent.intent_id in pruned_ids and intent.status == "active":
                            intent.status = "expired"
                            intent.terminal_reason = "Dropped by in-band state size limit."
                    self._log(
                        "inband_state_prune",
                        {
                            "max_chars": INBAND_STATE_MAX_CHARS,
                            "before_count": len(parsed_intents),
                            "after_count": len(carried_intents),
                            "pruned_intent_ids": pruned_ids,
                            "state_chars": inband_state_chars,
                        },
                    )
            assistant_with_state = assistant_text
            if assistant_with_state:
                assistant_with_state = assistant_with_state.rstrip() + "\n\n"
            assistant_with_state += f"{RCL_STATE_OPEN}{state_blob}{RCL_STATE_CLOSE}"
            self.history.append(ChatMessage(role="assistant", content=assistant_with_state))
        else:
            planned_intents = self._probe_deferred_intent_plans()
            planned_intent = planned_intents[0] if planned_intents else None
            if (
                planned_intents
                and self.config.deferred_intent_ablation
                == DeferredIntentAblation.DELETE_PLANNED
            ):
                for item in planned_intents:
                    status_before = item.status
                    item.status = "canceled"
                    item.terminal_reason = "Ablation: deleted planned intent immediately."
                    status_after = item.status
                    action = {
                        "intent_id": item.intent_id,
                        "action": "cancel",
                        "reason": item.terminal_reason,
                        "plan_strategy": item.plan_strategy,
                        "plan_signals": list(item.plan_signals),
                        "plan_rationale": item.plan_rationale,
                        "decision_strategy": item.decision_strategy,
                        "decision_signals": list(item.decision_signals),
                        "decision_rationale": item.decision_rationale,
                        "status_before": status_before,
                        "status_after": status_after,
                        "intent": item.intent,
                        "kind": item.kind,
                        "created_turn": item.created_turn,
                        "earliest_turn": item.earliest_turn,
                        "latest_turn": item.latest_turn,
                        "priority": item.priority,
                        "confidence": item.confidence,
                        "revision_count": item.revision_count,
                        "updated_intent": None,
                        "ablated": True,
                    }
                    ablation_actions.append(action)
                    self._log(
                        "deferred_intent_ablation",
                        {
                            "mode": self.config.deferred_intent_ablation.value,
                            "intent_id": item.intent_id,
                            "status_before": status_before,
                            "status_after": status_after,
                            "reason": item.terminal_reason,
                        },
                    )

            planned_intents_payload = [item.to_dict() for item in planned_intents]

            due_intents, deferred_actions = self._schedule_deferred_intents()
            if ablation_actions:
                deferred_actions = list(deferred_actions) + ablation_actions

            system_prompt = self._build_system_prompt(
                due_intents=due_intents,
                injected_delayed_mentions=injected_delayed_mentions,
                suppressed_delayed_mentions=suppressed_delayed_mentions,
                adaptive_delayed_mention_adjustments=adaptive_delayed_mention_adjustments,
            )
            reply = self.adapter.generate(
                system=system_prompt or None,
                messages=self._recent_window(),
                config=self.config.reply_config,
            )

            assistant_text = reply.text.strip()
            self.history.append(ChatMessage(role="assistant", content=assistant_text))

        overlap = 0.0
        if conclusion:
            overlap = lexical_overlap(conclusion, assistant_text)

        latest_conclusion_line = self.latest_conclusion_line or ""
        latest_conclusion_line_text = strip_conclusion_prefix(latest_conclusion_line)
        latest_conclusion_line_reply_overlap: Optional[float] = None
        latest_conclusion_line_mentioned: Optional[bool] = None
        latest_conclusion_line_age_turns: Optional[int] = None
        if latest_conclusion_line_text:
            latest_conclusion_line_reply_overlap = lexical_overlap(
                latest_conclusion_line_text, assistant_text
            )
            latest_conclusion_line_mentioned = (
                latest_conclusion_line_reply_overlap >= self.CONCLUSION_MENTION_THRESHOLD
            )
            if self.latest_conclusion_probe_turn is not None:
                latest_conclusion_line_age_turns = (
                    self.turn_index - self.latest_conclusion_probe_turn
                )

        latest_conclusion_keywords = list(self.latest_conclusion_keywords or [])
        latest_conclusion_keyword_hits: Optional[int] = None
        latest_conclusion_keyword_coverage: Optional[float] = None
        latest_conclusion_keyword_mentioned: Optional[bool] = None
        if latest_conclusion_keywords:
            latest_conclusion_keyword_hits = keyword_hits(
                latest_conclusion_keywords, assistant_text
            )
            latest_conclusion_keyword_coverage = keyword_coverage(
                latest_conclusion_keywords, assistant_text
            )
            required_hits = max(
                self.CONCLUSION_KEYWORD_MENTION_MIN_HITS,
                (len(latest_conclusion_keywords) + 1) // 2,
            )
            latest_conclusion_keyword_mentioned = (
                latest_conclusion_keyword_hits is not None
                and latest_conclusion_keyword_hits >= required_hits
            )

        latest_conclusion_mentioned_any: Optional[bool] = None
        if latest_conclusion_line_mentioned is not None or latest_conclusion_keyword_mentioned is not None:
            latest_conclusion_mentioned_any = bool(
                latest_conclusion_line_mentioned or latest_conclusion_keyword_mentioned
            )

        latest_conclusion_plan_earliest_turn: Optional[int] = None
        latest_conclusion_plan_latest_turn: Optional[int] = None
        if self.latest_conclusion_probe_turn is not None:
            if self.latest_conclusion_mention_delay_min_turns is not None:
                latest_conclusion_plan_earliest_turn = (
                    self.latest_conclusion_probe_turn
                    + self.latest_conclusion_mention_delay_min_turns
                )
            if self.latest_conclusion_mention_delay_max_turns is not None:
                latest_conclusion_plan_latest_turn = (
                    self.latest_conclusion_probe_turn
                    + self.latest_conclusion_mention_delay_max_turns
                )
        latest_conclusion_plan_hazard_turn_prob: Optional[float] = None
        if self.latest_conclusion_probe_turn is not None and self.latest_conclusion_mention_hazard_profile:
            latest_conclusion_plan_hazard_turn_prob = hazard_turn_probability(
                self.latest_conclusion_mention_hazard_profile,
                created_turn=self.latest_conclusion_probe_turn,
                turn_index=self.turn_index,
            )
        latest_conclusion_plan_adaptive_hazard_turn_prob: Optional[float] = None
        latest_conclusion_plan_adaptive_multiplier: Optional[float] = None
        latest_conclusion_plan_adaptive_threshold: Optional[float] = None
        latest_conclusion_plan_adaptive_threshold_shift: Optional[float] = None
        latest_conclusion_plan_adaptive_reasons: Optional[list[str]] = None
        if isinstance(adaptive_conclusion_hazard, dict):
            latest_conclusion_plan_adaptive_hazard_turn_prob = clamp01(
                adaptive_conclusion_hazard.get("effective_turn_prob"),
                default=latest_conclusion_plan_hazard_turn_prob or 0.0,
            )
            multiplier = adaptive_conclusion_hazard.get("turn_prob_multiplier")
            if isinstance(multiplier, (int, float)):
                latest_conclusion_plan_adaptive_multiplier = float(multiplier)
            adaptive_threshold = adaptive_conclusion_hazard.get("effective_threshold")
            if isinstance(adaptive_threshold, (int, float)):
                latest_conclusion_plan_adaptive_threshold = float(adaptive_threshold)
            threshold_shift = adaptive_conclusion_hazard.get("threshold_shift")
            if isinstance(threshold_shift, (int, float)):
                latest_conclusion_plan_adaptive_threshold_shift = float(threshold_shift)
            reasons = adaptive_conclusion_hazard.get("reasons") or []
            if isinstance(reasons, list):
                latest_conclusion_plan_adaptive_reasons = [
                    str(reason) for reason in reasons if reason
                ]

        latent_convergence = None
        if self.config.semantic_judge_backend in {
            SemanticJudgeBackend.LLM,
            SemanticJudgeBackend.BOTH,
        }:
            latent_convergence = self._probe_latent_convergence(
                assistant_text=assistant_text,
                planned_earliest_turn=latest_conclusion_plan_earliest_turn,
                planned_latest_turn=latest_conclusion_plan_latest_turn,
                explicit_mention_present=latest_conclusion_mentioned_any,
            )
        embedding_convergence = self._probe_embedding_convergence(
            assistant_text=assistant_text,
            planned_earliest_turn=latest_conclusion_plan_earliest_turn,
            planned_latest_turn=latest_conclusion_plan_latest_turn,
            explicit_mention_present=latest_conclusion_mentioned_any,
        )
        semantic_judge_alignment_gap: Optional[float] = None
        if (
            isinstance(latent_convergence, dict)
            and isinstance(embedding_convergence, dict)
            and isinstance(latent_convergence.get("alignment"), (int, float))
            and isinstance(embedding_convergence.get("alignment"), (int, float))
        ):
            semantic_judge_alignment_gap = abs(
                float(latent_convergence.get("alignment"))
                - float(embedding_convergence.get("alignment"))
            )

        injected_delayed_mention_ids = {item.item_id for item in injected_delayed_mentions}
        delayed_mention_actions: list[dict[str, Any]] = []
        for item in self.delayed_mentions:
            if item.status != "active":
                continue
            overlap_score = lexical_overlap(item.text, assistant_text)
            hits = keyword_hits(item.keywords, assistant_text) if item.keywords else None
            adjustment = self._delayed_mention_adjustment(
                item,
                adaptive_delayed_mention_adjustments,
            )
            hazard_turn_prob = clamp01(adjustment.get("base_turn_prob"), default=0.0)
            adaptive_hazard_turn_prob = clamp01(
                adjustment.get("effective_turn_prob"),
                default=hazard_turn_prob,
            )
            adaptive_hazard_multiplier = adjustment.get("turn_prob_multiplier")
            adaptive_hazard_threshold = adjustment.get("effective_threshold")
            adaptive_hazard_threshold_shift = adjustment.get("threshold_shift")
            adaptive_hazard_reasons = list(adjustment.get("reasons") or [])
            required_hits: Optional[int] = None
            keyword_mentioned = False
            if item.keywords:
                required_hits = max(
                    self.CONCLUSION_KEYWORD_MENTION_MIN_HITS,
                    (len(item.keywords) + 1) // 2,
                )
                keyword_mentioned = (hits or 0) >= required_hits
            mentioned = (
                overlap_score >= self.CONCLUSION_MENTION_THRESHOLD or keyword_mentioned
            )
            if mentioned:
                item.status = "mentioned"
                item.mention_turn = self.turn_index
                item.terminal_reason = "Mentioned in assistant reply."
                action = {
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "release_stage_role": item.release_stage_role,
                    "action": "mention",
                    "status_before": "active",
                    "status_after": item.status,
                    "text": item.text,
                    "created_turn": item.created_turn,
                    "earliest_turn": item.earliest_turn,
                    "latest_turn": item.latest_turn,
                    "hazard_profile": list(item.hazard_profile),
                    "hazard_turn_prob": hazard_turn_prob,
                    "adaptive_hazard_turn_prob": adaptive_hazard_turn_prob,
                    "adaptive_hazard_multiplier": adaptive_hazard_multiplier,
                    "adaptive_hazard_threshold": adaptive_hazard_threshold,
                    "adaptive_hazard_threshold_shift": adaptive_hazard_threshold_shift,
                    "adaptive_embedding_prepeak_penalty": adjustment.get(
                        "embedding_prepeak_penalty"
                    ),
                    "adaptive_embedding_prepeak_peak_gap_factor": adjustment.get(
                        "embedding_prepeak_peak_gap_factor"
                    ),
                    "adaptive_hazard_stage_policy": adjustment.get(
                        "adaptive_hazard_stage_policy"
                    ),
                    "adaptive_stage_boost_bias": adjustment.get("stage_boost_bias"),
                    "adaptive_stage_suppress_bias": adjustment.get("stage_suppress_bias"),
                    "adaptive_stage_threshold_bias": adjustment.get("stage_threshold_bias"),
                    "adaptive_hazard_reasons": adaptive_hazard_reasons,
                    "hazard_peak_prob": (
                        hazard_peak_probability(item.hazard_profile)
                        if item.hazard_profile
                        else None
                    ),
                    "adaptive_peak_support_ratio": adjustment.get("peak_support_ratio"),
                    "adaptive_before_peak": adjustment.get("before_peak"),
                    "adaptive_at_peak": adjustment.get("at_peak"),
                    "mention_turn": item.mention_turn,
                    "delay_turns": (item.mention_turn or 0) - item.created_turn,
                    "within_window": item.earliest_turn <= self.turn_index <= item.latest_turn,
                    "assistant_overlap": overlap_score,
                    "keyword_hits": hits,
                    "keyword_required_hits": required_hits,
                    "injected": item.item_id in injected_delayed_mention_ids,
                }
                delayed_mention_actions.append(action)
                self._log("delayed_mention_action", dict(action))
            elif self.turn_index > item.latest_turn:
                item.status = "expired"
                item.terminal_reason = "Mention window passed."
                action = {
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "release_stage_role": item.release_stage_role,
                    "action": "expire",
                    "status_before": "active",
                    "status_after": item.status,
                    "text": item.text,
                    "created_turn": item.created_turn,
                    "earliest_turn": item.earliest_turn,
                    "latest_turn": item.latest_turn,
                    "hazard_profile": list(item.hazard_profile),
                    "hazard_turn_prob": hazard_turn_prob,
                    "adaptive_hazard_turn_prob": adaptive_hazard_turn_prob,
                    "adaptive_hazard_multiplier": adaptive_hazard_multiplier,
                    "adaptive_hazard_threshold": adaptive_hazard_threshold,
                    "adaptive_hazard_threshold_shift": adaptive_hazard_threshold_shift,
                    "adaptive_embedding_prepeak_penalty": adjustment.get(
                        "embedding_prepeak_penalty"
                    ),
                    "adaptive_embedding_prepeak_peak_gap_factor": adjustment.get(
                        "embedding_prepeak_peak_gap_factor"
                    ),
                    "adaptive_hazard_stage_policy": adjustment.get(
                        "adaptive_hazard_stage_policy"
                    ),
                    "adaptive_stage_boost_bias": adjustment.get("stage_boost_bias"),
                    "adaptive_stage_suppress_bias": adjustment.get("stage_suppress_bias"),
                    "adaptive_stage_threshold_bias": adjustment.get("stage_threshold_bias"),
                    "adaptive_hazard_reasons": adaptive_hazard_reasons,
                    "hazard_peak_prob": (
                        hazard_peak_probability(item.hazard_profile)
                        if item.hazard_profile
                        else None
                    ),
                    "adaptive_peak_support_ratio": adjustment.get("peak_support_ratio"),
                    "adaptive_before_peak": adjustment.get("before_peak"),
                    "adaptive_at_peak": adjustment.get("at_peak"),
                    "mention_turn": None,
                    "delay_turns": None,
                    "within_window": False,
                    "assistant_overlap": overlap_score,
                    "keyword_hits": hits,
                    "keyword_required_hits": required_hits,
                    "injected": item.item_id in injected_delayed_mention_ids,
                }
                delayed_mention_actions.append(action)
                self._log("delayed_mention_action", dict(action))

        action_map = {action["intent_id"]: dict(action) for action in deferred_actions}
        due_payloads: list[dict[str, Any]] = []
        for intent in due_intents:
            fire_overlap = lexical_overlap(intent.intent, assistant_text)
            realized = fire_overlap >= self.DEFERRED_REALIZATION_THRESHOLD
            payload = {
                "intent_id": intent.intent_id,
                "intent": intent.intent,
                "kind": intent.kind,
                "created_turn": intent.created_turn,
                "earliest_turn": intent.earliest_turn,
                "latest_turn": intent.latest_turn,
                "hazard_profile": list(intent.hazard_profile),
                "hazard_turn_prob": hazard_turn_probability(
                    intent.hazard_profile,
                    created_turn=intent.created_turn,
                    turn_index=self.turn_index,
                ),
                "fire_turn": self.turn_index,
                "plan_strategy": intent.plan_strategy,
                "plan_signals": list(intent.plan_signals),
                "plan_rationale": intent.plan_rationale,
                "decision_strategy": intent.decision_strategy,
                "decision_signals": list(intent.decision_signals),
                "decision_rationale": intent.decision_rationale,
                "assistant_overlap": fire_overlap,
                "realized": realized,
                "injected": self.config.deferred_intent_mode
                in {DeferredIntentMode.SOFT_FIRE, DeferredIntentMode.HARD_FIRE},
            }
            due_payloads.append(payload)
            if intent.intent_id in action_map:
                action_map[intent.intent_id].update(payload)

        payload = {
            "user": user_text,
            "assistant": assistant_text,
            "semantic_judge_backend": self.config.semantic_judge_backend.value,
            "adaptive_hazard_policy": self.config.adaptive_hazard_policy.value,
            "adaptive_hazard_profile": self.config.adaptive_hazard_profile.value,
            "adaptive_hazard_stage_policy": (
                self.config.adaptive_hazard_stage_policy.value
            ),
            "adaptive_hazard_embedding_guard": (
                self.config.adaptive_hazard_embedding_guard.value
            ),
            "memory_capsule": memory_capsule,
            "conclusion_probe": conclusion,
            "conclusion_steer_strength": self.config.conclusion_steer_strength.value,
            "conclusion_steer_injection": self.config.conclusion_steer_injection.value,
            "latest_conclusion_probe_turn": self.latest_conclusion_probe_turn,
            "latest_conclusion_line": latest_conclusion_line or None,
            "latest_conclusion_keywords": latest_conclusion_keywords or None,
            "latest_conclusion_keyword_hits": latest_conclusion_keyword_hits,
            "latest_conclusion_keyword_coverage": latest_conclusion_keyword_coverage,
            "latest_conclusion_keyword_mentioned": latest_conclusion_keyword_mentioned,
            "latest_conclusion_line_reply_overlap": latest_conclusion_line_reply_overlap,
            "latest_conclusion_line_mentioned": latest_conclusion_line_mentioned,
            "latest_conclusion_mentioned_any": latest_conclusion_mentioned_any,
            "latest_conclusion_line_age_turns": latest_conclusion_line_age_turns,
            "latest_conclusion_plan_delay_min_turns": self.latest_conclusion_mention_delay_min_turns,
            "latest_conclusion_plan_delay_max_turns": self.latest_conclusion_mention_delay_max_turns,
            "latest_conclusion_plan_earliest_turn": latest_conclusion_plan_earliest_turn,
            "latest_conclusion_plan_latest_turn": latest_conclusion_plan_latest_turn,
            "latest_conclusion_plan_hazard_profile": (
                list(self.latest_conclusion_mention_hazard_profile)
                if self.latest_conclusion_mention_hazard_profile
                else None
            ),
            "latest_conclusion_plan_hazard_turn_prob": latest_conclusion_plan_hazard_turn_prob,
            "latest_conclusion_plan_adaptive_hazard_turn_prob": (
                latest_conclusion_plan_adaptive_hazard_turn_prob
            ),
            "latest_conclusion_plan_adaptive_multiplier": (
                latest_conclusion_plan_adaptive_multiplier
            ),
            "latest_conclusion_plan_adaptive_threshold": (
                latest_conclusion_plan_adaptive_threshold
            ),
            "latest_conclusion_plan_adaptive_threshold_shift": (
                latest_conclusion_plan_adaptive_threshold_shift
            ),
            "latest_conclusion_plan_adaptive_reasons": (
                latest_conclusion_plan_adaptive_reasons
            ),
            "latest_conclusion_plan_likelihood": self.latest_conclusion_mention_likelihood,
            "latest_conclusion_plan_strategy": self.latest_conclusion_delay_strategy or None,
            "latest_conclusion_plan_signals": list(self.latest_conclusion_delay_signals or [])
            or None,
            "latent_convergence_alignment": (
                latent_convergence.get("alignment") if latent_convergence else None
            ),
            "latent_convergence_readiness": (
                latent_convergence.get("articulation_readiness")
                if latent_convergence
                else None
            ),
            "latent_convergence_leakage_risk": (
                latent_convergence.get("leakage_risk") if latent_convergence else None
            ),
            "latent_convergence_stage": (
                latent_convergence.get("trajectory_stage") if latent_convergence else None
            ),
            "latent_convergence_signals": (
                list(latent_convergence.get("shift_signals") or [])
                if latent_convergence
                else None
            ),
            "latent_convergence_explicit": (
                latent_convergence.get("explicit_mention_present")
                if latent_convergence
                else None
            ),
            "latent_convergence_evidence": (
                latent_convergence.get("evidence") if latent_convergence else None
            ),
            "latent_convergence_judge_source": (
                latent_convergence.get("judge_source") if latent_convergence else None
            ),
            "latent_convergence_judge_provider": (
                latent_convergence.get("judge_provider") if latent_convergence else None
            ),
            "latent_convergence_judge_model": (
                latent_convergence.get("judge_model") if latent_convergence else None
            ),
            "latent_convergence_trace": latent_convergence,
            "embedding_convergence_alignment": (
                embedding_convergence.get("alignment") if embedding_convergence else None
            ),
            "embedding_convergence_line_alignment": (
                embedding_convergence.get("line_alignment") if embedding_convergence else None
            ),
            "embedding_convergence_keyword_alignment": (
                embedding_convergence.get("keyword_alignment")
                if embedding_convergence
                else None
            ),
            "embedding_convergence_judge_source": (
                embedding_convergence.get("judge_source") if embedding_convergence else None
            ),
            "embedding_convergence_judge_provider": (
                embedding_convergence.get("judge_provider") if embedding_convergence else None
            ),
            "embedding_convergence_judge_model": (
                embedding_convergence.get("judge_model") if embedding_convergence else None
            ),
            "embedding_convergence_trace": embedding_convergence,
            "semantic_judge_alignment_gap": semantic_judge_alignment_gap,
            "adaptive_hazard_trace": self.latest_adaptive_hazard_trace,
            "planned_delayed_mentions": (
                [item.to_dict() for item in planned_delayed_mentions]
                if planned_delayed_mentions
                else []
            ),
            "planned_delayed_mention_count": len(planned_delayed_mentions),
            "due_delayed_mentions": [
                {
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "release_stage_role": item.release_stage_role,
                    "created_turn": item.created_turn,
                    "earliest_turn": item.earliest_turn,
                    "latest_turn": item.latest_turn,
                    "hazard_profile": list(item.hazard_profile),
                    "hazard_turn_prob": (
                        self._delayed_mention_adjustment(
                            item,
                            adaptive_delayed_mention_adjustments,
                        ).get("base_turn_prob")
                    ),
                    "adaptive_hazard_turn_prob": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("effective_turn_prob"),
                    "adaptive_hazard_multiplier": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("turn_prob_multiplier"),
                    "adaptive_hazard_threshold": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("effective_threshold"),
                    "adaptive_hazard_threshold_shift": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("threshold_shift"),
                    "adaptive_hazard_reasons": list(
                        self._delayed_mention_adjustment(
                            item,
                            adaptive_delayed_mention_adjustments,
                        ).get("reasons")
                        or []
                    ),
                    "likelihood": item.likelihood,
                    "text": truncate_text(item.text, 140),
                }
                for item in due_delayed_mentions
            ],
            "injected_delayed_mentions": [
                {
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "release_stage_role": item.release_stage_role,
                    "created_turn": item.created_turn,
                    "earliest_turn": item.earliest_turn,
                    "latest_turn": item.latest_turn,
                    "hazard_profile": list(item.hazard_profile),
                    "hazard_turn_prob": (
                        self._delayed_mention_adjustment(
                            item,
                            adaptive_delayed_mention_adjustments,
                        ).get("base_turn_prob")
                    ),
                    "adaptive_hazard_turn_prob": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("effective_turn_prob"),
                    "adaptive_hazard_multiplier": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("turn_prob_multiplier"),
                    "adaptive_hazard_threshold": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("effective_threshold"),
                    "adaptive_hazard_threshold_shift": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("threshold_shift"),
                    "adaptive_hazard_reasons": list(
                        self._delayed_mention_adjustment(
                            item,
                            adaptive_delayed_mention_adjustments,
                        ).get("reasons")
                        or []
                    ),
                    "likelihood": item.likelihood,
                    "text": truncate_text(item.text, 140),
                }
                for item in injected_delayed_mentions
            ],
            "delayed_mention_actions": delayed_mention_actions,
            "delayed_mention_mode": self.config.delayed_mention_mode.value,
            "delayed_mention_min_nonconclusion_items": (
                self.config.delayed_mention_min_nonconclusion_items
            ),
            "delayed_mention_min_kind_diversity": (
                self.config.delayed_mention_min_kind_diversity
            ),
            "delayed_mention_diversity_repair": (
                self.config.delayed_mention_diversity_repair.value
            ),
            "delayed_mention_fire_prob": delayed_mention_fire_prob,
            "delayed_mention_fire_max_items": delayed_mention_fire_max_items,
            "delayed_mention_leak_policy": self.config.delayed_mention_leak_policy.value,
            "delayed_mention_leak_threshold": self._delayed_mention_leak_threshold(),
            "suppressed_delayed_mentions": [
                {
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "release_stage_role": item.release_stage_role,
                    "created_turn": item.created_turn,
                    "earliest_turn": item.earliest_turn,
                    "latest_turn": item.latest_turn,
                    "hazard_profile": list(item.hazard_profile),
                    "hazard_turn_prob": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("base_turn_prob"),
                    "adaptive_hazard_turn_prob": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("effective_turn_prob"),
                    "adaptive_hazard_multiplier": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("turn_prob_multiplier"),
                    "threshold": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("effective_threshold"),
                    "base_threshold": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("base_threshold"),
                    "adaptive_hazard_threshold_shift": self._delayed_mention_adjustment(
                        item,
                        adaptive_delayed_mention_adjustments,
                    ).get("threshold_shift"),
                    "adaptive_hazard_reasons": list(
                        self._delayed_mention_adjustment(
                            item,
                            adaptive_delayed_mention_adjustments,
                        ).get("reasons")
                        or []
                    ),
                    "likelihood": item.likelihood,
                    "text": truncate_text(item.text, 140),
                }
                for item in suppressed_delayed_mentions
            ],
            "delayed_mention_injection_draws": delayed_mention_injection_draws,
            "planned_deferred_intent": planned_intent.to_dict() if planned_intent else None,
            "planned_deferred_intents": planned_intents_payload,
            "planned_deferred_intent_count": len(planned_intents_payload),
            "due_deferred_intents": due_payloads,
            "deferred_intent_actions": list(action_map.values()),
            "deferred_intent_backend": self.config.deferred_intent_backend.value,
            "deferred_intent_timing": self.config.deferred_intent_timing.value,
            "deferred_intent_plan_policy": self.config.deferred_intent_plan_policy.value,
            "deferred_intent_plan_budget": self.config.deferred_intent_plan_budget,
            "deferred_intent_plan_probe_calls": self.deferred_intent_plan_probe_calls,
            "deferred_intent_latent_injection": self.config.deferred_intent_latent_injection.value,
            "deferred_intent_ablation": self.config.deferred_intent_ablation.value,
            "deferred_intent_ablation_actions": ablation_actions,
            "inband_state": inband_state,
            "inband_state_error": inband_state_error,
            "inband_state_chars": inband_state_chars,
            "system_prompt": system_prompt,
            "usage": reply.usage,
            "finish_reason": reply.finish_reason,
            "request_id": reply.request_id,
            "probe_reply_overlap": overlap,
        }
        self._log("assistant_reply", payload)
        return payload


# ----------------------------
# Scripted comparison harness
# ----------------------------

def load_script(path: Path) -> tuple[str, list[str]]:
    def normalize_turns(turns_raw: list[Any]) -> list[str]:
        turns: list[str] = []
        for idx, item in enumerate(turns_raw, start=1):
            turn = str(item).strip()
            if not turn:
                raise ValueError(f"Script turn {idx} must not be empty or whitespace.")
            turns.append(turn)
        return turns

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        turns = normalize_turns(data)
        return "", turns
    if isinstance(data, dict):
        system = str(data.get("system", "") or "")
        turns_raw = data.get("turns")
        if not isinstance(turns_raw, list):
            raise ValueError("Script JSON object must contain a list field 'turns'.")
        turns = normalize_turns(turns_raw)
        return system, turns
    raise ValueError("Script JSON must be either a list[str] or {system, turns}.")


def parse_provider_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Provider spec {spec!r} is invalid. Use provider=model, "
                "e.g. openai=gpt-5-mini"
            )
        provider, model = spec.split("=", 1)
        provider = provider.strip()
        model = model.strip()
        if not provider or not model:
            raise ValueError(f"Invalid provider spec: {spec!r}")
        parsed.append((provider, model))
    return parsed


def make_experiment_config_from_args(args: argparse.Namespace, *, base_system: str = "") -> ExperimentConfig:
    base = base_system or args.system or ""
    cfg = ExperimentConfig(
        base_system=base,
        recent_window_messages=args.window,
        memory_every=args.memory_every,
        memory_capsule_limit=args.memory_limit,
        memory_word_budget=args.memory_words,
        conclusion_every=args.conclusion_every,
        conclusion_mode=ConclusionMode(args.conclusion_mode),
        conclusion_steer_strength=SteerStrength(args.conclusion_steer_strength),
        conclusion_steer_injection=ConclusionSteerInjection(args.conclusion_steer_injection),
        delayed_mention_every=getattr(args, "delayed_mention_every", 0),
        delayed_mention_item_limit=max(0, int(getattr(args, "delayed_mention_item_limit", 3) or 0)),
        delayed_mention_min_nonconclusion_items=max(
            0,
            int(getattr(args, "delayed_mention_min_nonconclusion_items", 1) or 0),
        ),
        delayed_mention_min_kind_diversity=max(
            1,
            int(getattr(args, "delayed_mention_min_kind_diversity", 2) or 1),
        ),
        delayed_mention_diversity_repair=DelayedMentionDiversityRepairPolicy(
            getattr(
                args,
                "delayed_mention_diversity_repair",
                DelayedMentionDiversityRepairPolicy.ON.value,
            )
        ),
        delayed_mention_mode=DelayedMentionMode(
            getattr(args, "delayed_mention_mode", DelayedMentionMode.OBSERVE.value)
        ),
        delayed_mention_fire_prob=float(getattr(args, "delayed_mention_fire_prob", 0.35) or 0.0),
        delayed_mention_fire_max_items=max(
            0, int(getattr(args, "delayed_mention_fire_max_items", 2) or 0)
        ),
        delayed_mention_leak_policy=DelayedMentionLeakPolicy(
            getattr(
                args,
                "delayed_mention_leak_policy",
                DelayedMentionLeakPolicy.ON.value,
            )
        ),
        delayed_mention_leak_threshold=clamp01(
            getattr(args, "delayed_mention_leak_threshold", 0.05),
            default=0.05,
        ),
        adaptive_hazard_policy=AdaptiveHazardPolicy(
            getattr(args, "adaptive_hazard_policy", AdaptiveHazardPolicy.ADAPTIVE.value)
        ),
        adaptive_hazard_profile=AdaptiveHazardProfile(
            getattr(
                args,
                "adaptive_hazard_profile",
                AdaptiveHazardProfile.BALANCED.value,
            )
        ),
        adaptive_hazard_stage_policy=AdaptiveHazardStagePolicy(
            getattr(
                args,
                "adaptive_hazard_stage_policy",
                AdaptiveHazardStagePolicy.FLAT.value,
            )
        ),
        adaptive_hazard_embedding_guard=AdaptiveHazardEmbeddingGuard(
            getattr(
                args,
                "adaptive_hazard_embedding_guard",
                AdaptiveHazardEmbeddingGuard.OFF.value,
            )
        ),
        latent_convergence_every=max(
            0, int(getattr(args, "latent_convergence_every", 0) or 0)
        ),
        semantic_judge_backend=SemanticJudgeBackend(
            getattr(args, "semantic_judge_backend", SemanticJudgeBackend.LLM.value)
        ),
        deferred_intent_every=args.deferred_intent_every,
        deferred_intent_mode=DeferredIntentMode(args.deferred_intent_mode),
        deferred_intent_strategy=DeferredIntentStrategy(args.deferred_intent_strategy),
        deferred_intent_timing=DeferredIntentTiming(
            getattr(args, "deferred_intent_timing", DeferredIntentTiming.OFFSET.value)
        ),
        deferred_intent_offset=args.deferred_intent_offset,
        deferred_intent_grace=args.deferred_intent_grace,
        deferred_intent_limit=args.deferred_intent_limit,
        deferred_intent_plan_policy=DeferredIntentPlanPolicy(
            getattr(
                args,
                "deferred_intent_plan_policy",
                DeferredIntentPlanPolicy.PERIODIC.value,
            )
        ),
        deferred_intent_plan_budget=max(
            0, int(getattr(args, "deferred_intent_plan_budget", 0) or 0)
        ),
        deferred_intent_plan_max_new=max(0, int(getattr(args, "deferred_intent_plan_max_new", 1) or 0)),
        deferred_intent_backend=DeferredIntentBackend(args.deferred_intent_backend),
        deferred_intent_latent_injection=DeferredIntentLatentInjection(
            args.deferred_intent_latent_injection
        ),
        deferred_intent_ablation=DeferredIntentAblation(args.deferred_intent_ablation),
        show_probe_outputs=bool(args.show_probes),
        reply_config=GenerationConfig(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout,
        ),
        probe_config=GenerationConfig(
            temperature=0.0,
            max_tokens=args.probe_max_tokens,
            timeout_seconds=args.timeout,
        ),
    )

    if (
        cfg.deferred_intent_plan_policy == DeferredIntentPlanPolicy.AUTO
        and cfg.deferred_intent_plan_budget <= 0
    ):
        raise ValueError(
            "deferred_intent_plan_policy='auto' requires --deferred-intent-plan-budget >= 1."
        )

    return cfg


def run_repl(args: argparse.Namespace) -> int:
    adapter = build_adapter(args.provider, args.model)
    observer_adapter = build_optional_observer_adapter(args)
    embedding_adapter = build_optional_embedding_adapter(args)
    cfg = make_experiment_config_from_args(args)
    log_path = Path(args.log) if args.log else None
    session = RecursiveConclusionSession(
        adapter=adapter,
        observer_adapter=observer_adapter,
        embedding_adapter=embedding_adapter,
        config=cfg,
        log_path=log_path,
    )

    print(f"[provider={adapter.provider_name} model={adapter.model}]")
    print("Type /exit to quit.\n")

    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            break
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break

        result = session.user_turn(user_text)
        if cfg.show_probe_outputs:
            print_probe_outputs(result)
        print(f"assistant> {result['assistant']}\n")
    return 0


def print_probe_outputs(result: dict[str, Any]) -> None:
    if result.get("memory_capsule"):
        print(f"\n[memory capsule]\n{result['memory_capsule']}\n")
    if result.get("conclusion_probe"):
        print(f"[conclusion probe]\n{result['conclusion_probe']}\n")
    latent_trace = result.get("latent_convergence_trace")
    if isinstance(latent_trace, dict):
        print(
            "[latent convergence]\n"
            f"judge={latent_trace.get('judge_source')} "
            f"{latent_trace.get('judge_provider')}/{latent_trace.get('judge_model')} "
            f"alignment={latent_trace.get('alignment')} "
            f"readiness={latent_trace.get('articulation_readiness')} "
            f"leakage_risk={latent_trace.get('leakage_risk')} "
            f"stage={latent_trace.get('trajectory_stage')}\n"
        )
    embedding_trace = result.get("embedding_convergence_trace")
    if isinstance(embedding_trace, dict):
        print(
            "[embedding convergence]\n"
            f"judge={embedding_trace.get('judge_source')} "
            f"{embedding_trace.get('judge_provider')}/{embedding_trace.get('judge_model')} "
            f"alignment={embedding_trace.get('alignment')} "
            f"line={embedding_trace.get('line_alignment')} "
            f"keywords={embedding_trace.get('keyword_alignment')}\n"
        )
    adaptive_trace = result.get("adaptive_hazard_trace")
    if isinstance(adaptive_trace, dict):
        signals = adaptive_trace.get("signals") or {}
        print(
            "[adaptive hazard]\n"
            f"policy={adaptive_trace.get('policy')} "
            f"profile={adaptive_trace.get('profile')} "
            f"alignment={signals.get('alignment')} "
            f"readiness={signals.get('readiness')} "
            f"leakage_risk={signals.get('leakage_risk')} "
            f"judge_gap={signals.get('judge_gap')} "
            f"stage={signals.get('trajectory_stage')}\n"
        )

    planned_delayed_mentions = result.get("planned_delayed_mentions") or []
    if isinstance(planned_delayed_mentions, list) and planned_delayed_mentions:
        print("[delayed mentions planned]")
        for item in planned_delayed_mentions:
            if not isinstance(item, dict):
                continue
            hazard_brief = format_hazard_profile_brief(item.get("hazard_profile") or [])
            hazard_suffix = f"; hazard: {hazard_brief}" if hazard_brief else ""
            print(
                f"- {item.get('item_id')}: {item.get('text')} "
                f"(window: turn {item.get('earliest_turn')}..{item.get('latest_turn')}{hazard_suffix})"
            )
        print()

    suppressed_delayed_mentions = result.get("suppressed_delayed_mentions") or []
    if isinstance(suppressed_delayed_mentions, list) and suppressed_delayed_mentions:
        print("[delayed mentions suppressed]")
        for item in suppressed_delayed_mentions:
            if not isinstance(item, dict):
                continue
            turn_prob = item.get("hazard_turn_prob")
            adaptive_turn_prob = item.get("adaptive_hazard_turn_prob")
            threshold = item.get("threshold")
            print(
                f"- {item.get('item_id')}: p_base={float(turn_prob):.3f} "
                f"p_eff={float(adaptive_turn_prob):.3f} "
                f"threshold={float(threshold):.3f} | {item.get('text')}"
            )
        print()

    planned_intents = result.get("planned_deferred_intents") or []
    if not planned_intents and result.get("planned_deferred_intent"):
        planned_intents = [result["planned_deferred_intent"]]
    if isinstance(planned_intents, list) and planned_intents:
        print("[deferred intents planned]")
        for item in planned_intents:
            if not isinstance(item, dict):
                continue
            hazard_brief = format_hazard_profile_brief(item.get("hazard_profile") or [])
            hazard_suffix = f"; hazard: {hazard_brief}" if hazard_brief else ""
            print(
                f"- {item.get('intent_id')}: {item.get('intent')} "
                f"(window: turn {item.get('earliest_turn')}..{item.get('latest_turn')}{hazard_suffix})"
            )
        print()

    due_intents = result.get("due_deferred_intents") or []
    if isinstance(due_intents, list) and due_intents:
        print("[deferred intents due]")
        for item in due_intents:
            if not isinstance(item, dict):
                continue
            overlap = float(item.get("assistant_overlap", 0.0) or 0.0)
            hazard_turn_prob = item.get("hazard_turn_prob")
            hazard_suffix = (
                f" hazard_turn_prob={float(hazard_turn_prob):.3f}"
                if isinstance(hazard_turn_prob, (int, float))
                else ""
            )
            print(
                f"- {item.get('intent_id')}: overlap={overlap:.3f} "
                f"realized={item.get('realized')}{hazard_suffix} | {item.get('intent')}"
            )
        print()


def write_summary_rows(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def derive_repeat_seed(
    base_seed: Optional[int],
    *,
    arm_name: str,
    provider: str,
    model: str,
    run_name: str,
) -> Optional[int]:
    if base_seed is None:
        return None
    material = f"{int(base_seed)}|{arm_name}|{provider}|{model}|{run_name}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def build_compare_log_path(
    out_dir: Path,
    *,
    provider: str,
    model: str,
    arm_name: str = "",
    run_name: str = "",
) -> Path:
    prefix = f"arm_{sanitize_filename(arm_name)}___" if arm_name else ""
    suffix = f"__{sanitize_filename(run_name)}" if run_name else ""
    return out_dir / f"{prefix}{provider}__{sanitize_filename(model)}{suffix}.jsonl"


def write_compare_summary(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    arm_name: str = "",
    run_name: str = "",
    prefix: str = "summary",
) -> Path:
    parts = [prefix]
    if arm_name:
        parts.append(sanitize_filename(arm_name))
    if run_name:
        parts.append(sanitize_filename(run_name))
    filename = "__".join(parts) + ".json"
    return write_summary_rows(out_dir / filename, rows)


def execute_compare(args: argparse.Namespace) -> CompareExecutionResult:
    script_path = Path(args.script)
    script_system, turns = load_script(script_path)
    provider_specs = parse_provider_specs(args.providers)

    out_dir = Path(args.out_dir or "compare_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_name = compact_text(str(getattr(args, "arm_name", "") or ""))
    arm_description = compact_text(str(getattr(args, "arm_description", "") or ""))
    run_name = compact_text(str(getattr(args, "run_name", "") or ""))
    run_index = coerce_int(getattr(args, "run_index", None))
    base_seed = coerce_int(getattr(args, "random_seed", None))

    rows: list[dict[str, Any]] = []
    log_paths: list[Path] = []
    if arm_name:
        print(f"=== arm: {arm_name} ===")
    if run_name:
        print(f"=== repeat: {run_name} ===")
    for provider, model in provider_specs:
        effective_seed = derive_repeat_seed(
            base_seed,
            arm_name=arm_name,
            provider=provider,
            model=model,
            run_name=run_name,
        )
        if effective_seed is not None:
            random.seed(effective_seed)
        adapter = build_adapter(provider, model)
        observer_adapter = build_optional_observer_adapter(args)
        embedding_adapter = build_optional_embedding_adapter(args)
        cfg = make_experiment_config_from_args(args, base_system=script_system)
        log_path = build_compare_log_path(
            out_dir,
            provider=provider,
            model=model,
            arm_name=arm_name,
            run_name=run_name,
        )
        log_paths.append(log_path)
        session = RecursiveConclusionSession(
            adapter=adapter,
            observer_adapter=observer_adapter,
            embedding_adapter=embedding_adapter,
            config=cfg,
            log_path=log_path,
        )

        print(f"=== {provider} / {model} ===")
        for turn_no, user_text in enumerate(turns, start=1):
            result = session.user_turn(user_text)
            fire_count = sum(1 for item in result["deferred_intent_actions"] if item.get("action") == "fire")
            if cfg.show_probe_outputs:
                print_probe_outputs(result)
            rows.append(
                {
                    "arm": arm_name or None,
                    "arm_description": arm_description or None,
                    "run_name": run_name or None,
                    "run_index": run_index,
                    "random_seed": effective_seed,
                    "semantic_judge_backend": result["semantic_judge_backend"],
                    "adaptive_hazard_policy": result["adaptive_hazard_policy"],
                    "adaptive_hazard_profile": result["adaptive_hazard_profile"],
                    "adaptive_hazard_stage_policy": result[
                        "adaptive_hazard_stage_policy"
                    ],
                    "adaptive_hazard_embedding_guard": result[
                        "adaptive_hazard_embedding_guard"
                    ],
                    "provider": provider,
                    "model": model,
                    "turn": turn_no,
                    "user": user_text,
                    "assistant": result["assistant"],
                    "memory_capsule": result["memory_capsule"],
                    "conclusion_probe": result["conclusion_probe"],
                    "delayed_mention_leak_policy": result["delayed_mention_leak_policy"],
                    "delayed_mention_leak_threshold": result["delayed_mention_leak_threshold"],
                    "delayed_mention_min_nonconclusion_items": result[
                        "delayed_mention_min_nonconclusion_items"
                    ],
                    "delayed_mention_min_kind_diversity": result[
                        "delayed_mention_min_kind_diversity"
                    ],
                    "delayed_mention_diversity_repair": result[
                        "delayed_mention_diversity_repair"
                    ],
                    "suppressed_delayed_mention_count": len(
                        result.get("suppressed_delayed_mentions") or []
                    ),
                    "planned_deferred_intent": result["planned_deferred_intent"],
                    "due_deferred_intents": result["due_deferred_intents"],
                    "deferred_intent_actions": result["deferred_intent_actions"],
                    "deferred_intent_timing": result["deferred_intent_timing"],
                    "latent_convergence_alignment": result["latent_convergence_alignment"],
                    "latent_convergence_readiness": result["latent_convergence_readiness"],
                    "latent_convergence_leakage_risk": result["latent_convergence_leakage_risk"],
                    "latent_convergence_stage": result["latent_convergence_stage"],
                    "latent_convergence_judge_source": result["latent_convergence_judge_source"],
                    "latent_convergence_judge_provider": result["latent_convergence_judge_provider"],
                    "latent_convergence_judge_model": result["latent_convergence_judge_model"],
                    "embedding_convergence_alignment": result["embedding_convergence_alignment"],
                    "embedding_convergence_judge_provider": result["embedding_convergence_judge_provider"],
                    "embedding_convergence_judge_model": result["embedding_convergence_judge_model"],
                    "semantic_judge_alignment_gap": result["semantic_judge_alignment_gap"],
                    "latest_conclusion_plan_hazard_turn_prob": result[
                        "latest_conclusion_plan_hazard_turn_prob"
                    ],
                    "latest_conclusion_plan_adaptive_hazard_turn_prob": result[
                        "latest_conclusion_plan_adaptive_hazard_turn_prob"
                    ],
                    "latest_conclusion_plan_adaptive_multiplier": result[
                        "latest_conclusion_plan_adaptive_multiplier"
                    ],
                    "latest_conclusion_plan_adaptive_threshold": result[
                        "latest_conclusion_plan_adaptive_threshold"
                    ],
                    "latest_conclusion_plan_adaptive_threshold_shift": result[
                        "latest_conclusion_plan_adaptive_threshold_shift"
                    ],
                    "probe_reply_overlap": result["probe_reply_overlap"],
                    "usage": result["usage"],
                }
            )
            print(
                f"[turn {turn_no}] overlap={result['probe_reply_overlap']:.3f} "
                f"deferred_fire={fire_count}"
            )
        print()

    return CompareExecutionResult(rows=rows, log_paths=log_paths)


def run_compare(args: argparse.Namespace) -> int:
    execution = execute_compare(args)
    summary_path = write_compare_summary(
        Path(args.out_dir or "compare_outputs"),
        execution.rows,
        arm_name=compact_text(str(getattr(args, "arm_name", "") or "")),
        run_name=compact_text(str(getattr(args, "run_name", "") or "")),
    )
    print(f"Wrote {summary_path}")
    return 0


def to_cli_flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def extend_argv_from_args_dict(argv: list[str], args_dict: dict[str, Any]) -> None:
    for key, value in args_dict.items():
        if value is None:
            continue
        flag = to_cli_flag(str(key))
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, list):
            argv.append(flag)
            argv.extend(str(item) for item in value)
            continue
        argv.extend([flag, str(value)])


def build_compare_args_from_config(
    data: dict[str, Any],
    *,
    out_dir_override: Optional[str] = None,
    args_override: Optional[dict[str, Any]] = None,
    arm_name: str = "",
    arm_description: str = "",
) -> argparse.Namespace:
    script = data.get("script")
    providers = data.get("providers")
    out_dir = out_dir_override or data.get("out_dir") or data.get("out-dir") or "compare_outputs"
    if not script or not isinstance(script, str):
        raise ValueError("compare config requires string field 'script'.")
    if not isinstance(providers, list) or not all(isinstance(x, str) for x in providers):
        raise ValueError("compare config requires list[str] field 'providers'.")

    argv: list[str] = ["compare", "--script", script, "--providers", *providers]
    if out_dir:
        argv.extend(["--out-dir", str(out_dir)])

    base_args = data.get("args") or {}
    if not isinstance(base_args, dict):
        raise ValueError("Field 'args' must be an object if present.")
    merged_args = dict(base_args)
    if args_override:
        merged_args.update(args_override)
    if merged_args:
        extend_argv_from_args_dict(argv, merged_args)

    parser = build_parser()
    inner_args = parser.parse_args(argv)
    inner_args.arm_name = arm_name
    inner_args.arm_description = arm_description
    return inner_args


def run_compare_matrix_from_config_data(data: dict[str, Any]) -> int:
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object.")

    arms = data.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("compare-matrix config requires non-empty list field 'arms'.")
    repeats = max(1, int(coerce_int(data.get("repeats")) or 1))
    base_seed = coerce_int(data.get("seed"))
    if base_seed is None:
        args_block = data.get("args") or {}
        if isinstance(args_block, dict):
            base_seed = coerce_int(args_block.get("random_seed"))

    out_dir = Path(str(data.get("out_dir") or data.get("out-dir") or "compare_outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    from analyze_runs import aggregate_summary_rows, load_evaluation_spec, summarize_log

    script_value = data.get("script")
    script_path = Path(str(script_value)) if isinstance(script_value, str) and script_value else None
    evaluation = load_evaluation_spec(script_path)
    for idx, arm in enumerate(arms, start=1):
        if not isinstance(arm, dict):
            raise ValueError(f"Arm {idx} must be an object.")
        arm_name = compact_text(str(arm.get("name") or f"arm_{idx:02d}"))
        arm_description = compact_text(str(arm.get("description") or ""))
        arm_args = arm.get("args") or {}
        if not isinstance(arm_args, dict):
            raise ValueError(f"Arm {arm_name!r} field 'args' must be an object.")
        arm_rows: list[dict[str, Any]] = []
        for repeat_idx in range(1, repeats + 1):
            run_name = f"run_{repeat_idx:03d}" if repeats > 1 else ""
            compare_args = build_compare_args_from_config(
                data,
                out_dir_override=str(out_dir),
                args_override=arm_args,
                arm_name=arm_name,
                arm_description=arm_description,
            )
            compare_args.run_name = run_name
            compare_args.run_index = repeat_idx
            if base_seed is not None:
                compare_args.random_seed = base_seed + (repeat_idx - 1)
            execution = execute_compare(compare_args)
            arm_rows.extend(execution.rows)
            combined_rows.extend(execution.rows)
            if run_name:
                run_summary_path = write_compare_summary(
                    out_dir,
                    execution.rows,
                    arm_name=arm_name,
                    run_name=run_name,
                )
                print(f"Wrote {run_summary_path}")

            for log_path in execution.log_paths:
                summary_row = summarize_log(log_path, evaluation)
                summary_row["arm_description"] = arm_description or None
                if repeat_idx and summary_row.get("run_index") is None:
                    summary_row["run_index"] = repeat_idx
                if run_name and not summary_row.get("run_name"):
                    summary_row["run_name"] = run_name
                if base_seed is not None:
                    summary_row["random_seed"] = derive_repeat_seed(
                        base_seed + (repeat_idx - 1),
                        arm_name=arm_name,
                        provider=str(summary_row.get("provider") or ""),
                        model=str(summary_row.get("model") or ""),
                        run_name=run_name,
                    )
                analysis_rows.append(summary_row)

        rows = arm_rows
        arm_summary_path = write_compare_summary(out_dir, rows, arm_name=arm_name)
        print(f"Wrote {arm_summary_path}")

    matrix_summary_path = write_compare_summary(out_dir, combined_rows)
    analysis_runs_path = write_summary_rows(out_dir / "analysis_runs.json", analysis_rows)
    analysis_aggregate_rows = aggregate_summary_rows(analysis_rows)
    analysis_aggregate_path = write_summary_rows(
        out_dir / "analysis_aggregate.json",
        analysis_aggregate_rows,
    )
    manifest_path = write_summary_rows(
        out_dir / "arms.json",
        {
            "repeats": repeats,
            "seed": base_seed,
            "arms": [
                {
                    "name": compact_text(str(arm.get("name") or f"arm_{idx:02d}")),
                    "description": compact_text(str(arm.get("description") or "")) or None,
                    "args": arm.get("args") or {},
                }
                for idx, arm in enumerate(arms, start=1)
                if isinstance(arm, dict)
            ],
        },
    )
    print(f"Wrote {matrix_summary_path}")
    print(f"Wrote {analysis_runs_path}")
    print(f"Wrote {analysis_aggregate_path}")
    print(f"Wrote {manifest_path}")
    return 0


def run_compare_matrix(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    return run_compare_matrix_from_config_data(data)


def run_config(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object.")

    command = data.get("command")
    if command is None:
        if "providers" in data and "arms" in data:
            command = "compare-matrix"
        elif "providers" in data:
            command = "compare"
        elif "provider" in data and "model" in data:
            command = "repl"
    command = str(command or "").strip().lower().replace("_", "-")
    if command not in {"compare", "repl", "compare-matrix"}:
        raise ValueError(
            "Config must specify command='compare'|'repl'|'compare-matrix', or include either "
            "('providers' for compare, optionally with 'arms' for compare-matrix) or "
            "('provider' and 'model' for repl)."
        )
    if command == "compare-matrix":
        return run_compare_matrix_from_config_data(data)

    argv: list[str] = [command]
    if command == "compare":
        script = data.get("script")
        providers = data.get("providers")
        out_dir = data.get("out_dir") or data.get("out-dir")
        if not script or not isinstance(script, str):
            raise ValueError("compare config requires string field 'script'.")
        if not isinstance(providers, list) or not all(isinstance(x, str) for x in providers):
            raise ValueError("compare config requires list[str] field 'providers'.")
        argv.extend(["--script", script, "--providers", *providers])
        if out_dir:
            argv.extend(["--out-dir", str(out_dir)])
    else:
        provider = data.get("provider")
        model = data.get("model")
        if not provider or not model:
            raise ValueError("repl config requires fields 'provider' and 'model'.")
        argv.extend(["--provider", str(provider), "--model", str(model)])
        log_path = data.get("log")
        if log_path:
            argv.extend(["--log", str(log_path)])

    args_dict = data.get("args") or {}
    if not isinstance(args_dict, dict):
        raise ValueError("Field 'args' must be an object if present.")
    extend_argv_from_args_dict(argv, args_dict)

    parser = build_parser()
    inner_args = parser.parse_args(argv)
    return int(inner_args.func(inner_args))


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text)


# ----------------------------
# CLI
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified multi-provider LLM experiment harness with recursive memory capsules, "
            "periodic conclusion probes, and deferred utterance intents."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser, *, single_provider: bool) -> None:
        if single_provider:
            p.add_argument("--provider", required=True, help="openai | anthropic | mistral | gemini | hf | dummy")
            p.add_argument("--model", required=True, help="Model id for the selected provider.")
        p.add_argument("--system", default="", help="Base system prompt.")
        p.add_argument(
            "--random-seed",
            type=int,
            default=None,
            help="Optional seed for harness-side stochastic components such as soft-fire sampling.",
        )
        p.add_argument("--window", type=int, default=8, help="How many recent messages to reload each turn.")
        p.add_argument("--memory-every", type=int, default=3, help="Create a new memory capsule every N user turns. 0 disables.")
        p.add_argument("--memory-limit", type=int, default=4, help="Maximum number of memory capsules to retain.")
        p.add_argument("--memory-words", type=int, default=140, help="Soft word budget for each memory capsule.")
        p.add_argument("--conclusion-every", type=int, default=3, help="Probe the likely end-state every N user turns. 0 disables.")
        p.add_argument(
            "--conclusion-mode",
            choices=[m.value for m in ConclusionMode],
            default=ConclusionMode.OBSERVE.value,
            help="Whether the latest conclusion hypothesis is only logged or softly injected into the next reply.",
        )
        p.add_argument(
            "--conclusion-steer-strength",
            choices=[s.value for s in SteerStrength],
            default=SteerStrength.MEDIUM.value,
            help="How strongly the injected conclusion hypothesis should steer the next reply (only affects --conclusion-mode soft_steer).",
        )
        p.add_argument(
            "--conclusion-steer-injection",
            choices=[i.value for i in ConclusionSteerInjection],
            default=ConclusionSteerInjection.FULL.value,
            help="Which parts of the conclusion probe to inject (only affects --conclusion-mode soft_steer).",
        )
        p.add_argument(
            "--delayed-mention-every",
            type=int,
            default=0,
            help="Probe for delayed mention targets every N user turns. 0 disables.",
        )
        p.add_argument(
            "--delayed-mention-item-limit",
            type=int,
            default=3,
            help="Maximum number of delayed mention targets to plan per probe.",
        )
        p.add_argument(
            "--delayed-mention-min-nonconclusion-items",
            type=int,
            default=1,
            help=(
                "Soft planner target: ask for at least this many delayed mentions whose kind is not "
                "'conclusion' when plausible."
            ),
        )
        p.add_argument(
            "--delayed-mention-min-kind-diversity",
            type=int,
            default=2,
            help=(
                "Soft planner target for the minimum number of distinct delayed-mention kinds "
                "to return when plausible."
            ),
        )
        p.add_argument(
            "--delayed-mention-diversity-repair",
            choices=[m.value for m in DelayedMentionDiversityRepairPolicy],
            default=DelayedMentionDiversityRepairPolicy.ON.value,
            help=(
                "If the first delayed-mention plan lacks enough non-conclusion items or kind diversity, "
                "run one supplemental non-conclusion probe."
            ),
        )
        p.add_argument(
            "--delayed-mention-mode",
            choices=[m.value for m in DelayedMentionMode],
            default=DelayedMentionMode.OBSERVE.value,
            help="Whether delayed mention targets are only observed or softly fired into due turns.",
        )
        p.add_argument(
            "--delayed-mention-fire-prob",
            type=float,
            default=0.35,
            help="Per-item probability of injecting a due delayed-mention hint (only affects --delayed-mention-mode soft_fire).",
        )
        p.add_argument(
            "--delayed-mention-fire-max-items",
            type=int,
            default=2,
            help="Maximum number of due delayed-mention hints to inject per turn (only affects --delayed-mention-mode soft_fire).",
        )
        p.add_argument(
            "--delayed-mention-leak-policy",
            choices=[m.value for m in DelayedMentionLeakPolicy],
            default=DelayedMentionLeakPolicy.ON.value,
            help=(
                "Whether active delayed mentions that are still below the current hazard threshold "
                "should be explicitly suppressed from surfacing."
            ),
        )
        p.add_argument(
            "--delayed-mention-leak-threshold",
            type=float,
            default=0.05,
            help=(
                "Current-turn hazard probability required before an active delayed mention is allowed "
                "to surface without suppression when --delayed-mention-leak-policy is on."
            ),
        )
        p.add_argument(
            "--adaptive-hazard-policy",
            choices=[m.value for m in AdaptiveHazardPolicy],
            default=AdaptiveHazardPolicy.ADAPTIVE.value,
            help=(
                "Whether current-turn hazard probabilities remain static or are adaptively "
                "scaled from recent semantic convergence traces."
            ),
        )
        p.add_argument(
            "--adaptive-hazard-profile",
            choices=[m.value for m in AdaptiveHazardProfile],
            default=AdaptiveHazardProfile.BALANCED.value,
            help=(
                "Adaptive hazard temperament: conservative delays more, balanced mixes hold/release, "
                "and eager releases earlier."
            ),
        )
        p.add_argument(
            "--adaptive-hazard-stage-policy",
            choices=[m.value for m in AdaptiveHazardStagePolicy],
            default=AdaptiveHazardStagePolicy.FLAT.value,
            help=(
                "Whether adaptive hazard treats delayed mentions uniformly or separates "
                "option-stage items from final-stage risk packets."
            ),
        )
        p.add_argument(
            "--adaptive-hazard-embedding-guard",
            choices=[m.value for m in AdaptiveHazardEmbeddingGuard],
            default=AdaptiveHazardEmbeddingGuard.OFF.value,
            help=(
                "Optional pre-peak semantic leakage guard driven by embedding alignment. "
                "Keep this off unless you are explicitly comparing its effect."
            ),
        )
        p.add_argument(
            "--latent-convergence-every",
            type=int,
            default=0,
            help=(
                "Run semantic judges every N turns after replies. "
                "0 disables."
            ),
        )
        p.add_argument(
            "--semantic-judge-backend",
            choices=[m.value for m in SemanticJudgeBackend],
            default=SemanticJudgeBackend.LLM.value,
            help=(
                "Which semantic judge to run at each latent-convergence step: "
                "llm, embedding, both, or off."
            ),
        )
        p.add_argument(
            "--observer-provider",
            default="",
            help=(
                "Optional independent observer provider for latent convergence judging. "
                "If unset, the generator adapter judges its own replies."
            ),
        )
        p.add_argument(
            "--observer-model",
            default="",
            help="Model id for --observer-provider.",
        )
        p.add_argument(
            "--embedding-provider",
            default="",
            help=(
                "Optional embedding provider for semantic judging. "
                "Required when --semantic-judge-backend is embedding or both."
            ),
        )
        p.add_argument(
            "--embedding-model",
            default="",
            help="Model id for --embedding-provider.",
        )
        p.add_argument(
            "--deferred-intent-every",
            type=int,
            default=0,
            help="Plan one deferred utterance every N user turns. 0 disables.",
        )
        p.add_argument(
            "--deferred-intent-mode",
            choices=[m.value for m in DeferredIntentMode],
            default=DeferredIntentMode.OBSERVE.value,
            help="Whether deferred intents are only observed or softly fired into due turns.",
        )
        p.add_argument(
            "--deferred-intent-strategy",
            choices=[m.value for m in DeferredIntentStrategy],
            default=DeferredIntentStrategy.TRIGGER.value,
            help="Scheduling style for deferred intents: fixed, trigger, or adaptive.",
        )
        p.add_argument(
            "--deferred-intent-timing",
            choices=[m.value for m in DeferredIntentTiming],
            default=DeferredIntentTiming.OFFSET.value,
            help=(
                "How to set intent timing windows: 'offset' derives from --deferred-intent-offset/--deferred-intent-grace; "
                "'model' asks the planner to propose timing bounds; "
                "'hazard' asks the planner to emit a per-delay probability profile."
            ),
        )
        p.add_argument(
            "--deferred-intent-offset",
            type=int,
            default=3,
            help="Target delay in user turns before a deferred intent becomes eligible.",
        )
        p.add_argument(
            "--deferred-intent-grace",
            type=int,
            default=2,
            help="Extra slack turns after the initial deferred-intent offset for trigger/adaptive strategies.",
        )
        p.add_argument(
            "--deferred-intent-limit",
            type=int,
            default=6,
            help="Maximum number of simultaneously active deferred intents.",
        )
        p.add_argument(
            "--deferred-intent-plan-policy",
            choices=[m.value for m in DeferredIntentPlanPolicy],
            default=DeferredIntentPlanPolicy.PERIODIC.value,
            help=(
                "Planning cadence for new deferred intents: "
                "'periodic' uses --deferred-intent-every; "
                "'auto' allows planning every turn until the plan budget is exhausted."
            ),
        )
        p.add_argument(
            "--deferred-intent-plan-budget",
            type=int,
            default=0,
            help=(
                "Maximum number of deferred-intent planning opportunities per run. "
                "0 means unlimited (useful with --deferred-intent-plan-policy periodic). "
                "Required for --deferred-intent-plan-policy auto. "
                "Counts planner probe calls (external) / eligible planning turns (inband)."
            ),
        )
        p.add_argument(
            "--deferred-intent-plan-max-new",
            type=int,
            default=1,
            help="Maximum number of new deferred intents the model may create on an eligible planning turn.",
        )
        p.add_argument(
            "--deferred-intent-backend",
            choices=[m.value for m in DeferredIntentBackend],
            default=DeferredIntentBackend.EXTERNAL.value,
            help=(
                "Deferred-intent backend: 'external' uses explicit planner/scheduler probes; "
                "'inband' stores state in a hidden RCL_STATE block carried in the dialogue."
            ),
        )
        p.add_argument(
            "--deferred-intent-latent-injection",
            choices=[m.value for m in DeferredIntentLatentInjection],
            default=DeferredIntentLatentInjection.OFF.value,
            help="Whether active deferred intents are injected as latent (private) context every turn.",
        )
        p.add_argument(
            "--deferred-intent-ablation",
            choices=[m.value for m in DeferredIntentAblation],
            default=DeferredIntentAblation.NONE.value,
            help="Ablation mode for deferred intents (e.g., delete planned intents immediately).",
        )
        p.add_argument("--temperature", type=float, default=0.2, help="Temperature for main assistant replies.")
        p.add_argument("--max-tokens", type=int, default=900, help="Max output tokens for main assistant replies.")
        p.add_argument("--probe-max-tokens", type=int, default=220, help="Max output tokens for memory/conclusion/intent probes.")
        p.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
        p.add_argument("--show-probes", action="store_true", help="Print probe outputs during REPL or scripted runs.")

    repl = sub.add_parser("repl", help="Interactive chat loop.")
    add_common(repl, single_provider=True)
    repl.add_argument("--log", default="", help="Optional JSONL log path.")
    repl.set_defaults(func=run_repl)

    compare = sub.add_parser("compare", help="Run the same script across multiple providers.")
    add_common(compare, single_provider=False)
    compare.add_argument("--script", required=True, help="Path to JSON script. Either list[str] or {system, turns}.")
    compare.add_argument(
        "--providers",
        nargs="+",
        required=True,
        help="Provider specs formatted as provider=model. Example: openai=gpt-5-mini anthropic=claude-opus-4-6",
    )
    compare.add_argument("--out-dir", default="compare_outputs", help="Directory for JSONL logs and summary.")
    compare.set_defaults(func=run_compare)

    compare_matrix = sub.add_parser(
        "compare-matrix",
        help="Run an arm matrix from a JSON config file.",
    )
    compare_matrix.add_argument(
        "--config",
        required=True,
        help="Path to JSON config with script/providers/arms/out_dir.",
    )
    compare_matrix.set_defaults(func=run_compare_matrix)

    run_cfg = sub.add_parser("run-config", help="Run repl/compare from a JSON config file.")
    run_cfg.add_argument(
        "--config",
        required=True,
        help=(
            "Path to JSON config (see templates/compare_config_template.json or "
            "templates/compare_matrix_config_template.json)."
        ),
    )
    run_cfg.set_defaults(func=run_config)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
