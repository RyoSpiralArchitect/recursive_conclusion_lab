#!/usr/bin/env python3
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


class DeferredIntentMode(str, Enum):
    OBSERVE = "observe"
    SOFT_FIRE = "soft_fire"


class DeferredIntentStrategy(str, Enum):
    FIXED = "fixed"
    TRIGGER = "trigger"
    ADAPTIVE = "adaptive"


class DeferredIntentTiming(str, Enum):
    OFFSET = "offset"
    MODEL = "model"


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
class DeferredIntent:
    intent_id: str
    created_turn: int
    kind: str
    intent: str
    why_not_now: str = ""
    earliest_turn: int = 0
    latest_turn: int = 0
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


def extract_conclusion_line(text: str) -> str:
    match = re.search(r"(?im)^\s*conclusion\s*:\s*(.+?)\s*$", text or "")
    if not match:
        return ""
    return f"CONCLUSION: {match.group(1).strip()}"


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

    return DeferredIntent(
        intent_id=intent_id,
        created_turn=created_turn,
        kind=kind,
        intent=intent,
        why_not_now=why_not_now,
        earliest_turn=earliest_turn,
        latest_turn=max(earliest_turn, latest_turn),
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

    if earliest_turn is None:
        earliest_turn = max(current_turn + 1, intent.earliest_turn)
    if latest_turn is None:
        latest_turn = max(earliest_turn, intent.latest_turn, current_turn + 1 + max(0, default_grace))

    if latest_turn < earliest_turn:
        latest_turn = earliest_turn

    intent.earliest_turn = earliest_turn
    intent.latest_turn = latest_turn
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
                earliest = coerce_int(item.get("earliest_turn"))
                latest = coerce_int(item.get("latest_turn"))
                if earliest is None:
                    earliest = created_turn + max(1, deferred_intent_offset)
                if latest is None:
                    if deferred_intent_strategy == DeferredIntentStrategy.FIXED.value:
                        latest = earliest
                    else:
                        latest = earliest + max(0, deferred_intent_grace)
                if latest < earliest:
                    latest = earliest

                if current_turn < earliest:
                    continue
                if current_turn > latest:
                    item["status"] = "expired"
                    item["terminal_reason"] = "Dummy: timing window passed."
                    item["decision_strategy"] = "window_expired"
                    item["decision_signals"] = ["past_latest_turn"]
                    item["decision_rationale"] = "The intended timing window has passed."
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
            if deferred_intent_mode == DeferredIntentMode.SOFT_FIRE.value and fired_this_turn:
                fired_block = "\n".join(f"- {text}" for text in fired_this_turn)
                visible_reply = visible_reply.rstrip() + "\n\nDeferred follow-up:\n" + fired_block

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
            output = textwrap.dedent(
                f"""\
                CONCLUSION: The dialogue is converging toward a concrete experiment design with explicit instrumentation and comparison conditions.
                CONFIDENCE: 0.74
                EVIDENCE: The recent turns emphasize recursive memory, periodic probes, and operational evaluation. Latest user turn: {final_user[:120]}
                """
            ).strip()
        elif "deferred utterance planner" in (system or "").lower() or ("create_intent" in lower and "why_not_now" in lower and "deferred utterance" in lower):
            latest_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            timing = None
            if "delay_min_turns" in lower or "delay_max_turns" in lower:
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
                earliest = coerce_int(item.get("earliest_turn")) or current_turn + 1
                latest = coerce_int(item.get("latest_turn")) or earliest
                if current_turn < earliest:
                    action = "hold"
                    reason = "Too early. Keep the utterance in reserve."
                    decision_strategy = "too_early"
                    decision_signals = ["before_earliest_turn"]
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

        return ProviderResponse(
            provider=self.provider_name,
            model=self.model,
            text=output,
            raw={"dummy": True, "text": output},
            usage={"input_tokens": 0, "output_tokens": 0},
            finish_reason="stop",
            request_id=f"dummy-{random.randint(1000, 9999)}",
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

    def __init__(
        self,
        *,
        adapter: BaseAdapter,
        config: ExperimentConfig,
        log_path: Optional[Path] = None,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.log_path = log_path
        self.history: list[ChatMessage] = []
        self.memory_capsules: list[str] = []
        self.conclusion_hypotheses: list[str] = []
        self.deferred_intents: list[DeferredIntent] = []
        self.turn_index = 0
        self.next_deferred_intent_index = 1
        self.deferred_intent_plan_probe_calls = 0
        self._deferred_intent_plan_compact_ok = False
        self._deferred_intent_scheduler_compact_ok = False

    def _recent_window(self) -> list[ChatMessage]:
        if self.config.recent_window_messages <= 0:
            return list(self.history)
        return self.history[-self.config.recent_window_messages :]

    def _active_deferred_intents(self) -> list[DeferredIntent]:
        return [intent for intent in self.deferred_intents if intent.status == "active"]

    def _build_system_prompt(self, *, due_intents: Optional[list[DeferredIntent]] = None) -> str:
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

        if (
            self.config.deferred_intent_mode == DeferredIntentMode.SOFT_FIRE
            and due_intents
        ):
            bullets = "\n".join(
                f"- [{intent.intent_id}] {intent.intent}" for intent in due_intents
            )
            parts.append(
                "Deferred utterances are due now. Integrate them only if they still fit the current turn.\n"
                f"{bullets}\n\n"
                "Realize them naturally instead of quoting them verbatim. "
                "If the latest user turn conflicts with a deferred utterance, prioritize the current evidence."
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
            self.conclusion_hypotheses.append(hypothesis)
            self._log(
                "conclusion_probe",
                {
                    "hypothesis": hypothesis,
                    "usage": response.usage,
                    "finish_reason": response.finish_reason,
                    "request_id": response.request_id,
                },
            )
        return hypothesis or None

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
        planned_intent: Optional[DeferredIntent] = None
        planned_intents_payload: list[dict[str, Any]] = []
        due_intents: list[DeferredIntent] = []
        deferred_actions: list[dict[str, Any]] = []
        ablation_actions: list[dict[str, Any]] = []
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

            base_prompt = self._build_system_prompt(due_intents=None)
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
                  - If you set status=fired, set fire_turn={self.turn_index} and set terminal_reason.
                - If deferred_intent_mode is 'soft_fire', integrate fired intents naturally into the visible reply.
                  If deferred_intent_mode is 'observe', do NOT include fired intent content in the visible reply.
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
                    else:
                        if intent.earliest_turn <= self.turn_index:
                            intent.earliest_turn = self.turn_index + 1
                        if intent.latest_turn < intent.earliest_turn:
                            intent.latest_turn = intent.earliest_turn
                        if self.config.deferred_intent_strategy == DeferredIntentStrategy.FIXED:
                            intent.latest_turn = intent.earliest_turn

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

            system_prompt = self._build_system_prompt(due_intents=due_intents)
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
                "fire_turn": self.turn_index,
                "plan_strategy": intent.plan_strategy,
                "plan_signals": list(intent.plan_signals),
                "plan_rationale": intent.plan_rationale,
                "decision_strategy": intent.decision_strategy,
                "decision_signals": list(intent.decision_signals),
                "decision_rationale": intent.decision_rationale,
                "assistant_overlap": fire_overlap,
                "realized": realized,
                "injected": self.config.deferred_intent_mode == DeferredIntentMode.SOFT_FIRE,
            }
            due_payloads.append(payload)
            if intent.intent_id in action_map:
                action_map[intent.intent_id].update(payload)

        payload = {
            "user": user_text,
            "assistant": assistant_text,
            "memory_capsule": memory_capsule,
            "conclusion_probe": conclusion,
            "conclusion_steer_strength": self.config.conclusion_steer_strength.value,
            "conclusion_steer_injection": self.config.conclusion_steer_injection.value,
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
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        turns = [str(x) for x in data]
        return "", turns
    if isinstance(data, dict):
        system = str(data.get("system", "") or "")
        turns_raw = data.get("turns")
        if not isinstance(turns_raw, list):
            raise ValueError("Script JSON object must contain a list field 'turns'.")
        turns = [str(x) for x in turns_raw]
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
    cfg = make_experiment_config_from_args(args)
    log_path = Path(args.log) if args.log else None
    session = RecursiveConclusionSession(adapter=adapter, config=cfg, log_path=log_path)

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
            if result["memory_capsule"]:
                print(f"\n[memory capsule]\n{result['memory_capsule']}\n")
            if result["conclusion_probe"]:
                print(f"[conclusion probe]\n{result['conclusion_probe']}\n")
            if result["planned_deferred_intent"]:
                plan = result["planned_deferred_intent"]
                print(
                    "[deferred intent planned]\n"
                    f"{plan['intent_id']} -> {plan['intent']} "
                    f"(window: turn {plan['earliest_turn']}..{plan['latest_turn']})\n"
                )
            if result["due_deferred_intents"]:
                print("[deferred intents due]")
                for item in result["due_deferred_intents"]:
                    print(
                        f"- {item['intent_id']}: overlap={item['assistant_overlap']:.3f} realized={item['realized']} | {item['intent']}"
                    )
                print()
        print(f"assistant> {result['assistant']}\n")
    return 0


def run_compare(args: argparse.Namespace) -> int:
    script_path = Path(args.script)
    script_system, turns = load_script(script_path)
    provider_specs = parse_provider_specs(args.providers)

    out_dir = Path(args.out_dir or "compare_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for provider, model in provider_specs:
        adapter = build_adapter(provider, model)
        cfg = make_experiment_config_from_args(args, base_system=script_system)
        log_path = out_dir / f"{provider}__{sanitize_filename(model)}.jsonl"
        session = RecursiveConclusionSession(adapter=adapter, config=cfg, log_path=log_path)

        print(f"=== {provider} / {model} ===")
        for turn_no, user_text in enumerate(turns, start=1):
            result = session.user_turn(user_text)
            fire_count = sum(1 for item in result["deferred_intent_actions"] if item.get("action") == "fire")
            rows.append(
                {
                    "provider": provider,
                    "model": model,
                    "turn": turn_no,
                    "user": user_text,
                    "assistant": result["assistant"],
                    "memory_capsule": result["memory_capsule"],
                    "conclusion_probe": result["conclusion_probe"],
                    "planned_deferred_intent": result["planned_deferred_intent"],
                    "due_deferred_intents": result["due_deferred_intents"],
                    "deferred_intent_actions": result["deferred_intent_actions"],
                    "probe_reply_overlap": result["probe_reply_overlap"],
                    "usage": result["usage"],
                }
            )
            print(
                f"[turn {turn_no}] overlap={result['probe_reply_overlap']:.3f} "
                f"deferred_fire={fire_count}"
            )
        print()

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


def run_config(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object.")

    command = data.get("command")
    if command is None:
        if "providers" in data:
            command = "compare"
        elif "provider" in data and "model" in data:
            command = "repl"
    command = str(command or "").strip().lower()
    if command not in {"compare", "repl"}:
        raise ValueError(
            "Config must specify command='compare'|'repl', or include either "
            "('providers' for compare) or ('provider' and 'model' for repl)."
        )

    def to_flag(name: str) -> str:
        return "--" + name.replace("_", "-")

    def extend_from_args_dict(argv: list[str], args_dict: dict[str, Any]) -> None:
        for key, value in args_dict.items():
            if value is None:
                continue
            flag = to_flag(str(key))
            if isinstance(value, bool):
                if value:
                    argv.append(flag)
                continue
            if isinstance(value, list):
                argv.append(flag)
                argv.extend(str(item) for item in value)
                continue
            argv.extend([flag, str(value)])

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
    extend_from_args_dict(argv, args_dict)

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
                "'model' asks the planner to propose timing (external backend)."
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

    run_cfg = sub.add_parser("run-config", help="Run repl/compare from a JSON config file.")
    run_cfg.add_argument("--config", required=True, help="Path to JSON config (see templates/compare_config_template.json).")
    run_cfg.set_defaults(func=run_config)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
