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
    deferred_intent_offset: int = 3
    deferred_intent_grace: int = 2
    deferred_intent_limit: int = 6
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
        chunks.append(f"{idx:02d}. {msg.role.upper()}: {msg.content}")
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

    if strategy == DeferredIntentStrategy.FIXED:
        earliest_turn = created_turn + max(1, offset)
        latest_turn = earliest_turn
        trigger = []
        cancel_if = []
    else:
        earliest_turn = created_turn + max(1, offset)
        latest_turn = earliest_turn + max(0, grace)
        trigger = coerce_str_list(data.get("trigger"))
        cancel_if = coerce_str_list(data.get("cancel_if"))

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

        if "compress conversation state into a compact memory capsule" in (system or "").lower() or "existing memory capsules:" in lower and "produce one new memory capsule" in lower:
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
            output = json.dumps(
                {
                    "create_intent": True,
                    "kind": "proposal",
                    "intent": f"After a few more turns, offer a concise operational recommendation grounded in: {compact_text(latest_user)[:100]}",
                    "why_not_now": "The conversation is still gathering constraints and timing cues.",
                    "trigger": [
                        "the user asks for a summary or concrete next step",
                        "enough constraints have accumulated"
                    ],
                    "cancel_if": [
                        "the topic changes",
                        "the user explicitly rejects recommendations"
                    ],
                    "confidence": 0.66,
                    "priority": 0.72,
                },
                ensure_ascii=False,
            )
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
                elif current_turn > latest:
                    action = "expire"
                    reason = "The intended timing window has passed."
                else:
                    action = "fire"
                    reason = "The timing window is open and the utterance can be delivered now."
                decisions.append(
                    {
                        "intent_id": item.get("intent_id", "unknown"),
                        "action": action,
                        "reason": reason,
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

    def _probe_deferred_intent_plan(self) -> Optional[DeferredIntent]:
        if self.config.deferred_intent_every <= 0:
            return None
        if self.turn_index == 0 or self.turn_index % self.config.deferred_intent_every != 0:
            return None
        if len(self._active_deferred_intents()) >= self.config.deferred_intent_limit:
            return None

        if self.config.deferred_intent_strategy == DeferredIntentStrategy.FIXED:
            strategy_text = (
                f"Imagine ONE future utterance that would be more appropriate about {self.config.deferred_intent_offset} user turns later. "
                "It should be useful later but not appropriate yet."
            )
        elif self.config.deferred_intent_strategy == DeferredIntentStrategy.TRIGGER:
            strategy_text = (
                f"Imagine ONE future utterance that should stay in reserve for roughly {self.config.deferred_intent_offset} turns, "
                f"with up to {self.config.deferred_intent_grace} turns of slack. Include timing triggers and cancel conditions."
            )
        else:
            strategy_text = (
                f"Imagine ONE future utterance that should stay in reserve for roughly {self.config.deferred_intent_offset} turns, "
                f"with up to {self.config.deferred_intent_grace} turns of slack. Include timing triggers and cancel conditions. "
                "This plan may be revised later if the dialogue shifts."
            )

        source = textwrap.dedent(
            f"""\
            Memory capsules:
            {render_capsules(self.memory_capsules)}

            Recent dialogue window:
            {render_messages(self._recent_window())}

            Current turn index: {self.turn_index}

            {strategy_text}

            Return strict JSON only.
            Use this schema:
            {{
              "create_intent": true or false,
              "kind": "summary|proposal|correction|question|reminder|other",
              "intent": "what should be said later",
              "why_not_now": "why it should be delayed",
              "trigger": ["condition 1", "condition 2"],
              "cancel_if": ["condition 1", "condition 2"],
              "confidence": 0.0,
              "priority": 0.0
            }}

            If no future utterance should be held, return:
            {{"create_intent": false, "reason": "..."}}
            """
        ).strip()

        response = self.adapter.generate(
            system=(
                "You are a deferred utterance planner. Do not answer the user directly. "
                "Propose at most one future utterance to hold in reserve."
            ),
            messages=[ChatMessage(role="user", content=source)],
            config=self.config.probe_config,
        )
        parsed = extract_json_value(response.text)
        if not isinstance(parsed, dict):
            self._log(
                "deferred_intent_plan_error",
                {
                    "raw_text": response.text,
                    "usage": response.usage,
                    "finish_reason": response.finish_reason,
                    "request_id": response.request_id,
                },
            )
            return None

        intent_id = f"di-{self.next_deferred_intent_index:04d}"
        planned = build_deferred_intent_from_plan(
            parsed,
            intent_id=intent_id,
            created_turn=self.turn_index,
            strategy=self.config.deferred_intent_strategy,
            offset=self.config.deferred_intent_offset,
            grace=self.config.deferred_intent_grace,
        )
        if planned is None:
            self._log(
                "deferred_intent_plan",
                {
                    "intent_id": None,
                    "created": False,
                    "raw_plan": parsed,
                    "usage": response.usage,
                    "finish_reason": response.finish_reason,
                    "request_id": response.request_id,
                },
            )
            return None

        self.next_deferred_intent_index += 1
        self.deferred_intents.append(planned)
        self._log(
            "deferred_intent_plan",
            {
                "intent_id": planned.intent_id,
                "created": True,
                "intent": planned.to_dict(),
                "raw_plan": parsed,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
                "request_id": response.request_id,
            },
        )
        return planned

    def _schedule_deferred_intents(self) -> tuple[list[DeferredIntent], list[dict[str, Any]]]:
        active = self._active_deferred_intents()
        if not active:
            return [], []

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
            source = textwrap.dedent(
                f"""\
                Memory capsules:
                {render_capsules(self.memory_capsules)}

                Recent dialogue window:
                {render_messages(self._recent_window())}

                Current turn index: {self.turn_index}

                Deferred intents:
                {json.dumps([intent.to_dict() for intent in active], ensure_ascii=False, indent=2)}

                Decide whether each deferred utterance should be held, fired, canceled, expired, or revised.
                Use "fire" only when the utterance would feel timely and appropriate now.
                Never fire before earliest_turn.
                Prefer "expire" if current_turn is past latest_turn and the opportunity has passed.
                Use "cancel" if the topic moved on or its assumptions broke.
                "revise" is allowed only if the core idea is still useful but the wording or timing should change.

                Return strict JSON only:
                {{
                  "decisions": [
                    {{
                      "intent_id": "di-0001",
                      "action": "hold|fire|cancel|expire__REVISE_SUFFIX__",
                      "reason": "brief reason",
                      "updated_intent": {{
                        "kind": "optional",
                        "intent": "optional revised intent",
                        "why_not_now": "optional",
                        "trigger": ["optional"],
                        "cancel_if": ["optional"],
                        "confidence": 0.0,
                        "priority": 0.0,
                        "earliest_turn": 0,
                        "latest_turn": 0
                      }}
                    }}
                  ]
                }}
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
            if not isinstance(decisions_payload, dict):
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

            applied.append(
                {
                    "intent_id": intent.intent_id,
                    "action": action,
                    "reason": reason,
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
        planned_intent = self._probe_deferred_intent_plan()
        due_intents, deferred_actions = self._schedule_deferred_intents()

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
            "due_deferred_intents": due_payloads,
            "deferred_intent_actions": list(action_map.values()),
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
    return ExperimentConfig(
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
        deferred_intent_offset=args.deferred_intent_offset,
        deferred_intent_grace=args.deferred_intent_grace,
        deferred_intent_limit=args.deferred_intent_limit,
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

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
