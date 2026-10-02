#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from inquiry_state import InquiryState

from blind_review import (
    BlindReviewManager,
    ReviewConflictError,
    ReviewNotFoundError,
    ReviewValidationError,
)
from recursive_conclusion_lab import (
    AdaptiveHazardEmbeddingGuard,
    AdaptiveHazardPolicy,
    AdaptiveHazardProfile,
    AdaptiveHazardStagePolicy,
    ChatMessage,
    ConclusionMode,
    ConclusionSteerInjection,
    DelayedMentionDiversityRepairPolicy,
    DelayedMentionItem,
    DelayedMentionLeakPolicy,
    DelayedMentionMode,
    DeferredIntent,
    DeferredIntentAblation,
    DeferredIntentBackend,
    DeferredIntentLatentInjection,
    DeferredIntentMode,
    DeferredIntentPlanPolicy,
    DeferredIntentStrategy,
    DeferredIntentTiming,
    ExperimentConfig,
    GenerationConfig,
    ModelProfilePinError,
    RecursiveConclusionSession,
    SemanticJudgeBackend,
    SteerStrength,
    available_embedding_provider_names,
    available_provider_names,
    build_adapter,
    build_embedding_adapter,
    canonical_embedding_provider_name,
    canonical_provider_name,
    compact_text,
    load_script,
    model_profile_catalog,
    strip_rcl_state,
)


ROOT_DIR = Path(__file__).resolve().parent
PROTOCOL_SCRIPTS_DIR = ROOT_DIR / "protocol_scripts"
PLAYTEST_UI_DIST_DIR = ROOT_DIR / "playtest_ui" / "dist"
DEFAULT_SESSIONS_DIR = ROOT_DIR / "playtest_sessions"
DEFAULT_EVAL_SETS_DIR = ROOT_DIR / "human_eval_sets"
DEFAULT_REVIEW_SESSIONS_DIR = ROOT_DIR / "blind_review_sessions"

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
SESSION_SCHEMA_VERSION = 2
PUBLIC_LOAD_ERROR_LIMIT = 20

DEFAULT_ALLOWED_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

TOKEN_USAGE_FIELDS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "prompttokencount",
        "candidatestokencount",
        "totaltokencount",
        "thoughtstokencount",
        "tooluseprompttokencount",
    }
)


class AmbiguousLegacyProfileError(ModelProfilePinError):
    pass


def json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {
            field.name: json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def value_or_default(data: dict[str, Any], key: str, default: Any) -> Any:
    value = data.get(key)
    return default if value is None else value


def safe_turn_error(error: Exception) -> str:
    kind = type(error).__name__
    status = re.match(r"HTTP ([1-5][0-9]{2})\b", str(error))
    if status:
        return f"{kind}: provider HTTP {status.group(1)}"
    return f"{kind}: turn failed"


def generation_config_from_dict(data: dict[str, Any]) -> GenerationConfig:
    max_tokens = int(value_or_default(data, "max_tokens", 900))
    timeout_seconds = int(value_or_default(data, "timeout_seconds", 120))
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    profile_id_raw = data.get("model_profile_id")
    profile_version_raw = data.get("model_profile_version")
    return GenerationConfig(
        temperature=float(value_or_default(data, "temperature", 0.2)),
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        reasoning_effort=optional_generation_setting(data.get("reasoning_effort")),
        reasoning_mode=optional_generation_setting(data.get("reasoning_mode")),
        reasoning_context=optional_generation_setting(data.get("reasoning_context")),
        text_verbosity=optional_generation_setting(data.get("text_verbosity")),
        model_profile_id=(
            str(profile_id_raw).strip() if profile_id_raw is not None else None
        ),
        model_profile_version=(
            int(profile_version_raw) if profile_version_raw is not None else None
        ),
    )


def optional_generation_setting(value: Any) -> Optional[str]:
    cleaned = str(value or "").strip().lower()
    return None if not cleaned or cleaned == "auto" else cleaned


def experiment_config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        base_system=str(data.get("base_system", "") or ""),
        recent_window_messages=int(value_or_default(data, "recent_window_messages", 8)),
        memory_every=int(value_or_default(data, "memory_every", 3)),
        memory_capsule_limit=int(value_or_default(data, "memory_capsule_limit", 4)),
        memory_word_budget=int(value_or_default(data, "memory_word_budget", 140)),
        conclusion_every=int(value_or_default(data, "conclusion_every", 3)),
        conclusion_mode=ConclusionMode(str(data.get("conclusion_mode", ConclusionMode.OBSERVE.value))),
        conclusion_steer_strength=SteerStrength(
            str(data.get("conclusion_steer_strength", SteerStrength.MEDIUM.value))
        ),
        conclusion_steer_injection=ConclusionSteerInjection(
            str(
                data.get(
                    "conclusion_steer_injection", ConclusionSteerInjection.FULL.value
                )
            )
        ),
        delayed_mention_every=int(value_or_default(data, "delayed_mention_every", 0)),
        delayed_mention_item_limit=int(value_or_default(data, "delayed_mention_item_limit", 3)),
        delayed_mention_min_nonconclusion_items=int(
            value_or_default(data, "delayed_mention_min_nonconclusion_items", 1)
        ),
        delayed_mention_min_kind_diversity=int(
            value_or_default(data, "delayed_mention_min_kind_diversity", 2)
        ),
        delayed_mention_diversity_repair=DelayedMentionDiversityRepairPolicy(
            str(
                data.get(
                    "delayed_mention_diversity_repair",
                    DelayedMentionDiversityRepairPolicy.ON.value,
                )
            )
        ),
        delayed_mention_mode=DelayedMentionMode(
            str(data.get("delayed_mention_mode", DelayedMentionMode.OBSERVE.value))
        ),
        delayed_mention_fire_prob=float(value_or_default(data, "delayed_mention_fire_prob", 0.35)),
        delayed_mention_fire_max_items=int(value_or_default(data, "delayed_mention_fire_max_items", 2)),
        delayed_mention_leak_policy=DelayedMentionLeakPolicy(
            str(
                data.get(
                    "delayed_mention_leak_policy", DelayedMentionLeakPolicy.ON.value
                )
            )
        ),
        delayed_mention_leak_threshold=float(
            value_or_default(data, "delayed_mention_leak_threshold", 0.05)
        ),
        adaptive_hazard_policy=AdaptiveHazardPolicy(
            str(data.get("adaptive_hazard_policy", AdaptiveHazardPolicy.ADAPTIVE.value))
        ),
        adaptive_hazard_profile=AdaptiveHazardProfile(
            str(
                data.get(
                    "adaptive_hazard_profile", AdaptiveHazardProfile.BALANCED.value
                )
            )
        ),
        adaptive_hazard_stage_policy=AdaptiveHazardStagePolicy(
            str(
                data.get(
                    "adaptive_hazard_stage_policy",
                    AdaptiveHazardStagePolicy.FLAT.value,
                )
            )
        ),
        adaptive_hazard_embedding_guard=AdaptiveHazardEmbeddingGuard(
            str(
                data.get(
                    "adaptive_hazard_embedding_guard",
                    AdaptiveHazardEmbeddingGuard.OFF.value,
                )
            )
        ),
        latent_convergence_every=int(value_or_default(data, "latent_convergence_every", 0)),
        semantic_judge_backend=SemanticJudgeBackend(
            str(data.get("semantic_judge_backend", SemanticJudgeBackend.LLM.value))
        ),
        deferred_intent_every=int(value_or_default(data, "deferred_intent_every", 0)),
        deferred_intent_mode=DeferredIntentMode(
            str(data.get("deferred_intent_mode", DeferredIntentMode.OBSERVE.value))
        ),
        deferred_intent_strategy=DeferredIntentStrategy(
            str(
                data.get(
                    "deferred_intent_strategy", DeferredIntentStrategy.TRIGGER.value
                )
            )
        ),
        deferred_intent_timing=DeferredIntentTiming(
            str(data.get("deferred_intent_timing", DeferredIntentTiming.OFFSET.value))
        ),
        deferred_intent_offset=int(value_or_default(data, "deferred_intent_offset", 3)),
        deferred_intent_grace=int(value_or_default(data, "deferred_intent_grace", 2)),
        deferred_intent_limit=int(value_or_default(data, "deferred_intent_limit", 6)),
        deferred_intent_plan_policy=DeferredIntentPlanPolicy(
            str(
                data.get(
                    "deferred_intent_plan_policy",
                    DeferredIntentPlanPolicy.PERIODIC.value,
                )
            )
        ),
        deferred_intent_plan_budget=int(value_or_default(data, "deferred_intent_plan_budget", 0)),
        deferred_intent_plan_max_new=int(value_or_default(data, "deferred_intent_plan_max_new", 1)),
        deferred_intent_backend=DeferredIntentBackend(
            str(
                data.get(
                    "deferred_intent_backend", DeferredIntentBackend.EXTERNAL.value
                )
            )
        ),
        deferred_intent_latent_injection=DeferredIntentLatentInjection(
            str(
                data.get(
                    "deferred_intent_latent_injection",
                    DeferredIntentLatentInjection.OFF.value,
                )
            )
        ),
        deferred_intent_ablation=DeferredIntentAblation(
            str(data.get("deferred_intent_ablation", DeferredIntentAblation.NONE.value))
        ),
        show_probe_outputs=bool(data.get("show_probe_outputs", False)),
        reply_config=generation_config_from_dict(dict(data.get("reply_config") or {})),
        probe_config=generation_config_from_dict(dict(data.get("probe_config") or {})),
        observer_config=(
            generation_config_from_dict(dict(data["observer_config"]))
            if isinstance(data.get("observer_config"), dict)
            else None
        ),
    )


def serialize_session_state(session: RecursiveConclusionSession) -> dict[str, Any]:
    return {
        "history": [json_ready(message) for message in session.history],
        "memory_capsules": list(session.memory_capsules),
        "conclusion_hypotheses": list(session.conclusion_hypotheses),
        "inquiry_state": session.inquiry_state.to_dict() if session.inquiry_state else None,
        "inquiry_state_turn": session.inquiry_state_turn,
        "latest_conclusion_probe_turn": session.latest_conclusion_probe_turn,
        "latest_conclusion_line": session.latest_conclusion_line,
        "latest_conclusion_keywords": list(session.latest_conclusion_keywords),
        "latest_conclusion_mention_delay_min_turns": (
            session.latest_conclusion_mention_delay_min_turns
        ),
        "latest_conclusion_mention_delay_max_turns": (
            session.latest_conclusion_mention_delay_max_turns
        ),
        "latest_conclusion_mention_hazard_profile": list(
            session.latest_conclusion_mention_hazard_profile
        ),
        "latest_conclusion_mention_likelihood": session.latest_conclusion_mention_likelihood,
        "latest_conclusion_delay_strategy": session.latest_conclusion_delay_strategy,
        "latest_conclusion_delay_signals": list(
            session.latest_conclusion_delay_signals
        ),
        "latest_conclusion_delay_rationale": session.latest_conclusion_delay_rationale,
        "latest_latent_convergence_trace": json_ready(
            session.latest_latent_convergence_trace
        ),
        "latest_embedding_convergence_trace": json_ready(
            session.latest_embedding_convergence_trace
        ),
        "latest_adaptive_hazard_trace": json_ready(
            session.latest_adaptive_hazard_trace
        ),
        "delayed_mentions": [item.to_dict() for item in session.delayed_mentions],
        "deferred_intents": [item.to_dict() for item in session.deferred_intents],
        "turn_index": session.turn_index,
        "next_deferred_intent_index": session.next_deferred_intent_index,
        "next_delayed_mention_index": session.next_delayed_mention_index,
        "deferred_intent_plan_probe_calls": session.deferred_intent_plan_probe_calls,
        "deferred_intent_plan_compact_ok": session._deferred_intent_plan_compact_ok,
        "deferred_intent_scheduler_compact_ok": session._deferred_intent_scheduler_compact_ok,
    }


def restore_session_state(
    session: RecursiveConclusionSession,
    state: dict[str, Any],
) -> RecursiveConclusionSession:
    # Parse on an isolated candidate so any early or late validation failure
    # leaves the live session (including its adapters and log path) untouched.
    candidate = copy.copy(session)
    _restore_session_state(candidate, copy.deepcopy(state))
    session.__dict__.update(candidate.__dict__)
    return session


def _restore_session_state(
    session: RecursiveConclusionSession,
    state: dict[str, Any],
) -> None:
    turn_index = state.get("turn_index", 0)
    if type(turn_index) is not int or turn_index < 0:
        raise ValueError("Invalid turn index in session snapshot.")
    history = state.get("history", [])
    if not isinstance(history, list) or len(history) != turn_index * 2:
        raise ValueError("Session snapshot turn index does not match completed dialogue.")
    for index, item in enumerate(history):
        if (
            not isinstance(item, dict)
            or item.get("role") != ("user" if index % 2 == 0 else "assistant")
            or not isinstance(item.get("content"), str)
        ):
            raise ValueError("Invalid dialogue history in session snapshot.")
    session.history = [
        ChatMessage(
            role=str(item.get("role", "user")),
            content=str(item.get("content", "")),
        )
        for item in history
    ]
    session.memory_capsules = [
        str(item) for item in list(state.get("memory_capsules") or [])
    ]
    session.conclusion_hypotheses = [
        str(item) for item in list(state.get("conclusion_hypotheses") or [])
    ]
    inquiry = state.get("inquiry_state")
    session.inquiry_state = InquiryState.from_dict(inquiry) if inquiry is not None else None
    session.inquiry_state_turn = state.get("inquiry_state_turn")
    if session.inquiry_state is None and session.inquiry_state_turn is not None:
        raise ValueError("Inquiry workpad turn has no workpad in session snapshot.")
    if session.inquiry_state is not None and (
        type(session.inquiry_state_turn) is not int
        or session.inquiry_state_turn < 1
        or session.inquiry_state_turn > turn_index
    ):
        raise ValueError("Invalid inquiry workpad turn in session snapshot.")
    session.latest_conclusion_probe_turn = state.get("latest_conclusion_probe_turn")
    session.latest_conclusion_line = str(state.get("latest_conclusion_line", "") or "")
    session.latest_conclusion_keywords = [
        str(item) for item in list(state.get("latest_conclusion_keywords") or [])
    ]
    session.latest_conclusion_mention_delay_min_turns = state.get(
        "latest_conclusion_mention_delay_min_turns"
    )
    session.latest_conclusion_mention_delay_max_turns = state.get(
        "latest_conclusion_mention_delay_max_turns"
    )
    session.latest_conclusion_mention_hazard_profile = list(
        state.get("latest_conclusion_mention_hazard_profile") or []
    )
    session.latest_conclusion_mention_likelihood = state.get(
        "latest_conclusion_mention_likelihood"
    )
    session.latest_conclusion_delay_strategy = str(
        state.get("latest_conclusion_delay_strategy", "") or ""
    )
    session.latest_conclusion_delay_signals = [
        str(item) for item in list(state.get("latest_conclusion_delay_signals") or [])
    ]
    session.latest_conclusion_delay_rationale = str(
        state.get("latest_conclusion_delay_rationale", "") or ""
    )
    session.latest_latent_convergence_trace = (
        dict(state.get("latest_latent_convergence_trace") or {})
        if isinstance(state.get("latest_latent_convergence_trace"), dict)
        else None
    )
    session.latest_embedding_convergence_trace = (
        dict(state.get("latest_embedding_convergence_trace") or {})
        if isinstance(state.get("latest_embedding_convergence_trace"), dict)
        else None
    )
    session.latest_adaptive_hazard_trace = (
        dict(state.get("latest_adaptive_hazard_trace") or {})
        if isinstance(state.get("latest_adaptive_hazard_trace"), dict)
        else None
    )
    session.delayed_mentions = [
        DelayedMentionItem(**item)
        for item in list(state.get("delayed_mentions") or [])
        if isinstance(item, dict)
    ]
    session.deferred_intents = [
        DeferredIntent(**item)
        for item in list(state.get("deferred_intents") or [])
        if isinstance(item, dict)
    ]
    session.turn_index = turn_index
    session.next_deferred_intent_index = int(
        state.get("next_deferred_intent_index", 1) or 1
    )
    session.next_delayed_mention_index = int(
        state.get("next_delayed_mention_index", 1) or 1
    )
    session.deferred_intent_plan_probe_calls = int(
        state.get("deferred_intent_plan_probe_calls", 0) or 0
    )
    session._deferred_intent_plan_compact_ok = bool(
        state.get("deferred_intent_plan_compact_ok", False)
    )
    session._deferred_intent_scheduler_compact_ok = bool(
        state.get("deferred_intent_scheduler_compact_ok", False)
    )


def failed_event_receipts(discarded_bytes: bytes) -> list[dict[str, Any]]:
    """Keep event identity and provider receipts without copying prompt or reply text."""
    receipts: list[dict[str, Any]] = []
    for raw_line in discarded_bytes.splitlines(keepends=True):
        if not raw_line.strip():
            continue
        receipt: dict[str, Any] = {"raw_line_sha256": hashlib.sha256(raw_line).hexdigest()}
        try:
            event = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            receipts.append(receipt)
            continue
        if not isinstance(event, dict):
            receipts.append(receipt)
            continue
        for key in ("timestamp", "turn_index"):
            value = event.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                receipt[key] = value
        event_type = event.get("event_type")
        if isinstance(event_type, str) and len(event_type) <= 64 and event_type.replace("_", "").isalnum():
            receipt["event_type"] = event_type
        payload = event.get("payload")
        if isinstance(payload, dict):
            safe_payload: dict[str, Any] = {}
            request_id = payload.get("request_id")
            if isinstance(request_id, str) and 0 < len(request_id) <= 128:
                allowed = all(char.isascii() and (char.isalnum() or char in "._:-") for char in request_id)
                secret_prefix = request_id.lower().startswith(
                    ("sk-", "sk_", "hf_", "aiza", "xox", "ghp_", "github_pat_")
                )
                if allowed and not secret_prefix:
                    safe_payload["request_id"] = request_id
            usage = payload.get("usage")
            if isinstance(usage, dict):
                token_usage = {
                    key: value
                    for key, value in usage.items()
                    if isinstance(key, str)
                    and key.lower() in TOKEN_USAGE_FIELDS
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                }
                if token_usage:
                    safe_payload["usage"] = token_usage
            if safe_payload:
                receipt["payload"] = safe_payload
        receipts.append(receipt)
    return receipts


def save_failed_attempt(
    record: PlaytestRecord,
    *,
    user_text: str,
    turn_index: int,
    error: Exception,
    discarded_bytes: bytes,
) -> None:
    """Persist discarded event receipts before removing them from the live session log."""
    receipts = failed_event_receipts(discarded_bytes)
    audit = {
        "recorded_at": time.time(),
        "attempt_turn_index": turn_index,
        "error_type": type(error).__name__,
        "user_text_sha256": hashlib.sha256(user_text.encode("utf-8")).hexdigest(),
        "discarded_bytes_sha256": hashlib.sha256(discarded_bytes).hexdigest(),
        "discarded_event_count": len(receipts),
        "discarded_events": receipts,
    }
    audit_path = record.storage_dir / "failed_attempts.jsonl"
    with audit_path.open("a", encoding="utf-8") as audit_log:
        audit_log.write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + "\n")
        audit_log.flush()
        os.fsync(audit_log.fileno())


def humanize_name(name: str) -> str:
    parts = [part for part in name.replace("-", "_").split("_") if part]
    return " ".join(part.capitalize() for part in parts) if parts else name


def load_script_catalog() -> list[dict[str, Any]]:
    scripts: list[dict[str, Any]] = [
        {
            "id": "free_chat",
            "label": "Free Chat",
            "path": None,
            "system": "",
            "turns": [],
            "evaluation": {},
        }
    ]
    for path in sorted(PROTOCOL_SCRIPTS_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        system, turns = load_script(path)
        evaluation = raw.get("evaluation") if isinstance(raw, dict) else {}
        scripts.append(
            {
                "id": path.stem,
                "label": humanize_name(path.stem),
                "path": str(path.relative_to(ROOT_DIR)),
                "system": system,
                "turns": turns,
                "evaluation": evaluation if isinstance(evaluation, dict) else {},
            }
        )

    preferred_order = {
        "free_chat": 0,
        "shortlist_then_commit": 1,
        "deferred_multi_release": 2,
    }
    scripts.sort(key=lambda item: (preferred_order.get(item["id"], 50), item["label"]))
    return scripts


SCRIPT_CATALOG = load_script_catalog()
SCRIPT_INDEX = {item["id"]: item for item in SCRIPT_CATALOG}

ARM_PRESETS: dict[str, dict[str, str]] = {
    "hold": {
        "label": "Open Inquiry (hold)",
        "adaptive_hazard_policy": AdaptiveHazardPolicy.STATIC.value,
        "adaptive_hazard_stage_policy": AdaptiveHazardStagePolicy.FLAT.value,
    },
    "static": {
        "label": "Static",
        "adaptive_hazard_policy": AdaptiveHazardPolicy.STATIC.value,
        "adaptive_hazard_stage_policy": AdaptiveHazardStagePolicy.FLAT.value,
    },
    "adaptive_flat": {
        "label": "Adaptive Flat",
        "adaptive_hazard_policy": AdaptiveHazardPolicy.ADAPTIVE.value,
        "adaptive_hazard_stage_policy": AdaptiveHazardStagePolicy.FLAT.value,
    },
    "adaptive_kind_aware": {
        "label": "Adaptive Kind-Aware",
        "adaptive_hazard_policy": AdaptiveHazardPolicy.ADAPTIVE.value,
        "adaptive_hazard_stage_policy": AdaptiveHazardStagePolicy.KIND_AWARE.value,
    },
}


def default_experiment_config(
    *,
    script: dict[str, Any],
    arm_preset: str,
    semantic_judge_backend: str,
) -> ExperimentConfig:
    evaluation = dict(script.get("evaluation") or {})
    arm = ARM_PRESETS[arm_preset]
    hold = arm_preset == "hold"
    return ExperimentConfig(
        base_system=str(script.get("system", "") or ""),
        recent_window_messages=8,
        memory_every=2,
        memory_capsule_limit=4,
        memory_word_budget=140,
        conclusion_every=1 if hold else 2,
        conclusion_mode=ConclusionMode.HOLD if hold else ConclusionMode.OBSERVE,
        conclusion_steer_strength=SteerStrength.MEDIUM,
        conclusion_steer_injection=ConclusionSteerInjection.FULL,
        delayed_mention_every=0 if hold else 2,
        delayed_mention_item_limit=4,
        delayed_mention_min_nonconclusion_items=int(
            value_or_default(evaluation, "delayed_mention_min_nonconclusion_items", 2)
        ),
        delayed_mention_min_kind_diversity=int(
            value_or_default(evaluation, "delayed_mention_min_kind_diversity", 3)
        ),
        delayed_mention_diversity_repair=DelayedMentionDiversityRepairPolicy.ON,
        delayed_mention_mode=(
            DelayedMentionMode.OBSERVE if hold else DelayedMentionMode.SOFT_FIRE
        ),
        delayed_mention_fire_prob=0.35,
        delayed_mention_fire_max_items=2,
        delayed_mention_leak_policy=DelayedMentionLeakPolicy.ON,
        delayed_mention_leak_threshold=0.05,
        adaptive_hazard_policy=AdaptiveHazardPolicy(arm["adaptive_hazard_policy"]),
        adaptive_hazard_profile=AdaptiveHazardProfile.BALANCED,
        adaptive_hazard_stage_policy=AdaptiveHazardStagePolicy(
            arm["adaptive_hazard_stage_policy"]
        ),
        adaptive_hazard_embedding_guard=AdaptiveHazardEmbeddingGuard.OFF,
        latent_convergence_every=0 if hold else 1,
        semantic_judge_backend=(
            SemanticJudgeBackend.OFF if hold else SemanticJudgeBackend(semantic_judge_backend)
        ),
        deferred_intent_every=0,
        deferred_intent_mode=DeferredIntentMode.OBSERVE,
        deferred_intent_strategy=DeferredIntentStrategy.TRIGGER,
        deferred_intent_timing=DeferredIntentTiming.OFFSET,
        deferred_intent_offset=3,
        deferred_intent_grace=2,
        deferred_intent_limit=6,
        deferred_intent_plan_policy=DeferredIntentPlanPolicy.PERIODIC,
        deferred_intent_plan_budget=0,
        deferred_intent_plan_max_new=1,
        deferred_intent_backend=DeferredIntentBackend.EXTERNAL,
        deferred_intent_latent_injection=DeferredIntentLatentInjection.OFF,
        deferred_intent_ablation=DeferredIntentAblation.NONE,
        show_probe_outputs=False,
        reply_config=GenerationConfig(
            temperature=0.2, max_tokens=900, timeout_seconds=120
        ),
        probe_config=GenerationConfig(
            temperature=0.0, max_tokens=220, timeout_seconds=120
        ),
    )


def sanitize_turn_payload(
    payload: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    data = dict(json_ready(payload))
    data.pop("system_prompt", None)
    data.pop("inband_state", None)
    for trace_key in ("latent_convergence_trace", "embedding_convergence_trace"):
        trace = data.get(trace_key)
        if isinstance(trace, dict):
            trace.pop("raw", None)
            trace.pop("usage", None)
            trace.pop("request_id", None)
            trace.pop("finish_reason", None)
    return data


def build_public_history(session: RecursiveConclusionSession) -> list[dict[str, Any]]:
    return [
        {"role": message.role, "content": strip_rcl_state(message.content)}
        for message in session.history
    ]


def read_event_log_tail(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    tail = lines[-max(1, limit) :]
    events: list[dict[str, Any]] = []
    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except Exception:
        return str(path)


@dataclass
class PlaytestRecord:
    session_id: str
    title: str
    created_at: float
    updated_at: float
    provider: str
    model: str
    observer_provider: str
    observer_model: str
    embedding_provider: str
    embedding_model: str
    script_id: str
    arm_preset: str
    notes: str
    pending_user_text: str
    last_error: str
    last_result: Optional[dict[str, Any]]
    storage_dir: Path
    log_path: Path
    session: RecursiveConclusionSession


class CreateSessionRequest(BaseModel):
    title: str = ""
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    observer_provider: Optional[str] = None
    observer_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    script_id: str = "free_chat"
    arm_preset: str = "adaptive_kind_aware"
    semantic_judge_backend: str = SemanticJudgeBackend.BOTH.value
    reasoning_effort: Optional[str] = None
    probe_reasoning_effort: Optional[str] = None
    reasoning_mode: Optional[str] = None
    probe_reasoning_mode: Optional[str] = None
    reasoning_context: Optional[str] = None
    probe_reasoning_context: Optional[str] = None
    text_verbosity: Optional[str] = None
    probe_text_verbosity: Optional[str] = None
    observer_reasoning_effort: Optional[str] = None
    observer_reasoning_mode: Optional[str] = None
    observer_reasoning_context: Optional[str] = None
    observer_text_verbosity: Optional[str] = None


class TurnRequest(BaseModel):
    user_text: str = Field(min_length=1)


class NotesRequest(BaseModel):
    notes: str = ""


class CreateReviewSessionRequest(BaseModel):
    eval_set_id: str = Field(min_length=1)
    rater_id: str = Field(min_length=1, max_length=120)


class SubmitJudgmentRequest(BaseModel):
    submission_id: str = Field(min_length=1)
    answers: dict[str, str] = Field(default_factory=dict)
    confidence: str
    evidence: str
    counterevidence: str = ""
    abstain: bool = False


class SessionManager:
    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, PlaytestRecord] = {}
        self._load_errors: list[dict[str, str]] = []
        self._lock = threading.RLock()
        self._load_existing()

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _session_json_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session.json"

    def _load_existing(self) -> None:
        for session_json in sorted(self.sessions_dir.glob("*/session.json")):
            try:
                payload = json.loads(session_json.read_text(encoding="utf-8"))
                schema_version = int(payload.get("version", 1) or 1)
                if schema_version < 1 or schema_version > SESSION_SCHEMA_VERSION:
                    raise ValueError(
                        f"Unsupported saved session schema version: {schema_version}."
                    )
                record = self._restore_record(
                    payload,
                    session_json.parent,
                    schema_version=schema_version,
                )
                if schema_version < SESSION_SCHEMA_VERSION:
                    self._save_record(record, touch_updated_at=False)
            except Exception as exc:
                self._load_errors.append(
                    self._public_load_error(session_json.parent.name, exc)
                )
                continue
            self._records[record.session_id] = record

    @staticmethod
    def _public_load_error(session_id: str, exc: Exception) -> dict[str, str]:
        if isinstance(exc, AmbiguousLegacyProfileError):
            return {
                "session_id": session_id,
                "code": "legacy_profile_ambiguous",
                "message": (
                    "This legacy session predates exact model-profile provenance; "
                    "it was not resumed with inferred generation semantics."
                ),
            }
        if isinstance(exc, ModelProfilePinError):
            return {
                "session_id": session_id,
                "code": "model_profile_mismatch",
                "message": (
                    "The pinned model profile is missing or no longer matches; "
                    "the session was not resumed."
                ),
            }
        if isinstance(exc, OSError):
            return {
                "session_id": session_id,
                "code": "storage_error",
                "message": "The saved session could not be read or migrated.",
            }
        return {
            "session_id": session_id,
            "code": "invalid_snapshot",
            "message": "The saved session is invalid or incompatible and was not resumed.",
        }

    def _restore_record(
        self,
        payload: dict[str, Any],
        storage_dir: Path,
        *,
        schema_version: int,
    ) -> PlaytestRecord:
        session_id = compact_text(str(payload.get("session_id") or ""))
        if not session_id or session_id != storage_dir.name:
            raise ValueError("Saved session id does not match its storage directory.")
        provider = canonical_provider_name(
            str(payload.get("provider", DEFAULT_PROVIDER) or DEFAULT_PROVIDER)
        )
        model = str(payload.get("model", DEFAULT_MODEL) or DEFAULT_MODEL)
        observer_provider = canonical_provider_name(
            str(payload.get("observer_provider", provider) or provider)
        )
        observer_model = str(payload.get("observer_model", model) or model)
        embedding_provider = canonical_embedding_provider_name(
            str(
                payload.get("embedding_provider", DEFAULT_EMBEDDING_PROVIDER)
                or DEFAULT_EMBEDDING_PROVIDER
            )
        )
        embedding_model = str(
            payload.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
            or DEFAULT_EMBEDDING_MODEL
        )
        config = experiment_config_from_dict(dict(payload.get("config") or {}))
        generation_adapter = build_adapter(provider, model)
        observer_generation_adapter = build_adapter(
            observer_provider, observer_model
        )
        if schema_version >= SESSION_SCHEMA_VERSION:
            pinned_configs = {
                "reply_config": config.reply_config,
                "probe_config": config.probe_config,
                "observer_config": config.observer_config,
            }
            for label, generation_config in pinned_configs.items():
                if generation_config is None or (
                    generation_config.model_profile_id is None
                    or generation_config.model_profile_version is None
                ):
                    raise ModelProfilePinError(
                        f"Saved {label} is missing its model profile pin."
                    )
        else:
            legacy_configs = (
                ("reply_config", config.reply_config, generation_adapter),
                ("probe_config", config.probe_config, generation_adapter),
                (
                    "observer_config",
                    config.observer_config,
                    observer_generation_adapter,
                ),
            )
            for label, generation_config, adapter in legacy_configs:
                has_profile_id = (
                    generation_config is not None
                    and generation_config.model_profile_id is not None
                )
                has_profile_version = (
                    generation_config is not None
                    and generation_config.model_profile_version is not None
                )
                if has_profile_id != has_profile_version:
                    raise ModelProfilePinError(
                        f"Saved {label} has an incomplete model profile pin."
                    )
                if not has_profile_id and adapter.model_profile.model_ids:
                    raise AmbiguousLegacyProfileError(
                        f"Saved {label} predates exact profile provenance for "
                        f"{adapter.provider_name}={adapter.model!r}."
                    )
        log_path = storage_dir / "events.jsonl"
        session = RecursiveConclusionSession(
            adapter=generation_adapter,
            observer_adapter=observer_generation_adapter,
            embedding_adapter=(
                build_embedding_adapter(embedding_provider, embedding_model)
                if config.semantic_judge_backend
                in {SemanticJudgeBackend.EMBEDDING, SemanticJudgeBackend.BOTH}
                else None
            ),
            config=config,
            log_path=log_path,
        )
        restore_session_state(session, dict(payload.get("session_state") or {}))
        return PlaytestRecord(
            session_id=session_id,
            title=str(payload.get("title", "") or ""),
            created_at=float(payload.get("created_at", time.time()) or time.time()),
            updated_at=float(payload.get("updated_at", time.time()) or time.time()),
            provider=provider,
            model=model,
            observer_provider=observer_provider,
            observer_model=observer_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            script_id=str(payload.get("script_id", "free_chat") or "free_chat"),
            arm_preset=str(
                payload.get("arm_preset", "adaptive_kind_aware")
                or "adaptive_kind_aware"
            ),
            notes=str(payload.get("notes", "") or ""),
            pending_user_text=str(payload.get("pending_user_text", "") or ""),
            last_error=str(payload.get("last_error", "") or ""),
            last_result=(
                dict(payload.get("last_result") or {})
                if isinstance(payload.get("last_result"), dict)
                else None
            ),
            storage_dir=storage_dir,
            log_path=log_path,
            session=session,
        )

    def _save_record(
        self, record: PlaytestRecord, *, touch_updated_at: bool = True
    ) -> None:
        if touch_updated_at:
            record.updated_at = time.time()
        record.storage_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SESSION_SCHEMA_VERSION,
            "session_id": record.session_id,
            "title": record.title,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "provider": record.provider,
            "model": record.model,
            "observer_provider": record.observer_provider,
            "observer_model": record.observer_model,
            "embedding_provider": record.embedding_provider,
            "embedding_model": record.embedding_model,
            "script_id": record.script_id,
            "arm_preset": record.arm_preset,
            "notes": record.notes,
            "pending_user_text": record.pending_user_text,
            "last_error": record.last_error,
            "last_result": sanitize_turn_payload(record.last_result),
            "config": json_ready(record.session.config),
            "session_state": serialize_session_state(record.session),
        }
        session_json = self._session_json_path(record.session_id)
        temporary = session_json.with_name(
            f".{session_json.name}.{secrets.token_hex(4)}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, session_json)
        finally:
            temporary.unlink(missing_ok=True)

    def _default_title(self, script_id: str, arm_preset: str) -> str:
        script_label = SCRIPT_INDEX.get(script_id, {}).get("label") or humanize_name(
            script_id
        )
        arm_label = ARM_PRESETS.get(arm_preset, {}).get("label") or humanize_name(
            arm_preset
        )
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return f"{script_label} · {arm_label} · {timestamp}"

    def list_summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            items = []
            for record in sorted(
                self._records.values(),
                key=lambda item: item.updated_at,
                reverse=True,
            ):
                items.append(
                    {
                        "session_id": record.session_id,
                        "title": record.title,
                        "script_id": record.script_id,
                        "script_label": SCRIPT_INDEX.get(record.script_id, {}).get(
                            "label"
                        ),
                        "arm_preset": record.arm_preset,
                        "arm_label": ARM_PRESETS.get(record.arm_preset, {}).get(
                            "label"
                        ),
                        "provider": record.provider,
                        "model": record.model,
                        "created_at": record.created_at,
                        "updated_at": record.updated_at,
                        "turn_index": record.session.turn_index,
                        "message_count": len(record.session.history),
                        "pending_user_text": record.pending_user_text,
                        "last_error": record.last_error,
                    }
                )
            return items

    def load_errors(self) -> list[dict[str, str]]:
        with self._lock:
            return [
                dict(item) for item in self._load_errors[:PUBLIC_LOAD_ERROR_LIMIT]
            ]

    def load_error_count(self) -> int:
        with self._lock:
            return len(self._load_errors)

    def get_record(self, session_id: str) -> PlaytestRecord:
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                raise KeyError(session_id)
            return record

    def session_detail(self, session_id: str) -> dict[str, Any]:
        record = self.get_record(session_id)
        script = SCRIPT_INDEX.get(record.script_id, SCRIPT_INDEX["free_chat"])
        last_result = sanitize_turn_payload(record.last_result)
        if last_result is not None:
            # The inspector must show the validated live state after restore,
            # not a second, independently persisted copy of the workpad.
            last_result["inquiry_state"] = (
                record.session.inquiry_state.to_dict()
                if record.session.inquiry_state else None
            )
            last_result["inquiry_state_turn"] = record.session.inquiry_state_turn
        return {
            "session_id": record.session_id,
            "title": record.title,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "provider": record.provider,
            "model": record.model,
            "observer_provider": record.observer_provider,
            "observer_model": record.observer_model,
            "embedding_provider": record.embedding_provider,
            "embedding_model": record.embedding_model,
            "script": script,
            "arm_preset": record.arm_preset,
            "arm_label": ARM_PRESETS.get(record.arm_preset, {}).get("label"),
            "notes": record.notes,
            "pending_user_text": record.pending_user_text,
            "last_error": record.last_error,
            "turn_index": record.session.turn_index,
            "history": build_public_history(record.session),
            "last_result": last_result,
            "config": json_ready(record.session.config),
            "log_path": display_path(record.log_path),
        }

    def create_session(self, request: CreateSessionRequest) -> dict[str, Any]:
        script = SCRIPT_INDEX.get(request.script_id)
        if script is None:
            raise ValueError(f"Unknown script_id: {request.script_id}")
        if request.arm_preset not in ARM_PRESETS:
            raise ValueError(f"Unknown arm preset: {request.arm_preset}")
        semantic_judge_backend = (
            request.semantic_judge_backend or SemanticJudgeBackend.BOTH.value
        )
        if request.arm_preset == "hold":
            semantic_judge_backend = SemanticJudgeBackend.OFF.value

        provider = canonical_provider_name(
            compact_text(request.provider).lower() or DEFAULT_PROVIDER
        )
        model = compact_text(request.model)
        if not model:
            raise ValueError("model must not be blank")
        observer_provider = canonical_provider_name(
            compact_text(request.observer_provider or "").lower() or provider
        )
        observer_model = compact_text(request.observer_model or "") or model
        embedding_provider = compact_text(request.embedding_provider or "").lower()
        embedding_model = compact_text(request.embedding_model or "")
        if semantic_judge_backend in {
            SemanticJudgeBackend.EMBEDDING.value,
            SemanticJudgeBackend.BOTH.value,
        }:
            if not embedding_provider:
                embedding_provider = (
                    "dummy" if provider == "dummy" else DEFAULT_EMBEDDING_PROVIDER
                )
            if not embedding_model:
                embedding_model = (
                    "hash-128"
                    if embedding_provider == "dummy"
                    else DEFAULT_EMBEDDING_MODEL
                )
            embedding_provider = canonical_embedding_provider_name(
                embedding_provider
            )

        session_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        storage_dir = self._session_dir(session_id)
        log_path = storage_dir / "events.jsonl"
        config = default_experiment_config(
            script=script,
            arm_preset=request.arm_preset,
            semantic_judge_backend=semantic_judge_backend,
        )
        config.reply_config = dataclasses.replace(
            config.reply_config,
            reasoning_effort=optional_generation_setting(request.reasoning_effort),
            reasoning_mode=optional_generation_setting(request.reasoning_mode),
            reasoning_context=optional_generation_setting(request.reasoning_context),
            text_verbosity=optional_generation_setting(request.text_verbosity),
        )
        config.probe_config = dataclasses.replace(
            config.probe_config,
            reasoning_effort=optional_generation_setting(
                request.probe_reasoning_effort
            ),
            reasoning_mode=optional_generation_setting(request.probe_reasoning_mode),
            reasoning_context=optional_generation_setting(
                request.probe_reasoning_context
            ),
            text_verbosity=optional_generation_setting(request.probe_text_verbosity),
        )
        observer_controls = {
            "reasoning_effort": optional_generation_setting(
                request.observer_reasoning_effort
            ),
            "reasoning_mode": optional_generation_setting(
                request.observer_reasoning_mode
            ),
            "reasoning_context": optional_generation_setting(
                request.observer_reasoning_context
            ),
            "text_verbosity": optional_generation_setting(
                request.observer_text_verbosity
            ),
        }
        if any(value is not None for value in observer_controls.values()):
            config.observer_config = GenerationConfig(
                temperature=config.probe_config.temperature,
                max_tokens=config.probe_config.max_tokens,
                timeout_seconds=config.probe_config.timeout_seconds,
                **observer_controls,
            )
        session = RecursiveConclusionSession(
            adapter=build_adapter(provider, model),
            observer_adapter=build_adapter(observer_provider, observer_model),
            embedding_adapter=(
                build_embedding_adapter(embedding_provider, embedding_model)
                if semantic_judge_backend
                in {
                    SemanticJudgeBackend.EMBEDDING.value,
                    SemanticJudgeBackend.BOTH.value,
                }
                else None
            ),
            config=config,
            log_path=log_path,
        )
        record = PlaytestRecord(
            session_id=session_id,
            title=compact_text(request.title)
            or self._default_title(request.script_id, request.arm_preset),
            created_at=time.time(),
            updated_at=time.time(),
            provider=provider,
            model=model,
            observer_provider=observer_provider,
            observer_model=observer_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            script_id=request.script_id,
            arm_preset=request.arm_preset,
            notes="",
            pending_user_text="",
            last_error="",
            last_result=None,
            storage_dir=storage_dir,
            log_path=log_path,
            session=session,
        )
        with self._lock:
            self._records[session_id] = record
            self._save_record(record)
        return self.session_detail(session_id)

    def update_notes(self, session_id: str, notes: str) -> dict[str, Any]:
        with self._lock:
            record = self.get_record(session_id)
            record.notes = notes or ""
            self._save_record(record)
        return self.session_detail(session_id)

    def append_turn(self, session_id: str, user_text: str) -> dict[str, Any]:
        cleaned = user_text.strip()
        if not cleaned:
            raise ValueError("user_text must not be empty")
        with self._lock:
            record = self.get_record(session_id)
            state_before = serialize_session_state(record.session)
            log_size_before = record.log_path.stat().st_size if record.log_path.exists() else None
            record.pending_user_text = cleaned
            record.last_error = ""
            self._save_record(record)
            try:
                result = record.session.user_turn(cleaned)
            except Exception as exc:
                discarded_bytes = b""
                if record.log_path.exists():
                    with record.log_path.open("rb") as event_log:
                        event_log.seek(log_size_before or 0)
                        discarded_bytes = event_log.read()
                restore_session_state(record.session, state_before)
                try:
                    save_failed_attempt(
                        record,
                        user_text=cleaned,
                        turn_index=int(state_before["turn_index"]) + 1,
                        error=exc,
                        discarded_bytes=discarded_bytes,
                    )
                except Exception as audit_error:
                    record.last_error = (
                        f"{type(exc).__name__}: failed attempt audit write failed "
                        f"({type(audit_error).__name__}); event log retained"
                    )
                    self._save_record(record)
                    raise RuntimeError("Failed attempt audit write failed; event log retained") from audit_error
                if log_size_before is None:
                    record.log_path.unlink(missing_ok=True)
                else:
                    with record.log_path.open("r+b") as event_log:
                        event_log.truncate(log_size_before)
                record.last_error = safe_turn_error(exc)
                self._save_record(record)
                raise
            record.pending_user_text = ""
            record.last_error = ""
            record.last_result = result
            self._save_record(record)
        return self.session_detail(session_id)


def build_app(
    *,
    sessions_dir: Path,
    allowed_origins: list[str],
    eval_sets_dir: Optional[Path] = None,
    review_sessions_dir: Optional[Path] = None,
    workspace_mode: str = "full",
) -> FastAPI:
    if workspace_mode not in {"full", "review"}:
        raise ValueError("workspace_mode must be 'full' or 'review'.")
    manager = SessionManager(sessions_dir=sessions_dir)
    review_manager = BlindReviewManager(
        eval_sets_dir=(eval_sets_dir or DEFAULT_EVAL_SETS_DIR),
        sessions_dir=(review_sessions_dir or DEFAULT_REVIEW_SESSIONS_DIR),
    )

    reviewer_only = workspace_mode == "review"
    app = FastAPI(
        title="Recursive Conclusion Lab Playtest",
        openapi_url=None if reviewer_only else "/openapi.json",
        docs_url=None if reviewer_only else "/docs",
        redoc_url=None if reviewer_only else "/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if reviewer_only:

        @app.middleware("http")
        async def enforce_reviewer_boundary(request: Request, call_next: Any) -> Any:
            path = request.url.path.rstrip("/")
            if (
                path == "/api/options"
                or path == "/api/sessions"
                or path.startswith("/api/sessions/")
            ):
                return JSONResponse(
                    status_code=404,
                    content={"detail": "Playtest APIs are unavailable in review mode."},
                )
            return await call_next(request)

    @app.get("/api/options")
    def get_options() -> dict[str, Any]:
        return {
            "default_provider": DEFAULT_PROVIDER,
            "default_model": DEFAULT_MODEL,
            "default_embedding_provider": DEFAULT_EMBEDDING_PROVIDER,
            "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
            "default_script_id": "shortlist_then_commit",
            "default_arm_preset": "adaptive_kind_aware",
            "default_semantic_judge_backend": "both",
            "scripts": SCRIPT_CATALOG,
            "arm_presets": [
                {"id": arm_id, **payload} for arm_id, payload in ARM_PRESETS.items()
            ],
            "semantic_judge_backends": [
                backend.value for backend in SemanticJudgeBackend
            ],
            "providers": list(available_provider_names()),
            "embedding_providers": list(available_embedding_provider_names()),
            "model_profiles": model_profile_catalog(),
        }

    @app.get("/api/sessions")
    def list_sessions() -> dict[str, Any]:
        return {
            "sessions": manager.list_summaries(),
            "load_errors": manager.load_errors(),
            "load_error_count": manager.load_error_count(),
        }

    @app.post("/api/sessions")
    def create_session(request: CreateSessionRequest) -> dict[str, Any]:
        try:
            return manager.create_session(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        try:
            return manager.session_detail(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc

    @app.post("/api/sessions/{session_id}/turn")
    def append_turn(session_id: str, request: TurnRequest) -> dict[str, Any]:
        try:
            return manager.append_turn(session_id, request.user_text)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found.") from None
        except ValueError as exc:
            if str(exc) == "user_text must not be empty":
                raise HTTPException(status_code=400, detail="user_text must not be empty") from None
            raise HTTPException(status_code=500, detail=safe_turn_error(exc)) from None
        except Exception as exc:
            raise HTTPException(status_code=500, detail=safe_turn_error(exc)) from None

    @app.put("/api/sessions/{session_id}/notes")
    def update_notes(session_id: str, request: NotesRequest) -> dict[str, Any]:
        try:
            return manager.update_notes(session_id, request.notes)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc

    @app.get("/api/sessions/{session_id}/events")
    def get_session_events(
        session_id: str,
        limit: int = Query(default=80, ge=1, le=400),
    ) -> dict[str, Any]:
        try:
            record = manager.get_record(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc
        return {"events": read_event_log_tail(record.log_path, limit=limit)}

    @app.get("/api/review/sets")
    def list_review_sets() -> dict[str, Any]:
        try:
            return {"sets": review_manager.list_sets()}
        except ReviewValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/review/sessions")
    def list_review_sessions() -> dict[str, Any]:
        return {"sessions": review_manager.list_sessions()}

    @app.post("/api/review/sessions")
    def create_review_session(request: CreateReviewSessionRequest) -> dict[str, Any]:
        try:
            return review_manager.create_session(
                eval_set_id=request.eval_set_id,
                rater_id=request.rater_id,
            )
        except ReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ReviewValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/review/sessions/{session_id}")
    def get_review_session(session_id: str) -> dict[str, Any]:
        try:
            return review_manager.session_detail(session_id)
        except ReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ReviewValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ReviewConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.put("/api/review/sessions/{session_id}/items/{item_id}")
    def submit_review_judgment(
        session_id: str,
        item_id: str,
        request: SubmitJudgmentRequest,
    ) -> dict[str, Any]:
        try:
            return review_manager.submit_judgment(
                session_id=session_id,
                item_id=item_id,
                submission_id=request.submission_id,
                answers=request.answers,
                confidence=request.confidence,
                evidence=request.evidence,
                counterevidence=request.counterevidence,
                abstain=request.abstain,
            )
        except ReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ReviewValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ReviewConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/review/sessions/{session_id}/seal")
    def seal_review_session(session_id: str) -> dict[str, Any]:
        try:
            return review_manager.seal_session(session_id)
        except ReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ReviewValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ReviewConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "workspace_mode": workspace_mode}

    if PLAYTEST_UI_DIST_DIR.exists():
        app.mount(
            "/",
            StaticFiles(directory=PLAYTEST_UI_DIST_DIR, html=True),
            name="playtest-ui",
        )

    return app


def app_from_env() -> FastAPI:
    sessions_dir = Path(
        os.environ.get("RCL_PLAYTEST_SESSIONS_DIR", str(DEFAULT_SESSIONS_DIR))
    ).resolve()
    origins_raw = os.environ.get("RCL_PLAYTEST_ALLOW_ORIGINS", "")
    extra_origins = [item.strip() for item in origins_raw.split(",") if item.strip()]
    eval_sets_dir = Path(
        os.environ.get("RCL_EVAL_SETS_DIR", str(DEFAULT_EVAL_SETS_DIR))
    ).resolve()
    review_sessions_dir = Path(
        os.environ.get("RCL_REVIEW_SESSIONS_DIR", str(DEFAULT_REVIEW_SESSIONS_DIR))
    ).resolve()
    workspace_mode = os.environ.get("RCL_WORKSPACE_MODE", "full").strip().lower()
    return build_app(
        sessions_dir=sessions_dir,
        allowed_origins=DEFAULT_ALLOWED_ORIGINS + extra_origins,
        eval_sets_dir=eval_sets_dir,
        review_sessions_dir=review_sessions_dir,
        workspace_mode=workspace_mode,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local playtest server for Recursive Conclusion Lab."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Expose only blind-review APIs and hide the unblinded Playtest workspace.",
    )
    parser.add_argument(
        "--sessions-dir",
        default=str(DEFAULT_SESSIONS_DIR),
        help="Directory where playtest session snapshots and logs are stored.",
    )
    parser.add_argument(
        "--eval-sets-dir",
        default=str(DEFAULT_EVAL_SETS_DIR),
        help="Directory containing blinded human-evaluation packet sets.",
    )
    parser.add_argument(
        "--review-sessions-dir",
        default=str(DEFAULT_REVIEW_SESSIONS_DIR),
        help="Directory where append-only blind-review sessions are stored.",
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Extra allowed CORS origin. Repeat for multiple origins.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    sessions_dir = Path(args.sessions_dir).resolve()
    os.environ["RCL_PLAYTEST_SESSIONS_DIR"] = str(sessions_dir)
    os.environ["RCL_EVAL_SETS_DIR"] = str(Path(args.eval_sets_dir).resolve())
    os.environ["RCL_REVIEW_SESSIONS_DIR"] = str(
        Path(args.review_sessions_dir).resolve()
    )
    os.environ["RCL_WORKSPACE_MODE"] = "review" if args.review_only else "full"
    os.environ["RCL_PLAYTEST_ALLOW_ORIGINS"] = ",".join(list(args.allow_origin or []))
    if args.reload:
        uvicorn.run("playtest_server:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app_from_env(), host=args.host, port=args.port, reload=False)
    return 0


app = app_from_env()


if __name__ == "__main__":
    raise SystemExit(main())
