#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

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
    RecursiveConclusionSession,
    SemanticJudgeBackend,
    SteerStrength,
    build_adapter,
    build_embedding_adapter,
    compact_text,
    load_script,
    strip_rcl_state,
)


ROOT_DIR = Path(__file__).resolve().parent
PROTOCOL_SCRIPTS_DIR = ROOT_DIR / "protocol_scripts"
PLAYTEST_UI_DIST_DIR = ROOT_DIR / "playtest_ui" / "dist"
DEFAULT_SESSIONS_DIR = ROOT_DIR / "playtest_sessions"

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

DEFAULT_ALLOWED_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


def json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {field.name: json_ready(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def generation_config_from_dict(data: dict[str, Any]) -> GenerationConfig:
    return GenerationConfig(
        temperature=float(data.get("temperature", 0.2) or 0.2),
        max_tokens=int(data.get("max_tokens", 900) or 900),
        timeout_seconds=int(data.get("timeout_seconds", 120) or 120),
    )


def experiment_config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        base_system=str(data.get("base_system", "") or ""),
        recent_window_messages=int(data.get("recent_window_messages", 8) or 8),
        memory_every=int(data.get("memory_every", 3) or 3),
        memory_capsule_limit=int(data.get("memory_capsule_limit", 4) or 4),
        memory_word_budget=int(data.get("memory_word_budget", 140) or 140),
        conclusion_every=int(data.get("conclusion_every", 3) or 3),
        conclusion_mode=ConclusionMode(str(data.get("conclusion_mode", ConclusionMode.OBSERVE.value))),
        conclusion_steer_strength=SteerStrength(
            str(data.get("conclusion_steer_strength", SteerStrength.MEDIUM.value))
        ),
        conclusion_steer_injection=ConclusionSteerInjection(
            str(data.get("conclusion_steer_injection", ConclusionSteerInjection.FULL.value))
        ),
        delayed_mention_every=int(data.get("delayed_mention_every", 0) or 0),
        delayed_mention_item_limit=int(data.get("delayed_mention_item_limit", 3) or 3),
        delayed_mention_min_nonconclusion_items=int(
            data.get("delayed_mention_min_nonconclusion_items", 1) or 1
        ),
        delayed_mention_min_kind_diversity=int(
            data.get("delayed_mention_min_kind_diversity", 2) or 2
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
        delayed_mention_fire_prob=float(data.get("delayed_mention_fire_prob", 0.35) or 0.35),
        delayed_mention_fire_max_items=int(data.get("delayed_mention_fire_max_items", 2) or 2),
        delayed_mention_leak_policy=DelayedMentionLeakPolicy(
            str(data.get("delayed_mention_leak_policy", DelayedMentionLeakPolicy.ON.value))
        ),
        delayed_mention_leak_threshold=float(
            data.get("delayed_mention_leak_threshold", 0.05) or 0.05
        ),
        adaptive_hazard_policy=AdaptiveHazardPolicy(
            str(data.get("adaptive_hazard_policy", AdaptiveHazardPolicy.ADAPTIVE.value))
        ),
        adaptive_hazard_profile=AdaptiveHazardProfile(
            str(data.get("adaptive_hazard_profile", AdaptiveHazardProfile.BALANCED.value))
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
        latent_convergence_every=int(data.get("latent_convergence_every", 0) or 0),
        semantic_judge_backend=SemanticJudgeBackend(
            str(data.get("semantic_judge_backend", SemanticJudgeBackend.LLM.value))
        ),
        deferred_intent_every=int(data.get("deferred_intent_every", 0) or 0),
        deferred_intent_mode=DeferredIntentMode(
            str(data.get("deferred_intent_mode", DeferredIntentMode.OBSERVE.value))
        ),
        deferred_intent_strategy=DeferredIntentStrategy(
            str(data.get("deferred_intent_strategy", DeferredIntentStrategy.TRIGGER.value))
        ),
        deferred_intent_timing=DeferredIntentTiming(
            str(data.get("deferred_intent_timing", DeferredIntentTiming.OFFSET.value))
        ),
        deferred_intent_offset=int(data.get("deferred_intent_offset", 3) or 3),
        deferred_intent_grace=int(data.get("deferred_intent_grace", 2) or 2),
        deferred_intent_limit=int(data.get("deferred_intent_limit", 6) or 6),
        deferred_intent_plan_policy=DeferredIntentPlanPolicy(
            str(
                data.get(
                    "deferred_intent_plan_policy",
                    DeferredIntentPlanPolicy.PERIODIC.value,
                )
            )
        ),
        deferred_intent_plan_budget=int(data.get("deferred_intent_plan_budget", 0) or 0),
        deferred_intent_plan_max_new=int(data.get("deferred_intent_plan_max_new", 1) or 1),
        deferred_intent_backend=DeferredIntentBackend(
            str(data.get("deferred_intent_backend", DeferredIntentBackend.EXTERNAL.value))
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
    )


def serialize_session_state(session: RecursiveConclusionSession) -> dict[str, Any]:
    return {
        "history": [json_ready(message) for message in session.history],
        "memory_capsules": list(session.memory_capsules),
        "conclusion_hypotheses": list(session.conclusion_hypotheses),
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
        "latest_conclusion_delay_signals": list(session.latest_conclusion_delay_signals),
        "latest_conclusion_delay_rationale": session.latest_conclusion_delay_rationale,
        "latest_latent_convergence_trace": json_ready(session.latest_latent_convergence_trace),
        "latest_embedding_convergence_trace": json_ready(session.latest_embedding_convergence_trace),
        "latest_adaptive_hazard_trace": json_ready(session.latest_adaptive_hazard_trace),
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
    session.history = [
        ChatMessage(
            role=str(item.get("role", "user")),
            content=str(item.get("content", "")),
        )
        for item in list(state.get("history") or [])
        if isinstance(item, dict)
    ]
    session.memory_capsules = [str(item) for item in list(state.get("memory_capsules") or [])]
    session.conclusion_hypotheses = [
        str(item) for item in list(state.get("conclusion_hypotheses") or [])
    ]
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
    session.turn_index = int(state.get("turn_index", 0) or 0)
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
    return session


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
    return ExperimentConfig(
        base_system=str(script.get("system", "") or ""),
        recent_window_messages=8,
        memory_every=2,
        memory_capsule_limit=4,
        memory_word_budget=140,
        conclusion_every=2,
        conclusion_mode=ConclusionMode.OBSERVE,
        conclusion_steer_strength=SteerStrength.MEDIUM,
        conclusion_steer_injection=ConclusionSteerInjection.FULL,
        delayed_mention_every=2,
        delayed_mention_item_limit=4,
        delayed_mention_min_nonconclusion_items=int(
            evaluation.get("delayed_mention_min_nonconclusion_items", 2) or 2
        ),
        delayed_mention_min_kind_diversity=int(
            evaluation.get("delayed_mention_min_kind_diversity", 3) or 3
        ),
        delayed_mention_diversity_repair=DelayedMentionDiversityRepairPolicy.ON,
        delayed_mention_mode=DelayedMentionMode.SOFT_FIRE,
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
        latent_convergence_every=1,
        semantic_judge_backend=SemanticJudgeBackend(semantic_judge_backend),
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
        reply_config=GenerationConfig(temperature=0.2, max_tokens=900, timeout_seconds=120),
        probe_config=GenerationConfig(temperature=0.0, max_tokens=220, timeout_seconds=120),
    )


def sanitize_turn_payload(payload: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
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


class TurnRequest(BaseModel):
    user_text: str = Field(min_length=1)


class NotesRequest(BaseModel):
    notes: str = ""


class SessionManager:
    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, PlaytestRecord] = {}
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
                record = self._restore_record(payload, session_json.parent)
            except Exception:
                continue
            self._records[record.session_id] = record

    def _restore_record(self, payload: dict[str, Any], storage_dir: Path) -> PlaytestRecord:
        provider = str(payload.get("provider", DEFAULT_PROVIDER) or DEFAULT_PROVIDER)
        model = str(payload.get("model", DEFAULT_MODEL) or DEFAULT_MODEL)
        observer_provider = str(payload.get("observer_provider", provider) or provider)
        observer_model = str(payload.get("observer_model", model) or model)
        embedding_provider = str(
            payload.get("embedding_provider", DEFAULT_EMBEDDING_PROVIDER)
            or DEFAULT_EMBEDDING_PROVIDER
        )
        embedding_model = str(
            payload.get("embedding_model", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL
        )
        config = experiment_config_from_dict(dict(payload.get("config") or {}))
        log_path = storage_dir / "events.jsonl"
        session = RecursiveConclusionSession(
            adapter=build_adapter(provider, model),
            observer_adapter=build_adapter(observer_provider, observer_model),
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
            session_id=str(payload.get("session_id")),
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
            arm_preset=str(payload.get("arm_preset", "adaptive_kind_aware") or "adaptive_kind_aware"),
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

    def _save_record(self, record: PlaytestRecord) -> None:
        record.updated_at = time.time()
        record.storage_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
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
        self._session_json_path(record.session_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _default_title(self, script_id: str, arm_preset: str) -> str:
        script_label = SCRIPT_INDEX.get(script_id, {}).get("label") or humanize_name(script_id)
        arm_label = ARM_PRESETS.get(arm_preset, {}).get("label") or humanize_name(arm_preset)
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
                        "script_label": SCRIPT_INDEX.get(record.script_id, {}).get("label"),
                        "arm_preset": record.arm_preset,
                        "arm_label": ARM_PRESETS.get(record.arm_preset, {}).get("label"),
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

    def get_record(self, session_id: str) -> PlaytestRecord:
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                raise KeyError(session_id)
            return record

    def session_detail(self, session_id: str) -> dict[str, Any]:
        record = self.get_record(session_id)
        script = SCRIPT_INDEX.get(record.script_id, SCRIPT_INDEX["free_chat"])
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
            "last_result": sanitize_turn_payload(record.last_result),
            "config": json_ready(record.session.config),
            "log_path": display_path(record.log_path),
        }

    def create_session(self, request: CreateSessionRequest) -> dict[str, Any]:
        script = SCRIPT_INDEX.get(request.script_id)
        if script is None:
            raise ValueError(f"Unknown script_id: {request.script_id}")
        if request.arm_preset not in ARM_PRESETS:
            raise ValueError(f"Unknown arm preset: {request.arm_preset}")
        semantic_judge_backend = request.semantic_judge_backend or SemanticJudgeBackend.BOTH.value

        provider = compact_text(request.provider).lower() or DEFAULT_PROVIDER
        model = compact_text(request.model) or DEFAULT_MODEL
        observer_provider = (
            compact_text(request.observer_provider or "").lower() or provider
        )
        observer_model = compact_text(request.observer_model or "") or model
        embedding_provider = compact_text(request.embedding_provider or "").lower()
        embedding_model = compact_text(request.embedding_model or "")
        if semantic_judge_backend in {SemanticJudgeBackend.EMBEDDING.value, SemanticJudgeBackend.BOTH.value}:
            if not embedding_provider:
                embedding_provider = (
                    "dummy" if provider == "dummy" else DEFAULT_EMBEDDING_PROVIDER
                )
            if not embedding_model:
                embedding_model = "hash-128" if embedding_provider == "dummy" else DEFAULT_EMBEDDING_MODEL

        session_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        storage_dir = self._session_dir(session_id)
        log_path = storage_dir / "events.jsonl"
        config = default_experiment_config(
            script=script,
            arm_preset=request.arm_preset,
            semantic_judge_backend=semantic_judge_backend,
        )
        session = RecursiveConclusionSession(
            adapter=build_adapter(provider, model),
            observer_adapter=build_adapter(observer_provider, observer_model),
            embedding_adapter=(
                build_embedding_adapter(embedding_provider, embedding_model)
                if semantic_judge_backend
                in {SemanticJudgeBackend.EMBEDDING.value, SemanticJudgeBackend.BOTH.value}
                else None
            ),
            config=config,
            log_path=log_path,
        )
        record = PlaytestRecord(
            session_id=session_id,
            title=compact_text(request.title) or self._default_title(request.script_id, request.arm_preset),
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
            record.pending_user_text = cleaned
            record.last_error = ""
            self._save_record(record)
            try:
                result = record.session.user_turn(cleaned)
            except Exception as exc:
                record.last_error = str(exc)
                self._save_record(record)
                raise
            record.pending_user_text = ""
            record.last_error = ""
            record.last_result = result
            self._save_record(record)
        return self.session_detail(session_id)


def build_app(*, sessions_dir: Path, allowed_origins: list[str]) -> FastAPI:
    manager = SessionManager(sessions_dir=sessions_dir)

    app = FastAPI(title="Recursive Conclusion Lab Playtest")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/options")
    def get_options() -> dict[str, Any]:
        return {
            "default_provider": DEFAULT_PROVIDER,
            "default_model": DEFAULT_MODEL,
            "default_embedding_provider": DEFAULT_EMBEDDING_PROVIDER,
            "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
            "scripts": SCRIPT_CATALOG,
            "arm_presets": [
                {"id": arm_id, **payload}
                for arm_id, payload in ARM_PRESETS.items()
            ],
            "semantic_judge_backends": [
                backend.value for backend in SemanticJudgeBackend
            ],
            "providers": ["openai", "anthropic", "mistral", "gemini", "hf", "dummy"],
        }

    @app.get("/api/sessions")
    def list_sessions() -> dict[str, Any]:
        return {"sessions": manager.list_summaries()}

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
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    if PLAYTEST_UI_DIST_DIR.exists():
        app.mount("/", StaticFiles(directory=PLAYTEST_UI_DIST_DIR, html=True), name="playtest-ui")

    return app


def app_from_env() -> FastAPI:
    sessions_dir = Path(
        os.environ.get("RCL_PLAYTEST_SESSIONS_DIR", str(DEFAULT_SESSIONS_DIR))
    ).resolve()
    origins_raw = os.environ.get("RCL_PLAYTEST_ALLOW_ORIGINS", "")
    extra_origins = [item.strip() for item in origins_raw.split(",") if item.strip()]
    return build_app(
        sessions_dir=sessions_dir,
        allowed_origins=DEFAULT_ALLOWED_ORIGINS + extra_origins,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local playtest server for Recursive Conclusion Lab.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--sessions-dir",
        default=str(DEFAULT_SESSIONS_DIR),
        help="Directory where playtest session snapshots and logs are stored.",
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
    os.environ["RCL_PLAYTEST_ALLOW_ORIGINS"] = ",".join(list(args.allow_origin or []))
    if args.reload:
        uvicorn.run("playtest_server:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app_from_env(), host=args.host, port=args.port, reload=False)
    return 0


app = app_from_env()


if __name__ == "__main__":
    raise SystemExit(main())
