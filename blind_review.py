#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Any


SCHEMA_VERSION = 1
CONFIDENCE_LEVELS = {"low", "medium", "high"}
DISPLAY_LABELS = ("A", "B")
PAIRWISE_CHOICES = {"A", "B", "Tie"}


class ReviewError(Exception):
    pass


class ReviewNotFoundError(ReviewError):
    pass


class ReviewValidationError(ReviewError):
    pass


class ReviewConflictError(ReviewError):
    pass


def compact_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def repair_jsonl_tail(path: Path) -> None:
    if not path.exists():
        return
    with path.open("rb+") as stream:
        content = stream.read()
        if not content or content.endswith((b"\n", b"\r")):
            return
        line_start = max(content.rfind(b"\n"), content.rfind(b"\r")) + 1
        tail = content[line_start:]
        try:
            parsed = json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            tail_digest = hashlib.sha256(tail).hexdigest()
            quarantine = path.with_name(
                f"{path.name}.torn-{time.time_ns()}-{tail_digest[:12]}.bin"
            )
            with quarantine.open("xb") as evidence:
                evidence.write(tail)
                evidence.flush()
                os.fsync(evidence.fileno())
            stream.seek(line_start)
            stream.truncate()
            recovery_event = {
                "event_type": "log_tail_recovered",
                "event_id": f"event-{secrets.token_hex(8)}",
                "created_at": time.time(),
                "payload": {
                    "byte_count": len(tail),
                    "quarantine_file": quarantine.name,
                    "sha256": tail_digest,
                },
            }
            stream.write((canonical_json(recovery_event) + "\n").encode("utf-8"))
        else:
            if not isinstance(parsed, dict):
                raise ReviewValidationError(
                    f"Expected an object on the final JSONL line: {path}"
                )
            stream.seek(0, os.SEEK_END)
            stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    repair_jsonl_tail(path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(payload) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewValidationError(f"Cannot read JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ReviewValidationError(f"Expected a JSON object: {path}")
    return payload


def load_jsonl(
    path: Path,
    *,
    tolerate_torn_tail: bool = False,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReviewValidationError(f"Cannot read JSONL file: {path}") from exc
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if (
                tolerate_torn_tail
                and line_number == len(lines)
                and not text.endswith(("\n", "\r"))
            ):
                break
            raise ReviewValidationError(
                f"Invalid JSONL at {path}:{line_number}"
            ) from exc
        if not isinstance(payload, dict):
            raise ReviewValidationError(f"Expected an object at {path}:{line_number}")
        items.append(payload)
    return items


def validated_key_items(
    blind_key: dict[str, Any],
    *,
    expected_item_ids: set[str],
) -> dict[str, dict[str, Any]]:
    raw_items = blind_key.get("items")
    if not isinstance(raw_items, dict):
        raw_items = {
            str(key): value
            for key, value in blind_key.items()
            if not str(key).startswith("_") and isinstance(value, dict)
        }
    actual_item_ids = {str(item_id) for item_id in raw_items}
    if actual_item_ids != expected_item_ids:
        missing = sorted(expected_item_ids - actual_item_ids)
        extra = sorted(actual_item_ids - expected_item_ids)
        raise ReviewValidationError(
            f"blind_key.json item coverage mismatch (missing={missing}, extra={extra})."
        )

    normalized: dict[str, dict[str, Any]] = {}
    for item_id in sorted(expected_item_ids):
        raw_item = raw_items.get(item_id)
        if not isinstance(raw_item, dict):
            raise ReviewValidationError(
                f"blind_key.json is missing an object for {item_id}."
            )
        raw_mapping = raw_item.get("label_to_arm")
        if not isinstance(raw_mapping, dict) or set(raw_mapping) != set(DISPLAY_LABELS):
            raise ReviewValidationError(
                f"blind_key.json label map for {item_id} must contain exactly A and B."
            )
        mapping = {
            label: compact_text(raw_mapping.get(label)) for label in DISPLAY_LABELS
        }
        if any(not arm for arm in mapping.values()) or len(set(mapping.values())) != 2:
            raise ReviewValidationError(
                f"blind_key.json label map for {item_id} needs two distinct, non-empty arms."
            )
        arm_pair = raw_item.get("arm_pair_sorted")
        if arm_pair is not None and sorted(
            compact_text(arm) for arm in arm_pair
        ) != sorted(mapping.values()):
            raise ReviewValidationError(
                f"blind_key.json arm pair for {item_id} does not match its label map."
            )
        raw_labels = raw_item.get("labels")
        if raw_labels is not None:
            if not isinstance(raw_labels, dict) or set(raw_labels) != set(
                DISPLAY_LABELS
            ):
                raise ReviewValidationError(
                    f"blind_key.json source labels for {item_id} must contain A and B."
                )
            for label in DISPLAY_LABELS:
                source = raw_labels[label]
                if not isinstance(source, dict):
                    raise ReviewValidationError(
                        f"blind_key.json source label {item_id}/{label} must be an object."
                    )
                source_arm = compact_text(source.get("arm"))
                if source_arm and source_arm != mapping[label]:
                    raise ReviewValidationError(
                        f"blind_key.json source arm for {item_id}/{label} is inconsistent."
                    )
                source_hash = compact_text(source.get("source_sha256"))
                if source_hash and (
                    len(source_hash) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in source_hash.lower()
                    )
                ):
                    raise ReviewValidationError(
                        f"blind_key.json source hash for {item_id}/{label} is invalid."
                    )
        normalized[item_id] = {**raw_item, "label_to_arm": mapping}
    return normalized


def normalize_questions(raw_questions: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ReviewValidationError("A review set needs at least one rubric question.")
    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_questions:
        if not isinstance(raw, dict):
            raise ReviewValidationError("Rubric questions must be objects.")
        question_id = compact_text(raw.get("id"))
        prompt = compact_text(raw.get("prompt"))
        choices = [compact_text(choice) for choice in list(raw.get("choices") or [])]
        if not question_id or not prompt or not choices:
            raise ReviewValidationError(
                "Each rubric question needs id, prompt, and choices."
            )
        if question_id in seen_ids:
            raise ReviewValidationError(f"Duplicate rubric question id: {question_id}")
        if any(not choice for choice in choices) or len(set(choices)) != len(choices):
            raise ReviewValidationError(
                f"Invalid choices for rubric question: {question_id}"
            )
        seen_ids.add(question_id)
        questions.append({"id": question_id, "prompt": prompt, "choices": choices})
    return questions


def sanitize_item(
    raw_item: dict[str, Any],
    *,
    default_questions: list[dict[str, Any]],
) -> dict[str, Any]:
    item_id = compact_text(raw_item.get("item_id"))
    if not item_id:
        raise ReviewValidationError("Every review item needs an item_id.")
    raw_labels = raw_item.get("labels")
    if not isinstance(raw_labels, dict):
        raise ReviewValidationError(f"Review item {item_id} is missing labels.")
    labels: dict[str, dict[str, str]] = {}
    for label in DISPLAY_LABELS:
        raw_label = raw_labels.get(label)
        if not isinstance(raw_label, dict):
            raise ReviewValidationError(
                f"Review item {item_id} is missing label {label}."
            )
        transcript = str(raw_label.get("transcript") or "").strip()
        if not transcript:
            raise ReviewValidationError(
                f"Review item {item_id} label {label} has an empty transcript."
            )
        labels[label] = {"transcript": transcript}
    questions = normalize_questions(raw_item.get("questions") or default_questions)
    return {
        "item_id": item_id,
        "scenario_name": compact_text(raw_item.get("scenario_name")),
        "scenario_display_name": compact_text(
            raw_item.get("scenario_display_name") or raw_item.get("scenario_name")
        ),
        "run_index": raw_item.get("run_index"),
        "labels": labels,
        "questions": questions,
    }


@dataclass(frozen=True)
class ReviewSet:
    eval_set_id: str
    path: Path
    manifest: dict[str, Any]
    items: tuple[dict[str, Any], ...]
    review_bundle_digest: str
    rubric_digest: str
    blind_key_digest: str
    source_digest: str

    @property
    def items_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(item["item_id"]): item for item in self.items}

    def summary(self) -> dict[str, Any]:
        return {
            "eval_set_id": self.eval_set_id,
            "title": self.manifest["title"],
            "rubric_version": self.manifest["rubric_version"],
            "item_count": len(self.items),
            "questions": self.manifest["questions"],
            "scenario_item_counts": self.manifest["scenario_item_counts"],
            "review_bundle_digest": self.review_bundle_digest,
            "rubric_digest": self.rubric_digest,
        }


class ReviewSetCatalog:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        direct = self.root / "eval_items.jsonl"
        candidates = (
            [direct] if direct.exists() else sorted(self.root.rglob("eval_items.jsonl"))
        )
        for items_path in candidates:
            set_dir = items_path.parent.resolve()
            if (
                not (set_dir / "manifest.json").exists()
                or not (set_dir / "READY.json").exists()
            ):
                continue
            if set_dir == self.root:
                eval_set_id = self.root.name
            else:
                eval_set_id = set_dir.relative_to(self.root).as_posix()
            paths[eval_set_id] = set_dir
        return paths

    def list_sets(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for eval_set_id in sorted(self._paths()):
            summaries.append(self.load(eval_set_id).summary())
        return summaries

    def load(self, eval_set_id: str) -> ReviewSet:
        set_dir = self._paths().get(eval_set_id)
        if set_dir is None:
            raise ReviewNotFoundError(f"Review set not found: {eval_set_id}")
        raw_manifest = load_json(set_dir / "manifest.json")
        manifest_questions = normalize_questions(raw_manifest.get("questions"))
        for question in manifest_questions:
            if set(question["choices"]) != PAIRWISE_CHOICES:
                raise ReviewValidationError(
                    f"Blind pairwise question {question['id']} must use A, B, and Tie."
                )
        raw_items = load_jsonl(set_dir / "eval_items.jsonl")
        items = tuple(
            sanitize_item(raw_item, default_questions=manifest_questions)
            for raw_item in raw_items
        )
        item_ids = [str(item["item_id"]) for item in items]
        if not items:
            raise ReviewValidationError(f"Review set {eval_set_id} has no items.")
        if len(item_ids) != len(set(item_ids)):
            raise ReviewValidationError(
                f"Review set {eval_set_id} has duplicate item ids."
            )
        for item in items:
            if item["questions"] != manifest_questions:
                raise ReviewValidationError(
                    f"Review item {item['item_id']} does not match the manifest rubric."
                )
        declared_count = raw_manifest.get("item_count")
        if declared_count is not None and int(declared_count) != len(items):
            raise ReviewValidationError(
                f"Review set {eval_set_id} declares {declared_count} items but has {len(items)}."
            )
        scenario_counts: dict[str, int] = {}
        for item in items:
            scenario = str(item.get("scenario_name") or "unspecified")
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        rubric_version = (
            compact_text(raw_manifest.get("rubric_version")) or "unversioned"
        )
        public_manifest = {
            "schema": compact_text(raw_manifest.get("schema"))
            or "rcl.blind_pairwise_eval_set.v1",
            "title": compact_text(raw_manifest.get("title")) or eval_set_id,
            "rubric_version": rubric_version,
            "item_count": len(items),
            "questions": manifest_questions,
            "scenario_item_counts": scenario_counts,
        }
        review_bundle = {"manifest": public_manifest, "items": items}
        review_bundle_digest = json_digest(review_bundle)
        ready = load_json(set_dir / "READY.json")
        if ready.get("schema") != "rcl.blind_pairwise_ready.v1":
            raise ReviewValidationError(
                f"Review set {eval_set_id} has an invalid readiness marker."
            )
        if ready.get("review_bundle_digest") != review_bundle_digest:
            raise ReviewValidationError(
                f"Review set {eval_set_id} is not a completely published packet."
            )
        blind_key_path = set_dir / "blind_key.json"
        if not blind_key_path.exists():
            raise ReviewValidationError(
                f"Review set {eval_set_id} is missing blind_key.json."
            )
        blind_key = load_json(blind_key_path)
        private_meta = blind_key.get("_meta")
        if not isinstance(private_meta, dict):
            raise ReviewValidationError(
                f"Review set {eval_set_id} has no private key metadata."
            )
        if private_meta.get("review_bundle_digest") != review_bundle_digest:
            raise ReviewValidationError(
                f"Review set {eval_set_id} has a blind key for a different public packet."
            )
        source_digest = compact_text(private_meta.get("source_digest"))
        if len(source_digest) != 64 or any(
            character not in "0123456789abcdef" for character in source_digest.lower()
        ):
            raise ReviewValidationError(
                f"Review set {eval_set_id} has an invalid source digest."
            )
        if ready.get("source_digest") != source_digest:
            raise ReviewValidationError(
                f"Review set {eval_set_id} readiness marker has a different source digest."
            )
        validated_key_items(blind_key, expected_item_ids=set(item_ids))
        return ReviewSet(
            eval_set_id=eval_set_id,
            path=set_dir,
            manifest=public_manifest,
            items=items,
            review_bundle_digest=review_bundle_digest,
            rubric_digest=json_digest(
                {"rubric_version": rubric_version, "questions": manifest_questions}
            ),
            blind_key_digest=file_digest(blind_key_path),
            source_digest=source_digest,
        )


class BlindReviewManager:
    def __init__(self, *, eval_sets_dir: Path, sessions_dir: Path) -> None:
        self.catalog = ReviewSetCatalog(eval_sets_dir)
        self.sessions_dir = sessions_dir.resolve()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _session_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session.json"

    def _events_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "judgments.jsonl"

    def _load_session(self, session_id: str) -> dict[str, Any]:
        path = self._session_path(session_id)
        if not path.exists():
            raise ReviewNotFoundError(f"Review session not found: {session_id}")
        session = load_json(path)
        events = self._load_events(session_id)
        for event in events:
            if event.get("event_type") == "review_sealed":
                payload = event.get("payload") or {}
                session["sealed_at"] = payload.get("sealed_at")
                session["receipt_path"] = payload.get("receipt_path")
                session["result_path"] = payload.get("result_path")
        return session

    def _load_events(self, session_id: str) -> list[dict[str, Any]]:
        path = self._events_path(session_id)
        if not path.exists():
            return []
        return load_jsonl(path, tolerate_torn_tail=True)

    def _save_session(self, session: dict[str, Any]) -> None:
        atomic_write_json(self._session_path(str(session["session_id"])), session)

    def _current_responses(
        self, events: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        responses: dict[str, dict[str, Any]] = {}
        for event in events:
            if event.get("event_type") != "judgment_submitted":
                continue
            item_id = compact_text(event.get("item_id"))
            if not item_id:
                continue
            responses[item_id] = {
                **dict(event.get("payload") or {}),
                "event_id": event.get("event_id"),
                "revision": event.get("revision"),
                "submitted_at": event.get("created_at"),
            }
        return responses

    def _assignment(
        self,
        review_set: ReviewSet,
        rater_id: str,
    ) -> tuple[list[str], dict[str, dict[str, str]]]:
        seed_prefix = f"{review_set.review_bundle_digest}:{rater_id}"
        item_order = sorted(
            (str(item["item_id"]) for item in review_set.items),
            key=lambda item_id: hashlib.sha256(
                f"{seed_prefix}:order:{item_id}".encode("utf-8")
            ).hexdigest(),
        )
        presentation: dict[str, dict[str, str]] = {}
        for item_id in item_order:
            marker = hashlib.sha256(
                f"{seed_prefix}:labels:{item_id}".encode("utf-8")
            ).digest()[0]
            presentation[item_id] = (
                {"A": "B", "B": "A"} if marker % 2 else {"A": "A", "B": "B"}
            )
        return item_order, presentation

    def _validate_dataset(self, session: dict[str, Any]) -> ReviewSet:
        review_set = self.catalog.load(str(session["eval_set_id"]))
        if review_set.review_bundle_digest != session.get("review_bundle_digest"):
            raise ReviewConflictError(
                "The review packet changed after this session was created."
            )
        if review_set.blind_key_digest != session.get("blind_key_digest"):
            raise ReviewConflictError(
                "The private blind key changed after this session was created."
            )
        if review_set.source_digest != session.get("source_digest"):
            raise ReviewConflictError(
                "The source evidence changed after this session was created."
            )
        events = self._load_events(str(session["session_id"]))
        creation_events = [
            event for event in events if event.get("event_type") == "session_created"
        ]
        if len(creation_events) != 1:
            raise ReviewConflictError("The review session creation record is invalid.")
        creation_payload = creation_events[0].get("payload") or {}
        anchored_fields = (
            "eval_set_id",
            "rater_id",
            "review_bundle_digest",
            "rubric_digest",
            "blind_key_digest",
            "source_digest",
        )
        if any(
            session.get(field) != creation_payload.get(field)
            for field in anchored_fields
        ):
            raise ReviewConflictError(
                "The review session identity changed after creation."
            )
        expected_order, expected_presentation = self._assignment(
            review_set, str(session["rater_id"])
        )
        if session.get("item_order") != expected_order:
            raise ReviewConflictError(
                "The review item assignment changed after creation."
            )
        if session.get("presentation") != expected_presentation:
            raise ReviewConflictError(
                "The blinded presentation changed after creation."
            )
        return review_set

    def _verified_sealed_artifacts(
        self,
        session: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        session_id = str(session["session_id"])
        seal_events = [
            event for event in events if event.get("event_type") == "review_sealed"
        ]
        if len(seal_events) != 1:
            raise ReviewConflictError("The sealed review event record is invalid.")
        seal_event = seal_events[0]
        receipt_path = self._session_dir(session_id) / "seal_receipt.json"
        result_path = self._session_dir(session_id) / "unblinded_results.json"
        if not receipt_path.exists() or not result_path.exists():
            raise ReviewConflictError("A sealed review artifact is missing.")
        receipt = load_json(receipt_path)
        results = load_json(result_path)

        claimed_receipt_digest = compact_text(receipt.get("receipt_digest"))
        unsigned_receipt = dict(receipt)
        unsigned_receipt.pop("receipt_digest", None)
        if (
            not claimed_receipt_digest
            or json_digest(unsigned_receipt) != claimed_receipt_digest
        ):
            raise ReviewConflictError("The seal receipt digest does not verify.")
        if json_digest(results) != receipt.get("results_digest"):
            raise ReviewConflictError("The unblinded results digest does not verify.")
        event_payload = seal_event.get("payload") or {}
        if event_payload.get("receipt_digest") != claimed_receipt_digest:
            raise ReviewConflictError("The seal event does not match its receipt.")

        protected_fields = (
            "session_id",
            "eval_set_id",
            "rater_id",
            "review_bundle_digest",
            "rubric_digest",
            "source_digest",
            "blind_key_digest",
            "sealed_at",
        )
        for field in protected_fields:
            if receipt.get(field) != session.get(field) or results.get(
                field
            ) != session.get(field):
                raise ReviewConflictError(
                    f"The sealed artifacts do not match session field '{field}'."
                )
        if receipt.get("item_count") != len(session["item_order"]):
            raise ReviewConflictError("The seal receipt item count does not verify.")
        if len(results.get("judgments") or []) != len(session["item_order"]):
            raise ReviewConflictError("The sealed judgment count does not verify.")

        event_log_path = self._events_path(session_id)
        try:
            raw_lines = event_log_path.read_bytes().splitlines(keepends=True)
            parsed_lines = [json.loads(line) for line in raw_lines]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReviewConflictError(
                "The sealed event log cannot be verified."
            ) from exc
        seal_positions = [
            index
            for index, event in enumerate(parsed_lines)
            if isinstance(event, dict) and event.get("event_type") == "review_sealed"
        ]
        if seal_positions != [len(raw_lines) - 1]:
            raise ReviewConflictError("The seal event is not the final ledger event.")
        prefix = b"".join(raw_lines[: seal_positions[0]])
        if hashlib.sha256(prefix).hexdigest() != receipt.get(
            "judgment_event_prefix_digest"
        ):
            raise ReviewConflictError(
                "The pre-seal judgment ledger digest does not verify."
            )
        return receipt, results

    def list_sets(self) -> list[dict[str, Any]]:
        return self.catalog.list_sets()

    def list_sessions(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        with self._lock:
            for path in sorted(self.sessions_dir.glob("*/session.json")):
                try:
                    session = self._load_session(path.parent.name)
                    events = self._load_events(path.parent.name)
                    responses = self._current_responses(events)
                except ReviewError:
                    continue
                summaries.append(
                    {
                        "session_id": session["session_id"],
                        "eval_set_id": session["eval_set_id"],
                        "title": session["title"],
                        "rater_id": session["rater_id"],
                        "item_count": len(session["item_order"]),
                        "completed_count": len(responses),
                        "created_at": session["created_at"],
                        "updated_at": session["updated_at"],
                        "sealed_at": session.get("sealed_at"),
                    }
                )
        summaries.sort(key=lambda item: float(item["updated_at"]), reverse=True)
        return summaries

    def create_session(self, *, eval_set_id: str, rater_id: str) -> dict[str, Any]:
        clean_rater_id = compact_text(rater_id)
        if not clean_rater_id:
            raise ReviewValidationError("rater_id must not be empty.")
        if len(clean_rater_id) > 120:
            raise ReviewValidationError("rater_id must be 120 characters or fewer.")
        review_set = self.catalog.load(eval_set_id)
        item_order, presentation = self._assignment(review_set, clean_rater_id)
        now = time.time()
        session_id = f"review-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        session = {
            "version": SCHEMA_VERSION,
            "session_id": session_id,
            "title": f"{review_set.manifest['title']} / {clean_rater_id}",
            "eval_set_id": review_set.eval_set_id,
            "rater_id": clean_rater_id,
            "review_bundle_digest": review_set.review_bundle_digest,
            "rubric_digest": review_set.rubric_digest,
            "blind_key_digest": review_set.blind_key_digest,
            "source_digest": review_set.source_digest,
            "item_order": item_order,
            "presentation": presentation,
            "created_at": now,
            "updated_at": now,
            "sealed_at": None,
            "receipt_path": None,
            "result_path": None,
        }
        with self._lock:
            for path in self.sessions_dir.glob("*/session.json"):
                try:
                    existing = self._load_session(path.parent.name)
                except ReviewError:
                    continue
                if (
                    existing.get("eval_set_id") == eval_set_id
                    and existing.get("rater_id") == clean_rater_id
                ):
                    return self.session_detail(str(existing["session_id"]))
            append_jsonl(
                self._events_path(session_id),
                {
                    "event_type": "session_created",
                    "event_id": f"event-{secrets.token_hex(8)}",
                    "created_at": now,
                    "payload": {
                        "eval_set_id": review_set.eval_set_id,
                        "rater_id": clean_rater_id,
                        "review_bundle_digest": review_set.review_bundle_digest,
                        "rubric_digest": review_set.rubric_digest,
                        "blind_key_digest": review_set.blind_key_digest,
                        "source_digest": review_set.source_digest,
                    },
                },
            )
            self._save_session(session)
        return self.session_detail(session_id)

    def session_detail(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._load_session(session_id)
            review_set = self._validate_dataset(session)
            events = self._load_events(session_id)
            responses = self._current_responses(events)
            seal_receipt = None
            if session.get("sealed_at"):
                seal_receipt, _ = self._verified_sealed_artifacts(session, events)
        source_items = review_set.items_by_id
        presented_items: list[dict[str, Any]] = []
        for item_id in session["item_order"]:
            source_item = source_items[item_id]
            label_map = session["presentation"][item_id]
            presented_items.append(
                {
                    **{
                        key: value
                        for key, value in source_item.items()
                        if key != "labels"
                    },
                    "labels": {
                        display_label: source_item["labels"][source_label]
                        for display_label, source_label in label_map.items()
                    },
                }
            )
        complete_count = len(responses)
        detail = {
            "session_id": session["session_id"],
            "title": session["title"],
            "eval_set_id": session["eval_set_id"],
            "rater_id": session["rater_id"],
            "review_bundle_digest": session["review_bundle_digest"],
            "rubric_digest": session["rubric_digest"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "sealed_at": session.get("sealed_at"),
            "item_count": len(presented_items),
            "completed_count": complete_count,
            "items": presented_items,
            "responses": responses,
        }
        if seal_receipt is not None:
            detail["seal_receipt"] = seal_receipt
        return detail

    def submit_judgment(
        self,
        *,
        session_id: str,
        item_id: str,
        submission_id: str,
        answers: dict[str, Any],
        confidence: str,
        evidence: str,
        counterevidence: str = "",
        abstain: bool = False,
    ) -> dict[str, Any]:
        clean_submission_id = compact_text(submission_id)
        if not clean_submission_id:
            raise ReviewValidationError("submission_id must not be empty.")
        if len(clean_submission_id) > 200:
            raise ReviewValidationError(
                "submission_id must be 200 characters or fewer."
            )
        normalized_payload = {
            "answers": {
                compact_text(question_id): compact_text(choice)
                for question_id, choice in dict(answers or {}).items()
                if compact_text(question_id) and compact_text(choice)
            },
            "confidence": compact_text(confidence).lower(),
            "evidence": str(evidence or "").strip(),
            "counterevidence": str(counterevidence or "").strip(),
            "abstain": bool(abstain),
        }
        submission_digest = json_digest(
            {"item_id": item_id, "payload": normalized_payload}
        )
        with self._lock:
            session = self._load_session(session_id)
            events = self._load_events(session_id)
            for event in events:
                if event.get("submission_id") != clean_submission_id:
                    continue
                if event.get("submission_digest") == submission_digest:
                    return self.session_detail(session_id)
                raise ReviewConflictError(
                    "submission_id was already used for a different judgment."
                )
            if session.get("sealed_at"):
                raise ReviewConflictError("This review session is sealed.")
            review_set = self._validate_dataset(session)
            item = review_set.items_by_id.get(item_id)
            if item is None or item_id not in session["item_order"]:
                raise ReviewNotFoundError(f"Review item not found: {item_id}")
            if normalized_payload["confidence"] not in CONFIDENCE_LEVELS:
                raise ReviewValidationError("confidence must be low, medium, or high.")
            if not normalized_payload["evidence"]:
                raise ReviewValidationError("A short evidence note is required.")
            if len(normalized_payload["evidence"]) > 10_000:
                raise ReviewValidationError(
                    "evidence must be 10000 characters or fewer."
                )
            if len(normalized_payload["counterevidence"]) > 10_000:
                raise ReviewValidationError(
                    "counterevidence must be 10000 characters or fewer."
                )
            questions = {question["id"]: question for question in item["questions"]}
            if normalized_payload["abstain"]:
                if normalized_payload["answers"]:
                    raise ReviewValidationError(
                        "An abstained item must not include rubric answers."
                    )
            else:
                if set(normalized_payload["answers"]) != set(questions):
                    raise ReviewValidationError(
                        "Every rubric question must be answered before saving."
                    )
                for question_id, choice in normalized_payload["answers"].items():
                    if choice not in questions[question_id]["choices"]:
                        raise ReviewValidationError(
                            f"Invalid choice for rubric question {question_id}: {choice}"
                        )
            current = self._current_responses(events).get(item_id)
            revision = int(current.get("revision", 0) or 0) + 1 if current else 1
            now = time.time()
            append_jsonl(
                self._events_path(session_id),
                {
                    "event_type": "judgment_submitted",
                    "event_id": f"event-{secrets.token_hex(8)}",
                    "submission_id": clean_submission_id,
                    "submission_digest": submission_digest,
                    "created_at": now,
                    "item_id": item_id,
                    "revision": revision,
                    "supersedes": current.get("event_id") if current else None,
                    "payload": normalized_payload,
                },
            )
            session["updated_at"] = now
            self._save_session(session)
        return self.session_detail(session_id)

    def seal_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            repair_jsonl_tail(self._events_path(session_id))
            session = self._load_session(session_id)
            if session.get("sealed_at"):
                return self.session_detail(session_id)
            review_set = self._validate_dataset(session)
            events = self._load_events(session_id)
            responses = self._current_responses(events)
            missing = [
                item_id for item_id in session["item_order"] if item_id not in responses
            ]
            if missing:
                raise ReviewValidationError(
                    f"Cannot seal: {len(missing)} review items are incomplete."
                )
            blind_key_path = review_set.path / "blind_key.json"
            if not blind_key_path.exists():
                raise ReviewValidationError("Cannot seal without blind_key.json.")
            blind_key = load_json(blind_key_path)
            key_items = validated_key_items(
                blind_key,
                expected_item_ids=set(session["item_order"]),
            )
            sealed_at = time.time()
            question_summaries: dict[str, dict[str, Any]] = {
                question["id"]: {
                    "prompt": question["prompt"],
                    "wins": {},
                    "ties": 0,
                    "abstentions": 0,
                    "display_choices": {"A": 0, "B": 0, "Tie": 0},
                }
                for question in review_set.manifest["questions"]
            }
            judgments: list[dict[str, Any]] = []
            for item_id in session["item_order"]:
                response = responses[item_id]
                key_item = key_items.get(item_id)
                if not isinstance(key_item, dict):
                    raise ReviewValidationError(f"blind_key.json is missing {item_id}.")
                display_to_packet = session["presentation"][item_id]
                normalized_answers: dict[str, dict[str, Any]] = {}
                if response["abstain"]:
                    for summary in question_summaries.values():
                        summary["abstentions"] += 1
                else:
                    for question_id, display_choice in response["answers"].items():
                        summary = question_summaries[question_id]
                        summary["display_choices"][display_choice] = (
                            summary["display_choices"].get(display_choice, 0) + 1
                        )
                        packet_choice = display_to_packet.get(
                            display_choice, display_choice
                        )
                        arm_choice = None
                        if packet_choice == "Tie":
                            summary["ties"] += 1
                        elif packet_choice in DISPLAY_LABELS:
                            arm_choice = key_item["label_to_arm"].get(packet_choice)
                            if not arm_choice:
                                raise ReviewValidationError(
                                    f"blind_key.json cannot resolve {item_id}/{packet_choice}."
                                )
                            summary["wins"][arm_choice] = (
                                summary["wins"].get(arm_choice, 0) + 1
                            )
                        normalized_answers[question_id] = {
                            "display_choice": display_choice,
                            "packet_choice": packet_choice,
                            "arm_choice": arm_choice,
                        }
                judgments.append(
                    {
                        "item_id": item_id,
                        "scenario_name": review_set.items_by_id[item_id][
                            "scenario_name"
                        ],
                        "run_index": review_set.items_by_id[item_id]["run_index"],
                        "revision": response["revision"],
                        "abstain": response["abstain"],
                        "confidence": response["confidence"],
                        "evidence": response["evidence"],
                        "counterevidence": response["counterevidence"],
                        "presentation_display_to_packet": display_to_packet,
                        "packet_label_to_arm": dict(key_item["label_to_arm"]),
                        "source_labels": dict(key_item.get("labels") or {}),
                        "answers": normalized_answers,
                    }
                )
            event_log_path = self._events_path(session_id)
            results = {
                "schema": "rcl.blind_review_results.v1",
                "session_id": session_id,
                "eval_set_id": review_set.eval_set_id,
                "rater_id": session["rater_id"],
                "sealed_at": sealed_at,
                "review_bundle_digest": review_set.review_bundle_digest,
                "rubric_digest": review_set.rubric_digest,
                "source_digest": session["source_digest"],
                "blind_key_digest": session["blind_key_digest"],
                "question_summaries": question_summaries,
                "judgments": judgments,
            }
            results_digest = json_digest(results)
            receipt = {
                "schema": "rcl.blind_review_seal_receipt.v1",
                "session_id": session_id,
                "eval_set_id": review_set.eval_set_id,
                "rater_id": session["rater_id"],
                "sealed_at": sealed_at,
                "item_count": len(session["item_order"]),
                "review_bundle_digest": review_set.review_bundle_digest,
                "rubric_digest": review_set.rubric_digest,
                "source_digest": session["source_digest"],
                "blind_key_digest": session["blind_key_digest"],
                "judgment_event_prefix_digest": (
                    file_digest(event_log_path) if event_log_path.exists() else None
                ),
                "results_digest": results_digest,
            }
            receipt["receipt_digest"] = json_digest(receipt)
            result_path = self._session_dir(session_id) / "unblinded_results.json"
            receipt_path = self._session_dir(session_id) / "seal_receipt.json"
            atomic_write_json(result_path, results)
            atomic_write_json(receipt_path, receipt)
            append_jsonl(
                event_log_path,
                {
                    "event_type": "review_sealed",
                    "event_id": f"event-{secrets.token_hex(8)}",
                    "created_at": sealed_at,
                    "payload": {
                        "sealed_at": sealed_at,
                        "receipt_path": str(receipt_path),
                        "result_path": str(result_path),
                        "receipt_digest": receipt["receipt_digest"],
                    },
                },
            )
            session["sealed_at"] = sealed_at
            session["updated_at"] = sealed_at
            session["receipt_path"] = str(receipt_path)
            session["result_path"] = str(result_path)
            self._save_session(session)
        return self.session_detail(session_id)
