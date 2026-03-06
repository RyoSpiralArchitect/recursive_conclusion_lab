#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional


def collect_logs(*, logs: list[str] | None, log_dir: str | None) -> list[Path]:
    paths: list[Path] = []
    if logs:
        paths.extend(Path(p) for p in logs)
    if log_dir:
        paths.extend(sorted(Path(log_dir).glob("*.jsonl")))
    if not paths:
        raise SystemExit("No logs found. Use --logs or --log-dir.")
    deduped: list[Path] = []
    seen = set()
    for path in paths:
        rp = path.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(path)
    return deduped


def coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS log_files (
          source_file TEXT PRIMARY KEY,
          imported_at REAL NOT NULL,
          file_size INTEGER,
          file_mtime REAL,
          line_count INTEGER,
          inserted INTEGER,
          ignored INTEGER,
          errors INTEGER
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
          source_file TEXT NOT NULL,
          source_seq INTEGER NOT NULL,
          timestamp REAL,
          provider TEXT,
          model TEXT,
          turn_index INTEGER,
          event_type TEXT,
          payload_json TEXT,
          row_json TEXT NOT NULL,
          PRIMARY KEY (source_file, source_seq)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_turn ON events(turn_index);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_provider_model ON events(provider, model);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);")
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?);",
        ("schema_version", "1"),
    )


def iter_jsonl(path: Path) -> Iterable[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            yield idx, raw


def import_file(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
    inserted = 0
    ignored = 0
    errors = 0
    line_count = 0
    source_file = str(path.resolve())

    batch: list[tuple[Any, ...]] = []
    insert_sql = (
        "INSERT OR IGNORE INTO events("
        "source_file, source_seq, timestamp, provider, model, turn_index, event_type, payload_json, row_json"
        ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?);"
    )

    cur = conn.cursor()
    start = time.time()
    for seq, raw in iter_jsonl(path):
        line_count += 1
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            errors += 1
            continue
        if not isinstance(row, dict):
            errors += 1
            continue

        payload = row.get("payload")
        try:
            payload_json = json.dumps(payload, ensure_ascii=False) if payload is not None else None
            row_json = json.dumps(row, ensure_ascii=False)
        except TypeError:
            errors += 1
            continue

        batch.append(
            (
                source_file,
                int(seq),
                coerce_float(row.get("timestamp")),
                str(row.get("provider") or "") or None,
                str(row.get("model") or "") or None,
                coerce_int(row.get("turn_index")),
                str(row.get("event_type") or "") or None,
                payload_json,
                row_json,
            )
        )
        if len(batch) >= 1000:
            before = conn.total_changes
            cur.executemany(insert_sql, batch)
            batch_inserted = conn.total_changes - before
            inserted += batch_inserted
            ignored += max(0, len(batch) - batch_inserted)
            batch.clear()

    if batch:
        before = conn.total_changes
        cur.executemany(insert_sql, batch)
        batch_inserted = conn.total_changes - before
        inserted += batch_inserted
        ignored += max(0, len(batch) - batch_inserted)
        batch.clear()

    file_stat = path.stat()
    conn.execute(
        """
        INSERT OR REPLACE INTO log_files(
          source_file, imported_at, file_size, file_mtime, line_count, inserted, ignored, errors
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            source_file,
            float(time.time()),
            int(file_stat.st_size),
            float(file_stat.st_mtime),
            int(line_count),
            int(inserted),
            int(ignored),
            int(errors),
        ),
    )
    conn.commit()
    return {
        "source_file": source_file,
        "inserted": inserted,
        "ignored": ignored,
        "errors": errors,
        "line_count": line_count,
        "seconds": time.time() - start,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Import Recursive Conclusion Lab JSONL event logs into a SQLite database."
    )
    p.add_argument("--db", required=True, help="SQLite DB path (created if missing).")
    p.add_argument("--log-dir", default="", help="Directory containing *.jsonl logs.")
    p.add_argument("--logs", nargs="*", default=[], help="Explicit JSONL file paths.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_dir = args.log_dir.strip() or None
    logs = [p for p in (args.logs or []) if p.strip()]
    paths = collect_logs(logs=logs or None, log_dir=log_dir)

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        totals = {"files": 0, "inserted": 0, "ignored": 0, "errors": 0, "line_count": 0}
        for path in paths:
            result = import_file(conn, path)
            totals["files"] += 1
            totals["inserted"] += int(result["inserted"])
            totals["ignored"] += int(result["ignored"])
            totals["errors"] += int(result["errors"])
            totals["line_count"] += int(result["line_count"])
            print(
                f"- {Path(result['source_file']).name}: +{result['inserted']} "
                f"ignored={result['ignored']} errors={result['errors']} "
                f"lines={result['line_count']} ({result['seconds']:.2f}s)"
            )
        print(
            "Done: "
            f"files={totals['files']} inserted={totals['inserted']} "
            f"ignored={totals['ignored']} errors={totals['errors']} "
            f"lines={totals['line_count']} db={db_path}"
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
