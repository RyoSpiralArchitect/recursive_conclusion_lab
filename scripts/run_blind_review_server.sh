#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8787}"
RELOAD="${RELOAD:-1}"
EVAL_SETS_DIR="${EVAL_SETS_DIR:-human_eval_sets}"
REVIEW_SESSIONS_DIR="${REVIEW_SESSIONS_DIR:-blind_review_sessions}"

SERVER_ARGS=(
  --host "$HOST"
  --port "$PORT"
  --eval-sets-dir "$EVAL_SETS_DIR"
  --review-sessions-dir "$REVIEW_SESSIONS_DIR"
  --review-only
)

if [[ "$RELOAD" == "1" ]]; then
  python3 playtest_server.py "${SERVER_ARGS[@]}" --reload
else
  python3 playtest_server.py "${SERVER_ARGS[@]}"
fi
