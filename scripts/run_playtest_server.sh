#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PSYCHOID_NET_GUARD="${PSYCHOID_NET_GUARD:-0}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8787}"
RELOAD="${RELOAD:-1}"
SESSIONS_DIR="${SESSIONS_DIR:-playtest_sessions}"

if [[ "${RELOAD}" == "1" ]]; then
  python3 playtest_server.py --host "$HOST" --port "$PORT" --reload --sessions-dir "$SESSIONS_DIR"
else
  python3 playtest_server.py --host "$HOST" --port "$PORT" --sessions-dir "$SESSIONS_DIR"
fi
