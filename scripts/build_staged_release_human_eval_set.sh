#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${CONFIG_PATH:-templates/human_eval_staged_release_pairwise_v1.json}"

python3 build_human_eval_set.py --config "$CONFIG_PATH"
