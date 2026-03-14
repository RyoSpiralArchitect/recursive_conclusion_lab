#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set." >&2
  exit 1
fi

GENERATOR_MODEL="${OPENAI_MODEL:-gpt-4.1-mini-2025-04-14}"
OBSERVER_MODEL="${OPENAI_OBSERVER_MODEL:-$GENERATOR_MODEL}"
EMBEDDING_MODEL="${OPENAI_EMBEDDING_MODEL:-text-embedding-3-large}"
REPEATS="${REPEATS:-3}"
SEED="${SEED:-7}"
OUT_DIR="${OUT_DIR:-compare_outputs/triplet_shortlist_stage_policy_gpt4mini}"

TMP_CONFIG="$(mktemp /tmp/rcl_shortlist_stage_policy.XXXXXX.json)"
trap 'rm -f "$TMP_CONFIG"' EXIT

cat >"$TMP_CONFIG" <<JSON
{
  "command": "compare-matrix",
  "script": "protocol_scripts/shortlist_then_commit.json",
  "providers": [
    "openai=${GENERATOR_MODEL}"
  ],
  "out_dir": "${OUT_DIR}",
  "repeats": ${REPEATS},
  "seed": ${SEED},
  "args": {
    "window": 8,
    "memory_every": 2,
    "conclusion_every": 2,
    "delayed_mention_every": 2,
    "delayed_mention_mode": "soft_fire",
    "delayed_mention_item_limit": 4,
    "delayed_mention_fire_prob": 0.35,
    "delayed_mention_fire_max_items": 2,
    "delayed_mention_leak_policy": "on",
    "delayed_mention_leak_threshold": 0.05,
    "delayed_mention_min_nonconclusion_items": 3,
    "delayed_mention_min_kind_diversity": 4,
    "delayed_mention_diversity_repair": "on",
    "adaptive_hazard_policy": "adaptive",
    "adaptive_hazard_profile": "balanced",
    "adaptive_hazard_stage_policy": "flat",
    "adaptive_hazard_embedding_guard": "off",
    "latent_convergence_every": 1,
    "semantic_judge_backend": "both",
    "observer_provider": "openai",
    "observer_model": "${OBSERVER_MODEL}",
    "embedding_provider": "openai",
    "embedding_model": "${EMBEDDING_MODEL}",
    "temperature": 0.2,
    "probe_max_tokens": 220,
    "show_probes": false
  },
  "arms": [
    {
      "name": "static",
      "args": {
        "adaptive_hazard_policy": "static",
        "adaptive_hazard_stage_policy": "flat"
      }
    },
    {
      "name": "adaptive_flat",
      "args": {
        "adaptive_hazard_policy": "adaptive",
        "adaptive_hazard_stage_policy": "flat"
      }
    },
    {
      "name": "adaptive_kind_aware",
      "args": {
        "adaptive_hazard_policy": "adaptive",
        "adaptive_hazard_stage_policy": "kind_aware"
      }
    }
  ]
}
JSON

export PSYCHOID_NET_GUARD="${PSYCHOID_NET_GUARD:-0}"

python3 recursive_conclusion_lab.py compare-matrix --config "$TMP_CONFIG"
python3 analyze_runs.py --log-dir "$OUT_DIR" --script protocol_scripts/shortlist_then_commit.json
