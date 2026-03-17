# Recursive Conclusion Lab

[Japanese README](README.ja.md)

Cross-provider experiment harness for observing (and optionally steering) the *time-structure* of LLM dialogue:

- Recursive **memory capsules** (compressed context you can reload every turn)
- Periodic **conclusion probes** (predict the likely end-state)
- Observe-only **latent convergence traces** (semantic drift before explicit mention)
- Optional **independent observer** for latent convergence judging
- Optional **embedding judge** for semantic drift measurement
- **Deferred utterance intents** (plan now, say later)

## Providers

- `openai` (Responses API)
- `anthropic` (Messages API)
- `mistral` (Chat Completions API)
- `gemini` (generateContent API)
- `hf` (Inference Providers; OpenAI-compatible chat-completions)
- `dummy` (local deterministic mock; no API keys)

Embedding-based semantic judging currently supports:

- `openai`
- `dummy`

## Requirements

- Python 3.9+
- `requests`

```bash
python -m pip install requests
```

## API keys (set only what you use)

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `MISTRAL_API_KEY`
- `GEMINI_API_KEY`
- `HF_TOKEN`

## Quick start (REPL)

Observe conclusion probes:

```bash
python recursive_conclusion_lab.py repl \
  --provider openai \
  --model <model_id> \
  --window 8 \
  --conclusion-every 2 \
  --conclusion-mode observe \
  --show-probes \
  --log runs/openai_observe.jsonl
```

Steer with the conclusion hypothesis (tunable strength/injection):

```bash
python recursive_conclusion_lab.py repl \
  --provider openai \
  --model <model_id> \
  --window 8 \
  --conclusion-every 2 \
  --conclusion-mode soft_steer \
  --conclusion-steer-strength strong \
  --conclusion-steer-injection conclusion_line \
  --show-probes
```

## Scripted comparisons

Run the same script across providers:

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/convergent_protocol.json \
  --providers openai=<model_id> anthropic=<model_id> \
  --window 8 \
  --memory-every 2 \
  --conclusion-every 2 \
  --conclusion-mode soft_steer \
  --out-dir compare_outputs/conclusion
```

Templates:

- `templates/script_template.json`
- `templates/compare_config_template.json` (CLI-to-JSON mapping reference)
- `templates/compare_matrix_config_template.json` (arm matrix example)

Run from a JSON config file:

```bash
python recursive_conclusion_lab.py run-config \
  --config templates/compare_config_template.json
```

## Deferred intents (baseline → full → ablation)

### Backend

- `external` (default): explicit planner/scheduler probes (more controlled, more API calls).
- `inband`: the assistant carries deferred-intent state by appending a hidden `<RCL_STATE>...</RCL_STATE>` JSON block to each reply (more portable, fewer moving parts).

### Planning cadence & timing

- `--deferred-intent-plan-policy periodic|auto`
  - `periodic`: plan on `--deferred-intent-every` turns.
  - `auto`: allow planning every turn until the plan budget is exhausted.
- `--deferred-intent-plan-budget N`
  - required for `--deferred-intent-plan-policy auto`
  - limits planner calls (`external`) / eligible planning turns (`inband`)
- `--deferred-intent-plan-max-new N`: cap new intents per eligible planning turn (both backends).
- `--deferred-intent-timing offset|model|hazard`
  - `offset` (default): timing window derives from `--deferred-intent-offset` / `--deferred-intent-grace`.
  - `model`: the planner proposes a timing window (fully supported in `external`; `inband` uses the state's `earliest_turn`/`latest_turn`).
  - `hazard`: the planner proposes a per-delay probability profile (`hazard_profile`) instead of a single fixed offset.

### Latent convergence trace

Track semantic convergence toward the latest conclusion even before explicit mention:

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/latent_resurfacing.json \
  --providers openai=<model_id> \
  --conclusion-every 2 \
  --latent-convergence-every 1 \
  --semantic-judge-backend both \
  --observer-provider anthropic \
  --observer-model <observer_model_id> \
  --embedding-provider openai \
  --embedding-model <embedding_model_id> \
  --out-dir compare_outputs/latent_trace
```

`--semantic-judge-backend` accepts `off|llm|embedding|both`.
This logs `latent_convergence_trace` and/or `embedding_convergence_trace` events and analyzer
fields like `avg_latent_alignment`, `avg_embedding_alignment`,
`latent_semantic_leakage_rate`, `embedding_semantic_leakage_rate`,
`avg_articulation_gap_turns`, `avg_embedding_articulation_gap_turns`, and
`semantic_judge_disagreement_rate`.
If `--observer-provider/--observer-model` are set, the LLM latent judge is decoupled from the
generator and `analyze_runs.py` reports `latent_judge_source`, `latent_judge_provider`,
`latent_judge_model`, plus `embedding_judge_provider` / `embedding_judge_model`.

Enable in-band deferred intents:

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers openai=<model_id> \
  --window 8 \
  --deferred-intent-backend inband \
  --deferred-intent-every 2 \
  --deferred-intent-mode soft_fire \
  --deferred-intent-strategy trigger \
  --deferred-intent-offset 3 \
  --deferred-intent-grace 2 \
  --out-dir compare_outputs/deferred_inband_trigger
```

Trigger-based deferred intents (soft fire):

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers openai=<model_id> \
  --window 8 \
  --deferred-intent-every 2 \
  --deferred-intent-mode soft_fire \
  --deferred-intent-strategy trigger \
  --deferred-intent-offset 3 \
  --deferred-intent-grace 2 \
  --out-dir compare_outputs/deferred_trigger
```

Latent injection (intents influence the trajectory before they are due):

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers openai=<model_id> \
  --window 8 \
  --deferred-intent-every 2 \
  --deferred-intent-mode soft_fire \
  --deferred-intent-strategy trigger \
  --deferred-intent-offset 3 \
  --deferred-intent-grace 2 \
  --deferred-intent-latent-injection active \
  --out-dir compare_outputs/deferred_latent
```

Ablation (intent planned → deleted immediately):

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers openai=<model_id> \
  --window 8 \
  --deferred-intent-every 2 \
  --deferred-intent-mode soft_fire \
  --deferred-intent-strategy trigger \
  --deferred-intent-offset 3 \
  --deferred-intent-grace 2 \
  --deferred-intent-latent-injection active \
  --deferred-intent-ablation delete_planned \
  --out-dir compare_outputs/deferred_latent_deleted
```

## Analyze logs

### Arm matrix

Run a fixed arm set (for example `observe`, `latent_only`, `soft_fire`, `hard_fire`, `delete_planned`) from JSON:

```bash
python recursive_conclusion_lab.py compare-matrix \
  --config templates/compare_matrix_config_template.json
```

You can add top-level `repeats` and `seed` to rerun the full arm matrix multiple times with stable
harness-side randomness.

This writes arm-tagged logs like `arm_soft_fire___openai__model.jsonl`, arm-specific summaries
(`summary__soft_fire.json`), per-repeat summaries such as `summary__soft_fire__run_001.json`,
a combined `summary.json`, plus repeat-level analyzer outputs in `analysis_runs.json` and
`analysis_aggregate.json`.

If a script's `evaluation` block includes an optional `perturbation` spec, `analyze_runs.py`
also reports recovery metrics such as `recovery_after_perturbation_rate`,
`time_to_recover_turns`, `probe_recovery_after_perturbation_rate`,
`probe_time_to_recover_turns`, `probe_to_reply_recovery_gap_turns`, and
`post_perturbation_forbidden_turn_rate`.

### Conclusion mention “delay” (observe-only)

Each `conclusion_probe` also emits an observe-only “mention plan” (not injected into replies):

- `keywords`: 3–5 distinctive phrases for mention detection
- `mention_delay_min_turns` / `mention_delay_max_turns`: predicted mention window (turns after probe)
- `mention_hazard_profile`: per-delay mention mass inside that window
- `mention_likelihood`, `delay_strategy`, `delay_signals`

`analyze_runs.py` reports planned-vs-actual metrics like `conclusion_plan_within_window_rate`,
`conclusion_on_support_rate`, and `avg_conclusion_hazard_turn_prob_at_mention`.

### Delayed mention targets (multi-item; optional)

Enable an additional probe where the LLM chooses **which items** should be mentioned later (not immediately):

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers openai=<model_id> \
  --delayed-mention-every 2 \
  --delayed-mention-item-limit 3
```

Optional soft-fire (probabilistic) hinting inside each planned window:

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers openai=<model_id> \
  --delayed-mention-every 2 \
  --delayed-mention-mode soft_fire \
  --delayed-mention-fire-prob 0.35 \
  --delayed-mention-leak-policy on \
  --delayed-mention-leak-threshold 0.05 \
  --delayed-mention-fire-max-items 2
```

Internally each delayed mention is now normalized to a `mention_hazard_profile`; `soft_fire`
uses that per-delay mass to weight the per-turn injection probability instead of treating the
whole support window as flat.

The leakage guard is configurable:

- `--delayed-mention-leak-policy on|off`
- `--delayed-mention-leak-threshold <0.00-1.00>`
- `--delayed-mention-min-nonconclusion-items <int>`
- `--delayed-mention-min-kind-diversity <int>`
- `--delayed-mention-diversity-repair on|off`
- `--adaptive-hazard-policy static|adaptive`
- `--adaptive-hazard-profile conservative|balanced|eager`
- `--adaptive-hazard-stage-policy flat|kind_aware`
- `--adaptive-hazard-embedding-guard off|on`

When the guard is on, active delayed mentions whose current turn probability is still below the
threshold are injected into the private system prompt as “keep latent” targets. This is meant to
reduce early explicit surfacing without removing latent trajectory pressure.

The planner can also be nudged away from collapsing everything into `conclusion`. The delayed-mention
probe now asks for at least some non-conclusion items and a minimum kind diversity when plausible,
for example `caveat`, `option`, or `constraint`.

For a stronger delayed-mention timing discriminator, use
`protocol_scripts/shortlist_then_commit.json`. It creates a staged release:
the model must surface a two-item shortlist one turn before it is allowed to
commit to the winner and release the late caveat / fallback / migration-risk
packet. This tends to separate `static`, `adaptive`, and `adaptive_guard` more
clearly than simple single-release scripts.

To rerun the current OpenAI baseline comparison directly, use:

```bash
OPENAI_API_KEY=... scripts/run_shortlist_stage_policy_gpt4mini.sh
```

To build the matching `deferred_multi_release` stage-policy comparison, use:

```bash
OPENAI_API_KEY=... scripts/run_deferred_multi_release_stage_policy_gpt4mini.sh
```

To build a blinded pairwise human-eval packet set from those two compare outputs, use:

```bash
scripts/build_staged_release_human_eval_set.sh
```

This writes `human_eval_sets/staged_release_pairwise_v1/` with:
`manifest.json`, `eval_items.jsonl`, `booklet.md`, `answer_sheet.csv`, `blind_key.json`,
and one Markdown packet per item under `packets/`.

When `--delayed-mention-diversity-repair on`, the harness will make one compact supplemental probe if
the first delayed-mention plan fails the non-conclusion or kind-diversity minimums. This keeps the
pressure probabilistic, but makes the planner pay a real cost for collapsing everything into a single
conclusion item.

When adaptive hazard is on, the harness does not replace the planned hazard support. Instead, it
rescales the current-turn hazard mass and leak threshold from recent semantic signals
(`latent_alignment`, `articulation_readiness`, `leakage_risk`, and judge gap), while also pulling
release decisions toward the hazard-profile support peak rather than simply lowering thresholds.
This is meant to create more “delay” without collapsing articulation into a deterministic schedule.

`--adaptive-hazard-stage-policy kind_aware` further separates staged option items from final-stage
risk packets. That adds kind-aware hazard multipliers / threshold shifts plus a small hazard-profile
reshape so option-stage items and final packet items do not share exactly the same hold/release
policy. Keep the default at `flat` unless you are explicitly comparing staged-release behavior.

This logs `delayed_mention_plan` / `delayed_mention_action` events and adds analyzer columns like:
`delayed_mention_nonconclusion_mention_rate`, `delayed_mention_within_window_rate`,
`delayed_mention_on_support_rate`, `avg_delayed_mention_hazard_turn_prob_at_mention`,
`avg_delayed_mention_peak_support_ratio_at_mention`,
`delayed_mention_kind_diversity`, `delayed_mention_required_kind_coverage`,
`delayed_mention_min_nonconclusion_satisfied`, `delayed_mention_min_kind_diversity_satisfied`,
`delayed_mention_leak_policy`, `delayed_mention_leak_threshold`, and
`avg_suppressed_delayed_mention_count`. Adaptive control also adds
`adaptive_hazard_policy`, `adaptive_hazard_profile`, `adaptive_hazard_stage_policy`,
`avg_adaptive_hazard_multiplier`, `adaptive_hazard_intervention_rate`,
`avg_adaptive_hazard_turn_prob_shift`, `avg_option_stage_adaptive_hazard_multiplier`,
`avg_option_stage_adaptive_threshold_shift`,
`avg_final_risk_packet_adaptive_hazard_multiplier`,
`avg_final_risk_packet_adaptive_threshold_shift`, and
`avg_conclusion_adaptive_hazard_multiplier`. Conclusion timing also exposes
`avg_conclusion_peak_support_ratio_at_mention`.

`--adaptive-hazard-embedding-guard on` adds an extra pre-peak penalty when the embedding judge sees
strong semantic drift before the hazard peak. This is best treated as an experimental arm, not as
the default adaptive policy, because it can reduce leakage in some runs but also over-hold release
timing in others.

```bash
python analyze_runs.py \
  --log-dir compare_outputs/deferred_trigger \
  --script protocol_scripts/gather_then_recommend.json
```

Import JSONL into SQLite:

```bash
python jsonl_to_sqlite.py \
  --db runs/rcl.sqlite \
  --log-dir compare_outputs/deferred_trigger
```

## Protocol docs

- `EXPERIMENT_PROTOCOL.md` (JP)
- `DEFERRED_INTENT_PROTOCOL.md` (JP)

## License

GNU Affero General Public License v3.0 or later (`AGPL-3.0-or-later`). See `LICENSE`.
