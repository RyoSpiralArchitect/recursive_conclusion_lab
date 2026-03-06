# Recursive Conclusion Lab

[Japanese README](README.ja.md)

Cross-provider experiment harness for observing (and optionally steering) the *time-structure* of LLM dialogue:

- Recursive **memory capsules** (compressed context you can reload every turn)
- Periodic **conclusion probes** (predict the likely end-state)
- **Deferred utterance intents** (plan now, say later)

## Providers

- `openai` (Responses API)
- `anthropic` (Messages API)
- `mistral` (Chat Completions API)
- `gemini` (generateContent API)
- `hf` (Inference Providers; OpenAI-compatible chat-completions)
- `dummy` (local deterministic mock; no API keys)

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
- `--deferred-intent-timing offset|model`
  - `offset` (default): timing window derives from `--deferred-intent-offset` / `--deferred-intent-grace`.
  - `model`: the planner proposes a timing window (fully supported in `external`; `inband` uses the state's `earliest_turn`/`latest_turn`).

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

### Conclusion mention “delay” (observe-only)

Each `conclusion_probe` also emits an observe-only “mention plan” (not injected into replies):

- `keywords`: 3–5 distinctive phrases for mention detection
- `mention_delay_min_turns` / `mention_delay_max_turns`: predicted mention window (turns after probe)
- `mention_likelihood`, `delay_strategy`, `delay_signals`

`analyze_runs.py` reports planned-vs-actual metrics like `conclusion_plan_within_window_rate`.

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
