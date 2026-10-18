# Bounded hold-dialogue pilot

`hold_dialogue_pilot.py` extracts the reusable runner from the October 18 ZIP.
It uses this repository's existing protocol and review templates. The ZIP, its
bundled dummy outputs, machine-specific launcher, credential-file parser, and
duplicate protocol/templates are not dependencies.

Run a wiring check from the repository root:

```bash
python3 hold_dialogue_pilot.py --offline --out runs/hold_pilot_dummy_01
```

For an explicitly chosen live run, install `requests`, provide the selected
provider's existing environment credential, and choose a fresh output directory:

```bash
python3 hold_dialogue_pilot.py --provider-model mistral=mistral-small-2603 \
  --out runs/hold_pilot_live_01
```

The command requires an explicit offline/live choice. It never reads dotenv or
shell configuration, obtains new credentials, retries a failed call, resumes an
interrupted run, or overwrites an existing run directory. Other supported live
provider names are `openai`, `anthropic`, `gemini`, and `hf`, using the existing
adapter environment conventions. Availability must be checked for the chosen
model; a model name in documentation is not an availability guarantee.

The design is fixed at three conditions × two repetitions × six turns: 36 replies
and 12 workpad generations, capped at 48 adapter calls. Replies allow 900 output
tokens; workpads allow 600. HTTP timeout is 60 seconds per call. This is not a
total-input-token, wall-clock, or monetary cap. The protocol is one English
scenario, not a representative benchmark. `seed=7` does not make the Mistral
requests paired random draws: that adapter does not transmit the local seed.

## What is recorded

- `run_manifest.json`: exact source commit, dirty-state flag, source/input file
  hashes, settings, expected and observed sizes, and execution status.
- `matrix.json` and `human_eval.json`: resolved repository inputs and paths.
- `generation_receipts.json`: call intent saved before dispatch, dialogue/turn,
  reply or workpad kind, usage, response identifier, finish reason, and time.
- `turn_timings.json`: time until each reply returns, including workpad generation.
- `compare/`: the existing event logs, summaries, and analyses.
- `metrics.json`: event/receipt reconciliation, per-condition usage and time,
  workpad validity, dialogue exclusions, and eligible comparison item IDs.
- `review/`: all six blinded pairs, including fallback behavior.
- `complete_review/`: eligible pairs only; absent if no pair qualifies.

Before any call or output reservation, the runner checks all arms, the credential
presence for live execution, the protocol length, and the fixed controller design.
After reservation, failed runs are marked `incomplete` when the process can still
write. Abrupt process termination can leave a `running` manifest and a `started`
receipt. That is an unknown outcome, not evidence of zero cost. Preserve the run
and reconcile it before deciding on another invocation.

`complete` means execution and receipt reconciliation completed. It does not mean
every workpad was accepted, every response was useful, or human review occurred.
Unknown usage remains null, with missing-call counts, rather than becoming zero.
Gemini output totals use `candidatesTokenCount`; the raw usage receipt also retains
any separately reported thought tokens. These totals are not normalized billing
estimates or cross-provider quality measures.

## Rejected workpads and review subsets

The harness still discards a rejected workpad and clears its update turn. It
continues the reply using the existing hold instructions and dialogue history.
Prompts, parsing/admission rules, and fallback behavior are unchanged.

New `inquiry_update.validation` diagnostics record the failure stage, a stable
field/code, up to 4,096 characters of rejected generated text, its full character
count and SHA-256, and whether the excerpt was truncated. `missing_or_null_json`
reflects the existing extractor's behavior: both failed extraction and a JSON
null yield no usable value. Schema codes distinguish wrong fields, types, lengths,
and blank strings. These diagnostics are research logs, never reply context or
public blind-review fields. They can contain dialogue-derived content and should
be handled with the same access restrictions as the underlying transcripts.

A complete-treatment pair requires both dialogues to have normal recorded finish
reasons for every generation and a valid state for every scheduled workpad. The
all-dialogue packet always remains available. Excluding workpad failures changes
the analysis population and may remove the entire primary comparison; it does
not justify quietly replacing the all-dialogue question with a favorable subset.

Give blinded reviewers the booklet/packets and blank answer sheet. Keep the
condition-labelled metrics, diagnostic logs, and `blind_key.json` separate.

See the [October 18 readout](experiments/2026-10-18-hold-pilot.md) and the
[next diagnostic replication](experiments/hold-pilot-next.md). The recorded live
results predate these diagnostics; this implementation cannot reconstruct the
two missing rejected texts.
