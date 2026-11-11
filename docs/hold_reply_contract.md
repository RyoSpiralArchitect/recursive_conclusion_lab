# Reply targets and offline diagnostics

The hold harness can request a concise reply and record whether the returned
text meets the configured target. This supports the goal of useful, revisable
conversation without concealing unresolved feasibility or cutting off a final
constraint. It is a soft model instruction with a deterministic measurement,
not a semantic validator or a hard output guarantee.

```sh
python3 recursive_conclusion_lab.py compare \
  --script protocol_scripts/hold_open_then_revise.json \
  --providers dummy=dummy-v1 --conclusion-mode hold \
  --conclusion-every 0 --memory-every 0 --hold-user-replay \
  --hold-reply-max-words 180 --out-dir runs/reply_contract_demo
```

`hold_reply_max_words` is a nonnegative integer in saved configuration and JSON
compare configurations. Zero disables the additional instruction and diagnostic.
A positive target requires hold mode. Invalid values are rejected before output
reservation and provider construction, including invalid later matrix arms.
Playtest persistence retains the value; older configurations default to zero.
This change does not add a Playtest UI control.

The instruction asks for a complete direct answer or provisional move, preserving
material uncertainty, purpose and constraints. A reply that exceeds the target
is retained in full. There is no automatic retry, rewriting call, or truncation.
The instruction is new and has not yet been tested in a new live comparison.

The `reply_contract` field accompanies the reply event, return value and compare
summary. It is not fed back to the model, added to conversation history, or shown
in a blind transcript. For example:

```json
{
  "schema": "rcl.reply_contract.v1",
  "word_counter": "whitespace_v1",
  "max_words": 180,
  "words": 198,
  "status": "exceeded",
  "excess_words": 18
}
```

`whitespace_v1` is `len(text.split())`, including standalone Markdown markers.
It preserves the existing English experiment's definition. It is not a linguistic
tokenizer or a suitable measure of unspaced Japanese text. `empty`, `exceeded`
and `within_budget` describe length; provider `finish_reason` is a separate field.
A short truncated answer can be within the word budget and still be incomplete.
Neither normal finish nor length compliance establishes semantic quality.

`analyze_runs.py` retains the existing regex-based `avg_reply_words` and adds
`avg_reply_whitespace_words`, its counter identifier, and configured-budget
counts. Absent diagnostics do not imply a target; malformed or inconsistent
diagnostics are counted separately rather than accepted as compliance.

## Complete and interrupted source-replay experiments

```sh
python3 -S analyze_hold_run.py \
  --run /path/to/recorded_source_run \
  --out reports/source_run.json --max-words 180
```

This offline command reads the three-arm `hold_source_comparison.py` format,
including its frozen inputs, execution order, generation receipts, timings and
JSONL events. Raw run data is local and is not included in the public repository.
The command imports no provider adapters and makes no network requests.

It distinguishes finished, partial and unstarted conversations, including empty
files reserved for unstarted work. `failed_or_unknown` and write-ahead `started`
requests retain unknown remote outcomes and usage. Usage totals cover returned
requests and separately report missing usage. Valid workpads are not required
for the two arms that deliberately make no workpad calls.

All planned pair identities are retained, with available and complete-treatment
subsets labelled separately. Unknown finish, empty reply and invalid expected
workpad prevent complete-treatment inclusion. Per-turn word counts and format
counts are descriptive diagnostics, not semantic scoring.

The analysis limit is explicitly an analysis criterion; it is not inferred to
have been a runtime instruction. A stopped run is required. Duplicate or
inconsistent receipts/events, unexpected calls, source drift, or escaping paths
are rejected before creating the report. The report must be a new file outside
the evidence directory. The original run is never resumed or modified.
