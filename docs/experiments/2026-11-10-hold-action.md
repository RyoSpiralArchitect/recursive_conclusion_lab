# 2026-11-10 action-wording comparison: incomplete live execution

The fixed comparison and three fresh scenarios are implemented. The frozen
revision `e9b5dd3dc20c37f96846b6f01e45fad217133cbf` passed 96 local tests and a
144-generation dummy simulation, with 72 valid workpads and six paired packets.
Dummy outputs establish wiring only.

The authenticated model catalog listed the exact `mistral-small-2603` model.
Live execution then stopped at its first workpad generation (museum, current
instruction, repetition 1, turn 1). The write-ahead receipt records one attempt,
`failed_or_unknown`, `RuntimeError`; zero completed replies. No automatic retry
or replacement model was used. The original receipt did not capture the HTTP
status or response identifier. Neither the cause, provider execution outcome,
usage, nor charge can be established from this receipt. Do not interpret unknown
usage as zero or classify this as a quota, credential, or model failure without
additional evidence.

No live paired comparisons or quality judgments exist. The intervention's
behavioral effect remains unresolved. Frozen inputs, partial logs and the failed
receipt are retained locally under `runs/hold_action_live_20261110/`.

A subsequent offline-only change records an HTTP status only when the exception
exactly matches the harness's bounded transport error template. Arbitrary error
text is never copied. This does not reconstruct the missing status from the
failed run, and no further live generation was made after the change. The full
suite now passes 97 tests.

A separately initiated run must use a new directory, retain the old unknown
attempt, pass simulation again and record its new revision. Do not resume or
silently overwrite the interrupted comparison. The product's default inquiry
prompt remains unchanged.

## Explicitly requested retry

The user subsequently requested a retry. Revision `996cc51` passed a fresh
144-generation dummy simulation, then began a separate live run under
`runs/hold_action_retry_live_20261110/`. The first workpad request returned HTTP
402 and the runner stopped: one attempted generation, zero completed replies.
No automatic retry or replacement model was used. The prior run remains intact;
its missing HTTP status cannot be inferred from this new result.

Mistral documents 402 as Payment required. Billing configuration, payments and
credit availability need checking before another live run. The account's actual
balance or invoice state was not inspected, so this is not a confirmed diagnosis
of exhausted credit. See [error glossary](https://docs.mistral.ai/resources/error-glossary)
and [billing documentation](https://docs.mistral.ai/admin/billing-usage/billing).
The behavioral comparison remains unresolved.

## Separately requested GPT replication

The subsequent OpenAI run completed all 144 generations. See the
[GPT readout](2026-11-10-hold-action-gpt.md) for validity losses, preserved
comparisons and unblinded observations. The two Mistral attempts above remain
unchanged; their missing outcomes are not filled in by the GPT run.
