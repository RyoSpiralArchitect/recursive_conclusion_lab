# Current-response action comparison

The development goal remains useful dialogue while conclusions stay open.
This experiment follows the completed October diagnostic replication: compare
its unchanged workpad instruction with the previously proposed wording that
assigns `next_step` to the assistant's current reply and places facts requiring
outside confirmation in `open_questions`. The product default is unchanged.

## Frozen design

- Exact model: `mistral-small-2603`, checked against the authenticated catalog.
- Three fresh English scripts: museum feedback, seed lending, newsletter distribution.
- Two workpad conditions, two repetitions, six turns each: 72 replies and 72
  workpads, at most 144 generations. No retry, judge, repair, or replacement calls.
- Pair order: S1R1 current/action, S2R1 action/current, S3R1 current/action,
  S1R2 action/current, S2R2 current/action, S3R2 action/current.
- Reply limit 900, probe limit 600, timeout 60 seconds. Recent window eight messages,
  unchanged from the diagnostic runner; memory and other controllers disabled.
  Local seed does not create matched provider random draws.
- Only the workpad system instruction differs. Schema, validators, fallback and
  reply instructions remain identical. Prompt hashes verify the exact proposal.
- User facts and unestablished assumptions are frozen in
  `docs/experiments/data/hold_action_facts.json` before any generation.

## Outcomes

Primary: independent blinded paired judgments of how replies fulfill the trial,
recommendation and revision requests at turns 4–6, using the whole conversation.
Record A/B/Tie or abstain, with evidence and counterevidence. Do not score by first
words or require one preferred activity. Without independent judgments, the
primary comparison remains unresolved; unblinded mechanism observations cannot
select a winning default.

Secondary: whether the memo's action addresses the current request; unsupported
assertions versus explicitly labelled proposals; superseded hypotheses; schema
validity, finish reasons, usage and missingness. A necessary or requested question
is allowed. An invented example is an error only when asserted as an established
fact or obligation. Preserve all six pairs; report complete-treatment pairs
separately, excluding both members' dialogue if any generation has abnormal or
unknown termination or any workpad is invalid. Never silently substitute that
selected subset for the full comparison.

## Execution

```sh
python3 -m unittest discover -s tests -q
python3 hold_action_comparison.py --offline --out runs/hold_action_dummy_new
python3 hold_action_comparison.py --live --out runs/hold_action_live_new
```

Use a new directory every time. Live execution requires committed source files
and an existing environment credential. It checks the exact model before output
reservation. Call intent is written before dispatch; interrupted requests remain
unknown and must not be retried automatically. Source copies, prompt texts,
call/event reconciliation, complete and all-pair review packets, and blank
answer sheets are retained. A `complete` run is execution evidence, not a quality
claim. The output-token ceilings are not an input-token or monetary cap.
