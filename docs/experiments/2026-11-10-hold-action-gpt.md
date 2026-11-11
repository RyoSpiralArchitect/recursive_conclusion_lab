# GPT action-wording comparison: execution complete, primary outcome unresolved

The goal remains useful dialogue while conclusions stay open. After the two
interrupted Mistral attempts, an explicitly requested OpenAI replication ran at
revision `55927b869ad92bc9ab6d313a499ffd09ee635497`, using catalog-confirmed
`gpt-5.6-luna`. The frozen scripts, order, prompts, limits and exclusion policy
were unchanged. This is not a provider ranking: there are no completed Mistral
pairs from this experiment.

## Execution and validity

All 144 generations returned: 72 replies and 72 workpads, with no transport
failures, automatic retries, repair or judge calls. Receipts, events and 72 turn
timings reconciled. Source-copy hashes and the credential scan passed.

| Observation | Current instruction | Current-action instruction |
| --- | ---: | ---: |
| Generations returned | 72 | 72 |
| Valid workpads | 36/36 | 32/36 |
| Replies ending normally | 28/36 | 27/36 |
| Replies stopped at the 900-token limit | 8/36 | 9/36 |
| Input tokens reported | 163,039 | 168,918 |
| Output tokens reported | 34,296 | 35,296 |
| Calls missing input/output usage | 0 | 0 |

All 72 workpad generations ended normally at the provider; four then failed the
local schema validator. Each failure was `next_step` exceeding 400 characters:
museum repetition 1 turn 6 (440), seeds repetition 1 turn 5 (583), and seeds
repetition 2 turns 5 and 6 (562 and 476). All four occurred in the action arm.
Those turns proceeded with cleared workpad state; they are not successful
applications of the intended intervention. No rejected memo was repaired.

All six comparison pairs are retained. Only two satisfy the frozen
complete-treatment rule, both newsletter repetitions. Museum and seeds are
excluded from that subset because of reply truncation and/or invalid workpads.
That topic-selected subset must not replace the full comparison. Usage totals
are provider-reported token counts, not a verified monetary bill.

## Unblinded observations, not quality scores

- Both conditions produced concrete trials and provisional recommendations in
  inspected turns. For example, both seeds repetition 1 turn 5 replies recommend
  a small documented-seed trial with optional returns and list assumptions that
  could change it. The action arm's memo was invalid on that turn, so this reply
  cannot establish that the candidate memo caused the behavior.
- The candidate makes current action explicit in some valid memos, but its four
  length failures show a competing operational cost in this sample. Memo verbs
  alone do not establish better dialogue.
- Spanish workpads appeared despite English dialogue and the instruction to use
  the dialogue's language; inspected user-facing replies remained English.
  Schema validity does not check this semantic requirement. This mismatch is
  retained, with no translation or prompt adjustment during the run.
- Seeds repetition 1 turn 6 exposes a constraint-retention concern in both arms.
  The user prohibits overnight storage of returned packets, without exempting
  reviewed packets. The action reply says an accepted packet is placed with
  verified stock; the current reply permits approved stock in the cabinet and
  asks whether approved packets can stay overnight. Both revise the return
  process, but neither establishes permission for the possible reviewed-return
  exception. This is a concern for independent review, not an assigned score.
- Newsletter repetition 2 turn 6 replies in both arms shift toward authorized
  pickup points and preserve aggregate-only records. The current reply also
  notes that low pickup may reflect access barriers rather than low interest.
  These examples show useful revision without proving comparative superiority.

The primary outcome remains unresolved: independent blinded judgments have not
been collected. Blank answer sheets and all six blind conversation packets are
available locally, separately from the unblinded mechanism readout. Neither
model behavior nor infrastructure checks justify changing the product default.

## Validation and next decision

The execution revision passed 99 local tests, changed-file Ruff, whitespace
checks and a fresh offline simulation (144 generations, 72 valid workpads, six
complete pairs) before live calls. Dummy success established wiring, not the
live validity rates above. No CI workflow is configured in this checkout.

Local evidence is retained under `runs/hold_action_gpt_live_20261110/`, including
the frozen inputs, raw generation receipts, event logs, metrics, review packets,
derived turn excerpts and `ARTIFACTS.sha256`. Earlier failed attempts remain
separate. Raw provider metadata and credentials are not included in this PR.

Before another experiment, use these existing packets to judge useful progress,
premise revision and retention of hard constraints. A subsequent, separately
frozen intervention could target brief current actions and explicit constraint
retention; it should distinguish instruction changes from token-budget changes.
No additional live run or default change was made as part of this readout.
