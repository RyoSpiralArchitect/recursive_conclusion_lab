# GPT comparison: exact scope and concise completion

The purpose remains a chat harness that makes useful progress while conclusions
stay open. Two reply additions were tested independently: preserve original
purpose and exact constraint scope; finish concisely. The workpad instruction
and product defaults were unchanged.

**Result:** the brevity addition eliminated provider truncation in this selected
set and substantially reduced length. It did not reliably satisfy the declared
180-word counter. The scope addition did not resolve the returned-tool failure;
neither condition is established as a semantic-quality winner.

## Execution and mechanical results

Frozen source: `eb0d4bcadf5804289021577a4032fa22bad31467`.
See the [prospective design](hold-constraint-brevity.md) and
[machine-readable aggregate](data/hold_constraint_brevity_20261110.json).

Four reused regression scenarios, four arms, two repetitions, six user turns:
**384 generation calls, 32 conversations, 192 replies and 192 workpads**.
Both 192-call suites completed with the exact `gpt-5.6-luna` model, no transport
failures, retries, repairs, judge calls or replacements. Two authenticated
catalog GETs preceded generation and output reservation. The reply/workpad
ceilings remained 900/600 tokens in every arm.

| Reply condition | Replies | Truncated replies | Median word count | At most 180 and normal finish | Valid workpads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 48 | 14 | 584 | 0/48 | 46/48 |
| Exact scope and purpose | 48 | 5 | 481 | 0/48 | 47/48 |
| Brief | 48 | 0 | 183 | 16/48 | 48/48 |
| Both additions | 48 | 0 | 183.5 | 20/48 | 47/48 |

The predefined counter splits on whitespace, including Markdown markers and
list numbers; it is not a linguistic word tokenizer. Keep this definition and
threshold fixed rather than changing them after observing results.

For turns 4–6, where trial design, recommendation and revision are requested:

| Condition | Normal finish | Median word count | At most 180 and normal finish |
| --- | ---: | ---: | ---: |
| Baseline | 15/24 | 576.5 | 0/24 |
| Exact scope and purpose | 19/24 | 504.5 | 0/24 |
| Brief | 24/24 | 189 | 5/24 |
| Both additions | 24/24 | 189.5 | 5/24 |

Adding brevity improved finish counts over baseline in six of eight matched
case/repetition blocks and tied in the two craft blocks, which already finished.
Adding brevity to the scope condition improved four blocks and tied four.
These are descriptive comparisons of a deliberately selected small set, with
dependent turns and shared conversations, not independent effect estimates.
The scope addition also reduced observed truncation relative to baseline, but
that does not establish better constraint fidelity.

Reported usage totals are **681,365 input and 139,332 output tokens**, with no
missing usage entries. These are provider-reported counts, not a verified bill.

## Semantic reading and counterexamples

All 96 turn 4–6 replies were read unblinded against the frozen user history.
This is not full-dialogue scoring: earlier assistant replies were not all
independently rated. Eight final tool-return workpads were additionally read.
Independent judgments remain uncollected.

**Tool returns: scope remains unresolved.** The baseline repetition 1 reply
first restates that the prohibition applies regardless of inspection outcome,
then allows the cabinet to hold tools that have passed inspection. Its valid
workpad had preserved the overnight restriction, so a correct memo did not
guarantee a consistent reply.

The scope-only repetition 1 answer instead says:

> A returned tool cannot be left overnight awaiting inspection.

That narrows the prohibition to a processing state. Repetition 2 mixes similar
narrowing with a final requirement for a same-day path for both passing and
failing returns. Preserve that counterevidence; it is not a clean categorical
pass or a single unambiguous permission.

Both combined-condition repetitions return passing tools to the cabinet or
available inventory, while specifying separate end-of-day disposition for failed
tools. Daytime movement itself is allowed; the unresolved point is the overnight
treatment of every returned tool. Brief repetition 2 explicitly requires no
returned tools left onsite, but lawful, safe disposition still needs confirmation.
The added instruction did not consistently separate passing inspection from
permission to retain a returned tool overnight.

**Craft pickup: the actor boundary can be retained without the new instruction.**
Scope-only repetition 2 prohibits volunteers entering apartment buildings and
separately asks whether an authorized intermediary can operate under the same
privacy constraint. Baseline replies also preserve exterior-only volunteer
handoffs in this run. Permission for an exterior location or an independently
authorized actor is not itself an exception to the volunteer-entry prohibition.
Some wording about doors or intermediaries remains conditional and warrants
careful reading; these examples are not full quality certifications.

**Gallery access: do not score paper selection as a violation.** Baseline
repetition 2 and brief repetition 2 explicitly preserve a privacy-checked offline
typing option, conditional on actual availability. Other replies choose paper
and speaking and acknowledge excluded preferences. A paper trial is allowed by
the case; omission of typing alone does not prove an invented blanket ban.
This set does not show a clear semantic advantage for the scope addition.

**Repair learning: all eight final replies explicitly resist completion-only
evaluation, including both baseline replies.** Three of those eight replies are
truncated. Short replies still contain learning, participant choice and remaining
uncertainty. However, combined repetition 2 ends by suggesting the sponsor's
purpose may change the trial's emphasis while preserving participant choice.
That might be a compatible emphasis change, but choice alone is not the user's
original learning objective. Do not turn the eight explicit objections into an
8/8 purpose-retention success rate.

## Integrity, attrition and limits

- 188/192 workpads were schema-valid. Three failures exceeded the open-question
  list limit; one had invalid fields. All four were cleared before the reply.
  All workpad provider finishes were normal. There were 19 truncated replies.
- All 32 planned review pairs were retained. Only **15/32 pairs** meet the
  complete-treatment rule (scope 10/16; anchor 5/16). Only **2/8 four-arm blocks**
  are complete, both craft pickup repetitions. The complete subset therefore
  cannot stand in for the whole experiment.
- Every generation receipt reconciled with its event; full system prompt hashes,
  temperature, caps, source copies and unchanged source hashes were verified.
  Two available credential values were checked for exact matches in the local
  artifacts; none were found. Raw provider records remain local.
- All 111 local tests, changed-file Ruff and whitespace checks passed. Both
  committed-source dummy suites completed 192 calls and 16 review pairs each.
  No CI workflow is configured in this repository.
- Probe instructions were identical, but their outputs and subsequent histories
  were not. In the eight tool final memos inspected, the two brief-arm valid
  memos used Chinese despite English dialogue; language was not scored over all
  192 memos. This run does not resolve that separate issue.

The next focused hypothesis is that the harness needs to keep **inspection
status, storage permission and the original purpose as distinct state**, so
changing one status cannot silently erase another constraint. That is a proposal
for a later comparison, not an implemented or validated remedy in this run.
Product defaults remain unchanged; no further live calls were added.
