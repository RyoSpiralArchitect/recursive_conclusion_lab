# October 18 hold-dialogue pilot: execution, attrition, and premise handling

The pilot completed all 48 generations, but accepted only 10 of 12 workpads.
Both workpad dialogues rejected their first memo. Under the declared
complete-treatment rule, no pair remains for the primary comparison of hold
instructions versus hold instructions plus a valid memo at every turn.
This run supports concrete implementation and evaluation questions, not a claim
that the workpad improves dialogue.

The [compact evidence record](data/hold_pilot_20261018.json) contains the observed
aggregates, all 12 workpad events, and nine source-linked reply excerpt entries.
Event line numbers, event hashes, and full-log hashes refer to the preserved
original local run. Full logs and blind keys are not bundled in this repository;
these excerpts are a bounded evidence selection, not a replacement for all raw
artifacts. No human quality judgments were collected. The reading below is
post-hoc and unblinded.

## Provenance and design

The local `hold_dialogue_pilot.zip` was 76,237 bytes, SHA-256
`9cc0edf66e85f635a334a23638e52346a70e20db01782e4e664fd4bb70b3411d`.
Its independent byte identity with the Space copy was not verified. Its specified
source was `b44474358ed95dc1cde8e5cad2668bc1a29d4e8e`. At execution, the four
experiment modules were byte-identical to `main` at
`798244c540b99cef71984f42e93b188955cedacc`; additional changes on the pinned branch
concerned Playtest UI/request handling, which this CLI run did not exercise.

After 88 offline tests and a fresh 48-generation dummy run, the unmodified ZIP
runner used the authenticated, fixed model ID `mistral-small-2603`. It started at
2026-10-18 10:19:35 UTC (19:19:35 Asia/Tokyo) and ran for 180.58 seconds.
There were no retries, extra judge calls, embeddings, memory compression,
deferred-intent planning, or changes to generation limits during the run.

The three conditions were executed in fixed order, each with two six-turn
dialogues: baseline, hold instructions, and hold instructions plus a workpad.
Replies used temperature 0.2 and 900 output tokens; workpads used temperature 0
and 600 output tokens. All requests had a 60-second HTTP timeout.

The script asks how to use a shared room, then adds a low budget, adjacent therapy
rooms, two volunteers, a request for a provisional recommendation, and finally
evening-only volunteer availability. The initial user turn already asks to keep
the final choice open. Therefore baseline is **not absence of an open-inquiry
goal**; the experiment concerns the added system instructions and external memo
on top of that shared task request.

## What completed, and what did not

| Condition | Reply / workpad calls | Valid workpads | Mean turn time | Input tokens | Output tokens |
|---|---:|---:|---:|---:|---:|
| baseline | 12 / 0 | — | 4.18 s | 10,182 | 4,695 |
| hold_prompt | 12 / 0 | — | 3.82 s | 10,433 | 3,549 |
| hold_workpad | 12 / 12 | 10 / 12 | 7.05 s | 29,009 | 5,714 |

All 48 finish reasons were `stop`; all 36 replies reached the last scripted turn.
Both rejected workpads were at turn 1 and reported 205 output tokens. Normal
termination and equal token counts do not identify the rejected texts or their
cause. The old event format recorded `invalid_inquiry_state` without the text or
specific validation reason. An extra field, invalid field type, excessive length,
or extraction failure remains possible; none is established by these receipts.

For both failed updates, the corresponding reply recorded a null workpad and null
update turn. Later updates were accepted and matched the states attached to their
replies. This demonstrates the observed fallback path, not that the model's answer
was correct. At turn 1 there was no earlier memo to discard; later-turn stale-state
clearing is covered by offline tests, not demonstrated by these two live failures.

The complete-treatment subset contains four dialogues and two baseline-versus-
hold_prompt pairs. Both workpad dialogues are excluded, leaving **zero complete
primary pairs**. Full packets still contain all six pairs and could support a
review of the deployed policy including failures. That is a different question
from the benefit of consistently available valid workpads. Attrition occurred
only in the treatment arm, so simply dropping it cannot repair the causal
comparison. Later turns are also downstream of the initial fallback and are not
independent replacement replications.

## Reading the dialogue and the memo separately

**An underspecified next-step field tends toward asking.** Nine of the ten valid
`next_step` values begin with “Ask”; the other begins with “Clarify”. Both turn-5
memos preserve the turn-4 next step asking volunteers to select an activity,
although the current request asks the assistant for a recommendation. In
repetition 2, the entire memo is unchanged between turns 4 and 5. This motivates
testing whether the field represents an action for the current response or an
unresolved external information need. It does not prove that the assistant only
asks questions: both workpad dialogues produce an experiment at turn 4 and a
provisional recommendation at turn 5.

**Reopening occurs, but added detail can become an asserted constraint.** In
workpad repetition 1, turn 6 moves the proposed activity to evening slots and
asks about evening demand, lighting, and access rules. In repetition 2 it states,
“The room can only be staffed 6–9 p.m. on weekdays.” The script establishes
evening-only availability, not those hours or weekdays. The same reply asks
whether two volunteers can cover five evenings a week, inheriting a demanding
schedule from the assistant's own earlier plan. Offering an example schedule can
be useful; presenting its details as already established is the problematic step.

**The control conditions show related failures and useful behavior.** Baseline
repetition 1 speculates that the therapy rooms imply evening/night quiet hours.
Hold_prompt repetition 1 treats unsupervised daytime operation as necessary,
although evening-only opening is another possibility. Hold_prompt repetition 2
changes a previously proposed weekend schedule into an “original ... weekday
window” when describing what must change. Yet the controls also propose trials,
give recommendations, and reopen them. These examples support a shared target
for improvement—preserving the difference between given constraints and generated
proposals—not a ranking of the three conditions.

**Syntactic validity does not establish semantic fidelity.** A valid memo may
repeat a stale next step, retain an unsupported candidate, or summarize a proposal
as if it were settled. Conversely, the reply can satisfy the current request
despite an unhelpful next step. Workpad acceptance, memo revision, and visible
response behavior need separate measurements. “Ask” frequency is a descriptive
clue, not a success/failure rubric.

## Time, usage, and limits of interpretation

The workpad condition took about 1.85 times the hold_prompt mean turn time and
used 2.78 times its input tokens and 1.61 times its output tokens. Workpad output
comprised 3,056 reply tokens plus 2,658 memo tokens. Raw totals were 49,624 input
and 13,958 output tokens; these are not invoice amounts.

Fixed execution order, different response lengths, and different cache hits
confound a general latency estimate. Two repetitions of one English script are
not two scenarios, and the local seeds were not transmitted by the Mistral
adapter. No significance test, provider ranking, long-term-memory claim, Japanese
quality claim, or general improvement claim follows from this pilot.

The immediate implementation response is therefore diagnostic: bounded rejected
text, static validation codes, write-ahead call receipts, and explicit full-versus-
complete comparison accounting. Prompts and admission rules remain unchanged.
The [next experiment](hold-pilot-next.md) asks what actually failed before a
behavioral prompt revision is evaluated.
