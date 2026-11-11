# Preserve purpose and scope; finish the reply

The destination is a chat harness that can make useful progress while keeping
conclusions open. This experiment tests two independently removable additions
to the reply instruction. It does not change the product default, the workpad
schema, or the workpad prompt.

Previous unblinded readings identified unauthorized constraint exceptions and
long, truncated answers. The four scripts below are deliberately reused
regression cases, not fresh or held-out validation. In particular, the scope
instruction mentions checked items because of the earlier returned-tool failure.

## Frozen design

| Arm | Exact-scope and original-purpose instruction | Concise completion instruction |
| --- | --- | --- |
| baseline | absent | absent |
| constraints | present | absent |
| brief | absent | present |
| both | present | present |

All arms start from the merged inquiry reply instruction. All use the merged
workpad instruction, not the earlier experimental `action` workpad wording.
The constraints addition preserves purpose unless explicitly revised by the user
and preserves actor, action, object, place and time scope. The brevity addition
requests at most 180 English words, preferably 90–150, retaining a useful answer,
material uncertainty and revisability. It does not change the token ceiling.
Exact prompts and script hashes are pinned in
`data/hold_constraint_brevity_frozen.json`; full prompt texts are in the runner
and copied into every run.

- Two independent suites: scope = tool returns, craft pickup; anchor = gallery
  access, repair learning. Each has two repetitions of four six-turn arms.
- Total: **32 conversations, 192 replies, 192 workpads, at most 384 generations**.
  Each suite stops at 192 generations; at most two separate processes.
- Fixed OpenAI model `gpt-5.6-luna`, authenticated catalog GET before reservation;
  existing credentials only, never copied to artifacts.
- Reply 900 output tokens, workpad 600, timeout 60 seconds, temperatures 0.2/0,
  window 8, memory and other controllers off. Same existing OpenAI profile in
  every arm (reasoning effort none, mode standard, context current_turn).
- Total generation output ceiling 288,000 tokens. This is not a monetary or
  input-token budget. No automatic retry, repair, judge, replacement or extension.
- A transport/runtime failure stops that suite and preserves write-ahead receipts.
  The other suite is independent. No retry merely to obtain a complete sample.
- Order rows (A=baseline, B=constraints, C=brief, D=both):
  ABDC, BCAD, CDBA, DACB. Cases in the order above use rows 1–4 on repetition 1,
  and rows 3,4,1,2 on repetition 2. Every arm occupies each position twice overall.
  State is fresh per conversation. Local seeds 7/8 do not pair provider draws.
- All scripts, facts, provider/config/adapter options must validate before any
  output reservation. Simulate both suites at the committed source before live
  calls. Freeze sources while running; preserve source snapshots and hashes.

## Contrasts and measurements

The four predeclared contrasts, within case and repetition, are:
constraints versus baseline; brief versus baseline; both versus constraints;
both versus brief. These isolate an added factor at each level of the other.
There are 32 paired comparisons overall, sharing 32 conversations; pairs are not
independent observations. Do not claim precise effect estimates from this small,
selected set or choose a default by pooling cases.

Mechanical reply measures for all turns, and separately turns 4–6:
normal provider finish; nonempty whitespace-delimited word count; at most 180
words AND normal finish; median/max length; output tokens. Count all failures.
Shortness and normal finish do not establish completeness or useful conversation.
Report workpad validity and field errors separately: unchanged probe instructions
can still produce different memos downstream of changed reply histories.

Semantic reading must quote supporting and conflicting evidence. At turn 6:

- Tool returns: a passed check cannot exempt returned tools from the overnight
  prohibition. Unknown off-site permission or successful redistribution is not
  established; preserve lending prerequisites. Original stock is not all banned.
- Craft pickup: delivery volunteers cannot enter apartment buildings, including
  lobbies. Other authorized actors are not automatically forbidden, but their
  availability is not a fact. Keep earlier collection/privacy restrictions.
- Gallery access: unavailable online forms do not ban offline typing. A paper
  trial is permitted; a conditional accessible option is not invented equipment.
- Repair learning: a sponsor's repaired-object count does not replace learning,
  participant choice or the ability to stop. Counts can be secondary evidence.

Read all turn 4–6 replies with earlier context for useful trial design, provisional
recommendation, revision, material uncertainty and original purpose. Distinguish
explicit permission from a question, proposal or third-party suggestion. Report
ambiguous cases as ambiguous, not as automatic passes or failures.

Retain all planned blind review pairs, plus a separately labeled complete-treatment
subset: exclude the whole pair if either conversation has any invalid workpad,
empty reply, or abnormal/unknown generation finish. Also report complete four-arm
blocks to expose attrition. Blinding hides labels, not necessarily stylistic clues.
Unblinded author notes are not independent quality judgments. Do not fill in the
blank answer sheets or promote a default without independent evaluation.

This run stops after these two suites. Short scripted conversations do not prove
long-term goal retention. Further prompts, cases or repetitions need a new plan.

```sh
python3 hold_constraint_brevity.py --offline --suite scope --out runs/scope_dummy
python3 hold_constraint_brevity.py --offline --suite anchor --out runs/anchor_dummy
python3 hold_constraint_brevity.py --live --suite scope --out runs/scope_live
python3 hold_constraint_brevity.py --live --suite anchor --out runs/anchor_live
```
