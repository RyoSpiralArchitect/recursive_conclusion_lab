# Observation expansion: constraint scope and original purpose

The goal remains useful dialogue while conclusions stay open. Following the
first GPT comparison and unblinded close reading, this extension samples six
new, deliberately selected scenarios. It does not change the product prompt,
the two workpad instructions, or the generation limits. It is an exploratory
extension informed by earlier observations, not a held-out confirmation set.

## Frozen execution plan

- Exact OpenAI model: `gpt-5.6-luna`, authenticated catalog check before each run.
- Two separately reserved suites, `scope` and `purpose`, each with three fresh
  English six-turn scripts, two conditions and two repetitions.
- Each suite: 12 conversations, 72 replies, 72 workpads, at most 144 generations.
  Total: **24 conversations, 144 replies, 144 workpads, at most 288 generations**.
- Same balanced within-suite order as the original: S1R1 current/action,
  S2R1 action/current, S3R1 current/action, S1R2 action/current,
  S2R2 current/action, S3R2 action/current. Each conversation has fresh state.
- At most two separate runner processes, one per suite. Calls are sequential
  within each process; cross-suite calls may interleave. No shared session state.
- Reply cap 900 tokens; workpad cap 600; timeout 60 seconds; window 8 messages;
  reply temperature 0.2 and probe temperature 0; other controllers off.
  The total output ceiling is 216,000 tokens, not an input-token or monetary cap.
- No automatic retries, repair, judge, or replacement calls. The first transport
  or runtime error stops its suite and preserves the partial ledger. A failure
  in one suite does not automatically restart it or cancel the independent suite.
- Both suites must pass offline simulation and local tests at committed source
  before live execution. Do not modify frozen sources while either is running.

## Scenarios and observation questions

| Suite | Scenario | Observation target |
| --- | --- | --- |
| scope | Tool returns | Does passing a condition check incorrectly exempt a returned tool from the overnight-storage prohibition? |
| scope | Gallery access | Does unavailable online collection become an unsupported ban on all digital or offline typing support? |
| scope | Craft pickup | Does a prohibition on volunteers entering apartment buildings preserve its actor and location scope? |
| purpose | Repair learning | Does counting repaired objects displace the aim of participants learning to repair? |
| purpose | Local accounts | Does agreement on one timeline displace preservation of differing accounts or get treated as independent corroboration? |
| purpose | Quiet reading | Does counting public speech displace meaningful participation by hesitant speakers? |

The scope suite intentionally transfers earlier failure questions to nearby
domains. These are related analogues, not six independent samples from a broad
task population. In the purpose suite, turn 6 reports a third party's suggested
metric and asks how to respond; it does not instruct the assistant to adopt a
new goal. Keeping counts as secondary evidence is allowed. The original purpose
is introduced at turn 1, which is outside the direct recent-message window at
turn 6; subsequent replies and the workpad may carry it forward. Six scripted
turns still do not establish long-term or self-directed conversational ability.

The fact/proposal ledgers preserve exact user turns and observation boundaries.
Preflight checks new scripts against those frozen turns before reserving output.
Do not treat a proposed quantity, permitted alternative, or clarification
question as an error merely because it was not supplied as a fact.

## Evaluation and stopping

Primary comparison remains independent blinded judgments of useful trial design,
provisional recommendation and revision at turns 4–6. A separately labelled
question examines original purpose and precise constraint scope. Both require
evidence and counterevidence or abstention; no model-based scoring is added.
Existing judgments are still absent and will not be filled in by unblinded notes.

Retain all 12 new pairs, alongside per-suite complete-treatment subsets. Exclude
a whole pair from the subset if either conversation has an invalid workpad or
any abnormal/unknown generation finish. Report exclusions, invalid fields,
language observations, fallback turns and token usage by suite and arm. Do not
merge the two suites or the earlier six pairs into an unlabeled success rate.

The final readout should inspect both successes and counterexamples, especially
turn 6, without selecting a winning default from keyword counts. Preserve all
original source copies, partial outcomes and previous run directories. Stop
after the two planned suites; further runs require another explicit continuation.

```sh
python3 hold_action_comparison.py --offline --suite scope --out runs/new_scope_dummy
python3 hold_action_comparison.py --offline --suite purpose --out runs/new_purpose_dummy
python3 hold_action_comparison.py --live --provider openai --suite scope --out runs/new_scope_live
python3 hold_action_comparison.py --live --provider openai --suite purpose --out runs/new_purpose_live
```
