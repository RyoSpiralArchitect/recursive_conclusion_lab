# Scale and cost: workpad ablation with verbatim user replay

The destination remains useful conversation with conclusions held open. This
study asks whether the per-turn generated workpad is worth its latency and input
cost, and whether deterministic replay of missing user source text can preserve
context without that auxiliary generation. It does not claim that more context
will repair semantic mistakes about permissions.

## Before-run audit

The previous 384-call run's brief arm spent 53.1% of measured adapter time in
workpad generation. Its median turn was 11.75 seconds (median probe 6.35,
reply 5.37; component medians are not additive). Total local residual was
0.704 seconds over 48 turns, about 15 ms per turn. The timer includes adapter
and network time, not just remote model computation. These are recorded,
unpaired timing observations, not a hardware benchmark.

The eight-message window loses original user turns 1 and 2 by turn 6.
Generated workpads and earlier replies may carry their content, but are not
verbatim source. The new opt-in `--hold-user-replay` quotes only missing user
messages. It retains the origin and then recent missing user turns, with at most
16 messages and 12,000 text characters. Messages are never cut mid-constraint;
omissions are disclosed with count and endpoints. This is not an active-constraint
classifier. Later explicit user revisions remain in chronological order.
Assistant claims cannot become user source. Quotes are data below system
authority and are not added to the durable dialogue. Existing saved history
reconstructs them on resume; no second mutable ledger is introduced.
Replay is supplied to replies; the probe input path is unchanged.

## Frozen design and budget

All arms use the same merged hold reply prompt plus the previous brevity
addition. No exact-scope prompt addition or new answer-length instruction is
introduced in this experiment.

| Arm | Workpad calls per turn | Verbatim missing-user replay |
| --- | ---: | --- |
| workpad | 1 | off |
| no_workpad | 0 | off |
| source_replay | 0 | on |

The first contrast isolates removing the workpad. The second isolates adding
source replay without a workpad. The third measures the replacement package.
All three pairings are retained; no winner is selected by a keyword score.

- **8 cases × 3 repetitions × 3 arms × 6 turns = 72 conversations and 432 replies.**
  The workpad arm adds 144 workpad generations: **at most 576 generations total**.
- Two independent suites of four cases: regression and transfer. Each has
  36 conversations, 216 replies, 72 workpads and a 288-generation ceiling.
- Regression repeats tool returns, craft pickup, gallery access and repair
  learning. Transfer adds camera returns, mending skills, explicit storage
  revision and explicit primary-purpose revision.
- The new cases are deliberately constructed analogues and positive revision
  controls, not a random sample or held-out confirmation of a selected winner.
  In storage revision, the user now permits checked passing returns overnight.
  In purpose revision, the user explicitly changes the learning goal to repair
  completion while retaining voluntary participation. Sticking to superseded
  restrictions or refusing that explicit goal change is not successful anchoring.
- Cyclic ABC/BCA/CAB orders rotate by case and repetition. Each case puts every
  arm in every position once. State is fresh per conversation. Seeds 7–9 are
  local bookkeeping, not matched provider draws.
- Exact model `gpt-5.6-luna`; authenticated catalog GET before each reservation.
  Reply/probe caps 900/600, temperatures 0.2/0, timeout 60 seconds, window 8.
  Other controllers off. Two processes at most; each runs sequentially.
- Output ceiling 475,200 tokens; this is not a monetary or input-token cap.
  No retries, repair, judge, replacement or unplanned continuation calls.
  The first transport/runtime failure stops its suite and retains partial receipts.
- Both suites must pass offline simulation at committed source before live calls.
  Validate every script, fact ledger, provider/config and optional adapter before
  reservation. Do not change frozen sources while either suite is running.

## Measurements and validity

Report call counts, provider-reported input/output usage, per-dialogue mean turn
latency and matched case/repetition differences, alongside per-arm medians.
Timing is observational under two concurrent processes; do not claim causal
hardware speedup. Log input character counts and request-message hashes.

Retain the previous whitespace counter, including Markdown markers, and the
180 threshold. Report normal finish, nonempty response, length and short-normal
completion separately from semantic quality. The no-workpad arms have no
workpads by design; they are not counted as 100% valid generated workpads.

All 72 planned pairs remain available. A separate complete-treatment subset
excludes the whole pair if either dialogue has any abnormal/unknown finish,
empty reply or invalid expected workpad. Also report all-arm complete blocks.
Keep the regression and transfer strata labeled; do not pool earlier experiments
into a success rate.

Read all 72 final replies unblinded, including the explicit revision controls.
Also read the full turn 4–6 text for repetition 1 of every case/arm (72 replies,
24 overlapping with final-turn coverage), then use full preceding transcripts
for ambiguous cases. Preserve counterevidence. Independent blind review
packets and blank answers are generated, but author reading does not fill them.
Style may reveal the treatment despite label hiding.

New replay source must exactly match the corresponding historical user turns,
contain no assistant-derived assertions, and disclose any omission. It must not
silently make an obsolete constraint active again. Track failed workpads and
their clearance separately. Any behavioral gain is limited to these scripts.

Engineering improvement is supported if the replay arm makes exactly one
generation per turn, preserves missing source within its declared bounds,
round-trips through persistence, and passes offline contracts. Behavioral
improvement additionally requires useful replies and correct handling of both
retained and explicitly revised constraints; lower latency does not establish it.

```sh
python3 hold_source_comparison.py --offline --suite regression --out runs/new_regression_dummy
python3 hold_source_comparison.py --offline --suite transfer --out runs/new_transfer_dummy
python3 hold_source_comparison.py --live --suite regression --out runs/new_regression_live
python3 hold_source_comparison.py --live --suite transfer --out runs/new_transfer_live
```
