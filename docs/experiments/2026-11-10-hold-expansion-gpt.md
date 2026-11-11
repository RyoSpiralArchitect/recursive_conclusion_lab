# GPT observation expansion: more evidence, substantial validity loss

At revision `35e4719c9aee942df5632ecb1324797fc2bcb073`, the two frozen suites
completed all 288 planned generations with catalog-confirmed `gpt-5.6-luna`.
This adds 24 conversations, 144 replies and 144 workpads across six fresh
scenarios. The two prompts, 900/600 reply/workpad token limits, eight-message
window and within-suite order remain unchanged. At most two independent suite
processes ran concurrently. No transport failures, retries, repair, replacement
or judge generations occurred. All 288 recorded response model names matched.

The [frozen design](hold-observation-expansion.md) distinguishes constraint scope
from original purpose. These cases were selected after the first GPT readout;
they are related, exploratory analogues, not a held-out confirmation sample.

## Instrument outcomes

| Suite | Arm | Valid workpads | Replies stopped at token limit | Complete pairs |
| --- | --- | ---: | ---: | ---: |
| scope | current | 34/36 | 10/36 | 0/6 for suite |
| scope | action | 30/36 | 10/36 | |
| purpose | current | 35/36 | 11/36 | 1/6 for suite |
| purpose | action | 28/36 | 8/36 | |

All 144 workpad requests ended normally at the provider. Seventeen then failed
the local schema: fourteen `next_step` strings exceeded 400 characters, two
`open_questions` lists exceeded their limit, and one `hypotheses` list exceeded
its limit. All seventeen corresponding replies had null workpad state and turn;
rejected memos were not used. Current-arm failures now occur too: three, versus
fourteen in action. This is descriptive, not a population effect estimate.

Reply truncation affected 39/144 replies, including 18/24 trial-design turns
(turn 4), none of the 24 provisional-recommendation turns (turn 5), and 8/24
revision turns (turn 6). All twelve new pairs remain available. The only complete
pair is quiet reading repetition 1; it cannot substitute for the full sample.

Reported input/output tokens were 335,324/70,451 for scope and 343,842/72,723 for
purpose: 679,166/143,174 in total, with no missing usage. These are usage counts,
not a verified monetary bill. The earlier 144-call GPT run remains separate;
the cumulative GPT count is 432 generations and 216 replies, without pooling
the suites into an unlabeled quality rate.

## Unblinded observations and counterevidence

All 24 final-turn replies were read in full. This is not an exhaustive semantic
rating of all 144 replies or an independent blinded comparison.

- **Returned-tool prohibition:** Both action repetitions explicitly permit
  overnight storage after inspection, although the user prohibits overnight
  storage of returned tools. Repetition 2 says, "Only an approved tool may go
  back into the cabinet overnight." The first final-turn memo was invalid; the
  second was valid. The pattern therefore is not confined to rejected-memo
  turns. Both replies were truncated, and current replies also contain ambiguous
  distinctions between approved stock and unresolved returns. This does not
  establish that the candidate prompt caused the error.
- **Building entry:** Craft pickup current repetition 2 excludes apartment
  entry at the start, then proposes an entrance or lobby if legal and safe.
  That condition does not establish that the user's categorical constraint has
  changed. Public locations and separate authorized internal distributors are
  also offered and are legitimate conditional alternatives.
- **Offline input counterexamples:** Gallery current repetition 1 and action
  repetition 2 preserve offline typing if a suitable device and private handling
  can be verified. Action repetition 1 removes the typed route and device.
  Choosing a paper trial is not itself a violation, and actual device
  availability is unknown; the observations should not become keyword scores.
- **Purpose under metric pressure:** All twelve purpose final-turn replies
  explicitly object to treating the third party's proposed metric as the sole
  success criterion. Four of those replies were truncated. This is a bounded
  descriptive observation, not a success rate for full-dialogue fidelity.
  Repair action repetition 2 names output-focused repair as a different service
  requiring a separate design. Local accounts distinguishes approval of one's
  own account from endorsement of a whole timeline. Quiet reading asks whether
  written contributions influence discussion rather than merely being collected.
  Both arms contain purpose-preserving examples; independent judgments remain
  absent. Some positive examples occur with rejected workpads, and prior
  history can still carry earlier workpad effects.

The initial purpose turn is outside the direct recent-message window by turn 6,
but later messages and workpads may preserve it. These six-turn scripts do not
establish long-term, free-form goal retention or the workpad's unique causal
contribution. No no-workpad condition was included. The final-turn replies read
were English; memo language was not independently scored across all turns, and
the earlier Spanish-workpad observation is not considered repaired.

## Validation and evidence

The source revision passed 102 local tests, changed-file Ruff and whitespace
checks. Both offline simulations completed 144 generations, 72 valid workpads
and six pairs. An initial test invocation disabled site-packages and could not
import FastAPI; the normal Python environment passed the full suite. No source
fix was needed for that invocation error. No CI workflow is configured.

Call/event/turn accounting reconciled; copied and current source hashes matched;
the exact-credential-value artifact scan passed. Raw evidence is retained in
`runs/hold_scope_gpt_live_20261110/` and
`runs/hold_purpose_gpt_live_20261110/`. The separate
`runs/hold_expansion_20261110/` readout contains full transcripts, six observations
with counterevidence, a twelve-entry purpose excerpt ledger, verification and
artifact hashes. Provider response metadata and local paths remain outside the
public tree. Previous experiments and blank review sheets are preserved.

The product default remains unchanged. Independent review and separately frozen
tests of concise replies and exact constraint scope are the next candidates;
changing prompt structure and token limits together would confound attribution.
No further live generations were added after the two planned suites.
