# Next experiment: diagnostic replication of the hold pilot

Status: specified; **not executed**. This plan does not schedule or authorize an
automatic continuation. No additional live generations were made to prepare this
PR. The next run should use the merged revision containing the pilot runner and
diagnostics and record that exact clean revision in its manifest.

## Question and intervention

What made both initial workpads unusable in the October 18 pilot, and does that
failure recur under the unchanged dialogue protocol? The only intended change
from the measured harness is observability: structured validation diagnostics and
durable call accounting. Keep the system prompts, admission rules, fallback,
script, model ID, temperatures, token limits, and fixed arm order unchanged.
The previous failure cannot be reconstructed from token counts; do not fabricate
an invalid-response fixture and present it as the observed provider output.

Use `mistral-small-2603` if the authenticated catalog still lists it. If that exact
ID is unavailable, stop and revise the design explicitly; do not silently use a
`latest` alias or another model. Pin and hash all inputs before calls. Keep the
existing credential in the environment and use a new output directory.

## Budget and stopping policy

- Three conditions × two repetitions × six turns: 36 replies and 12 workpads.
- At most 48 adapter calls; no additional judge or repair calls.
- Reply limit 900 output tokens; workpad limit 600; HTTP timeout 60 seconds.
- Aggregate output ceilings sum to 39,600 tokens. Input and billed cost are not
  capped by that number. Record raw usage and missingness.
- Stop on the first transport/runtime failure. Preserve receipts and mark the run
  incomplete; never retry automatically or call a replacement model.
- An invalid workpad is an observed outcome: retain its diagnostic and continue
  with the existing no-workpad fallback. Do not relax validation within the run.
- Retain truncated replies and workpads in the all-dialogue record. Exclude their
  dialogues from complete-treatment pairs using the recorded policy.

Offline prerequisite:

```bash
python3 -m unittest discover -s tests -q
python3 hold_dialogue_pilot.py --offline --out runs/hold_diagnostic_dummy_01
```

The selected live command, when this experiment is started, is:

```bash
python3 hold_dialogue_pilot.py --provider-model mistral=mistral-small-2603 \
  --out runs/hold_diagnostic_live_01
```

Do not reuse those directories if they exist. The runner is a single-process CLI;
do not call its instrumented entrypoint concurrently inside an application server.

## Readout fixed before execution

The primary outcome is diagnostic coverage of rejected workpads: counts by
turn, extraction/schema stage, stable code and field, finish reason, and whether
the saved text is complete or a bounded excerpt. Report all 12 attempts, including
the two initial workpads, and distinguish a non-reproduced failure from a repaired
one. For a complete saved rejected response, replay that exact text through the
same parser offline, without a provider call. For a truncated diagnostic excerpt,
do not claim exact-text replay of the full response.

Report execution completion, schema validity, and response behavior separately.
For both repetitions, inspect turns 4–6 for these descriptive annotations:

1. Did the memo's next step address the current request (develop a trial, give a
   recommendation, or reopen it), or defer that work to someone else?
2. Did the reply nevertheless perform the requested action? Record evidence and
   counterevidence; a next step starting with “Ask” is not itself a failure.
3. Did an invented hour, weekday, access rule, staffing obligation, or success
   threshold become an asserted fact? Distinguish a clearly labelled proposal
   from an unsupported assertion, and quote the originating turn.

These are unblinded mechanism observations unless collected through a separately
declared blinded procedure. Keep all six comparison packets and the blank human
rubric. Report how many complete-treatment primary pairs survive, even if zero.
No after-the-fact renaming of the primary population is allowed.

## Decision after the readout

If failures recur with a specific schema/extraction reason, create an exact-text
offline regression from that newly observed response before changing code or
prompts. A syntax fix must preserve rejection of unsupported extra fields,
overlong states, and stale-state fallback. Do not broaden admission merely to
make the acceptance percentage increase.

If no failure recurs, report non-reproduction at this small sample size; the old
failure remains unresolved. In either case, this diagnostic replication cannot
establish improved dialogue quality because it introduces no intended behavioral
intervention.

Only after the diagnostic decision, design a separate behavioral comparison of
the current workpad instruction and a version that separates a concrete action
for the current reply from information requiring external confirmation. Use
multiple previously unused scripts, counterbalance order, predeclare how to
score unsupported premise additions, and retain all-dialogue outcomes alongside
valid-workpad subsets. Its model budget and final prompt should be fixed in that
later design; they are not implicitly added to this 48-call experiment.
