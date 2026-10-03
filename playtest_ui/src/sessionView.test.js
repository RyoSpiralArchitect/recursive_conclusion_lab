import assert from "node:assert/strict";
import test from "node:test";

import { emptySessionView, mergeSessionSummary, withCommittedTurn, withSessionDetail } from "./sessionView.js";

function detail(sessionId, turnIndex, updatedAt, notes = "", pendingText = "") {
  return {
    session_id: sessionId,
    turn_index: turnIndex,
    updated_at: updatedAt,
    notes,
    pending_user_text: pendingText,
  };
}

test("session drafts and notes stay with their own session", () => {
  const a = withSessionDetail(emptySessionView(), detail("a", 0, 1, "A note"));
  const b = withSessionDetail(emptySessionView(), detail("b", 0, 1, "B note"));
  const editedA = { ...a, composer: "A draft", notesDraft: "A changed" };
  const refreshedB = withSessionDetail(b, detail("b", 1, 2, "B newer"));
  assert.equal(editedA.composer, "A draft");
  assert.equal(editedA.notesDraft, "A changed");
  assert.equal(refreshedB.composer, "");
  assert.equal(refreshedB.notesDraft, "B newer");
});

test("late older session response does not overwrite a newer turn", () => {
  const current = withSessionDetail(emptySessionView(), detail("a", 2, 5, "latest"));
  assert.equal(withSessionDetail(current, detail("a", 1, 9, "older")), current);
  assert.equal(withSessionDetail(current, detail("a", 2, 4, "older")), current);
  const summary = { session_id: "a", turn_index: 2, updated_at: 5 };
  assert.equal(mergeSessionSummary(summary, detail("a", 1, 9, "older")), summary);
});

test("a dirty notes draft survives a server response", () => {
  const loaded = withSessionDetail(emptySessionView(), detail("a", 0, 1, "saved"));
  const dirty = { ...loaded, notesDraft: "typing", composer: "new draft" };
  const next = withSessionDetail(dirty, detail("a", 1, 2, "saved"));
  assert.equal(next.notesDraft, "typing");
  assert.equal(next.notesSaved, "saved");
  assert.equal(next.composer, "new draft");
});

test("a reverted edit survives an earlier notes save response", () => {
  const loaded = withSessionDetail(emptySessionView(), detail("a", 0, 1, ""));
  const savingA = { ...loaded, notesDraft: "", notesSaving: true };
  const response = withSessionDetail(savingA, detail("a", 0, 2, "A"));
  assert.equal(response.notesDraft, "");
  assert.equal(response.notesSaved, "");
  const markedSaved = { ...response, notesSaved: "A", notesSaving: false };
  assert.notEqual(markedSaved.notesDraft, markedSaved.notesSaved);
});

test("recovered text seeds only a previously untouched composer", () => {
  const recovered = withSessionDetail(emptySessionView(), detail("a", 0, 1, "", "unfinished"));
  assert.equal(recovered.composer, "unfinished");
  const erased = { ...recovered, composer: "" };
  assert.equal(withSessionDetail(erased, detail("a", 0, 2, "", "unfinished")).composer, "");
});

test("commit clears only the unedited submitted composer", () => {
  const request = { userText: "sent", composerRevision: 3 };
  const untouched = { ...emptySessionView(), composer: "sent", composerRevision: 3 };
  assert.equal(withCommittedTurn(untouched, request).composer, "");
  const edited = { ...untouched, composerRevision: 4 };
  assert.equal(withCommittedTurn(edited, request).composer, "sent");
  const newDraft = { ...untouched, composer: "next", composerRevision: 4 };
  assert.equal(withCommittedTurn(newDraft, request).composer, "next");
});
