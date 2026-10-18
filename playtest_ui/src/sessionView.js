export function emptySessionView() {
  return {
    detail: null,
    composer: "",
    composerRevision: 0,
    composerInitialized: false,
    notesDraft: "",
    notesSaved: "",
    notesInitialized: false,
    notesSaving: false,
    notesError: "",
    turnRequest: null,
    error: "",
  };
}

export function withSessionDetail(view, detail) {
  const current = view.detail;
  const older = current && (
    (detail.turn_index ?? 0) < (current.turn_index ?? 0) ||
    ((detail.turn_index ?? 0) === (current.turn_index ?? 0) &&
      (detail.updated_at ?? 0) < (current.updated_at ?? 0))
  );
  if (older) {
    return view;
  }
  const notesClean = !view.notesSaving &&
    (!view.notesInitialized || view.notesDraft === view.notesSaved);
  return {
    ...view,
    detail,
    composer: view.composerInitialized
      ? view.composer
      : detail.pending_user_text || "",
    composerInitialized: true,
    notesDraft: notesClean ? detail.notes || "" : view.notesDraft,
    notesSaved: notesClean ? detail.notes || "" : view.notesSaved,
    notesInitialized: true,
  };
}

export function mergeSessionSummary(summary, detail) {
  if ((detail.turn_index ?? 0) < (summary.turn_index ?? 0) ||
    ((detail.turn_index ?? 0) === (summary.turn_index ?? 0) &&
      (detail.updated_at ?? 0) < (summary.updated_at ?? 0))) {
    return summary;
  }
  return {
    ...summary,
    turn_index: detail.turn_index,
    updated_at: detail.updated_at,
    pending_user_text: detail.pending_user_text,
  };
}

export function withCommittedTurn(view, request) {
  return {
    ...view,
    composer: view.composerRevision === request.composerRevision &&
      view.composer === request.userText ? "" : view.composer,
    turnRequest: null,
    error: "",
  };
}
