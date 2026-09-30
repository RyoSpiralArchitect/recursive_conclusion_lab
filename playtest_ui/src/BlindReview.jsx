import { useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  FileCheck2,
  LockKeyhole,
  Plus,
  Save,
} from "lucide-react";

import { fetchJson } from "./api";


function createEmptyDraft() {
  return {
    answers: {},
    confidence: "",
    evidence: "",
    counterevidence: "",
    abstain: false,
  };
}


function draftFromResponse(response) {
  if (!response) {
    return createEmptyDraft();
  }
  return {
    answers: response.answers || {},
    confidence: response.confidence || "",
    evidence: response.evidence || "",
    counterevidence: response.counterevidence || "",
    abstain: Boolean(response.abstain),
  };
}


function draftSignature(draft) {
  const answers = Object.fromEntries(
    Object.entries(draft?.answers || {}).sort(([left], [right]) =>
      left.localeCompare(right),
    ),
  );
  return JSON.stringify({
    answers,
    confidence: draft?.confidence || "",
    evidence: draft?.evidence || "",
    counterevidence: draft?.counterevidence || "",
    abstain: Boolean(draft?.abstain),
  });
}


function formatTimestamp(epochSeconds) {
  if (!epochSeconds) {
    return "Not sealed";
  }
  return new Date(epochSeconds * 1000).toLocaleString();
}


function shortDigest(value) {
  if (!value) {
    return "Unavailable";
  }
  return `${value.slice(0, 10)}…${value.slice(-8)}`;
}


function parseTranscript(transcript) {
  const chunks = String(transcript || "")
    .trim()
    .split(/\n\n(?=Turn\s+\d+\n)/);
  const turns = chunks.map((chunk) => {
    const match = chunk.match(/^Turn\s+(\d+)\nUser:\s*([\s\S]*?)\nAssistant:\s*([\s\S]*)$/);
    if (!match) {
      return null;
    }
    return {
      turn: Number(match[1]),
      user: match[2].trim(),
      assistant: match[3].trim(),
    };
  });
  return turns.every(Boolean) ? turns : null;
}


function TranscriptComparison({ item }) {
  const turnsA = useMemo(
    () => parseTranscript(item?.labels?.A?.transcript),
    [item],
  );
  const turnsB = useMemo(
    () => parseTranscript(item?.labels?.B?.transcript),
    [item],
  );
  const aligned =
    turnsA &&
    turnsB &&
    turnsA.length === turnsB.length &&
    turnsA.every((turn, index) => turn.user === turnsB[index].user);

  if (!aligned) {
    return (
      <div className="raw-comparison">
        {(["A", "B"]).map((label) => (
          <article className="raw-transcript" key={label}>
            <header>Transcript {label}</header>
            <pre>{item.labels[label].transcript}</pre>
          </article>
        ))}
      </div>
    );
  }

  return (
    <div className="turn-comparison">
      {turnsA.map((turnA, index) => {
        const turnB = turnsB[index];
        return (
          <section className="comparison-turn" key={`${turnA.turn}-${index}`}>
            <div className="shared-user-turn">
              <span>Turn {turnA.turn}</span>
              <p>{turnA.user}</p>
            </div>
            <div className="assistant-grid">
              <article className="assistant-answer">
                <header>Response A</header>
                <p>{turnA.assistant}</p>
              </article>
              <article className="assistant-answer">
                <header>Response B</header>
                <p>{turnB.assistant}</p>
              </article>
            </div>
          </section>
        );
      })}
    </div>
  );
}


function RubricForm({ disabled, draft, item, onChange }) {
  function selectChoice(questionId, choice) {
    onChange({
      ...draft,
      abstain: false,
      answers: { ...draft.answers, [questionId]: choice },
    });
  }

  return (
    <section className="rubric-section">
      <div className="section-heading compact-heading">
        <p className="eyebrow">Judgment</p>
        <h2>Timing and articulation</h2>
      </div>
      <div className="rubric-list">
        {item.questions.map((question) => (
          <fieldset disabled={disabled || draft.abstain} key={question.id}>
            <legend>{question.prompt}</legend>
            <div className="choice-segment" role="group" aria-label={question.prompt}>
              {question.choices.map((choice) => (
                <button
                  aria-pressed={draft.answers[question.id] === choice}
                  className={draft.answers[question.id] === choice ? "selected" : ""}
                  key={choice}
                  onClick={() => selectChoice(question.id, choice)}
                  type="button"
                >
                  {draft.answers[question.id] === choice ? <Check size={15} /> : null}
                  {choice}
                </button>
              ))}
            </div>
          </fieldset>
        ))}
      </div>

      <div className="review-fields">
        <fieldset disabled={disabled}>
          <legend>Confidence</legend>
          <div className="choice-segment" role="group" aria-label="Confidence">
            {(["low", "medium", "high"]).map((level) => (
              <button
                aria-pressed={draft.confidence === level}
                className={draft.confidence === level ? "selected" : ""}
                key={level}
                onClick={() => onChange({ ...draft, confidence: level })}
                type="button"
              >
                {level}
              </button>
            ))}
          </div>
        </fieldset>

        <label className="review-text-field">
          <span>{draft.abstain ? "Reason" : "Evidence"}</span>
          <textarea
            disabled={disabled}
            onChange={(event) => onChange({ ...draft, evidence: event.target.value })}
            rows={3}
            value={draft.evidence}
          />
        </label>

        <label className="review-text-field">
          <span>Counterevidence</span>
          <textarea
            disabled={disabled}
            onChange={(event) => onChange({ ...draft, counterevidence: event.target.value })}
            rows={2}
            value={draft.counterevidence}
          />
        </label>

        <label className="abstain-control">
          <input
            checked={draft.abstain}
            disabled={disabled}
            onChange={(event) =>
              onChange({
                ...draft,
                abstain: event.target.checked,
                answers: event.target.checked ? {} : draft.answers,
              })
            }
            type="checkbox"
          />
          <span>Abstain</span>
        </label>
      </div>
    </section>
  );
}


function BlindReview() {
  const [sets, setSets] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [activeSession, setActiveSession] = useState(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [draft, setDraft] = useState(createEmptyDraft);
  const [draftItemId, setDraftItemId] = useState("");
  const [createForm, setCreateForm] = useState({ eval_set_id: "", rater_id: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [sealing, setSealing] = useState(false);
  const [pendingSubmission, setPendingSubmission] = useState(null);
  const activeSessionIdRef = useRef("");
  const busy = loading || creating || saving || sealing;

  function selectActiveSessionId(sessionId) {
    activeSessionIdRef.current = sessionId;
    setActiveSessionId(sessionId);
  }

  async function refreshSessions() {
    const payload = await fetchJson("/api/review/sessions");
    setSessions(payload.sessions || []);
  }

  useEffect(() => {
    async function bootstrap() {
      try {
        const [setsPayload, sessionsPayload] = await Promise.all([
          fetchJson("/api/review/sets"),
          fetchJson("/api/review/sessions"),
        ]);
        const availableSets = setsPayload.sets || [];
        const savedSessions = sessionsPayload.sessions || [];
        setSets(availableSets);
        setSessions(savedSessions);
        setCreateForm((previous) => ({
          ...previous,
          eval_set_id: previous.eval_set_id || availableSets[0]?.eval_set_id || "",
        }));
        if (savedSessions.length) {
          selectActiveSessionId(savedSessions[0].session_id);
        }
      } catch (loadError) {
        setError(loadError.message);
      } finally {
        setLoading(false);
      }
    }
    bootstrap();
  }, []);

  useEffect(() => {
    if (!activeSessionId) {
      setActiveSession(null);
      return;
    }
    const requestedSessionId = activeSessionId;
    let cancelled = false;
    async function loadSession() {
      setLoading(true);
      setError("");
      setActiveSession(null);
      setActiveIndex(0);
      setDraft(createEmptyDraft());
      setDraftItemId("");
      setPendingSubmission(null);
      try {
        const payload = await fetchJson(`/api/review/sessions/${requestedSessionId}`);
        if (cancelled || activeSessionIdRef.current !== requestedSessionId) {
          return;
        }
        setActiveSession(payload);
        const nextIncomplete = payload.items.findIndex(
          (item) => !payload.responses?.[item.item_id],
        );
        setActiveIndex(nextIncomplete >= 0 ? nextIncomplete : 0);
      } catch (loadError) {
        if (!cancelled && activeSessionIdRef.current === requestedSessionId) {
          setError(loadError.message);
          setLoading(false);
          selectActiveSessionId("");
        }
      } finally {
        if (!cancelled && activeSessionIdRef.current === requestedSessionId) {
          setLoading(false);
        }
      }
    }
    loadSession();
    return () => {
      cancelled = true;
    };
  }, [activeSessionId]);

  const currentItem = activeSession?.items?.[activeIndex] || null;
  useEffect(() => {
    if (!currentItem) {
      setDraft(createEmptyDraft());
      setDraftItemId("");
      setPendingSubmission(null);
      return;
    }
    const response = activeSession?.responses?.[currentItem.item_id];
    setDraft(draftFromResponse(response));
    setDraftItemId(currentItem.item_id);
    setPendingSubmission(null);
  }, [activeSession, currentItem]);

  const savedResponse = currentItem
    ? activeSession?.responses?.[currentItem.item_id]
    : null;
  const dirtyDraft = Boolean(
    currentItem &&
      draftItemId === currentItem.item_id &&
      draftSignature(draft) !== draftSignature(draftFromResponse(savedResponse)),
  );

  function confirmDiscard(message) {
    return !dirtyDraft || window.confirm(message);
  }

  function openItem(index) {
    if (busy || !activeSession || index === activeIndex) {
      return;
    }
    if (!confirmDiscard("Discard unsaved changes for this item?")) {
      return;
    }
    const item = activeSession.items[index];
    setActiveIndex(index);
    setDraft(draftFromResponse(activeSession.responses?.[item.item_id]));
    setDraftItemId(item.item_id);
    setPendingSubmission(null);
  }

  function openSession(sessionId) {
    if (busy || sessionId === activeSessionId) {
      return;
    }
    if (!confirmDiscard("Discard unsaved changes and switch reviews?")) {
      return;
    }
    setLoading(true);
    setActiveSession(null);
    setActiveIndex(0);
    setDraft(createEmptyDraft());
    setDraftItemId("");
    setPendingSubmission(null);
    selectActiveSessionId(sessionId);
  }

  async function createReview(event) {
    event.preventDefault();
    if (busy || !confirmDiscard("Discard unsaved changes and open a new review?")) {
      return;
    }
    setCreating(true);
    setError("");
    try {
      const payload = await fetchJson("/api/review/sessions", {
        method: "POST",
        body: JSON.stringify(createForm),
      });
      setActiveSession(null);
      setActiveIndex(0);
      setDraft(createEmptyDraft());
      setDraftItemId("");
      setPendingSubmission(null);
      selectActiveSessionId(payload.session_id);
      try {
        await refreshSessions();
      } catch (refreshError) {
        if (activeSessionIdRef.current === payload.session_id) {
          setError(
            `Review opened, but the saved-review list could not refresh: ${refreshError.message}`,
          );
        }
      }
    } catch (createError) {
      setError(createError.message);
    } finally {
      setCreating(false);
    }
  }

  const completeDraft = Boolean(
    currentItem &&
      draftItemId === currentItem.item_id &&
      draft.confidence &&
      draft.evidence.trim() &&
      (draft.abstain ||
        currentItem.questions.every((question) => draft.answers[question.id])),
  );

  async function saveJudgment() {
    if (
      busy ||
      !activeSession ||
      !currentItem ||
      !completeDraft ||
      activeSession.session_id !== activeSessionIdRef.current
    ) {
      return;
    }
    const sessionId = activeSession.session_id;
    const itemId = currentItem.item_id;
    const itemIndex = activeIndex;
    const draftSnapshot = {
      ...draft,
      answers: { ...draft.answers },
    };
    setSaving(true);
    setError("");
    const signature = `${itemId}:${draftSignature(draftSnapshot)}`;
    const submission =
      pendingSubmission?.signature === signature
        ? pendingSubmission
        : { id: crypto.randomUUID(), signature };
    setPendingSubmission(submission);
    try {
      const payload = await fetchJson(
        `/api/review/sessions/${sessionId}/items/${itemId}`,
        {
          method: "PUT",
          body: JSON.stringify({
            ...draftSnapshot,
            submission_id: submission.id,
          }),
        },
      );
      if (activeSessionIdRef.current !== sessionId) {
        return;
      }
      setActiveSession(payload);
      setPendingSubmission(null);
      const nextIncomplete = payload.items.findIndex(
        (item, index) => index > itemIndex && !payload.responses?.[item.item_id],
      );
      if (nextIncomplete >= 0) {
        setActiveIndex(nextIncomplete);
      }
      try {
        await refreshSessions();
      } catch (refreshError) {
        if (activeSessionIdRef.current === sessionId) {
          setError(
            `Judgment saved, but the saved-review list could not refresh: ${refreshError.message}`,
          );
        }
      }
    } catch (saveError) {
      if (activeSessionIdRef.current === sessionId) {
        setError(saveError.message);
      }
    } finally {
      setSaving(false);
    }
  }

  async function sealReview() {
    if (
      busy ||
      dirtyDraft ||
      !activeSession ||
      activeSession.session_id !== activeSessionIdRef.current ||
      activeSession.completed_count !== activeSession.item_count
    ) {
      return;
    }
    const sessionId = activeSession.session_id;
    setSealing(true);
    setError("");
    try {
      const payload = await fetchJson(
        `/api/review/sessions/${sessionId}/seal`,
        { method: "POST", body: "{}" },
      );
      if (activeSessionIdRef.current !== sessionId) {
        return;
      }
      setActiveSession(payload);
      try {
        await refreshSessions();
      } catch (refreshError) {
        if (activeSessionIdRef.current === sessionId) {
          setError(
            `Review sealed, but the saved-review list could not refresh: ${refreshError.message}`,
          );
        }
      }
    } catch (sealError) {
      if (activeSessionIdRef.current === sessionId) {
        setError(sealError.message);
      }
    } finally {
      setSealing(false);
    }
  }

  return (
    <div className={`review-workspace ${activeSession ? "has-active" : ""}`}>
      {error ? (
        <div aria-live="assertive" className="error-banner review-error" role="alert">
          {error}
        </div>
      ) : null}

      <aside className="review-sidebar">
        <section className="review-side-section">
          <div className="section-heading compact-heading">
            <p className="eyebrow">New Review</p>
            <h2>Open packet set</h2>
          </div>
          <form className="stack" onSubmit={createReview}>
            <label className="field">
              <span>Packet set</span>
              <select
                disabled={busy || !sets.length}
                onChange={(event) =>
                  setCreateForm((previous) => ({
                    ...previous,
                    eval_set_id: event.target.value,
                  }))
                }
                value={createForm.eval_set_id}
              >
                {sets.map((reviewSet) => (
                  <option key={reviewSet.eval_set_id} value={reviewSet.eval_set_id}>
                    {reviewSet.title} ({reviewSet.item_count})
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Rater ID</span>
              <input
                disabled={busy}
                onChange={(event) =>
                  setCreateForm((previous) => ({
                    ...previous,
                    rater_id: event.target.value,
                  }))
                }
                value={createForm.rater_id}
              />
            </label>
            <button
              className="primary-button icon-text-button"
              disabled={busy || !createForm.eval_set_id || !createForm.rater_id.trim()}
              type="submit"
            >
              <Plus size={17} />
              {creating ? "Opening…" : "Open Review"}
            </button>
          </form>
          {!sets.length && !loading ? (
            <p className="empty-note">No packet sets found.</p>
          ) : null}
        </section>

        <section className="review-side-section saved-review-section">
          <div className="section-heading compact-heading">
            <p className="eyebrow">Saved Reviews</p>
            <h2>Resume</h2>
          </div>
          <div className="review-session-list">
            {sessions.map((session) => (
              <button
                aria-current={session.session_id === activeSessionId ? "page" : undefined}
                aria-pressed={session.session_id === activeSessionId}
                className={session.session_id === activeSessionId ? "active" : ""}
                disabled={busy}
                key={session.session_id}
                onClick={() => openSession(session.session_id)}
                type="button"
              >
                <strong>{session.title}</strong>
                <span>
                  {session.completed_count}/{session.item_count}
                  {session.sealed_at ? " · sealed" : ""}
                </span>
              </button>
            ))}
          </div>
        </section>
      </aside>

      <main aria-busy={busy} className="review-main">
        {activeSession && currentItem ? (
          <>
            <header className="review-head">
              <div>
                <p className="eyebrow">{currentItem.scenario_display_name}</p>
                <h1>{activeSession.title}</h1>
              </div>
              <div aria-label="Review item navigation" className="review-navigation" role="group">
                <button
                  aria-label="Previous item"
                  disabled={busy || activeIndex === 0}
                  onClick={() => openItem(Math.max(0, activeIndex - 1))}
                  title="Previous item"
                  type="button"
                >
                  <ChevronLeft size={18} />
                </button>
                <strong>
                  {activeIndex + 1} / {activeSession.item_count}
                </strong>
                <button
                  aria-label="Next item"
                  disabled={busy || activeIndex >= activeSession.item_count - 1}
                  onClick={() =>
                    openItem(Math.min(activeSession.item_count - 1, activeIndex + 1))
                  }
                  title="Next item"
                  type="button"
                >
                  <ChevronRight size={18} />
                </button>
              </div>
            </header>

            <nav className="item-progress" aria-label="Review item progress">
              {activeSession.items.map((item, index) => (
                <button
                  aria-current={index === activeIndex ? "step" : undefined}
                  aria-label={`Open item ${index + 1}, ${
                    activeSession.responses?.[item.item_id] ? "completed" : "incomplete"
                  }`}
                  className={`${index === activeIndex ? "active" : ""} ${
                    activeSession.responses?.[item.item_id] ? "complete" : ""
                  }`}
                  disabled={busy}
                  key={item.item_id}
                  onClick={() => openItem(index)}
                  title={`Item ${index + 1}`}
                  type="button"
                >
                  {activeSession.responses?.[item.item_id] ? <Check size={14} /> : index + 1}
                </button>
              ))}
            </nav>

            <TranscriptComparison item={currentItem} />
            <RubricForm
              disabled={busy || Boolean(activeSession.sealed_at)}
              draft={draft}
              item={currentItem}
              onChange={setDraft}
            />

            <footer className="review-actions">
              <div aria-live="polite" className="review-save-state" role="status">
                {dirtyDraft ? (
                  "Unsaved changes"
                ) : activeSession.responses?.[currentItem.item_id] ? (
                  <>
                    <CheckCircle2 size={17} />
                    Revision {activeSession.responses[currentItem.item_id].revision}
                  </>
                ) : (
                  "Unsaved"
                )}
              </div>
              <button
                className="primary-button icon-text-button"
                disabled={busy || !completeDraft || Boolean(activeSession.sealed_at)}
                onClick={saveJudgment}
                type="button"
              >
                <Save size={17} />
                {saving ? "Saving…" : "Save Judgment"}
              </button>
            </footer>
          </>
        ) : (
          <div aria-live="polite" className="review-empty" role="status">
            <FileCheck2 size={28} />
            <p>{loading ? "Loading reviews…" : "No review selected."}</p>
          </div>
        )}
      </main>

      <aside className="review-ledger">
        <section>
          <div className="section-heading compact-heading">
            <p className="eyebrow">Progress</p>
            <h2>Review ledger</h2>
          </div>
          <dl className="review-ledger-list">
            <div>
              <dt>Completed</dt>
              <dd aria-live="polite">
                {activeSession?.completed_count || 0} / {activeSession?.item_count || 0}
              </dd>
            </div>
            <div>
              <dt>Rater</dt>
              <dd>{activeSession?.rater_id || "—"}</dd>
            </div>
            <div>
              <dt>Packet</dt>
              <dd title={activeSession?.review_bundle_digest}>
                {shortDigest(activeSession?.review_bundle_digest)}
              </dd>
            </div>
            <div>
              <dt>Rubric</dt>
              <dd title={activeSession?.rubric_digest}>
                {shortDigest(activeSession?.rubric_digest)}
              </dd>
            </div>
            <div>
              <dt>Sealed</dt>
              <dd>{formatTimestamp(activeSession?.sealed_at)}</dd>
            </div>
          </dl>
        </section>

        <section className="seal-section">
          {activeSession?.sealed_at ? (
            <div aria-live="polite" className="sealed-state" role="status">
              <LockKeyhole size={20} />
              <strong>Sealed</strong>
              <span title={activeSession.seal_receipt?.receipt_digest}>
                {shortDigest(activeSession.seal_receipt?.receipt_digest)}
              </span>
            </div>
          ) : (
            <button
              className="seal-button icon-text-button"
              disabled={
                busy ||
                dirtyDraft ||
                !activeSession ||
                activeSession.completed_count !== activeSession.item_count
              }
              onClick={sealReview}
              type="button"
            >
              <LockKeyhole size={17} />
              {sealing ? "Sealing…" : "Seal Review"}
            </button>
          )}
        </section>
      </aside>
    </div>
  );
}


export default BlindReview;
