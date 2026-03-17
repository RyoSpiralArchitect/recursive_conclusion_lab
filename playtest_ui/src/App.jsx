import { startTransition, useDeferredValue, useEffect, useState } from "react";

const emptyCreateForm = {
  title: "",
  script_id: "shortlist_then_commit",
  arm_preset: "adaptive_kind_aware",
  provider: "openai",
  model: "gpt-4.1-mini-2025-04-14",
  semantic_judge_backend: "both",
};

async function fetchJson(url, options) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        detail = String(payload.detail);
      }
    } catch {}
    throw new Error(detail);
  }
  return response.json();
}

function formatNumber(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  return value.toFixed(3);
}

function formatTimestamp(epochSeconds) {
  if (!epochSeconds) {
    return "—";
  }
  return new Date(epochSeconds * 1000).toLocaleString();
}

function summaryMetric(label, value, tone = "neutral") {
  return { label, value, tone };
}

function collectMetrics(session) {
  const result = session?.last_result || {};
  return [
    summaryMetric("Turn", session?.turn_index ?? 0, "neutral"),
    summaryMetric("Overlap", formatNumber(result.probe_reply_overlap), "neutral"),
    summaryMetric(
      "Latent Align",
      formatNumber(result.latent_convergence_alignment),
      "warm",
    ),
    summaryMetric(
      "Embedding Align",
      formatNumber(result.embedding_convergence_alignment),
      "cool",
    ),
    summaryMetric(
      "Judge Gap",
      formatNumber(result.semantic_judge_alignment_gap),
      "neutral",
    ),
    summaryMetric(
      "Suppressed",
      Array.isArray(result.suppressed_delayed_mentions)
        ? result.suppressed_delayed_mentions.length
        : 0,
      "neutral",
    ),
  ];
}

function App() {
  const [options, setOptions] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [activeSession, setActiveSession] = useState(null);
  const [createForm, setCreateForm] = useState(emptyCreateForm);
  const [composer, setComposer] = useState("");
  const [notesDraft, setNotesDraft] = useState("");
  const deferredNotes = useDeferredValue(notesDraft);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("Loading…");
  const [creating, setCreating] = useState(false);
  const [sending, setSending] = useState(false);
  const [loadingSession, setLoadingSession] = useState(false);
  const [savingNotes, setSavingNotes] = useState(false);

  useEffect(() => {
    async function bootstrap() {
      try {
        const [optionsPayload, sessionsPayload] = await Promise.all([
          fetchJson("/api/options"),
          fetchJson("/api/sessions"),
        ]);
        startTransition(() => {
          setOptions(optionsPayload);
          setSessions(sessionsPayload.sessions || []);
          if ((sessionsPayload.sessions || []).length > 0) {
            setActiveSessionId(sessionsPayload.sessions[0].session_id);
          }
        });
        setStatus("Ready");
      } catch (loadError) {
        setError(loadError.message);
        setStatus("Failed to load");
      }
    }
    bootstrap();
  }, []);

  useEffect(() => {
    if (!activeSessionId) {
      setActiveSession(null);
      return;
    }
    let cancelled = false;
    async function loadSession() {
      setLoadingSession(true);
      setError("");
      try {
        const payload = await fetchJson(`/api/sessions/${activeSessionId}`);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setActiveSession(payload);
          setNotesDraft(payload.notes || "");
          if (payload.pending_user_text && !composer) {
            setComposer(payload.pending_user_text);
          }
        });
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError.message);
        }
      } finally {
        if (!cancelled) {
          setLoadingSession(false);
        }
      }
    }
    loadSession();
    return () => {
      cancelled = true;
    };
  }, [activeSessionId]);

  useEffect(() => {
    if (!activeSession?.session_id) {
      return;
    }
    if (deferredNotes === (activeSession.notes || "")) {
      return;
    }
    const timer = window.setTimeout(async () => {
      try {
        setSavingNotes(true);
        const payload = await fetchJson(
          `/api/sessions/${activeSession.session_id}/notes`,
          {
            method: "PUT",
            body: JSON.stringify({ notes: deferredNotes }),
          },
        );
        startTransition(() => {
          setActiveSession(payload);
          setSessions((previous) =>
            previous.map((session) =>
              session.session_id === payload.session_id
                ? { ...session, updated_at: payload.updated_at }
                : session,
            ),
          );
        });
      } catch (saveError) {
        setError(saveError.message);
      } finally {
        setSavingNotes(false);
      }
    }, 500);
    return () => window.clearTimeout(timer);
  }, [deferredNotes, activeSession?.session_id, activeSession?.notes]);

  async function refreshSessions() {
    const payload = await fetchJson("/api/sessions");
    startTransition(() => {
      setSessions(payload.sessions || []);
    });
  }

  async function handleCreateSession(event) {
    event.preventDefault();
    setCreating(true);
    setError("");
    try {
      const payload = await fetchJson("/api/sessions", {
        method: "POST",
        body: JSON.stringify(createForm),
      });
      startTransition(() => {
        setActiveSession(payload);
        setActiveSessionId(payload.session_id);
        setNotesDraft(payload.notes || "");
        setComposer("");
      });
      await refreshSessions();
    } catch (createError) {
      setError(createError.message);
    } finally {
      setCreating(false);
    }
  }

  async function handleSendTurn() {
    if (!activeSession?.session_id || !composer.trim()) {
      return;
    }
    setSending(true);
    setError("");
    const userText = composer;
    setComposer("");
    try {
      const payload = await fetchJson(
        `/api/sessions/${activeSession.session_id}/turn`,
        {
          method: "POST",
          body: JSON.stringify({ user_text: userText }),
        },
      );
      startTransition(() => {
        setActiveSession(payload);
        setNotesDraft(payload.notes || "");
      });
      await refreshSessions();
    } catch (turnError) {
      setComposer(userText);
      setError(turnError.message);
    } finally {
      setSending(false);
    }
  }

  function applySeedTurn(text) {
    setComposer(text);
  }

  function onComposerKeyDown(event) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      handleSendTurn();
    }
  }

  const metrics = collectMetrics(activeSession);
  const scriptTurns = activeSession?.script?.turns || [];
  const lastResult = activeSession?.last_result || {};
  const delayedCollections = [
    {
      label: "Planned",
      items: lastResult.planned_delayed_mentions || [],
    },
    {
      label: "Due",
      items: lastResult.due_delayed_mentions || [],
    },
    {
      label: "Injected",
      items: lastResult.injected_delayed_mentions || [],
    },
    {
      label: "Suppressed",
      items: lastResult.suppressed_delayed_mentions || [],
    },
  ];

  return (
    <div className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Recursive Conclusion Lab</p>
          <h1>Playtest Console</h1>
        </div>
        <div className="topbar-status">
          <span className="status-chip">{status}</span>
          {loadingSession ? <span className="status-chip muted">Loading session…</span> : null}
          {savingNotes ? <span className="status-chip muted">Saving notes…</span> : null}
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="layout">
        <aside className="panel sidebar">
          <section className="panel-section">
            <div className="section-heading">
              <p className="eyebrow">New Session</p>
              <h2>Spin up a playtest</h2>
            </div>
            <form className="stack" onSubmit={handleCreateSession}>
              <label className="field">
                <span>Title</span>
                <input
                  value={createForm.title}
                  onChange={(event) =>
                    setCreateForm((previous) => ({
                      ...previous,
                      title: event.target.value,
                    }))
                  }
                  placeholder="Optional custom title"
                />
              </label>
              <label className="field">
                <span>Scenario</span>
                <select
                  value={createForm.script_id}
                  onChange={(event) =>
                    setCreateForm((previous) => ({
                      ...previous,
                      script_id: event.target.value,
                    }))
                  }
                >
                  {(options?.scripts || []).map((script) => (
                    <option key={script.id} value={script.id}>
                      {script.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Arm</span>
                <select
                  value={createForm.arm_preset}
                  onChange={(event) =>
                    setCreateForm((previous) => ({
                      ...previous,
                      arm_preset: event.target.value,
                    }))
                  }
                >
                  {(options?.arm_presets || []).map((arm) => (
                    <option key={arm.id} value={arm.id}>
                      {arm.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Provider</span>
                <select
                  value={createForm.provider}
                  onChange={(event) =>
                    setCreateForm((previous) => ({
                      ...previous,
                      provider: event.target.value,
                    }))
                  }
                >
                  {(options?.providers || []).map((provider) => (
                    <option key={provider} value={provider}>
                      {provider}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Model</span>
                <input
                  value={createForm.model}
                  onChange={(event) =>
                    setCreateForm((previous) => ({
                      ...previous,
                      model: event.target.value,
                    }))
                  }
                />
              </label>
              <button className="primary-button" disabled={creating} type="submit">
                {creating ? "Creating…" : "Create Session"}
              </button>
            </form>
          </section>

          <section className="panel-section">
            <div className="section-heading">
              <p className="eyebrow">Saved Sessions</p>
              <h2>Resume</h2>
            </div>
            <div className="session-list">
              {sessions.map((session) => (
                <button
                  key={session.session_id}
                  className={`session-card ${
                    session.session_id === activeSessionId ? "active" : ""
                  }`}
                  onClick={() => setActiveSessionId(session.session_id)}
                  type="button"
                >
                  <strong>{session.title}</strong>
                  <span>
                    {session.script_label || session.script_id} ·{" "}
                    {session.arm_label || session.arm_preset}
                  </span>
                  <span>
                    turn {session.turn_index} · {formatTimestamp(session.updated_at)}
                  </span>
                  {session.pending_user_text ? (
                    <span className="warning-inline">Recovered pending draft</span>
                  ) : null}
                </button>
              ))}
              {sessions.length === 0 ? <p className="muted-copy">No saved sessions yet.</p> : null}
            </div>
          </section>
        </aside>

        <main className="panel transcript-panel">
          <section className="panel-section transcript-head">
            <div className="section-heading">
              <p className="eyebrow">Conversation</p>
              <h2>{activeSession?.title || "No session selected"}</h2>
            </div>
            {activeSession ? (
              <div className="chip-row">
                <span className="info-chip">{activeSession.script?.label}</span>
                <span className="info-chip">{activeSession.arm_label}</span>
                <span className="info-chip">
                  {activeSession.provider}/{activeSession.model}
                </span>
              </div>
            ) : null}
          </section>

          {activeSession?.pending_user_text ? (
            <div className="recovery-banner">
              A previous request was saved before completion. The draft is back in the composer.
            </div>
          ) : null}

          {scriptTurns.length > 0 ? (
            <section className="panel-section seed-panel">
              <div className="section-heading tight">
                <p className="eyebrow">Seed Turns</p>
                <h3>Use scripted turns as probes</h3>
              </div>
              <div className="seed-grid">
                {scriptTurns.map((turn, index) => (
                  <button
                    key={`${index + 1}`}
                    className="seed-card"
                    onClick={() => applySeedTurn(turn)}
                    type="button"
                  >
                    <span className="seed-index">Turn {index + 1}</span>
                    <span>{turn}</span>
                  </button>
                ))}
              </div>
            </section>
          ) : null}

          <section className="chat-scroll">
            {(activeSession?.history || []).map((message, index) => (
              <article
                key={`${message.role}-${index}`}
                className={`message-card ${message.role}`}
              >
                <div className="message-meta">
                  <span>{message.role === "user" ? "User" : "Assistant"}</span>
                  {message.role === "assistant" && index === (activeSession.history.length - 1) ? (
                    <span className="message-badge">Latest</span>
                  ) : null}
                </div>
                <p>{message.content}</p>
              </article>
            ))}
            {!activeSession?.history?.length ? (
              <div className="empty-state">
                Create a session, then type freely or click one of the scripted seed turns.
              </div>
            ) : null}
          </section>

          <section className="composer">
            <textarea
              value={composer}
              onChange={(event) => setComposer(event.target.value)}
              onKeyDown={onComposerKeyDown}
              placeholder="Type a user turn. Cmd/Ctrl+Enter sends."
              rows={7}
            />
            <div className="composer-actions">
              <span className="muted-copy">
                {composer.length} chars · saved per turn on the backend
              </span>
              <button
                className="primary-button"
                disabled={sending || !activeSession?.session_id || !composer.trim()}
                onClick={handleSendTurn}
                type="button"
              >
                {sending ? "Sending…" : "Send Turn"}
              </button>
            </div>
          </section>
        </main>

        <aside className="panel inspector">
          <section className="panel-section">
            <div className="section-heading">
              <p className="eyebrow">Live Trace</p>
              <h2>Human-side reading aid</h2>
            </div>
            <div className="metric-grid">
              {metrics.map((metric) => (
                <div key={metric.label} className={`metric-card ${metric.tone}`}>
                  <span>{metric.label}</span>
                  <strong>{metric.value}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="panel-section">
            <div className="section-heading tight">
              <p className="eyebrow">Conclusion</p>
              <h3>Current plan</h3>
            </div>
            <div className="inspector-card">
              <p className="highlight-line">
                {lastResult.latest_conclusion_line || "No conclusion probe yet."}
              </p>
              <dl className="kv-list">
                <div>
                  <dt>Window</dt>
                  <dd>
                    {lastResult.latest_conclusion_plan_earliest_turn ?? "—"} →{" "}
                    {lastResult.latest_conclusion_plan_latest_turn ?? "—"}
                  </dd>
                </div>
                <div>
                  <dt>Hazard</dt>
                  <dd>
                    {formatNumber(lastResult.latest_conclusion_plan_hazard_turn_prob)} /{" "}
                    {formatNumber(
                      lastResult.latest_conclusion_plan_adaptive_hazard_turn_prob,
                    )}
                  </dd>
                </div>
                <div>
                  <dt>Stage</dt>
                  <dd>{lastResult.latent_convergence_stage || "—"}</dd>
                </div>
              </dl>
            </div>
          </section>

          <section className="panel-section">
            <div className="section-heading tight">
              <p className="eyebrow">Delayed Mentions</p>
              <h3>Release pressure</h3>
            </div>
            {delayedCollections.map((group) => (
              <div key={group.label} className="inspector-group">
                <h4>{group.label}</h4>
                {group.items.length > 0 ? (
                  <div className="item-list">
                    {group.items.map((item) => (
                      <div key={item.item_id} className="item-card">
                        <div className="item-meta">
                          <span>{item.release_stage_role || item.kind}</span>
                          <span>
                            {item.earliest_turn ?? "—"} → {item.latest_turn ?? "—"}
                          </span>
                        </div>
                        <p>{item.text}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="muted-copy">None.</p>
                )}
              </div>
            ))}
          </section>

          <section className="panel-section">
            <div className="section-heading tight">
              <p className="eyebrow">Observations</p>
              <h3>Human notes</h3>
            </div>
            <textarea
              className="notes-box"
              value={notesDraft}
              onChange={(event) => setNotesDraft(event.target.value)}
              placeholder="Write qualitative notes: awkward timing, unnatural shortlist, good earned ending, etc."
              rows={12}
            />
          </section>
        </aside>
      </div>
    </div>
  );
}

export default App;
