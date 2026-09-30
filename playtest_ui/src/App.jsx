import { startTransition, useDeferredValue, useEffect, useState } from "react";
import { FileCheck2, MessagesSquare } from "lucide-react";

import { fetchJson } from "./api";
import BlindReview from "./BlindReview";
const modelControlDefaults = {
  reasoning_effort: "auto",
  probe_reasoning_effort: "auto",
  reasoning_mode: "auto",
  probe_reasoning_mode: "auto",
  reasoning_context: "auto",
  probe_reasoning_context: "auto",
  text_verbosity: "auto",
  probe_text_verbosity: "auto",
};
const emptyCreateForm = {
  title: "",
  script_id: "",
  arm_preset: "",
  provider: "",
  model: "",
  semantic_judge_backend: "",
  ...modelControlDefaults,
};

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

function formatRestoreWarning(loadErrors, totalCount) {
  if (!Array.isArray(loadErrors) || loadErrors.length === 0) {
    return "";
  }
  const visibleErrors = loadErrors.slice(0, 3);
  const count = Math.max(Number(totalCount) || 0, loadErrors.length);
  const details = visibleErrors
    .map(
      (item) =>
        `${item.session_id || "unknown"}: ${item.message || "restore failed"}`,
    )
    .join(" · ");
  const remaining = count - visibleErrors.length;
  const suffix = remaining > 0 ? ` · ${remaining} more` : "";
  return `Saved session restore failed (${count}). ${details}${suffix}`;
}

function App() {
  const [workspace, setWorkspace] = useState("");
  const [serverMode, setServerMode] = useState("");
  const [options, setOptions] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [activeSession, setActiveSession] = useState(null);
  const [createForm, setCreateForm] = useState(emptyCreateForm);
  const [composer, setComposer] = useState("");
  const [notesDraft, setNotesDraft] = useState("");
  const deferredNotes = useDeferredValue(notesDraft);
  const [error, setError] = useState("");
  const [restoreWarning, setRestoreWarning] = useState("");
  const [status, setStatus] = useState("Loading…");
  const [creating, setCreating] = useState(false);
  const [sending, setSending] = useState(false);
  const [loadingSession, setLoadingSession] = useState(false);
  const [savingNotes, setSavingNotes] = useState(false);

  useEffect(() => {
    async function bootstrap() {
      try {
        const health = await fetchJson("/api/health");
        const mode = health.workspace_mode === "review" ? "review" : "full";
        setServerMode(mode);
        if (mode === "review") {
          setWorkspace("review");
          setStatus("Ready");
          return;
        }
        const [optionsPayload, sessionsPayload] = await Promise.all([
          fetchJson("/api/options"),
          fetchJson("/api/sessions"),
        ]);
        startTransition(() => {
          setWorkspace("playtest");
          setOptions(optionsPayload);
          setSessions(sessionsPayload.sessions || []);
          setRestoreWarning(
            formatRestoreWarning(
              sessionsPayload.load_errors,
              sessionsPayload.load_error_count,
            ),
          );
          setCreateForm((previous) => ({
            ...previous,
            script_id: previous.script_id || optionsPayload.default_script_id,
            arm_preset: previous.arm_preset || optionsPayload.default_arm_preset,
            provider: previous.provider || optionsPayload.default_provider,
            model: previous.model || optionsPayload.default_model,
            semantic_judge_backend:
              previous.semantic_judge_backend ||
              optionsPayload.default_semantic_judge_backend,
          }));
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
      setRestoreWarning(
        formatRestoreWarning(payload.load_errors, payload.load_error_count),
      );
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
  const holdingConclusion = activeSession?.config?.conclusion_mode === "hold";
  const inquiry = lastResult.inquiry_state;
  const providerModelProfiles = (options?.model_profiles || []).filter(
    (profile) => profile.provider === createForm.provider,
  );
  const suggestedModels = providerModelProfiles.flatMap(
    (profile) => profile.models || [],
  );
  const selectedModelProfile = providerModelProfiles.find((profile) =>
    (profile.models || []).includes(createForm.model),
  );
  const selectedCapabilities = selectedModelProfile?.capabilities || {};
  const selectedDefaults = selectedModelProfile?.defaults || {};

  function renderSettingOptions(values, defaultValue) {
    return (
      <>
        <option value="auto">
          Default{defaultValue ? `: ${defaultValue}` : ""}
        </option>
        {(values || []).map((value) => (
          <option key={value} value={value}>
            {value}
          </option>
        ))}
      </>
    );
  }
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
          <h1>
            {workspace === "playtest"
              ? "Playtest Console"
              : workspace === "review"
                ? "Blind Review"
                : "Loading"}
          </h1>
        </div>
        <div className="topbar-tools">
          {serverMode === "full" ? (
            <div className="workspace-switch" role="tablist" aria-label="Workspace">
              <button
                aria-selected={workspace === "playtest"}
                className={workspace === "playtest" ? "active" : ""}
                onClick={() => setWorkspace("playtest")}
                role="tab"
                type="button"
              >
                <MessagesSquare size={16} />
                Playtest
              </button>
              <button
                aria-selected={workspace === "review"}
                className={workspace === "review" ? "active" : ""}
                onClick={() => setWorkspace("review")}
                role="tab"
                type="button"
              >
                <FileCheck2 size={16} />
                Blind Review
              </button>
            </div>
          ) : null}
          {workspace === "playtest" ? (
            <div className="topbar-status">
              <span className="status-chip">{status}</span>
              {loadingSession ? <span className="status-chip muted">Loading session…</span> : null}
              {savingNotes ? <span className="status-chip muted">Saving notes…</span> : null}
            </div>
          ) : null}
        </div>
      </header>

      {workspace === "playtest" && restoreWarning ? (
        <div className="error-banner" role="alert">
          {restoreWarning}
        </div>
      ) : null}
      {workspace === "playtest" && error ? (
        <div className="error-banner" role="alert">
          {error}
        </div>
      ) : null}

      {!workspace ? null : workspace === "review" ? (
        <BlindReview />
      ) : (
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
                      model:
                        event.target.value === options?.default_provider
                          ? options?.default_model || ""
                          : "",
                      ...modelControlDefaults,
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
                  list="model-suggestions"
                  required
                  value={createForm.model}
                  onChange={(event) =>
                    setCreateForm((previous) => ({
                      ...previous,
                      model: event.target.value,
                      ...modelControlDefaults,
                    }))
                  }
                />
                <datalist id="model-suggestions">
                  {suggestedModels.map((model) => (
                    <option key={model} value={model} />
                  ))}
                </datalist>
              </label>
              {selectedModelProfile ? (
                <details className="model-controls">
                  <summary>Generation controls</summary>
                  <div className="model-control-grid">
                    <label className="field">
                      <span>Reply effort</span>
                      <select
                        value={createForm.reasoning_effort}
                        onChange={(event) =>
                          setCreateForm((previous) => ({
                            ...previous,
                            reasoning_effort: event.target.value,
                          }))
                        }
                      >
                        {renderSettingOptions(
                          selectedCapabilities.reasoning_efforts,
                          selectedDefaults.reasoning_effort,
                        )}
                      </select>
                    </label>
                    <label className="field">
                      <span>Probe effort</span>
                      <select
                        value={createForm.probe_reasoning_effort}
                        onChange={(event) =>
                          setCreateForm((previous) => ({
                            ...previous,
                            probe_reasoning_effort: event.target.value,
                          }))
                        }
                      >
                        {renderSettingOptions(
                          selectedCapabilities.reasoning_efforts,
                          selectedDefaults.reasoning_effort,
                        )}
                      </select>
                    </label>
                    <label className="field">
                      <span>Reply mode</span>
                      <select
                        value={createForm.reasoning_mode}
                        onChange={(event) =>
                          setCreateForm((previous) => ({
                            ...previous,
                            reasoning_mode: event.target.value,
                          }))
                        }
                      >
                        {renderSettingOptions(
                          selectedCapabilities.reasoning_modes,
                          selectedDefaults.reasoning_mode,
                        )}
                      </select>
                    </label>
                    <label className="field">
                      <span>Probe mode</span>
                      <select
                        value={createForm.probe_reasoning_mode}
                        onChange={(event) =>
                          setCreateForm((previous) => ({
                            ...previous,
                            probe_reasoning_mode: event.target.value,
                          }))
                        }
                      >
                        {renderSettingOptions(
                          selectedCapabilities.reasoning_modes,
                          selectedDefaults.reasoning_mode,
                        )}
                      </select>
                    </label>
                    <label className="field">
                      <span>Reply context</span>
                      <select
                        value={createForm.reasoning_context}
                        onChange={(event) =>
                          setCreateForm((previous) => ({
                            ...previous,
                            reasoning_context: event.target.value,
                          }))
                        }
                      >
                        {renderSettingOptions(
                          selectedCapabilities.reasoning_contexts,
                          selectedDefaults.reasoning_context,
                        )}
                      </select>
                    </label>
                    <label className="field">
                      <span>Probe context</span>
                      <select
                        value={createForm.probe_reasoning_context}
                        onChange={(event) =>
                          setCreateForm((previous) => ({
                            ...previous,
                            probe_reasoning_context: event.target.value,
                          }))
                        }
                      >
                        {renderSettingOptions(
                          selectedCapabilities.reasoning_contexts,
                          selectedDefaults.reasoning_context,
                        )}
                      </select>
                    </label>
                    <label className="field">
                      <span>Reply verbosity</span>
                      <select
                        value={createForm.text_verbosity}
                        onChange={(event) =>
                          setCreateForm((previous) => ({
                            ...previous,
                            text_verbosity: event.target.value,
                          }))
                        }
                      >
                        {renderSettingOptions(
                          selectedCapabilities.text_verbosity_levels,
                          selectedDefaults.text_verbosity,
                        )}
                      </select>
                    </label>
                    <label className="field">
                      <span>Probe verbosity</span>
                      <select
                        value={createForm.probe_text_verbosity}
                        onChange={(event) =>
                          setCreateForm((previous) => ({
                            ...previous,
                            probe_text_verbosity: event.target.value,
                          }))
                        }
                      >
                        {renderSettingOptions(
                          selectedCapabilities.text_verbosity_levels,
                          selectedDefaults.text_verbosity,
                        )}
                      </select>
                    </label>
                  </div>
                </details>
              ) : null}
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

          {holdingConclusion ? (
            <section className="panel-section">
              <div className="section-heading tight">
                <p className="eyebrow">Open inquiry</p>
                <h3>Keep exploring</h3>
              </div>
              <div className="inspector-card">
                <p className="highlight-line">
                  {inquiry?.focus || "Explore the question without choosing a conclusion yet."}
                </p>
                {inquiry ? (
                  <>
                    <p className="muted">
                      These notes are provisional and can change with the conversation.
                    </p>
                    <h4>Possibilities</h4>
                    <ul>
                      {inquiry.hypotheses.map((text, index) => <li key={index}>{text}</li>)}
                    </ul>
                    {inquiry.hypotheses.length === 0 && <p>No specific hypothesis yet.</p>}
                    <h4>Open questions</h4>
                    <ul>
                      {inquiry.open_questions.map((text, index) => <li key={index}>{text}</li>)}
                    </ul>
                    <h4>Next useful step</h4>
                    <p>{inquiry.next_step}</p>
                    {inquiry.revision_note && <p>{inquiry.revision_note}</p>}
                    <p className="muted">
                      Updated on turn {lastResult.inquiry_state_turn}. Ask for a synthesis whenever you want one.
                    </p>
                  </>
                ) : (
                  <p>No current inquiry notes. Continue the conversation to update them.</p>
                )}
              </div>
            </section>
          ) : (
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
          )}

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
      )}
    </div>
  );
}

export default App;
