import { startTransition, useEffect, useRef, useState } from "react";
import { FileCheck2, MessagesSquare } from "lucide-react";

import { fetchJson } from "./api";
import BlindReview from "./BlindReview";
import { emptySessionView, mergeSessionSummary, withCommittedTurn, withSessionDetail } from "./sessionView";
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
  const activeSessionIdRef = useRef("");
  const [sessionViews, setSessionViews] = useState({});
  const sessionViewsRef = useRef({});
  const inFlightTurnsRef = useRef(new Set());
  const reconcilingTurnsRef = useRef(new Set());
  const notesTimersRef = useRef(new Map());
  const noteSavesRef = useRef(new Set());
  const [createForm, setCreateForm] = useState(emptyCreateForm);
  const [error, setError] = useState("");
  const [restoreWarning, setRestoreWarning] = useState("");
  const [status, setStatus] = useState("Loading…");
  const [creating, setCreating] = useState(false);
  const [loadingSessionId, setLoadingSessionId] = useState("");

  const activeView = sessionViews[activeSessionId] || emptySessionView();
  const activeSession = activeView.detail?.session_id === activeSessionId
    ? activeView.detail
    : null;
  const composer = activeView.composer;
  const notesDraft = activeView.notesDraft;
  const sending = activeView.turnRequest?.status === "submitting";
  const savingNotes = activeView.notesSaving;
  const loadingSession = Boolean(activeSessionId) && loadingSessionId === activeSessionId;

  function updateView(sessionId, updater) {
    const current = sessionViewsRef.current[sessionId] || emptySessionView();
    const updated = updater(current);
    const next = { ...sessionViewsRef.current, [sessionId]: updated };
    sessionViewsRef.current = next;
    setSessionViews(next);
    return updated;
  }

  function applySessionDetail(detail) {
    if (!detail?.session_id) {
      return null;
    }
    const updated = updateView(detail.session_id, (view) => withSessionDetail(view, detail));
    if (updated.detail !== detail) {
      return updated.detail;
    }
    setSessions((previous) => previous.map((session) =>
      session.session_id === detail.session_id
        ? mergeSessionSummary(session, detail)
        : session,
    ));
    return detail;
  }

  function selectSession(sessionId) {
    activeSessionIdRef.current = sessionId;
    setActiveSessionId(sessionId);
  }

  async function refreshSession(sessionId) {
    const detail = await fetchJson(`/api/sessions/${sessionId}`);
    return applySessionDetail(detail);
  }

  useEffect(() => {
    let cancelled = false;
    async function bootstrap() {
      try {
        const health = await fetchJson("/api/health");
        if (cancelled) {
          return;
        }
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
        if (cancelled) {
          return;
        }
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
            selectSession(sessionsPayload.sessions[0].session_id);
          }
        });
        setStatus("Ready");
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError.message);
          setStatus("Failed to load");
        }
      }
    }
    bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!activeSessionId) {
      return;
    }
    const sessionId = activeSessionId;
    let cancelled = false;
    async function loadSession() {
      setLoadingSessionId(sessionId);
      updateView(sessionId, (view) => ({ ...view, error: "" }));
      try {
        const payload = await fetchJson(`/api/sessions/${sessionId}`);
        if (cancelled) {
          return;
        }
        applySessionDetail(payload);
      } catch (loadError) {
        if (!cancelled) {
          updateView(sessionId, (view) => ({ ...view, error: loadError.message }));
        }
      } finally {
        if (!cancelled) {
          setLoadingSessionId((current) => current === sessionId ? "" : current);
        }
      }
    }
    loadSession();
    return () => {
      cancelled = true;
    };
  }, [activeSessionId]);

  function scheduleNotesSave(sessionId, delay = 500) {
    window.clearTimeout(notesTimersRef.current.get(sessionId));
    const timer = window.setTimeout(() => {
      notesTimersRef.current.delete(sessionId);
      void saveNotes(sessionId);
    }, delay);
    notesTimersRef.current.set(sessionId, timer);
  }

  async function saveNotes(sessionId) {
    const view = sessionViewsRef.current[sessionId];
    if (!view?.detail || view.notesDraft === view.notesSaved || noteSavesRef.current.has(sessionId)) {
      return;
    }
    const notes = view.notesDraft;
    noteSavesRef.current.add(sessionId);
    updateView(sessionId, (current) => ({ ...current, notesSaving: true, notesError: "" }));
    let succeeded = false;
    try {
      const payload = await fetchJson(`/api/sessions/${sessionId}/notes`, {
        method: "PUT",
        body: JSON.stringify({ notes }),
      });
      applySessionDetail(payload);
      updateView(sessionId, (current) => ({ ...current, notesSaved: notes }));
      succeeded = true;
    } catch (saveError) {
      updateView(sessionId, (current) => ({ ...current, notesError: saveError.message }));
    } finally {
      noteSavesRef.current.delete(sessionId);
      updateView(sessionId, (current) => ({ ...current, notesSaving: false }));
      const latest = sessionViewsRef.current[sessionId];
      if (succeeded && latest.notesDraft !== latest.notesSaved) {
        scheduleNotesSave(sessionId, 0);
      }
    }
  }

  function handleNotesChange(value) {
    const sessionId = activeSessionIdRef.current;
    if (!sessionId || sessionViewsRef.current[sessionId]?.detail?.session_id !== sessionId) {
      return;
    }
    updateView(sessionId, (view) => ({ ...view, notesDraft: value, notesInitialized: true }));
    scheduleNotesSave(sessionId);
  }

  async function refreshSessions() {
    const payload = await fetchJson("/api/sessions");
    startTransition(() => {
      setSessions((previous) => (payload.sessions || []).map((session) => {
        const cached = sessionViewsRef.current[session.session_id]?.detail;
        const fromCache = cached ? mergeSessionSummary(session, cached) : session;
        const prior = previous.find((item) => item.session_id === session.session_id);
        return prior ? mergeSessionSummary(fromCache, prior) : fromCache;
      }));
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
        applySessionDetail(payload);
        selectSession(payload.session_id);
      });
      await refreshSessions();
    } catch (createError) {
      setError(createError.message);
    } finally {
      setCreating(false);
    }
  }

  function setTurnRequest(sessionId, request, status, message = "") {
    updateView(sessionId, (view) => ({
      ...view,
      turnRequest: { ...request, status, message },
      error: "",
    }));
  }

  async function acceptCommittedTurn(sessionId, request, payload, committedTurnIndex, serverCurrentIndex, replayed) {
    const currentIndex = sessionViewsRef.current[sessionId]?.detail?.turn_index ?? 0;
    const committedIndex = committedTurnIndex ?? payload.turn_index ?? 0;
    if (replayed || currentIndex > committedIndex ||
      (serverCurrentIndex ?? 0) > committedIndex) {
      try {
        await refreshSession(sessionId);
      } catch (refreshError) {
        setTurnRequest(
          sessionId, request, "committed_refresh_needed",
          `The turn was committed, but the current session could not be loaded (${refreshError.message}). Check status to refresh it.`,
        );
        return;
      }
    } else {
      applySessionDetail(payload);
    }
    updateView(sessionId, (view) => withCommittedTurn(view, request));
    try {
      await refreshSessions();
    } catch {
      // Session detail was updated; a list refresh can be retried on navigation.
    }
  }

  async function reconcileTurn(sessionId, request) {
    if (reconcilingTurnsRef.current.has(sessionId)) {
      return;
    }
    reconcilingTurnsRef.current.add(sessionId);
    setTurnRequest(sessionId, request, "checking", "Checking whether this turn was saved…");
    try {
      const receipt = await fetchJson(
        `/api/sessions/${sessionId}/turn-requests/${request.requestId}`,
      );
      if (receipt.status === "committed" && receipt.session) {
        await acceptCommittedTurn(
          sessionId, request, receipt.session, receipt.committed_turn_index,
          receipt.current_turn_index, true,
        );
        return;
      }
      const messages = {
        in_progress: "This turn is still running. Check its status before sending another turn.",
        unknown: "The turn outcome is unknown. A model call may have happened; do not send it again automatically. Start a new session to continue safely.",
        not_found: "No saved receipt was found. Delivery is uncertain. You can explicitly retry this same request after checking the session state.",
        failed: "The turn failed. Your draft is still here; you can send a new request after reviewing it.",
      };
      let refreshed = null;
      if (receipt.status !== "in_progress") {
        try {
          refreshed = await refreshSession(sessionId);
        } catch {
          // The receipt remains visible even if refreshing the session fails.
        }
      }
      if (receipt.status === "failed" && !refreshed) {
        setTurnRequest(
          sessionId, request, "unresolved",
          "The request failed, but the current session could not be loaded. Check status again before sending a new turn.",
        );
      } else if (receipt.status === "not_found" && refreshed?.pending_turn_status === "unknown") {
        setTurnRequest(
          sessionId, request, "unknown",
          "This session has an earlier turn with an unknown outcome. Keep it for review and start a new session to continue.",
        );
      } else {
        setTurnRequest(sessionId, request, receipt.status, messages[receipt.status] || "Turn status could not be determined.");
      }
    } catch (checkError) {
      setTurnRequest(
        sessionId, request, "unresolved",
        `Could not check this turn (${checkError.message}). Check its status before sending another turn.`,
      );
    } finally {
      reconcilingTurnsRef.current.delete(sessionId);
    }
  }

  async function submitTurnRequest(sessionId, request) {
    if (inFlightTurnsRef.current.has(sessionId)) {
      return;
    }
    inFlightTurnsRef.current.add(sessionId);
    setTurnRequest(sessionId, request, "submitting", "Sending turn…");
    try {
      const payload = await fetchJson(`/api/sessions/${sessionId}/turn`, {
        method: "POST",
        body: JSON.stringify({
          user_text: request.userText,
          request_id: request.requestId,
          expected_turn_index: request.expectedTurnIndex,
        }),
      });
      await acceptCommittedTurn(
        sessionId, request, payload, payload.committed_turn_index,
        payload.current_turn_index, payload.replayed,
      );
    } catch (turnError) {
      const code = turnError.code || turnError.detail?.code;
      if (turnError.status === 409) {
        if (code === "turn_outcome_unknown") {
          setTurnRequest(sessionId, request, "unknown", turnError.message);
        } else if (code === "turn_in_progress") {
          setTurnRequest(sessionId, request, "in_progress", turnError.message);
        } else {
          setTurnRequest(sessionId, request, "failed", turnError.message);
        }
        try {
          await refreshSession(sessionId);
        } catch (refreshError) {
          setTurnRequest(
            sessionId, request, "unresolved",
            `${turnError.message} The session could not be reloaded (${refreshError.message}); check status before sending another turn.`,
          );
        }
      } else if (turnError.status && turnError.status >= 400 && turnError.status < 500) {
        setTurnRequest(sessionId, request, "failed", turnError.message);
      } else {
        await reconcileTurn(sessionId, request);
      }
    } finally {
      inFlightTurnsRef.current.delete(sessionId);
    }
  }

  function handleSendTurn() {
    const sessionId = activeSessionIdRef.current;
    const view = sessionViewsRef.current[sessionId];
    if (!sessionId || view?.detail?.session_id !== sessionId || !view.composer.trim() ||
      inFlightTurnsRef.current.has(sessionId) ||
      ["submitting", "checking", "in_progress", "unknown", "not_found", "unresolved", "committed_refresh_needed"]
        .includes(view.turnRequest?.status) ||
      ["in_progress", "unknown"].includes(view.detail.pending_turn_status)) {
      return;
    }
    const request = {
      requestId: crypto.randomUUID(),
      userText: view.composer,
      expectedTurnIndex: view.detail.turn_index ?? 0,
      composerRevision: view.composerRevision,
    };
    void submitTurnRequest(sessionId, request);
  }

  async function retrySameTurn() {
    const sessionId = activeSessionIdRef.current;
    const request = sessionViewsRef.current[sessionId]?.turnRequest;
    if (!sessionId || request?.status !== "not_found" || inFlightTurnsRef.current.has(sessionId)) {
      return;
    }
    setTurnRequest(sessionId, request, "checking", "Checking the current session before retrying the same request…");
    try {
      const detail = await refreshSession(sessionId);
      if (detail.pending_turn_status === "unknown") {
        setTurnRequest(sessionId, request, "unknown", "This session has a turn with an unknown outcome. Keep it for review and start a new session to continue.");
        return;
      }
      if (detail.pending_turn_status === "in_progress") {
        setTurnRequest(sessionId, request, "in_progress", "Another turn is still running. Check status before retrying.");
        return;
      }
      if (detail.turn_index !== request.expectedTurnIndex) {
        setTurnRequest(sessionId, request, "failed", "The session changed. Review the current turn and your draft before sending again.");
        return;
      }
      void submitTurnRequest(sessionId, request);
    } catch (retryError) {
      setTurnRequest(sessionId, request, "unresolved", `Could not verify the session (${retryError.message}). Check status before retrying.`);
    }
  }

  function applySeedTurn(text) {
    const sessionId = activeSessionIdRef.current;
    if (sessionId && sessionViewsRef.current[sessionId]?.detail?.session_id === sessionId) {
      updateView(sessionId, (view) => ({
        ...view, composer: text, composerInitialized: true,
        composerRevision: view.composerRevision + 1,
      }));
    }
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
      {workspace === "playtest" && (activeView.error || activeView.notesError) ? (
        <div className="error-banner" role="alert">
          {activeView.error || activeView.notesError}
          {activeView.notesError ? (
            <button className="inline-action" onClick={() => void saveNotes(activeSessionId)} type="button">
              Retry notes save
            </button>
          ) : null}
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
                  onClick={() => selectSession(session.session_id)}
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
              {activeSession.pending_turn_status === "failed"
                ? "A previous turn failed. Its text was recovered as a draft; review it before sending a new request."
                : "A previous turn has an unknown outcome. A model call may have happened. Keep this session for review and start a new session to continue."}
            </div>
          ) : null}

          {activeView.turnRequest ? (
            <div className="turn-status-banner" role="status">
              <strong>Turn request: {activeView.turnRequest.status.replaceAll("_", " ")}</strong>
              <p>{activeView.turnRequest.message}</p>
              <div className="turn-status-actions">
                {["in_progress", "unknown", "not_found", "unresolved", "committed_refresh_needed"].includes(activeView.turnRequest.status) ? (
                  <button
                    className="inline-action"
                    disabled={inFlightTurnsRef.current.has(activeSessionId)}
                    onClick={() => void reconcileTurn(activeSessionId, activeView.turnRequest)}
                    type="button"
                  >
                    Check status
                  </button>
                ) : null}
                {activeView.turnRequest.status === "not_found" ? (
                  <button className="inline-action" onClick={() => void retrySameTurn()} type="button">
                    Retry same request
                  </button>
                ) : null}
              </div>
              {activeView.turnRequest.status !== "submitting" && activeView.turnRequest.status !== "checking" ? (
                <p className="muted-copy">Submitted text: {activeView.turnRequest.userText}</p>
              ) : null}
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
              onChange={(event) => {
                const sessionId = activeSessionIdRef.current;
                if (sessionId && sessionViewsRef.current[sessionId]?.detail?.session_id === sessionId) {
                  updateView(sessionId, (view) => ({
                    ...view, composer: event.target.value, composerInitialized: true,
                    composerRevision: view.composerRevision + 1,
                  }));
                }
              }}
              onKeyDown={onComposerKeyDown}
              placeholder="Type a user turn. Cmd/Ctrl+Enter sends."
              disabled={!activeSession}
              rows={7}
            />
            <div className="composer-actions">
              <span className="muted-copy">
                {composer.length} chars · draft stays with this session
              </span>
              <button
                className="primary-button"
                disabled={sending || !activeSession || !composer.trim() ||
                  ["submitting", "checking", "in_progress", "unknown", "not_found", "unresolved", "committed_refresh_needed"]
                    .includes(activeView.turnRequest?.status) ||
                  ["in_progress", "unknown"].includes(activeSession?.pending_turn_status)}
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
              onChange={(event) => handleNotesChange(event.target.value)}
              placeholder="Write qualitative notes: awkward timing, unnatural shortlist, good earned ending, etc."
              disabled={!activeSession}
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
