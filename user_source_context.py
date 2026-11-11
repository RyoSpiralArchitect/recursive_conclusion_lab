# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded verbatim user history, with explicit omissions and no semantic inference."""

MAX_SOURCE_CHARS = 12_000
MAX_SOURCE_TURNS = 16


def source_user_history(history, recent_window_messages):
    """Replay only user messages outside the direct window, in original order.

    Prefer the origin and then the newest missing user messages. Never truncate
    a message mid-constraint, classify a statement as active, or infer permission.
    Full history remains in the session; this is only a bounded request view.
    """
    if recent_window_messages <= 0:
        return None
    boundary = max(0, len(history) - recent_window_messages)
    candidates, user_turn = [], 0
    for index, message in enumerate(history):
        if message.role == "user":
            user_turn += 1
            if index < boundary:
                candidates.append({"turn": user_turn, "text": message.content})
    if not candidates:
        return None
    selected, used = [], 0
    for row in [candidates[0], *reversed(candidates[1:])]:
        size = len(row["text"])
        if len(selected) < MAX_SOURCE_TURNS and used + size <= MAX_SOURCE_CHARS:
            selected.append(row)
            used += size
    selected.sort(key=lambda row: row["turn"])
    selected_turns = {row["turn"] for row in selected}
    omitted = [row["turn"] for row in candidates if row["turn"] not in selected_turns]
    return {
        "kind": "verbatim_historical_user_messages",
        "messages": selected,
        "complete_outside_window": not omitted,
        "omitted_count": len(omitted),
        "first_omitted_turn": omitted[0] if omitted else None,
        "last_omitted_turn": omitted[-1] if omitted else None,
        "text_characters": used,
        "max_text_characters": MAX_SOURCE_CHARS,
        "max_messages": MAX_SOURCE_TURNS,
    }
