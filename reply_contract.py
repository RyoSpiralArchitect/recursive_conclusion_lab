# SPDX-License-Identifier: AGPL-3.0-or-later
"""Versioned length measurements, separate from model finish and semantic quality."""

WORD_COUNTER = "whitespace_v1"
SCHEMA = "rcl.reply_contract.v1"


def validate_max_words(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("hold_reply_max_words must be a nonnegative integer.")


def count_words(text: str) -> int:
    """Count whitespace-separated units, including standalone Markdown markers."""
    return len(text.split())


def reply_contract(text: str, max_words: int = 0) -> dict:
    validate_max_words(max_words)
    words = count_words(text)
    status = (
        "empty"
        if not text.strip()
        else "not_configured"
        if not max_words
        else "exceeded"
        if words > max_words
        else "within_budget"
    )
    return dict(
        schema=SCHEMA,
        word_counter=WORD_COUNTER,
        max_words=max_words,
        words=words,
        status=status,
        excess_words=max(0, words - max_words) if max_words else None,
    )


def budget_instruction(max_words: int) -> str:
    validate_max_words(max_words)
    if not max_words:
        return ""
    return (
        f"Complete this reply in at most {max_words} whitespace-separated words, "
        "including standalone headings and list markers. Lead with the answer or "
        "one useful provisional move. Keep the user's purpose, active constraints, "
        "and material uncertainty explicit where they affect that move. Prefer "
        "a compact paragraph; use a list only when needed. Do not fill the budget "
        "with repeated context or alternative menus. If feasibility remains "
        "unresolved, say so briefly instead of inventing permission or forcing "
        "a conclusion. Finish the thought within the budget."
    )


def summarize_contracts(rows: list[dict]) -> dict:
    """Validate logged diagnostics; never infer a configured target from length."""
    statuses = dict(within_budget=0, exceeded=0, empty=0)
    invalid = 0
    for row in rows:
        logged = row.get("reply_contract")
        if logged is None:
            continue
        try:
            expected = reply_contract(row.get("assistant", ""), logged["max_words"])
            if logged != expected:
                raise ValueError("Diagnostic mismatch")
        except (ValueError, TypeError, KeyError):
            invalid += 1
            continue
        if expected["max_words"]:
            statuses[expected["status"]] += 1
    return dict(
        reply_budget_configured_count=sum(statuses.values()),
        reply_budget_within_count=statuses["within_budget"],
        reply_budget_exceeded_count=statuses["exceeded"],
        reply_budget_empty_count=statuses["empty"],
        reply_budget_invalid_diagnostic_count=invalid,
    )
