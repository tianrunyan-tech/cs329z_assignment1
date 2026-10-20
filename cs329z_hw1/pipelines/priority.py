"""Problem (priority): label one email urgent / normal / ignore with one LLM call."""

from __future__ import annotations

import re
from datetime import datetime

from cs329z_hw1.data import REPO_ROOT
from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.types import Email, LMCallable

LABELS = ("urgent", "normal", "ignore")

MAX_BODY_CHARS = 6000  # bodies longer than this are cut; the gold emails are all shorter
MAX_RECIPIENTS_SHOWN = 10  # a long "to" list is shown as a count; it is a mass-mailing signal


def _load_rubric() -> str:
    """The course rubric, minus its last section (how the gold set was built),
    which says nothing about how to label."""
    text = (REPO_ROOT / "data" / "priority_rubric.md").read_text(encoding="utf-8")
    return text.split("## How the gold labels were made")[0].strip()


SYSTEM_PROMPT = f"""You triage email. Read one email and decide what its recipient should do with it, following this rubric exactly.

{_load_rubric()}

Reply with exactly two lines and nothing else:
LABEL: <urgent|normal|ignore>
REASON: <one sentence explaining the label>"""


def format_email(email: Email) -> str:
    """Render the email for the prompt, with the weekday of its date so the
    model can count working days."""
    date = email.get("date") or ""
    try:
        sent = datetime.strptime(date[:19], "%Y-%m-%dT%H:%M:%S")
        date = f"{sent:%A}, {sent:%Y-%m-%d %H:%M} UTC"
    except ValueError:
        pass
    to = list(email.get("to") or [])
    if len(to) > MAX_RECIPIENTS_SHOWN:
        to_line = ", ".join(to[:MAX_RECIPIENTS_SHOWN]) + f" ... and {len(to) - MAX_RECIPIENTS_SHOWN} more recipients"
    else:
        to_line = ", ".join(to) or "(no recipients listed)"
    body = (email.get("body") or "").strip()
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n[... body truncated ...]"
    return (
        f"From: {email.get('from') or '(unknown sender)'}\n"
        f"To: {to_line}\n"
        f"Date: {date}\n"
        f"Subject: {email.get('subject') or '(no subject)'}\n\n"
        f"{body or '(empty body)'}"
    )


# Our own format, "LABEL: urgent"; also matches JSON-ish "category": "urgent".
_FORMATTED = re.compile(r'\b(?:label|category)"?\s*[:=]\s*["*]*\s*(urgent|normal|ignore)\b', re.IGNORECASE)
_ANY_LABEL = re.compile(r"\b(urgent|normal|ignore)\b", re.IGNORECASE)
_REASON = re.compile(r"\breason\"?\s*[:=]\s*\"?(.+)", re.IGNORECASE)


def parse_label(reply: str) -> str:
    """The label the reply states, or "unknown".

    1. Our format wins: the last "LABEL: x" line (a model that corrects
       itself writes the final one last).
    2. Otherwise, if exactly one label word appears anywhere, use it.
    3. No label word, or several different ones: "unknown", not a guess.
    """
    formatted = _FORMATTED.findall(reply)
    if formatted:
        return formatted[-1].lower()
    found = {word.lower() for word in _ANY_LABEL.findall(reply)}
    if len(found) == 1:
        return found.pop()
    return "unknown"


def parse_reason(reply: str, category: str) -> str:
    """The REASON line if there is one, else something non-empty."""
    match = _REASON.search(reply)
    if match:
        reason = match.group(1).strip().strip('"}').strip()
        if reason:
            return reason
    if category == "unknown":
        return "The model's reply did not state a label."
    flat = " ".join(reply.split())
    return flat[:300] or f"The model labeled this email {category}."


def classify(email: Email, lm: LMCallable) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_email(email)},
    ]
    try:
        reply = lm(messages)
    except OutputTruncated as exc:  # a cut-off reply may still hold the label
        reply = exc.text or ""
    reply = reply if isinstance(reply, str) else ""
    category = parse_label(reply)
    return {"category": category, "reason": parse_reason(reply, category)}
