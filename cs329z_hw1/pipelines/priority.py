"""Problem (priority): label one email with one LLM call."""

import re

from cs329z_hw1.data import REPO_ROOT

RUBRIC = (REPO_ROOT / "data" / "priority_rubric.md").read_text(encoding="utf-8")

SYSTEM_PROMPT = f"""You triage email for its recipient. Follow this rubric:

{RUBRIC}

Reply with exactly two lines:
LABEL: <urgent|normal|ignore>
REASON: <one sentence>"""


def format_email(email):
    return (
        f"From: {email['from']}\n"
        f"To: {', '.join(email['to'])}\n"
        f"Date: {email['date']}\n"
        f"Subject: {email['subject']}\n\n"
        f"{email['body']}"
    )


def parse_label(reply):
    match = re.search(r"label:\s*(urgent|normal|ignore)", reply.lower())
    if match:
        return match.group(1)
    return "unknown"


def classify(email, lm):
    reply = lm([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_email(email)},
    ])
    label = parse_label(reply)
    match = re.search(r"reason:\s*(.+)", reply, re.IGNORECASE)
    reason = match.group(1).strip() if match else "No reason given by the model."
    return {"category": label, "reason": reason}
