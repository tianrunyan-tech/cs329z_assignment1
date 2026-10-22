"""Problem (daily_digest): classify a day's emails, then write one digest."""

from concurrent.futures import ThreadPoolExecutor

from cs329z_hw1.pipelines.priority import classify, format_email

DIGEST_PROMPT = """You write a short morning digest for a busy employee.
You get today's emails, urgent ones first.

Rules:
- At most 250 words in total.
- Start with the urgent items. For each one, say what action is needed and by when,
  and name the company, project, or site it is about.
- Then summarize the other items briefly, grouping similar ones.
- Only use the emails given."""


def daily_digest(emails, lm):
    if not emails:
        return "No emails today."

    # 1. Classify every email concurrently
    with ThreadPoolExecutor(max_workers=8) as pool:
        labels = list(pool.map(lambda e: classify(e, lm)["category"], emails))

    # 2. Drop ignore, treat unknown as normal, urgent first, keep one email per thread
    urgent = [e for e, label in zip(emails, labels) if label == "urgent"]
    normal = [e for e, label in zip(emails, labels) if label in ("normal", "unknown")]
    kept, seen_threads = [], set()
    for email in urgent + normal:
        if email["thread_id"] not in seen_threads:
            seen_threads.add(email["thread_id"])
            kept.append(email)

    # 3. Ask the model to write the digest
    text = "\n\n---\n\n".join(
        ("[URGENT]\n" if email in urgent else "") + format_email(email)
        for email in kept
    )
    return lm([
        {"role": "system", "content": DIGEST_PROMPT},
        {"role": "user", "content": text},
    ])
