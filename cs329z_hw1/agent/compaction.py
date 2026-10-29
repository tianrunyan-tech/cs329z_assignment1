"""Problem (compaction): keep every model call within config.context_budget."""

from cs329z_hw1.agent.terminal import head_tail
from cs329z_hw1.tokens import count_message_tokens

SUMMARY_PROMPT = """You compress the earlier part of a conversation between a user and an
assistant that uses tools, so the assistant can continue the work without it.
Keep, word for word where you can: every constraint, requirement, preference or fact
the user stated (amounts, places, dates, names, numbers), decisions made, open tasks,
and the key results of tool calls (ids, numbers, file paths).
Leave out greetings and raw tool output. Write at most 200 words."""

REMOVED_NOTE = "[Older result removed to save space. Call the tool again if you need it.]"


# Cap one tool result so a single huge result can never fill the context.
# The model gets about a quarter of the budget from one result.
def cap_tool_result(content, context_budget):
    max_chars = (context_budget // 4) * 4
    if len(content) <= max_chars:
        return content

    return head_tail(content, max_chars - 120) + "\n[Result shortened to fit the context.]"


# Bring messages within the budget, cheapest step first. Returns the new list
# of messages and the index of the current user message in it.
#   1. Replace older tool results (never the newest message) with a short note.
#   2. Replace everything before the current user message with one summary.
#   3. Last resort: drop the oldest steps of the current turn.
# The system prompt, the current user message and the newest message are never removed.
def fit_to_budget(messages, turn_start, budget, summarize):
    if count_message_tokens(messages) <= budget:
        return messages, turn_start

    messages = list(messages)

    # Step 1: older tool results become a one-line note.
    for i in range(1, len(messages) - 1):
        content = messages[i]["content"]
        if content.startswith("<tool_result") and REMOVED_NOTE not in content:
            opening_tag = content.split("\n", 1)[0]
            messages[i] = {
                "role": "user",
                "content": f"{opening_tag}\n{REMOVED_NOTE}\n</tool_result>",
            }

    if count_message_tokens(messages) <= budget:
        return messages, turn_start

    # Step 2: earlier turns become one summary written by the aux model.
    older = messages[1:turn_start]
    if older:
        summary = {"role": "user", "content": summarize(older)}
        messages = [messages[0], summary] + messages[turn_start:]
        turn_start = 2

    # Step 3: drop the oldest steps of the current turn, keeping the user message
    # (at turn_start) and the newest message.
    while count_message_tokens(messages) > budget and len(messages) > turn_start + 2:
        del messages[turn_start + 1]

    return messages, turn_start


# Build the summary block: the user's earlier messages word for word, then the
# aux model's summary of everything else. User messages are never summarized away.
def summary_block(older, earlier_user_messages, aux_lm, budget):
    conversation = "\n\n".join(f"{m['role']}: {m['content']}" for m in older)

    try:
        summary = aux_lm([
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": conversation},
        ])
    except Exception:
        summary = "(No summary available.)"

    # Keep the summary and the list of user messages small.
    summary = head_tail(summary, (budget // 8) * 4)
    user_lines = "\n".join(f"- {message[:400]}" for message in earlier_user_messages)
    user_lines = head_tail(user_lines, (budget // 8) * 4)

    return (
        "<conversation_summary>\n"
        "Earlier messages from the user, word for word:\n"
        f"{user_lines}\n\n"
        "Summary of the earlier conversation:\n"
        f"{summary}\n"
        "</conversation_summary>"
    )
