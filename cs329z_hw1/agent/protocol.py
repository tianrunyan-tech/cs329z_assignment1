"""Problem (tool_protocol): how tools are described to the model and how calls are written and read."""

import json

from cs329z_hw1.types import ParsedResponse, ToolCall

START = "<tool_call>"
END = "</tool_call>"


# Describe every tool and explain the call format, for the system prompt.
def render_tools(schemas):
    lines = [
        "To call a tool, write one block exactly like this:",
        "",
        START,
        '{"name": "tool_name", "args": {"argument_name": "value"}}',
        END,
        "",
        "Rules:",
        '- Inside the block, write one JSON object with the keys "name" and "args".',
        "- Use the parameter names and types from the tool's schema.",
        "- Do not put code fences or other text inside the block.",
        "- Call at most one tool per reply.",
        "- If the request requires a tool, emit the tool call in this reply.",
        "- Saying that you will use a tool is not a tool call.",
        "- Use lookup tools for facts available through those tools; do not guess the answer.",
        "- If the user explicitly asks to run a tool, call it even if it is described as failing.",
        "- Do not return an empty reply.",
        "- When you need no tool, reply with plain text and no block. That is your final answer.",
        "",
        "Available tools:",
    ]

    for schema in schemas:
        lines.append("")
        lines.append(f"Tool: {schema['name']}")
        lines.append(f"Description: {schema['description']}")
        # json.dumps(data)  Python 数据 → JSON 字符串
        # Convert the parameter schema into JSON text, so the model can read the tool's argument rules.
        lines.append(f"Parameters (JSON Schema): {json.dumps(schema['parameters'])}")

    return "\n".join(lines)


# Find at most one call in a reply; never raise.
def parse_response(reply):
    start = reply.find(START)

    # No call at all: the whole reply is the final answer.
    if start == -1:
        return ParsedResponse(text=reply.strip())

    text_before = reply[:start].strip()
    after_start = reply[start + len(START):].lstrip()

    # Decode one JSON value and record where it ends.
    try:
        payload, length = json.JSONDecoder().raw_decode(after_start)
    except Exception:
        return ParsedResponse(
            text=text_before,
            error=f"The tool call is not complete, valid JSON. Write {START}, one JSON object, then {END}.",
        )

    # Read the text after the JSON and remove the closing marker if present.
    # A missing end tag is accepted, because the JSON object is already complete.
    rest = after_start[length:].lstrip()
    if rest.startswith(END):
        rest = rest[len(END):]

    # Reject additional calls outside the decoded JSON.
    if START in rest:
        return ParsedResponse(text=text_before, error="Call only one tool per reply.")

    # Require an object with a string tool name.
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        return ParsedResponse(text=text_before, error='The tool call needs a "name" string.')

    # Tool-specific argument validation belongs to the registry.
    args = payload.get("args", {})
    if not isinstance(args, dict):
        return ParsedResponse(text=text_before, error='"args" must be a JSON object.')

    # Success: keep any text written before or after the call, then return the call.
    text_after = rest.strip()
    text = "\n".join(part for part in (text_before, text_after) if part)
    return ParsedResponse(text=text, tool_call=ToolCall(name=payload["name"], args=args))


# Write a reply that contains the text and one call in our format.
def format_tool_call(text, call):
    payload = json.dumps({"name": call.name, "args": call.args}, ensure_ascii=False)
    block = f"{START}\n{payload}\n{END}"

    text = text.strip()
    if text:
        return f"{text}\n{block}"
    return block
