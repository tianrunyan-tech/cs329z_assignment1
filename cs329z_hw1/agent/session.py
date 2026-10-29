"""Problem (agent_loop): one conversation with the agent, driven by a tool-calling loop."""

import copy

from cs329z_hw1.agent.cardinal_tools import choose_tools
from cs329z_hw1.agent.memory import MemoryStore
from cs329z_hw1.agent.compaction import cap_tool_result, fit_to_budget, summary_block
from cs329z_hw1.agent.protocol import parse_response, render_tools
from cs329z_hw1.agent.registry import ToolRegistry
from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.tokens import count_message_tokens, count_tokens
from cs329z_hw1.types import AgentResult

AGENT_PROMPT = """You are a helpful assistant that can use tools.
You have no built-in function calling. The only way to use a tool is to write a
<tool_call> block as plain text in your reply, as shown below. A tool runs only
when your reply contains that block.
The tools listed below are available in this conversation. Never tell the user that you cannot access them.
Call a tool when it helps, read each tool result, then decide the next step.
If a tool is needed, write the actual call rather than only announcing it.
Tool results come back in a user message inside <tool_result> tags.
Problems with your reply come back inside <error> tags; fix the problem and try again.
Before you say you cannot find something, use the tools to look further, for example by reading the full document or email.
When you have the answer, reply with plain text and no tool call."""

# Statuses that mean the model wrote a bad call, so it should correct itself.
MALFORMED_STATUSES = ("unknown_tool", "invalid_args")

# Sent (at most 3 times per send) when a reply has no tool call but only announces one or is empty.
# The example uses a tool this session really has, so the model can copy it.
def reminder(example_tool):
    return (
        "No tool ran: a tool call made any other way than text is lost.\n"
        "Write the call as plain text in your reply, starting with <tool_call>, for example:\n"
        "<tool_call>\n"
        f'{{"name": "{example_tool}", "args": {{"query": "your search words"}}}}\n'
        "</tool_call>\n"
        "If you already have the answer, give it in plain text instead."
    )


# The aux_lm reads a reply that has no tool call and decides whether it really answers.
JUDGE_PROMPT = """You check one reply from an assistant that can use tools.
Decide whether the reply gives the user an answer, or only says what the assistant
will do next (for example "I'll look that up") without doing it.
A reply that uses a placeholder such as [name] or [date] instead of a real value
has not answered either.
Reply with exactly one word: ANSWER or ANNOUNCE."""


class AgentSession:
    # Set up one conversation: the tools, the system prompt, and empty logs.
    def __init__(self, lm, tools, config):
        self.lm = lm
        self.config = config
        self.mode = config.mode

        # Tools that call a model use aux_lm, so they never use up the agent's own turns.
        self.aux_lm = config.aux_lm if config.aux_lm is not None else lm
        self.registry = ToolRegistry(choose_tools(tools, config, self.aux_lm))

        # The first tool's name is used in the reminder's example call.
        schemas = self.registry.schemas()
        self.example_tool = schemas[0]["name"] if schemas else "tool_name"

        # messages: what the model sees. transcript: every event, never changed.
        system_prompt = AGENT_PROMPT + "\n\n" + render_tools(self.registry.schemas())

        # Memory policy: load the saved facts into the system prompt at the start of
        # every session (at most 1/8 of the context budget); recall finds the rest.
        if config.memory_dir is not None:
            memory = MemoryStore(config.memory_dir)
            max_chars = (config.context_budget // 8) * 4
            system_prompt += "\n\n" + memory.prompt_section(max_chars)
        self.messages = [{"role": "system", "content": system_prompt}]
        self.transcript = []

        # For compaction: every message the user sent, and where the current one is in messages.
        self.user_messages = []
        self.turn_start = 1

    # Change between "confirm" and "auto" mode (used from Problem 13 on).
    def set_mode(self, mode):
        self.mode = mode

    # Run the loop for one user message until a final answer or a limit.
    def send(self, user_message):
        self.messages.append({"role": "user", "content": user_message})
        self.transcript.append({"type": "user", "content": user_message})
        self.user_messages.append(user_message)
        self.turn_start = len(self.messages) - 1

        # These limits apply to each send separately.
        turns = 0
        tokens_used = 0
        malformed_in_a_row = 0
        reminders = 0

        while True:
            # 1. Check both budgets before every model call.
            if turns >= self.config.max_turns:
                return self.finish(
                    "max_turns",
                    f"I stopped after {turns} model calls without finishing the task.",
                )
            if tokens_used >= self.config.token_budget:
                return self.finish(
                    "token_budget",
                    f"I stopped after using {tokens_used} tokens without finishing the task.",
                )

            # 2. Call the model. Each call has an output limit (max_tokens).
            # A reply cut off at that limit (OutputTruncated) is malformed: log it
            turns += 1
            self.compact()
            try:
                reply = self.lm(self.messages)
            except OutputTruncated as error:
                tokens_used += count_message_tokens(self.messages) + count_tokens(error.text)
                self.transcript.append({"type": "assistant", "content": error.text})
                self.report_error(
                    "Your reply was cut off before it finished. Write a shorter reply."
                )
                malformed_in_a_row += 1
            else:
                tokens_used += count_message_tokens(self.messages) + count_tokens(reply)
                self.messages.append({"role": "assistant", "content": reply})
                self.transcript.append({"type": "assistant", "content": reply})

                # 3. Read the reply: a parse error, a final answer, or a tool call.
                parsed = parse_response(reply)

                if parsed.error is not None:
                    self.report_error(parsed.error)
                    malformed_in_a_row += 1

                # End the send when a well-formed reply contains no tool call.
                elif parsed.tool_call is None:
                    # An empty reply, or one the aux_lm judges as only announcing an action,
                    # gets a reminder instead of ending the send (at most 3 times per send).
                    if reminders < 3 and (
                        not parsed.text.strip()
                        or self.only_announces(user_message, parsed.text)
                    ):
                        reminders += 1
                        text = reminder(self.example_tool)
                        self.transcript.append({"type": "reminder", "content": text})
                        self.messages.append({"role": "user", "content": f"<error>\n{text}\n</error>"})
                    else:
                        text = parsed.text or "(The model returned an empty answer.)"
                        return self.finish("done", text)

                else:
                    status = self.run_tool(parsed.tool_call)

                    # MALFORMED_STATUSES includes unknown tools and invalid arguments.
                    if status in MALFORMED_STATUSES:
                        malformed_in_a_row += 1
                    # A valid call resets the counter, even if the tool fails during execution (tool_error)
                    else:
                        malformed_in_a_row = 0

            # 4. Three malformed replies in a row end the send.
            if malformed_in_a_row >= 3:
                return self.finish(
                    "malformed",
                    "I stopped because my last three replies could not be used.",
                )

    # Ask the aux_lm whether a reply only announces an action.
    # Anything unclear (an odd verdict, or an error) counts as a real answer.
    def only_announces(self, user_message, reply):
        try:
            verdict = self.aux_lm([
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"User request:\n{user_message}\n\nAssistant reply:\n{reply}"},
            ])
        except Exception:
            return False

        return verdict.strip().upper().startswith("ANNOUNCE")

    # Run one tool call, log it, and show the result to the model.
    def run_tool(self, call):
        self.transcript.append({"type": "tool_call", "name": call.name, "args": call.args})

        result = self.registry.execute(call.name, call.args)

        # A huge result is shortened first; the transcript keeps exactly what the model sees.
        content = cap_tool_result(result.content, self.config.context_budget)

        self.transcript.append({
            "type": "tool_result",
            "name": call.name,
            "status": result.status,
            "content": content,
        })
        self.messages.append({
            "role": "user",
            "content": (
                f'<tool_result name="{call.name}" status="{result.status}">\n'
                f"{content}\n"
                f"</tool_result>"
            ),
        })
        return result.status

    # Keep the next model call within the context budget. The transcript is not changed,
    # only a "compaction" event is added to record what happened.
    def compact(self):
        before = self.messages
        self.messages, self.turn_start = fit_to_budget(
            self.messages, self.turn_start, self.config.context_budget, self.summarize
        )
        if self.messages is not before:
            self.transcript.append({
                "type": "compaction",
                "messages_before": len(before),
                "messages_after": len(self.messages),
            })

    # Summarize older messages with the aux model (never the agent's own model).
    def summarize(self, older):
        return summary_block(
            older, self.user_messages[:-1], self.aux_lm, self.config.context_budget
        )

    # Send an error about the model's reply back to the model, and log it.
    def report_error(self, message):
        self.transcript.append({"type": "error", "content": message})
        self.messages.append({"role": "user", "content": f"<error>\n{message}\n</error>"})

    # End the send: the text, why it stopped, and a copy of the transcript now.
    def finish(self, status, text):
        return AgentResult(text=text, status=status, transcript=copy.deepcopy(self.transcript))
