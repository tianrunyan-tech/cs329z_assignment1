"""Problem (agent_loop): one conversation with the agent, driven by a tool-calling loop."""

import copy

from cs329z_hw1.agent.cardinal_tools import choose_tools
from cs329z_hw1.agent.protocol import parse_response, render_tools
from cs329z_hw1.agent.registry import ToolRegistry
from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.tokens import count_message_tokens, count_tokens
from cs329z_hw1.types import AgentResult

AGENT_PROMPT = """You are a helpful assistant that can use tools.
You have no built-in function calling. The only way to use a tool is to write a
<tool_call> block as plain text in your reply, as shown below. A tool runs only
when your reply contains that block.
Call a tool when it helps, read each tool result, then decide the next step.
If a tool is needed, write the actual call rather than only announcing it.
Tool results come back in a user message inside <tool_result> tags.
Problems with your reply come back inside <error> tags; fix the problem and try again.
When you have the answer, reply with plain text and no tool call."""

# Statuses that mean the model wrote a bad call, so it should correct itself.
MALFORMED_STATUSES = ("unknown_tool", "invalid_args")

# Sent once per send when a reply has no tool call but only announces one (or is empty).
REMINDER = (
    "You did not write a <tool_call> block, so no tool ran. To use a tool, write the "
    "<tool_call> block now. If you already have the answer, give it in plain text."
)


# True for an empty reply, or a short reply that only announces an action
# ("I'll look that up."). Such a reply gets one reminder instead of ending the send.
def looks_like_announcement(text):
    if not text.strip():
        return True

    starts = ("I'll", "I’ll", "I will", "Let me", "We need")
    short = len(text.split()) < 40
    return text.strip().startswith(starts) and short


class AgentSession:
    # Set up one conversation: the tools, the system prompt, and empty logs.
    def __init__(self, lm, tools, config):
        self.lm = lm
        self.config = config
        self.mode = config.mode

        # Tools that call a model use aux_lm, so they never use up the agent's own turns.
        aux_lm = config.aux_lm if config.aux_lm is not None else lm
        self.registry = ToolRegistry(choose_tools(tools, config, aux_lm))

        # messages: what the model sees. transcript: every event, never changed.
        system_prompt = AGENT_PROMPT + "\n\n" + render_tools(self.registry.schemas())
        self.messages = [{"role": "system", "content": system_prompt}]
        self.transcript = []

    # Change between "confirm" and "auto" mode (used from Problem 13 on).
    def set_mode(self, mode):
        self.mode = mode

    # Run the loop for one user message until a final answer or a limit.
    def send(self, user_message):
        self.messages.append({"role": "user", "content": user_message})
        self.transcript.append({"type": "user", "content": user_message})

        # These limits apply to each send separately.
        turns = 0
        tokens_used = 0
        malformed_in_a_row = 0
        reminded = False

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
                    # An empty or announcing reply gets one reminder per send.
                    if not reminded and looks_like_announcement(parsed.text):
                        reminded = True
                        self.transcript.append({"type": "reminder", "content": REMINDER})
                        self.messages.append({"role": "user", "content": f"<error>\n{REMINDER}\n</error>"})
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

    # Run one tool call, log it, and show the result to the model.
    def run_tool(self, call):
        self.transcript.append({"type": "tool_call", "name": call.name, "args": call.args})

        result = self.registry.execute(call.name, call.args)

        self.transcript.append({
            "type": "tool_result",
            "name": call.name,
            "status": result.status,
            "content": result.content,
        })
        self.messages.append({
            "role": "user",
            "content": (
                f'<tool_result name="{call.name}" status="{result.status}">\n'
                f"{result.content}\n"
                f"</tool_result>"
            ),
        })
        return result.status

    # Send an error about the model's reply back to the model, and log it.
    def report_error(self, message):
        self.transcript.append({"type": "error", "content": message})
        self.messages.append({"role": "user", "content": f"<error>\n{message}\n</error>"})

    # End the send: the text, why it stopped, and a copy of the transcript now.
    def finish(self, status, text):
        return AgentResult(text=text, status=status, transcript=copy.deepcopy(self.transcript))
