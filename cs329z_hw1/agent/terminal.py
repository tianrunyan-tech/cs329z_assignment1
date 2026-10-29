"""Problem (terminal): execute sandboxed commands and bound observations."""

from pathlib import Path

from cs329z_hw1 import sandbox
from cs329z_hw1.tokens import count_tokens


# Keep both the beginning and end of a long output stream.
def head_tail(text, max_chars):
    if len(text) <= max_chars:
        return text

    marker = "\n... output omitted ...\n"
    available = max(0, max_chars - len(marker))
    head_size = available // 2
    tail_size = available - head_size

    head = text[:head_size]
    tail = text[-tail_size:] if tail_size else ""

    return head + marker + tail


def make_terminal_tool(config):
    output_number = 0

    # Save captured output inside the sandbox workspace for later inspection.
    def save_output(text):
        nonlocal output_number

        if config.workspace is None:
            workspace = sandbox.default_workspace()
        else:
            workspace = Path(config.workspace)

        folder = workspace / "terminal_outputs"
        folder.mkdir(parents=True, exist_ok=True)

        # Use relative, numbered paths and never overwrite an earlier output.
        while True:
            output_number += 1
            relative_path = (
                f"terminal_outputs/output-{output_number:04d}.txt"
            )
            path = workspace / relative_path

            try:
                with path.open("x", encoding="utf-8") as file:
                    file.write(text)
                return relative_path
            except FileExistsError:
                continue

    # Every model-generated command goes through the provided sandbox.
    def run_terminal(cmd):
        result = sandbox.run_terminal(
            cmd,
            workspace=config.workspace,
            timeout=config.terminal_timeout,
        )

        details = [f"Exit code: {result.exit_code}"]

        if result.rejected:
            details.append(f"Rejected: {result.reason}")
        elif result.timed_out:
            details.append(f"Timed out: {result.reason}")
        elif result.reason:
            details.append(f"Reason: {result.reason}")

        header = "\n".join(details)

        captured = (
            f"{header}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

        # Reserve most of the context for instructions and conversation.
        # This is an initial observation-size policy, not the context check.
        observation_budget = max(
            200,
            min(1000, config.context_budget // 8),
        )

        shown = captured

        if count_tokens(captured) > observation_budget:
            relative_path = save_output(captured)

            # Leave room for status information and the saved-file pointer.
            stream_chars = max(
                0,
                (observation_budget * 4 - 800) // 2,
            )

            shown = (
                f"{header}\n\n"
                f"STDOUT:\n{head_tail(result.stdout, stream_chars)}\n\n"
                f"STDERR:\n{head_tail(result.stderr, stream_chars)}\n\n"
                f"Output shortened. Captured output saved to: "
                f"{relative_path}\n"
                "Use grep, head, tail, or python3 to inspect that file."
            )

        successful = (
            not result.rejected
            and not result.timed_out
            and result.exit_code == 0
        )

        if not successful:
            # The registry converts this exception into tool_error,
            # preserving the observation in the exception message.
            raise RuntimeError(shown)

        return shown

    return run_terminal
