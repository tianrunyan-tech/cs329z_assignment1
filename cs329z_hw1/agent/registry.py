"""Problem (tool_registry): register, validate, and execute tools."""

from jsonschema import Draft202012Validator, ValidationError

from cs329z_hw1.types import ToolResult


class ToolRegistry:
    # Store tools by name so calls can find the correct tool quickly.
    def __init__(self, tools):
        self.tools = {}

        for tool in tools:
            self.tools[tool.name] = tool

    # Expose tool descriptions and argument schemas to the model.
    # Python dictionaries preserve registration order.
    def schemas(self):
        schemas = []

        for tool in self.tools.values():
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })

        return schemas

    # Check the tool name and arguments without executing the tool.
    # Return None when valid, or a ToolResult describing the error.
    def validate(self, name, args):
        try:
            if name not in self.tools:
                return ToolResult(
                    status="unknown_tool",
                    content=f"Unknown tool: {name}. Available tools: {', '.join(self.tools)}",
                )

            tool = self.tools[name]

            # Create a validator from the tool's argument schema.
            validator = Draft202012Validator(tool.parameters)

            # Check the supplied arguments; raise an exception if they violate the schema.
            validator.validate(args)

            return None

        # Handle schema validation failures with a specific, concise error message.
        except ValidationError as error:
            return ToolResult(
                status="invalid_args",
                content=f"Invalid arguments for {name}: {error.message}",
            )

        # Catch other exceptions so validation returns an error result instead of crashing.
        except Exception as error:
            return ToolResult(
                status="invalid_args",
                content=str(error),
            )

    # Reject invalid calls first, then execute valid calls.
    # Convert failures into results so the agent can continue.
    def execute(self, name, args):
        validation_error = self.validate(name, args)

        if validation_error is not None:
            return validation_error

        try:
            tool = self.tools[name]
            result = tool.fn(**args)

            return ToolResult(
                status="ok",
                content=str(result),
            )

        except Exception as error:
            return ToolResult(
                status="tool_error",
                content=f"Tool {name} failed: {error}",
            )
