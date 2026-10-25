"""Regression tests for the ACP adapter's tool-result handling (#248).

The adapter read ``event.result`` on pydantic-ai's ``FunctionToolResultEvent``,
which does not exist — every tool call crashed the turn with an
``AttributeError`` right after the tool's side effect ran. The event carries
the returned part on ``event.part`` instead.
"""

from pydantic_ai.messages import FunctionToolResultEvent, ToolReturnPart

from apps.acp.server import _tool_result_display


def _event(content) -> FunctionToolResultEvent:
    part = ToolReturnPart(tool_name="write_file", content=content, tool_call_id="call-1")
    return FunctionToolResultEvent(part=part)


def test_reads_part_not_result():
    """The event's part is read; ``event.result`` would raise AttributeError."""
    event = _event("wrote notes.txt")
    assert event.result if False else True  # the attribute genuinely does not exist
    assert not hasattr(event, "result")
    tool_call_id, text = _tool_result_display(event.part)
    assert tool_call_id == "call-1"
    assert text == "wrote notes.txt"


def test_none_content_renders_empty():
    _, text = _tool_result_display(_event(None).part)
    assert text == ""


def test_falsy_content_is_not_swallowed():
    """0 / False / [] are legitimate tool returns and must keep their string form."""
    assert _tool_result_display(_event(0).part)[1] == "0"
    assert _tool_result_display(_event(False).part)[1] == "False"
    assert _tool_result_display(_event([]).part)[1] == "[]"


def test_long_content_truncated_for_display():
    _, text = _tool_result_display(_event("x" * 800).part)
    assert text == "x" * 500 + "..."
