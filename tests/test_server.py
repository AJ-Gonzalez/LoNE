"""The OpenAI message a `serve` chat completion returns when tools are in play."""

from lone.server import assistant_message


def test_message_splits_reasoning_and_parses_a_tool_call():
    message = assistant_message(
        ">\nuse read\n</think\nreading now\n"
        "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert message == {
        "role": "assistant",
        "content": "reading now",
        "reasoning_content": "use read",
        "tool_calls": [{
            "id": "call_0",
            "type": "function",
            "function": {"name": "read", "arguments": '{"path": "a.txt"}'},
        }],
    }


def test_message_without_calls_is_text_only():
    assert assistant_message("Hello.") == {"role": "assistant", "content": "Hello."}
