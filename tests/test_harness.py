"""The Diverge harness contract: the feed in, the turn's chunks out.

The pure half is pinned here — feed validation, the stored history as chat messages, tool schema
normalization, both LFM tool-call grammars and the envelope — so the contract can be tested
without a bundle. The one turn that needs a model runs only when `LONE_HARNESS_BUNDLE` names a
converted bundle:

    LONE_HARNESS_BUNDLE=/tmp/lone-230m-spec pytest -q tests/test_harness.py
"""

import json
import os

import pytest

from lone.harness import (
    HarnessError,
    chunks,
    endpoint_url,
    envelope,
    history_to_messages,
    load_feed,
    normalize_tools,
    parse_tool_calls,
    run_endpoint,
    split_thinking,
)

BUNDLE = os.environ.get("LONE_HARNESS_BUNDLE")
requires_bundle = pytest.mark.skipif(not BUNDLE, reason="set LONE_HARNESS_BUNDLE to a bundle")


def test_feed_defaults_the_three_lists():
    feed = load_feed('{"input": []}')
    assert feed == {"input": [], "tools": [], "resources": []}


@pytest.mark.parametrize(
    "text",
    ["", "[]", '{"input": {}}', '{"tools": "x"}', "{not json"],
)
def test_feed_refuses_what_is_not_the_contract(text):
    with pytest.raises(HarnessError):
        load_feed(text)


def test_a_prompt_is_a_user_message_of_text_blocks():
    messages = history_to_messages([[
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]])
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "hello\nworld"}]}]


def test_non_text_blocks_are_marked_not_dropped():
    messages = history_to_messages([[
        {"type": "image", "data": "...", "mimeType": "image/png"},
        {"type": "resource_link", "uri": "file:///notes.md"},
        {"type": "text", "text": "read these"},
    ]])
    text = messages[0]["content"][0]["text"]
    assert "[image]" in text and "[resource file:///notes.md]" in text and "read these" in text


def test_assistant_chunks_become_one_message_with_tool_calls():
    messages = history_to_messages([
        [{"type": "text", "text": "list the files"}],
        {"type": "assistant_reasoning", "text": "use ls"},
        {"type": "assistant_text_content", "text": "Sure."},
        {"type": "assistant_tool_call", "id": "c1", "name": "ls", "arguments": '{"path":'},
        {"type": "assistant_tool_call", "id": "c1", "arguments": ' "/tmp"}'},
        {"type": "tool_response", "id": "c1", "content": [{"type": "text", "text": "a.txt"}]},
        {"type": "assistant_text_content", "text": "One file."},
    ])
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "assistant"]
    assistant = messages[1]
    assert assistant["thinking"] == "use ls"
    assert assistant["content"] == "Sure."
    assert assistant["tool_calls"] == [{
        "id": "c1",
        "type": "function",
        "function": {"name": "ls", "arguments": {"path": "/tmp"}},
    }]
    assert messages[2] == {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}


def test_tool_response_keeps_structured_content_when_there_is_no_text():
    messages = history_to_messages([
        {"type": "tool_response", "id": "c1", "content": [], "structuredContent": {"n": 1}},
    ])
    assert messages == [{"role": "tool", "tool_call_id": "c1", "content": '{"n": 1}'}]


def test_usage_and_user_chunks_are_not_replayed():
    messages = history_to_messages([
        {"type": "usage", "input_tokens": 1},
        {"type": "notification", "message": "a warning"},
        {"type": "user_text_content", "text": "already in the prompt"},
    ])
    assert messages == []


def test_tools_are_normalized_from_rmcp_and_from_the_wrapped_shape():
    rmcp = {"name": "read", "description": "read a file", "inputSchema": {"type": "object"}}
    wrapped = {"type": "function", "function": {"name": "ls", "parameters": {"type": "object"}}}
    assert normalize_tools([rmcp, wrapped, {"description": "nameless"}]) == [
        {
            "type": "function",
            "function": {"name": "read", "description": "read a file",
                         "parameters": {"type": "object"}},
        },
        {
            "type": "function",
            "function": {"name": "ls", "description": "",
                         "parameters": {"type": "object"}},
        },
    ]


def test_reasoning_is_split_off_the_generated_turn():
    assert split_thinking(">\nweigh it\n</think\n\nanswer") == ("weigh it", "answer")
    assert split_thinking("answer only") == ("", "answer only")
    assert split_thinking("<think</think\nanswer") == ("<think", "answer")


def test_lfm3_tool_calls_are_parsed_with_json_values():
    text, calls = parse_tool_calls(
        "before\n<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n"
        "<parameter=lines>\n10\n</parameter>\n</function>\n</tool_call>\nafter"
    )
    assert text == "before\n\nafter"
    assert calls == [{"name": "read", "arguments": {"path": "a.txt", "lines": 10}}]


def test_lfm25_tool_calls_are_parsed_with_quoted_values():
    text, calls = parse_tool_calls(
        "checking <|tool_call_start|>[read(path='a.txt', lines=10), ls()]<|tool_call_end|>"
    )
    assert text == "checking"
    assert calls == [
        {"name": "read", "arguments": {"path": "a.txt", "lines": 10}},
        {"name": "ls", "arguments": {}},
    ]


def test_text_without_calls_is_left_whole():
    text, calls = parse_tool_calls("plain answer")
    assert (text, calls) == ("plain answer", [])


def test_chunks_are_the_six_kinds_the_loop_accepts():
    turn = chunks("why", "hello", [{"name": "ls", "arguments": {"path": "/"}}])
    assert [chunk["type"] for chunk in turn] == [
        "assistant_reasoning", "assistant_text_content", "assistant_tool_call",
    ]
    assert turn[1] == {"type": "assistant_text_content", "text": "hello"}
    assert turn[2] == {
        "type": "assistant_tool_call",
        "id": "call_0",
        "name": "ls",
        "arguments": '{"path": "/"}',
    }


def test_an_empty_turn_says_nothing():
    assert chunks("", "  ", []) == []


def test_the_envelope_is_one_json_line_with_the_turn_under_eval():
    turn = chunks("", "hi", [])
    decoded = json.loads(envelope(turn))
    assert decoded == {"eval": turn, "stdout": ""}


@requires_bundle
def test_one_turn_on_a_bundle_answers_in_chunks():
    from lone.harness import run

    feed = load_feed(json.dumps({
        "input": [[{"type": "text", "text": "Say hi."}]],
        "tools": [],
        "resources": [],
    }))
    turn = run(feed, BUNDLE, max_new_tokens=4, cpu_only=True)
    assert isinstance(turn, list)
    for chunk in turn:
        assert chunk["type"] in {
            "assistant_reasoning", "assistant_text_content", "assistant_tool_call",
        }


def test_endpoint_url_takes_a_base_prefix_or_full_endpoint():
    assert endpoint_url("127.0.0.1:8082") == "http://127.0.0.1:8082/v1/chat/completions"
    assert endpoint_url("http://host:8082/") == "http://host:8082/v1/chat/completions"
    assert endpoint_url("http://host:8082/v1") == "http://host:8082/v1/chat/completions"
    assert (
        endpoint_url("http://host:8082/v1/chat/completions")
        == "http://host:8082/v1/chat/completions"
    )


def test_endpoint_turn_matches_the_local_parsing_path(canned_server):
    text = (
        ">\nweigh it\n</think\nreading\n"
        "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    reasoning, content = split_thinking(text)
    content, calls = parse_tool_calls(content)
    canned_server.payload = {"choices": [{
        "message": {
            "role": "assistant",
            "content": "reading",
            "reasoning_content": "weigh it",
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path": "a.txt"}'},
            }],
        },
        "finish_reason": "tool_calls",
    }]}
    feed = load_feed(json.dumps({
        "input": [[{"type": "text", "text": "read a.txt"}]],
        "tools": [{"name": "read", "inputSchema": {"type": "object"}}],
        "resources": [],
    }))
    turn = run_endpoint(feed, canned_server.url)
    assert turn == chunks(reasoning, content, calls)
    request = canned_server.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["body"]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "read a.txt"}]}
    ]
    assert request["body"]["tools"][0]["function"]["name"] == "read"
    assert request["body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_endpoint_parses_a_plain_text_answer(canned_server):
    canned_server.payload = {"choices": [{"message": {
        "role": "assistant",
        "content": ">\nwhy\n</think\n<|tool_call_start|>[ls()]<|tool_call_end|>",
    }}]}
    feed = load_feed('{"input": [[{"type": "text", "text": "list"}]]}')
    assert run_endpoint(feed, canned_server.url) == chunks(
        "why", "", [{"name": "ls", "arguments": {}}]
    )


def test_endpoint_failure_is_the_turn_raising(canned_server):
    canned_server.status = 500
    canned_server.payload = {"detail": "no model"}
    feed = load_feed('{"input": [[{"type": "text", "text": "hi"}]]}')
    with pytest.raises(HarnessError, match="HTTP 500"):
        run_endpoint(feed, canned_server.url)
