"""One turn of a Diverge agentic loop, run on a LoNE bundle.

The container's loop spawns `lone harness` once per turn with one JSON object on stdin —
`{"input": [...], "tools": [...], "resources": [...]}` — and reads one JSON envelope from the
last line of stdout: `{"eval": <chunks>, "stdout": ""}`. `input` is the history in its stored
form: chunk objects, and each user message as an array of MCP content blocks, the newest item
last. Nothing is streamed; the turn is one answer, and the loop does the rest (tool calls, the
history, the queue).

What the harness may say is the six assistant kinds: reasoning, text, an image, audio, a tool
call and a refusal. This one is text-only, so it says reasoning, text and tool calls. A turn that
says nothing is the empty array, which the loop reads as a turn that said nothing.

The two tool-call grammars the LFM chat templates emit are both parsed back:

    <tool_call><function=name><parameter=k>v</parameter></function></tool_call>     LFM3
    <|tool_call_start|>[name(k='v')]<|tool_call_end|>                             LFM2.5

Everything here is pure except [`run`], which loads the bundle and generates.
"""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Any

ASSISTANT_KINDS = frozenset({
    "assistant_reasoning",
    "assistant_text_content",
    "assistant_image_content",
    "assistant_audio_content",
    "assistant_tool_call",
    "assistant_refusal",
    "notification",
})

# `<tool_call>` grammar, one function per block, one parameter per tag.
_LFM3_CALL = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)</function>\s*</tool_call>", re.S
)
_LFM3_PARAM = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.S)
# `<|tool_call_start|>[name(k='v', n=3)]<|tool_call_end|>` grammar.
_LFM25_CALL = re.compile(
    r"<\|tool_call_start\|>\s*\[(.*?)\]\s*<\|tool_call_end\|>", re.S
)


class HarnessError(RuntimeError):
    """The feed or the bundle cannot produce a turn."""


# --------------------------------------------------------------------------- feed


def load_feed(text: str) -> dict:
    """Parse and validate stdin. The three members are lists; absent means empty."""
    try:
        feed = json.loads(text)
    except json.JSONDecodeError as error:
        raise HarnessError(f"stdin is not JSON: {error}") from None
    if not isinstance(feed, dict):
        raise HarnessError("stdin is not a JSON object")
    for name in ("input", "tools", "resources"):
        value = feed.get(name, [])
        if not isinstance(value, list):
            raise HarnessError(f"`{name}` is not a list")
        feed[name] = value
    return feed


# ------------------------------------------------------------- history -> chat


def _content_text(block: Any) -> str:
    """One MCP content block as plain text. LoNE is text-only; the rest is marked, not dropped."""
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return json.dumps(block, ensure_ascii=False)
    kind = block.get("type")
    if kind == "text":
        return block.get("text") or ""
    if kind == "image":
        return "[image]"
    if kind == "audio":
        return "[audio]"
    if kind == "resource":
        resource = block.get("resource") or {}
        if isinstance(resource.get("text"), str):
            return resource["text"]
        return f"[resource {resource.get('uri', '')}]"
    if kind == "resource_link":
        return f"[resource {block.get('uri', '')}]"
    return json.dumps(block, ensure_ascii=False)


def _prompt_message(blocks: list) -> dict:
    texts = [_content_text(block) for block in blocks]
    text = "\n".join(text for text in texts if text)
    return {"role": "user", "content": [{"type": "text", "text": text}] if text else ""}


def _arguments(text: str) -> dict:
    """A stored tool call's arguments as the mapping the chat template requires."""
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def history_to_messages(items: list) -> list[dict]:
    """The stored history as chat messages, in order.

    Consecutive assistant chunks are one message: reasoning under `thinking`, text under
    `content`, tool calls coalesced by id under `tool_calls` (fragments concatenate). A
    `tool_response` is a `tool` message with the id it answers. `usage`, `notification` and the
    user chunk kinds say nothing a model replays.
    """
    messages: list[dict] = []
    assistant: dict | None = None
    calls: dict[str, dict] = {}

    def flush() -> None:
        nonlocal assistant, calls
        if assistant is not None:
            if calls:
                assistant["tool_calls"] = list(calls.values())
            messages.append(assistant)
        assistant, calls = None, {}

    for item in items:
        if isinstance(item, list):
            flush()
            messages.append(_prompt_message(item))
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "tool_response":
            flush()
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("id", ""),
                "content": _response_text(item),
            })
        elif kind in ASSISTANT_KINDS and kind != "notification":
            if assistant is None:
                assistant = {"role": "assistant", "content": ""}
            if kind == "assistant_reasoning":
                assistant["thinking"] = (assistant.get("thinking") or "") + (item.get("text") or "")
            elif kind == "assistant_text_content":
                assistant["content"] = (assistant.get("content") or "") + (item.get("text") or "")
            elif kind == "assistant_tool_call":
                call_id = item.get("id") or f"call_{len(calls)}"
                call = calls.get(call_id)
                if call is None:
                    call = calls[call_id] = {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": item.get("name", ""), "arguments": {}},
                    }
                fragment = item.get("arguments") or ""
                if fragment:
                    raw = call["function"].get("_raw", "") + fragment
                    call["function"]["_raw"] = raw
                    call["function"]["arguments"] = _arguments(raw)
        # Anything else (usage, notification, user_* chunks) says nothing to a model.
    flush()
    # `_raw` exists only so fragments accumulate; the template must not see it.
    for message in messages:
        for call in message.get("tool_calls", ()):
            call["function"].pop("_raw", None)
    return messages


def _response_text(chunk: dict) -> str:
    """A tool response's content as text: text blocks joined, else the structured content."""
    blocks = chunk.get("content")
    if isinstance(blocks, list) and blocks:
        text = "\n".join(_content_text(block) for block in blocks if _content_text(block))
        if text:
            return text
    structured = chunk.get("structuredContent")
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)
    return ""


def normalize_tools(tools: list) -> list[dict]:
    """rmcp `Tool`s as the chat template's `{"type":"function","function":{...}}` entries."""
    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if not name:
            continue
        schema = (
            function.get("parameters")
            or function.get("inputSchema")
            or function.get("input_schema")
            or {"type": "object", "properties": {}}
        )
        normalized.append({
            "type": "function",
            "function": {
                "name": name,
                "description": function.get("description") or "",
                "parameters": schema,
            },
        })
    return normalized


# --------------------------------------------------------------- output parsing


def split_thinking(text: str) -> tuple[str, str]:
    """`(reasoning, content)` from a generated turn.

    The LFM3 generation prompt ends with `<think` (the `>` is generated), so a reasoning turn
    starts with `>` and ends the reasoning at `</think`. A turn without the closing tag is all
    content.
    """
    end = text.find("</think")
    if end < 0:
        return "", text
    reasoning = text[:end]
    if reasoning.startswith(">\n"):
        reasoning = reasoning[2:]
    elif reasoning.startswith(">"):
        reasoning = reasoning[1:]
    content = text[end + len("</think"):]
    if content.startswith(">"):
        content = content[1:]
    return reasoning.strip(), content.lstrip("\n")


def _split_top(text: str, separator: str = ",") -> list[str]:
    """Split on `separator` at nesting depth zero, respecting single and double quotes."""
    parts, start, depth, quote = [], 0, 0, ""
    for i, char in enumerate(text):
        if quote:
            if char == quote and (i == 0 or text[i - 1] != "\\"):
                quote = ""
        elif char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _literal(text: str) -> Any:
    """A template-rendered argument value: quoted strings bare, numbers and objects as JSON."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _lfm3_call(match: re.Match) -> dict:
    name, body = match.group(1), match.group(2)
    arguments = {key: _literal(value) for key, value in _LFM3_PARAM.findall(body)}
    return {"name": name, "arguments": arguments}


def _lfm25_call(text: str) -> dict | None:
    """`name(k='v', n=3)` — the LFM2.5 template's one-call form inside its bracket."""
    text = text.strip()
    opening = text.find("(")
    if opening <= 0 or not text.endswith(")"):
        return None
    name = text[:opening].strip()
    arguments = {}
    for argument in _split_top(text[opening + 1:-1]):
        key, separator, value = argument.partition("=")
        if separator:
            arguments[key.strip()] = _literal(value)
    return {"name": name, "arguments": arguments}


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """`(text without the call blocks, calls)` for both LFM grammars, in source order."""
    spans: list[tuple[int, int, list[dict]]] = []
    for match in _LFM3_CALL.finditer(text):
        spans.append((match.start(), match.end(), [_lfm3_call(match)]))
    for match in _LFM25_CALL.finditer(text):
        calls = [call for call in (
            _lfm25_call(part) for part in _split_top(match.group(1))
        ) if call is not None]
        spans.append((match.start(), match.end(), calls))
    spans.sort(key=lambda span: span[0])

    calls: list[dict] = []
    out, cursor = [], 0
    for start, end, found in spans:
        out.append(text[cursor:start])
        cursor = end
        calls.extend(found)
    out.append(text[cursor:])
    return "".join(out).strip(), calls


def chunks(reasoning: str, text: str, calls: list[dict]) -> list[dict]:
    """The turn's chunks: what was thought, what was said, what was called."""
    out: list[dict] = []
    if reasoning.strip():
        out.append({"type": "assistant_reasoning", "text": reasoning.strip()})
    if text.strip():
        out.append({"type": "assistant_text_content", "text": text.strip()})
    for index, call in enumerate(calls):
        chunk = {
            "type": "assistant_tool_call",
            "id": f"call_{index}",
            "name": call["name"],
        }
        arguments = json.dumps(call["arguments"], ensure_ascii=False)
        chunk["arguments"] = arguments
        out.append(chunk)
    return out


def envelope(turn: list[dict]) -> str:
    """The last stdout line: the turn under `eval`, nothing captured under `stdout`."""
    return json.dumps({"eval": turn, "stdout": ""}, ensure_ascii=False)


# -------------------------------------------------------------------- the turn


def _stop_ids(model) -> list[int]:
    """The bundle's EOS ids plus the template's own end-of-turn token, if it is a token."""
    ids = set(model.spec.eos_token_ids or ())
    for marker in ("<|im_end|>", "</s>"):
        encoded = model.tokenizer.encode(marker, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(encoded[0])
    return sorted(ids)


def run(
    feed: dict,
    bundle: str,
    system: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
    thinking: bool = True,
    compute_units: str = "cpu_and_ne",
    wire_caches: bool = False,
) -> list[dict]:
    """One turn on the bundle: the feed in, the chunks out."""
    from .runtime import LoneModel

    messages = history_to_messages(feed["input"])
    if system:
        messages.insert(0, {"role": "system", "content": system})
    tools = normalize_tools(feed["tools"])

    model = LoneModel(bundle, compute_units, wire_caches=wire_caches)
    prompt_ids = model.tokenizer.chat(
        messages,
        tools=tools or None,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    room = model.ctx - len(prompt_ids)
    if room <= 1:
        raise HarnessError(
            f"the turn needs {len(prompt_ids)} tokens of a {model.ctx}-token context; "
            "shorten the conversation or convert a larger bundle"
        )
    max_new_tokens = min(max_new_tokens, room - 1)

    started = time.perf_counter()
    ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        stop_ids=_stop_ids(model),
    )
    elapsed = time.perf_counter() - started
    text = model.tokenizer.decode(ids, skip_special_tokens=True)
    reasoning, content = split_thinking(text) if thinking else ("", text)
    content, calls = parse_tool_calls(content)
    turn = chunks(reasoning, content, calls)

    print(
        f"[harness: {len(prompt_ids)} prompt + {len(ids)} new tokens in {elapsed:.2f} s "
        f"| {len(turn)} chunks]",
        file=sys.stderr,
    )
    return turn
