"""OpenAI-compatible HTTP server for a LoNE bundle.

Endpoints: /health, /v1/models, /v1/completions, /v1/chat/completions, /metrics.
One model instance, one in-flight request: Core ML serialises predictions anyway, so concurrency
would only queue behind the same ANE.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time

from . import __version__
from .metrics import ANECounters
from .runtime import LoneModel

MAX_TOKENS = 4096
_DONE = ("done",)


def _server_time() -> int:
    return int(time.time())


def create_app(bundle_dir: str, compute_units: str = "cpu_and_ne", wire_caches: bool = False):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    model = LoneModel(bundle_dir, compute_units, wire_caches=wire_caches)
    model_name = os.path.basename(os.path.abspath(bundle_dir))
    lock = asyncio.Lock()
    counters = ANECounters()
    stats = {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "decode_tokens": 0,
        "prefill_seconds": 0.0,
        "decode_seconds": 0.0,
        "last": None,
    }
    app = FastAPI(title="LoNE", version=__version__)
    app.state.wired_bytes = model.wired_bytes

    def _params(body: dict) -> dict:
        return {
            "max_new_tokens": min(int(body.get("max_tokens") or body.get("max_completion_tokens") or 256), MAX_TOKENS),
            "temperature": float(body.get("temperature") or 0.0),
            "top_p": float(body.get("top_p") or 1.0),
            "top_k": int(body.get("top_k") or 0),
            "seed": body.get("seed"),
        }

    def _stop_strings(body: dict) -> list[str]:
        stop = body.get("stop")
        if stop is None:
            return []
        return [stop] if isinstance(stop, str) else list(stop)

    def _split(started: float, first_at: float | None, ended: float) -> tuple[float, float]:
        """Prefill = until the first token is sampled; decode = everything after it."""
        if first_at is None:
            return ended - started, 0.0
        return first_at - started, ended - first_at

    def _record(prompt_tokens: int, ids: list[int], started: float, first_at: float | None,
                ended: float) -> dict:
        prefill_s, decode_s = _split(started, first_at, ended)
        decode_tokens = max(len(ids) - 1, 0)
        stats["requests"] += 1
        stats["prompt_tokens"] += prompt_tokens
        stats["completion_tokens"] += len(ids)
        stats["decode_tokens"] += decode_tokens
        stats["prefill_seconds"] += prefill_s
        stats["decode_seconds"] += decode_s
        total = prefill_s + decode_s
        stats["last"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(ids),
            "prefill_seconds": round(prefill_s, 3),
            "decode_seconds": round(decode_s, 3),
            "seconds": round(total, 3),
            "prefill_tokens_per_second": round(prompt_tokens / prefill_s, 2) if prefill_s else None,
            "decode_tokens_per_second": round(decode_tokens / decode_s, 2) if decode_s else None,
            "tokens_per_second": round(len(ids) / total, 2) if total else None,
            "ane_bytes": stats.get("_ane_delta") or {},
        }
        return stats["last"]

    def _run(prompt_ids: list[int], params: dict) -> tuple[list[int], float, float, float | None]:
        before = counters.read() if counters.ok else {}
        started = time.perf_counter()
        first: list[float | None] = [None]

        def on_token(_token: int) -> None:
            if first[0] is None:
                first[0] = time.perf_counter()

        ids = model.generate(prompt_ids, on_token=on_token, **params)
        ended = time.perf_counter()
        after = counters.read() if counters.ok else {}
        stats["_ane_delta"] = counters.delta(before, after) if after else {}
        return ids, started, ended, first[0]

    def _start_worker(prompt_ids: list[int], params: dict, q: queue.Queue,
                      collected: list[int], timing: dict) -> None:
        """Generate into a queue; read the ANE counters around the whole generation (stream path)."""

        def worker() -> None:
            before = counters.read() if counters.ok else {}
            timing["started"] = time.perf_counter()
            first: list[float | None] = [None]

            def on_token(token: int) -> None:
                if first[0] is None:
                    first[0] = time.perf_counter()
                collected.append(token)
                q.put(token)

            try:
                model.generate(prompt_ids, on_token=on_token, **params)
                timing["first"], timing["ended"] = first[0], time.perf_counter()
                after = counters.read() if counters.ok else {}
                stats["_ane_delta"] = counters.delta(before, after) if after else {}
                q.put(_DONE)
            except Exception as exc:  # surface to the client, then stop
                q.put(exc)

        threading.Thread(target=worker, daemon=True).start()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "lone"}]}

    @app.get("/metrics")
    def metrics():
        prefill = stats["prefill_seconds"]
        decode = stats["decode_seconds"]
        total = prefill + decode
        return {
            "model": model_name,
            "requests": stats["requests"],
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["completion_tokens"],
            "prefill_seconds": round(prefill, 3),
            "decode_seconds": round(decode, 3),
            "total_seconds": round(total, 3),
            "prefill_tokens_per_second": round(stats["prompt_tokens"] / prefill, 2) if prefill else None,
            "decode_tokens_per_second": round(stats["decode_tokens"] / decode, 2) if decode else None,
            "tokens_per_second": round(stats["completion_tokens"] / total, 2) if total else None,
            "ane_counters_available": counters.ok,
            "last": stats["last"],
        }

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    @app.post("/v1/completions")
    async def completions(body: dict):
        prompt = body.get("prompt") or ""
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        prompt_ids = model.tokenizer.encode(prompt)
        params = _params(body)
        streams = bool(body.get("stream"))
        stops = _stop_strings(body)

        if not streams:
            async with lock:
                ids, started, ended, first_at = await asyncio.to_thread(_run, prompt_ids, params)
            text = _apply_stop(model.tokenizer.decode(ids), stops)
            usage = _record(len(prompt_ids), ids, started, first_at, ended)
            return {
                "id": f"cmpl-{_server_time()}",
                "object": "text_completion",
                "created": _server_time(),
                "model": model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": _finish(ids, params)}],
                "usage": {"prompt_tokens": usage["prompt_tokens"],
                          "completion_tokens": usage["completion_tokens"],
                          "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"]},
            }

        async def stream():
            async with lock:
                q: queue.Queue = queue.Queue()
                collected: list[int] = []
                timing: dict = {}
                _start_worker(prompt_ids, params, q, collected, timing)
                yield _sse({"id": f"cmpl-{_server_time()}", "object": "text_completion.chunk",
                            "model": model_name, "choices": [{"index": 0, "text": "", "finish_reason": None}]})
                while True:
                    item = await asyncio.to_thread(q.get)
                    if isinstance(item, tuple) and item and item[0] == "done":
                        break
                    if isinstance(item, Exception):
                        yield _sse({"error": str(item)})
                        return
                    yield _sse({"id": f"cmpl-{_server_time()}", "object": "text_completion.chunk",
                                "model": model_name,
                                "choices": [{"index": 0, "text": model.tokenizer.decode([item]),
                                             "finish_reason": None}]})
                _record(len(prompt_ids), collected, timing["started"], timing["first"], timing["ended"])
                yield _sse({"id": f"cmpl-{_server_time()}", "object": "text_completion.chunk",
                            "model": model_name, "choices": [{"index": 0, "text": "",
                                                              "finish_reason": _finish(collected, params)}]})
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        messages = body.get("messages") or []
        prompt_ids = model.tokenizer.chat(messages, add_generation_prompt=True)
        params = _params(body)
        streams = bool(body.get("stream"))
        stops = _stop_strings(body)
        created = _server_time()

        if not streams:
            async with lock:
                ids, started, ended, first_at = await asyncio.to_thread(_run, prompt_ids, params)
            text = _apply_stop(model.tokenizer.decode(ids), stops)
            usage = _record(len(prompt_ids), ids, started, first_at, ended)
            return {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": _finish(ids, params)}],
                "usage": {"prompt_tokens": usage["prompt_tokens"],
                          "completion_tokens": usage["completion_tokens"],
                          "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"]},
            }

        async def stream():
            async with lock:
                q: queue.Queue = queue.Queue()
                collected: list[int] = []
                timing: dict = {}
                _start_worker(prompt_ids, params, q, collected, timing)
                yield _sse({"id": f"chatcmpl-{created}", "object": "chat.completion.chunk", "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                while True:
                    item = await asyncio.to_thread(q.get)
                    if isinstance(item, tuple) and item and item[0] == "done":
                        break
                    if isinstance(item, Exception):
                        yield _sse({"error": str(item)})
                        return
                    yield _sse({"id": f"chatcmpl-{created}", "object": "chat.completion.chunk",
                                "created": created, "model": model_name,
                                "choices": [{"index": 0, "delta": {"content": model.tokenizer.decode([item])},
                                             "finish_reason": None}]})
                _record(len(prompt_ids), collected, timing["started"], timing["first"], timing["ended"])
                yield _sse({"id": f"chatcmpl-{created}", "object": "chat.completion.chunk", "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": _finish(collected, params)}]})
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def _finish(ids: list[int], params: dict) -> str:
    return "length" if len(ids) >= params["max_new_tokens"] else "stop"


def _apply_stop(text: str, stops: list[str]) -> str:
    for stop in stops:
        idx = text.find(stop)
        if idx >= 0:
            return text[:idx]
    return text
