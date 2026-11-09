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

from .metrics import ANECounters
from .runtime import LoneModel

MAX_TOKENS = 4096


def _server_time() -> int:
    return int(time.time())


def create_app(bundle_dir: str, compute_units: str = "cpu_and_ne"):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    model = LoneModel(bundle_dir, compute_units)
    model_name = os.path.basename(os.path.abspath(bundle_dir))
    lock = asyncio.Lock()
    counters = ANECounters()
    stats = {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "generation_seconds": 0.0,
        "last": None,
    }
    app = FastAPI(title="LoNE", version="0.0.1")

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

    def _record(prompt_tokens: int, ids: list[int], seconds: float) -> dict:
        stats["requests"] += 1
        stats["prompt_tokens"] += prompt_tokens
        stats["completion_tokens"] += len(ids)
        stats["generation_seconds"] += seconds
        stats["last"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(ids),
            "seconds": round(seconds, 3),
            "tokens_per_second": round(len(ids) / seconds, 2) if seconds else None,
            "ane_bytes": stats.get("_ane_delta"),
        }
        return stats["last"]

    def _run(prompt_ids: list[int], params: dict) -> tuple[list[int], float, dict]:
        before = counters.read() if counters.ok else {}
        started = time.perf_counter()
        ids = model.generate(prompt_ids, **params)
        seconds = time.perf_counter() - started
        after = counters.read() if counters.ok else {}
        delta = counters.delta(before, after) if after else {}
        stats["_ane_delta"] = delta
        return ids, seconds, delta

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "lone"}]}

    @app.get("/metrics")
    def metrics():
        total = stats["generation_seconds"]
        return {
            "model": model_name,
            "requests": stats["requests"],
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["completion_tokens"],
            "generation_seconds": round(total, 3),
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
                ids, seconds, _ = await asyncio.to_thread(_run, prompt_ids, params)
            text = _apply_stop(model.tokenizer.decode(ids), stops)
            usage = _record(len(prompt_ids), ids, seconds)
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
                started = time.perf_counter()

                def worker():
                    try:
                        model.generate(prompt_ids, **params, on_token=lambda t: (collected.append(t), q.put(t)))
                        q.put(None)
                    except Exception as exc:  # surface to the client, then stop
                        q.put(exc)

                threading.Thread(target=worker, daemon=True).start()
                yield _sse({"id": f"cmpl-{_server_time()}", "object": "text_completion.chunk",
                            "model": model_name, "choices": [{"index": 0, "text": "", "finish_reason": None}]})
                while True:
                    item = await asyncio.to_thread(q.get)
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        yield _sse({"error": str(item)})
                        return
                    yield _sse({"id": f"cmpl-{_server_time()}", "object": "text_completion.chunk",
                                "model": model_name,
                                "choices": [{"index": 0, "text": model.tokenizer.decode([item]),
                                             "finish_reason": None}]})
                _record(len(prompt_ids), collected, time.perf_counter() - started)
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
                ids, seconds, _ = await asyncio.to_thread(_run, prompt_ids, params)
            text = _apply_stop(model.tokenizer.decode(ids), stops)
            usage = _record(len(prompt_ids), ids, seconds)
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
                started = time.perf_counter()

                def worker():
                    try:
                        model.generate(prompt_ids, **params, on_token=lambda t: (collected.append(t), q.put(t)))
                        q.put(None)
                    except Exception as exc:
                        q.put(exc)

                threading.Thread(target=worker, daemon=True).start()
                yield _sse({"id": f"chatcmpl-{created}", "object": "chat.completion.chunk", "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                while True:
                    item = await asyncio.to_thread(q.get)
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        yield _sse({"error": str(item)})
                        return
                    yield _sse({"id": f"chatcmpl-{created}", "object": "chat.completion.chunk",
                                "created": created, "model": model_name,
                                "choices": [{"index": 0, "delta": {"content": model.tokenizer.decode([item])},
                                             "finish_reason": None}]})
                _record(len(prompt_ids), collected, time.perf_counter() - started)
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
