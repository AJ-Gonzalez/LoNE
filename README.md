# LoNE: Liquid on Neural Engine

LFM3 and LFM2.5 on the Apple Neural Engine, via Core ML.

Core ML is an energy play, not a speed play: decode is bandwidth-bound on the ANE, so expect
MLX-class throughput at a much lower power envelope.

## Requirements

- Apple Silicon, macOS 14+ (measured on M1, 8 GB, macOS 26.6)
- Python 3.11+
- Conversion: `torch==2.7.0` (the version coremltools 9.0 tests against)

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[convert,chat]'
```

`chat` adds `transformers` for exact chat-template rendering; without it LoNE renders
`chat_template.jinja` itself.

## Convert

```bash
lone convert --model /path/to/LFM3-1.7B --out bundles/lfm3-1.7b --ctx 2048 --prefill 64
```

Each run prints the `MLComputePlan` placement per model and writes `manifest.json`. Useful flags:

| Flag | Default | Effect |
|---|---|---|
| `--precision` | `fp16` | `lut6` palettizes weights (≈2.6× smaller, placement untested) |
| `--layers-per-model` | `1` | decoder layers per Core ML graph; 1 is the most accurate (see below) |
| `--head-chunks` | `8` | LM-head vocabulary slices, one ANE model each |
| `--decode-only` | off | skip prefill models; prompts run token by token |
| `--no-plan` | off | skip the placement report (faster conversion) |

Weights are stored once per sequence shape (prefill + decode), so a bundle is roughly 2× the fp16
checkpoint size. `--decode-only` halves that.

## Run

```bash
lone run --bundle bundles/lfm3-1.7b --chat --prompt "Explain RoPE in two sentences" --max-new-tokens 128
lone inspect --bundle bundles/lfm3-1.7b
```

## Serve

```bash
lone serve --bundle bundles/lfm3-1.7b --port 8082

curl -s localhost:8082/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "lfm3-1.7b",
  "messages": [{"role": "user", "content": "Hi"}],
  "max_tokens": 64
}'
```

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat schema, SSE when `stream: true` |
| `POST /v1/completions` | raw prompt, SSE when `stream: true` |
| `GET /v1/models`, `/health` | harness probes |
| `GET /metrics` | requests, tokens/s, last-request ANE bytes (IOReport) |

Point EvalPlus (`--backend openai --base-url http://127.0.0.1:8082/v1`), lm-eval
(`local-chat-completions`) or Harbor (`--api-format openai_chat_completions`) at it. One request is
processed at a time; Core ML serialises predictions anyway.

## Measured (pre-release, M1 8 GB)

ANE residency and cost of the LFM3-1.7B layers (fp16, real weights, `MLComputePlan` + IOReport
`ANE RD/WR`):

| Unit | NE ops | p50 | ANE MB/pass |
|---|---:|---:|---:|
| 13 conv layers, one graph | 546/546 | 29.97 ms | 1,665 |
| 6 full-attention layers, one graph | 506/506 | 17.26 ms | 694 |
| per token (13 conv + 11 attention layers) | — | 61.6 ms | 2,943 |

That is **16.2 tok/s at fp16**; weight traffic matches the fp16 layer weights to 0.4%, i.e. every
token re-reads every weight. A LUT6 bundle projects to ~1.10 GB/token and **~46 tok/s**
`[INFERENCE]` — MLX-4bit parity (47.6 tok/s) at lower power.

LFM2.5-230M end-to-end (ctx 128, 1 layer per graph): logits cosine **0.99986** vs the HF fp16
model; 50.7 tok/s decode over 32 tokens, 138 ms to the first token. Chat, streaming, `/metrics`
and the ANE byte counters were verified through the HTTP server.

Chunk size is a precision knob on the ANE, not just a call-count knob: 1 layer per graph gives
0.99986 logits cosine, 2 layers 0.9968, 4 layers 0.9969 on the same prompt. Keep
`--layers-per-model 1` unless you have re-measured the tradeoff.

## Limits

- **Static shapes.** Context and prefill length are frozen at conversion; unused context is masked.
  Prompt remainders beyond the last full prefill chunk are decoded one token at a time.
- **fp16 only.** The ANE has no fp32 path; the reference tolerance is per-layer, and drift
  accumulates with stack depth (hence the 1-layer default).
- **No batching.** One request at a time, no continuous batching, no left-padded prompts.
- **Embedding gather runs on the CPU**; the LM head runs on the ANE in vocabulary slices.
- **Prefill models load lazily** (decode models load at startup), but a bundle still stores both
  sequence shapes on disk. On 8 GB, prefer LUT6 or `--decode-only`.
- **Tools / structured output / logit_bias** are not implemented; `stop` strings are applied to the
  decoded text after generation.
- Conversion expects a single-file `model.safetensors` (no sharded checkpoint support yet).

## Models tested

| Model | Status |
|---|---|
| LFM2.5-230M | full convert + generate + server smoke; per-layer math matches HF (cosine ≥ 0.99999) |
| LFM3-1.7B | per-layer math verified (conv / full / sliding, cosine ≥ 0.99999); stack placement and per-token cost measured; full bundle not yet published |
| LFM2.5-350M | same config path, not yet run |

## How it works

The 24-layer stack is split into contiguous chunks; each chunk is one `mlprogram` per sequence
shape (prefill, decode). Caches are ordinary tensors: K/V per attention layer and the
`conv_kernel-1` token window per conv layer, concatenated as `past || new` inside the graph. The
runtime owns the buffers and passes an additive mask that encodes both the causal structure and
which rows of the preallocated cache are real. LM head and final norm are separate vocabulary
slices. No Core ML state buffers are used, so conversion stays on the TorchScript path.
