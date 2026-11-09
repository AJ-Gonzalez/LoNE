# LoNE: Liquid on Neural Engine

LFM3 and LFM2.5 on the Apple Neural Engine, via Core ML.

Core ML is an energy play, not a speed play: decode is bandwidth-bound on the ANE, so expect
MLX-class throughput at a much lower power envelope.

LoNE splits and rescales wide reductions because the ANE loses precision on them; the measured
numbers and the failure modes are below.

## Conversion precision rules

The ANE's arithmetic differs from torch's fp16 in four measurable ways. `torch_model` implements
all four; `lone verify` exists because per-op conversion parity does **not** catch three of them.

| Rule | Why |
|---|---|
| **1 layer per Core ML graph** | 1 → 0.99986 logits cosine, 2 → 0.9968, 4 → 0.9969 on LFM2.5-230M |
| **Wide reductions split** (FFN > 4096, LM head hidden > 1024) | one 6656-wide output projection measured 0.9972 cosine; two 3328-wide ones 0.999994 |
| **Wide reductions scaled by 2^4** (any `k > 1024`; weights ×16, output ×1/16) | the ANE's linear core loses precision when products fall subnormal: LFM3's conv `out_proj` (input absmax 0.066 × weights ~0.02) measured 0.99916 cosine, 0.9999998 after the rescale. Exact in real arithmetic, and no extra I/O |
| **SiLU via `x / (1 + exp(-x))`** | Core ML's fused `silu` (and `x * sigmoid(x)`) is an NE approximation ~1e-3 off; the exp form is exact. FFN gate 0.99935 → 0.999991 on real LFM3 activations |

Scaling a power of two is exact in fp16 unless it overflows; 2^4 keeps every measured accumulator
two orders of magnitude below the fp16 maximum. The norm reductions do not need any of this:
`mean` over 2048 measured exact on the ANE.


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

That per-layer sum is **16.2 tok/s at fp16**. The shipped bundle decodes at **13.2 tok/s** in a
warm process (75.8 ms/token; 24 chunks x ~2.9 ms, the precision ops add ~14 ms/token over the
spike), and **~2 tok/s when the 3.9 GB of weights are evicted** and stream from SSD (measured
1.55 GB/s ceiling, NOTES §10). Weight traffic matches the fp16 layer weights to 0.4%, i.e. every
token re-reads every weight, so residency is what decides throughput. A LUT6 bundle projects to
~1.10 GB/token and **~46 tok/s** `[INFERENCE]`, small enough to stay resident — MLX-4bit parity
(47.6 tok/s) at lower power.

LFM2.5-230M end-to-end (ctx 128, 1 layer per graph): logits cosine **0.99987** vs the HF fp16
model; 50.7 tok/s decode over 32 tokens, 138 ms to the first token.

LFM2.5-350M end-to-end: logits cosine **0.99989**, greedy output identical to HF for 12 tokens
(FFN split into two wide reductions). Chat, streaming, `/metrics` and the ANE byte counters were
verified through the HTTP server.

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
- **Wide reductions need splitting *and* rescaling.** A single output projection loses precision
  on the ANE when the reduction is wide, and any wide reduction loses precision when its products
  fall subnormal. LoNE splits the FFN at 4096 (350M measured 0.9687 before the split, 0.99989
  after) and the LM head at 1024, and scales every `k > 1024` reduction by 2^4 with the output
  rescaled (rule table above). Always run `lone verify` before trusting a new model on the engine.
- **Depth costs precision, slowly.** With the rules above, LFM3's per-layer hidden cosine stays
  ≥0.99958 over 24 layers and end-to-end logits measure **0.99977** — above the 0.999 bar, but
  with less headroom than LFM2.5.
- **Tools / structured output / logit_bias** are not implemented; `stop` strings are applied to the
  decoded text after generation.
- Conversion expects a single-file `model.safetensors` (no sharded checkpoint support yet).

## Models tested

| Model | Status |
|---|---|
| LFM2.5-230M | full convert + generate + server smoke; logits cosine 0.99987, per-layer math matches HF (cosine ≥ 0.99999) |
| LFM3-1.7B | 24 layers, ctx 256, decode-only bundle. Logits cosine **0.99977** vs HF, argmax matches; per-layer hidden cosine 0.999995 → 0.99958 over the 24 layers; greedy diverges from HF after 6 of 12 tokens |
| LFM2.5-350M | logits cosine **0.99989** vs HF, greedy output identical for 12 tokens (with the FFN split below) |

## Verify

End-to-end parity is one command. The HF checkpoint and the Core ML bundle never share a process:
the reference phase runs in a subprocess and caches the hidden states, then the bundle runs its
decode path token by token.

```bash
lone verify --bundle bundles/lfm3-1.7b --model models/lfm3-1.7b-torch \
    --prompt "Explain in one sentence why the sky is blue." --reference /tmp/ref-lfm3.npz
```

It prints logits cosine, argmax agreement, the per-layer hidden-state drift curve, and — with
`--max-new-tokens N` — the greedy token sequences and the index of the first divergence. It exits
non-zero below `--min-cos` (default 0.999). Reuse the saved `.npz` for repeat runs (`--reference`
only) or with `--cpu-only` to separate a mapping error from an ANE arithmetic error.

Per-layer parity against the reference torch modules (fast, but it only compares our module to its
converted form, so it cannot catch the ANE-specific rules above):

```bash
LONE_TEST_MODEL=models/lfm2.5-230m pytest -q tests/test_parity.py
```

## How it works

The 24-layer stack is split into contiguous chunks; each chunk is one `mlprogram` per sequence
shape (prefill, decode). Caches are ordinary tensors: K/V per attention layer and the
`conv_kernel-1` token window per conv layer, concatenated as `past || new` inside the graph. The
runtime owns the buffers and passes an additive mask that encodes both the causal structure and
which rows of the preallocated cache are real. LM head and final norm are separate vocabulary
slices. No Core ML state buffers are used, so conversion stays on the TorchScript path.
