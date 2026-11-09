# LoNE: Liquid on Neural Engine

LFM3 and LFM2.5 on the Apple Neural Engine, via Core ML. Decode is bandwidth-bound on the ANE, so
LoNE targets MLX-class throughput at a lower power envelope; it also works around the ANE's fp16
arithmetic failures, which are the precision rules below.

## Install

Apple Silicon, macOS 15+, Python 3.11+.

```bash
pip install lone-ane                  # run / serve
pip install 'lone-ane[convert,chat]'  # conversion + HF chat templates
```

For a development checkout: `uv venv && source .venv/bin/activate && uv pip install -e '.[convert,chat]'`.

`convert` adds torch 2.7.0 and scikit-learn (coremltools' k-means palettizer); `chat` adds
transformers for exact chat-template rendering (otherwise the bundled Jinja template is used).

## Homebrew

```bash
brew tap aj-gonzalez/lone https://github.com/AJ-Gonzalez/LoNE
brew trust aj-gonzalez/lone
brew install lone-ane
```

The tap builds the `v0.0.1` source tag and installs the runtime (run/serve/verify) into an isolated
Python 3.12 venv. The runtime dependencies are resolved from PyPI at install time (no vendored
Homebrew resources yet). Conversion extras are opt-in in that venv:

```bash
"$(brew --prefix lone-ane)/libexec/bin/pip" install "torch==2.7.0" "scikit-learn>=1.4,<1.6" "transformers>=5.0"
```

## Quick start

```bash
lone convert --model /path/to/LFM3-1.7B --out bundles/lfm3-1.7b --ctx 2048 --prefill 64
lone run --bundle bundles/lfm3-1.7b --chat --prompt "Explain RoPE in two sentences" --max-new-tokens 128
lone serve --bundle bundles/lfm3-1.7b --port 8082
lone inspect --bundle bundles/lfm3-1.7b
lone verify --bundle bundles/lfm3-1.7b --model /path/to/LFM3-1.7B \
    --prompt "Explain in one sentence why the sky is blue."
```

Convert flags: `--precision fp16|lut4|lut6|lut8`, `--layers-per-model 1`, `--head-chunks 8`,
`--decode-only`/`--prefill-only`, `--no-plan`. Weights are stored once per sequence shape, so a
full bundle is ~2x the fp16 checkpoint; `--decode-only` halves it.

`lone verify` runs the HF checkpoint in a subprocess, then the bundle's decode path in the parent;
it prints logits cosine, argmax, the per-layer hidden drift and (with `--max-new-tokens`) the first
greedy divergence. It exits 1 below `--min-cos` (default 0.999). Run it before trusting a
conversion.

## Serve

```bash
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
| `GET /metrics` | prefill/decode seconds and rates, last-request ANE bytes (IOReport; streams included) |

One request at a time; Core ML serialises predictions anyway. `--cpu-only` and
`--compute-units {cpu_and_ne,cpu_and_gpu,cpu_only,all}` select the engine. Point EvalPlus
(`--backend openai`), lm-eval (`local-chat-completions`) or Harbor at the OpenAI endpoint.

## Precision rules

The ANE's arithmetic differs from torch's fp16 in four measurable ways; `torch_model` implements
all four and `lone verify` exists because per-op conversion parity does not catch them.

| Rule | Why (measured) |
|---|---|
| **1 layer per graph** | 1 layer → 0.99986 logits cosine; 2 → 0.9968; 4 → 0.9969 on 230M |
| **Wide reductions split** (FFN > 4096, head hidden > 1024) | one 6656-wide projection 0.9972; two 3328-wide 0.999994 |
| **Wide reductions scaled by 2^4** (any k > 1024; weights ×16, output ×1/16) | products otherwise fall subnormal: LFM3 conv `out_proj` 0.99916 → 0.9999998 |
| **SiLU as `x / (1 + exp(-x))`** | Core ML's fused `silu`/`sigmoid` is an NE approximation ~1e-3 off; gate 0.99935 → 0.999991 |

Power-of-two scaling is exact in fp16 unless it overflows; 2^4 stays far below the fp16 maximum.
Norm reductions need none of this — `mean` over 2048 measured exact on the ANE.

## Measured (M1, 8 GB)

Parity (`lone verify`, decode path):

| Model | Placement | Logits cosine | Notes |
|---|---|---:|---|
| LFM2.5-230M (ctx 128) | CPU | 0.99987 | greedy 2 tokens identical, then EOS differs |
| LFM2.5-350M (ctx 128) | CPU | 0.99989 | greedy 12/12 identical |
| LFM3-1.7B (ctx 256) | NE | **0.99977** | hidden min 0.99958; greedy diverges at token 6/12 |

Placement is `MLComputePlan` over every chunk. The M1 planner declines the small LFM2.5 graphs, so
only LFM3 runs on the NE here; the LFM2.5 rows are CPU-path parity results.

LFM3 speeds and power by engine. Power is the engine's PMP rail, idle-subtracted, via IOReport
without root; rates and power are from the same window.

| Engine | Weights | Prefill tok/s | Decode tok/s | Prefill power | Decode power | J/token (decode) |
|---|---|---:|---:|---:|---:|---:|
| ANE (`CPU_AND_NE`) | fp16, 4.0 GB | 50.9 (32 tok) | 8.5 | 0.25 W | 0.59 W | 0.070 |
| CPU (`CPU_ONLY`) | fp16, 4.0 GB | 2.73 (32 tok) | 0.20 | 0.45 W | 1.64 W | 8.2 |
| GPU (MLX 4-bit, Metal) | 4-bit, 0.95 GB | 385.9 (512 tok) | 53.3 | 9.12 W | 6.04 W | 0.113 |

![LFM3 throughput: prefill and decode](https://raw.githubusercontent.com/AJ-Gonzalez/LoNE/main/docs/tokens.svg)

![Engine power during decode and prefill](https://raw.githubusercontent.com/AJ-Gonzalez/LoNE/main/docs/power.svg)

These are engine numbers, not wall power. During decode the ANE path's CPU orchestration adds
~1.7 W and DRAM ~0.9 W; the CPU row's work power is 2.82 W (1.64 W above idle). The GPU row is
4-bit while ANE/CPU are fp16, so it is not a quantization-controlled comparison. LFM3 ANE decode
varies **2–13 tok/s** with weight residency (8.5 tok/s in the instrumented window, 13.2 tok/s with
`lone run` warm, ~2 tok/s when the 4 GB stream from SSD), so pair rate and power from one run.
`CPU_AND_GPU` is not a real GPU path: 0.51 tok/s with 0.07 W on the GPU channel.

Quantized ANE (LFM3-1.7B) — all of these place every op on the NE; none meets the fp16 parity
bar:

| Weights | Bundle | Decode | ANE power | Energy/token | Logits cosine |
|---|---:|---:|---:|---:|---:|
| fp16 | 4.0 GB | 13.2 tok/s | 0.59 W | 0.070 J | 0.99977 |
| LUT8 | 2.2 GB | 18.7 tok/s* | — | — | 0.9985 |
| LUT6 | 1.8 GB | 20.5 tok/s | 0.87 W | 0.039 J | 0.9526 |
| LUT4 | 1.4 GB | 30.7 tok/s | 0.88 W | 0.037 J | 0.786 |

\* LUT8's parity and size are from a 57-token code prompt; its rate was sampled while a
conversion was running and is a lower bound. Power for 8-bit weights is not measured.

Rates are `lone run`; energy uses the paired instrumented window (LUT6 22.3, LUT4 23.6, fp16 8.5
tok/s). LUT6 per-layer accuracy is 0.9973 (conv) / 0.9981 (attention) but compounds to 0.9526 over
24 layers; LUT4 is 0.896 / 0.935. The precision cliff is measured in NOTES §14.13: each extra LUT
bit buys ~0.04 logits cosine in the 6–8 range, post-training 8-bit reaches 0.9985 and is the only
LUT format near the bar, and 6-bit parity needs QAT.

Those power figures are **1–2.5 s window averages**. A 60 s sustained LUT6 run measured 23.7 tok/s
with the ANE rail at 1.04 W (43.9 mJ/token) and no throttling (10 s bins: 22.6 → 25.2 tok/s).
With CPU and DRAM, whole-system LUT6 decode is ~3.7 W dynamic (~155 mJ/token). A wall/package
figure needs `powermetrics` (root) or a meter; PMP is per-rail.

## Limits

- **Static shapes.** `ctx` and prefill length are frozen at conversion; unused context is masked.
  Prompt remainders are decoded one token at a time. Context memory is 22.5 KB/token for LFM3
  (0.7 GB at 32k, 1.4 GB at 64k, 2.9 GB at 128k); 32k is comfortable on 8 GB and 128k is
  swap-bound `[INFERENCE]` (no long-context run beyond 256 tokens yet).
- **fp16 / palettized only.** No fp32 ANE path; palettized weights trade accuracy (see above).
- **One request, no batching**; no left-padded prompts.
- **Embedding gather is CPU**; the LM head runs on the ANE in vocabulary slices.
- **Prefill models load lazily**, decode models at startup; a full bundle stores both shapes.
- **Not implemented:** tools/structured output, `logit_bias`, `n`, `logprobs`; `stop` strings are
  applied to the decoded text after generation. Sharded checkpoints are unsupported.
- **Depth costs accuracy.** LFM3's hidden cosine falls 0.999995 → 0.99958 over 24 layers; greedy
  output diverges from HF after 6 tokens despite the 0.99977 logits cosine.

## How it works

The stack is split into contiguous chunks, one `mlprogram` per sequence shape
(prefill and decode). Caches are ordinary tensors owned by the runtime — K/V per attention layer
and the `conv_kernel-1` window per conv layer — concatenated as `past || new` inside the graph with
an additive mask keyed by absolute position. No Core ML state buffers, so the conversion path stays
on TorchScript. The LM head is vocabulary slices on the ANE; the embedding gather runs on the CPU.
