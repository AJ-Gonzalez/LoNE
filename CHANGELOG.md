# Changelog

## 0.1.0

N-gram speculative decoding, and two fixes it exposed.

- **Speculative decoding.** `convert --spec-drafts N` converts a verify shape (the prefill shape
  becomes `N+1` tokens and the LM head is traced at that length); `run --spec-ngram` drafts with
  llama.cpp's `ngram-simple` proposer (`--spec-n 12`, `--spec-m 16`) and verifies up to `N` drafts
  per pass. Acceptance is exact for any sampling configuration, not greedy-only: a draft is accepted
  with the target's probability and a rejection draws from the target minus that token, so
  temperature / top-k / top-p keep their distribution. Measured on the LUT8 1.7B coder bundle on
  the NE: 23.2 → 84.4 tok/s on a repeated-text workload (91% acceptance, 64.8 → 23.2 mJ/token) and
  22.8 → 24.0 tok/s on prose, where the proposer finds nothing.
- **Conv kernel normalization.** `mlx_lm.fuse` writes the depthwise conv kernel in MLX's
  `(C, K, 1)` order under HF's tensor name; conversion now normalizes it to `(C, 1, K)` instead of
  failing on a channel mismatch, so fused MLX checkpoints convert directly.
- **No more `$TMPDIR` accumulation.** Core ML compiles every `mlpackage` into
  `$TMPDIR/<name>_<uuid>.mlmodelc` on load (465 MB per chunk graph, ~7 GB per process for a
  two-shape 1.7B bundle) and never collects it. `LoneModel` removes its own copies at exit.
- `lone inspect` reports the verify shape.
- `bench/spec_ab.py`: interleaved speculative A/B with warmup, acceptance, tokens per verify pass,
  ANE watts and J/token.

## 0.0.1

First release: `convert`, `run`, `chat`, `serve`, `inspect`, `verify` for LFM2.5 and LFM3 on the
Apple Neural Engine, with the fp16/LUT precision rules, ring-buffered sliding caches, size-based
head slicing and `mlock` cache wiring.
