# Notes and unverified ideas

External suggestions, kept separate from the measured record in [`README.md`](../README.md) and
`NOTES.md`. Nothing here is a claim about LoNE's current behaviour.

## LUT4 + pruning + retraining (2026-11-11)

**Suggestion (external, unverified, via [@NayteThaGreat](https://x.com/NayteThaGreat)):** for the
ANE path, LUT4 palettization combined with pruning and retraining may work better than the current
LUT6/LUT8 trade-off.

**Current measured state:** simulated per-output-channel uniform logits cosine vs fp16 is 0.7515
at 4 bits, 0.8953 at 5, 0.9418 at 6, 0.9870 at 7, 0.9978 at 8. The real 6-bit kmeans bundle is
0.96158 (hidden min 0.92675) and LUT8 is 0.99853. QAT is the known route to 6-bit parity, and
coremltools' `DKMPalettizer` requires CUDA. So "LUT4 + pruning + retraining" is a QAT experiment,
not a post-training conversion change.

**What to test:**

- structured or unstructured pruning to x%, then fake-palettize 4 bits and fine-tune on a CUDA
  box (Kaggle T4×2, NOTES §8);
- compare logits cosine, task quality (held-out perplexity, EvalPlus/Harbor through `lone serve`),
  bundle size and NE placement against LUT6/LUT8;
- confirm pruning preserves 100% NE placement (the current dense conv shapes place fully);
- check whether the smaller LUT (16 clusters) buys decode rate or only size.

## Emit the MIL graph directly, in Swift (2026-11-11)

**Suggestion (external, unverified, quoted, via
[@NayteThaGreat](https://x.com/NayteThaGreat)):** "Ask your model to emit the MIL graph by putting
emitting it through the ANE. No more coremltools. Have it make it in swift."

**Interpretation:** have an LLM generate the MIL program (or Swift code that constructs/loads it)
and hand it to the ANE, instead of TorchScript → coremltools → `mlprogram`.

**Reality check:** MIL is Core ML's internal IR; the public, supported path is coremltools and
`MLModel` / `MLProgram` in Swift. Executing a hand-emitted MIL graph on the ANE likely needs
private APIs and entitlements, and MIL is versioned and undocumented. Treat this as a spike, not
a replacement:

- generate a small MIL graph for one chunk and compare numerics against the current bundle;
- check whether a hand-built `.mlpackage`/MIL is accepted by `MLModel` on macOS 15+;
- measure compile time, placement (`lone verify` op placement) and whether precision changes;
- do not switch the conversion path until numerics and placement are verified.

## Provenance

Both items came from [@NayteThaGreat](https://x.com/NayteThaGreat) on X, 2026-11-11. They are
recorded because they are plausible, not because they are validated.
