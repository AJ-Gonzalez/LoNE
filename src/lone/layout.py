"""Graph layout shared by converter and runtime.

A model is a sequence of contiguous decoder-layer chunks. Every chunk has one static tensor per
mutable cache, and the runtime owns those tensors — no Core ML state buffers, so the conversion
path stays plain TorchScript.
"""

from __future__ import annotations

from .config import ATTENTION_KINDS, ModelSpec


def is_attention(kind: str) -> bool:
    return kind in ATTENTION_KINDS


def chunk_layout(spec: ModelSpec, layer_ids: list[int]) -> dict:
    """Input/output names for one chunk, in trace order."""
    attn = [i for i in layer_ids if spec.is_attention(i)]
    conv = [i for i in layer_ids if not spec.is_attention(i)]
    kinds = [k for k in ATTENTION_KINDS if any(spec.kind(i) == k for i in attn)]

    inputs = ["hidden"]
    for kind in kinds:
        inputs += [f"cos_{kind}", f"sin_{kind}", f"mask_{kind}"]
    for i in attn:
        inputs += [f"k_{i}", f"v_{i}"]
    for i in conv:
        inputs += [f"conv_{i}"]

    # Core ML forbids an input and an output sharing a name, so updated caches are suffixed.
    outputs = ["hidden_out"]
    for i in attn:
        outputs += [f"k_{i}_out", f"v_{i}_out"]
    for i in conv:
        outputs += [f"conv_{i}_out"]

    return {
        "layers": list(layer_ids),
        "attention_layers": attn,
        "conv_layers": conv,
        "attention_kinds": kinds,
        "inputs": inputs,
        "outputs": outputs,
    }


def split_layers(num_layers: int, per_model: int) -> list[list[int]]:
    if per_model < 1:
        raise ValueError("layers per model must be >= 1")
    return [list(range(i, min(i + per_model, num_layers))) for i in range(0, num_layers, per_model)]


def attention_mask(kind: str, ctx: int, seq: int, pos: int, window: int | None = None):
    """Additive mask for `past || new` keys.

    Past rows are valid only up to `pos`; the rest of the preallocated buffer is empty, and the
    new block is causal within itself. Mask by absolute position, never by buffer index.
    """
    import numpy as np

    queries = pos + np.arange(seq)
    past_pos = np.arange(ctx)
    new_pos = pos + np.arange(seq)
    past_ok = (past_pos[None, :] < pos) & (past_pos[None, :] <= queries[:, None])
    new_ok = new_pos[None, :] <= queries[:, None]
    if kind == "sliding_attention":
        if window is None:
            raise ValueError("sliding attention needs a window")
        floor = queries[:, None] - (window - 1)
        past_ok &= past_pos[None, :] >= floor
        new_ok &= new_pos[None, :] >= floor
    allowed = np.concatenate([np.broadcast_to(past_ok, (seq, ctx)), new_ok], axis=1)
    mask = np.where(allowed, 0.0, -65504.0).astype(np.float16)
    return mask[None, None]


def cos_sin(inv_freq, positions):
    import numpy as np

    freqs = positions[:, None].astype(np.float32) * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb)[None].astype(np.float16), np.sin(emb)[None].astype(np.float16)


def cache_shapes(spec: ModelSpec, ctx: int) -> dict:
    """Shapes of every runtime-owned cache tensor."""
    attn = {
        i: ((1, spec.num_kv_heads, ctx, spec.head_dim), (1, spec.num_kv_heads, ctx, spec.head_dim))
        for i in range(spec.num_layers)
        if spec.is_attention(i)
    }
    conv = {i: (1, spec.hidden_size, spec.conv_kernel - 1) for i in range(spec.num_layers) if not spec.is_attention(i)}
    return {"attention": attn, "conv": conv}
