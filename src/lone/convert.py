"""Convert an LFM checkpoint into a LoNE bundle of Core ML mlprograms.

Bundle layout::

    <out>/manifest.json
    <out>/chunks/chunk_<i>_prefill.mlpackage   (seq = prefill)
    <out>/chunks/chunk_<i>_decode.mlpackage    (seq = 1)
    <out>/head/head_<j>.mlpackage              (final norm + vocab slice, seq = 1)
    <out>/embedding.fp16.npy                   (token gather matrix)
    <out>/tokenizer/...                        (tokenizer.json, chat template, ...)

Chunks are contiguous decoder layers. Caches are ordinary tensors owned by the runtime, so no
Core ML state buffers are involved and conversion stays on the TorchScript path.
"""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import time

import numpy as np

from . import MANIFEST_FORMAT, __version__
from .config import ModelSpec
from .layout import chunk_layout, split_layers

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)

# One head graph re-reads its vocabulary slice every token. The ANE sustains ~49 GB/s on a ~130 MB
# read and ~34 GB/s on a 17 MB one (measured, README), so slices are sized by bytes, not by count.
HEAD_GRAPH_BYTES = 128 * 1024 * 1024
_BITS_PER_WEIGHT = {"fp16": 16, "lut8": 8, "lut6": 6, "lut4": 4}


def head_chunk_count(vocab: int, hidden: int, precision: str = "fp16",
                     budget_bytes: int = HEAD_GRAPH_BYTES) -> int:
    """Number of vocabulary slices, so one head graph's weights fit `budget_bytes`."""
    if vocab <= 0 or hidden <= 0:
        raise ValueError("vocab and hidden must be positive")
    row_bytes = max(1, hidden * _BITS_PER_WEIGHT.get(precision, 16) // 8)
    rows = max(1, budget_bytes // row_bytes)
    return max(1, math.ceil(vocab / rows))


# HF parameter names -> the short keys the static modules use.
_ATTN_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _short_name(hf_suffix: str, spec: ModelSpec) -> str | None:
    """Map a tensor name below `model.layers.<i>.` to a short key, or None if unused."""
    if hf_suffix.endswith(".weight"):
        hf_suffix = hf_suffix[: -len(".weight")]
    if hf_suffix.startswith("self_attn."):
        name = hf_suffix.split(".", 1)[1]
        if name == "out_proj":  # LFM2
            return "o_proj"
        if name in _ATTN_KEYS:
            return name
        if name in ("q_norm", "q_layernorm"):
            return spec.qk_norm_prefix
        if name in ("k_norm", "k_layernorm"):
            return "k_norm" if spec.qk_norm_prefix == "q_norm" else "k_layernorm"
        if name == "gate_proj":
            return "gate_proj"
        return None
    if hf_suffix.startswith("conv."):
        name = hf_suffix.split(".", 1)[1]
        return name if name in ("conv", "in_proj", "out_proj") else None
    if hf_suffix.startswith("feed_forward."):
        name = hf_suffix.split(".", 1)[1]
        return name if name in ("w1", "w2", "w3") else None
    if hf_suffix in ("operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"):
        return hf_suffix
    return None


def _normalize_conv(kernel):
    """Depthwise conv kernel as (C, 1, K), from either HF's (C, 1, K) or MLX's (C, K, 1).

    `mlx_lm.fuse` writes the fused checkpoint in MLX's parameter layout: the conv weight keeps
    MLX's (out, kernel, 1) order under HF's tensor name. Same taps, same order, so the fix is a
    transpose of the two unit axes — but skipping it makes the conv silently read the wrong filter.
    """
    if kernel.dim() == 3 and kernel.shape[1] != 1 and kernel.shape[2] == 1:
        return kernel.transpose(1, 2).contiguous()
    return kernel


def _read_layer(sd, spec: ModelSpec, layer: int, dtype) -> dict:
    prefix = f"model.layers.{layer}."
    params = {}
    for key in sd.keys():
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        short = _short_name(key[len(prefix) :], spec)
        if short is not None:
            tensor = sd.get_tensor(key).to(dtype)
            params[short] = _normalize_conv(tensor) if short == "conv" else tensor
    if spec.is_attention(layer):
        need = {"q_proj", "k_proj", "v_proj", "o_proj", spec.qk_norm_prefix}
        if spec.attention_gate:
            need.add("gate_proj")
    else:
        need = {"conv", "in_proj", "out_proj"}
    need |= {"operator_norm", "ffn_norm", "w1", "w2", "w3"}
    if spec.sandwich_norm:
        need |= {"post_operator_norm", "post_ffn_norm"}
    missing = need - params.keys()
    if missing:
        raise KeyError(f"layer {layer} ({spec.kind(layer)}) is missing tensors: {sorted(missing)}")
    return params


def _trace_and_convert(module, input_names: list[str], output_names: list[str], example: dict, name: str,
                       out_dir: str, precision: str):
    import coremltools as ct
    import torch

    module.eval()
    args = tuple(example[n] for n in input_names)
    with torch.no_grad():
        traced = torch.jit.trace(module, args, strict=False)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=tuple(example[n].shape), dtype=np.float16) for n in input_names],
        outputs=[ct.TensorType(name=n, dtype=np.float16) for n in output_names],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    bits = {"lut4": 4, "lut6": 6, "lut8": 8}.get(precision)
    if bits is not None:
        from coremltools.optimize.coreml import OptimizationConfig, OpPalettizerConfig, palettize_weights

        config = OptimizationConfig(global_config=OpPalettizerConfig(nbits=bits, mode="kmeans"))
        mlmodel = palettize_weights(mlmodel, config)
    elif precision != "fp16":
        raise ValueError(f"unknown precision {precision!r}")

    path = os.path.join(out_dir, f"{name}.mlpackage")
    mlmodel.save(path)
    return path


def plan_summary(pkg_path: str, compute_units: str = "cpu_and_ne") -> dict:
    """MLComputePlan placement for a saved mlpackage. Empty dict if the plan is unavailable."""
    import coremltools as ct

    units = {
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "all": ct.ComputeUnit.ALL,
    }[compute_units]
    try:
        model = ct.models.MLModel(pkg_path, compute_units=units)
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=units)
    except Exception as exc:  # pragma: no cover - depends on the OS / model
        return {"error": f"{type(exc).__name__}: {exc}"}

    counts: dict[str, int] = {}

    def walk(block):
        for op in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue
            dev = str(usage.preferred_compute_device)
            low = dev.lower()
            key = "NE" if ("neural" in low or "ane" in low) else ("GPU" if "gpu" in low else ("CPU" if "cpu" in low else dev))
            counts[key] = counts.get(key, 0) + 1
            for inner in getattr(op, "blocks", []) or []:
                walk(inner)

    walk(plan.model_structure.program.functions["main"].block)
    return {"devices": counts, "total": sum(counts.values())}


def _copy_tokenizer(model_dir: str, out_dir: str) -> None:
    dst = os.path.join(out_dir, "tokenizer")
    os.makedirs(dst, exist_ok=True)
    for name in TOKENIZER_FILES:
        src = os.path.join(model_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, name))


def convert_model(
    model_dir: str,
    out_dir: str,
    ctx: int = 2048,
    prefill: int = 64,
    layers_per_model: int = 4,
    precision: str = "fp16",
    head_chunks: int = 0,
    ring: bool = True,
    decode_only: bool = False,
    prefill_only: bool = False,
    spec_drafts: int = 0,
    check_plan: bool = True,
    compute_units: str = "cpu_and_ne",
    log=print,
) -> dict:
    import torch
    from safetensors import safe_open

    from .torch_model import Chunk, Head, example_inputs

    started = time.time()
    spec = ModelSpec.from_dir(model_dir)
    os.makedirs(out_dir, exist_ok=True)
    chunk_dir = os.path.join(out_dir, "chunks")
    head_dir = os.path.join(out_dir, "head")
    os.makedirs(chunk_dir, exist_ok=True)
    os.makedirs(head_dir, exist_ok=True)

    ckpt = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"{ckpt} not found; LoNE v0.1 expects a single-file safetensors checkpoint")
    log(f"{spec.model_type}: {spec.num_layers} layers, hidden {spec.hidden_size}, "
        f"{len(spec.layer_ids('conv'))} conv / {len(spec.layer_ids('full_attention'))} full / "
        f"{len(spec.layer_ids('sliding_attention'))} sliding, ctx {ctx}, prefill {prefill}")

    if decode_only and prefill_only:
        raise ValueError("--decode-only and --prefill-only are mutually exclusive")
    if spec_drafts:
        if spec_drafts < 1:
            raise ValueError("--spec-drafts must be >= 1")
        if decode_only:
            raise ValueError("--spec-drafts needs the prefill models it verifies with; drop --decode-only")
        # One extra position: speculative verification feeds the pending token plus the drafts.
        prefill = spec_drafts + 1
    if prefill_only:
        seqs = [("prefill", prefill)]
    elif decode_only:
        seqs = [("decode", 1)]
    else:
        seqs = [("prefill", prefill), ("decode", 1)]

    groups = split_layers(spec.num_layers, layers_per_model)
    chunks_meta = []
    with safe_open(ckpt, framework="pt") as sd:
        for gi, layer_ids in enumerate(groups):
            layout = chunk_layout(spec, layer_ids)
            params = {i: _read_layer(sd, spec, i, torch.float16) for i in layer_ids}
            entries = {"name": f"chunk_{gi}", "layers": layer_ids, "models": {}}

            for tag, seq in seqs:
                module = Chunk(spec, params, layer_ids, seq, ctx, ring, trace_conv=seq > 1)
                example = example_inputs(spec, layout, seq, ctx, ring=ring)
                name = f"chunk_{gi}_{tag}"
                path = _trace_and_convert(module, module.input_names, module.output_names, example,
                                          name, chunk_dir, precision)
                entry = {"file": f"chunks/{name}.mlpackage", "seq": seq, "trace_conv": seq > 1}
                if check_plan:
                    entry["plan"] = plan_summary(path, compute_units)
                    log(f"  {name}: {entry['plan'].get('devices', entry['plan'])}")
                entries["models"][tag] = entry
                gc.collect()

            chunks_meta.append(entries)
            del params, module
            gc.collect()

        # Final norm + vocabulary slices of the tied LM head.
        embed = sd.get_tensor("model.embed_tokens.weight").to(torch.float16)
        norm = sd.get_tensor("model.embedding_norm.weight").to(torch.float16)
        vocab = spec.vocab_size
        head_chunks = max(1, head_chunks or head_chunk_count(vocab, spec.hidden_size, precision))
        step = math.ceil(vocab / head_chunks)
        # Speculative decoding needs one logits row per verified position, so the head is traced at
        # the verify length; the decode path pads a single hidden row up to it.
        head_seq = prefill if spec_drafts else 1
        head_meta = []
        for j, start in enumerate(range(0, vocab, step)):
            end = min(start + step, vocab)
            module = Head(spec, embed, norm, start, end)
            example = {"hidden": torch.zeros(1, head_seq, spec.hidden_size, dtype=torch.float16)}
            name = f"head_{j}"
            path = _trace_and_convert(module, ["hidden"], ["logits"], example, name, head_dir, precision)
            entry = {"file": f"head/{name}.mlpackage", "start": start, "end": end}
            if check_plan:
                entry["plan"] = plan_summary(path, compute_units)
            head_meta.append(entry)
            gc.collect()
        log(f"  head: {len(head_meta)} chunks of {step} logits")

    np.save(os.path.join(out_dir, "embedding.fp16.npy"), embed.numpy())
    _copy_tokenizer(model_dir, out_dir)

    manifest = {
        "format": MANIFEST_FORMAT,
        "lone_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": {"model_dir": os.path.abspath(model_dir)},
        "spec": spec.to_dict(),
        "shapes": {"ctx": ctx, "prefill": prefill, "decode": 1},
        "ring": ring,
        "precision": precision,
        "chunks": chunks_meta,
        "head": {"chunks": head_meta, "vocab": vocab, "seq": head_seq},
        "embedding": {"file": "embedding.fp16.npy", "vocab": vocab, "hidden": spec.hidden_size},
        "tokenizer": {"dir": "tokenizer"},
        "speculative": {"drafts": spec_drafts} if spec_drafts else {},
        "seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"wrote {out_dir}/manifest.json ({manifest['seconds']} s)")
    return manifest
