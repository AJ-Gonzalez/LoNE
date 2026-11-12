"""Runtime for a LoNE bundle: tokenizer, caches, prefill, decode, sampling."""

from __future__ import annotations

import json
import os
import sys

import jinja2
import numpy as np
from jinja2.ext import Extension

from .config import ModelSpec
from .layout import attention_mask, cache_lengths, chunk_layout, cos_sin

_COMPUTE_UNITS = {
    "cpu_and_ne": "CPU_AND_NE",
    "cpu_and_gpu": "CPU_AND_GPU",
    "cpu_only": "CPU_ONLY",
    "all": "ALL",
}


class LoneError(RuntimeError):
    pass


class _GenerationExtension(Extension):
    """Jinja2 extension for HF's `{% generation %}` assistant-mask tags (no-op here)."""

    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)  # consume the tag name
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


class BundleTokenizer:
    """Token IDs plus chat-template rendering.

    Uses `transformers` when installed (exact template semantics, tools/documents included);
    otherwise falls back to `tokenizers` + Jinja2, which covers the LFM templates.
    """

    def __init__(self, tokenizer_dir: str):
        from tokenizers import Tokenizer

        self.dir = tokenizer_dir
        self._tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        template_path = os.path.join(tokenizer_dir, "chat_template.jinja")
        self._template = None
        if os.path.exists(template_path):
            with open(template_path) as f:
                source = f.read()
            env = jinja2.Environment(keep_trailing_newline=True, extensions=[_GenerationExtension])
            try:
                self._template = env.from_string(source)
            except Exception:
                self._template = None
        self._hf = None
        self._special = {}
        config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            for key in ("bos_token", "eos_token", "pad_token"):
                value = cfg.get(key)
                if isinstance(value, dict):
                    value = value.get("content")
                if value:
                    self._special[key] = value
        try:  # optional, better template fidelity
            from transformers import AutoTokenizer

            self._hf = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        except Exception:
            self._hf = None

    @property
    def eos_token_id(self) -> int | None:
        return self._tok.token_to_id("</s>")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def chat(self, messages: list[dict], add_generation_prompt: bool = True, **kwargs) -> list[int]:
        if self._hf is not None:
            text = self._hf.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
            )
            return self._hf(text)["input_ids"]
        if self._template is None:
            raise LoneError("bundle has no chat template; use /v1/completions")
        text = self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            bos_token=self._special.get("bos_token", ""),
            eos_token=self._special.get("eos_token", ""),
            **kwargs,
        )
        return self.encode(text)


_LIBMLOCK = None


def _libmlock():
    global _LIBMLOCK
    if _LIBMLOCK is None:
        import ctypes

        libc = ctypes.CDLL("libc.dylib", use_errno=True)
        libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mlock.restype = ctypes.c_int
        _LIBMLOCK = libc
    return _LIBMLOCK


def _wire_pages(arr: np.ndarray) -> tuple[int, int] | None:
    """Wire an array's pages with `mlock(2)` so the kernel cannot page them out.

    Returns the (page-aligned address, byte length) span on success, None when the platform or
    the process refuses. macOS releases wired pages when the mapping is freed, so no explicit
    unwire is needed at process exit.
    """
    if sys.platform != "darwin":
        return None
    import ctypes

    page = os.sysconf("SC_PAGE_SIZE")
    start = arr.ctypes.data & ~(page - 1)
    end = (arr.ctypes.data + arr.nbytes + page - 1) & ~(page - 1)
    if _libmlock().mlock(ctypes.c_void_p(start), ctypes.c_size_t(end - start)) != 0:
        return None
    return start, end - start


def _inv_freq(spec: ModelSpec, kind: str) -> np.ndarray:
    rope = spec.rope_for(kind)
    hd = spec.head_dim
    angles = int(rope.partial * hd) // 2
    idx = np.arange(0, 2 * angles, 2, dtype=np.float64) / hd
    inv = 1.0 / (rope.theta**idx)
    if angles < hd // 2:
        inv = np.concatenate([inv, np.zeros(hd // 2 - angles)])
    return inv.astype(np.float32)


class LoneModel:
    def __init__(self, bundle_dir: str, compute_units: str = "cpu_and_ne", wire_caches: bool = False):
        import coremltools as ct

        self.dir = bundle_dir
        with open(os.path.join(bundle_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        if self.manifest.get("format") != "lone-1":
            raise LoneError(f"unsupported bundle format {self.manifest.get('format')!r}")
        self.spec = ModelSpec.from_dict(self.manifest["spec"])
        self.ctx = int(self.manifest["shapes"]["ctx"])
        self.prefill_len = int(self.manifest["shapes"]["prefill"])

        units = getattr(ct.ComputeUnit, _COMPUTE_UNITS[compute_units])
        self._units = units
        load = lambda path: ct.models.MLModel(path, compute_units=units)

        self._chunks = []
        for entry in self.manifest["chunks"]:
            paths = {tag: os.path.join(bundle_dir, meta["file"]) for tag, meta in entry["models"].items()}
            # Decode models are always needed; prefill models are loaded on first use so an 8 GB
            # machine does not hold both shapes of the whole stack at once.
            loaded = {"decode": load(paths["decode"])} if "decode" in paths else {}
            self._chunks.append({"layout": chunk_layout(self.spec, entry["layers"]), "paths": paths,
                                 "models": loaded, "layers": entry["layers"]})
        self._heads = [load(os.path.join(bundle_dir, h["file"])) for h in self.manifest["head"]["chunks"]]
        self._head_ranges = [(h["start"], h["end"]) for h in self.manifest["head"]["chunks"]]

        emb = self.manifest["embedding"]
        self._embed = np.load(os.path.join(bundle_dir, emb["file"]), mmap_mode="r")
        self._tok = BundleTokenizer(os.path.join(bundle_dir, self.manifest["tokenizer"]["dir"]))

        self._ring = bool(self.manifest.get("ring", False))
        self._cache_len = cache_lengths(self.spec, self.ctx, self._ring)
        self._inv = {kind: _inv_freq(self.spec, kind) for kind in self.spec.rope}
        self._cache_dtype = np.float16
        self._wire_caches = bool(wire_caches)
        self.wired_bytes = 0
        self._wired: list[tuple[int, int]] = []
        self.reset()

    # ------------------------------------------------------------------ state
    @property
    def tokenizer(self) -> BundleTokenizer:
        return self._tok

    def _cache_arrays(self):
        yield from self._k.values()
        yield from self._v.values()
        yield from self._conv.values()

    def reset(self) -> None:
        """Zero the caches. Buffers are allocated once, then reused: at large `ctx` a
        reallocation would briefly hold two full caches, and wired pages must keep their address.
        """
        self.pos = 0
        allocating = not hasattr(self, "_k")
        if allocating:
            self._k = {
                i: np.zeros(
                    (1, self.spec.num_kv_heads, self._cache_len[self.spec.kind(i)], self.spec.head_dim),
                    dtype=self._cache_dtype,
                )
                for i in range(self.spec.num_layers)
                if self.spec.is_attention(i)
            }
            self._v = {i: np.zeros_like(a) for i, a in self._k.items()}
            self._conv = {
                i: np.zeros((1, self.spec.hidden_size, self.spec.conv_kernel - 1), dtype=self._cache_dtype)
                for i in range(self.spec.num_layers)
                if not self.spec.is_attention(i)
            }
        for arr in self._cache_arrays():
            arr.fill(0)
        if allocating and self._wire_caches:
            for arr in self._cache_arrays():
                span = _wire_pages(arr)
                if span is not None:
                    self._wired.append(span)
                    self.wired_bytes += span[1]

    # ------------------------------------------------------------------- math
    def _cos_sin(self, kind: str, positions: np.ndarray):
        return cos_sin(self._inv[kind], positions)

    def _mask(self, kind: str, seq: int, pos: int) -> np.ndarray:
        return attention_mask(kind, self._cache_len[kind], seq, pos, self.spec.sliding_window,
                              ring=self._ring)

    def _embed_tokens(self, ids: list[int]) -> np.ndarray:
        rows = np.asarray(ids, dtype=np.int64)
        return np.asarray(self._embed[rows], dtype=np.float16)[None]

    def _run_chunk(self, chunk: dict, tag: str, hidden: np.ndarray, pos: int) -> None:
        layout = chunk["layout"]
        model = chunk["models"].get(tag)
        if model is None:
            import coremltools as ct

            model = ct.models.MLModel(chunk["paths"][tag], compute_units=self._units)
            chunk["models"][tag] = model
        seq = hidden.shape[1]
        feed = {"hidden": hidden}
        for kind in layout["attention_kinds"]:
            cos, sin = self._cos_sin(kind, pos + np.arange(seq))
            feed[f"cos_{kind}"] = cos
            feed[f"sin_{kind}"] = sin
            feed[f"mask_{kind}"] = self._mask(kind, seq, pos)
        for i in layout["attention_layers"]:
            feed[f"k_{i}"] = self._k[i]
            feed[f"v_{i}"] = self._v[i]
        for i in layout["conv_layers"]:
            feed[f"conv_{i}"] = self._conv[i]

        out = model.predict(feed)
        for i in layout["attention_layers"]:
            self._store(self._k, i, out[f"k_{i}_out"][0], pos, seq)
            self._store(self._v, i, out[f"v_{i}_out"][0], pos, seq)
        for i in layout["conv_layers"]:
            # In place: the conv window keeps its address, so it stays wired and does not
            # allocate once per token.
            self._conv[i][...] = out[f"conv_{i}_out"]
        self._hidden = out["hidden_out"].astype(np.float16)

    @staticmethod
    def _store(cache: dict, i: int, new: np.ndarray, pos: int, seq: int) -> None:
        """Write the new rows at their absolute positions, wrapping when the cache is a ring.

        For a full-attention cache the modulus is the identity (`pos + seq <= ctx`), so one path
        covers both; a prefill longer than the window keeps only the last `cache rows`.
        """
        n = cache[i].shape[2]
        if seq > n:
            new, pos = new[:, -n:, :], pos + seq - n
        index = (pos + np.arange(new.shape[1])) % n
        cache[i][0][:, index, :] = new.astype(np.float16)

    def _forward(self, ids: list[int], tag: str, capture: list | None = None) -> None:
        seq = len(ids)
        pos = self.pos
        if pos + seq > self.ctx:
            raise LoneError(
                f"context exhausted: {pos} + {seq} > {self.ctx}; reconvert with a larger --ctx"
            )
        hidden = self._embed_tokens(ids)
        for chunk in self._chunks:
            self._run_chunk(chunk, tag, hidden, pos)
            hidden = self._hidden
            if capture is not None:
                capture.append(hidden[0].copy())
        self.pos = pos + seq

    def _logits(self) -> np.ndarray:
        parts = []
        for model, (start, end) in zip(self._heads, self._head_ranges):
            parts.append(model.predict({"hidden": self._hidden[:, -1:, :]})["logits"].astype(np.float32))
        return np.concatenate(parts, axis=-1)[0, 0]

    def unload_prefill(self) -> None:
        """Drop lazily loaded prefill models (they reload on the next prefill)."""
        for chunk in self._chunks:
            chunk["models"].pop("prefill", None)

    # --------------------------------------------------------------- generate
    def prefill(self, ids: list[int]) -> None:
        if not ids:
            raise LoneError("empty prompt")
        self.reset()
        remaining = list(ids)
        has_prefill = all("prefill" in c["paths"] for c in self._chunks)
        while has_prefill and len(remaining) >= self.prefill_len:
            self._forward(remaining[: self.prefill_len], "prefill")
            remaining = remaining[self.prefill_len :]
        while remaining:
            self._forward(remaining[:1], "decode")
            remaining = remaining[1:]

    def step(self, token_id: int) -> np.ndarray:
        self._forward([token_id], "decode")
        return self._logits()

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        stop_ids: list[int] | None = None,
        seed: int | None = None,
        on_token=None,
    ) -> list[int]:
        self.prefill(prompt_ids)
        logits = self._logits()
        stop = set(stop_ids or self.spec.eos_token_ids)
        rng = np.random.default_rng(seed)
        out: list[int] = []
        for _ in range(max_new_tokens):
            if self.pos >= self.ctx:
                break
            token = sample(logits, temperature, top_p, top_k, rng)
            if token in stop:
                break
            out.append(token)
            if on_token is not None:
                on_token(token)
            if len(out) >= max_new_tokens:
                break
            logits = self.step(token)
        return out


# Ranks past this can only be sampled when the distribution is flat enough that top-p keeps most
# of the vocabulary; the prefilter detects that and falls back to the full vector.
_SAMPLE_CANDIDATES = 1024


def sample(logits: np.ndarray, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
           rng: np.random.Generator | None = None) -> int:
    """Draw one token. Greedy at temperature 0; otherwise softmax over the sampled support.

    Sorting the whole vocabulary per token cost 9 ms at 128k, but only the head of the
    distribution can ever be drawn: `top_k` is a hard cut and `top_p < 1` keeps a few hundred
    ranks in practice. The support is found by `argpartition` and only that slice is sorted. The
    probabilities passed to `rng.choice` are the true ones (divided by the full-vocabulary mass),
    so the kept set matches the vocabulary-wide computation; when top-p would reach beyond the
    slice, the full vector is used instead.
    """
    if temperature <= 0:
        return int(np.argmax(logits))
    if rng is None:
        rng = np.random.default_rng()

    x = logits / np.float32(temperature)
    size = x.size

    if top_k and top_k < size:
        # Keep every token tied at the cut, as a vocabulary-wide mask does.
        cut = np.partition(x, -top_k)[-top_k]
        cand = np.flatnonzero(x >= cut)
        order = np.argsort(-x[cand])
        kept, probs = _top_p(_softmax(x[cand][order].astype(np.float64)), top_p)
        return int(rng.choice(cand[order[kept]], p=probs))

    if top_p < 1.0 and size > _SAMPLE_CANDIDATES:
        e = np.exp(x.astype(np.float64) - float(x.max()))
        total = e.sum()
        cand = np.argpartition(e, -_SAMPLE_CANDIDATES)[-_SAMPLE_CANDIDATES:]
        probs = e[cand] / total
        order = np.argsort(-probs)
        kept, normalised = _top_p(probs[order], top_p)
        if kept.size < order.size:
            return int(rng.choice(cand[order[kept]], p=normalised))
        # Top-p needs ranks outside the slice: the full distribution is already exponentiated.
        probs = e / total
    else:
        probs = _softmax(x.astype(np.float64))

    if top_p >= 1.0:
        return int(rng.choice(size, p=probs))
    order = np.argsort(-probs)
    kept, normalised = _top_p(probs[order], top_p)
    return int(rng.choice(order[kept], p=normalised))


def _top_p(sorted_probs: np.ndarray, top_p: float) -> tuple[np.ndarray, np.ndarray]:
    """Indices and renormalised probabilities of the top-p set of `sorted_probs` (descending)."""
    if top_p >= 1.0:
        return np.arange(sorted_probs.size), sorted_probs
    keep = np.cumsum(sorted_probs) <= top_p
    keep[0] = True
    probs = sorted_probs[keep]
    return np.flatnonzero(keep), probs / probs.sum()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
