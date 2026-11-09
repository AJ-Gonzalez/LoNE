"""Normalised model description shared by the converter and the runtime.

The converter never imports `transformers`: everything it needs is in `config.json`, and the
architecture differences between LFM3 and LFM2.5 are captured by three flags — `sandwich_norm`,
`attention_gate` and per-layer RoPE parameters.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

ATTENTION_KINDS = ("full_attention", "sliding_attention")
CONV = "conv"


class SpecError(ValueError):
    pass


@dataclass
class RopeSpec:
    theta: float
    partial: float = 1.0


@dataclass
class ModelSpec:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    layer_types: list[str]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    conv_kernel: int
    norm_eps: float
    vocab_size: int
    tie_word_embeddings: bool
    sandwich_norm: bool
    attention_gate: bool
    qk_norm_prefix: str
    rope: dict[str, RopeSpec]
    sliding_window: int | None = None
    eos_token_ids: list[int] = field(default_factory=list)
    bos_token_id: int | None = None
    conv_bias: bool = False

    # ---------------------------------------------------------------- helpers
    @property
    def kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    def kind(self, layer: int) -> str:
        return self.layer_types[layer]

    def is_attention(self, layer: int) -> bool:
        return self.layer_types[layer] in ATTENTION_KINDS

    def rope_for(self, kind: str) -> RopeSpec:
        try:
            return self.rope[kind]
        except KeyError:  # pragma: no cover - guarded at load time
            raise SpecError(f"no RoPE parameters for layer kind {kind!r}") from None

    def layer_ids(self, kind: str) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == kind]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rope"] = {k: asdict(v) for k, v in self.rope.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        d = dict(d)
        d["rope"] = {k: RopeSpec(**v) for k, v in d["rope"].items()}
        return cls(**d)

    # ------------------------------------------------------------------ load
    @classmethod
    def from_dir(cls, model_dir: str) -> "ModelSpec":
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)

        family = cfg.get("model_type")
        if family not in ("lfm3", "lfm2"):
            raise SpecError(
                f"unsupported model_type {family!r}; LoNE converts LFM3 (lfm3) and LFM2.5 (lfm2)"
            )

        layer_types = cfg.get("layer_types")
        if not layer_types:
            raise SpecError("config.json has no layer_types; cannot place conv/attention layers")
        bad = set(layer_types) - {CONV, *ATTENTION_KINDS}
        if bad:
            raise SpecError(f"unknown layer types {sorted(bad)}")

        num_heads = int(cfg["num_attention_heads"])
        num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))
        head_dim = int(cfg.get("head_dim") or cfg["hidden_size"] // num_heads)

        if family == "lfm3":
            sandwich = bool(cfg.get("use_sandwich_norm", True))
            gate = True
            prefix = "q_norm"
            rope_raw = cfg.get("rope_parameters") or {}
            rope = {}
            for kind in sorted({t for t in layer_types if t in ATTENTION_KINDS}):
                params = rope_raw.get(kind)
                if params is None:
                    raise SpecError(f"rope_parameters has no entry for {kind}")
                rope[kind] = RopeSpec(
                    theta=float(params.get("rope_theta", 10000.0)),
                    partial=float(params.get("partial_rotary_factor", 1.0)),
                )
            sliding = cfg.get("sliding_window")
        else:  # lfm2
            sandwich = bool(cfg.get("use_sandwich_norm", False))
            gate = False
            prefix = "q_layernorm"
            rope_raw = cfg.get("rope_parameters") or {}
            theta = float(rope_raw.get("rope_theta", cfg.get("rope_theta", 10000.0)))
            partial = float(rope_raw.get("partial_rotary_factor", 1.0))
            rope = {kind: RopeSpec(theta=theta, partial=partial) for kind in ATTENTION_KINDS}
            sliding = cfg.get("sliding_window")

        if any(t == "sliding_attention" for t in layer_types) and not sliding:
            raise SpecError("sliding_attention layers present but config has no sliding_window")
        if cfg.get("conv_bias"):
            raise SpecError("conv_bias=true is not implemented")

        eos = []
        gen_path = os.path.join(model_dir, "generation_config.json")
        if os.path.exists(gen_path):
            with open(gen_path) as f:
                gen = json.load(f)
            eos = gen.get("eos_token_id") or []
            if isinstance(eos, int):
                eos = [eos]
        cfg_eos = cfg.get("eos_token_id")
        if isinstance(cfg_eos, int):
            cfg_eos = [cfg_eos]
        eos = list(dict.fromkeys([*(eos or []), *(cfg_eos or [])]))

        spec = cls(
            model_type=family,
            hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg.get("intermediate_size") or cfg.get("block_ff_dim")),
            num_layers=len(layer_types),
            layer_types=list(layer_types),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            conv_kernel=int(cfg["conv_L_cache"]),
            norm_eps=float(cfg.get("norm_eps", cfg.get("block_norm_eps", 1e-5))),
            vocab_size=int(cfg["vocab_size"]),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", cfg.get("tie_embedding", True))),
            sandwich_norm=sandwich,
            attention_gate=gate,
            qk_norm_prefix=prefix,
            rope=rope,
            sliding_window=sliding,
            eos_token_ids=eos,
            bos_token_id=cfg.get("bos_token_id"),
            conv_bias=bool(cfg.get("conv_bias", False)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.num_heads % self.num_kv_heads:
            raise SpecError("num_attention_heads must be a multiple of num_key_value_heads")
        for i in range(self.num_layers):
            if self.is_attention(i) and self.kind(i) not in self.rope:
                raise SpecError(f"layer {i} kind {self.kind(i)} has no RoPE spec")
        if self.conv_kernel < 2:
            raise SpecError(f"conv kernel {self.conv_kernel} is not causal")
