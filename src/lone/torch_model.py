"""Static-shape TorchScript modules for conversion.

Everything here is a pure function of fixed tensors: no dynamic shapes, no `.shape` unpacking
(TorchScript lowers that to `aten::Int`, which coremltools cannot convert), no fp32 escapes. The
math mirrors `transformers`' LFM3 / LFM2 implementations, with the differences carried by the
spec: sandwich norms, the sigmoid attention gate, per-kind RoPE and the conv kernel width.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelSpec
from .layout import cache_len_for, chunk_layout


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = (x * x).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(var + eps))


# The ANE's linear cores lose precision when the products fall into fp16's subnormal range
# (LFM3's conv operator: input absmax 0.066 x weights ~0.02). A power-of-two rescale of the
# weights, undone on the output, is exact in real arithmetic and moves the accumulated terms
# back into the normal range: operator cosine 0.99916 -> 0.9999998 on real activations.
_SCALE = 16.0
_INV_SCALE = 1.0 / _SCALE
_WIDE = 1024


def scaled_weight(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Scale a wide reduction's weights by 2^4; return (weights, 1/scale) for the output."""
    if w.shape[1] > _WIDE:
        return (w * _SCALE).contiguous(), _INV_SCALE
    return w, 1.0


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU without the ANE's fused `silu` approximation.

    Core ML maps `aten::silu` (and `x * sigmoid(x)`) to an NE activation with ~1e-3 relative
    error; the exp form keeps the activation off that path (measured 0.99935 -> 0.999991 on a
    real LFM3 FFN).
    """
    return x / (1.0 + torch.exp(-x))


def rotate_half(x: torch.Tensor, half: int) -> torch.Tensor:
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class AttentionLayer(nn.Module):
    def __init__(self, spec: ModelSpec, params: dict, seq: int, cache_len: int, kind: str):
        super().__init__()
        self.seq = seq
        self.cache_len = cache_len
        self.heads = spec.num_heads
        self.kv = spec.num_kv_heads
        self.hd = spec.head_dim
        self.groups = spec.kv_groups
        self.scale = spec.head_dim**-0.5
        self.eps = spec.norm_eps
        self.kind = kind
        self.has_gate = spec.attention_gate

        self.wq, self.q_inv = scaled_weight(params["q_proj"])
        self.wk, self.k_inv = scaled_weight(params["k_proj"])
        self.wv, self.v_inv = scaled_weight(params["v_proj"])
        self.wo, self.o_inv = scaled_weight(params["o_proj"])
        self.qn = params[spec.qk_norm_prefix]
        self.kn = params["k_norm" if spec.qk_norm_prefix == "q_norm" else "k_layernorm"]
        if self.has_gate:
            self.wg, self.g_inv = scaled_weight(params["gate_proj"])

    def forward(self, x, cos, sin, mask, k_past, v_past):
        q = rms((F.linear(x, self.wq) * self.q_inv).view(1, self.seq, self.heads, self.hd),
                self.qn, self.eps).transpose(1, 2)
        k = rms((F.linear(x, self.wk) * self.k_inv).view(1, self.seq, self.kv, self.hd),
                self.kn, self.eps).transpose(1, 2)
        v = (F.linear(x, self.wv) * self.v_inv).view(1, self.seq, self.kv, self.hd).transpose(1, 2)

        half = self.hd // 2
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        q = q * c + rotate_half(q, half) * s
        k = k * c + rotate_half(k, half) * s

        keys = torch.cat((k_past, k), dim=2)
        vals = torch.cat((v_past, v), dim=2)
        total = self.cache_len + self.seq
        keys = keys[:, :, None].expand(1, self.kv, self.groups, total, self.hd).reshape(
            1, self.kv * self.groups, total, self.hd
        )
        vals = vals[:, :, None].expand(1, self.kv, self.groups, total, self.hd).reshape(
            1, self.kv * self.groups, total, self.hd
        )

        att = torch.matmul(q, keys.transpose(2, 3)) * self.scale + mask
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, vals).transpose(1, 2)  # (1, seq, heads, hd)
        if self.has_gate:
            out = out * torch.sigmoid(F.linear(x, self.wg) * self.g_inv).unsqueeze(-1)
        return F.linear(out.reshape(1, self.seq, self.heads * self.hd), self.wo) * self.o_inv, k, v


class ConvLayer(nn.Module):
    def __init__(self, spec: ModelSpec, params: dict, seq: int):
        super().__init__()
        self.seq = seq
        self.hidden = spec.hidden_size
        self.state_len = spec.conv_kernel - 1
        self.win, self.in_inv = scaled_weight(params["in_proj"])
        self.wout, self.out_inv = scaled_weight(params["out_proj"])
        self.kernel = params["conv"]  # (hidden, 1, K)

    def forward(self, x, state):
        bcx = (F.linear(x, self.win) * self.in_inv).transpose(1, 2)  # (1, 3H, T)
        b, c, g = bcx.chunk(3, dim=1)
        h = torch.cat((state, b * g), dim=-1)  # (1, H, T + K - 1)
        w = self.kernel[:, :, None, :]  # (H, 1, 1, K)
        y = F.conv2d(h.unsqueeze(2), w, groups=self.hidden)[:, :, 0, :]  # (1, H, T)
        out = F.linear((c * y).transpose(1, 2).contiguous(), self.wout) * self.out_inv
        return out, h[:, :, -self.state_len :]


class DecoderLayer(nn.Module):
    def __init__(self, spec: ModelSpec, params: dict, seq: int, cache_len: int, kind: str):
        super().__init__()
        self.eps = spec.norm_eps
        self.kind = kind
        self.is_attention = kind != "conv"
        if self.is_attention:
            self.op = AttentionLayer(spec, params, seq, cache_len, kind)
        else:
            self.op = ConvLayer(spec, params, seq)

        self.op_norm = params["operator_norm"]
        self.ffn_norm = params["ffn_norm"]
        self.post_op_norm = params.get("post_operator_norm") if spec.sandwich_norm else None
        self.post_ffn_norm = params.get("post_ffn_norm") if spec.sandwich_norm else None
        self.w1, self.w1_inv = scaled_weight(params["w1"])
        w2 = params["w2"]
        self.w2, self.w2_inv = scaled_weight(w2)
        self.w3, self.w3_inv = scaled_weight(params["w3"])
        # The ANE loses precision on a single wide output projection (k=6656 measured at 0.997
        # cosine versus torch, k=3328 at 0.99999). Split the reduction when the FFN is wide.
        self.ffn_split = 2 if spec.intermediate_size > 4096 else 1
        if self.ffn_split == 2:
            half = spec.intermediate_size // 2
            self.ffn_half = half
            # Scale the raw halves, not self.w2 (already scaled above).
            self.w2a, self.w2a_inv = scaled_weight(w2[:, :half].contiguous())
            self.w2b, self.w2b_inv = scaled_weight(w2[:, half:].contiguous())

    def forward(self, x, *op_args):
        residual = x
        h = rms(x, self.op_norm, self.eps)
        if self.is_attention:
            h, new_k, new_v = self.op(h, *op_args)
        else:
            h, new_state = self.op(h, *op_args)
        if self.post_op_norm is not None:
            h = rms(h, self.post_op_norm, self.eps)
        x = x + h

        ffn_in = rms(x, self.ffn_norm, self.eps)
        gate_in = F.linear(ffn_in, self.w1) * self.w1_inv
        gate = silu(gate_in) * (F.linear(ffn_in, self.w3) * self.w3_inv)
        if self.ffn_split == 2:
            f = (F.linear(gate[:, :, : self.ffn_half], self.w2a) * self.w2a_inv
                 + F.linear(gate[:, :, self.ffn_half :], self.w2b) * self.w2b_inv)
        else:
            f = F.linear(gate, self.w2) * self.w2_inv
        if self.post_ffn_norm is not None:
            f = rms(f, self.post_ffn_norm, self.eps)
        x = x + f
        return (x, new_k, new_v) if self.is_attention else (x, new_state)


class Chunk(nn.Module):
    """A contiguous group of decoder layers with explicit cache tensors."""

    def __init__(self, spec: ModelSpec, params_by_layer: dict, layer_ids: list[int], seq: int, ctx: int,
                 ring: bool = False):
        super().__init__()
        self.layout = chunk_layout(spec, layer_ids)
        self.input_names = self.layout["inputs"]
        self.output_names = self.layout["outputs"]
        self.kinds = {i: spec.kind(i) for i in layer_ids}
        self.layers = nn.ModuleDict(
            {str(i): DecoderLayer(spec, params_by_layer[i], seq,
                                  cache_len_for(spec, spec.kind(i), ctx, ring), spec.kind(i))
             for i in layer_ids}
        )
        self.layer_ids = list(layer_ids)

    def forward(self, *args):
        d = dict(zip(self.input_names, args))
        x = d["hidden"]
        out = {}
        for i in self.layer_ids:
            layer = self.layers[str(i)]
            if layer.is_attention:
                kind = self.kinds[i]
                x, out[f"k_{i}_out"], out[f"v_{i}_out"] = layer(
                    x, d[f"cos_{kind}"], d[f"sin_{kind}"], d[f"mask_{kind}"], d[f"k_{i}"], d[f"v_{i}"]
                )
            else:
                x, out[f"conv_{i}_out"] = layer(x, d[f"conv_{i}"])
        out["hidden_out"] = x
        return tuple(out[name] for name in self.output_names)


class Head(nn.Module):
    """Final RMSNorm plus one vocabulary slice of the (tied) LM head."""

    def __init__(self, spec: ModelSpec, embed: torch.Tensor, norm: torch.Tensor, start: int, end: int):
        super().__init__()
        self.eps = spec.norm_eps
        self.norm = norm
        self.weight = embed[start:end].contiguous()
        # Same wide-reduction limit as the FFN: hidden 2048 measured 0.992 cosine on one linear.
        self.split = 2 if spec.hidden_size > 1024 else 1
        if self.split == 2:
            half = spec.hidden_size // 2
            self.half = half
            self.wa, self.a_inv = scaled_weight(self.weight[:, :half].contiguous())
            self.wb, self.b_inv = scaled_weight(self.weight[:, half:].contiguous())
        else:
            self.wa, self.a_inv = scaled_weight(self.weight)
            self.half = 0

    def forward(self, x):
        h = rms(x, self.norm, self.eps)
        if self.split == 2:
            return (F.linear(h[:, :, : self.half], self.wa) * self.a_inv
                    + F.linear(h[:, :, self.half :], self.wb) * self.b_inv)
        return F.linear(h, self.wa) * self.a_inv


def example_inputs(spec: ModelSpec, layout: dict, seq: int, ctx: int, dtype=torch.float16,
                   ring: bool = False) -> dict:
    shapes = {"hidden": (1, seq, spec.hidden_size)}
    for kind in layout["attention_kinds"]:
        shapes[f"cos_{kind}"] = (1, seq, spec.head_dim)
        shapes[f"sin_{kind}"] = (1, seq, spec.head_dim)
        shapes[f"mask_{kind}"] = (1, 1, seq, cache_len_for(spec, kind, ctx, ring) + seq)
    for i in layout["attention_layers"]:
        n = cache_len_for(spec, spec.kind(i), ctx, ring)
        shapes[f"k_{i}"] = (1, spec.num_kv_heads, n, spec.head_dim)
        shapes[f"v_{i}"] = (1, spec.num_kv_heads, n, spec.head_dim)
    for i in layout["conv_layers"]:
        shapes[f"conv_{i}"] = (1, spec.hidden_size, spec.conv_kernel - 1)
    return {name: torch.zeros(shape, dtype=dtype) for name, shape in shapes.items()}
