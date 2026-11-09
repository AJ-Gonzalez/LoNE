import json
import os

import numpy as np
import pytest

from lone.config import ModelSpec, SpecError
from lone.layout import attention_mask, chunk_layout, cos_sin, split_layers

LFM3 = {
    "model_type": "lfm3",
    "hidden_size": 2048,
    "intermediate_size": 7680,
    "num_hidden_layers": 4,
    "layer_types": ["conv", "conv", "full_attention", "sliding_attention"],
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "conv_L_cache": 7,
    "norm_eps": 1e-5,
    "vocab_size": 128000,
    "tie_word_embeddings": True,
    "use_sandwich_norm": True,
    "sliding_window": 256,
    "rope_parameters": {
        "full_attention": {"rope_theta": 1e6, "partial_rotary_factor": 0.5},
        "sliding_attention": {"rope_theta": 1e4},
    },
}

LFM2 = {
    "model_type": "lfm2",
    "hidden_size": 1024,
    "intermediate_size": 2560,
    "num_hidden_layers": 4,
    "layer_types": ["conv", "conv", "full_attention", "full_attention"],
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "conv_L_cache": 3,
    "norm_eps": 1e-5,
    "vocab_size": 65536,
    "tie_word_embeddings": True,
    "rope_parameters": {"rope_theta": 1e6, "rope_type": "default"},
}


def write_model(tmp_path, cfg, gen=None):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    if gen is not None:
        (tmp_path / "generation_config.json").write_text(json.dumps(gen))
    return str(tmp_path)


def test_lfm3_spec(tmp_path):
    spec = ModelSpec.from_dir(write_model(tmp_path, LFM3, {"eos_token_id": [1, 2]}))
    assert spec.num_layers == 4
    assert spec.layer_ids("conv") == [0, 1]
    assert spec.layer_ids("sliding_attention") == [3]
    assert spec.kv_groups == 4
    assert spec.sandwich_norm and spec.attention_gate
    assert spec.qk_norm_prefix == "q_norm"
    assert spec.rope_for("full_attention").partial == 0.5
    assert spec.rope_for("sliding_attention").theta == 1e4
    assert spec.eos_token_ids == [1, 2]


def test_lfm2_spec(tmp_path):
    spec = ModelSpec.from_dir(write_model(tmp_path, LFM2))
    assert spec.sandwich_norm is False
    assert spec.attention_gate is False
    assert spec.qk_norm_prefix == "q_layernorm"
    assert spec.conv_kernel == 3
    assert spec.head_dim == 64  # derived from hidden / heads
    assert spec.rope_for("full_attention").partial == 1.0


def test_unsupported_and_invalid(tmp_path):
    with pytest.raises(SpecError):
        ModelSpec.from_dir(write_model(tmp_path, {**LFM2, "model_type": "llama"}))
    with pytest.raises(SpecError):
        ModelSpec.from_dir(write_model(tmp_path, {**LFM2, "layer_types": []}))
    with pytest.raises(SpecError):
        ModelSpec.from_dir(write_model(tmp_path, {**LFM2, "conv_bias": True}))


def test_spec_roundtrip(tmp_path):
    spec = ModelSpec.from_dir(write_model(tmp_path, LFM3))
    assert ModelSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


def test_split_layers():
    assert split_layers(5, 2) == [[0, 1], [2, 3], [4]]
    assert split_layers(3, 1) == [[0], [1], [2]]


def test_chunk_layout_names_are_unique(tmp_path):
    spec = ModelSpec.from_dir(write_model(tmp_path, LFM3))
    layout = chunk_layout(spec, [0, 1, 2, 3])
    assert len(set(layout["inputs"])) == len(layout["inputs"])
    assert len(set(layout["outputs"])) == len(layout["outputs"])
    assert not set(layout["inputs"]) & set(layout["outputs"])  # Core ML forbids this overlap
    assert layout["attention_layers"] == [2, 3]
    assert layout["conv_layers"] == [0, 1]
    assert layout["attention_kinds"] == ["full_attention", "sliding_attention"]


def test_mask_decode_ignores_unused_past():
    mask = attention_mask("full_attention", ctx=8, seq=1, pos=0)[0, 0, 0]
    assert list(np.where(mask == 0)[0]) == [8]  # only the new block, no zero-filled past rows

    mask = attention_mask("full_attention", ctx=8, seq=1, pos=5)[0, 0, 0]
    assert list(np.where(mask == 0)[0]) == [0, 1, 2, 3, 4, 8]


def test_mask_prefill_is_causal():
    mask = attention_mask("full_attention", ctx=4, seq=3, pos=0)
    rows = [np.where(mask[0, 0, i] == 0)[0].tolist() for i in range(3)]
    assert rows == [[4], [4, 5], [4, 5, 6]]


def test_sliding_mask_window():
    mask = attention_mask("sliding_attention", ctx=10, seq=1, pos=6, window=3)[0, 0, 0]
    # window 3 => positions 4, 5 and the new block
    assert list(np.where(mask == 0)[0]) == [4, 5, 10]


def test_cos_sin_zero_tail():
    inv = np.array([1.0, 0.0], np.float32)  # head_dim 4: one real frequency, one zero-tail
    cos, sin = cos_sin(inv, np.array([0, 1]))
    assert cos.shape == (1, 2, 4)
    # `cat(freqs, freqs)`, so the zero-frequency slot repeats at index 1 and 3 for head_dim 4.
    assert np.allclose(cos[0, :, [1, 3]], 1.0)  # zero frequencies are unrotated
    assert np.allclose(sin[0, :, [1, 3]], 0.0)
