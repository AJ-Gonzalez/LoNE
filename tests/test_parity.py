"""Conversion parity against the reference torch modules.

Skipped unless `LONE_TEST_MODEL` points at a local LFM checkpoint with `model.safetensors`.
Run it for LFM3 and LFM2.5 before trusting a bundle:

    LONE_TEST_MODEL=models/lfm3-1.7b-torch pytest -q
    LONE_TEST_MODEL=models/lfm2.5-230m    pytest -q
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lone.config import ModelSpec
from lone.convert import _read_layer, _trace_and_convert
from lone.layout import chunk_layout
from lone.torch_model import Chunk, example_inputs

MODEL = os.environ.get("LONE_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set LONE_TEST_MODEL to a checkpoint directory")


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize("seq", [1, 8])
def test_layer_conversion_parity(tmp_path, seq):
    from safetensors import safe_open

    spec = ModelSpec.from_dir(MODEL)
    ctx = 32
    kinds = {}
    for i in range(spec.num_layers):
        kinds.setdefault(spec.kind(i), i)

    with safe_open(os.path.join(MODEL, "model.safetensors"), framework="pt") as sd:
        for layer in kinds.values():
            params = {layer: _read_layer(sd, spec, layer, torch.float16)}
            module = Chunk(spec, params, [layer], seq, ctx)
            layout = chunk_layout(spec, [layer])
            example = example_inputs(spec, layout, seq, ctx)
            package = _trace_and_convert(
                module, layout["inputs"], layout["outputs"], example,
                f"layer_{layer}_s{seq}", str(tmp_path), "fp16",
            )

            import coremltools as ct

            runtime = ct.models.MLModel(package, compute_units=ct.ComputeUnit.CPU_AND_NE)
            torch.manual_seed(0)
            feed = {name: (torch.randn(*tuple(example[name].shape)) * 0.1).half() for name in layout["inputs"]}
            with torch.no_grad():
                expected = module(*[feed[name] for name in layout["inputs"]])
            got = runtime.predict({name: feed[name].numpy() for name in layout["inputs"]})
            for name, value in zip(layout["outputs"], expected):
                assert cosine(value.numpy(), got[name]) >= 0.999, f"{spec.kind(layer)} layer {layer} {name}"
