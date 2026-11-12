"""`sample` must draw from the same distribution as the vocabulary-wide top-p computation."""

import numpy as np
import pytest

from lone.runtime import sample

VOCAB = 128000


class RecordingRng:
    """Captures the support `sample` hands to `Generator.choice` and returns its first entry."""

    def __init__(self):
        self.indices = None
        self.probs = None

    def choice(self, indices, p=None, **kwargs):
        # `sample` passes either an index array or a vocabulary size.
        self.indices = np.arange(int(indices)) if np.ndim(indices) == 0 else np.asarray(indices)
        self.probs = np.asarray(p, dtype=np.float64)
        return int(self.indices[0])


def reference_support(logits, temperature, top_p, top_k):
    """The pre-prefilter algorithm: mask, sort the whole vocabulary, cumulate, renormalise."""
    x = logits.astype(np.float64) / float(temperature)
    if top_k and top_k < x.size:
        cut = np.partition(x, -top_k)[-top_k]
        x = np.where(x < cut, -np.inf, x)
    order = np.argsort(-x)
    probs = np.exp(x[order] - x[order].max())
    probs /= probs.sum()
    if top_p >= 1.0:
        return {i: q for i, q in zip(order.tolist(), probs.tolist()) if q > 0}
    keep = np.cumsum(probs) <= top_p
    keep[0] = True
    kept, probs = order[keep], probs[keep]
    return dict(zip(kept.tolist(), (probs / probs.sum()).tolist()))


def support_of(logits, temperature=0.7, top_p=1.0, top_k=0):
    rng = RecordingRng()
    sample(logits, temperature, top_p, top_k, rng)
    return dict(zip(rng.indices.tolist(), rng.probs.tolist()))


@pytest.fixture(scope="module")
def logits():
    rng = np.random.default_rng(0)
    peaked = rng.standard_normal(VOCAB).astype(np.float32) * 3
    peaked[:50] += 8.0
    flat = (rng.standard_normal(VOCAB) * 0.01).astype(np.float32)
    return peaked, flat


@pytest.mark.parametrize("temperature,top_p,top_k", [
    (0.7, 1.0, 0), (0.7, 0.95, 0), (1.0, 0.5, 0), (0.7, 1.0, 40), (0.7, 0.9, 40),
])
def test_support_matches_the_vocabulary_wide_computation(logits, temperature, top_p, top_k):
    for vector in logits:
        expected = reference_support(vector, temperature, top_p, top_k)
        actual = support_of(vector, temperature, top_p, top_k)
        assert actual.keys() == expected.keys()
        assert max(abs(actual[i] - expected[i]) for i in expected) < 1e-6


def test_tail_beyond_the_prefilter_falls_back(logits):
    flat = logits[1]
    # 1024 uniform-ish candidates hold far less than 0.9999 of the mass, so top-p needs the tail.
    expected = reference_support(flat, 1.0, 0.9999, 0)
    actual = support_of(flat, 1.0, 0.9999, 0)
    assert len(actual) > 1024
    assert actual.keys() == expected.keys()


def test_top_k_keeps_ties_at_the_cut():
    logits = np.full(64, -10.0, dtype=np.float32)
    logits[:5] = 3.0
    actual = support_of(logits, 0.7, 1.0, 2)
    assert set(actual) == {0, 1, 2, 3, 4}


def test_greedy_ignores_sampling_parameters():
    logits = np.zeros(32, dtype=np.float32)
    logits[7] = 5.0
    assert sample(logits, 0.0, 0.9, 5) == 7


def test_top_k_one_is_the_argmax():
    logits = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
    for seed in range(5):
        assert sample(logits, 1.0, 1.0, 1, np.random.default_rng(seed)) == 999
