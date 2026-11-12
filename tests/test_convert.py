"""Head slicing is a size decision; these are the sizes that matter."""

import pytest

from lone.convert import HEAD_GRAPH_BYTES, head_chunk_count


def test_slices_fit_the_budget():
    # 128k vocab, hidden 2048, fp16: 524 MB of head weights -> four ~131 MB graphs.
    assert head_chunk_count(128000, 2048) == 4
    # 65k vocab, hidden 1024: 134 MB total, one graph at the budget.
    assert head_chunk_count(65536, 1024) == 1
    # Wider rows need proportionally more slices.
    assert head_chunk_count(128000, 4096) == 8


def test_palettized_weights_need_fewer_slices():
    assert head_chunk_count(128000, 2048, "lut8") == 2
    assert head_chunk_count(128000, 2048, "lut6") == 2
    assert head_chunk_count(128000, 2048, "lut4") == 1


def test_slice_weights_stay_within_the_budget():
    for vocab, hidden, precision in ((128000, 2048, "fp16"), (65536, 1024, "fp16"),
                                     (128000, 2048, "lut6"), (32000, 512, "fp16")):
        n = head_chunk_count(vocab, hidden, precision)
        rows = -(-vocab // n)
        bits = {"fp16": 16, "lut8": 8, "lut6": 6, "lut4": 4}[precision]
        assert rows * hidden * bits // 8 <= HEAD_GRAPH_BYTES


def test_small_models_use_one_graph():
    assert head_chunk_count(32000, 512) == 1


def test_rejects_empty_shapes():
    with pytest.raises(ValueError):
        head_chunk_count(0, 2048)
    with pytest.raises(ValueError):
        head_chunk_count(128000, 0)
