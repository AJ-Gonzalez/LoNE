"""Cache wiring: the mlock span must cover exactly the array's page-aligned range."""

import os
import sys

import numpy as np
import pytest


def test_wire_pages_covers_the_array():
    if sys.platform != "darwin":
        pytest.skip("mlock cache wiring is macOS-only")
    from lone.runtime import _wire_pages

    page = os.sysconf("SC_PAGE_SIZE")
    arr = np.ones(3 * page + 17, dtype=np.uint8)
    span = _wire_pages(arr)
    assert span is not None, "mlock refused a small array"
    start, length = span
    assert start % page == 0 and length % page == 0
    assert start <= arr.ctypes.data
    assert start + length >= arr.ctypes.data + arr.nbytes
    assert length - arr.nbytes < page, "span must not extend past one trailing page"


def test_wire_pages_is_unavailable_off_darwin(monkeypatch):
    import lone.runtime as runtime

    monkeypatch.setattr(runtime.sys, "platform", "linux")
    assert runtime._wire_pages(np.zeros(16, dtype=np.uint8)) is None


def test_store_wraps_a_ring_buffer():
    from lone.runtime import LoneModel

    cache = {0: np.zeros((1, 2, 4, 3), dtype=np.float16)}
    new = np.arange(2 * 3 * 3, dtype=np.float16).reshape(2, 3, 3)
    LoneModel._store(cache, 0, new, pos=2, seq=3)  # absolute 2,3,4 -> rows 2,3,0
    assert cache[0][0, :, 2].tolist() == new[:, 0].tolist()
    assert cache[0][0, :, 3].tolist() == new[:, 1].tolist()
    assert cache[0][0, :, 0].tolist() == new[:, 2].tolist()
    assert cache[0][0, :, 1].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_store_keeps_only_the_last_rows_when_seq_exceeds_the_ring():
    from lone.runtime import LoneModel

    cache = {0: np.zeros((1, 1, 3, 2), dtype=np.float16)}
    new = np.arange(5 * 2, dtype=np.float16).reshape(1, 5, 2)
    LoneModel._store(cache, 0, new, pos=0, seq=5)  # absolute 2,3,4 survive at rows 2,0,1
    assert cache[0][0, 0, 2].tolist() == new[0, 2].tolist()
    assert cache[0][0, 0, 0].tolist() == new[0, 3].tolist()
    assert cache[0][0, 0, 1].tolist() == new[0, 4].tolist()
