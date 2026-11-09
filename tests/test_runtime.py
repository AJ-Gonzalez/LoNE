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
