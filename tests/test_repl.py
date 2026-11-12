"""REPL helpers: token-prefix bookkeeping and the per-turn stats block."""

import sys

import pytest

from lone.repl import common_prefix, format_stats


def test_common_prefix():
    assert common_prefix([], [1]) == 0
    assert common_prefix([1, 2, 3], [1, 2, 3]) == 3
    assert common_prefix([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix([1, 2], [1, 2, 3, 4]) == 2


def test_stats_report_rates_power_context_and_memory():
    lines = format_stats(
        prefill_tokens=10, prefill_s=0.2, prefill_w=0.25,
        decode_tokens=20, decode_s=1.0, decode_w=0.6,
        used=100, ctx=512, mem={"resident": 2**31, "free": 2**30}, ane_gb=4.1,
    )
    assert len(lines) == 2
    assert "prefill 10 tok 0.20 s 50.0 tok/s 0.25 W" in lines[0]
    assert "decode 20 tok 1.00 s 20.0 tok/s 0.60 W" in lines[0]
    assert "ctx 100/512 (19.5%)" in lines[1]
    assert "2.00 GB rss" in lines[1]
    assert "1.00 GB free" in lines[1]
    assert "ane io 4.10 GB" in lines[1]


def test_stats_omit_missing_power_memory_and_traffic():
    lines = format_stats(
        prefill_tokens=1, prefill_s=0.0, prefill_w=None,
        decode_tokens=0, decode_s=0.0, decode_w=None,
        used=0, ctx=128, mem={}, ane_gb=None,
    )
    assert "W" not in lines[0]
    assert lines[0].count("—") == 2
    assert "ctx 0/128 (0.0%)" in lines[1]
    assert "mem" not in lines[1] and "ane io" not in lines[1]


@pytest.mark.skipif(sys.platform != "darwin", reason="mach probes are macOS-only")
def test_memory_probe_reports_resident_and_free():
    from lone.metrics import memory

    mem = memory()
    assert mem["resident"] > 0
    assert mem["free"] > 0
