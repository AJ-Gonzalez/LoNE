"""ANE byte counters via IOReport (best effort, zero impact when unavailable).

On Apple Silicon the counters live in the `AMC Stats` group; channel identity is
`(group, name, unit)` and the ANE channels are `ANE RD` / `ANE WR` with unit `B`. Reading a sample
is a documented-by-observation private API: `libIOReport.dylib` is dyld-cache only on macOS 26, and
the subscription handle is returned by `IOReportCreateSubscription`, not by its out-parameter.
"""

from __future__ import annotations

import ctypes
import platform
from ctypes import POINTER, c_int64, c_uint32, c_uint64, c_void_p, byref

GROUP_CANDIDATES = ("AMC Stats", "AMC0 Stats")
UTF8 = 0x08000100
CHANNELS_KEY = "IOReportChannels"
_lib = None
_cf = None


def available() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _bind():
    global _lib, _cf
    if _lib is not None:
        return _lib, _cf
    _cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    _lib = ctypes.CDLL("libIOReport.dylib")

    _cf.CFStringCreateWithCString.restype = c_void_p
    _cf.CFStringCreateWithCString.argtypes = (c_void_p, ctypes.c_char_p, c_uint32)
    _cf.CFStringGetCString.restype = ctypes.c_bool
    _cf.CFStringGetCString.argtypes = (c_void_p, ctypes.c_char_p, c_int64, c_uint32)
    _cf.CFArrayGetCount.restype = c_int64
    _cf.CFArrayGetCount.argtypes = (c_void_p,)
    _cf.CFArrayGetValueAtIndex.restype = c_void_p
    _cf.CFArrayGetValueAtIndex.argtypes = (c_void_p, c_int64)
    _cf.CFDictionaryGetValue.restype = c_void_p
    _cf.CFDictionaryGetValue.argtypes = (c_void_p, c_void_p)
    _cf.CFRelease.restype = None
    _cf.CFRelease.argtypes = (c_void_p,)

    _lib.IOReportCopyChannelsInGroup.restype = c_void_p
    _lib.IOReportCopyChannelsInGroup.argtypes = (c_void_p, c_void_p, c_uint64, c_uint64, c_uint64)
    _lib.IOReportCreateSubscription.restype = c_void_p
    _lib.IOReportCreateSubscription.argtypes = (c_void_p, c_void_p, POINTER(c_void_p), c_uint64, c_void_p)
    _lib.IOReportCreateSamples.restype = c_void_p
    _lib.IOReportCreateSamples.argtypes = (c_void_p, c_void_p, c_void_p)
    _lib.IOReportCreateSamplesDelta.restype = c_void_p
    _lib.IOReportCreateSamplesDelta.argtypes = (c_void_p, c_void_p, c_void_p)
    for fn in ("IOReportChannelGetGroup", "IOReportChannelGetChannelName", "IOReportChannelGetUnitLabel"):
        getattr(_lib, fn).restype = c_void_p
        getattr(_lib, fn).argtypes = (c_void_p,)
    _lib.IOReportSimpleGetIntegerValue.restype = c_int64
    _lib.IOReportSimpleGetIntegerValue.argtypes = (c_void_p, ctypes.c_int)
    return _lib, _cf


def _cfstr(text: str) -> c_void_p:
    return _cf.CFStringCreateWithCString(None, text.encode(), UTF8)


def _to_str(ref: c_void_p) -> str:
    if not ref:
        return ""
    buf = ctypes.create_string_buffer(256)
    if _cf.CFStringGetCString(ref, buf, len(buf), UTF8):
        return buf.value.decode(errors="replace")
    return ""


class _Subscription:
    def __init__(self, group: str):
        lib, _ = _bind()
        self._channels = lib.IOReportCopyChannelsInGroup(_cfstr(group), None, 0, 0, 0)
        if not self._channels:
            raise RuntimeError(f"no channels for group {group!r}")
        self._sub = lib.IOReportCreateSubscription(None, self._channels, byref(c_void_p()), 0, None)
        if not self._sub:
            raise RuntimeError(f"subscription failed for group {group!r}")

    def sample(self) -> c_void_p:
        return _lib.IOReportCreateSamples(self._sub, self._channels, None)

    def delta(self, before: c_void_p, after: c_void_p) -> c_void_p:
        return _lib.IOReportCreateSamplesDelta(before, after, None)


def _rows(sample: c_void_p) -> list[dict]:
    array = _cf.CFDictionaryGetValue(sample, _cfstr(CHANNELS_KEY))
    if not array:
        return []
    out = []
    for i in range(_cf.CFArrayGetCount(array)):
        item = _cf.CFArrayGetValueAtIndex(array, i)
        out.append(
            {
                "group": _to_str(_lib.IOReportChannelGetGroup(item)),
                "name": _to_str(_lib.IOReportChannelGetChannelName(item)),
                "unit": _to_str(_lib.IOReportChannelGetUnitLabel(item)),
                "value": _lib.IOReportSimpleGetIntegerValue(item, 0),
            }
        )
    return out


class ANECounters:
    """Cumulative ANE read/write bytes, or None when the OS/group is unavailable."""

    def __init__(self):
        self._sub = None
        if not available():
            return
        for group in GROUP_CANDIDATES:
            try:
                self._sub = _Subscription(group)
                names = {(r["name"], r["unit"]) for r in _rows(self._sub.sample())}
                if not any(name.startswith("ANE") and unit == "B" for name, unit in names):
                    self._sub = None
                    continue
                self._group = group
                break
            except Exception:
                self._sub = None

    @property
    def ok(self) -> bool:
        return self._sub is not None

    def read(self) -> dict:
        if self._sub is None:
            return {}
        rows = _rows(self._sub.sample())
        return {r["name"]: r["value"] for r in rows if r["group"] == self._group and r["unit"] == "B"
                and r["name"].startswith("ANE")}

    def delta(self, before: dict, after: dict) -> dict:
        return {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)}
