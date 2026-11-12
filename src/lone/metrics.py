"""ANE byte counters via IOReport (best effort, zero impact when unavailable).

On Apple Silicon the counters live in the `AMC Stats` group; channel identity is
`(group, name, unit)` and the ANE channels are `ANE RD` / `ANE WR` with unit `B`. Reading a sample
is a documented-by-observation private API: `libIOReport.dylib` is dyld-cache only on macOS 26, and
the subscription handle is returned by `IOReportCreateSubscription`, not by its out-parameter.
"""

from __future__ import annotations

import ctypes
import os
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


# --------------------------------------------------------------------------- power and memory
POWER_GROUP = "PMP"
POWER_UNIT = "mJ"


class PowerCounters:
    """Per-rail energy counters (mJ) from the `PMP` group; None when unavailable.

    Channels are cumulative and system-wide: `ANE` is the engine rail LoNE drives, `DRAM` and the
    CPU rails are whole-machine and move with anything else running. Convert a window to watts with
    `watts(before, after, seconds)`.
    """

    def __init__(self):
        self._sub = None
        if not available():
            return
        try:
            sub = _Subscription(POWER_GROUP)
            rows = _rows(sub.sample())
            if not any(r["unit"] == POWER_UNIT and r["name"] == "ANE" for r in rows):
                return
            self._sub = sub
        except Exception:
            self._sub = None

    @property
    def ok(self) -> bool:
        return self._sub is not None

    def read(self) -> dict:
        if self._sub is None:
            return {}
        return {r["name"]: r["value"] for r in _rows(self._sub.sample())
                if r["unit"] == POWER_UNIT}

    def watts(self, before: dict, after: dict, seconds: float) -> dict:
        """Average watts per rail over a window; empty when the counters are unavailable."""
        if seconds <= 0 or not before or not after:
            return {}
        return {name: (after.get(name, 0) - before.get(name, 0)) / 1000.0 / seconds for name in before}


class _MachTaskBasicInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", c_uint64),
        ("resident_size", c_uint64),
        ("resident_size_max", c_uint64),
        ("user_time_sec", ctypes.c_int32),
        ("user_time_usec", ctypes.c_int32),
        ("system_time_sec", ctypes.c_int32),
        ("system_time_usec", ctypes.c_int32),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
    ]


class _VMStatistics64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", c_uint64),
        ("reactivations", c_uint64),
        ("pageins", c_uint64),
        ("pageouts", c_uint64),
        ("faults", c_uint64),
        ("cow_faults", c_uint64),
        ("lookups", c_uint64),
        ("hits", c_uint64),
        ("purges", c_uint64),
        ("purgeable_count", c_uint64),
        ("speculative_count", c_uint64),
        ("decompressions", c_uint64),
        ("compressions", c_uint64),
        ("swapins", c_uint64),
        ("swapouts", c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", c_uint64),
    ]


_MACH_TASK_BASIC_INFO = 20
_HOST_VM_INFO64 = 4


def memory() -> dict:
    """Process resident size and system memory state, in bytes; empty when unavailable."""
    if not available():
        return {}
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mach_task_self.restype = ctypes.c_uint32
    libc.mach_host_self.restype = ctypes.c_uint32
    page = os.sysconf("SC_PAGE_SIZE")

    info = _MachTaskBasicInfo()
    count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
    task = libc.mach_task_self()
    if libc.task_info(task, _MACH_TASK_BASIC_INFO, byref(info), byref(count)) != 0:
        return {}

    vm = _VMStatistics64()
    vm_count = ctypes.c_uint32(ctypes.sizeof(vm) // 4)
    if libc.host_statistics64(libc.mach_host_self(), _HOST_VM_INFO64, byref(vm), byref(vm_count)) != 0:
        return {"resident": info.resident_size}

    # Only fields whose offsets are verified against `vm_stat` on this machine: resident, peak and
    # free pages. Later vm_statistics64 fields drift between macOS releases.
    return {
        "resident": info.resident_size,
        "resident_peak": info.resident_size_max,
        "free": vm.free_count * page,
    }
