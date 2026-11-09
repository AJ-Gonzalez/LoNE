"""CLI option enums shared by `cli.py` and `verify.py`."""

from __future__ import annotations

from enum import Enum


class ComputeUnits(str, Enum):
    cpu_and_ne = "cpu_and_ne"
    cpu_and_gpu = "cpu_and_gpu"
    cpu_only = "cpu_only"
    all = "all"


class Precision(str, Enum):
    fp16 = "fp16"
    lut4 = "lut4"
    lut6 = "lut6"
    lut8 = "lut8"
