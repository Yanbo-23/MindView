"""Registry for spatial-understanding benchmark adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .base import BenchmarkAdapter
from .mindcube import MindCubeAdapter, add_mindcube_arguments
from .mmsi import MmsiBenchmark, add_mmsi_arguments
from .sat_v2 import SatV2Benchmark, add_sat_v2_arguments


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    adapter_class: type[BenchmarkAdapter]
    add_arguments: Callable[[Any], None]
    default_max_new_tokens: int


SAT_V2_SPEC = BenchmarkSpec(
    name="sat-v2",
    adapter_class=SatV2Benchmark,
    add_arguments=add_sat_v2_arguments,
    default_max_new_tokens=1024,
)

MMSI_SPEC = BenchmarkSpec(
    name="mmsi",
    adapter_class=MmsiBenchmark,
    add_arguments=add_mmsi_arguments,
    default_max_new_tokens=2048,
)

MINDCUBE_TINY_SPEC = BenchmarkSpec(
    name="mindcube-tiny",
    adapter_class=MindCubeAdapter,
    add_arguments=add_mindcube_arguments,
    default_max_new_tokens=4096,
)

BENCHMARKS = {
    "mindcube-tiny": MINDCUBE_TINY_SPEC,
    "mmsi": MMSI_SPEC,
    "sat-v2": SAT_V2_SPEC,
}


def get_benchmark_spec(name: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported benchmark {name!r}; choose from {sorted(BENCHMARKS)}"
        ) from exc


__all__ = [
    "BENCHMARKS",
    "BenchmarkAdapter",
    "BenchmarkSpec",
    "MindCubeAdapter",
    "MmsiBenchmark",
    "SatV2Benchmark",
    "get_benchmark_spec",
]
