"""Shared interfaces for spatial-understanding benchmark adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationTask:
    """A lightweight, serializable unit used for deterministic rank sharding."""

    global_index: int
    eval_id: str
    source_index: int
    source_id: str
    category: str
    variant: str


@dataclass
class EvaluationSample:
    """A fully materialized model input for one evaluation task."""

    task: EvaluationTask
    images: list[Any]
    prompt: str
    target_letter: str
    target_text: str
    options: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(ABC):
    """Interface implemented by every supported understanding benchmark."""

    name: str
    protocol_version: str

    @property
    @abstractmethod
    def tasks(self) -> Sequence[EvaluationTask]:
        """Return every expanded evaluation task in canonical order."""

    @property
    @abstractmethod
    def manifest_fragment(self) -> Mapping[str, Any]:
        """Return immutable dataset/protocol metadata for resume protection."""

    @abstractmethod
    def materialize(self, task: EvaluationTask) -> EvaluationSample:
        """Decode images and build the exact prompt for ``task``."""

    @abstractmethod
    def score_response(
        self, sample: EvaluationSample, response: str
    ) -> Mapping[str, Any]:
        """Parse and score one model response."""

    @abstractmethod
    def aggregate(
        self, records_by_id: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        """Aggregate selected per-task records into benchmark metrics."""

    @property
    def required_ok_boolean_fields(self) -> Sequence[str]:
        """Boolean score fields required on every successful record.

        ``parse_success`` is shared by the current multiple-choice adapters;
        benchmark-specific correctness fields are declared by each adapter.
        Keeping this contract here avoids teaching the common result writer
        about SAT- or MMSI-specific metric names.
        """

        return ("parse_success",)

    def validate_record(
        self, task: EvaluationTask, record: Mapping[str, Any]
    ) -> Sequence[str]:
        """Return benchmark-specific integrity errors for a persisted record."""

        return ()

    def summary_lines(self, metrics: Mapping[str, Any]) -> Sequence[str]:
        """Format a compact terminal summary.

        Benchmark-specific adapters should override this when their metric
        schema has a useful human-readable presentation. Keeping formatting in
        the adapter lets new benchmarks plug into the common runner unchanged.
        """

        overall = metrics.get("overall", {})
        return (
            f"=== {self.name} summary ===",
            f"Complete: {metrics.get('complete', False)}",
            f"Total evaluation items: {overall.get('total', len(self.tasks))}",
        )
