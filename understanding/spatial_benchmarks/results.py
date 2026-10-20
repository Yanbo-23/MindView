"""Crash-tolerant JSONL shards, manifest protection, and result merging."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typing_extensions import Self

from .base import BenchmarkAdapter

MANIFEST_SCHEMA = "bagel_spatial_eval_manifest_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent)


def spec_fingerprint(spec: Mapping[str, Any]) -> str:
    canonical = _json_dumps(spec).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, _json_dumps(value, indent=2) + "\n")


def ensure_manifest(output_dir: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_manifest.json"
    expected_fingerprint = spec_fingerprint(spec)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("schema_version") != MANIFEST_SCHEMA:
            raise RuntimeError(f"Unknown evaluation manifest schema in {path}")
        if existing.get("spec") != spec:
            old = _json_dumps(existing.get("spec"), indent=2)
            new = _json_dumps(spec, indent=2)
            raise RuntimeError(
                f"Evaluation settings differ from the frozen manifest {path}.\n"
                f"Existing:\n{old}\nRequested:\n{new}\n"
                "Choose a new --output-dir for a different checkpoint/protocol."
            )
        if existing.get("spec_fingerprint") != expected_fingerprint:
            raise RuntimeError(f"Invalid spec fingerprint in {path}")
        return existing

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at_utc": utc_now(),
        "spec_fingerprint": expected_fingerprint,
        "spec": dict(spec),
    }
    atomic_write_json(path, manifest)
    return manifest


def append_launch(output_dir: Path, launch: Mapping[str, Any]) -> None:
    path = output_dir / "launches.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json_dumps(dict(launch)) + "\n")
        handle.flush()


class OutputRunLock:
    """Hold an advisory exclusive lock for one output directory."""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / ".eval.lock"
        self.handle: Any | None = None

    def acquire(self) -> dict[str, Any]:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            handle.close()
            raise RuntimeError(
                f"Another evaluation is writing {self.path.parent}; lock owner: {owner}"
            ) from exc

        owner = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at_utc": utc_now(),
        }
        handle.seek(0)
        handle.truncate()
        handle.write(_json_dumps(owner) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        self.handle = handle
        return owner

    def release(self) -> None:
        if self.handle is None:
            return
        import fcntl

        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


def _repair_and_read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    byte_offset = 0
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        next_offset = byte_offset + len(raw_line)
        if not stripped:
            byte_offset = next_offset
            continue
        try:
            records.append(json.loads(stripped.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_truncated_tail = index == len(lines) - 1 and not data.endswith(b"\n")
            if not is_truncated_tail:
                raise RuntimeError(
                    f"Corrupt JSONL record in {path}:{index + 1}"
                ) from exc
            backup = path.with_name(f"{path.name}.truncated.{os.getpid()}")
            backup.write_bytes(data[byte_offset:])
            with path.open("r+b") as handle:
                handle.truncate(byte_offset)
                handle.flush()
                os.fsync(handle.fileno())
            break
        byte_offset = next_offset
    return records


def shard_paths(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "shards").glob("rank-*.jsonl"))


def collect_selected_records(
    output_dir: Path,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Select one record per eval_id, preferring any successful attempt."""

    selected: dict[str, dict[str, Any]] = {}
    duplicate_attempts = 0
    for path in shard_paths(output_dir):
        for record in _repair_and_read_jsonl(path):
            eval_id = record.get("eval_id")
            if not isinstance(eval_id, str) or not eval_id:
                raise RuntimeError(f"Record without eval_id in {path}")
            old = selected.get(eval_id)
            if old is not None:
                duplicate_attempts += 1
            if old is None or old.get("status") != "ok" or record.get("status") == "ok":
                selected[eval_id] = record
    return selected, duplicate_attempts


def _validate_selected_records(
    selected: Mapping[str, Mapping[str, Any]],
    adapter: BenchmarkAdapter,
    *,
    run_fingerprint: str,
    record_schema: str,
    checkpoint_path: str,
) -> None:
    tasks_by_id = {task.eval_id: task for task in adapter.tasks}
    errors: list[str] = []
    for eval_id, record in selected.items():
        task = tasks_by_id.get(eval_id)
        if task is None:
            errors.append(f"{eval_id}: eval_id is outside the benchmark manifest")
            continue
        expected_fields = {
            "schema_version": record_schema,
            "run_fingerprint": run_fingerprint,
            "eval_id": task.eval_id,
            "global_index": task.global_index,
            "source_index": task.source_index,
            "source_id": task.source_id,
            "category": task.category,
            "variant": task.variant,
            "checkpoint_path": checkpoint_path,
        }
        for key, expected in expected_fields.items():
            if record.get(key) != expected:
                errors.append(f"{eval_id}: invalid {key}")

        status = record.get("status")
        if status not in {"ok", "inference_error"}:
            errors.append(f"{eval_id}: invalid status {status!r}")
        elif status == "ok":
            if not isinstance(record.get("raw_response"), str):
                errors.append(f"{eval_id}: ok record has no string raw_response")
            elapsed = record.get("elapsed_seconds")
            if (
                not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool)
                or elapsed < 0
            ):
                errors.append(f"{eval_id}: ok record has invalid elapsed_seconds")
            for key in adapter.required_ok_boolean_fields:
                if not isinstance(record.get(key), bool):
                    errors.append(f"{eval_id}: ok record has invalid {key}")
        else:
            if not isinstance(record.get("error_type"), str):
                errors.append(f"{eval_id}: error record has no error_type")
            if not isinstance(record.get("error"), str):
                errors.append(f"{eval_id}: error record has no error message")

        errors.extend(
            f"{eval_id}: {message}" for message in adapter.validate_record(task, record)
        )

    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n  ... {len(errors) - 20} more"
        raise RuntimeError(
            "Persisted result shards do not match the frozen run manifest:\n"
            f"{preview}{suffix}\nUse a new --output-dir or repair the invalid shards."
        )


def completed_eval_ids(
    output_dir: Path,
    adapter: BenchmarkAdapter,
    *,
    run_fingerprint: str,
    record_schema: str,
    checkpoint_path: str,
) -> list[str]:
    selected, _ = collect_selected_records(output_dir)
    _validate_selected_records(
        selected,
        adapter,
        run_fingerprint=run_fingerprint,
        record_schema=record_schema,
        checkpoint_path=checkpoint_path,
    )
    return sorted(
        eval_id for eval_id, record in selected.items() if record.get("status") == "ok"
    )


class JsonlShardWriter:
    def __init__(self, output_dir: Path, rank: int, *, fsync_every: int) -> None:
        if fsync_every <= 0:
            raise ValueError("fsync_every must be positive")
        self.path = output_dir / "shards" / f"rank-{rank:05d}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Rank 0 repairs all existing shards before workers are released.
        self.handle = self.path.open("a", encoding="utf-8")
        self.fsync_every = fsync_every
        self.pending = 0

    def write(self, record: Mapping[str, Any]) -> None:
        self.handle.write(_json_dumps(dict(record)) + "\n")
        self.handle.flush()
        self.pending += 1
        if self.pending >= self.fsync_every:
            os.fsync(self.handle.fileno())
            self.pending = 0

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def finalize_results(
    output_dir: Path,
    adapter: BenchmarkAdapter,
    *,
    run_fingerprint: str,
    record_schema: str,
    checkpoint_path: str,
    partial: bool = False,
    source_limit: int | None = None,
) -> dict[str, Any]:
    selected, duplicate_attempts = collect_selected_records(output_dir)
    _validate_selected_records(
        selected,
        adapter,
        run_fingerprint=run_fingerprint,
        record_schema=record_schema,
        checkpoint_path=checkpoint_path,
    )
    expected_ids = {task.eval_id for task in adapter.tasks}

    ordered_records = [
        selected[task.eval_id] for task in adapter.tasks if task.eval_id in selected
    ]
    merged_path = output_dir / "predictions.jsonl"
    atomic_write_text(
        merged_path,
        "".join(_json_dumps(record) + "\n" for record in ordered_records),
    )

    metrics = dict(adapter.aggregate(selected))
    overall = metrics["overall"]
    complete = (
        overall["missing"] == 0
        and overall["inference_errors"] == 0
        and overall["completed_responses"] == overall["total"]
    )
    metrics.update(
        {
            "complete": complete,
            "partial": partial,
            "source_limit": source_limit,
            "finalized_at_utc": utc_now(),
            "duplicate_attempts": duplicate_attempts,
            "num_merged_records": len(ordered_records),
            "expected_eval_ids": len(expected_ids),
            "run_fingerprint": run_fingerprint,
        }
    )
    atomic_write_json(output_dir / "metrics.json", metrics)
    success_path = output_dir / "_SUCCESS"
    if complete:
        atomic_write_text(success_path, f"{utc_now()}\n")
    elif success_path.exists():
        raise RuntimeError(
            f"Stale _SUCCESS exists even though the merged run is incomplete: {success_path}"
        )
    return metrics


__all__ = [
    "JsonlShardWriter",
    "OutputRunLock",
    "append_launch",
    "completed_eval_ids",
    "ensure_manifest",
    "finalize_results",
    "spec_fingerprint",
    "utc_now",
]
