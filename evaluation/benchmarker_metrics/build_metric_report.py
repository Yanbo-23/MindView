#!/usr/bin/env python3
"""Build strictly validated, configuration-driven MindView paper reports.

The raw metric directories are read-only inputs.  Every reported value is
recomputed from per-sample ``results.json`` files after filtering against the
canonical benchmark JSONL.  Report tables are declared in a JSON config so new
models, metrics, aggregations, and table views do not require copying the
scoring logic.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "metric_report_config.json"
DEFAULT_DATASET = SCRIPT_DIR.parent / "data" / "merged_dataset_final.jsonl"
DEFAULT_METRICS_ROOT = SCRIPT_DIR.parents[1] / "outputs/metrics"
DEFAULT_OUTPUT = SCRIPT_DIR.parents[1] / "outputs/reports"


class ReportError(RuntimeError):
    """Raised when a paper value cannot be proven valid and comparable."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Tuple[Any, str]:
    if not path.is_file():
        raise ReportError(f"Required JSON file is missing: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"Invalid UTF-8 JSON in {path}: {exc}") from exc
    return value, sha256_bytes(payload)


def normalize(field: str, value: Any) -> Any:
    if field == "shot":
        try:
            return int(str(value).replace("shot-", ""))
        except (TypeError, ValueError) as exc:
            raise ReportError(f"Invalid shot value: {value!r}") from exc
    if field == "anchor_frame_id":
        text = "" if value is None else str(value).strip()
        return "" if text.lower() in {"none", "null"} else text
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value)
    return "" if value is None else str(value).strip()


def identity_key(record: Mapping[str, Any], fields: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(normalize(field, record.get(field)) for field in fields)


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"Non-numeric metric at {label}: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ReportError(f"Non-finite metric at {label}: {value!r}")
    return number


def arithmetic_mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ReportError("Cannot average an empty collection")
    return math.fsum(materialized) / len(materialized)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights) or not values:
        raise ReportError("Composite values and weights must be non-empty and aligned")
    denominator = math.fsum(weights)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ReportError("Composite weights must have a positive finite sum")
    return math.fsum(value * weight for value, weight in zip(values, weights)) / denominator


def is_subsequence(smaller: Sequence[Any], larger: Sequence[Any]) -> bool:
    iterator = iter(larger)
    return all(any(item == candidate for candidate in iterator) for item in smaller)


def trailing_integer(text: str) -> Tuple[str, int, int]:
    match = re.match(r"^(.*?)(\d+)$", text)
    if not match:
        raise ReportError(f"sample_id has no numeric suffix: {text!r}")
    digits = match.group(2)
    return match.group(1), int(digits), len(digits)


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"benchmark", "models", "metrics", "composites", "tables", "output"}
    missing = sorted(required.difference(config))
    if missing:
        raise ReportError(f"Report config is missing sections: {missing}")

    metrics = config["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ReportError("Config metrics must be a non-empty list")
    metric_names = [metric.get("name") for metric in metrics]
    if any(not name for name in metric_names) or len(metric_names) != len(set(metric_names)):
        raise ReportError("Metric names must be non-empty and unique")
    for metric in metrics:
        for field in ("source", "field", "scale", "raw_range", "direction"):
            if field not in metric:
                raise ReportError(f"Metric {metric['name']} is missing {field}")
        if metric["direction"] not in {"higher", "lower"}:
            raise ReportError(f"Invalid direction for metric {metric['name']}")
        bounds = metric["raw_range"]
        if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ReportError(f"Invalid raw_range for metric {metric['name']}")

    known_scores = set(metric_names)
    for composite in config["composites"]:
        name = composite.get("name")
        members = composite.get("members")
        weights = composite.get("weights")
        if not name or name in known_scores:
            raise ReportError(f"Invalid or duplicate composite name: {name!r}")
        if not isinstance(members, list) or any(member not in known_scores for member in members):
            raise ReportError(f"Composite {name} references unknown metrics")
        if not isinstance(weights, list) or len(weights) != len(members):
            raise ReportError(f"Composite {name} weights do not match its members")
        if composite.get("direction") not in {"higher", "lower"}:
            raise ReportError(f"Invalid direction for composite {name}")
        known_scores.add(name)

    table_names = [table.get("name") for table in config["tables"]]
    if any(not name for name in table_names) or len(table_names) != len(set(table_names)):
        raise ReportError("Table names must be non-empty and unique")
    primary_score = config["output"].get("primary_score")
    if primary_score and primary_score not in known_scores:
        raise ReportError(f"Output primary_score is unknown: {primary_score}")
    leader_table = config["output"].get("leader_table")
    if leader_table and leader_table not in table_names:
        raise ReportError(f"Output leader_table is unknown: {leader_table}")


def validate_dataset(
    path: Path, benchmark: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[Any, ...], Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    if not path.is_file():
        raise ReportError(f"Canonical benchmark JSONL is missing: {path}")
    payload = path.read_bytes()
    records: List[Dict[str, Any]] = []
    key_fields = list(benchmark["key_fields"])
    keys: set[Tuple[Any, ...]] = set()
    sample_ids: set[str] = set()
    shot_counts: Counter[int] = Counter()
    dataset_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    cell_counts: Counter[Tuple[int, str, str]] = Counter()

    for line_number, raw_line in enumerate(payload.splitlines(), 1):
        if not raw_line.strip():
            raise ReportError(f"Blank line in benchmark JSONL at line {line_number}")
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReportError(f"Invalid benchmark line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ReportError(f"Benchmark line {line_number} is not an object")
        missing = [field for field in key_fields if field not in record]
        if missing:
            raise ReportError(f"Benchmark line {line_number} is missing {missing}")

        key = identity_key(record, key_fields)
        sample_id = str(record.get("sample_id", "")).strip()
        if key in keys:
            raise ReportError(f"Duplicate benchmark key at line {line_number}: {key}")
        if not sample_id or sample_id in sample_ids:
            raise ReportError(f"Missing or duplicate sample_id at line {line_number}: {sample_id!r}")
        keys.add(key)
        sample_ids.add(sample_id)

        shot = normalize("shot", record.get("shot"))
        dataset = normalize("dataset", record.get("dataset"))
        task = normalize("task_type", record.get("task_type"))
        contexts = record.get("contexts_frame_id")
        variable_contexts = (
            benchmark.get("allow_variable_contexts", False) and shot == 10
            and isinstance(contexts, list) and 6 <= len(contexts) <= 10
        )
        if not isinstance(contexts, list) or (len(contexts) != shot and not variable_contexts):
            raise ReportError(
                f"Context count mismatch for {sample_id}: shot={shot}, "
                f"contexts={len(contexts) if isinstance(contexts, list) else 'invalid'}"
            )
        if not str(record.get("instruction", "")).strip():
            raise ReportError(f"Missing instruction for {sample_id}")
        records.append(record)
        shot_counts[shot] += 1
        dataset_counts[dataset] += 1
        task_counts[task] += 1
        cell_counts[(shot, dataset, task)] += 1

    expected_total = benchmark.get("expected_num_samples")
    if expected_total is not None and len(records) != int(expected_total):
        raise ReportError(f"Expected {expected_total} benchmark rows, found {len(records)}")

    shots = [int(value) for value in benchmark["shots"]]
    datasets = [str(value) for value in benchmark["datasets"]]
    tasks = [str(value) for value in benchmark["tasks"]]
    if set(shot_counts) != set(shots):
        raise ReportError(f"Shot set mismatch: expected={shots}, actual={sorted(shot_counts)}")
    if set(dataset_counts) != set(datasets):
        raise ReportError(
            f"Dataset set mismatch: expected={datasets}, actual={sorted(dataset_counts)}"
        )
    if set(task_counts) != set(tasks):
        raise ReportError(f"Task set mismatch: expected={tasks}, actual={sorted(task_counts)}")
    expected_per_shot = benchmark.get("expected_samples_per_shot")
    if expected_per_shot is not None:
        expected = Counter({shot: int(expected_per_shot) for shot in shots})
        if shot_counts != expected:
            raise ReportError(f"Per-shot sample count mismatch: {shot_counts}")
    for shot, dataset, task in itertools.product(shots, datasets, tasks):
        if not benchmark.get("require_full_cells", True):
            break
        if cell_counts[(shot, dataset, task)] <= 0:
            raise ReportError(f"Empty benchmark cell: C{shot}/{dataset}/{task}")

    if benchmark.get("require_adjacent_shot_groups", False):
        group_size = len(shots)
        if len(records) % group_size:
            raise ReportError("Benchmark row count is not divisible by the shot-group size")
        equal_fields = list(benchmark.get("triplet_equal_fields", []))
        for start in range(0, len(records), group_size):
            group = records[start : start + group_size]
            actual_shots = [normalize("shot", record.get("shot")) for record in group]
            if actual_shots != shots:
                raise ReportError(
                    f"Non-canonical shot order at rows {start + 1}-{start + group_size}: "
                    f"{actual_shots}"
                )
            for field in equal_fields:
                values = [normalize(field, record.get(field)) for record in group]
                if any(value != values[0] for value in values[1:]):
                    raise ReportError(
                        f"Shot-group field mismatch at rows {start + 1}-{start + group_size}: "
                        f"field={field}"
                    )
            parsed_ids = [trailing_integer(str(record["sample_id"])) for record in group]
            if (
                len({prefix for prefix, _, _ in parsed_ids}) != 1
                or len({width for _, _, width in parsed_ids}) != 1
                or [number for _, number, _ in parsed_ids]
                != list(range(parsed_ids[0][1], parsed_ids[0][1] + group_size))
            ):
                raise ReportError(f"Non-consecutive sample IDs in shot group at row {start + 1}")
            if benchmark.get("require_context_subsequence", False):
                contexts = [
                    [str(item) for item in record["contexts_frame_id"]] for record in group
                ]
                for smaller, larger in zip(contexts, contexts[1:]):
                    if not is_subsequence(smaller, larger):
                        raise ReportError(
                            f"Context order is not preserved in shot group at row {start + 1}"
                        )

    index = {identity_key(record, key_fields): record for record in records}
    count_rows = [
        {
            "shot": shot,
            "dataset": dataset,
            "task_type": task,
            "num_samples": cell_counts[(shot, dataset, task)],
        }
        for shot, dataset, task in itertools.product(shots, datasets, tasks)
    ]
    audit = {
        "path": str(path),
        "sha256": sha256_bytes(payload),
        "num_samples": len(records),
        "num_unique_keys": len(keys),
        "num_unique_sample_ids": len(sample_ids),
        "shot_counts": {str(shot): shot_counts[shot] for shot in shots},
        "dataset_counts": {dataset: dataset_counts[dataset] for dataset in datasets},
        "task_counts": {task: task_counts[task] for task in tasks},
        "adjacent_shot_groups": len(records) // len(shots),
    }
    return records, index, audit, count_rows


def metrics_by_source(config: Mapping[str, Any]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for metric in config["metrics"]:
        grouped[str(metric["source"])].append(metric)
    return dict(grouped)


def discover_models(
    metrics_root: Path,
    model_config: Mapping[str, Any],
    required_sources: Sequence[str],
    cli_models: str,
) -> List[str]:
    candidate_dirs = {
        path.name
        for path in metrics_root.iterdir()
        if path.is_dir() and any((path / source).is_dir() for source in required_sources)
    }
    include = [str(value) for value in model_config.get("include", [])]
    exclude = {str(value) for value in model_config.get("exclude", [])}
    if cli_models and cli_models != "all":
        selected = [part.strip() for part in cli_models.split(",") if part.strip()]
    elif include:
        selected = include
    elif model_config.get("mode", "discover") == "discover" or cli_models == "all":
        selected = sorted(candidate_dirs)
    else:
        raise ReportError("Model configuration selected neither discovery nor an include list")
    selected = [model for model in selected if model not in exclude]
    if not selected:
        raise ReportError("No models selected")
    if len(selected) != len(set(selected)):
        raise ReportError(f"Duplicate model IDs in selection: {selected}")
    missing_models = [model for model in selected if model not in candidate_dirs]
    if missing_models:
        raise ReportError(f"Selected model metric directories are missing: {missing_models}")
    order = [str(value) for value in model_config.get("order", [])]
    order_index = {model: index for index, model in enumerate(order)}
    return sorted(selected, key=lambda model: (order_index.get(model, len(order)), model))


def merged_assertions(source_metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    assertions: Dict[str, Any] = {}
    for metric in source_metrics:
        for field, expected in metric.get("row_assertions", {}).items():
            if field in assertions and assertions[field] != expected:
                raise ReportError(
                    f"Conflicting assertions for source {metric['source']}: {field}"
                )
            assertions[field] = expected
    return assertions


def source_config_hash_field(source_metrics: Sequence[Mapping[str, Any]]) -> str:
    fields = {
        str(metric["config_hash_field"])
        for metric in source_metrics
        if metric.get("config_hash_field")
    }
    if len(fields) > 1:
        raise ReportError(f"A metric source declares multiple config hash fields: {fields}")
    return next(iter(fields), "")


def load_model_scores(
    metrics_root: Path,
    models: Sequence[str],
    source_specs: Mapping[str, Sequence[Mapping[str, Any]]],
    benchmark_index: Mapping[Tuple[Any, ...], Mapping[str, Any]],
    benchmark_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    allow_mixed_config: bool,
) -> Tuple[
    Dict[str, Dict[Tuple[Any, ...], Dict[str, float]]],
    List[Dict[str, Any]],
    Dict[str, str],
]:
    key_fields = list(benchmark_config["key_fields"])
    metadata_fields = list(benchmark_config["metadata_fields"])
    record_ids = {str(k): str(v) for k, v in model_config.get("record_ids", {}).items()}
    values: Dict[str, Dict[Tuple[Any, ...], Dict[str, float]]] = {}
    source_audit: List[Dict[str, Any]] = []
    shared_hashes: Dict[str, set[str]] = defaultdict(set)
    benchmark_keys = set(benchmark_index)

    for model in models:
        values[model] = {key: {} for key in benchmark_index}
        expected_record_model = record_ids.get(model, model)
        for source, specs in source_specs.items():
            path = metrics_root / model / source / "results.json"
            payload, file_digest = load_json(path)
            if not isinstance(payload, list):
                raise ReportError(f"Metric results are not a JSON list: {path}")
            assertions = merged_assertions(specs)
            config_hash_field = source_config_hash_field(specs)
            config_hashes: set[str] = set()
            selected: Dict[Tuple[Any, ...], Dict[str, float]] = {}
            extras = 0

            for row_index, row in enumerate(payload):
                if not isinstance(row, dict):
                    raise ReportError(f"Non-object metric row at {path}[{row_index}]")
                key = identity_key(row, key_fields)
                if key not in benchmark_keys:
                    extras += 1
                    continue
                if key in selected:
                    raise ReportError(f"Duplicate final metric row in {model}/{source}: {key}")
                if row.get("metric_success") is not True:
                    raise ReportError(f"Failed final metric row in {model}/{source}: {key}")
                if normalize("model", row.get("model")) != expected_record_model:
                    raise ReportError(
                        f"Model identity mismatch in {model}/{source}: "
                        f"expected={expected_record_model!r}, actual={row.get('model')!r}"
                    )
                expected_record = benchmark_index[key]
                for field in metadata_fields:
                    expected = normalize(field, expected_record.get(field))
                    actual = normalize(field, row.get(field))
                    if actual != expected:
                        raise ReportError(
                            f"Metadata mismatch in {model}/{source}/{key}: "
                            f"field={field}, expected={expected!r}, actual={actual!r}"
                        )
                for field, expected in assertions.items():
                    if row.get(field) != expected:
                        raise ReportError(
                            f"Assertion failed in {model}/{source}/{key}: "
                            f"{field} expected {expected!r}, found {row.get(field)!r}"
                        )

                row_values: Dict[str, float] = {}
                for spec in specs:
                    raw = finite_number(
                        row.get(spec["field"]),
                        f"{model}/{source}/{key}/{spec['field']}",
                    )
                    lower, upper = (float(value) for value in spec["raw_range"])
                    if raw < lower - 1e-9 or raw > upper + 1e-9:
                        raise ReportError(
                            f"Metric outside configured range in {model}/{source}/{key}: "
                            f"{spec['field']}={raw}, range=[{lower}, {upper}]"
                        )
                    row_values[str(spec["name"])] = raw * float(spec["scale"])
                if config_hash_field:
                    config_hash = str(row.get(config_hash_field, "")).strip()
                    if not config_hash:
                        raise ReportError(
                            f"Missing {config_hash_field} in {model}/{source}/{key}"
                        )
                    config_hashes.add(config_hash)
                selected[key] = row_values

            missing = benchmark_keys.difference(selected)
            if missing:
                first = sorted(missing, key=str)[:3]
                raise ReportError(
                    f"Incomplete final coverage in {model}/{source}: "
                    f"missing={len(missing)}, first={first}"
                )
            if len(config_hashes) > 1:
                raise ReportError(
                    f"Mixed metric configurations within {model}/{source}: {sorted(config_hashes)}"
                )
            shared_hashes[source].update(config_hashes)
            for key, row_values in selected.items():
                overlap = set(values[model][key]).intersection(row_values)
                if overlap:
                    raise ReportError(f"Metric collision in {model}/{key}: {sorted(overlap)}")
                values[model][key].update(row_values)
            source_audit.append(
                {
                    "model": model,
                    "metric_source": source,
                    "results_path": str(path),
                    "results_sha256": file_digest,
                    "raw_rows": len(payload),
                    "selected_rows": len(selected),
                    "ignored_nonfinal_rows": extras,
                    "duplicate_final_rows": 0,
                    "failed_final_rows": 0,
                    "missing_final_rows": 0,
                    "metric_config_sha256": next(iter(config_hashes), ""),
                }
            )

    mixed = {source: hashes for source, hashes in shared_hashes.items() if len(hashes) > 1}
    if mixed and not allow_mixed_config:
        formatted = {source: sorted(hashes) for source, hashes in mixed.items()}
        raise ReportError(f"Metric configurations differ across models: {formatted}")
    shared = {
        source: next(iter(hashes)) if len(hashes) == 1 else "MIXED:" + ",".join(sorted(hashes))
        for source, hashes in sorted(shared_hashes.items())
    }
    return values, source_audit, shared


def score_definitions(config: Mapping[str, Any]) -> Tuple[List[str], Dict[str, str]]:
    names = [str(metric["name"]) for metric in config["metrics"]]
    directions = {str(metric["name"]): str(metric["direction"]) for metric in config["metrics"]}
    for composite in config["composites"]:
        name = str(composite["name"])
        names.append(name)
        directions[name] = str(composite["direction"])
    return names, directions


def apply_composites(row: MutableMapping[str, Any], composites: Sequence[Mapping[str, Any]]) -> None:
    for composite in composites:
        members = [str(member) for member in composite["members"]]
        weights = [finite_number(weight, f"composite {composite['name']} weight") for weight in composite["weights"]]
        row[str(composite["name"])] = weighted_mean(
            [finite_number(row[member], f"composite {composite['name']}/{member}") for member in members],
            weights,
        )


def build_sample_rows(
    benchmark_records: Sequence[Mapping[str, Any]],
    model_values: Mapping[str, Mapping[Tuple[Any, ...], Mapping[str, float]]],
    models: Sequence[str],
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    key_fields = list(config["benchmark"]["key_fields"])
    display_names = config["models"].get("display_names", {})
    metric_names = [str(metric["name"]) for metric in config["metrics"]]
    rows: List[Dict[str, Any]] = []
    for model in models:
        for record in benchmark_records:
            key = identity_key(record, key_fields)
            values = model_values[model][key]
            if set(values) != set(metric_names):
                raise ReportError(f"Incomplete joined metric row for {model}/{key}")
            row: Dict[str, Any] = {
                "model": model,
                "model_display": display_names.get(model, model),
                "dataset": normalize("dataset", record.get("dataset")),
                "shot": normalize("shot", record.get("shot")),
                "task_type": normalize("task_type", record.get("task_type")),
                "sample_id": normalize("sample_id", record.get("sample_id")),
                "scene_id": normalize("scene_id", record.get("scene_id")),
                "gt_frame_id": normalize("gt_frame_id", record.get("gt_frame_id")),
                "anchor_frame_id": normalize("anchor_frame_id", record.get("anchor_frame_id")),
                "num_samples": 1,
                **values,
            }
            apply_composites(row, config["composites"])
            rows.append(row)
    return rows


def filter_rows(rows: Sequence[Mapping[str, Any]], filters: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not filters:
        return [dict(row) for row in rows]
    normalized_filters = {
        field: {normalize(field, value) for value in values}
        for field, values in filters.items()
    }
    return [
        dict(row)
        for row in rows
        if all(normalize(field, row.get(field)) in allowed for field, allowed in normalized_filters.items())
    ]


def expected_dimension_values(field: str, config: Mapping[str, Any], models: Sequence[str]) -> List[Any]:
    if field == "model":
        return list(models)
    if field == "shot":
        return [int(value) for value in config["benchmark"]["shots"]]
    if field == "dataset":
        return [str(value) for value in config["benchmark"]["datasets"]]
    if field == "task_type":
        return [str(value) for value in config["benchmark"]["tasks"]]
    raise ReportError(f"No configured values for equal-over dimension: {field}")


def validate_equal_over(
    source_rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    config: Mapping[str, Any],
    models: Sequence[str],
    label: str,
) -> None:
    if not fields:
        return
    expected_lists = [expected_dimension_values(field, config, models) for field in fields]
    expected = Counter(itertools.product(*expected_lists))
    observed = Counter(tuple(row.get(field) for field in fields) for row in source_rows)
    if observed != expected:
        raise ReportError(
            f"Table {label} does not contain exactly one row per equal-weight cell: "
            f"fields={fields}, expected={dict(expected)}, observed={dict(observed)}"
        )


def decorate_row(row: MutableMapping[str, Any], config: Mapping[str, Any]) -> None:
    if "model" in row:
        row["model_display"] = config["models"].get("display_names", {}).get(
            row["model"], row["model"]
        )
    if "task_type" in row:
        task = row["task_type"]
        row["task_display"] = config.get("task_display", {}).get(task, task)
        row["task_group"] = ""
        for group, tasks in config.get("task_groups", {}).items():
            if task in tasks:
                row["task_group"] = group
                break


def aggregate_table(
    source_rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    score_names: Sequence[str],
    config: Mapping[str, Any],
    models: Sequence[str],
) -> List[Dict[str, Any]]:
    filtered = filter_rows(source_rows, spec.get("filters", {}))
    group_by = [str(field) for field in spec["group_by"]]
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in filtered:
        groups[tuple(row.get(field) for field in group_by)].append(row)
    if not groups:
        raise ReportError(f"Table {spec['name']} has no rows after filtering")
    output: List[Dict[str, Any]] = []
    equal_over = [str(field) for field in spec.get("equal_over", [])]
    for key, rows in groups.items():
        validate_equal_over(rows, equal_over, config, models, str(spec["name"]))
        result: Dict[str, Any] = dict(zip(group_by, key))
        result["num_samples"] = sum(int(row.get("num_samples", 0)) for row in rows)
        result["num_source_rows"] = len(rows)
        for score in score_names:
            result[score] = arithmetic_mean(
                finite_number(row.get(score), f"{spec['name']}/{key}/{score}") for row in rows
            )
        decorate_row(result, config)
        output.append(result)
    return output


def rank_rows(
    rows: List[Dict[str, Any]],
    spec: Mapping[str, Any],
    score_names: Sequence[str],
    directions: Mapping[str, str],
) -> None:
    rank_columns = spec.get("rank_columns", [])
    if rank_columns == "all_scores":
        rank_columns = list(score_names)
    else:
        rank_columns = [str(column) for column in rank_columns]
    if not rank_columns:
        return
    within = [str(field) for field in spec.get("rank_within", [])]
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in within)].append(row)
    for group_key, group_rows in groups.items():
        if len({row.get("model") for row in group_rows}) != len(group_rows):
            raise ReportError(
                f"Ranking group in {spec['name']} has duplicate model rows: {group_key}"
            )
        for column in rank_columns:
            direction = directions.get(column)
            if direction not in {"higher", "lower"}:
                raise ReportError(f"No score direction configured for ranking column {column}")
            ordered = sorted(
                group_rows,
                key=lambda row: (
                    -finite_number(row[column], f"rank/{column}")
                    if direction == "higher"
                    else finite_number(row[column], f"rank/{column}"),
                    str(row.get("model", "")),
                ),
            )
            previous: float | None = None
            competition_rank = 0
            for position, row in enumerate(ordered, 1):
                value = float(row[column])
                if previous is None or value != previous:
                    competition_rank = position
                row[f"rank_{column}"] = competition_rank
                previous = value
    primary = spec.get("primary_rank")
    if primary:
        for row in rows:
            row["rank"] = row[f"rank_{primary}"]


def select_table(source_rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = filter_rows(source_rows, spec.get("filters", {}))
    if not rows:
        raise ReportError(f"Table {spec['name']} has no rows after filtering")
    return rows


def shot_comparison_table(
    source_rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    score_names: Sequence[str],
    directions: Mapping[str, str],
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    filtered = filter_rows(source_rows, spec.get("filters", {}))
    index_fields = [str(field) for field in spec.get("index", ["model"])]
    dimension = str(spec.get("dimension", "shot"))
    levels = [normalize(dimension, value) for value in spec["levels"]]
    deltas = [[normalize(dimension, value) for value in pair] for pair in spec.get("deltas", [])]
    groups: Dict[Tuple[Any, ...], Dict[Any, Mapping[str, Any]]] = defaultdict(dict)
    for row in filtered:
        key = tuple(row.get(field) for field in index_fields)
        level = row.get(dimension)
        if level in groups[key]:
            raise ReportError(f"Duplicate {dimension}={level} for table {spec['name']}/{key}")
        groups[key][level] = row
    output: List[Dict[str, Any]] = []
    for key, level_rows in groups.items():
        if set(level_rows) != set(levels):
            raise ReportError(
                f"Incomplete levels for table {spec['name']}/{key}: "
                f"expected={levels}, actual={sorted(level_rows)}"
            )
        result: Dict[str, Any] = dict(zip(index_fields, key))
        for level in levels:
            result[f"num_samples_c{level}"] = int(level_rows[level].get("num_samples", 0))
        for score in score_names:
            values = {level: float(level_rows[level][score]) for level in levels}
            for level in levels:
                result[f"{score}_c{level}"] = values[level]
            for left, right in deltas:
                result[f"{score}_delta_c{left}_c{right}"] = values[left] - values[right]
            result[f"{score}_mean"] = arithmetic_mean(values.values())
        primary_base = str(spec.get("primary_rank", "")).removesuffix("_mean")
        if primary_base in score_names:
            best = (
                max(levels, key=lambda level: result[f"{primary_base}_c{level}"])
                if directions[primary_base] == "higher"
                else min(levels, key=lambda level: result[f"{primary_base}_c{level}"])
            )
            result["best_setting"] = f"C{best}"
        decorate_row(result, config)
        output.append(result)

    primary = str(spec.get("primary_rank", ""))
    if primary:
        direction = directions.get(primary.removesuffix("_mean"))
        if direction not in {"higher", "lower"}:
            raise ReportError(f"No direction for comparison rank {primary}")
        ordered = sorted(
            output,
            key=lambda row: (
                -float(row[primary]) if direction == "higher" else float(row[primary]),
                str(row.get("model", "")),
            ),
        )
        previous: float | None = None
        rank = 0
        for position, row in enumerate(ordered, 1):
            value = float(row[primary])
            if previous is None or value != previous:
                rank = position
            row["rank"] = rank
            previous = value
    return output


def dimension_sort_key(
    row: Mapping[str, Any], fields: Sequence[str], config: Mapping[str, Any], models: Sequence[str]
) -> Tuple[Any, ...]:
    order_maps = {
        "model": {value: index for index, value in enumerate(models)},
        "shot": {int(value): index for index, value in enumerate(config["benchmark"]["shots"])},
        "dataset": {
            str(value): index for index, value in enumerate(config["benchmark"]["datasets"])
        },
        "task_type": {
            str(value): index for index, value in enumerate(config["benchmark"]["tasks"])
        },
    }
    output: List[Any] = []
    for field in fields:
        value = row.get(field)
        if field in order_maps:
            output.append((0, order_maps[field].get(value, len(order_maps[field])), str(value)))
        else:
            output.append((1, str(value)))
    return tuple(output)


def build_tables(
    sample_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    models: Sequence[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    score_names, directions = score_definitions(config)
    sources: Dict[str, Sequence[Mapping[str, Any]]] = {"samples": sample_rows}
    tables: Dict[str, List[Dict[str, Any]]] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    for spec in config["tables"]:
        name = str(spec["name"])
        source_name = str(spec["source"])
        if source_name not in sources:
            raise ReportError(
                f"Table {name} references unavailable source {source_name}; "
                "table declarations must be topologically ordered"
            )
        kind = str(spec.get("type", "aggregate"))
        if kind == "aggregate":
            rows = aggregate_table(sources[source_name], spec, score_names, config, models)
            rank_rows(rows, spec, score_names, directions)
            sort_fields = [str(field) for field in spec.get("sort_by", spec["group_by"])]
        elif kind == "select":
            rows = select_table(sources[source_name], spec)
            rank_rows(rows, spec, score_names, directions)
            sort_fields = [str(field) for field in spec.get("sort_by", [])]
            if not sort_fields:
                sort_fields = [
                    field for field in ("shot", "task_type", "model") if field in rows[0]
                ]
        elif kind == "shot_comparison":
            rows = shot_comparison_table(
                sources[source_name], spec, score_names, directions, config
            )
            sort_fields = [str(field) for field in spec.get("index", ["model"])]
        else:
            raise ReportError(f"Unsupported table type for {name}: {kind}")
        rows.sort(key=lambda row: dimension_sort_key(row, sort_fields, config, models))
        tables[name] = rows
        sources[name] = rows
        metadata[name] = {
            "type": kind,
            "source": source_name,
            "num_rows": len(rows),
            "group_by": spec.get("group_by"),
            "equal_over": spec.get("equal_over", []),
            "filters": spec.get("filters", {}),
            "rank_within": spec.get("rank_within", []),
            "rank_policy": "competition rank on full-precision values; model ID breaks sort ties",
        }
    return tables, metadata


def ordered_fieldnames(
    rows: Sequence[Mapping[str, Any]], score_names: Sequence[str]
) -> List[str]:
    preferred = [
        "model",
        "model_display",
        "shot",
        "dataset",
        "task_type",
        "task_display",
        "task_group",
        "sample_id",
        "scene_id",
        "gt_frame_id",
        "anchor_frame_id",
        "num_samples",
        "num_source_rows",
    ]
    all_fields = {field for row in rows for field in row}
    fields = [field for field in preferred if field in all_fields]
    fields.extend(score for score in score_names if score in all_fields and score not in fields)
    for score in score_names:
        related = sorted(
            field
            for field in all_fields
            if field.startswith(f"{score}_") and not field.startswith(f"rank_{score}")
        )
        fields.extend(field for field in related if field not in fields)
    fields.extend(
        f"rank_{score}"
        for score in score_names
        if f"rank_{score}" in all_fields and f"rank_{score}" not in fields
    )
    if "rank" in all_fields:
        fields.append("rank")
    fields.extend(sorted(all_fields.difference(fields)))
    return fields


def csv_value(value: Any, precision: int, paper_decimals: int | None) -> Any:
    if isinstance(value, float):
        if paper_decimals is not None:
            return f"{value:.{paper_decimals}f}"
        return format(value, f".{precision}g")
    return value


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    precision: int,
    score_names: Sequence[str] = (),
    paper_decimals: int | None = None,
) -> None:
    if not rows:
        raise ReportError(f"Refusing to write an empty CSV: {path}")
    fieldnames = ordered_fieldnames(rows, score_names)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: csv_value(row.get(field, ""), precision, paper_decimals)
                    for field in fieldnames
                }
            )


def report_readme(models: Sequence[str], num_samples: int, score_names: Sequence[str]) -> str:
    return f"""# MindView final metric report

This directory is generated from the canonical benchmark JSONL and per-sample
metric `results.json` files.  It contains {len(models)} models x {num_samples}
final samples.  Raw metric outputs are never modified.

## Official aggregation

For every model, shot, task, and metric, samples are averaged within the task.
The shot-level score then averages all configured task means with equal weight.
The cross-shot model score averages C3, C6, and C10 with equal weight.
Composite scores are calculated from the full-precision metric means according
to `metric_report_config.json`; ranking is performed before display rounding.

Reported score fields: {', '.join(score_names)}.

`DINO-I (x100)` is cosine similarity multiplied by 100 and is not clipped.
VGGT-RScore and VGGT-TScore are the stored per-sample normalized scores; they
are averaged directly rather than reconstructed from averaged raw errors.

Files ending in `.paper.csv` are rounded for typesetting.  The corresponding
`.csv` and `.json` files retain full numeric precision and are authoritative.
`report_manifest.json` records every source and generated-file SHA-256 digest.
"""


def write_report_tree(
    staging: Path,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    dataset_audit: Mapping[str, Any],
    dataset_counts: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    source_audit: Sequence[Mapping[str, Any]],
    shared_config_hashes: Mapping[str, str],
    sample_rows: Sequence[Mapping[str, Any]],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    table_metadata: Mapping[str, Mapping[str, Any]],
    requested_tables: Sequence[str],
) -> None:
    output_config = config["output"]
    precision = int(output_config.get("float_precision", 17))
    paper_decimals = int(output_config.get("paper_decimals", 2))
    score_names, _ = score_definitions(config)

    (staging / "README.md").write_text(
        report_readme(models, int(dataset_audit["num_samples"]), score_names),
        encoding="utf-8",
    )
    shutil.copy2(config_path, staging / "metric_report_config.json")
    write_csv(staging / "dataset_counts.csv", dataset_counts, precision)
    write_csv(staging / "source_audit.csv", source_audit, precision)
    if output_config.get("write_sample_scores", True):
        write_csv(
            staging / "sample_scores.csv", sample_rows, precision, score_names=score_names
        )

    for name in requested_tables:
        rows = list(tables[name])
        write_csv(
            staging / f"{name}.csv", rows, precision, score_names=score_names
        )
        write_csv(
            staging / f"{name}.paper.csv",
            rows,
            precision,
            score_names=score_names,
            paper_decimals=paper_decimals,
        )
        (staging / f"{name}.json").write_text(
            json.dumps(
                {"meta": table_metadata[name], "rows": rows},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    leader_table = str(output_config.get("leader_table", "scores_by_model_shot"))
    primary_score = str(output_config.get("primary_score", score_names[-1]))
    model_shot_rows = list(tables.get(leader_table, []))
    leaders: Dict[str, Dict[str, Any]] = {}
    for shot in config["benchmark"]["shots"]:
        candidates = [row for row in model_shot_rows if int(row["shot"]) == int(shot)]
        if candidates:
            leader = min(candidates, key=lambda row: int(row.get("rank", 10**9)))
            leaders[f"C{shot}"] = {
                "model": leader["model"],
                primary_score: leader.get(primary_score),
                "rank": leader.get("rank"),
            }
    summary = {
        "schema_version": "mindview-metric-report-v1",
        "aggregation": {
            "task": "arithmetic mean of final samples within each task",
            "shot": "unweighted arithmetic mean of task means",
            "model": "unweighted arithmetic mean of shot-level task-macro means",
            "composites": config["composites"],
        },
        "models": list(models),
        "score_fields": score_names,
        "dataset": dataset_audit,
        "shared_metric_config_sha256": dict(shared_config_hashes),
        "leaders": leaders,
        "tables": table_metadata,
    }
    (staging / "report_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    generated_hashes = {
        str(path.relative_to(staging)): sha256_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": "mindview-metric-report-manifest-v1",
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "dataset": dataset_audit,
        "models": list(models),
        "num_models": len(models),
        "num_final_samples_per_model": dataset_audit["num_samples"],
        "num_joined_sample_rows": len(sample_rows),
        "metric_sources": list(shared_config_hashes),
        "shared_metric_config_sha256": dict(shared_config_hashes),
        "source_results": list(source_audit),
        "generated_files_sha256": generated_hashes,
    }
    (staging / "report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def atomic_publish(staging: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise ReportError(f"Output exists; pass --overwrite to replace it: {output_dir}")
    backup: Path | None = None
    if output_dir.exists():
        backup = output_dir.parent / f".{output_dir.name}.backup-{os.getpid()}"
        if backup.exists():
            raise ReportError(f"Unexpected report backup already exists: {backup}")
        os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except Exception:
        if backup is not None and backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly validate final per-sample metrics and build task-macro paper reports."
    )
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--protocol", choices=("paper", "selected"), default="paper",
        help="paper validates frozen C3/C6/C10; selected reports the dimensions in --dataset-jsonl.",
    )
    parser.add_argument("--canonical-jsonl", type=Path, default=DEFAULT_DATASET,
                        help="Frozen benchmark used to verify selected sample identities and C6 cores.")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model directory IDs, or 'all'; default follows config discovery.",
    )
    parser.add_argument(
        "--tables",
        default="all",
        help="Comma-separated configured table names, or 'all'.",
    )
    parser.add_argument("--check-only", action="store_true", help="Validate and compute without writing.")
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace metric_report.")
    parser.add_argument(
        "--allow-mixed-config",
        action="store_true",
        help="Allow different metric_config_sha256 values across models (not for paper use).",
    )
    return parser


def validate_selected_identity(records: Sequence[Mapping[str, Any]], canonical: Path) -> None:
    frozen = [json.loads(line) for line in canonical.read_bytes().splitlines()]
    by_id = {row["sample_id"]: row for row in frozen}
    fields = ("dataset", "task_type", "shot", "scene_id", "gt_frame_id", "anchor_frame_id", "instruction")
    semantic_fields = ("dataset", "task_type", "scene_id", "gt_frame_id", "anchor_frame_id")
    cores = {identity_key(row, semantic_fields): row["contexts_frame_id"]
             for row in frozen if int(row["shot"]) == 6}
    for row in records:
        reference = by_id.get(row.get("sample_id"))
        if reference is None:
            raise ReportError(f"Selected sample is absent from frozen benchmark: {row.get('sample_id')}")
        for field in fields:
            if normalize(field, row.get(field)) != normalize(field, reference.get(field)):
                raise ReportError(f"Selected identity mismatch: {row['sample_id']}/{field}")
        contexts = row.get("contexts_frame_id")
        if not isinstance(contexts, list):
            raise ReportError(f"Selected contexts are not a list: {row['sample_id']}")
        selected = list(map(str, contexts))
        baseline = list(map(str, reference["contexts_frame_id"]))
        if int(row["shot"]) == 10:
            core = cores.get(identity_key(reference, semantic_fields))
            if (core is None or not is_subsequence(selected, baseline)
                    or not is_subsequence(list(map(str, core)), selected)):
                raise ReportError(f"Selected C10 must preserve the frozen C6 core and C10 order: {row['sample_id']}")
        elif selected != baseline:
            raise ReportError(f"Selected C3/C6 contexts differ from frozen benchmark: {row['sample_id']}")


def selected_protocol_config(config: Mapping[str, Any], dataset: Path,
                             canonical: Path = DEFAULT_DATASET) -> Dict[str, Any]:
    """Adapt validation/table dimensions while retaining all six score definitions."""
    effective = copy.deepcopy(config)
    records = [json.loads(line) for line in dataset.read_bytes().splitlines()]
    if not records:
        raise ReportError("Selected benchmark is empty")
    validate_selected_identity(records, canonical)
    benchmark = effective["benchmark"]
    for field, dimension in (("shot", "shots"), ("dataset", "datasets"), ("task_type", "tasks")):
        values = {normalize(field, row.get(field)) for row in records}
        if not values.issubset(set(benchmark[dimension])):
            raise ReportError(f"Unsupported selected {dimension}: {values}")
        benchmark[dimension] = [value for value in benchmark[dimension] if value in values]
    benchmark["expected_num_samples"] = len(records)
    benchmark.pop("expected_samples_per_shot", None)
    benchmark.update(require_adjacent_shot_groups=False, require_context_subsequence=False,
                     require_full_cells=False, allow_variable_contexts=True)
    benchmark.update(protocol="selected", canonical_jsonl=str(canonical.resolve()),
                     canonical_sha256=sha256_file(canonical))
    tables = []
    for table in effective["tables"]:
        if table["name"].startswith("table_c6_") and 6 not in benchmark["shots"]:
            continue
        if table["type"] == "shot_comparison":
            if len(benchmark["shots"]) < 2:
                continue
            table["levels"] = benchmark["shots"]
            table["deltas"] = [list(pair) for pair in zip(benchmark["shots"][1:], benchmark["shots"][:-1])]
        tables.append(table)
    effective["tables"] = tables
    return effective


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset_jsonl.expanduser().resolve()
    metrics_root = args.metrics_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not metrics_root.is_dir():
        raise ReportError(f"Metrics root is missing: {metrics_root}")
    if args.check_only and args.overwrite:
        raise ReportError("--check-only and --overwrite are mutually exclusive")
    if not args.check_only and output_dir.exists() and not args.overwrite:
        raise ReportError(f"Output exists; pass --overwrite to replace it: {output_dir}")
    if output_dir == metrics_root:
        raise ReportError("Report output must not replace the raw metrics root")

    config, config_digest = load_json(config_path)
    if not isinstance(config, dict):
        raise ReportError("Report config must be a JSON object")
    release, _ = load_json(SCRIPT_DIR.parent / "data/manifest.json")
    canonical_path = args.canonical_jsonl.expanduser().resolve()
    verified_path = canonical_path if args.protocol == "selected" else dataset_path
    if sha256_file(verified_path) != release["sha256"]:
        raise ReportError("Paper/canonical benchmark SHA256 differs from the frozen release manifest")
    if args.protocol == "selected":
        config = selected_protocol_config(config, dataset_path, canonical_path)
    validate_config(config)
    benchmark_records, benchmark_index, dataset_audit, dataset_counts = validate_dataset(
        dataset_path, config["benchmark"]
    )
    source_specs = metrics_by_source(config)
    models = discover_models(
        metrics_root,
        config["models"],
        list(source_specs),
        args.models,
    )
    model_roots = [(metrics_root / model).resolve() for model in models]
    if any(output_dir == root or root in output_dir.parents for root in model_roots):
        raise ReportError("Report output must not be inside a raw model metric directory")

    model_values, source_audit, shared_config_hashes = load_model_scores(
        metrics_root,
        models,
        source_specs,
        benchmark_index,
        config["benchmark"],
        config["models"],
        args.allow_mixed_config,
    )
    sample_rows = build_sample_rows(
        benchmark_records, model_values, models, config
    )
    tables, table_metadata = build_tables(sample_rows, config, models)
    if args.tables == "all":
        requested_tables = list(tables)
    else:
        requested_tables = [part.strip() for part in args.tables.split(",") if part.strip()]
        unknown = sorted(set(requested_tables).difference(tables))
        if unknown:
            raise ReportError(f"Unknown requested tables: {unknown}")
        if not requested_tables:
            raise ReportError("No report tables requested")

    expected_joined = len(models) * len(benchmark_records)
    if len(sample_rows) != expected_joined:
        raise ReportError(
            f"Joined sample cube is incomplete: expected={expected_joined}, actual={len(sample_rows)}"
        )

    if args.check_only:
        print("[check] strict report validation passed; no files written")
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
        )
        try:
            write_report_tree(
                staging,
                config=config,
                config_path=config_path,
                config_sha256=config_digest,
                dataset_audit=dataset_audit,
                dataset_counts=dataset_counts,
                models=models,
                source_audit=source_audit,
                shared_config_hashes=shared_config_hashes,
                sample_rows=sample_rows,
                tables=tables,
                table_metadata=table_metadata,
                requested_tables=requested_tables,
            )
            atomic_publish(staging, output_dir, args.overwrite)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        print(f"[write] metric report saved to: {output_dir}")

    print(
        f"[audit] models={len(models)}, final_samples/model={len(benchmark_records)}, "
        f"joined_rows={len(sample_rows)}, dataset_sha256={dataset_audit['sha256']}"
    )
    leader_table = str(config["output"].get("leader_table", "scores_by_model_shot"))
    primary_score = str(config["output"].get("primary_score", score_definitions(config)[0][-1]))
    model_shot = tables.get(leader_table, [])
    for shot in config["benchmark"]["shots"]:
        candidates = [row for row in model_shot if int(row["shot"]) == int(shot)]
        if candidates:
            leader = min(candidates, key=lambda row: int(row.get("rank", 10**9)))
            print(
                f"[summary] C{shot}: rank-1={leader['model']}, "
                f"{primary_score}={float(leader[primary_score]):.6f}"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReportError as exc:
        raise SystemExit(f"[Error] {exc}") from exc
