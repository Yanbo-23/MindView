from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "build_metric_report.py"
SPEC = importlib.util.spec_from_file_location("build_metric_report", SCRIPT)
assert SPEC and SPEC.loader
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def minimal_config():
    return {
        "benchmark": {
            "shots": [6],
            "datasets": ["dataset"],
            "tasks": ["small", "large"],
            "key_fields": ["dataset", "shot", "task_type", "sample_id"],
            "metadata_fields": [
                "dataset",
                "task_type",
                "shot",
                "scene_id",
                "gt_frame_id",
                "anchor_frame_id",
            ],
        },
        "models": {"display_names": {}},
        "metrics": [
            {
                "name": "score",
                "source": "metric",
                "field": "raw_score",
                "scale": 100.0,
                "raw_range": [-1.0, 1.0],
                "direction": "higher",
                "config_hash_field": "metric_config_sha256",
            }
        ],
        "composites": [
            {
                "name": "average",
                "members": ["score"],
                "weights": [1],
                "direction": "higher",
            }
        ],
        "tables": [],
        "output": {},
    }


def test_task_macro_is_not_sample_micro_average():
    config = minimal_config()
    samples = [
        {"model": "m", "shot": 6, "task_type": "large", "num_samples": 1, "score": 0.0, "average": 0.0},
        {"model": "m", "shot": 6, "task_type": "large", "num_samples": 1, "score": 0.0, "average": 0.0},
        {"model": "m", "shot": 6, "task_type": "small", "num_samples": 1, "score": 100.0, "average": 100.0},
    ]
    task_spec = {
        "name": "task",
        "group_by": ["model", "shot", "task_type"],
    }
    task_rows = report.aggregate_table(
        samples, task_spec, ["score", "average"], config, ["m"]
    )
    shot_spec = {
        "name": "shot",
        "group_by": ["model", "shot"],
        "equal_over": ["task_type"],
    }
    shot_rows = report.aggregate_table(
        task_rows, shot_spec, ["score", "average"], config, ["m"]
    )
    assert shot_rows[0]["score"] == pytest.approx(50.0)
    assert shot_rows[0]["score"] != pytest.approx(100.0 / 3.0)


def test_metric_scaling_is_applied_exactly_once(tmp_path: Path):
    config = minimal_config()
    model_root = tmp_path / "model" / "metric"
    model_root.mkdir(parents=True)
    row = {
        "model": "model",
        "dataset": "dataset",
        "shot": 6,
        "task_type": "small",
        "sample_id": "sample_0",
        "scene_id": "scene",
        "gt_frame_id": "gt",
        "anchor_frame_id": None,
        "metric_success": True,
        "metric_config_sha256": "same-config",
        "raw_score": 0.5,
    }
    (model_root / "results.json").write_text(json.dumps([row]), encoding="utf-8")
    key = report.identity_key(row, config["benchmark"]["key_fields"])
    benchmark_index = {key: row}
    values, audit, hashes = report.load_model_scores(
        tmp_path,
        ["model"],
        report.metrics_by_source(config),
        benchmark_index,
        config["benchmark"],
        config["models"],
        allow_mixed_config=False,
    )
    assert values["model"][key]["score"] == pytest.approx(50.0)
    assert audit[0]["selected_rows"] == 1
    assert hashes == {"metric": "same-config"}


def test_competition_rank_uses_unrounded_values():
    rows = [
        {"model": "a", "average": 1.0},
        {"model": "b", "average": 1.0},
        {"model": "c", "average": 0.999999999},
    ]
    spec = {
        "name": "ranked",
        "rank_within": [],
        "rank_columns": ["average"],
        "primary_rank": "average",
    }
    report.rank_rows(rows, spec, ["average"], {"average": "higher"})
    assert {row["model"]: row["rank"] for row in rows} == {"a": 1, "b": 1, "c": 3}


def test_model_discovery_adds_new_complete_model_without_code_changes(tmp_path: Path):
    for model in ("existing", "new-model"):
        (tmp_path / model / "metric").mkdir(parents=True)
    selected = report.discover_models(
        tmp_path,
        {"mode": "discover", "include": [], "exclude": [], "order": ["existing"]},
        ["metric"],
        cli_models="",
    )
    assert selected == ["existing", "new-model"]


def test_missing_final_metric_row_fails_fast(tmp_path: Path):
    config = minimal_config()
    (tmp_path / "model" / "metric").mkdir(parents=True)
    (tmp_path / "model" / "metric" / "results.json").write_text("[]", encoding="utf-8")
    benchmark = {
        "model": "model",
        "dataset": "dataset",
        "shot": 6,
        "task_type": "small",
        "sample_id": "sample_0",
        "scene_id": "scene",
        "gt_frame_id": "gt",
        "anchor_frame_id": None,
    }
    key = report.identity_key(benchmark, config["benchmark"]["key_fields"])
    with pytest.raises(report.ReportError, match="Incomplete final coverage"):
        report.load_model_scores(
            tmp_path,
            ["model"],
            report.metrics_by_source(config),
            {key: benchmark},
            config["benchmark"],
            config["models"],
            allow_mixed_config=False,
        )
