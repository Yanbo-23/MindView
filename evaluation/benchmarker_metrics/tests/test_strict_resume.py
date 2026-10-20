import json
import math
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="Metric module tests require PyTorch; no model weights are loaded")
from PIL import Image

from benchmarker_metrics import run as metrics_run
from benchmarker_metrics.common import (
    EvalPair,
    MetricRunLockedError,
    append_jsonl,
    merge_shards,
    metric_run_lock,
    pair_to_record,
    read_jsonl,
)
from benchmarker_metrics.resume import _read_jsonl_rows


RESUME_FIELDS = {
    "resume_schema_version",
    "eval_key",
    "input_sha256",
    "metric_config_sha256",
    "evaluation_sha256",
}


def _write_rgb(path: Path, value: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), (value, value, value)).save(path)
    return str(path.resolve())


def _write_depth(path: Path, value: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("I;16", (8, 6), value).save(path)
    return str(path.resolve())


def _pair(
    tmp_path: Path,
    sample_id: str = "sample_00000000",
    *,
    contexts=None,
    gt_source_path: str | None = None,
) -> EvalPair:
    pred_path = _write_rgb(tmp_path / sample_id / "pred.png", 20)
    gt_path = _write_rgb(tmp_path / sample_id / "gt.png", 40)
    raw_record = {"global_index": int(sample_id.rsplit("_", 1)[-1])}
    if gt_source_path is not None:
        raw_record["ground_truth"] = {"source_path": gt_source_path}
    return EvalPair(
        model="test-model",
        run_name="test-run",
        run_dir=str(tmp_path),
        dataset="scannet",
        shot=6,
        task_type="rotation",
        sample_id=sample_id,
        scene_id="scene0000_00",
        gt_frame_id="100",
        anchor_frame_id="90",
        pred_path=pred_path,
        gt_path=gt_path,
        context_source_paths=list(contexts or []),
        context_vis_path=None,
        raw_record=raw_record,
    )


def _install_fake_worker(monkeypatch, module, calls):
    metric_keys = list(module.METRIC_KEYS)

    def fake_run_worker(args, pairs):
        pairs = list(pairs)
        calls.append([pair.sample_id for pair in pairs])
        shard_path = (
            Path(args.metrics_root)
            / module.METRIC_NAME
            / "shards"
            / f"rank_{args.rank}.jsonl"
        )
        for index, pair in enumerate(pairs):
            record = pair_to_record(pair)
            record["metric_success"] = True
            for key_index, key in enumerate(metric_keys):
                record[key] = 0.5 + index * 0.01 + key_index * 0.001
            append_jsonl(shard_path, record)

    monkeypatch.setattr(module, "run_worker", fake_run_worker)


def _invoke(monkeypatch, tmp_path: Path, metric: str, *extra: str) -> None:
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir(exist_ok=True)
    argv = [
        "benchmarker_metrics.run",
        "--metric",
        metric,
        "--benchmark_output_root",
        str(benchmark_root),
        "--metrics_root",
        str(tmp_path / "metrics"),
        "--num_gpus",
        "1",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    metrics_run.main()


def _results(tmp_path: Path, metric: str):
    return json.loads(
        (tmp_path / "metrics" / metric / "results.json").read_text(
            encoding="utf-8"
        )
    )


def _replace_cached_row(tmp_path: Path, metric: str, row) -> None:
    metric_dir = tmp_path / "metrics" / metric
    shard = metric_dir / "shards" / "rank_0.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (metric_dir / "results.json").write_text(
        json.dumps([row]), encoding="utf-8"
    )


def test_exact_success_is_skipped_without_loading_worker(monkeypatch, tmp_path):
    pair = _pair(tmp_path)
    monkeypatch.setattr(
        metrics_run, "_discover_for_metric", lambda _args, require_contexts: [pair]
    )
    calls = []
    _install_fake_worker(monkeypatch, metrics_run.clip_dino, calls)

    _invoke(monkeypatch, tmp_path, "clip_dino", "--overwrite")
    assert calls == [[pair.sample_id]]
    first = _results(tmp_path, "clip_dino")
    assert len(first) == 1
    assert RESUME_FIELDS <= first[0].keys()

    calls.clear()
    _invoke(monkeypatch, tmp_path, "clip_dino", "--skip_completed")

    # A zero-pending rank must return before run_worker loads any model.
    assert calls == []
    assert _results(tmp_path, "clip_dino") == first


def test_failed_and_legacy_rows_are_never_skipped(monkeypatch, tmp_path):
    pair = _pair(tmp_path)
    monkeypatch.setattr(
        metrics_run, "_discover_for_metric", lambda _args, require_contexts: [pair]
    )
    calls = []
    _install_fake_worker(monkeypatch, metrics_run.clip_dino, calls)

    _invoke(monkeypatch, tmp_path, "clip_dino", "--overwrite")
    signed_success = _results(tmp_path, "clip_dino")[0]

    failed = dict(signed_success, metric_success=False)
    _replace_cached_row(tmp_path, "clip_dino", failed)
    calls.clear()
    _invoke(monkeypatch, tmp_path, "clip_dino", "--skip_completed")
    assert calls == [[pair.sample_id]]
    rows = _results(tmp_path, "clip_dino")
    assert len(rows) == 1 and rows[0]["metric_success"] is True

    legacy = {
        key: value
        for key, value in rows[0].items()
        if key not in RESUME_FIELDS
    }
    _replace_cached_row(tmp_path, "clip_dino", legacy)
    calls.clear()
    _invoke(monkeypatch, tmp_path, "clip_dino", "--skip_completed")
    assert calls == [[pair.sample_id]]
    rows = _results(tmp_path, "clip_dino")
    assert len(rows) == 1
    assert RESUME_FIELDS <= rows[0].keys()


def test_nan_or_missing_metric_value_is_never_skipped(monkeypatch, tmp_path):
    pair = _pair(tmp_path)
    monkeypatch.setattr(
        metrics_run, "_discover_for_metric", lambda _args, require_contexts: [pair]
    )
    calls = []
    _install_fake_worker(monkeypatch, metrics_run.clip_dino, calls)

    _invoke(monkeypatch, tmp_path, "clip_dino", "--overwrite")
    signed_success = _results(tmp_path, "clip_dino")[0]

    for invalid_kind in ("nan", "missing"):
        invalid = dict(signed_success)
        if invalid_kind == "nan":
            invalid["dino_i"] = math.nan
        else:
            invalid.pop("dino_i")
        _replace_cached_row(tmp_path, "clip_dino", invalid)
        # Remove the cache written by a preceding loop iteration so this case
        # is judged solely from the deliberately invalid current records.
        cache_path = tmp_path / "metrics" / "clip_dino" / "resume_records.json"
        if cache_path.exists():
            cache_path.unlink()

        calls.clear()
        _invoke(monkeypatch, tmp_path, "clip_dino", "--skip_completed")
        assert calls == [[pair.sample_id]]
        rows = _results(tmp_path, "clip_dino")
        assert len(rows) == 1
        assert rows[0]["metric_success"] is True
        assert math.isfinite(rows[0]["dino_i"])


def test_input_or_metric_config_change_recomputes_and_drops_stale_row(
    monkeypatch, tmp_path
):
    pair = _pair(tmp_path)
    monkeypatch.setattr(
        metrics_run, "_discover_for_metric", lambda _args, require_contexts: [pair]
    )
    calls = []
    _install_fake_worker(monkeypatch, metrics_run.clip_dino, calls)

    _invoke(
        monkeypatch, tmp_path, "clip_dino", "--image_size", "16", "--overwrite"
    )
    original = _results(tmp_path, "clip_dino")[0]

    # Rewrite the same path with another image. Path/size based caching would
    # incorrectly reuse it; byte-content SHA-256 must force recomputation.
    _write_rgb(Path(pair.pred_path), 90)
    calls.clear()
    _invoke(
        monkeypatch,
        tmp_path,
        "clip_dino",
        "--image_size",
        "16",
        "--skip_completed",
    )
    assert calls == [[pair.sample_id]]
    changed_input = _results(tmp_path, "clip_dino")
    assert len(changed_input) == 1
    assert changed_input[0]["input_sha256"] != original["input_sha256"]
    assert (
        changed_input[0]["metric_config_sha256"]
        == original["metric_config_sha256"]
    )

    # Identical files but a different preprocessing configuration must also be
    # recomputed and must not leave the stale row in aggregate results.
    calls.clear()
    _invoke(
        monkeypatch,
        tmp_path,
        "clip_dino",
        "--image_size",
        "32",
        "--skip_completed",
    )
    assert calls == [[pair.sample_id]]
    changed_config = _results(tmp_path, "clip_dino")
    assert len(changed_config) == 1
    assert changed_config[0]["input_sha256"] == changed_input[0]["input_sha256"]
    assert (
        changed_config[0]["metric_config_sha256"]
        != changed_input[0]["metric_config_sha256"]
    )


def test_identical_bytes_at_new_paths_are_reused_and_metadata_is_refreshed(
    monkeypatch, tmp_path
):
    pair = _pair(tmp_path)
    monkeypatch.setattr(
        metrics_run, "_discover_for_metric", lambda _args, require_contexts: [pair]
    )
    calls = []
    _install_fake_worker(monkeypatch, metrics_run.clip_dino, calls)

    _invoke(monkeypatch, tmp_path, "clip_dino", "--overwrite")
    original = _results(tmp_path, "clip_dino")[0]

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    new_pred = relocated / "prediction.png"
    new_gt = relocated / "ground_truth.png"
    shutil.copyfile(pair.pred_path, new_pred)
    shutil.copyfile(pair.gt_path, new_gt)
    pair.pred_path = str(new_pred.resolve())
    pair.gt_path = str(new_gt.resolve())

    calls.clear()
    _invoke(monkeypatch, tmp_path, "clip_dino", "--skip_completed")

    # Snapshot/output relocation is not a new metric input. Cache identity is
    # content based, while the exported row should still describe current paths.
    assert calls == []
    current = _results(tmp_path, "clip_dino")
    assert len(current) == 1
    assert current[0]["input_sha256"] == original["input_sha256"]
    assert current[0]["pred_path"] == pair.pred_path
    assert current[0]["gt_path"] == pair.gt_path


def test_vggt_context_order_and_metric_depth_are_in_input_signature(
    monkeypatch, tmp_path
):
    scene = tmp_path / "scene0000_00"
    context_0 = _write_rgb(scene / "color" / "000.jpg", 10)
    context_1 = _write_rgb(scene / "color" / "001.jpg", 20)
    gt_source = _write_rgb(scene / "color" / "100.jpg", 30)
    _write_depth(scene / "depth" / "000.png", 1000)
    depth_1 = _write_depth(scene / "depth" / "001.png", 1100)
    _write_depth(scene / "depth" / "100.png", 1200)
    pair = _pair(
        tmp_path,
        contexts=[context_0, context_1],
        gt_source_path=gt_source,
    )
    monkeypatch.setattr(
        metrics_run, "_discover_for_metric", lambda _args, require_contexts: [pair]
    )
    calls = []
    _install_fake_worker(monkeypatch, metrics_run.vggt_geometry, calls)

    _invoke(monkeypatch, tmp_path, "vggt", "--overwrite")
    baseline = _results(tmp_path, "vggt")[0]

    pair.context_source_paths[:] = [context_1, context_0]
    calls.clear()
    _invoke(monkeypatch, tmp_path, "vggt", "--skip_completed")
    assert calls == [[pair.sample_id]]
    reordered = _results(tmp_path, "vggt")[0]
    assert reordered["input_sha256"] != baseline["input_sha256"]

    # Depth is not a displayed VGGT frame, but it calibrates translation into
    # metres and is therefore a true metric input that must invalidate cache.
    _write_depth(Path(depth_1), 2100)
    calls.clear()
    _invoke(monkeypatch, tmp_path, "vggt", "--skip_completed")
    assert calls == [[pair.sample_id]]
    changed_depth = _results(tmp_path, "vggt")[0]
    assert changed_depth["input_sha256"] != reordered["input_sha256"]


def test_resume_after_eight_to_one_worker_change_keeps_every_sample(
    monkeypatch, tmp_path
):
    pairs = [_pair(tmp_path, f"sample_{index:08d}") for index in range(8)]
    monkeypatch.setattr(
        metrics_run, "_discover_for_metric", lambda _args, require_contexts: pairs
    )
    calls = []
    _install_fake_worker(monkeypatch, metrics_run.clip_dino, calls)

    # Simulate the child processes of an earlier eight-GPU run without spawning
    # subprocesses or loading a real metric model.
    for rank in range(8):
        _invoke(
            monkeypatch,
            tmp_path,
            "clip_dino",
            "--rank",
            str(rank),
            "--world_size",
            "8",
            "--skip_completed",
        )
    assert sorted(sample for call in calls for sample in call) == sorted(
        pair.sample_id for pair in pairs
    )

    calls.clear()
    _invoke(monkeypatch, tmp_path, "clip_dino", "--skip_completed")

    # rank_4..rank_7 must remain discoverable even though the current run has
    # only one worker, and world_size/rank must not alter the metric signature.
    assert calls == []
    rows = _results(tmp_path, "clip_dino")
    assert len(rows) == 8
    assert {row["sample_id"] for row in rows} == {
        pair.sample_id for pair in pairs
    }


def test_truncated_final_shard_line_is_recomputed_without_losing_valid_rows(
    monkeypatch, tmp_path
):
    pairs = [_pair(tmp_path, f"sample_{index:08d}") for index in range(2)]
    monkeypatch.setattr(
        metrics_run, "_discover_for_metric", lambda _args, require_contexts: pairs
    )
    calls = []
    _install_fake_worker(monkeypatch, metrics_run.clip_dino, calls)

    _invoke(monkeypatch, tmp_path, "clip_dino", "--overwrite")
    baseline = _results(tmp_path, "clip_dino")
    assert len(baseline) == 2

    metric_dir = tmp_path / "metrics" / "clip_dino"
    (metric_dir / "results.json").unlink()
    shard = metric_dir / "shards" / "rank_0.jsonl"
    shard.write_text(
        json.dumps(baseline[0]) + "\n" + json.dumps(baseline[1])[:37],
        encoding="utf-8",
    )

    calls.clear()
    _invoke(monkeypatch, tmp_path, "clip_dino", "--skip_completed")
    assert calls == [[pairs[1].sample_id]]
    recovered = _results(tmp_path, "clip_dino")
    assert len(recovered) == 2
    assert {row["sample_id"] for row in recovered} == {
        pair.sample_id for pair in pairs
    }

    calls.clear()
    _invoke(monkeypatch, tmp_path, "clip_dino", "--skip_completed")
    assert calls == []
    assert len(_results(tmp_path, "clip_dino")) == 2


def test_conflicting_values_for_same_signature_are_recomputed_fail_closed(
    monkeypatch, tmp_path
):
    pair = _pair(tmp_path)
    monkeypatch.setattr(
        metrics_run, "_discover_for_metric", lambda _args, require_contexts: [pair]
    )
    calls = []
    _install_fake_worker(monkeypatch, metrics_run.clip_dino, calls)

    _invoke(monkeypatch, tmp_path, "clip_dino", "--overwrite")
    signed = _results(tmp_path, "clip_dino")[0]
    conflicting = dict(signed)
    conflicting["dino_i"] = signed["dino_i"] + 0.25

    metric_dir = tmp_path / "metrics" / "clip_dino"
    (metric_dir / "results.json").write_text(
        json.dumps([signed]), encoding="utf-8"
    )
    (metric_dir / "shards" / "rank_0.jsonl").write_text(
        json.dumps(conflicting) + "\n", encoding="utf-8"
    )

    calls.clear()
    _invoke(monkeypatch, tmp_path, "clip_dino", "--skip_completed")

    # Ambiguous cached values must not be selected arbitrarily even though
    # both rows claim the same otherwise-valid evaluation signature.
    assert calls == [[pair.sample_id]]
    rows = _results(tmp_path, "clip_dino")
    assert len(rows) == 1
    assert rows[0]["dino_i"] == signed["dino_i"]


def test_current_merge_ignores_truncated_tail_without_losing_valid_rows(
    tmp_path,
):
    metric_dir = tmp_path / "metrics" / "clip_dino"
    shard = metric_dir / "shards" / "rank_0.jsonl"
    shard.parent.mkdir(parents=True)
    valid = {
        "model": "test-model",
        "run_name": "test-run",
        "dataset": "scannet",
        "shot": 6,
        "task_type": "rotation",
        "sample_id": "sample_00000000",
        "metric_success": True,
        "clip_i": 0.5,
        "dino_i": 0.6,
    }
    shard.write_text(
        json.dumps(valid) + "\n" + '{"sample_id":"truncated',
        encoding="utf-8",
    )

    rows = merge_shards(
        metric_dir, 1, "clip_dino", ["clip_i", "dino_i"]
    )

    assert rows == [valid]
    assert json.loads((metric_dir / "results.json").read_text()) == [valid]


def test_utf8_truncated_tail_is_ignored_by_both_jsonl_readers(tmp_path):
    shard = tmp_path / "rank_0.jsonl"
    valid = {"sample_id": "sample_00000000", "reason": "valid"}
    shard.write_bytes(
        (json.dumps(valid, ensure_ascii=False) + "\n").encode("utf-8")
        + b'{"reason":"\xe4'
    )

    assert read_jsonl(shard) == [valid]
    rows, invalid = _read_jsonl_rows(shard)
    assert rows == [valid]
    assert invalid == 1


def test_existing_output_rejects_neither_overwrite_nor_resume(
    monkeypatch, tmp_path
):
    pair = _pair(tmp_path)
    monkeypatch.setattr(
        metrics_run, "_discover_for_metric", lambda _args, require_contexts: [pair]
    )
    calls = []
    _install_fake_worker(monkeypatch, metrics_run.clip_dino, calls)

    _invoke(monkeypatch, tmp_path, "clip_dino", "--overwrite")
    baseline = _results(tmp_path, "clip_dino")
    calls.clear()

    with pytest.raises(SystemExit, match="Use exactly one"):
        _invoke(monkeypatch, tmp_path, "clip_dino")

    assert calls == []
    assert _results(tmp_path, "clip_dino") == baseline


def test_metric_run_lock_rejects_concurrent_writer_and_is_reusable(tmp_path):
    metric_dir = tmp_path / "metrics" / "clip_dino"

    with metric_run_lock(metric_dir):
        with pytest.raises(MetricRunLockedError, match="already being written"):
            with metric_run_lock(metric_dir):
                pass

    # Releasing the first holder must make the lock immediately reusable.
    with metric_run_lock(metric_dir):
        pass
