import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image

EVALUATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION))
sys.path.insert(0, str(EVALUATION / "benchmarker_metrics"))
import import_predictions
import sample_subset
from benchmarker_metrics.common import discover_eval_pairs


def test_subset_keeps_original_lines_and_refuses_changed_or_partial_reuse(tmp_path):
    output = tmp_path / "selected.jsonl"
    args = ["--output-jsonl", str(output), "--count", "10", "--seed", "42"]
    sample_subset.main(args + ["--dry-run"])
    assert not output.exists() and not output.parent.joinpath("selected.manifest.json").exists()
    sample_subset.main(args)
    original = output.read_bytes()
    manifest = output.with_suffix(".manifest.json")
    metadata = json.loads(manifest.read_text())
    frozen_lines = set(sample_subset.FROZEN_JSONL.read_bytes().splitlines(keepends=True))
    assert len(original.splitlines()) == 10
    assert all(line in frozen_lines for line in original.splitlines(keepends=True))
    rows = [json.loads(line) for line in original.splitlines()]
    assert all(row["shot"] == 6 and set(row) == sample_subset.FIELDS for row in rows)
    assert metadata["selected_sample_ids"] == [row["sample_id"] for row in rows]
    assert metadata["source_sha256"] == sample_subset.sha256(sample_subset.FROZEN_JSONL.read_bytes())
    sample_subset.main(args)
    assert output.read_bytes() == original
    with pytest.raises(ValueError, match="differs"):
        sample_subset.main(["--output-jsonl", str(output), "--seed", "43"])
    output.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="differs"):
        sample_subset.main(args)
    output.write_bytes(original)
    manifest.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        sample_subset.main(args)


def test_subset_protects_frozen_input_and_rejects_an_altered_source(tmp_path):
    source = tmp_path / "changed.jsonl"
    source.write_bytes(sample_subset.FROZEN_JSONL.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        sample_subset.plan(source, tmp_path / "new.jsonl", 10, 42)
    with pytest.raises(ValueError, match="frozen input"):
        sample_subset.plan(sample_subset.FROZEN_JSONL, sample_subset.FROZEN_JSONL, 10, 42)
    with pytest.raises(ValueError, match="--count"):
        sample_subset.plan(sample_subset.FROZEN_JSONL, tmp_path / "new.jsonl", 1501, 42)


def external_input(tmp_path):
    input_path = tmp_path / "input.jsonl"
    sample_subset.main(["--output-jsonl", str(input_path), "--count", "1"])
    row = json.loads(input_path.read_text())
    root = tmp_path / "datasets" / row["dataset"]
    roots = {name: "" for name in ("scannet", "scannetpp", "arkitscenes")}
    roots[row["dataset"]] = str(root)
    from types import SimpleNamespace
    resolver = SimpleNamespace(**{name + "_root": value for name, value in roots.items()}, data_split="val")
    for frame in row["contexts_frame_id"] + [row["gt_frame_id"]]:
        path = Path(import_predictions.image_path(resolver, row["dataset"], row["scene_id"], frame))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), (10, 20, 30)).save(path)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    pred = predictions / (row["sample_id"] + ".png")
    Image.new("RGB", (512, 512), (30, 40, 50)).save(pred)
    args = ["--input-jsonl", str(input_path), "--predictions-dir", str(predictions),
            "--model", "my-model", "--generation-root", str(tmp_path / "generations"),
            "--" + row["dataset"] + "-root", str(root)]
    return row, input_path, pred, roots, args


def test_external_import_is_discovered_without_scores_or_image_conversion(tmp_path):
    row, input_path, pred, roots, args = external_input(tmp_path)
    import_predictions.main(args + ["--dry-run"])
    assert not (tmp_path / "generations").exists()
    import_predictions.main(args)
    pairs = discover_eval_pairs(str(tmp_path / "generations"), require_contexts=True,
                                **{name + "_root": value for name, value in roots.items()})
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.sample_id == row["sample_id"] and pair.shot == 6
    assert Path(pair.pred_path).read_bytes() == pred.read_bytes()
    assert len(pair.context_source_paths) == 6
    with Image.open(pair.gt_path) as gt:
        assert gt.size == (512, 512) and gt.mode == "RGB"
    assert pair.raw_record["ground_truth"]["source_path"] not in {pair.gt_path, pair.pred_path}
    assert not any("score" in key for key in pair.raw_record)
    with pytest.raises(ValueError, match="destination exists"):
        import_predictions.main(args)


def test_external_import_rejects_wrong_ids_metadata_and_non_rgb_predictions(tmp_path):
    row, input_path, pred, _, args = external_input(tmp_path)
    extra = pred.parent / "sample_wrong.png"
    Image.new("RGB", (512, 512)).save(extra)
    with pytest.raises(ValueError, match="Prediction IDs"):
        import_predictions.main(args)
    extra.unlink()
    Image.new("RGBA", (512, 512)).save(pred)
    with pytest.raises(ValueError, match="512x512 RGB PNG"):
        import_predictions.main(args)
    Image.new("RGB", (512, 512)).save(pred)
    altered = dict(row, gt_frame_id="wrong-target")
    input_path.write_text(json.dumps(altered) + "\n")
    with pytest.raises(ValueError, match="differs from the frozen"):
        import_predictions.main(args)
    input_path.write_text("[]\n")
    with pytest.raises(ValueError, match="record objects"):
        import_predictions.main(args)
    assert not (tmp_path / "generations").exists()


def test_ten_c6_report_uses_observed_task_macro_and_validates_every_selected_id(tmp_path):
    input_path = tmp_path / "selected.jsonl"
    sample_subset.main(["--output-jsonl", str(input_path), "--count", "10", "--seed", "42"])
    rows = [json.loads(line) for line in input_path.read_text().splitlines()]
    tasks = sorted({row["task_type"] for row in rows})
    assert len(tasks) < 7
    metrics = tmp_path / "metrics"
    for source in ("clip_dino", "qwen_vl", "vggt"):
        directory = metrics / "my-model" / source
        directory.mkdir(parents=True)
        results = []
        for row in rows:
            value = 10.0 * (tasks.index(row["task_type"]) + 1)
            values = {
                "clip_dino": {"dino_i": value / 100},
                "qwen_vl": dict(semantic_score_100=value, spatial_score_100=value,
                                visual_quality_score_100=value, judge_version="qwen_judge_v2",
                                parse_success=True, fidelity_parse_success=True, quality_parse_success=True),
                "vggt": dict(rscore_0_100=value, tscore_0_100=value,
                             num_contexts=10, context_source_shot=10, trans_error_unit="m"),
            }[source]
            results.append(dict(row, model="my-model", metric_success=True,
                                metric_config_sha256=source, **values))
        (directory / "results.json").write_text(json.dumps(results))
    env = dict(os.environ, PYTHON_BIN=sys.executable, INPUT_JSON=str(input_path),
               METRICS_ROOT=str(metrics), REPORT_ROOT=str(tmp_path / "report"),
               REPORT_PROTOCOL="selected", CONFIG_FILE="", SKIP_CONFIG="1")
    command = ["bash", str(EVALUATION / "run.sh"), "report", "my-model"]
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / "report/scores_by_model.json").read_text())["rows"][0]
    assert result["num_samples"] == 10
    assert result["avg6"] == pytest.approx(sum(range(1, len(tasks) + 1)) * 10 / len(tasks))
    records = json.loads((metrics / "my-model/vggt/results.json").read_text())
    (metrics / "my-model/vggt/results.json").write_text(json.dumps(records[:-1]))
    env["REPORT_ROOT"] = str(tmp_path / "missing-report")
    failed = subprocess.run(command, env=env, text=True, capture_output=True)
    assert failed.returncode != 0 and "Incomplete final coverage" in failed.stderr


def test_ten_c6_pairs_receive_canonical_c10_reference_contexts(tmp_path):
    pytest.importorskip("torch", reason="Metric discovery module requires PyTorch; no weights are loaded")
    from benchmarker_metrics.run import _apply_vggt_common_reference
    from benchmarker_metrics.common import EvalPair
    from types import SimpleNamespace
    output = tmp_path / "selected.jsonl"
    sample_subset.main(["--output-jsonl", str(output)])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    pairs = [EvalPair(model="my-model", run_name="sample10", run_dir=str(tmp_path),
                      dataset=row["dataset"], shot=row["shot"], task_type=row["task_type"],
                      sample_id=row["sample_id"], scene_id=row["scene_id"],
                      gt_frame_id=row["gt_frame_id"], anchor_frame_id=row["anchor_frame_id"],
                      pred_path="saved-pred.png", gt_path="saved-gt.jpg",
                      context_source_paths=[], context_vis_path=None, raw_record={}) for row in rows]
    args = SimpleNamespace(vggt_canonical_json=str(sample_subset.FROZEN_JSONL),
                           scannet_root="/data/scannet", scannetpp_root="/data/scannetpp",
                           arkitscenes_root="/data/arkitscenes", scan_root="", data_split="val")
    _apply_vggt_common_reference(pairs, args)
    assert all(len(pair.context_source_paths) == 10 and pair.gt_path == "saved-gt.jpg" for pair in pairs)
