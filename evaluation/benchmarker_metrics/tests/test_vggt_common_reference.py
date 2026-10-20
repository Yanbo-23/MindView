import json
from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="Metric module tests require PyTorch; no model weights are loaded")

from benchmarker_metrics.common import EvalPair
from benchmarker_metrics.run import (
    _apply_vggt_common_reference,
    _discover_for_metric,
    build_parser,
)


def test_canonical_json_restores_ten_contexts_and_keeps_saved_gt(tmp_path):
    pair = _pair(tmp_path, shot=10, run_name="selected", contexts=6)
    original_gt = pair.gt_path
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text(json.dumps({
        "sample_id": "frozen_c10", "dataset": "scannet", "shot": 10,
        "task_type": "rotation", "scene_id": "scene0000_00",
        "gt_frame_id": "100", "anchor_frame_id": "90",
        "contexts_frame_id": list(map(str, range(10))),
    }) + "\n")
    args = Namespace(vggt_canonical_json=str(canonical), scan_root="",
                     scannet_root=str(tmp_path / "source"), scannetpp_root="",
                     arkitscenes_root="", data_split="val")
    _apply_vggt_common_reference([pair], args)
    assert len(pair.context_source_paths) == 10
    assert pair.gt_path == original_gt
    assert pair.raw_record["ground_truth"]["source_path"] == str(tmp_path / "source/scene0000_00/color/100.jpg")
    assert pair.raw_record["vggt_context_source_shot"] == 10


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path.resolve())


def _pair(tmp_path: Path, *, shot: int, run_name: str, contexts: int) -> EvalPair:
    pred_path = _touch(tmp_path / f"{run_name}_pred.png")
    gt_path = _touch(tmp_path / f"{run_name}_gt.jpg")
    context_paths = [
        _touch(tmp_path / f"{run_name}_context_{index}.jpg")
        for index in range(contexts)
    ]
    return EvalPair(
        model="bagel",
        run_name=run_name,
        run_dir=str(tmp_path),
        dataset="scannet",
        shot=shot,
        task_type="rotation",
        sample_id=f"sample_{shot}",
        scene_id="scene0000_00",
        gt_frame_id="100",
        anchor_frame_id="90",
        pred_path=pred_path,
        gt_path=gt_path,
        context_source_paths=context_paths,
        context_vis_path=None,
        raw_record={
            "ground_truth": {"image_path": gt_path, "source_path": gt_path},
            "vggt_original_num_contexts": contexts,
            "vggt_context_source_shot": shot,
        },
    )


def _reference_record(tmp_path: Path, *, contexts: int = 10, sample_id: str = "ref"):
    source_paths = [
        _touch(tmp_path / f"reference_context_{index}.jpg")
        for index in range(contexts)
    ]
    gt_path = _touch(tmp_path / "reference_gt.jpg")
    gt_source_path = _touch(tmp_path / "reference_gt_source.jpg")
    return {
        "dataset": "scannet",
        "task_type": "rotation",
        "scene_id": "scene0000_00",
        "gt_frame_id": "100",
        "anchor_frame_id": "90",
        "sample_id": sample_id,
        "shot": 10,
        "status": "success",
        "pred_image": _touch(tmp_path / "must_not_be_used.png"),
        "gt_image": gt_path,
        "ground_truth": {
            "image_path": gt_path,
            "source_path": gt_source_path,
        },
        "contexts": {"source_paths": source_paths},
    }


def _write_reference(tmp_path: Path, records) -> str:
    path = tmp_path / "reference_generation_results.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return str(path)


def test_default_parameter_is_disabled_and_does_not_mutate_pairs(tmp_path):
    args = build_parser().parse_args(
        ["--metric", "vggt", "--benchmark_output_root", str(tmp_path)]
    )
    assert not hasattr(args, "vggt_common_reference_generation_results")

    pair = _pair(tmp_path, shot=6, run_name="c6", contexts=6)
    before = (
        pair.pred_path,
        pair.gt_path,
        list(pair.context_source_paths),
        dict(pair.raw_record),
    )
    _apply_vggt_common_reference([pair], args)
    after = (
        pair.pred_path,
        pair.gt_path,
        list(pair.context_source_paths),
        dict(pair.raw_record),
    )
    assert after == before


def test_default_discovery_keeps_legacy_max_shot_contexts(tmp_path, monkeypatch):
    c6_pair = _pair(tmp_path, shot=6, run_name="shared_run", contexts=6)
    c10_pair = _pair(tmp_path, shot=10, run_name="shared_run", contexts=10)
    c6_prediction = c6_pair.pred_path
    expected_contexts = list(c10_pair.context_source_paths)

    monkeypatch.setattr(
        "benchmarker_metrics.run.discover_eval_pairs",
        lambda **_kwargs: [c6_pair, c10_pair],
    )
    args = build_parser().parse_args(
        [
            "--metric",
            "vggt",
            "--benchmark_output_root",
            str(tmp_path),
            "--shots",
            "6",
        ]
    )

    pairs = _discover_for_metric(args, require_contexts=True)

    assert pairs == [c6_pair]
    assert c6_pair.pred_path == c6_prediction
    assert c6_pair.context_source_paths == expected_contexts
    assert c6_pair.raw_record["vggt_original_num_contexts"] == 6
    assert c6_pair.raw_record["vggt_context_source_shot"] == 10
    assert "vggt_common_reference_active" not in c6_pair.raw_record


def test_common_reference_replaces_only_reference_inputs(tmp_path):
    reference = _reference_record(tmp_path)
    reference_path = _write_reference(tmp_path, [reference])
    args = Namespace(vggt_common_reference_generation_results=reference_path)
    pairs = [
        _pair(tmp_path, shot=6, run_name="c6", contexts=6),
        _pair(tmp_path, shot=10, run_name="oracle", contexts=7),
    ]
    original_predictions = [pair.pred_path for pair in pairs]
    original_samples = [pair.sample_id for pair in pairs]

    _apply_vggt_common_reference(pairs, args)

    expected_contexts = reference["contexts"]["source_paths"]
    for index, pair in enumerate(pairs):
        assert pair.pred_path == original_predictions[index]
        assert pair.sample_id == original_samples[index]
        assert pair.context_source_paths == expected_contexts
        assert len(pair.context_source_paths) == 10
        assert pair.gt_path == reference["gt_image"]
        assert (
            pair.raw_record["ground_truth"]["source_path"]
            == reference["ground_truth"]["source_path"]
        )
        assert pair.raw_record["vggt_original_num_contexts"] == (6 if index == 0 else 7)
        assert pair.raw_record["vggt_context_source_shot"] == 10
        assert pair.raw_record["vggt_common_reference_active"] is True
        assert pair.raw_record["vggt_common_reference_sample_id"] == "ref"
    assert all(pair.pred_path != reference["pred_image"] for pair in pairs)


@pytest.mark.parametrize(
    ("dataset", "scene_id", "reference_anchor", "pair_anchor"),
    [
        (
            "arkitscenes",
            "41159557",
            "41159557_33251.256",
            "33251.256",
        ),
        (
            "arkitscenes",
            "41159557",
            "33251.256",
            "41159557_33251.256",
        ),
        ("scannet", "scene0000_00", "", "None"),
    ],
)
def test_common_reference_normalizes_equivalent_anchor_encodings(
    tmp_path,
    dataset,
    scene_id,
    reference_anchor,
    pair_anchor,
):
    reference = _reference_record(tmp_path)
    reference.update(
        dataset=dataset,
        scene_id=scene_id,
        anchor_frame_id=reference_anchor,
    )
    pair = _pair(tmp_path, shot=10, run_name="candidate", contexts=10)
    pair.dataset = dataset
    pair.scene_id = scene_id
    pair.anchor_frame_id = pair_anchor
    args = Namespace(
        vggt_common_reference_generation_results=_write_reference(
            tmp_path, [reference]
        )
    )

    _apply_vggt_common_reference([pair], args)

    assert pair.anchor_frame_id == pair_anchor
    assert pair.raw_record["vggt_common_reference_active"] is True
    assert pair.raw_record["vggt_common_reference_sample_id"] == "ref"


@pytest.mark.parametrize(
    ("dataset", "scene_id", "reference_anchor", "pair_anchor"),
    [
        ("scannet", "scene0000_00", "scene0000_00_90", "90"),
        ("arkitscenes", "41159557", "other_scene_33251.256", "33251.256"),
        ("arkitscenes", "41159557", "41159557_33251.256", "33252.055"),
    ],
)
def test_common_reference_keeps_distinct_anchor_ids_separate(
    tmp_path,
    dataset,
    scene_id,
    reference_anchor,
    pair_anchor,
):
    reference = _reference_record(tmp_path)
    reference.update(
        dataset=dataset,
        scene_id=scene_id,
        anchor_frame_id=reference_anchor,
    )
    pair = _pair(tmp_path, shot=10, run_name="candidate", contexts=10)
    pair.dataset = dataset
    pair.scene_id = scene_id
    pair.anchor_frame_id = pair_anchor
    args = Namespace(
        vggt_common_reference_generation_results=_write_reference(
            tmp_path, [reference]
        )
    )

    with pytest.raises(ValueError, match="missing 1 selected semantic keys"):
        _apply_vggt_common_reference([pair], args)


def test_common_reference_rejects_duplicate_after_anchor_normalization(tmp_path):
    reference = _reference_record(tmp_path)
    reference.update(
        dataset="arkitscenes",
        scene_id="41159557",
        anchor_frame_id="41159557_33251.256",
    )
    duplicate = dict(
        reference,
        sample_id="duplicate",
        anchor_frame_id="33251.256",
    )
    pair = _pair(tmp_path, shot=10, run_name="candidate", contexts=10)
    pair.dataset = "arkitscenes"
    pair.scene_id = "41159557"
    pair.anchor_frame_id = "33251.256"
    args = Namespace(
        vggt_common_reference_generation_results=_write_reference(
            tmp_path, [reference, duplicate]
        )
    )

    with pytest.raises(ValueError, match="Duplicate VGGT common reference"):
        _apply_vggt_common_reference([pair], args)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "nine_contexts"])
def test_common_reference_fails_closed(tmp_path, failure):
    pair = _pair(tmp_path, shot=6, run_name="c6", contexts=6)
    reference = _reference_record(
        tmp_path,
        contexts=9 if failure == "nine_contexts" else 10,
    )
    if failure == "missing":
        reference["gt_frame_id"] = "different"
        records = [reference]
    elif failure == "duplicate":
        records = [reference, dict(reference, sample_id="duplicate")]
    else:
        records = [reference]
    args = Namespace(
        vggt_common_reference_generation_results=_write_reference(tmp_path, records)
    )

    with pytest.raises((ValueError, FileNotFoundError)):
        _apply_vggt_common_reference([pair], args)
