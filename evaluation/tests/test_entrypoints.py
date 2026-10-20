import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

EVALUATION = Path(__file__).resolve().parents[1]
REPO = EVALUATION.parent
sys.path.insert(0, str(EVALUATION / "benchmark_runner"))
from benchmark.run import build_parser, generation_signature, prepare_run

spec = importlib.util.spec_from_file_location("released_report", EVALUATION / "benchmarker_metrics/build_metric_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def sample_environment(tmp_path):
    data = tmp_path / "data/scene/color"
    data.mkdir(parents=True)
    for frame in range(4):
        Image.new("RGB", (12, 10), (frame * 30, 10, 20)).save(data / f"{frame}.jpg")
    sample = dict(sample_id="sample_00000000", dataset="scannet", task_type="rotation",
                  scene_id="scene", gt_frame_id="3", anchor_frame_id="0", shot=3,
                  contexts_frame_id=["0", "1", "2"], instruction="Turn left.")
    input_json = tmp_path / "input.jsonl"
    input_json.write_text(json.dumps(sample) + "\n")
    env = dict(os.environ, PYTHON_BIN=sys.executable, INPUT_JSON=str(input_json),
               SCANNET_ROOT=str(tmp_path / "data"), GENERATION_ROOT=str(tmp_path / "generations"),
               NUM_GPUS="0", API_NUM_WORKERS="1", CONFIG_FILE="", CUDA_VISIBLE_DEVICES="")
    return input_json, env


def test_dummy_generation_and_safe_resume_through_public_entrypoint(tmp_path):
    input_json, env = sample_environment(tmp_path)
    for iteration in range(2):
        completed = subprocess.run(["bash", str(EVALUATION / "run.sh"), "generate", "dummy"],
                                   env=env, text=True, capture_output=True)
        assert completed.returncode == 0, completed.stderr
    run = tmp_path / "generations/dummy/dummy_input"
    rows = json.loads((run / "generation_results.json").read_text())
    assert len(rows) == 1 and rows[0]["status"] == "success"
    assert rows[0]["extra"]["skipped_completed"] is True
    assert Path(rows[0]["pred_image"]).is_file()
    env["SEED"] = "123"
    completed = subprocess.run(["bash", str(EVALUATION / "run.sh"), "generate", "dummy"],
                               env=env, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "configuration or input changed" in completed.stderr


def generation_args(input_json):
    return build_parser().parse_args(["--input_json", str(input_json), "--model", "dummy", "--num_gpus", "0"])


def test_generation_signature_rejects_input_and_weight_changes_but_allows_worker_changes(tmp_path):
    input_json, _ = sample_environment(tmp_path)
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"first")
    args = generation_args(input_json)
    args.bagel_checkpoint_path = str(weights)
    run = tmp_path / "run"
    prepare_run(args, run)
    initial = generation_signature(args)
    args.num_gpus, args.num_workers, args.gpus, args.rank = 8, 8, "3,4", 5
    assert generation_signature(args) == initial
    args.generator_device = "cpu"
    assert generation_signature(args) != initial
    args.generator_device = "cuda"
    prepare_run(args, run)
    weights.write_bytes(b"different")
    with pytest.raises(ValueError, match="configuration or input changed"):
        prepare_run(args, run)
    weights.write_bytes(b"first")
    prepare_run(args, tmp_path / "another")
    input_json.write_text(input_json.read_text().replace("Turn left", "Turn right"))
    with pytest.raises(ValueError, match="configuration or input changed"):
        prepare_run(args, tmp_path / "another")


def selected_input(tmp_path):
    frozen = [json.loads(line) for line in (EVALUATION / "data/merged_dataset_final.jsonl").read_text().splitlines()]
    core = frozen[1]
    selected = copy.deepcopy(frozen[2])
    selected["contexts_frame_id"] = list(core["contexts_frame_id"])
    path = tmp_path / "selected.jsonl"
    path.write_text(json.dumps(selected) + "\n")
    return selected, path


def test_selected_protocol_preserves_frozen_identity_and_core(tmp_path):
    row, path = selected_input(tmp_path)
    config = json.loads((EVALUATION / "benchmarker_metrics/metric_report_config.json").read_text())
    effective = report.selected_protocol_config(config, path)
    report.validate_dataset(path, effective["benchmark"])
    assert effective["benchmark"]["shots"] == [10]
    with pytest.raises(report.ReportError, match="Context count mismatch"):
        report.validate_dataset(path, config["benchmark"])
    for field, value in [("gt_frame_id", "wrong"), ("task_type", "rotation" if row["task_type"] != "rotation" else "translation")]:
        changed = dict(row, **{field: value})
        path.write_text(json.dumps(changed) + "\n")
        with pytest.raises(report.ReportError, match="identity mismatch"):
            report.selected_protocol_config(config, path)
    changed = dict(row, contexts_frame_id=row["contexts_frame_id"][:-1])
    path.write_text(json.dumps(changed) + "\n")
    with pytest.raises(report.ReportError, match="preserve the frozen C6 core"):
        report.selected_protocol_config(config, path)


def test_report_selected_model_cli_reads_complete_metrics_and_writes_outside_source(tmp_path):
    row, input_json = selected_input(tmp_path)
    metrics = tmp_path / "metrics"
    for source, values in {
        "clip_dino": {"dino_i": .75},
        "qwen_vl": dict(semantic_score_100=70, spatial_score_100=60, visual_quality_score_100=80,
                        judge_version="qwen_judge_v2", parse_success=True, fidelity_parse_success=True, quality_parse_success=True),
        "vggt": dict(rscore_0_100=90, tscore_0_100=85, num_contexts=10, context_source_shot=10, trans_error_unit="m")
    }.items():
        directory = metrics / "bagel" / source
        directory.mkdir(parents=True)
        result = dict(row, model="bagel", metric_success=True, metric_config_sha256=source, **values)
        (directory / "results.json").write_text(json.dumps([result]))
    env = dict(os.environ, PYTHON_BIN=sys.executable, INPUT_JSON=str(input_json), METRICS_ROOT=str(metrics),
               REPORT_ROOT=str(tmp_path / "reports"), REPORT_PROTOCOL="selected", CONFIG_FILE="")
    completed = subprocess.run(["bash", str(EVALUATION / "run.sh"), "report", "bagel"], env=env,
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / "reports/scores_by_model.json").read_text())["rows"][0]
    assert result["avg6"] == pytest.approx(460 / 6)
    assert result["num_samples"] == 1


def test_dreamomni_missing_upstream_runtime_fails_without_inference(tmp_path):
    _, env = sample_environment(tmp_path)
    completed = subprocess.run(["bash", str(EVALUATION / "run.sh"), "generate", "dreamomni2"],
                               env=env, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "dreamomni2/pipeline_dreamomni2.py" in completed.stderr


def test_dreamomni_public_wrapper_preserves_recorded_generation_parameters(monkeypatch):
    from benchmark import dreamomni_runtime as runtime
    from types import SimpleNamespace
    calls = {}

    class Generator:
        def __init__(self, device):
            calls["device"] = device
        def manual_seed(self, seed):
            calls["seed"] = seed
            return self

    def pipe(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(images=["prediction"])

    monkeypatch.setattr(runtime, "torch", SimpleNamespace(Generator=Generator), raising=False)
    monkeypatch.setattr(runtime, "load_image", lambda path: path, raising=False)
    monkeypatch.setattr(runtime, "resizeinput", lambda image: image, raising=False)
    assert runtime.infer_image(["view2", "view1"], "refined", pipe, 1234, "cuda") == "prediction"
    assert calls["images"] == ["view2", "view1"]
    assert (calls["num_inference_steps"], calls["guidance_scale"], calls["seed"]) == (30, 3.5, 1234)
    assert calls["height"] == calls["width"] == 1024
    assert runtime.refine_prompt("<img> example </img>") == "example"


def test_dreamomni_vlm_wrapper_matches_processor_and_model_contract(monkeypatch):
    from benchmark import dreamomni_runtime as runtime
    calls = {}

    class Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]
        def to(self, device):
            calls["device"] = device
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls["messages"], calls["template_kwargs"] = messages, kwargs
            return "template"
        def __call__(self, **kwargs):
            calls["processor_kwargs"] = kwargs
            return Inputs(input_ids=[[1, 2, 3]])
        def batch_decode(self, sequences, **kwargs):
            calls["decoded"] = sequences
            return ["<img> refined </img>"]

    class Model:
        def generate(self, **kwargs):
            calls["generate_kwargs"] = kwargs
            return [[1, 2, 3, 8, 9]]

    monkeypatch.setattr(runtime, "process_vision_info", lambda messages: (["image2", "image1"], None), raising=False)
    raw, refined = runtime.infer_vlm(["path2", "path1"], "Render target.", Model(), Processor(), "cuda")
    content = calls["messages"][0]["content"]
    assert [item["image"] for item in content[:2]] == ["path2", "path1"]
    assert content[2]["text"] == "Render target. It is generation task."
    assert calls["generate_kwargs"]["do_sample"] is False
    assert calls["generate_kwargs"]["max_new_tokens"] == 4096
    assert calls["processor_kwargs"]["images"] == ["image2", "image1"]
    assert calls["decoded"] == [[8, 9]] and calls["device"] == "cuda"
    assert raw == "<img> refined </img>" and refined == "refined"


def test_child_workflow_can_skip_config_without_overwriting_runtime_input(tmp_path):
    input_json, env = sample_environment(tmp_path)
    config = tmp_path / "config.env"
    config.write_text('INPUT_JSON=/missing/config-input.jsonl\nGENERATION_ROOT=/missing/config-output\n')
    env.update(CONFIG_FILE=str(config), SKIP_CONFIG="1")
    completed = subprocess.run(["bash", str(EVALUATION / "run.sh"), "generate", "dummy"],
                               env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "generations/dummy/dummy_input/generation_results.json").is_file()
