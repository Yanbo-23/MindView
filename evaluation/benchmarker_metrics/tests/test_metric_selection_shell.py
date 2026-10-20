import json
import os
import subprocess
import sys
from pathlib import Path


EVALUATION = Path(__file__).resolve().parents[2]
TASK_SCRIPT = EVALUATION / "benchmarker_metrics/task_run_final_paper.sh"
RUNNER_SCRIPT = EVALUATION / "benchmarker_metrics/run_metrics.sh"


def metric_environment(tmp_path):
    source = tmp_path / "generations/bagel/run"
    source.mkdir(parents=True)
    (source / "generation_results.json").write_text(json.dumps([{
        "sample_id": "sample_0", "shot": 10, "dataset": "scannet",
        "task_type": "rotation", "status": "success"
    }]))
    capture = tmp_path / "captured.jsonl"
    interpreter = tmp_path / "fake python"
    interpreter.write_text(f"#!{sys.executable}\n" + r'''
import json, os, sys
if len(sys.argv) > 1 and sys.argv[1] == "-":
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
with open(os.environ["CAPTURE_PATH"], "a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\n")
''')
    interpreter.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(interpreter),
               GENERATION_ROOT=str(tmp_path / "generations"),
               METRICS_ROOT=str(tmp_path / "metrics"), CAPTURE_PATH=str(capture),
               CONFIG_FILE="", CUDA_VISIBLE_DEVICES="", NUM_GPUS="0")
    return env, capture


def test_shell_syntax():
    for script in EVALUATION.rglob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_selected_metric_and_model_use_requested_interpreter_and_output(tmp_path):
    env, capture = metric_environment(tmp_path)
    env.update(RUN_CLIP_DINO="0", RUN_QWEN_VL="1", RUN_VGGT="0")
    completed = subprocess.run(["bash", str(TASK_SCRIPT), "bagel"], env=env,
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    commands = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(commands) == 2
    run, summary = commands
    assert run[:2] == ["-m", "benchmarker_metrics.run"]
    assert run[run.index("--metric") + 1] == "qwen_vl"
    assert run[run.index("--models") + 1] == "bagel"
    assert run[run.index("--metrics_root") + 1] == str(tmp_path / "metrics/bagel")
    assert "--skip_completed" in run
    assert summary[-2:] == ["--metrics", "qwen_vl"]


def test_no_metrics_or_conflicting_resume_flags_fail_before_model_dispatch(tmp_path):
    for flags in [dict(RUN_CLIP_DINO="0", RUN_QWEN_VL="0", RUN_VGGT="0"),
                  dict(OVERWRITE="1", SKIP_COMPLETED="1")]:
        case = tmp_path / str(len(list(tmp_path.iterdir())))
        env, capture = metric_environment(case)
        env.update(flags)
        completed = subprocess.run(["bash", str(TASK_SCRIPT), "bagel"], env=env,
                                   text=True, capture_output=True)
        assert completed.returncode != 0
        assert not capture.exists()


def test_duplicate_models_fail_before_dispatch(tmp_path):
    env, capture = metric_environment(tmp_path)
    completed = subprocess.run(["bash", str(TASK_SCRIPT), "bagel", "bagel"], env=env,
                               text=True, capture_output=True)
    assert completed.returncode != 0
    assert "Duplicate model" in completed.stderr
    assert not capture.exists()
