"""Exercise the public GPU orchestration contract with isolated fake model runtimes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SingleGpuTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mindview orchestration ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "release"
        (self.root / "evaluation/data").mkdir(parents=True)
        (self.root / "understanding").mkdir()
        for name in ("test_single_gpu.sh", "evaluation/sample_subset.py",
                     "evaluation/data/merged_dataset_final.jsonl", "evaluation/data/manifest.json"):
            shutil.copy2(ROOT / name, self.root / name)
        model = self.root / "models/base"
        model.mkdir(parents=True)
        for name in ("llm_config.json", "vit_config.json", "tokenizer_config.json",
                     "tokenizer.json", "vocab.json", "merges.txt", "ae.safetensors"):
            (model / name).write_text("fixture")
        checkpoint = self.root / "models/sft/ema.safetensors"
        checkpoint.parent.mkdir()
        checkpoint.write_text("fixture EMA; never loaded by a real model")
        self.events = Path(self.temporary.name) / "events.jsonl"
        stub = self.root / "stub.py"
        stub.write_text('''import json, os, sys
from pathlib import Path
kind, *args = sys.argv[1:]
if '--dry-run' in args:
    print('dry-run understanding plan'); raise SystemExit(0)
stage = args[0] if kind == 'evaluation' else ('prepare' if '--prepare-only' in args else 'understanding')
keys = ('CUDA_VISIBLE_DEVICES', 'NUM_GPUS', 'UNDERSTANDING_NPROC_PER_NODE',
        'SKIP_CONFIG', 'IMAGE_SIZE', 'MAX_SAMPLES', 'SHOTS', 'REPORT_PROTOCOL',
        'REPORT_CONFIG', 'RUN_CLIP_DINO', 'RUN_QWEN_VL', 'RUN_VGGT', 'SEED',
        'VGGT_CANONICAL_JSON', 'VGGT_COMMON_REFERENCE_GENERATION_RESULTS')
def record(name):
    with open(os.environ['TEST_EVENT_LOG'], 'a') as stream:
        stream.write(json.dumps(dict(stage=name, argv=args, env={k:os.environ.get(k) for k in keys}))+'\\n')
    if os.environ.get('TEST_FAIL_STAGE') == name: raise SystemExit(7)
if stage != 'understanding': record(stage)
if stage == 'report':
    path=Path(os.environ['REPORT_ROOT'])/'report_summary.json'
    subset=json.loads(Path(os.environ['INPUT_JSON']).with_suffix('.manifest.json').read_text())
    report=dict(models=['bagel-sft-final'],dataset=dict(num_samples=10,sha256=subset['subset_sha256']),
                score_fields=['dino_i','semantic','spatial','quality','vggt_rscore','vggt_tscore'])
    if os.environ.get('TEST_BAD_SUMMARY')=='mindview':report['dataset']['num_samples']=9
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report))
if stage == 'understanding':
    for name in ('sat-v2','mmsi','mindcube-tiny'): record(name)
    path=Path(os.environ['UNDERSTANDING_OUTPUT_ROOT'])
    if '--limit' in args: path=path/'partial'/('limit_'+args[args.index('--limit')+1])
    status='partial' if '--limit' in args else 'complete'
    benchmarks={name:dict(status=status,checkpoint=dict(checkpoint_path=os.environ['BAGEL_SFT_FINAL_CHECKPOINT']))
                for name in ('sat-v2','mmsi','mindcube-tiny')}
    summary=dict(status=status,selected_runs_complete=True,benchmarks=benchmarks)
    if os.environ.get('TEST_BAD_SUMMARY')=='understanding':summary['selected_runs_complete']=False
    path.mkdir(parents=True,exist_ok=True); (path/'summary.json').write_text(json.dumps(summary))
''')
        interpreter = self.root / "fake-python"
        interpreter.write_text(f"#!{sys.executable}\nimport os,sys\nsys.stdin.read()\n"
                               "raise SystemExit(3 if os.environ.get('TEST_NO_GPU') else 0)\n")
        interpreter.chmod(0o755)
        for kind in ("evaluation", "understanding"):
            wrapper = self.root / kind / "run.sh"
            wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                               f'exec "{sys.executable}" "{stub}" {kind} "$@"\n')
        config = self.root / "config.env"
        values = dict(PYTHON_BIN=sys.executable, BAGEL_PYTHON=str(interpreter),
                      CLIP_DINO_PYTHON=str(interpreter), QWEN_VL_PYTHON=str(interpreter),
                      VGGT_PYTHON=str(interpreter), EVAL_PYTHON=str(interpreter),
                      BAGEL_MODEL_PATH=str(model), BAGEL_SFT_FINAL_CHECKPOINT=str(checkpoint),
                      NUM_GPUS="8", UNDERSTANDING_NPROC_PER_NODE="8", CUDA_VISIBLE_DEVICES="0,1,2,3",
                      REPORT_CONFIG="/wrong/report.json", IMAGE_SIZE="1024", MAX_SAMPLES="1",
                      SHOTS="3", REPORT_PROTOCOL="paper", RUN_VGGT="0",
                      VGGT_COMMON_REFERENCE_GENERATION_RESULTS="/old/reference.json")
        for dataset in ("SCANNET", "SCANNETPP", "ARKITSCENES"):
            path = self.root / "resources" / dataset.lower()
            path.mkdir(parents=True)
            values[dataset + "_ROOT"] = str(path)
        config.write_text("\n".join(f'{k}="{v}"' for k, v in values.items()) + "\n")
        self.config = config
        self.output = self.root / "outputs/acceptance"
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TEST_EVENT_LOG=str(self.events),
                        WORLD_SIZE="1", RANK="0")

    def run_test_script(self, *args, **environment):
        return subprocess.run(["bash", str(self.root / "test_single_gpu.sh"),
                               "--config", str(self.config), "--gpu", "7",
                               "--output-root", str(self.output), *args],
                              env=dict(self.env, **environment), text=True, capture_output=True)

    def read_events(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()]

    def test_full_sequence_forces_one_gpu_and_all_metrics_despite_config(self):
        result = self.run_test_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.read_events()
        self.assertEqual([e["stage"] for e in events],
                         ["prepare", "generate", "metrics", "report", "sat-v2", "mmsi", "mindcube-tiny"])
        for event in events:
            env = event["env"]
            self.assertEqual((env["CUDA_VISIBLE_DEVICES"], env["NUM_GPUS"], env["UNDERSTANDING_NPROC_PER_NODE"]), ("7", "1", "1"))
            self.assertEqual((env["SKIP_CONFIG"], env["IMAGE_SIZE"], env["MAX_SAMPLES"], env["SHOTS"]), ("1", "512", "-1", "6"))
            self.assertEqual((env["RUN_CLIP_DINO"], env["RUN_QWEN_VL"], env["RUN_VGGT"]), ("1", "1", "1"))
            self.assertEqual(env["REPORT_PROTOCOL"], "selected")
            self.assertEqual(env["SEED"], "42")
            self.assertEqual(env["REPORT_CONFIG"], str(self.root / "evaluation/benchmarker_metrics/metric_report_config.json"))
            self.assertIsNone(env["VGGT_COMMON_REFERENCE_GENERATION_RESULTS"])
        subset = [json.loads(line) for line in (self.output / "subset.jsonl").read_text().splitlines()]
        self.assertEqual(len(subset), 10)
        self.assertEqual({row["shot"] for row in subset}, {6})
        self.assertEqual(json.loads((self.output / "test_summary.json").read_text())["understanding_scope"], "full")

    def test_failure_stops_before_report_and_understanding(self):
        result = self.run_test_script(TEST_FAIL_STAGE="metrics")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual([e["stage"] for e in self.read_events()], ["prepare", "generate", "metrics"])
        self.assertFalse((self.output / "test_summary.json").exists())

    def test_partial_results_are_read_from_the_partial_directory(self):
        result = self.run_test_script("--understanding-limit", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((self.output / "test_summary.json").read_text())
        self.assertIn("partial", summary["understanding_scope"])
        self.assertTrue(summary["reports"][1].endswith("/partial/limit_2/summary.json"))

    def test_dry_run_creates_no_outputs_and_does_not_touch_gpu(self):
        result = self.run_test_script("--dry-run", TEST_NO_GPU="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.events.exists())
        self.assertIn('"count": 10', result.stdout)

    def test_gpu_preflight_fails_before_writing_outputs(self):
        result = self.run_test_script(TEST_NO_GPU="1")
        self.assertEqual(result.returncode, 3)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.events.exists())

    def test_successful_commands_with_incomplete_summaries_are_rejected(self):
        for source in ('mindview', 'understanding'):
            with self.subTest(source=source):
                result = self.run_test_script(TEST_BAD_SUMMARY=source)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.output / 'test_summary.json').exists())

    def test_multiple_devices_and_protected_outputs_are_rejected(self):
        for arguments in (("--gpu", "0,1"), ("--output-root", str(self.root / "evaluation"))):
            result = self.run_test_script(*arguments)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.events.exists())


if __name__ == "__main__":
    unittest.main()
