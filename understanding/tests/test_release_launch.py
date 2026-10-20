import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from understanding import eval_spatial_benchmarks as evaluator
from understanding.launch import BENCHMARKS, HERE, ROOT, build_plan, execution_environment, main, parser, results_root
from understanding.spatial_benchmarks import BENCHMARKS as REGISTRY
from understanding.spatial_benchmarks.bagel_model import _load_image_transform_identity, checkpoint_identity
from understanding.spatial_benchmarks.runtime import DistributedRuntime


class ReleaseLaunchTest(unittest.TestCase):
    def args(self, *arguments):
        with patch.dict(os.environ, {}, clear=True):
            return parser().parse_args(list(arguments))

    def test_default_all_is_exactly_three_full_protocols(self):
        with patch.dict(os.environ, {"SAT_V2_LIMIT": "1", "MMSI_LIMIT": "1", "MINDCUBE_LIMIT": "1", "MODEL": "vanilla"}, clear=True):
            plan = build_plan(parser().parse_args([]))
        self.assertEqual(set(REGISTRY), set(BENCHMARKS))
        self.assertEqual([x["benchmark"] for x in plan], list(BENCHMARKS))
        self.assertEqual([x["evaluation_items"] for x in plan], [300, 1000, 1050])
        self.assertEqual([x["max_new_tokens"] for x in plan], [1024, 2048, 4096])
        self.assertTrue(all(x["partial"] is False for x in plan))

    def test_data_download_requires_explicit_flag(self):
        offline = build_plan(self.args("sat-v2", "--prepare-only"))[0]["command"]
        allowed = build_plan(self.args("mmsi", "--prepare-only", "--allow-download"))[0]["command"]
        self.assertIn("--offline", offline)
        self.assertNotIn("--allow-download", offline)
        self.assertIn("--allow-download", allowed)
        self.assertNotIn("--offline", allowed)
        for name in BENCHMARKS[:2]:
            args = evaluator.parse_args(["--benchmark", name, "--prepare-only"])
            self.assertTrue(args.offline)

    def test_partial_scope_is_separate_and_limit_precedes_sat_expansion(self):
        args = self.args("all", "--output-root", "/tmp/understanding-fixture", "--limit", "2")
        self.assertEqual(results_root(args), Path("/tmp/understanding-fixture/partial/limit_2"))
        plan = build_plan(args)
        self.assertEqual([x["evaluation_items"] for x in plan], [4, 2, 2])
        self.assertTrue(all(x["partial"] for x in plan))

    def test_single_process_ignores_inherited_torchrun_ranks(self):
        args = self.args("all", "--nproc-per-node", "1", "--gpus", "2")
        with patch.dict(os.environ, {"RANK": "7", "WORLD_SIZE": "8", "LOCAL_RANK": "7"}):
            environment = execution_environment(args)
        self.assertEqual((environment["RANK"], environment["WORLD_SIZE"], environment["LOCAL_RANK"]), ("0", "1", "0"))
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")

    def test_dry_run_needs_no_weights_data_torch_or_config(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "no-results"
            environment = dict(os.environ, EVAL_PYTHON=sys.executable,
                               SKIP_CONFIG="1", CONFIG_FILE="/nonexistent/config.env",
                               PYTHONDONTWRITEBYTECODE="1", UNDERSTANDING_OUTPUT_ROOT=str(output))
            for name in ("BAGEL_MODEL_PATH", "BAGEL_SFT_FINAL_CHECKPOINT", "SAT_V2_PARQUET", "MMSI_PARQUET", "MINDCUBE_DATA_ROOT"):
                environment.pop(name, None)
            result = subprocess.run(["bash", str(HERE / "run.sh"), "all", "--dry-run"],
                                    env=environment, cwd=directory, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"evaluation_items": 300', result.stdout)
            self.assertIn('"evaluation_items": 1000', result.stdout)
            self.assertIn('"evaluation_items": 1050', result.stdout)
            self.assertIn('"partial": false', result.stdout)
            self.assertFalse(output.exists())

    def test_prepare_only_without_local_data_refuses_before_subprocess(self):
        with patch.dict(os.environ, {}, clear=True), patch("understanding.launch.subprocess.run") as run:
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                code = main(["sat-v2", "--prepare-only"])
        self.assertEqual(code, 2)
        self.assertIn("SAT_V2_PARQUET", stderr.getvalue())
        run.assert_not_called()

    def test_prepare_only_runtime_does_not_import_torch_or_write_scores(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"torch": None}):
            runtime = DistributedRuntime(prepare_only=True)
            self.assertEqual(runtime.primary_call(lambda: 7, label="fixture"), 7)
            runtime.close()
            adapter = types.SimpleNamespace(name="sat-v2", tasks=[], manifest_fragment={"selected_source_rows": 0})
            args = evaluator.parse_args(["--benchmark", "sat-v2", "--prepare-only", "--output-dir", str(Path(directory) / "results")])
            with patch.object(REGISTRY["sat-v2"].adapter_class, "from_args", return_value=adapter), \
                 patch.object(evaluator, "checkpoint_identity", side_effect=AssertionError("weights must stay unread")), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluator.run(args), 0)
            self.assertFalse((Path(directory) / "results").exists())

    def test_original_understanding_transform_is_preserved(self):
        identity = _load_image_transform_identity()
        self.assertEqual(identity["args"], {"max_image_size": 980, "min_image_size": 378,
                                           "image_stride": 14, "max_pixels": 2007040})
        self.assertEqual(Path(identity["config_path"]), ROOT / "understanding/image_transform.yaml")

    def test_checkpoint_identity_accepts_new_release_bytes_without_private_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "base"
            model.mkdir()
            for name in ("llm_config.json", "vit_config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"):
                (model / name).write_text("fixture")
            checkpoint = Path(directory) / "ema.safetensors"
            checkpoint.write_bytes(b"new public BF16 checkpoint fixture")
            identity = checkpoint_identity(model, checkpoint, allow_incomplete_checkpoint=False)
            self.assertEqual(identity["checkpoint_size_bytes"], checkpoint.stat().st_size)
            self.assertEqual(len(identity["checkpoint_sampled_sha256"]), 64)
            self.assertEqual(len(identity["assets"]), 6)


if __name__ == "__main__":
    unittest.main()
