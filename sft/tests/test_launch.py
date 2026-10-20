import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from train.launch import MODEL_ASSETS, SFT_ROOT, parser, prepare
from data.mindview_schema import MindViewDataError


class LaunchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "official model"
        self.model.mkdir()
        for name in MODEL_ASSETS:
            (self.model / name).write_bytes(b"model fixture")
        self.images = self.root / "images"
        self.images.mkdir()
        self.snapshot = self.root / "snapshot.jsonl"
        self.benchmark = self.root / "benchmark.jsonl"
        self.benchmark.write_text(json.dumps(dict(dataset="scannet", scene_id="heldout_scene")) + "\n")
        self.output = self.root / "run"
        self.rows = [dict(
            schema_version="mindview_bagel_single_shot_v1", preprocess_id="center_square_then_bagel_resize_v1",
            shot=6, record_uid=hashlib.sha256(f"record-{i}".encode()).hexdigest(),
            dataset="scannet", source_family="cam_loc", task_type="composition", scene_id="scene0000_00",
            source_sample_id=f"sample-{i}", prompt="Turn left", anchor_frame_id="0",
            contexts={"6": [f"scene0000_00/color/{frame}.png" for frame in range(6)]},
            target_frame_id="target", target_image="scene0000_00/color/target.png",
        ) for i in range(8)]
        self.write_rows()

    def tearDown(self):
        self.temp.cleanup()

    def write_rows(self):
        self.snapshot.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def arguments(self, *extra):
        with patch.dict(os.environ, {}, clear=True):
            return parser().parse_args([
                "--training-jsonl", str(self.snapshot), "--model-path", str(self.model),
                "--scannet-root", str(self.images), "--output-root", str(self.output),
                "--benchmark-jsonl", str(self.benchmark),
                "--expected-rows", "8", "--snapshot-sha256", "auto", *extra,
            ])

    def test_default_recipe_and_snapshot_binding(self):
        result = prepare(self.arguments())
        training = result["specification"]["training"]
        self.assertEqual((training["total_steps"], result["specification"]["world_size"]), (10000, 8))
        self.assertEqual(training["gradient_accumulation_steps"], 1)
        self.assertEqual((training["lr"], training["min_lr"], training["warmup_steps"]), (1e-5, 1e-6, 200))
        self.assertEqual((training["ema"], training["timestep_shift"]), (0.9995, 4.0))
        config = result["config"]["mindview_sft"]
        self.assertEqual(config["expected_manifest_sha256"], hashlib.sha256(self.snapshot.read_bytes()).hexdigest())
        self.assertEqual(config["num_used_data"], [8])
        self.assertFalse(self.output.exists())

    def test_snapshot_requires_explicit_custom_hash(self):
        arguments = self.arguments("--snapshot-sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            prepare(arguments)

    def test_target_leak_uses_shared_loader_schema(self):
        self.rows[0]["target_image"] = "scene0000_00/color/1.png"
        self.rows[0]["target_frame_id"] = "1"
        self.write_rows()
        with self.assertRaisesRegex(MindViewDataError, "target image leaks"):
            prepare(self.arguments())

    def test_duplicate_source_lineage_is_rejected(self):
        self.rows[1]["source_sample_id"] = self.rows[0]["source_sample_id"]
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "Duplicate UID or source lineage"):
            prepare(self.arguments())

    def test_wrong_shot_is_rejected_before_launch(self):
        self.rows[0]["shot"] = 3
        self.rows[0]["contexts"] = {"3": self.rows[0]["contexts"]["6"][:3]}
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "requires shot=6"):
            prepare(self.arguments())

    def test_training_scene_overlap_is_rejected_for_custom_snapshot(self):
        self.benchmark.write_text(json.dumps(dict(dataset="scannet", scene_id="scene0000_00")) + "\n")
        with self.assertRaisesRegex(ValueError, "Training scenes overlap"):
            prepare(self.arguments())

    def test_output_cannot_overlap_original_images(self):
        with self.assertRaisesRegex(ValueError, "overlaps protected input"):
            prepare(self.arguments("--output-root", str(self.images / "run")))

    def test_changed_recipe_cannot_resume_in_existing_run(self):
        result = prepare(self.arguments())
        self.output.mkdir()
        (self.output / "run_spec.json").write_text(json.dumps(result["specification"]))
        with self.assertRaisesRegex(ValueError, "Existing run configuration differs"):
            prepare(self.arguments("--total-steps", "2"))

    def test_multinode_launch_and_gpu_validation(self):
        result = prepare(self.arguments("--nproc-per-node", "1", "--nnodes", "2", "--node-rank", "1"))
        self.assertIn("--nnodes=2", result["command"])
        self.assertIn("--node_rank=1", result["command"])
        self.assertNotIn("--standalone", result["command"])
        with self.assertRaisesRegex(ValueError, "distinct device"):
            prepare(self.arguments("--gpus", "0,0", "--nproc-per-node", "2"))

    def test_image_symlink_cannot_escape_dataset_root(self):
        outside = self.root / "outside.png"
        outside.write_bytes(b"image fixture")
        directory = self.images / "scene0000_00/color"
        directory.mkdir(parents=True)
        (directory / "0.png").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "path outside dataset root"):
            prepare(self.arguments("--check-images"))

    def test_missing_images_and_modified_preprocessing_fail(self):
        with self.assertRaisesRegex(ValueError, "Missing RGB file"):
            prepare(self.arguments("--check-images"))
        config = prepare(self.arguments())["config"]
        modified = copy.deepcopy(config)
        modified["mindview_sft"]["image_transform_args"]["max_image_size"] = 256
        config_path = self.root / "modified.json"
        config_path.write_text(json.dumps(modified))
        with self.assertRaisesRegex(ValueError, "VAE/ViT image transforms"):
            prepare(self.arguments("--dataset-config", str(config_path)))

    def test_real_dry_run_writes_nothing_without_gpu_imports(self):
        arguments = self.arguments("--nproc-per-node", "1", "--gpus", "0", "--total-steps", "2")
        command = ["bash", str(SFT_ROOT / "run.sh"), "--training-jsonl", str(self.snapshot),
                   "--model-path", str(self.model), "--scannet-root", str(self.images),
                   "--benchmark-jsonl", str(self.benchmark),
                   "--output-root", str(self.output), "--expected-rows", "8", "--snapshot-sha256", "auto",
                   "--nproc-per-node", str(arguments.nproc_per_node), "--gpus", "0", "--total-steps", "2", "--dry-run"]
        environment = dict(os.environ, BAGEL_PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1")
        for name in ("CONFIG_FILE", "SCANNETPP_ROOT", "ARKITSCENES_ROOT"):
            environment.pop(name, None)
        result = subprocess.run(command, text=True, capture_output=True, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Dry run passed", result.stdout)
        self.assertIn("--total_steps 2", result.stdout)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
