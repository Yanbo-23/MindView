import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Install the SFT PyTorch dependencies to run dataset tensor tests") from error
from PIL import Image

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.dataset_info import DATASET_INFO, DATASET_REGISTRY
from data.interleave_datasets import MindViewIterableDataset
from data.interleave_datasets.mindview_dataset import MindViewDataError


class _FakeTokenizer:
    def encode(self, text):
        return [11, 12, 13] if text else []


class _RecordingTransform:
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride
        self.calls = []

    def __call__(self, image):
        center = image.getpixel((image.width // 2, image.height // 2))[0]
        self.calls.append((image.size, center))
        return torch.full((3, self.size, self.size), float(center))


class MindViewDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_root = self.root / "images"
        self.scene_id = "scene0000_00"
        self.colors = {}
        for index in range(10):
            relpath = f"{self.scene_id}/color/{index}.png"
            self._write_center_colored_image(relpath, 10 + index)
            self.colors[relpath] = 10 + index
        self.target_relpath = f"{self.scene_id}/color/target.png"
        self._write_center_colored_image(self.target_relpath, 99)
        self.colors[self.target_relpath] = 99

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_center_colored_image(self, relpath, color):
        path = self.image_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (40, 20), (0, 0, 0))
        image.paste((color, 0, 0), (10, 0, 30, 20))
        image.save(path)

    def _row(self, index=0, shot=3):
        def path(frame):
            return f"{self.scene_id}/color/{frame}.png"

        contexts_by_shot = {
            3: [path(0), path(1), path(2)],
            6: [path(0), path(3), path(1), path(4), path(2), path(5)],
            10: [
                path(0),
                path(6),
                path(3),
                path(7),
                path(1),
                path(8),
                path(4),
                path(9),
                path(2),
                path(5),
            ],
        }
        return {
            "schema_version": "mindview_bagel_single_shot_v1",
            "shot": shot,
            "record_uid": hashlib.sha256(
                f"record-{index}-shot-{shot}".encode()
            ).hexdigest(),
            "dataset": "scannet",
            "source_family": "cam_loc",
            "task_type": "composition",
            "scene_id": self.scene_id,
            "source_sample_id": f"sample-{index}",
            "anchor_frame_id": "0",
            "prompt": "Generate the target view.",
            "contexts": {str(shot): contexts_by_shot[shot]},
            "target_frame_id": "target",
            "target_image": self.target_relpath,
            "preprocess_id": "center_square_then_bagel_resize_v1",
        }

    def _write_snapshot(self, rows, name="snapshot.jsonl"):
        path = self.root / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        return path

    def _dataset(
        self,
        rows,
        world_size=1,
        local_rank=0,
        num_workers=1,
        image_roots=None,
    ):
        snapshot = self._write_snapshot(rows)
        vae_transform = _RecordingTransform(size=16, stride=4)
        vit_transform = _RecordingTransform(size=14, stride=7)
        dataset = MindViewIterableDataset(
            dataset_name="mindview_test",
            transform=vae_transform,
            tokenizer=_FakeTokenizer(),
            vit_transform=vit_transform,
            data_dir_list=[str(self.root)],
            jsonl_path_list=[str(snapshot)],
            num_used_data=[-1],
            image_roots=image_roots or {"scannet": str(self.image_root)},
            enable_condition_dropout=False,
            local_rank=local_rank,
            world_size=world_size,
            num_workers=num_workers,
        )
        return dataset, vae_transform, vit_transform

    def test_arkitscenes_scene_prefixed_filename_uses_suffix_frame_id(self):
        scene_id = "41048265"
        frame_ids = [f"{4900 + index}.001" for index in range(11)]

        def relpath(frame_id):
            return f"color/train/{scene_id}/{scene_id}_{frame_id}.png"

        for index, frame_id in enumerate(frame_ids):
            self._write_center_colored_image(relpath(frame_id), 30 + index)

        row = self._row(shot=3)
        row.update(
            {
                "dataset": "arkitscenes",
                "source_family": "re_facing",
                "task_type": "loc-re-facing",
                "scene_id": scene_id,
                "anchor_frame_id": frame_ids[0],
                "contexts": {
                    "3": [relpath(frame_ids[index]) for index in (0, 1, 2)]
                },
                "target_frame_id": frame_ids[10],
                "target_image": relpath(frame_ids[10]),
            }
        )

        dataset, _, _ = self._dataset(
            [row], image_roots={"arkitscenes": str(self.image_root)}
        )
        sample = dataset.parse_row(row)

        self.assertEqual(len(sample["image_tensor_list"]), 7)
        self.assertEqual(
            [item["type"] for item in sample["sequence_plan"]],
            [
                "vae_image",
                "vit_image",
                "vae_image",
                "vit_image",
                "vae_image",
                "vit_image",
                "text",
                "vae_image",
            ],
        )

    def test_c3_c6_c10_sequence_plan_order_crop_and_loss(self):
        for shot in (3, 6, 10):
            with self.subTest(shot=shot):
                row = self._row(shot=shot)
                dataset, vae_transform, vit_transform = self._dataset([row])
                sample = dataset.parse_row(row)

                expected_types = [
                    value
                    for _ in range(shot)
                    for value in ("vae_image", "vit_image")
                ] + ["text", "vae_image"]
                self.assertEqual(
                    [item["type"] for item in sample["sequence_plan"]],
                    expected_types,
                )
                self.assertEqual(
                    [item["loss"] for item in sample["sequence_plan"]],
                    [0] * (2 * shot + 1) + [1],
                )
                self.assertTrue(
                    all(item["enable_cfg"] == 0 for item in sample["sequence_plan"])
                )
                self.assertEqual(len(sample["image_tensor_list"]), 2 * shot + 1)
                self.assertEqual(len(sample["text_ids_list"]), 1)

                expected_context_colors = [
                    self.colors[path] for path in row["contexts"][str(shot)]
                ]
                self.assertEqual(
                    [color for _, color in vae_transform.calls],
                    expected_context_colors + [99],
                )
                self.assertEqual(
                    [color for _, color in vit_transform.calls],
                    expected_context_colors,
                )
                self.assertTrue(
                    all(size == (20, 20) for size, _ in vae_transform.calls)
                )
                self.assertTrue(
                    all(size == (20, 20) for size, _ in vit_transform.calls)
                )
                expected_tokens = (shot + 1) * 16 + shot * 4 + 3
                self.assertEqual(sample["num_tokens"], expected_tokens)

    def test_invalid_fixed_context_and_target_leak_fail_at_startup(self):
        bad_context = self._row(shot=6)
        bad_context["contexts"]["6"].pop()
        with self.assertRaisesRegex(MindViewDataError, "exactly 6 paths"):
            self._dataset([bad_context])

        target_leak = self._row(shot=6)
        target_leak["target_image"] = target_leak["contexts"]["6"][0]
        target_leak["target_frame_id"] = "0"
        with self.assertRaisesRegex(MindViewDataError, "target image leaks"):
            self._dataset([target_leak])

    def test_missing_image_fails_when_record_is_read(self):
        row = self._row()
        row["target_frame_id"] = "missing"
        row["target_image"] = f"{self.scene_id}/color/missing.png"
        dataset, _, _ = self._dataset([row])
        with self.assertRaisesRegex(MindViewDataError, "missing image"):
            next(iter(dataset))

    def test_global_stride_sharding_has_no_drop_or_overlap(self):
        rows = [self._row(index) for index in range(11)]
        shards = []
        for rank in range(2):
            dataset, _, _ = self._dataset(
                rows, world_size=2, local_rank=rank, num_workers=2
            )
            dataset.set_epoch(91)
            for worker_id in range(2):
                shards.append(
                    {
                        record.row["record_uid"]
                        for record in dataset._records_for_worker(worker_id, 2)
                    }
                )

        union = set().union(*shards)
        self.assertEqual(union, {row["record_uid"] for row in rows})
        self.assertEqual(sum(len(shard) for shard in shards), len(rows))
        for left_index, left in enumerate(shards):
            for right in shards[left_index + 1 :]:
                self.assertFalse(left & right)

    def test_row_shot_is_fixed_across_repeat_cycles(self):
        rows = [self._row(0, shot=3), self._row(1, shot=6)]
        dataset, _, _ = self._dataset(rows)
        dataset.set_epoch(42)
        iterator = iter(dataset)
        observed = [next(iterator)["data_indexes"] for _ in range(4)]
        expected = {row["record_uid"]: row["shot"] for row in rows}
        self.assertEqual(
            [item["shot"] for item in observed],
            [expected[item["record_uid"]] for item in observed],
        )

    def test_shared_lineage_across_shots_is_valid(self):
        c3 = self._row(shot=3)
        c6 = self._row(shot=6)
        dataset, _, _ = self._dataset([c3, c6])
        self.assertEqual(len(dataset._records), 2)

        duplicate_c3 = copy.deepcopy(c3)
        duplicate_c3["record_uid"] = hashlib.sha256(b"duplicate-c3").hexdigest()
        with self.assertRaisesRegex(MindViewDataError, "duplicate source lineage"):
            self._dataset([c3, duplicate_c3])

    def test_context_key_must_match_row_shot(self):
        row = self._row(shot=6)
        row["contexts"] = {"3": row["contexts"]["6"][:3]}
        with self.assertRaisesRegex(MindViewDataError, "contexts must contain exactly"):
            self._dataset([row])

    def test_registry_to_packed_dataset_keeps_mse_only_supervision(self):
        row = self._row()
        snapshot = self._write_snapshot([row])
        fixture_info = {
            "data_dir": str(self.root),
            "jsonl_path": str(snapshot),
            "num_total_samples": 1,
        }
        grouped_datasets = {
            "mindview_sft": {
                "dataset_names": ["fixture"],
                "image_transform_args": {
                    "image_stride": 16,
                    "max_image_size": 32,
                    "min_image_size": 32,
                    "max_pixels": 1024,
                },
                "vit_image_transform_args": {
                    "image_stride": 14,
                    "max_image_size": 28,
                    "min_image_size": 28,
                    "max_pixels": 784,
                },
                "image_roots": {"scannet": str(self.image_root)},
                "enable_condition_dropout": False,
                "num_used_data": [1],
                "is_mandatory": False,
                "weight": 1.0,
            }
        }

        with patch.dict(DATASET_INFO["mindview_sft"], {"fixture": fixture_info}):
            packed_dataset = PackedDataset(
                DataConfig(grouped_datasets=copy.deepcopy(grouped_datasets)),
                tokenizer=_FakeTokenizer(),
                special_tokens={
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                    "start_of_image": 3,
                    "end_of_image": 4,
                },
                local_rank=0,
                world_size=1,
                num_workers=1,
                expected_num_tokens=1,
                max_num_tokens_per_sample=256,
                max_num_tokens=256,
                prefer_buffer_before=0,
                use_flex=True,
            )
            packed = next(iter(packed_dataset))
            batch = collate_wrapper()([packed]).to_dict()

        self.assertIs(DATASET_REGISTRY["mindview_sft"], MindViewIterableDataset)
        self.assertIn("mse_loss_indexes", batch)
        self.assertGreater(len(batch["mse_loss_indexes"]), 0)
        self.assertNotIn("ce_loss_indexes", batch)
        self.assertEqual(batch["batch_data_indexes"][0]["shot"], 3)


if __name__ == "__main__":
    unittest.main()
