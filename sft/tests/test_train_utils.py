import tempfile
import unittest
from pathlib import Path

from train.train_utils import get_latest_ckpt


class LatestCheckpointTest(unittest.TestCase):
    def _write_complete_checkpoint(self, root, step, shards):
        checkpoint = root / f"{step:07d}"
        checkpoint.mkdir()
        for name in (
            "ema.safetensors",
            "model.safetensors",
            "scheduler.pt",
            "data_status.pt",
        ):
            (checkpoint / name).write_bytes(b"complete")
        for index in range(shards):
            name = f"optimizer.{index:05d}-of-{shards:05d}.pt"
            (checkpoint / name).write_bytes(b"complete")
        (checkpoint / "_SUCCESS").write_text(f"{step}\n", encoding="utf-8")
        return checkpoint

    def test_skips_newer_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            complete = self._write_complete_checkpoint(root, step=10, shards=4)
            partial = root / "0000020"
            partial.mkdir()
            (partial / "ema.safetensors").write_bytes(b"partial")

            selected = get_latest_ckpt(root, expected_optimizer_shards=4)

            self.assertEqual(Path(selected), complete)

    def test_skips_newer_checkpoint_without_completion_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            complete = self._write_complete_checkpoint(root, step=10, shards=4)
            uncommitted = self._write_complete_checkpoint(root, step=20, shards=4)
            (uncommitted / "_SUCCESS").unlink()

            selected = get_latest_ckpt(root, expected_optimizer_shards=4)

            self.assertEqual(Path(selected), complete)

    def test_rejects_checkpoint_with_wrong_world_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_complete_checkpoint(root, step=10, shards=4)

            with self.assertRaisesRegex(RuntimeError, "none is complete"):
                get_latest_ckpt(root, expected_optimizer_shards=2)

    def test_rejects_extra_optimizer_shards_from_an_old_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            complete = self._write_complete_checkpoint(root, step=10, shards=4)
            mixed = self._write_complete_checkpoint(root, step=20, shards=4)
            for index in range(2):
                name = f"optimizer.{index:05d}-of-00002.pt"
                (mixed / name).write_bytes(b"stale-layout")

            selected = get_latest_ckpt(root, expected_optimizer_shards=4)

            self.assertEqual(Path(selected), complete)
            with self.assertRaisesRegex(RuntimeError, "none is complete"):
                get_latest_ckpt(root, expected_optimizer_shards=2)

    def test_empty_checkpoint_root_starts_from_base_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(
                get_latest_ckpt(temp_dir, expected_optimizer_shards=4)
            )


if __name__ == "__main__":
    unittest.main()
