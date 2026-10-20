import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image
try:
    import torch
    import accelerate
    from safetensors.torch import save_file
    HAS_TENSOR_RUNTIME = True
except ImportError:
    HAS_TENSOR_RUNTIME = False

from understanding.spatial_benchmarks.bagel_model import (
    stream_safetensors_checkpoint,
)
from understanding.spatial_benchmarks.results import (
    JsonlShardWriter,
    OutputRunLock,
    completed_eval_ids,
    ensure_manifest,
    finalize_results,
)
from understanding.spatial_benchmarks.sat_v2 import (
    SatV2Benchmark,
    build_sat_prompt,
    parse_official_compat_answer,
    parse_strict_answer,
)

TEST_RECORD_SCHEMA = "test_prediction_v1"
TEST_FINGERPRINT = "f" * 64
TEST_CHECKPOINT = "/tmp/test_checkpoint.safetensors"


def _png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(buffer, format="PNG")
    return buffer.getvalue()


class SatV2BenchmarkTest(unittest.TestCase):
    def _rows(self):
        return [
            {
                "images": [
                    {"bytes": _png_bytes((255, 0, 0)), "path": None},
                    {"bytes": _png_bytes((0, 255, 0)), "path": None},
                ],
                "question": "Did the chair move left?",
                "answers": ["yes", "no"],
                "correct_answer": "yes",
                "question_type": "obj_movement",
            },
            {
                "images": [{"bytes": _png_bytes((0, 0, 255)), "path": None}],
                "question": "Should the camera turn right?",
                "answers": ["right", "left"],
                "correct_answer": "right",
                "question_type": "ego_movement",
            },
        ]

    def _benchmark(self, limit=None):
        return SatV2Benchmark(
            self._rows(),
            Path("/tmp/fake_sat.parquet"),
            {"sha256": "fixture"},
            limit=limit,
            allow_nonstandard_data=True,
        )

    def _valid_record(self, benchmark, task, response=None):
        sample = benchmark.materialize(task)
        response = response or f"<answer>{sample.target_letter}</answer>"
        record = {
            "schema_version": TEST_RECORD_SCHEMA,
            "run_fingerprint": TEST_FINGERPRINT,
            "eval_id": task.eval_id,
            "global_index": task.global_index,
            "source_index": task.source_index,
            "source_id": task.source_id,
            "category": task.category,
            "variant": task.variant,
            "checkpoint_path": TEST_CHECKPOINT,
            "status": "ok",
            "question": sample.metadata["question"],
            "prompt": sample.prompt,
            "options": list(sample.options),
            "target_letter": sample.target_letter,
            "target_text": sample.target_text,
            "num_images": len(sample.images),
            "raw_response": response,
            "elapsed_seconds": 0.1,
            **benchmark.score_response(sample, response),
        }
        for image in sample.images:
            image.close()
        return record

    def test_circular_expansion_is_all_original_then_all_reversed(self):
        benchmark = self._benchmark()
        self.assertEqual(len(benchmark.tasks), 4)
        self.assertEqual(
            [(task.source_index, task.variant) for task in benchmark.tasks],
            [
                (0, "original"),
                (1, "original"),
                (0, "reversed"),
                (1, "reversed"),
            ],
        )
        self.assertEqual(
            [task.global_index for task in benchmark.tasks], list(range(4))
        )
        self.assertEqual(len({task.eval_id for task in benchmark.tasks}), 4)

    def test_limit_applies_before_circular_expansion(self):
        benchmark = self._benchmark(limit=1)
        self.assertEqual(len(benchmark.tasks), 2)
        self.assertEqual(
            [(task.source_index, task.variant) for task in benchmark.tasks],
            [(0, "original"), (0, "reversed")],
        )

    def test_materialization_preserves_image_order_and_recomputes_gt(self):
        benchmark = self._benchmark(limit=1)
        original = benchmark.materialize(benchmark.tasks[0])
        reversed_sample = benchmark.materialize(benchmark.tasks[1])

        self.assertEqual(original.target_letter, "A")
        self.assertEqual(original.options, ("yes", "no"))
        self.assertEqual(reversed_sample.target_letter, "B")
        self.assertEqual(reversed_sample.options, ("no", "yes"))
        self.assertEqual(len(original.images), 2)
        self.assertEqual(original.images[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(original.images[1].getpixel((0, 0)), (0, 255, 0))

    def test_official_sft_prompt_snapshot(self):
        prompt = build_sat_prompt("Where did it move?", ["left", "right"])
        self.assertEqual(
            prompt,
            "Where did it move? Choose from the following options: \n"
            "(A) left\n"
            "(B) right\n"
            "Provide your final answer between the <answer> </answer> tags."
            " Please provide only the single option letter (e.g., A, B, C, D, etc.)"
            " within the <answer> </answer> tags.",
        )

    def test_strict_parser_is_target_independent(self):
        cases = [
            ("<answer>A</answer>", "A", "answer_tag"),
            ("<answer>(b)</answer>", "B", "answer_tag"),
            ("answer is (A)", "A", "answer_is"),
            ("Answer is b", "B", "answer_is"),
            ("Final answer: B", "B", "final_answer"),
            ("(A)", "A", "standalone_letter"),
        ]
        for response, expected, method in cases:
            with self.subTest(response=response):
                self.assertEqual(
                    parse_strict_answer(response, ["A", "B"]),
                    (expected, method),
                )
        self.assertEqual(
            parse_strict_answer(
                "<answer>A</answer> then <answer>B</answer>", ["A", "B"]
            ),
            (None, "conflicting_answer_tags"),
        )
        self.assertEqual(
            parse_strict_answer("I am uncertain", ["A", "B"]),
            (None, "parse_failure"),
        )

    def test_official_compat_parser_is_reported_separately(self):
        # The released scorer searches for the GT letter anywhere in the model
        # response. This fixture records that behavior rather than endorsing it.
        self.assertEqual(parse_official_compat_answer("Maybe B or A", "B"), "B")
        self.assertEqual(
            parse_official_compat_answer("<answer>a</answer>", "A"),
            "<answer>a</answer>",
        )

    def test_official_compat_scoring_matches_case_insensitive_aggregation(self):
        benchmark = self._benchmark(limit=1)
        sample = benchmark.materialize(benchmark.tasks[0])
        score = benchmark.score_response(sample, "a")
        self.assertTrue(score["official_compat_correct"])
        self.assertEqual(score["official_compat_prediction_text"], "yes")
        self.assertTrue(score["strict_correct"])

    def test_micro_aggregation_and_circular_pair_metrics(self):
        benchmark = self._benchmark()
        records = {}
        for task in benchmark.tasks:
            sample = benchmark.materialize(task)
            # Make only row 1's reversed variant wrong: 3/4 micro accuracy.
            correct = not (task.source_index == 1 and task.variant == "reversed")
            predicted = sample.target_letter if correct else "A"
            response = f"<answer>{predicted}</answer>"
            records[task.eval_id] = {
                "status": "ok",
                **benchmark.score_response(sample, response),
            }

        metrics = benchmark.aggregate(records)
        self.assertEqual(metrics["overall"]["total"], 4)
        self.assertEqual(metrics["overall"]["strict_correct"], 3)
        self.assertEqual(metrics["overall"]["strict_accuracy"], 0.75)
        self.assertEqual(metrics["circular_pairs"]["strict_both_correct"], 1)
        self.assertEqual(metrics["per_category"]["obj_movement"]["total"], 2)

    def test_modulo_sharding_has_no_padding_overlap_or_drop(self):
        benchmark = self._benchmark()
        shards = [
            {task.eval_id for task in benchmark.tasks if task.global_index % 3 == rank}
            for rank in range(3)
        ]
        self.assertEqual(
            set().union(*shards), {task.eval_id for task in benchmark.tasks}
        )
        self.assertEqual(sum(len(shard) for shard in shards), len(benchmark.tasks))
        for index, left in enumerate(shards):
            for right in shards[index + 1 :]:
                self.assertFalse(left & right)

    def test_jsonl_resume_repairs_only_a_truncated_tail(self):
        benchmark = self._benchmark(limit=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with JsonlShardWriter(output_dir, rank=0, fsync_every=1) as writer:
                writer.write(self._valid_record(benchmark, benchmark.tasks[0]))
            shard = output_dir / "shards" / "rank-00000.jsonl"
            with shard.open("ab") as handle:
                handle.write(b'{"eval_id":"truncated"')

            self.assertEqual(
                completed_eval_ids(
                    output_dir,
                    benchmark,
                    run_fingerprint=TEST_FINGERPRINT,
                    record_schema=TEST_RECORD_SCHEMA,
                    checkpoint_path=TEST_CHECKPOINT,
                ),
                [benchmark.tasks[0].eval_id],
            )
            self.assertTrue(list(shard.parent.glob("*.truncated.*")))

    def test_manifest_protects_protocol_and_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            ensure_manifest(output_dir, {"checkpoint": "base", "protocol": "sat"})
            ensure_manifest(output_dir, {"checkpoint": "base", "protocol": "sat"})
            with self.assertRaisesRegex(RuntimeError, "frozen manifest"):
                ensure_manifest(
                    output_dir, {"checkpoint": "different", "protocol": "sat"}
                )

    def test_output_directory_lock_rejects_a_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            first = OutputRunLock(output_dir)
            second = OutputRunLock(output_dir)
            first.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "Another evaluation"):
                    second.acquire()
            finally:
                first.release()
            second.acquire()
            second.release()

    def test_finalize_requires_both_circular_items(self):
        benchmark = self._benchmark(limit=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            response = "<answer>A</answer>"
            with JsonlShardWriter(output_dir, rank=0, fsync_every=1) as writer:
                writer.write(
                    self._valid_record(benchmark, benchmark.tasks[0], response=response)
                )
            metrics = finalize_results(
                output_dir,
                benchmark,
                run_fingerprint=TEST_FINGERPRINT,
                record_schema=TEST_RECORD_SCHEMA,
                checkpoint_path=TEST_CHECKPOINT,
            )
            self.assertFalse(metrics["complete"])
            self.assertEqual(metrics["overall"]["missing"], 1)
            self.assertFalse((output_dir / "_SUCCESS").exists())

    def test_resume_rejects_an_unversioned_minimal_ok_record(self):
        benchmark = self._benchmark(limit=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with JsonlShardWriter(output_dir, rank=0, fsync_every=1) as writer:
                writer.write(
                    {
                        "eval_id": benchmark.tasks[0].eval_id,
                        "status": "ok",
                    }
                )
            with self.assertRaisesRegex(RuntimeError, "frozen run manifest"):
                completed_eval_ids(
                    output_dir,
                    benchmark,
                    run_fingerprint=TEST_FINGERPRINT,
                    record_schema=TEST_RECORD_SCHEMA,
                    checkpoint_path=TEST_CHECKPOINT,
                )


@unittest.skipUnless(HAS_TENSOR_RUNTIME, "PyTorch/accelerate/safetensors are not installed")
class StreamingCheckpointTest(unittest.TestCase):
    def test_bf16_release_checkpoint_is_accepted_without_fp32_assumptions(self):
        from accelerate import init_empty_weights

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "release.safetensors"
            save_file({"weight": torch.eye(2).to(torch.bfloat16),
                       "bias": torch.tensor([1.5, -2.5], dtype=torch.bfloat16)}, path)
            with init_empty_weights():
                model = torch.nn.Linear(2, 2)
            self.assertEqual(stream_safetensors_checkpoint(model, path, device=torch.device("cpu"),
                                                          dtype=torch.bfloat16), ())
            self.assertEqual(model.weight.dtype, torch.bfloat16)
            torch.testing.assert_close(model.weight.float(), torch.eye(2), rtol=0, atol=0)

    def test_fp32_checkpoint_is_streamed_and_cast_without_unexpected_keys(self):
        from accelerate import init_empty_weights

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tiny.safetensors"
            save_file(
                {
                    "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                    "bias": torch.tensor([1.5, -2.5], dtype=torch.float32),
                    "generation_only": torch.ones(1, dtype=torch.float32),
                },
                path,
            )
            with init_empty_weights():
                model = torch.nn.Linear(3, 2)

            unused = stream_safetensors_checkpoint(
                model,
                path,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
            )

            self.assertEqual(unused, ("generation_only",))
            self.assertEqual(model.weight.device.type, "cpu")
            self.assertEqual(model.weight.dtype, torch.bfloat16)
            self.assertFalse(model.weight.is_meta)
            torch.testing.assert_close(
                model.bias.float(), torch.tensor([1.5, -2.5]), rtol=0, atol=0
            )

    def test_streaming_loader_rejects_missing_model_weights(self):
        from accelerate import init_empty_weights

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.safetensors"
            save_file({"weight": torch.ones((2, 3))}, path)
            with init_empty_weights():
                model = torch.nn.Linear(3, 2)
            with self.assertRaisesRegex(RuntimeError, "missing 1 understanding"):
                stream_safetensors_checkpoint(
                    model,
                    path,
                    device=torch.device("cpu"),
                    dtype=torch.bfloat16,
                )

    def test_streaming_loader_handles_nonpersistent_rotary_buffer(self):
        from accelerate import init_empty_weights

        class TinyRotary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(2, 2)
                self.register_buffer("inv_freq", torch.tensor([0.5]), persistent=False)
                self.original_inv_freq = torch.tensor([-1.0])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rotary.safetensors"
            save_file(
                {
                    "projection.weight": torch.eye(2),
                    "projection.bias": torch.zeros(2),
                },
                path,
            )
            with init_empty_weights():
                model = TinyRotary()

            stream_safetensors_checkpoint(
                model,
                path,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
            )

            self.assertNotIn("inv_freq", model.state_dict())
            self.assertEqual(model.inv_freq.device.type, "cpu")
            self.assertIs(model.original_inv_freq, model.inv_freq)


if __name__ == "__main__":
    unittest.main()
