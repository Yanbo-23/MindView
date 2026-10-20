import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from understanding.spatial_benchmarks.mmsi import (
    MMSI_CIRCULAR_ORDERS,
    MMSI_POST_PROMPT,
    MmsiBenchmark,
    build_mmsi_prompt,
    build_mmsi_question_with_options,
    extract_mmsi_question_options,
    parse_mmsi_official_answer,
)
from understanding.spatial_benchmarks.results import JsonlShardWriter, finalize_results

TEST_RECORD_SCHEMA = "test_prediction_v1"
TEST_FINGERPRINT = "f" * 64
TEST_CHECKPOINT = "/tmp/test_checkpoint.safetensors"


def _png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(buffer, format="PNG")
    return buffer.getvalue()


class MmsiBenchmarkTest(unittest.TestCase):
    def _rows(self):
        return [
            {
                "id": 0,
                "images": [
                    _png_bytes((255, 0, 0)),
                    _png_bytes((0, 255, 0)),
                ],
                "question_type": "Motion (Cam.)",
                "question": (
                    "Where is the camera moving?  \n"
                    "Options: A: left, B: right, C: forward, D: backward"
                ),
                "answer": "C",
                "mean_normed_duration_seconds": 0.25,
                "difficulty": "easy",
            },
            {
                "id": 1,
                "images": [
                    _png_bytes((0, 0, 255)),
                    _png_bytes((255, 255, 0)),
                    _png_bytes((255, 0, 255)),
                ],
                "question_type": "MSR",
                "question": (
                    "Which route reaches the chair?\n"
                    "Options: A: first, B: second, C: third, D: fourth"
                ),
                "answer": "A",
                "mean_normed_duration_seconds": 0.75,
                "difficulty": "hard",
            },
        ]

    def _benchmark(self, mode="vanilla", limit=None):
        return MmsiBenchmark(
            self._rows(),
            Path("/tmp/fake_mmsi.parquet"),
            {"sha256": "fixture"},
            mode=mode,
            limit=limit,
            allow_nonstandard_data=True,
        )

    def test_official_prompt_snapshot(self):
        question = "Question\nOptions: A: one, B: two, C: three, D: four"
        self.assertEqual(
            build_mmsi_prompt(question),
            question
            + "\nAnswer with the option's letter from the given choices directly. "
            "Enclose the option's letter within ``.",
        )
        self.assertEqual(len(MMSI_POST_PROMPT.encode("utf-8")), 103)

    def test_official_option_parser_and_rebuilder(self):
        question = (
            "Which way?\nOptions: A: left; B: right; C: forward; D: backward"
        )
        question_text, options = extract_mmsi_question_options(question)
        self.assertEqual(question_text, "Which way?")
        self.assertEqual(
            options,
            {
                "A": "left;",
                "B": "right;",
                "C": "forward;",
                "D": "backward",
            },
        )
        self.assertEqual(
            build_mmsi_question_with_options(question_text, options),
            "Which way?\nOptions: A: left;, B: right;, C: forward;, D: backward",
        )

    def test_official_regex_goldens(self):
        cases = [
            (None, None),
            ("", None),
            ("A", "A"),
            ("a", None),
            ("``C``", "C"),
            ("`B`", "B"),
            ("The answer is A.", "A"),
            ("Option D", "D"),
            ("(C)", "C"),
            ("A bike, but answer is B", "B"),
            ("Answer: A because it is visible", None),
            ("Answer: A, because it is visible", "A"),
            ("AB", None),
            ("``A bike`` B", None),
            ("{A}", "A"),
            ("B reason", None),
            ("D) explanation", "D"),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                prediction, _ = parse_mmsi_official_answer(response)  # type: ignore[arg-type]
                self.assertEqual(prediction, expected)

    def test_vanilla_preserves_raw_question_and_image_order(self):
        benchmark = self._benchmark(mode="vanilla", limit=1)
        self.assertEqual(len(benchmark.tasks), 1)
        self.assertEqual(benchmark.tasks[0].variant, "vanilla")
        self.assertEqual(benchmark.tasks[0].category, "运动感知 camera")

        sample = benchmark.materialize(benchmark.tasks[0])
        try:
            self.assertTrue(sample.metadata["question"].startswith(
                "Where is the camera moving?  \n"
            ))
            self.assertEqual(sample.target_letter, "C")
            self.assertEqual(sample.options, ("left", "right", "forward", "backward"))
            self.assertEqual(sample.images[0].getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(sample.images[1].getpixel((0, 0)), (0, 255, 0))
            self.assertNotIn("thought", sample.metadata)
        finally:
            for image in sample.images:
                image.close()

    def test_circular_expansion_and_answer_remapping_match_cp4(self):
        benchmark = self._benchmark(mode="circular", limit=1)
        self.assertEqual(
            [task.variant for task in benchmark.tasks],
            list(MMSI_CIRCULAR_ORDERS),
        )
        self.assertEqual(
            [task.global_index for task in benchmark.tasks], list(range(4))
        )
        self.assertEqual(len({task.eval_id for task in benchmark.tasks}), 4)

        samples = [benchmark.materialize(task) for task in benchmark.tasks]
        try:
            self.assertEqual(
                [sample.target_letter for sample in samples], ["C", "B", "A", "D"]
            )
            self.assertEqual(
                samples[1].options,
                ("right", "forward", "backward", "left"),
            )
            # Official Circular rebuilds even the identity question and strips
            # the two trailing spaces before the newline.
            self.assertTrue(samples[0].metadata["question"].startswith(
                "Where is the camera moving?\n"
            ))
        finally:
            for sample in samples:
                for image in sample.images:
                    image.close()

    def test_limit_applies_before_circular_expansion(self):
        benchmark = self._benchmark(mode="circular", limit=1)
        self.assertEqual(len(benchmark.tasks), 4)
        self.assertEqual({task.source_index for task in benchmark.tasks}, {0})

    @staticmethod
    def _scored_records(benchmark, response_for_task):
        records = {}
        for task in benchmark.tasks:
            sample = benchmark.materialize(task)
            try:
                response = response_for_task(task, sample)
                records[task.eval_id] = {
                    "status": "ok",
                    **benchmark.score_response(sample, response),
                }
            finally:
                for image in sample.images:
                    image.close()
        return records

    def test_circular_requires_all_four_variants_correct(self):
        benchmark = self._benchmark(mode="circular")
        all_correct = self._scored_records(
            benchmark, lambda _task, sample: f"``{sample.target_letter}``"
        )
        metrics = benchmark.aggregate(all_correct)
        self.assertEqual(metrics["vanilla"]["official_accuracy"], 1.0)
        self.assertEqual(metrics["circular"]["official_accuracy"], 1.0)

        constant_a = self._scored_records(
            benchmark, lambda _task, _sample: "``A``"
        )
        metrics = benchmark.aggregate(constant_a)
        self.assertEqual(metrics["vanilla"]["official_correct"], 1)
        self.assertEqual(metrics["circular"]["official_correct"], 0)

        one_wrong = self._scored_records(
            benchmark,
            lambda task, sample: (
                "``A``"
                if task.source_index == 0 and task.variant == "DABC"
                else f"``{sample.target_letter}``"
            ),
        )
        metrics = benchmark.aggregate(one_wrong)
        self.assertEqual(metrics["circular"]["official_correct"], 1)
        self.assertEqual(metrics["circular"]["total"], 2)

        one_unparsed = self._scored_records(
            benchmark,
            lambda task, sample: (
                "lowercase c"
                if task.source_index == 0 and task.variant == "DABC"
                else f"``{sample.target_letter}``"
            ),
        )
        metrics = benchmark.aggregate(one_unparsed)
        self.assertEqual(metrics["circular"]["official_correct"], 1)
        self.assertEqual(
            metrics["expanded_variants"]["prediction_distribution"]["unparsed"],
            1,
        )

    def test_score_uses_only_official_case_sensitive_parser(self):
        benchmark = self._benchmark(mode="vanilla", limit=1)
        sample = benchmark.materialize(benchmark.tasks[0])
        try:
            correct = benchmark.score_response(sample, "``C``")
            self.assertTrue(correct["parse_success"])
            self.assertTrue(correct["official_correct"])
            lower = benchmark.score_response(sample, "``c``")
            self.assertFalse(lower["parse_success"])
            self.assertFalse(lower["official_correct"])
            metrics = benchmark.aggregate(
                {
                    benchmark.tasks[0].eval_id: {
                        "status": "ok",
                        **lower,
                    }
                }
            )
            self.assertEqual(
                metrics["overall"]["prediction_distribution"]["unparsed"], 1
            )
        finally:
            for image in sample.images:
                image.close()

    def test_modulo_sharding_has_no_overlap_or_drop(self):
        benchmark = self._benchmark(mode="circular")
        shards = [
            {task.eval_id for task in benchmark.tasks if task.global_index % 3 == rank}
            for rank in range(3)
        ]
        self.assertEqual(
            set().union(*shards), {task.eval_id for task in benchmark.tasks}
        )
        self.assertEqual(sum(map(len, shards)), len(benchmark.tasks))
        for index, left in enumerate(shards):
            for right in shards[index + 1 :]:
                self.assertFalse(left & right)

    def test_common_result_pipeline_accepts_mmsi_score_fields(self):
        benchmark = self._benchmark(mode="vanilla")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with JsonlShardWriter(output_dir, rank=0, fsync_every=1) as writer:
                for task in benchmark.tasks:
                    sample = benchmark.materialize(task)
                    try:
                        response = f"``{sample.target_letter}``"
                        writer.write(
                            {
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
                                "question_type": sample.metadata["question_type"],
                                "difficulty": sample.metadata["difficulty"],
                                "official_index": sample.metadata["official_index"],
                                "official_group_index": sample.metadata[
                                    "official_group_index"
                                ],
                                "prompt": sample.prompt,
                                "options": list(sample.options),
                                "target_letter": sample.target_letter,
                                "target_text": sample.target_text,
                                "num_images": len(sample.images),
                                "raw_response": response,
                                "elapsed_seconds": 0.1,
                                **benchmark.score_response(sample, response),
                            }
                        )
                    finally:
                        for image in sample.images:
                            image.close()

            metrics = finalize_results(
                output_dir,
                benchmark,
                run_fingerprint=TEST_FINGERPRINT,
                record_schema=TEST_RECORD_SCHEMA,
                checkpoint_path=TEST_CHECKPOINT,
            )
            self.assertTrue(metrics["complete"])
            self.assertEqual(metrics["vanilla"]["official_accuracy"], 1.0)
            self.assertTrue((output_dir / "_SUCCESS").is_file())


if __name__ == "__main__":
    unittest.main()
