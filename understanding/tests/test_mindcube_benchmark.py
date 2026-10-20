import tempfile
import unittest
from pathlib import Path

from PIL import Image

from understanding.spatial_benchmarks.mindcube import (
    MindCubeAdapter,
    build_mindcube_raw_qa_prompt,
    build_mindcube_raw_qa_target,
    clean_mindcube_response,
    extract_mindcube_options,
    get_mindcube_setting,
    parse_mindcube_official_answer,
)
from understanding.spatial_benchmarks.results import JsonlShardWriter, finalize_results

TEST_RECORD_SCHEMA = "test_prediction_v1"
TEST_FINGERPRINT = "f" * 64
TEST_CHECKPOINT = "/tmp/test_checkpoint.safetensors"


class MindCubeBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)
        image_dir = self.data_root / "images"
        image_dir.mkdir()
        for index, color in enumerate(
            ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
        ):
            Image.new("RGB", (16, 12), color).save(image_dir / f"{index}.png")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _rows(self):
        return [
            {
                "id": "among_fixture_0",
                "category": ["perpendicular", "O-O", "meanwhile", "self"],
                "type": "1_frame",
                "meta_info": [],
                "question": "What is on the right? A. Chair B. Table C. Lamp",
                "images": ["images/0.png", "images/1.png"],
                "gt_answer": "B",
            },
            {
                "id": "rotation_fixture_1",
                "category": ["perpendicular", "P-P", "rotation", "self"],
                "type": "three_view",
                "meta_info": [],
                "question": "Which object is behind you? A. Door B. Sofa",
                "images": ["images/1.png", "images/2.png", "images/3.png"],
                "gt_answer": "A",
            },
            {
                "id": "around_fixture_2",
                "category": ["parallel", "P-O", "sequence", "self"],
                "type": 2,
                "meta_info": [],
                "question": "Did the camera move closer? A. Yes B. No.",
                "images": ["images/3.png", "images/0.png"],
                "gt_answer": "B",
            },
        ]

    def _benchmark(self, limit=None):
        return MindCubeAdapter(
            self._rows(),
            self.data_root,
            {"sha256": "fixture", "size_bytes": 1},
            limit=limit,
            allow_nonstandard_data=True,
        )

    def _valid_record(self, benchmark, task):
        sample = benchmark.materialize(task)
        try:
            response = f"<answer>{sample.target_letter}. {sample.target_text}</answer>"
            return {
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
                "difficulty": None,
                "official_index": sample.metadata["official_index"],
                "official_group_index": sample.metadata["official_group_index"],
                "prompt": sample.prompt,
                "options": list(sample.options),
                "target_letter": sample.target_letter,
                "target_text": sample.target_text,
                "num_images": len(sample.images),
                "raw_response": response,
                "elapsed_seconds": 0.1,
                **benchmark.score_response(sample, response),
            }
        finally:
            for image in sample.images:
                image.close()

    def test_official_raw_qa_prompt_snapshot_keeps_blank_line(self):
        prompt = build_mindcube_raw_qa_prompt("Where? A. Left B. Right")
        self.assertEqual(
            prompt,
            "[Task]\n"
            "Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.\n"
            "[Answer Instruction]\n"
            "You only need to provide *ONE* correct answer selecting from the options listed below. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should **only** be '<answer>A. Above</answer>'.\n"
            "\n"
            "[Question]\n"
            "Where? A. Left B. Right",
        )

    def test_official_option_and_target_extraction(self):
        question = "Where? A. Left B. Right C. Straight ahead."
        self.assertEqual(
            extract_mindcube_options(question),
            (("A", "Left"), ("B", "Right"), ("C", "Straight ahead.")),
        )
        self.assertEqual(
            build_mindcube_raw_qa_target(question, "C"),
            "<answer>C. Straight ahead.</answer>",
        )

    def test_official_parser_priority_and_case_sensitivity(self):
        cases = [
            ("<answer>A. Left</answer>", "A", "simple_period"),
            (
                "<answer>A. Left</answer> then <answer>B. Right</answer>",
                "B",
                "simple_period",
            ),
            ("<Answer>C</Answer>", "C", "answer_section_4"),
            ("<answer>D</answer>", "D", "final_word"),
            ("My answer is B", "B", "priority_5"),
            ("A", "A", "priority_2"),
            ("b.", None, "parse_failure"),
            ("no option", None, "parse_failure"),
        ]
        for response, expected, method in cases:
            with self.subTest(response=response):
                self.assertEqual(
                    parse_mindcube_official_answer(response), (expected, method)
                )

    def test_official_response_cleaning_precedes_scoring(self):
        self.assertEqual(clean_mindcube_response("  Answer:\n B  "), "Answer: B")
        self.assertEqual(clean_mindcube_response(""), "No response generated")
        self.assertEqual(
            clean_mindcube_response("   "), "Error: Generated response too short"
        )
        benchmark = self._benchmark(limit=1)
        sample = benchmark.materialize(benchmark.tasks[0])
        try:
            score = benchmark.score_response(sample, "My answer\n is B")
            self.assertTrue(score["parse_success"])
            self.assertTrue(score["official_correct"])
            self.assertEqual(score["official_response"], "My answer is B")
        finally:
            for image in sample.images:
                image.close()

    def test_setting_classifier_matches_official_priority(self):
        self.assertEqual(get_mindcube_setting("among_item"), "among")
        self.assertEqual(get_mindcube_setting("rotation_item"), "rotation")
        self.assertEqual(get_mindcube_setting("around_rotation_item"), "around")
        self.assertEqual(get_mindcube_setting("translation_item"), "translation")
        self.assertEqual(get_mindcube_setting("unknown"), "other")

    def test_materialization_preserves_image_and_option_order(self):
        benchmark = self._benchmark(limit=1)
        sample = benchmark.materialize(benchmark.tasks[0])
        try:
            self.assertEqual(sample.options, ("Chair", "Table", "Lamp"))
            self.assertEqual(sample.target_letter, "B")
            self.assertEqual(sample.target_text, "Table")
            self.assertEqual(
                sample.metadata["official_target_output"], "<answer>B. Table</answer>"
            )
            self.assertEqual(sample.images[0].getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(sample.images[1].getpixel((0, 0)), (0, 255, 0))
        finally:
            for image in sample.images:
                image.close()

    def test_limit_and_modulo_sharding_do_not_duplicate_tasks(self):
        benchmark = self._benchmark(limit=2)
        self.assertEqual(len(benchmark.tasks), 2)
        self.assertEqual([task.global_index for task in benchmark.tasks], [0, 1])
        shards = [
            {task.eval_id for task in benchmark.tasks if task.global_index % 3 == rank}
            for rank in range(3)
        ]
        self.assertEqual(
            set().union(*shards), {task.eval_id for task in benchmark.tasks}
        )
        self.assertEqual(sum(map(len, shards)), len(benchmark.tasks))

    def test_official_micro_aggregation_and_parse_failure_denominator(self):
        benchmark = self._benchmark()
        responses = ["<answer>B. Table</answer>", "unparsed", "<answer>A. Yes</answer>"]
        records = {}
        for task, response in zip(benchmark.tasks, responses):
            sample = benchmark.materialize(task)
            try:
                records[task.eval_id] = {
                    "status": "ok",
                    **benchmark.score_response(sample, response),
                }
            finally:
                for image in sample.images:
                    image.close()
        metrics = benchmark.aggregate(records)
        self.assertEqual(metrics["overall"]["total"], 3)
        self.assertEqual(metrics["overall"]["official_total"], 3)
        self.assertEqual(metrics["overall"]["official_correct"], 1)
        self.assertEqual(metrics["overall"]["parsed"], 2)
        self.assertEqual(metrics["per_setting"]["among"]["official_correct"], 1)
        self.assertEqual(metrics["per_setting"]["rotation"]["official_correct"], 0)

    def test_common_result_pipeline_validates_and_finalizes(self):
        benchmark = self._benchmark()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with JsonlShardWriter(output_dir, rank=0, fsync_every=1) as writer:
                for task in benchmark.tasks:
                    writer.write(self._valid_record(benchmark, task))
            metrics = finalize_results(
                output_dir,
                benchmark,
                run_fingerprint=TEST_FINGERPRINT,
                record_schema=TEST_RECORD_SCHEMA,
                checkpoint_path=TEST_CHECKPOINT,
            )
            self.assertTrue(metrics["complete"])
            self.assertEqual(metrics["overall"]["official_accuracy"], 1.0)
            self.assertTrue((output_dir / "_SUCCESS").is_file())

    def test_unsafe_image_path_is_rejected(self):
        rows = self._rows()[:1]
        rows[0]["images"] = ["../outside.png"]
        with self.assertRaisesRegex(ValueError, "escapes data root"):
            MindCubeAdapter(
                rows,
                self.data_root,
                {"sha256": "fixture", "size_bytes": 1},
                allow_nonstandard_data=True,
            )


if __name__ == "__main__":
    unittest.main()
