import json
from pathlib import Path
import tempfile
import unittest

from understanding.spatial_benchmarks import sat_v2
from understanding.spatial_benchmarks.results import atomic_write_json, ensure_manifest, spec_fingerprint
from understanding.summarize import summarize
from understanding.tests import test_sat_v2_benchmark as fixtures


class SummaryTest(unittest.TestCase):
    def create_partial_sat(self, root):
        directory = root / "sat-v2"
        helper = fixtures.SatV2BenchmarkTest()
        # A small synthetic run exercises the summary's official-protocol
        # bindings without pretending to download or evaluate real benchmark data.
        benchmark = sat_v2.SatV2Benchmark([helper._rows()[0]] * 150, Path("fixture.parquet"),
                                       {"sha256": sat_v2.SAT_TEST_SHA256}, limit=1, allow_nonstandard_data=True)
        model = {"checkpoint_path": "fixture-ema.safetensors", "checkpoint_size_bytes": 31,
                 "checkpoint_sampled_sha256": "a" * 64, "assets": {}, "image_transform": {"args": {"max_image_size": 980}}}
        spec = {"evaluator_version": "bagel_spatial_eval_v2", "benchmark": dict(benchmark.manifest_fragment),
                "scope": {"partial": True, "source_limit": 1}, "model": model,
                "decoding": {"do_sample": False, "temperature": 0.0, "thinking": False,
                             "max_new_tokens": 1024, "seed": 0,
                             "image_transform_source": "understanding/image_transform.yaml:vlm_sft"},
                "model_execution": {"parameter_dtype": "torch.bfloat16", "autocast_dtype": "torch.bfloat16",
                                    "checkpoint_loading": "stream_safetensors_tensor_by_tensor",
                                    "replication": "one_complete_model_per_process"}}
        fingerprint = spec_fingerprint(spec)
        ensure_manifest(directory, spec)
        records = []
        for task in benchmark.tasks:
            record = helper._valid_record(benchmark, task)
            record.update(checkpoint_path=model["checkpoint_path"], run_fingerprint=fingerprint,
                          schema_version="bagel_spatial_prediction_v2", do_sample=False,
                          temperature=0.0, max_new_tokens=1024)
            records.append(record)
        selected = {row["eval_id"]: row for row in records}
        metrics = dict(benchmark.aggregate(selected))
        metrics.update(complete=True, run_fingerprint=fingerprint, num_merged_records=2,
                       expected_eval_ids=2, duplicate_attempts=0, partial=True, source_limit=1)
        atomic_write_json(directory / "metrics.json", metrics)
        (directory / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        (directory / "_SUCCESS").write_text("fixture\n")
        return directory

    def test_partial_summary_recomputes_answers_and_never_claims_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_partial_sat(root)
            summary = summarize(root, limit=1)
            row = summary["benchmarks"]["sat-v2"]
            self.assertEqual((summary["status"], row["status"]), ("partial", "partial"))
            self.assertEqual((row["correct"], row["total"], row["accuracy_percent"]), (2, 2, 100.0))
            self.assertEqual(row["both_order_accuracy_percent"], 100.0)
            self.assertEqual(summary["benchmarks"]["mmsi"]["status"], "not_run")
            for name in ("summary.json", "summary.csv", "summary.md"):
                self.assertTrue((root / name).is_file())

    def test_tampered_raw_answer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.create_partial_sat(root)
            records = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
            records[0]["raw_response"] = "<answer>B</answer>"
            (run / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
            summary = summarize(root, limit=1)
            self.assertEqual(summary["benchmarks"]["sat-v2"]["status"], "invalid")
            self.assertNotIn("accuracy_percent", summary["benchmarks"]["sat-v2"])

    def test_partial_run_cannot_be_summarized_as_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_partial_sat(root)
            summary = summarize(root)
            self.assertEqual(summary["benchmarks"]["sat-v2"]["status"], "invalid")
            self.assertEqual(summary["status"], "partial")


if __name__ == "__main__":
    unittest.main()
