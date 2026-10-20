"""Recompute and summarize the three fixed understanding benchmark scores on CPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from understanding.spatial_benchmarks import mindcube, mmsi, sat_v2
from understanding.spatial_benchmarks.results import atomic_write_json, atomic_write_text

BENCHMARKS = {
    "sat-v2": (150, 300, 1024, sat_v2.SAT_PROTOCOL_VERSION, sat_v2.SAT_TEST_SHA256),
    "mmsi": (1000, 1000, 2048, mmsi.MMSI_VANILLA_PROTOCOL_VERSION, mmsi.MMSI_PARQUET_SHA256),
    "mindcube-tiny": (1050, 1050, 4096, mindcube.MINDCUBE_PROTOCOL_VERSION, mindcube.MINDCUBE_TINY_SHA256),
}

def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def prompt_digest(prompts):
    digest = hashlib.sha256()
    for prompt in prompts:
        encoded = prompt.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def checkpoint_key(model):
    size, digest = model["checkpoint_size_bytes"], model["checkpoint_sampled_sha256"]
    require(type(size) is int and size > 0, "invalid checkpoint size")
    require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "invalid checkpoint digest")
    return size, digest


def check_bucket(bucket, correct, total, prefix="official"):
    require(bucket[f"{prefix}_correct"] == correct, f"{prefix} correct count differs from predictions")
    fraction = correct / total
    for key, expected in ((f"{prefix}_accuracy", fraction), (f"{prefix}_accuracy_percent", 100 * fraction)):
        value = bucket[key]
        require(type(value) in (int, float) and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12), f"incorrect metric {key}")


def load_run(directory, benchmark, *, limit=None):
    official_sources, official_items, token_limit, protocol, data_hash = BENCHMARKS[benchmark]
    source_count = min(limit, official_sources) if limit is not None else official_sources
    item_count = source_count * (2 if benchmark == "sat-v2" else 1)
    for filename in ("run_manifest.json", "metrics.json", "predictions.jsonl", "_SUCCESS"):
        require((directory / filename).is_file(), f"missing {filename}")
    manifest = json.loads((directory / "run_manifest.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    spec = manifest["spec"]
    run_fingerprint = fingerprint(spec)
    require(manifest["schema_version"] == "bagel_spatial_eval_manifest_v2", "unsupported manifest schema")
    require(manifest["spec_fingerprint"] == run_fingerprint == metrics["run_fingerprint"], "invalid run fingerprint")
    require(spec["evaluator_version"] == "bagel_spatial_eval_v2", "unsupported evaluator version")
    bspec, model = spec["benchmark"], spec["model"]
    checkpoint_key(model)
    require(bspec["benchmark"] == metrics["benchmark"] == benchmark, "benchmark mismatch")
    require(bspec["protocol_version"] == metrics["protocol_version"] == protocol, "nonofficial protocol")
    require(bspec["dataset"]["sha256"] == data_hash, "nonofficial dataset hash")
    require(bspec["raw_rows"] == official_sources and bspec["selected_source_rows"] == metrics["num_source_questions"] == source_count, "incorrect source selection")
    require(spec.get("scope") == {"partial": limit is not None, "source_limit": limit}, "full/partial run scope mismatch")
    require(metrics.get("partial") is (limit is not None) and metrics.get("source_limit") == limit, "metric full/partial scope mismatch")
    require(bspec["evaluation_items"] == metrics["num_evaluation_items"] == item_count, "incomplete evaluation selection")
    expected_protocol = {
        "sat-v2": dict(circular_order=["all_original", "all_reversed"], prompt_mode="sft", primary_aggregation="micro_accuracy_over_expanded_items"),
        "mmsi": dict(option_orders=["vanilla"], human_thought_used_as_input=False, post_prompt=mmsi.MMSI_POST_PROMPT, post_prompt_sha256=mmsi.MMSI_POST_PROMPT_SHA256, vanilla_prompt_sequence_sha256=mmsi.MMSI_VANILLA_PROMPTS_SHA256, primary_aggregation="vanilla_exact_match_accuracy"),
        "mindcube-tiny": dict(task_variant="raw_qa", official_code_revision=mindcube.MINDCUBE_CODE_REVISION, official_overall_excludes=["translation"], primary_aggregation="official filtered micro accuracy"),
    }[benchmark]
    require(all(bspec.get(k) == v for k, v in expected_protocol.items()), "nonofficial benchmark settings")
    require(metrics["primary_metric"] == {"sat-v2": "official_compat_accuracy", "mmsi": "official_vanilla_accuracy", "mindcube-tiny": "official_accuracy"}[benchmark], "unexpected primary metric")
    require(spec["decoding"] == dict(do_sample=False, temperature=0.0, thinking=False, max_new_tokens=token_limit, seed=0, image_transform_source="understanding/image_transform.yaml:vlm_sft"), "nonofficial decoding settings")
    require(spec["model_execution"] == dict(parameter_dtype="torch.bfloat16", autocast_dtype="torch.bfloat16", checkpoint_loading="stream_safetensors_tensor_by_tensor", replication="one_complete_model_per_process"), "nonofficial model execution")
    require(metrics["complete"] is True, "run is incomplete")
    overall = metrics["overall"]
    require(overall["total"] == overall["completed_responses"] == metrics["num_merged_records"] == metrics["expected_eval_ids"] == item_count, "incomplete result counts")
    require(overall["missing"] == overall["inference_errors"] == 0, "missing or failed predictions")
    require(type(metrics["duplicate_attempts"]) is int and metrics["duplicate_attempts"] >= 0, "invalid retry history")
    with (directory / "predictions.jsonl").open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    require(len(records) == item_count, "incorrect prediction count")
    require(len({r["eval_id"] for r in records}) == item_count, "duplicate prediction IDs")
    require([r["global_index"] for r in records] == list(range(item_count)), "noncanonical prediction order")
    counts, strict_count = [], 0
    for index, record in enumerate(records):
        require(record["schema_version"] == "bagel_spatial_prediction_v2" and record["status"] == "ok", "invalid or failed prediction")
        require(record["run_fingerprint"] == run_fingerprint and record["checkpoint_path"] == model["checkpoint_path"], "prediction identity mismatch")
        require(record["do_sample"] is False and record["temperature"] == 0.0 and record["max_new_tokens"] == token_limit, "prediction decoding mismatch")
        source_index = index % source_count
        variant = ("original" if index < source_count else "reversed") if benchmark == "sat-v2" else ("vanilla" if benchmark == "mmsi" else "raw_qa")
        require(record["source_index"] == source_index and record["variant"] == variant, "source/variant ordering mismatch")
        source_id = record["source_id"]
        if benchmark in ("sat-v2", "mmsi"):
            require(source_id == str(source_index), "noncanonical source_id")
        expected_id = f"sat_v2:test:{source_index}:{variant}" if benchmark == "sat-v2" else (f"mmsi:test:{source_id}:vanilla" if benchmark == "mmsi" else f"mindcube_tiny:{source_id}:raw_qa")
        require(record["eval_id"] == expected_id, "noncanonical eval_id")
        response, target, options = record["raw_response"], record["target_letter"], record["options"]
        require(isinstance(response, str) and isinstance(record["prompt"], str) and record["prompt"], "missing response or prompt")
        require(isinstance(options, list) and 2 <= len(options) <= 4 and all(isinstance(x, str) for x in options), "invalid options")
        require(isinstance(target, str) and len(target) == 1 and target in "ABCD"[:len(options)], "invalid target")
        require(options[ord(target) - 65] == record["target_text"], "target text mismatch")
        if benchmark == "sat-v2":
            require(len(options) == 2 and target == ("A" if variant == "original" else "B"), "nonofficial SAT options/target")
            require(record["prompt"] == sat_v2.build_sat_prompt(record["question"], options), "SAT prompt mismatch")
            prediction = sat_v2.parse_official_compat_answer(response, target)
            correct = prediction.lower().strip() == target.lower().strip()
            strict, _ = sat_v2.parse_strict_answer(response, "AB"[:len(options)])
            require(record["strict_correct"] is (strict == target), "SAT strict score mismatch")
            require(record["official_compat_prediction"] == prediction, "SAT parsed answer mismatch")
            strict_count += strict == target
            score_field, parsed = "official_compat_correct", strict
        elif benchmark == "mmsi":
            require(len(options) == 4 and record["prompt"] == mmsi.build_mmsi_prompt(record["question"]), "MMSI prompt/options mismatch")
            parsed, _ = mmsi.parse_mmsi_official_answer(response)
            correct, score_field = parsed == target, "official_correct"
        else:
            require(record["prompt"] == mindcube.build_mindcube_raw_qa_prompt(record["question"]), "MindCube prompt mismatch")
            require(record["category"] in mindcube.OFFICIAL_SETTING_COUNTS, "unexpected MindCube setting")
            cleaned = mindcube.clean_mindcube_response(response)
            require(record["official_response"] == cleaned, "MindCube cleaned response mismatch")
            parsed, _ = mindcube.parse_mindcube_official_answer(cleaned)
            correct, score_field = parsed == target, "official_correct"
        require(record[score_field] is correct and record["parse_success"] is (parsed is not None), "stored score differs from raw response")
        require(record["prediction"] == parsed, "stored prediction differs from raw response")
        counts.append(correct)
    require(len({r["source_id"] for r in records}) == source_count, "duplicate source IDs")
    if benchmark == "mindcube-tiny":
        require(prompt_digest(r["prompt"] for r in records) == bspec["selected_prompt_sequence_sha256"], "selected prompt sequence differs from manifest")
    if benchmark != "sat-v2" and limit is None:
        expected_digest = mmsi.MMSI_VANILLA_PROMPTS_SHA256 if benchmark == "mmsi" else mindcube.MINDCUBE_TINY_PROMPT_SEQUENCE_SHA256
        require(prompt_digest(r["prompt"] for r in records) == expected_digest, "nonofficial prediction prompt sequence")
    score = sum(counts)
    check_bucket(overall, score, item_count, "official_compat" if benchmark == "sat-v2" else "official")
    both = None
    if benchmark == "sat-v2":
        check_bucket(overall, strict_count, item_count, "strict")
        for original, reverse in zip(records[:source_count], records[source_count:]):
            require(all(original.get(k) == reverse.get(k) for k in ("source_id", "source_index", "question", "category", "question_type", "num_images", "target_text")) and original["options"] == reverse["options"][::-1], "invalid SAT order pair")
        both = sum(counts[i] and counts[i + source_count] for i in range(source_count))
        pairs = metrics["circular_pairs"]
        require(pairs["total"] == pairs["complete"] == source_count and pairs["official_compat_both_correct"] == both, "SAT both-order counts mismatch")
        require(math.isclose(pairs["official_compat_both_correct_percent"], 100 * both / source_count, abs_tol=1e-12), "SAT both-order accuracy mismatch")
    elif benchmark == "mmsi":
        check_bucket(metrics["vanilla"], score, item_count)
    else:
        require(overall["official_total"] == item_count, "MindCube official denominator mismatch")
    return dict(spec=spec, records=records, correct=score, total=item_count, both=both, directory=str(directory))


def model_key(model):
    return fingerprint({"checkpoint": checkpoint_key(model),
                        "assets": {name: value["sha256"] for name, value in model["assets"].items()},
                        "image_transform": model["image_transform"]["args"]})


def summarize(results_root, *, limit=None):
    results_root = Path(results_root).expanduser().resolve()
    rows = {}
    identities = set()
    for name, (source_count, item_count, _, _, _) in BENCHMARKS.items():
        directory = results_root / name
        entry = {"benchmark": name, "directory": str(directory), "status": "not_run",
                 "partial": limit is not None, "official_source_questions": source_count,
                 "official_evaluation_items": item_count}
        if (directory / "metrics.json").exists():
            try:
                metrics = json.loads((directory / "metrics.json").read_text())
                if metrics.get("complete") is not True or not (directory / "_SUCCESS").is_file():
                    entry.update(status="incomplete", counts=metrics.get("overall", {}))
                else:
                    run = load_run(directory, name, limit=limit)
                    identities.add(model_key(run["spec"]["model"]))
                    entry.update(status="partial" if limit is not None else "complete",
                                 source_questions=run["spec"]["benchmark"]["selected_source_rows"],
                                 correct=run["correct"], total=run["total"],
                                 accuracy_percent=100 * run["correct"] / run["total"],
                                 checkpoint=run["spec"]["model"],
                                 spec_fingerprint=fingerprint(run["spec"]))
                    if name == "sat-v2":
                        entry["both_order_accuracy_percent"] = 100 * run["both"] / entry["source_questions"]
                        entry["strict_accuracy_percent"] = metrics["overall"]["strict_accuracy_percent"]
            except (OSError, ValueError, TypeError, KeyError) as error:
                entry.update(status="invalid", error=str(error))
        rows[name] = entry
    if len(identities) > 1:
        raise ValueError("Understanding runs use different checkpoints, tokenizer assets, or image transforms")
    selected_complete = all(row["status"] in {"complete", "partial"} for row in rows.values())
    summary = {"schema": "mindview_understanding_summary_v1", "model": "bagel-sft-final",
               "status": "complete" if selected_complete and limit is None else "partial",
               "selected_runs_complete": selected_complete, "source_limit": limit,
               "benchmarks": rows,
               "note": "Scores are recomputed from saved raw answers using the fixed official-compatible scorers. SAT also reports strict and both-order diagnostics. Partial runs are not full-benchmark results."}
    atomic_write_json(results_root / "summary.json", summary)
    buffer = io.StringIO()
    fields = ("benchmark", "status", "correct", "total", "accuracy_percent", "both_order_accuracy_percent", "strict_accuracy_percent")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows.values())
    atomic_write_text(results_root / "summary.csv", buffer.getvalue())
    lines = ["# BAGEL-SFT spatial understanding", "", f"Status: **{summary['status']}**", "",
             "| Benchmark | Status | Correct / evaluated | Accuracy (%) |", "|---|---|---:|---:|"]
    for name, row in rows.items():
        count = f"{row['correct']} / {row['total']}" if "correct" in row else "—"
        score = f"{row['accuracy_percent']:.2f}" if "accuracy_percent" in row else "—"
        lines.append(f"| {name} | {row['status']} | {count} | {score} |")
    sat = rows["sat-v2"]
    if "both_order_accuracy_percent" in sat:
        lines += ["", f"SAT-v2 both-order accuracy: {sat['both_order_accuracy_percent']:.2f}%; strict accuracy: {sat['strict_accuracy_percent']:.2f}%."]
    if limit is not None:
        lines += ["", f"Source-question limit: {limit}. These are explicitly partial runs."]
    for name, row in rows.items():
        if "error" in row:
            lines += ["", f"{name}: {row['error']}"]
    lines += ["", summary["note"], ""]
    atomic_write_text(results_root / "summary.md", "\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    summary = summarize(args.results_root, limit=args.limit)
    print(json.dumps(summary, indent=2))
    if any(row["status"] == "invalid" for row in summary["benchmarks"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
