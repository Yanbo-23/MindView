"""SAT-v2 adapter following the current official public protocol.

Official sources:
  * Dataset: https://huggingface.co/datasets/array/SAT-v2
  * Circular evaluator:
    https://github.com/arijitray1993/lmms-eval/tree/main/lmms_eval/tasks/sat_real

The official test set has 150 source questions. Circular evaluation appends a
second copy with the answer choices reversed, yielding 300 model calls. The
official score is micro accuracy over those 300 calls.
"""

from __future__ import annotations

import hashlib
import io
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .base import BenchmarkAdapter, EvaluationSample, EvaluationTask

SAT_REPO_ID = "array/SAT-v2"
SAT_REVISION = "77d83da806d0a31f837be7b36d0dfbda230d8864"
SAT_TEST_FILE = "data/test-00000-of-00001.parquet"
SAT_TEST_SHA256 = "4880cf7b331d4398cd537d5faf96d26fa880ce9b6fd3c7dcdb35e253fa16091a"
SAT_PROTOCOL_VERSION = "sat_v2_official_circular_sft_v1"

OFFICIAL_CATEGORY_COUNTS = {
    "action_conseq": 37,
    "ego_movement": 23,
    "goal_aim": 34,
    "obj_movement": 23,
    "perspective": 33,
}
OFFICIAL_CATEGORY_LABELS = {
    "action_conseq": "EgoAct",
    "ego_movement": "EgoM",
    "goal_aim": "GoalAim",
    "obj_movement": "ObjM",
    "perspective": "Pers",
}
OFFICIAL_IMAGE_COUNTS = {1: 104, 2: 46}

_TAG_PATTERN = re.compile(r"<answer>\s*\(?\s*([A-Z])\s*\)?\s*</answer>", re.IGNORECASE)
_OFFICIAL_ANSWER_IS_PATTERN = re.compile(r"answer is\s*\n?\s*\(?([A-Z])\)?")
_STRICT_ANSWER_IS_PATTERN = re.compile(
    r"answer is\s*\n?\s*\(?([A-Z])\)?", re.IGNORECASE
)
_FINAL_ANSWER_PATTERN = re.compile(
    r"(?:final\s+answer|answer)\s*:\s*\(?\s*([A-Z])\s*\)?",
    re.IGNORECASE,
)
_STANDALONE_PATTERN = re.compile(r"^\s*\(?\s*([A-Z])\s*\)?[.\s]*$", re.IGNORECASE)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def add_sat_v2_arguments(parser: Any) -> None:
    group = parser.add_argument_group("SAT-v2 data")
    group.add_argument(
        "--sat-parquet",
        default="",
        help=(
            "Local SAT-v2 test parquet. If omitted, only the pinned official "
            "test parquet is downloaded from Hugging Face."
        ),
    )
    group.add_argument("--sat-repo-id", default=SAT_REPO_ID)
    group.add_argument("--sat-revision", default=SAT_REVISION)
    group.add_argument("--sat-filename", default=SAT_TEST_FILE)
    group.add_argument(
        "--hf-cache-dir",
        default=None,
        help=(
            "Optional Hugging Face cache used when --sat-parquet is omitted. "
            "By default huggingface_hub uses its standard configured cache."
        ),
    )
    group.add_argument(
        "--offline",
        action="store_true",
        help="Require the pinned parquet to already exist in the HF cache.",
    )
    group.add_argument(
        "--allow-nonstandard-data",
        action="store_true",
        help="Disable the official 150-row/category/image-count invariants.",
    )


def resolve_sat_parquet(args: Any, runtime: Any) -> tuple[Path, dict[str, Any]]:
    """Resolve/download the parquet on rank 0 and share its identity."""

    def resolve_on_primary() -> dict[str, Any]:
        if args.sat_parquet:
            path = Path(args.sat_parquet).expanduser().resolve()
            source = "local"
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:  # pragma: no cover - environment guard
                raise RuntimeError(
                    "huggingface_hub is required when --sat-parquet is omitted"
                ) from exc

            path = Path(
                hf_hub_download(
                    repo_id=args.sat_repo_id,
                    filename=args.sat_filename,
                    repo_type="dataset",
                    revision=args.sat_revision,
                    cache_dir=(
                        str(Path(args.hf_cache_dir).expanduser().resolve())
                        if args.hf_cache_dir
                        else None
                    ),
                    local_files_only=args.offline,
                )
            ).resolve()
            source = "huggingface_hub"

        if not path.is_file():
            raise FileNotFoundError(f"SAT-v2 test parquet does not exist: {path}")
        stat = path.stat()
        file_sha256 = sha256_file(path)
        if not args.allow_nonstandard_data and file_sha256 != SAT_TEST_SHA256:
            raise RuntimeError(
                "SAT-v2 parquet SHA256 differs from the pinned official file: "
                f"expected {SAT_TEST_SHA256}, got {file_sha256}. "
                "Use --allow-nonstandard-data only for an intentional custom copy."
            )
        return {
            "path": str(path),
            "source": source,
            "size_bytes": stat.st_size,
            "sha256": file_sha256,
            "expected_official_sha256": SAT_TEST_SHA256,
            "repo_id": args.sat_repo_id,
            "revision": args.sat_revision,
            "filename": args.sat_filename,
        }

    identity = runtime.primary_call(resolve_on_primary, label="resolve SAT-v2 data")
    runtime.barrier()
    return Path(identity["path"]), identity


def load_sat_rows(parquet_path: Path) -> list[dict[str, Any]]:
    """Load embedded images directly with PyArrow (no datasets dependency)."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "pyarrow is required for SAT-v2; install the BAGEL requirements first"
        ) from exc

    table = pq.read_table(
        parquet_path,
        columns=["images", "question", "answers", "correct_answer", "question_type"],
    )
    return table.to_pylist()


def build_sat_prompt(question: str, answers: Sequence[str]) -> str:
    """Build the current official lmms-eval ``prompt_mode=sft`` prompt."""

    if not answers or len(answers) > 26:
        raise ValueError(f"SAT answer count must be in [1, 26], got {len(answers)}")
    choices = "\n".join(
        f"({chr(65 + index)}) {answer}" for index, answer in enumerate(answers)
    )
    full_prompt = f"{question} Choose from the following options: \n{choices}"
    return (
        f"{full_prompt}\n"
        "Provide your final answer between the <answer> </answer> tags."
        " Please provide only the single option letter (e.g., A, B, C, D, etc.)"
        " within the <answer> </answer> tags."
    )


def parse_strict_answer(
    response: str, valid_letters: Sequence[str]
) -> tuple[str | None, str]:
    """Parse an answer without consulting the ground truth.

    The requested ``<answer>`` tag is preferred. Conservative fallbacks cover
    common BAGEL outputs while conflicting tagged answers remain a parse error.
    """

    valid = {letter.upper() for letter in valid_letters}

    tagged = [match.upper() for match in _TAG_PATTERN.findall(response)]
    tagged = [letter for letter in tagged if letter in valid]
    if tagged:
        if len(set(tagged)) == 1:
            return tagged[-1], "answer_tag"
        return None, "conflicting_answer_tags"

    answer_is = _STRICT_ANSWER_IS_PATTERN.search(response)
    if answer_is and answer_is.group(1).upper() in valid:
        return answer_is.group(1).upper(), "answer_is"

    final_answer = _FINAL_ANSWER_PATTERN.findall(response)
    final_answer = [
        letter.upper() for letter in final_answer if letter.upper() in valid
    ]
    if final_answer:
        if len(set(final_answer)) == 1:
            return final_answer[-1], "final_answer"
        return None, "conflicting_final_answers"

    standalone = _STANDALONE_PATTERN.match(response)
    if standalone and standalone.group(1).upper() in valid:
        return standalone.group(1).upper(), "standalone_letter"
    return None, "parse_failure"


def parse_official_compat_answer(response: str, target_letter: str) -> str:
    """Mirror SAT's current target-aware lmms-eval parser for comparability.

    This is intentionally reported separately from the strict parser because
    the official fallback searches the response for the ground-truth letter.
    """

    match = _OFFICIAL_ANSWER_IS_PATTERN.search(response)
    prediction = match.group(1) if match else ""
    if prediction != target_letter:
        if target_letter in response:
            prediction = target_letter
        else:
            prediction = response.strip()
    return prediction


def _decode_image(payload: Mapping[str, Any], parquet_path: Path) -> Image.Image:
    raw = payload.get("bytes")
    path_value = payload.get("path")
    if raw is not None:
        with Image.open(io.BytesIO(raw)) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()
    if path_value:
        path = Path(path_value)
        if not path.is_absolute():
            path = parquet_path.parent / path
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()
    raise ValueError("SAT image payload contains neither embedded bytes nor a path")


def _percent(correct: int, total: int) -> float:
    return 100.0 * correct / total if total else 0.0


class SatV2Benchmark(BenchmarkAdapter):
    name = "sat-v2"
    protocol_version = SAT_PROTOCOL_VERSION

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        parquet_path: Path,
        dataset_identity: Mapping[str, Any],
        *,
        limit: int | None = None,
        allow_nonstandard_data: bool = False,
    ) -> None:
        self._rows = [dict(row) for row in rows]
        self._parquet_path = parquet_path
        self._dataset_identity = dict(dataset_identity)
        self._validate_rows(allow_nonstandard_data=allow_nonstandard_data)

        if limit is not None:
            if limit <= 0:
                raise ValueError(f"--limit must be positive, got {limit}")
            selected_count = min(limit, len(self._rows))
        else:
            selected_count = len(self._rows)
        self._selected_count = selected_count

        tasks: list[EvaluationTask] = []
        variants = ("original", "reversed")
        global_index = 0
        for variant in variants:
            for source_index in range(selected_count):
                row = self._rows[source_index]
                tasks.append(
                    EvaluationTask(
                        global_index=global_index,
                        eval_id=f"sat_v2:test:{source_index}:{variant}",
                        source_index=source_index,
                        source_id=str(source_index),
                        category=str(row["question_type"]),
                        variant=variant,
                    )
                )
                global_index += 1
        self._tasks = tuple(tasks)

    @classmethod
    def from_args(cls, args: Any, runtime: Any) -> "SatV2Benchmark":
        parquet_path, identity = resolve_sat_parquet(args, runtime)
        rows = load_sat_rows(parquet_path)
        return cls(
            rows,
            parquet_path,
            identity,
            limit=args.limit,
            allow_nonstandard_data=args.allow_nonstandard_data,
        )

    def _validate_rows(self, *, allow_nonstandard_data: bool) -> None:
        required = {"images", "question", "answers", "correct_answer", "question_type"}
        for index, row in enumerate(self._rows):
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"SAT row {index} is missing fields: {sorted(missing)}"
                )
            answers = list(row["answers"])
            target = row["correct_answer"]
            if len(answers) < 2 or len(answers) > 26:
                raise ValueError(f"SAT row {index} has {len(answers)} answer choices")
            if answers.count(target) != 1:
                raise ValueError(
                    f"SAT row {index} must contain the correct answer exactly once"
                )
            if not row["images"]:
                raise ValueError(f"SAT row {index} contains no images")

        if allow_nonstandard_data:
            return

        if len(self._rows) != 150:
            raise ValueError(
                f"Official SAT-v2 test must contain 150 rows, got {len(self._rows)}; "
                "use --allow-nonstandard-data only for intentional custom data"
            )
        category_counts = Counter(str(row["question_type"]) for row in self._rows)
        if dict(category_counts) != OFFICIAL_CATEGORY_COUNTS:
            raise ValueError(
                "SAT-v2 category counts differ from the pinned official test: "
                f"{dict(category_counts)}"
            )
        image_counts = Counter(len(row["images"]) for row in self._rows)
        if dict(image_counts) != OFFICIAL_IMAGE_COUNTS:
            raise ValueError(
                "SAT-v2 image counts differ from the pinned official test: "
                f"{dict(image_counts)}"
            )
        if any(len(row["answers"]) != 2 for row in self._rows):
            raise ValueError("Official SAT-v2 test is expected to be entirely binary")
        if any(row["answers"].index(row["correct_answer"]) != 0 for row in self._rows):
            raise ValueError(
                "Pinned SAT-v2 test is expected to place every original GT at A"
            )

    @property
    def tasks(self) -> Sequence[EvaluationTask]:
        return self._tasks

    @property
    def required_ok_boolean_fields(self) -> Sequence[str]:
        return (
            "parse_success",
            "strict_correct",
            "official_compat_correct",
        )

    @property
    def manifest_fragment(self) -> Mapping[str, Any]:
        return {
            "benchmark": self.name,
            "protocol_version": self.protocol_version,
            "dataset": self._dataset_identity,
            "raw_rows": len(self._rows),
            "selected_source_rows": self._selected_count,
            "evaluation_items": len(self._tasks),
            "circular_order": ["all_original", "all_reversed"],
            "prompt_mode": "sft",
            "answer_format": "<answer>LETTER</answer>",
            "primary_aggregation": "micro_accuracy_over_expanded_items",
        }

    def materialize(self, task: EvaluationTask) -> EvaluationSample:
        row = self._rows[task.source_index]
        answers = list(row["answers"])
        if task.variant == "reversed":
            answers = answers[::-1]
        elif task.variant != "original":
            raise ValueError(f"Unknown SAT circular variant: {task.variant}")

        target_text = str(row["correct_answer"])
        target_index = answers.index(target_text)
        target_letter = chr(65 + target_index)
        images = [
            _decode_image(payload, self._parquet_path) for payload in row["images"]
        ]
        return EvaluationSample(
            task=task,
            images=images,
            prompt=build_sat_prompt(str(row["question"]), answers),
            target_letter=target_letter,
            target_text=target_text,
            options=tuple(str(answer) for answer in answers),
            metadata={"question": str(row["question"])},
        )

    def score_response(
        self, sample: EvaluationSample, response: str
    ) -> Mapping[str, Any]:
        valid_letters = [chr(65 + index) for index in range(len(sample.options))]
        strict_prediction, parse_method = parse_strict_answer(response, valid_letters)
        official_prediction = parse_official_compat_answer(
            response, sample.target_letter
        )
        official_normalized = official_prediction.lower().strip()
        official_target = sample.target_letter.lower().strip()

        strict_prediction_text = None
        if strict_prediction in valid_letters:
            strict_prediction_text = sample.options[ord(strict_prediction) - 65]
        official_prediction_text = None
        official_prediction_letter = official_normalized.upper()
        if official_prediction_letter in valid_letters:
            official_prediction_text = sample.options[
                ord(official_prediction_letter) - 65
            ]

        return {
            "prediction": strict_prediction,
            "prediction_text": strict_prediction_text,
            "parse_method": parse_method,
            "parse_success": strict_prediction is not None,
            "strict_correct": strict_prediction == sample.target_letter,
            "official_compat_prediction": official_prediction,
            "official_compat_prediction_text": official_prediction_text,
            "official_compat_correct": official_normalized == official_target,
        }

    def validate_record(
        self, task: EvaluationTask, record: Mapping[str, Any]
    ) -> Sequence[str]:
        if record.get("status") != "ok":
            return ()

        errors: list[str] = []
        row = self._rows[task.source_index]
        expected_options = [str(answer) for answer in row["answers"]]
        if task.variant == "reversed":
            expected_options.reverse()
        expected_target_text = str(row["correct_answer"])
        expected_target_letter = chr(65 + expected_options.index(expected_target_text))
        expected_values = {
            "question": str(row["question"]),
            "prompt": build_sat_prompt(str(row["question"]), expected_options),
            "options": expected_options,
            "target_letter": expected_target_letter,
            "target_text": expected_target_text,
            "num_images": len(row["images"]),
        }
        for key, expected in expected_values.items():
            if record.get(key) != expected:
                errors.append(f"{key} does not match the canonical SAT task")

        response = record.get("raw_response")
        if isinstance(response, str):
            sample = EvaluationSample(
                task=task,
                images=[],
                prompt=expected_values["prompt"],
                target_letter=expected_target_letter,
                target_text=expected_target_text,
                options=tuple(expected_options),
            )
            expected_score = self.score_response(sample, response)
            for key, expected in expected_score.items():
                if record.get(key) != expected:
                    errors.append(f"{key} does not match rescoring raw_response")
        return tuple(errors)

    def _bucket(
        self,
        tasks: Sequence[EvaluationTask],
        records_by_id: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        records = [records_by_id.get(task.eval_id) for task in tasks]
        total = len(tasks)
        completed = sum(
            record is not None and record.get("status") == "ok" for record in records
        )
        official_correct = sum(
            bool(record and record.get("official_compat_correct")) for record in records
        )
        strict_correct = sum(
            bool(record and record.get("strict_correct")) for record in records
        )
        parsed = sum(bool(record and record.get("parse_success")) for record in records)
        errors = sum(
            bool(record and record.get("status") == "inference_error")
            for record in records
        )
        missing = sum(record is None for record in records)
        return {
            "total": total,
            "completed_responses": completed,
            "missing": missing,
            "inference_errors": errors,
            "official_compat_correct": official_correct,
            "official_compat_accuracy": official_correct / total if total else 0.0,
            "official_compat_accuracy_percent": _percent(official_correct, total),
            "strict_correct": strict_correct,
            "strict_accuracy": strict_correct / total if total else 0.0,
            "strict_accuracy_percent": _percent(strict_correct, total),
            "strict_parsed": parsed,
            "strict_parse_rate": parsed / total if total else 0.0,
            "strict_parse_rate_percent": _percent(parsed, total),
        }

    def aggregate(
        self, records_by_id: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        overall = self._bucket(self._tasks, records_by_id)
        categories: dict[str, Any] = {}
        for category in sorted({task.category for task in self._tasks}):
            category_tasks = [task for task in self._tasks if task.category == category]
            categories[category] = {
                "label": OFFICIAL_CATEGORY_LABELS.get(category, category),
                **self._bucket(category_tasks, records_by_id),
            }

        pair_total = self._selected_count
        official_both = strict_both = 0
        strict_semantic_consistent = 0
        complete_pairs = 0
        by_pair: dict[tuple[int, str], Mapping[str, Any] | None] = {}
        for task in self._tasks:
            by_pair[(task.source_index, task.variant)] = records_by_id.get(task.eval_id)
        for source_index in range(self._selected_count):
            original = by_pair.get((source_index, "original"))
            reversed_record = by_pair.get((source_index, "reversed"))
            if not original or not reversed_record:
                continue
            if original.get("status") == "ok" and reversed_record.get("status") == "ok":
                complete_pairs += 1
            if original.get("official_compat_correct") and reversed_record.get(
                "official_compat_correct"
            ):
                official_both += 1
            if original.get("strict_correct") and reversed_record.get("strict_correct"):
                strict_both += 1
            left = original.get("prediction_text")
            right = reversed_record.get("prediction_text")
            if left is not None and left == right:
                strict_semantic_consistent += 1

        return {
            "benchmark": self.name,
            "protocol_version": self.protocol_version,
            "primary_metric": "official_compat_accuracy",
            "note": (
                "official_compat mirrors the released SAT scorer, including its "
                "target-aware fallback; strict_accuracy is target-independent"
            ),
            "num_source_questions": self._selected_count,
            "num_evaluation_items": len(self._tasks),
            "overall": overall,
            "per_category": categories,
            "circular_pairs": {
                "total": pair_total,
                "complete": complete_pairs,
                "official_compat_both_correct": official_both,
                "official_compat_both_correct_percent": _percent(
                    official_both, pair_total
                ),
                "strict_both_correct": strict_both,
                "strict_both_correct_percent": _percent(strict_both, pair_total),
                "strict_semantic_consistent": strict_semantic_consistent,
                "strict_semantic_consistency_percent": _percent(
                    strict_semantic_consistent, pair_total
                ),
            },
        }

    def summary_lines(self, metrics: Mapping[str, Any]) -> Sequence[str]:
        overall = metrics["overall"]
        lines = [
            "=== SAT-v2 summary ===",
            f"Complete: {metrics['complete']}",
            (
                "Official-compatible accuracy: "
                f"{overall['official_compat_accuracy_percent']:.2f}% "
                f"({overall['official_compat_correct']}/{overall['total']})"
            ),
            (
                "Strict accuracy: "
                f"{overall['strict_accuracy_percent']:.2f}% "
                f"({overall['strict_correct']}/{overall['total']})"
            ),
            (
                "Strict parse rate: "
                f"{overall['strict_parse_rate_percent']:.2f}% "
                f"({overall['strict_parsed']}/{overall['total']})"
            ),
        ]
        for category, values in metrics["per_category"].items():
            lines.append(
                f"  {values['label']} ({category}): "
                f"official={values['official_compat_accuracy_percent']:.2f}% "
                f"strict={values['strict_accuracy_percent']:.2f}% "
                f"N={values['total']}"
            )
        if overall["missing"] or overall["inference_errors"]:
            lines.append(
                f"Incomplete items: missing={overall['missing']}, "
                f"inference_errors={overall['inference_errors']}"
            )
        return tuple(lines)
