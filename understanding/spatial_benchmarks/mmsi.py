"""MMSI-Bench adapter matching the released official evaluation protocol.

Official sources pinned by this adapter:

* Dataset: https://huggingface.co/datasets/RunsenXu/MMSI-Bench
* Evaluation code:
  https://github.com/InternRobotics/MMSI-Bench/blob/main/vlmeval/dataset/mmsi_bench.py

The Vanilla protocol evaluates the 1,000 original four-choice questions.  The
Circular protocol uses the official ``ABCD, BCDA, CDAB, DABC`` option orders
and counts a source question as correct only when all four variants are
correct.  MMSI's released, case-sensitive regular-expression extractor is the
primary and only answer extractor here; no ground truth is exposed to an
auxiliary answer-extraction LLM.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .base import BenchmarkAdapter, EvaluationSample, EvaluationTask

MMSI_REPO_ID = "RunsenXu/MMSI-Bench"
MMSI_REVISION = "ec7c92bfaf7728fcca1d61e3e224e190af309436"
MMSI_PARQUET_FILENAME = "MMSI_Bench.parquet"
MMSI_PARQUET_SHA256 = (
    "72a9c94699ca88f3eba79f330ab28965ee537bccd7cbf563265982a6ff8b74b4"
)
MMSI_OFFICIAL_CODE_COMMIT = "13e58a2b8b30d880d7e8a1e4a6aa1c0feda94cac"
MMSI_OFFICIAL_CODE_FILE = "vlmeval/dataset/mmsi_bench.py"
MMSI_OFFICIAL_CODE_SHA256 = (
    "e551ea1d3ca87c72196dd837d5beec545398d06359bfc5d235558904cf95bec7"
)
MMSI_POST_PROMPT = (
    "Answer with the option's letter from the given choices directly. "
    "Enclose the option's letter within ``."
)
MMSI_POST_PROMPT_SHA256 = (
    "53c4ef680df1519b9e56c300bcb25a637609f17bba653c8373bfa3b3196eb90d"
)
MMSI_VANILLA_PROMPTS_SHA256 = (
    "c6553d14b0d6ec834dddc1187fe359f5d6219271098a3ac64715297ee847cc06"
)
MMSI_CIRCULAR_ORDERS = ("ABCD", "BCDA", "CDAB", "DABC")
MMSI_OPTION_PATTERN = re.compile(
    r"([A-D])\s*:\s*(.*?)(?=\s+[A-D]\s*:|,\s*[A-D]\s*:|$)"
)
_DOUBLE_BACKTICK_PATTERN = re.compile(r"``([^`]*)``")
_SINGLE_BACKTICK_PATTERN = re.compile(r"`([^`]*)`")
_OFFICIAL_LETTER_PATTERN = re.compile(r"\b[A-D]\b(?!\s[a-zA-Z])")

MMSI_VANILLA_PROTOCOL_VERSION = "mmsi_official_vanilla_regex_13e58a2_v1"
MMSI_CIRCULAR_PROTOCOL_VERSION = "mmsi_official_circular_cp4_regex_13e58a2_v1"

OFFICIAL_CATEGORY_COUNTS = {
    "Attribute (Appr.)": 66,
    "Attribute (Meas.)": 64,
    "MSR": 198,
    "Motion (Cam.)": 74,
    "Motion (Obj.)": 76,
    "Positional Relationship (Cam.–Cam.)": 93,
    "Positional Relationship (Cam.–Obj.)": 86,
    "Positional Relationship (Cam.–Reg.)": 83,
    "Positional Relationship (Obj.–Obj.)": 94,
    "Positional Relationship (Obj.–Reg.)": 85,
    "Positional Relationship (Reg.–Reg.)": 81,
}
OFFICIAL_CATEGORY_MAP = {
    "Motion (Cam.)": "运动感知 camera",
    "Positional Relationship (Cam.–Obj.)": "位置关系 cam-object",
    "Attribute (Meas.)": "几何属性 measurement",
    "Positional Relationship (Reg.–Reg.)": "位置关系 scene-scene",
    "MSR": "多步推理",
    "Motion (Obj.)": "运动感知 object",
    "Positional Relationship (Cam.–Cam.)": "位置关系 cam-cam",
    "Positional Relationship (Cam.–Reg.)": "位置关系 cam-scene",
    "Attribute (Appr.)": "几何属性 shape",
    "Positional Relationship (Obj.–Reg.)": "位置关系 object-scene",
    "Positional Relationship (Obj.–Obj.)": "位置关系 object-object",
}
# This is the first-occurrence order produced by pandas ``unique()`` on the
# official TSV and therefore matches the released category report ordering.
OFFICIAL_CATEGORY_ORDER = tuple(OFFICIAL_CATEGORY_MAP.values())
OFFICIAL_DIFFICULTY_COUNTS = {"easy": 334, "hard": 333, "medium": 333}
OFFICIAL_ANSWER_COUNTS = {"A": 265, "B": 250, "C": 255, "D": 230}
OFFICIAL_IMAGE_COUNTS = {
    2: 810,
    3: 58,
    4: 40,
    5: 19,
    6: 38,
    7: 15,
    8: 15,
    9: 2,
    10: 3,
}


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def add_mmsi_arguments(parser: Any) -> None:
    group = parser.add_argument_group("MMSI-Bench data")
    group.add_argument(
        "--mmsi-parquet",
        default="",
        help=(
            "Optional local official MMSI_Bench.parquet. If omitted, the pinned "
            "file is fetched from Hugging Face."
        ),
    )
    group.add_argument("--mmsi-repo-id", default=MMSI_REPO_ID)
    group.add_argument("--mmsi-revision", default=MMSI_REVISION)
    group.add_argument("--mmsi-filename", default=MMSI_PARQUET_FILENAME)
    group.add_argument(
        "--mmsi-hf-cache-dir",
        default=None,
        help=(
            "Optional Hugging Face cache used when --mmsi-parquet is omitted. "
            "By default huggingface_hub uses its standard configured cache."
        ),
    )
    group.add_argument(
        "--offline",
        action="store_true",
        help="Require the pinned MMSI parquet to exist locally/in the HF cache.",
    )
    group.add_argument(
        "--allow-nonstandard-data",
        action="store_true",
        help="Disable the official hash and 1,000-row distribution invariants.",
    )


def resolve_mmsi_parquet(args: Any, runtime: Any) -> tuple[Path, dict[str, Any]]:
    """Resolve and fingerprint the dataset once on rank zero."""

    def resolve_on_primary() -> dict[str, Any]:
        if args.mmsi_parquet:
            path = Path(args.mmsi_parquet).expanduser().resolve()
            source = "local"
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:  # pragma: no cover - environment guard
                raise RuntimeError(
                    "huggingface_hub is required when --mmsi-parquet is empty"
                ) from exc
            path = Path(
                hf_hub_download(
                    repo_id=args.mmsi_repo_id,
                    filename=args.mmsi_filename,
                    repo_type="dataset",
                    revision=args.mmsi_revision,
                    cache_dir=(
                        str(Path(args.mmsi_hf_cache_dir).expanduser().resolve())
                        if args.mmsi_hf_cache_dir
                        else None
                    ),
                    local_files_only=args.offline,
                )
            ).resolve()
            source = "huggingface_hub"

        if not path.is_file():
            raise FileNotFoundError(f"MMSI-Bench parquet does not exist: {path}")
        stat = path.stat()
        file_sha256 = _sha256_file(path)
        if (
            not args.allow_nonstandard_data
            and file_sha256 != MMSI_PARQUET_SHA256
        ):
            raise RuntimeError(
                "MMSI-Bench parquet SHA256 differs from the pinned official file: "
                f"expected {MMSI_PARQUET_SHA256}, got {file_sha256}. "
                "Use --allow-nonstandard-data only for an intentional custom copy."
            )
        return {
            "path": str(path),
            "source": source,
            "size_bytes": stat.st_size,
            "sha256": file_sha256,
            "expected_official_sha256": MMSI_PARQUET_SHA256,
            "repo_id": args.mmsi_repo_id,
            "revision": args.mmsi_revision,
            "filename": args.mmsi_filename,
        }

    identity = runtime.primary_call(
        resolve_on_primary, label="resolve MMSI-Bench data"
    )
    runtime.barrier()
    return Path(identity["path"]), identity


def load_mmsi_rows(parquet_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load official columns without loading the human ``thought`` as model input."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("pyarrow is required for MMSI-Bench") from exc

    parquet = pq.ParquetFile(parquet_path)
    schema_names = list(parquet.schema_arrow.names)
    required_schema = {
        "id",
        "images",
        "question_type",
        "question",
        "answer",
        "thought",
        "mean_normed_duration_seconds",
        "difficulty",
    }
    missing = required_schema - set(schema_names)
    if missing:
        raise ValueError(
            f"MMSI-Bench parquet is missing columns: {sorted(missing)}"
        )

    # ``thought`` is deliberately excluded: it is a human rationale/label-side
    # annotation and must never enter the BAGEL prompt.
    table = parquet.read(
        columns=[
            "id",
            "images",
            "question_type",
            "question",
            "answer",
            "mean_normed_duration_seconds",
            "difficulty",
        ]
    )
    return table.to_pylist(), schema_names


def extract_mmsi_question_options(question: str) -> tuple[str, dict[str, str]]:
    """Mirror ``MMSIBenchCircular.extract_options_from_question`` exactly."""

    parts = question.split("Options:", 1)
    if len(parts) < 2:
        return question.strip(), {}
    question_text = parts[0].strip()
    options_text = parts[1].strip()
    options = {
        letter: value.strip()
        for letter, value in MMSI_OPTION_PATTERN.findall(options_text)
    }
    return question_text, options


def build_mmsi_question_with_options(
    question_text: str, options: Mapping[str, str]
) -> str:
    """Mirror the official Circular question reconstruction."""

    options_text = "Options: " + ", ".join(
        f"{letter}: {value}" for letter, value in options.items()
    )
    return f"{question_text}\n{options_text}"


def build_mmsi_prompt(question: str) -> str:
    """Append the released MMSI-specific post-prompt verbatim."""

    return f"{question}\n{MMSI_POST_PROMPT}"


def parse_mmsi_official_answer(response: str) -> tuple[str | None, str]:
    """Mirror MMSI's released regex extractor for the default model path.

    The official implementation is intentionally case-sensitive and selects
    the first standalone uppercase A-D after applying its backtick handling.
    """

    prediction = str(response)

    method = "word_boundary"
    match = _DOUBLE_BACKTICK_PATTERN.search(prediction)
    if match:
        prediction = match.group(1)
        method = "double_backticks"
    match = _SINGLE_BACKTICK_PATTERN.search(prediction)
    if match:
        prediction = match.group(1)
        method = "single_backticks"
    match = _OFFICIAL_LETTER_PATTERN.search(prediction)
    if not match:
        return None, "parse_failure"
    return match.group(), method


def _decode_image(payload: Any, parquet_path: Path) -> Image.Image:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        raw = bytes(payload)
        with Image.open(io.BytesIO(raw)) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()
    if isinstance(payload, Mapping):
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
    raise ValueError("MMSI image payload contains neither bytes nor a path")


def _percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


class MmsiBenchmark(BenchmarkAdapter):
    """Official MMSI Vanilla or Circular evaluation over embedded images."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        parquet_path: Path,
        dataset_identity: Mapping[str, Any],
        *,
        mode: str,
        limit: int | None = None,
        allow_nonstandard_data: bool = False,
    ) -> None:
        if mode not in {"vanilla", "circular"}:
            raise ValueError(f"Unknown MMSI protocol mode: {mode}")
        self.mode = mode
        self.name = "mmsi" if mode == "vanilla" else "mmsi-circular"
        self.protocol_version = (
            MMSI_VANILLA_PROTOCOL_VERSION
            if mode == "vanilla"
            else MMSI_CIRCULAR_PROTOCOL_VERSION
        )
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
        self._orders = (
            ("vanilla",) if mode == "vanilla" else MMSI_CIRCULAR_ORDERS
        )

        tasks: list[EvaluationTask] = []
        for source_index in range(selected_count):
            row = self._rows[source_index]
            source_id = str(row["id"])
            for variant in self._orders:
                tasks.append(
                    EvaluationTask(
                        global_index=len(tasks),
                        eval_id=f"mmsi:test:{source_id}:{variant}",
                        source_index=source_index,
                        source_id=source_id,
                        category=OFFICIAL_CATEGORY_MAP.get(
                            str(row["question_type"]), str(row["question_type"])
                        ),
                        variant=variant,
                    )
                )
        self._tasks = tuple(tasks)
        self._task_lookup = {
            (task.source_index, task.variant): task for task in self._tasks
        }

    @classmethod
    def from_args(cls, args: Any, runtime: Any) -> MmsiBenchmark:
        parquet_path, identity = resolve_mmsi_parquet(args, runtime)
        rows, schema_names = load_mmsi_rows(parquet_path)
        identity["schema_columns"] = schema_names
        if args.benchmark == "mmsi":
            mode = "vanilla"
        elif args.benchmark == "mmsi-circular":
            mode = "circular"
        else:  # pragma: no cover - registry guard
            raise ValueError(f"MMSI adapter received benchmark={args.benchmark!r}")
        return cls(
            rows,
            parquet_path,
            identity,
            mode=mode,
            limit=args.limit,
            allow_nonstandard_data=args.allow_nonstandard_data,
        )

    def _validate_rows(self, *, allow_nonstandard_data: bool) -> None:
        required = {
            "id",
            "images",
            "question_type",
            "question",
            "answer",
            "mean_normed_duration_seconds",
            "difficulty",
        }
        ids: list[int] = []
        for index, row in enumerate(self._rows):
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"MMSI row {index} is missing fields: {sorted(missing)}"
                )
            try:
                source_id = int(row["id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"MMSI row {index} has an invalid id") from exc
            ids.append(source_id)

            question = row["question"]
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"MMSI row {index} has an invalid question")
            _, options = extract_mmsi_question_options(question)
            if list(options) != list("ABCD"):
                raise ValueError(
                    f"MMSI row {index} does not contain canonical A-D options"
                )
            answer = str(row["answer"])
            if answer not in options:
                raise ValueError(
                    f"MMSI row {index} has invalid answer {answer!r}"
                )
            images = row["images"]
            if not isinstance(images, Sequence) or isinstance(
                images, (str, bytes, bytearray)
            ):
                raise TypeError(f"MMSI row {index} has invalid images")
            if not 2 <= len(images) <= 10:
                raise ValueError(
                    f"MMSI row {index} has {len(images)} images; expected 2-10"
                )
            if any(
                not isinstance(image, (bytes, bytearray, memoryview, Mapping))
                for image in images
            ):
                raise ValueError(f"MMSI row {index} has an invalid image payload")
            if not str(row["question_type"]).strip():
                raise ValueError(f"MMSI row {index} has no question_type")
            if str(row["difficulty"]) not in {"easy", "medium", "hard"}:
                raise ValueError(f"MMSI row {index} has invalid difficulty")
            duration = row["mean_normed_duration_seconds"]
            if not isinstance(duration, (int, float)) or not math.isfinite(duration):
                raise ValueError(f"MMSI row {index} has invalid response duration")

        if len(ids) != len(set(ids)):
            raise ValueError("MMSI source ids must be unique")
        if allow_nonstandard_data:
            return

        if len(self._rows) != 1000:
            raise ValueError(
                f"Official MMSI-Bench must contain 1,000 rows, got {len(self._rows)}"
            )
        if ids != list(range(1000)):
            raise ValueError("Official MMSI-Bench ids must be the canonical 0-999")
        observed = {
            "category": dict(
                sorted(Counter(str(row["question_type"]) for row in self._rows).items())
            ),
            "difficulty": dict(
                sorted(Counter(str(row["difficulty"]) for row in self._rows).items())
            ),
            "answer": dict(
                sorted(Counter(str(row["answer"]) for row in self._rows).items())
            ),
            "image_count": dict(
                sorted(Counter(len(row["images"]) for row in self._rows).items())
            ),
        }
        expected = {
            "category": OFFICIAL_CATEGORY_COUNTS,
            "difficulty": OFFICIAL_DIFFICULTY_COUNTS,
            "answer": OFFICIAL_ANSWER_COUNTS,
            "image_count": OFFICIAL_IMAGE_COUNTS,
        }
        for key, expected_counts in expected.items():
            if observed[key] != expected_counts:
                raise ValueError(
                    f"MMSI official {key} distribution differs: {observed[key]}"
                )

        prompt_digest = hashlib.sha256()
        for row in self._rows:
            prompt_bytes = build_mmsi_prompt(str(row["question"])).encode("utf-8")
            prompt_digest.update(len(prompt_bytes).to_bytes(8, "big"))
            prompt_digest.update(prompt_bytes)
        if prompt_digest.hexdigest() != MMSI_VANILLA_PROMPTS_SHA256:
            raise ValueError(
                "MMSI Vanilla prompts differ from the pinned official prompts"
            )

    @property
    def tasks(self) -> Sequence[EvaluationTask]:
        return self._tasks

    @property
    def required_ok_boolean_fields(self) -> Sequence[str]:
        return ("parse_success", "official_correct")

    @property
    def manifest_fragment(self) -> Mapping[str, Any]:
        return {
            "benchmark": self.name,
            "protocol_version": self.protocol_version,
            "dataset": self._dataset_identity,
            "raw_rows": len(self._rows),
            "selected_source_rows": self._selected_count,
            "evaluation_items": len(self._tasks),
            "official_code": {
                "repository": "https://github.com/InternRobotics/MMSI-Bench",
                "commit": MMSI_OFFICIAL_CODE_COMMIT,
                "file": MMSI_OFFICIAL_CODE_FILE,
                "file_sha256": MMSI_OFFICIAL_CODE_SHA256,
            },
            "task_order": "source_row_then_option_order",
            "option_orders": list(self._orders),
            "post_prompt": MMSI_POST_PROMPT,
            "post_prompt_sha256": MMSI_POST_PROMPT_SHA256,
            "vanilla_prompt_sequence_sha256": MMSI_VANILLA_PROMPTS_SHA256,
            "answer_format": "uppercase A-D enclosed within double backticks",
            "answer_extraction": {
                "mode": "official_case_sensitive_regex_without_llm_fallback",
                "patterns": [
                    _DOUBLE_BACKTICK_PATTERN.pattern,
                    _SINGLE_BACKTICK_PATTERN.pattern,
                    _OFFICIAL_LETTER_PATTERN.pattern,
                ],
            },
            "human_thought_used_as_input": False,
            "parquet_question_type_to_official_tsv_category": dict(
                OFFICIAL_CATEGORY_MAP
            ),
            "primary_aggregation": (
                "vanilla_exact_match_accuracy"
                if self.mode == "vanilla"
                else "four_variants_all_correct_per_source_question"
            ),
        }

    def _canonical_sample_fields(
        self, task: EvaluationTask
    ) -> tuple[str, dict[str, str], str, str, Mapping[str, Any]]:
        row = self._rows[task.source_index]
        raw_question = str(row["question"])
        question_text, original_options = extract_mmsi_question_options(raw_question)
        original_answer = str(row["answer"])

        if self.mode == "vanilla":
            if task.variant != "vanilla":
                raise ValueError(f"Unknown MMSI Vanilla variant: {task.variant}")
            question = raw_question
            options = original_options
            target_letter = original_answer
        else:
            order = task.variant
            if order not in MMSI_CIRCULAR_ORDERS:
                raise ValueError(f"Unknown MMSI Circular variant: {order}")
            options = {
                new_letter: original_options[old_letter]
                for new_letter, old_letter in zip("ABCD", order)
            }
            question = build_mmsi_question_with_options(question_text, options)
            target_letter = "ABCD"[order.index(original_answer)]

        target_text = options[target_letter]
        return question, options, target_letter, target_text, row

    def materialize(self, task: EvaluationTask) -> EvaluationSample:
        question, options, target_letter, target_text, row = (
            self._canonical_sample_fields(task)
        )
        images = [
            _decode_image(payload, self._parquet_path) for payload in row["images"]
        ]
        source_id = int(row["id"])
        rotation_index = (
            0
            if self.mode == "vanilla"
            else MMSI_CIRCULAR_ORDERS.index(task.variant)
        )
        return EvaluationSample(
            task=task,
            images=images,
            prompt=build_mmsi_prompt(question),
            target_letter=target_letter,
            target_text=target_text,
            options=tuple(options[letter] for letter in "ABCD"),
            metadata={
                "question": question,
                "question_type": str(row["question_type"]),
                "difficulty": str(row["difficulty"]),
                "official_index": source_id + rotation_index * 1_000_000,
                "official_group_index": source_id,
            },
        )

    def score_response(
        self, sample: EvaluationSample, response: str
    ) -> Mapping[str, Any]:
        prediction, parse_method = parse_mmsi_official_answer(response)
        prediction_text = None
        if prediction is not None and prediction in "ABCD":
            prediction_text = sample.options[ord(prediction) - ord("A")]
        return {
            "prediction": prediction,
            "prediction_text": prediction_text,
            "parse_method": parse_method,
            "parse_success": prediction is not None,
            "official_correct": prediction == sample.target_letter,
        }

    def validate_record(
        self, task: EvaluationTask, record: Mapping[str, Any]
    ) -> Sequence[str]:
        if record.get("status") != "ok":
            return ()

        question, options, target_letter, target_text, row = (
            self._canonical_sample_fields(task)
        )
        expected_values = {
            "question": question,
            "question_type": str(row["question_type"]),
            "difficulty": str(row["difficulty"]),
            "official_index": int(row["id"])
            + (
                0
                if self.mode == "vanilla"
                else MMSI_CIRCULAR_ORDERS.index(task.variant) * 1_000_000
            ),
            "official_group_index": int(row["id"]),
            "prompt": build_mmsi_prompt(question),
            "options": [options[letter] for letter in "ABCD"],
            "target_letter": target_letter,
            "target_text": target_text,
            "num_images": len(row["images"]),
        }
        errors: list[str] = []
        for key, expected in expected_values.items():
            if record.get(key) != expected:
                errors.append(f"{key} does not match the canonical MMSI task")

        response = record.get("raw_response")
        if isinstance(response, str):
            sample = EvaluationSample(
                task=task,
                images=[],
                prompt=expected_values["prompt"],
                target_letter=target_letter,
                target_text=target_text,
                options=tuple(expected_values["options"]),
            )
            for key, expected in self.score_response(sample, response).items():
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
        errors = sum(
            record is not None and record.get("status") == "inference_error"
            for record in records
        )
        missing = sum(record is None for record in records)
        parsed = sum(bool(record and record.get("parse_success")) for record in records)
        correct = sum(bool(record and record.get("official_correct")) for record in records)
        prediction_distribution: Counter[str] = Counter()
        for record in records:
            if record is None or record.get("status") != "ok":
                continue
            prediction = record.get("prediction")
            key = (
                prediction
                if isinstance(prediction, str) and prediction in "ABCD"
                else "unparsed"
            )
            prediction_distribution[key] += 1
        return {
            "total": total,
            "completed_responses": completed,
            "missing": missing,
            "inference_errors": errors,
            "official_correct": correct,
            "official_accuracy": correct / total if total else 0.0,
            "official_accuracy_percent": _percent(correct, total),
            "parsed": parsed,
            "parse_rate": parsed / total if total else 0.0,
            "parse_rate_percent": _percent(parsed, total),
            "prediction_distribution": {
                key: prediction_distribution.get(key, 0)
                for key in ("A", "B", "C", "D", "unparsed")
            },
        }

    def _circular_bucket(
        self,
        source_indices: Sequence[int],
        records_by_id: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        total = len(source_indices)
        complete = 0
        correct = 0
        for source_index in source_indices:
            records = [
                records_by_id.get(
                    self._task_lookup[(source_index, order)].eval_id
                )
                for order in MMSI_CIRCULAR_ORDERS
            ]
            group_complete = all(
                record is not None and record.get("status") == "ok"
                for record in records
            )
            if group_complete:
                complete += 1
            if group_complete and all(
                bool(record and record.get("official_correct")) for record in records
            ):
                correct += 1
        return {
            "total": total,
            "complete_groups": complete,
            "missing_or_error_groups": total - complete,
            "official_correct": correct,
            "official_accuracy": correct / total if total else 0.0,
            "official_accuracy_percent": _percent(correct, total),
        }

    def _source_indices_for(self, field: str, value: Any) -> list[int]:
        def field_value(index: int) -> Any:
            if field == "images":
                return len(self._rows[index][field])
            if field == "official_category":
                question_type = str(self._rows[index]["question_type"])
                return OFFICIAL_CATEGORY_MAP.get(question_type, question_type)
            return str(self._rows[index][field])

        return [
            index
            for index in range(self._selected_count)
            if field_value(index) == value
        ]

    def _dimension_metrics(
        self,
        field: str,
        values: Sequence[Any],
        records_by_id: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for value in values:
            source_indices = self._source_indices_for(field, value)
            if self.mode == "vanilla":
                tasks = [
                    self._task_lookup[(source_index, "vanilla")]
                    for source_index in source_indices
                ]
                output[str(value)] = self._bucket(tasks, records_by_id)
            else:
                vanilla_tasks = [
                    self._task_lookup[(source_index, "ABCD")]
                    for source_index in source_indices
                ]
                output[str(value)] = {
                    "vanilla": self._bucket(vanilla_tasks, records_by_id),
                    "circular": self._circular_bucket(
                        source_indices, records_by_id
                    ),
                }
        return output

    def _image_bucket_metrics(
        self, records_by_id: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        bucket_ranges = {
            "2": {2},
            "3-6": {3, 4, 5, 6},
            "7-10": {7, 8, 9, 10},
        }
        output: dict[str, Any] = {}
        for label, counts in bucket_ranges.items():
            source_indices = [
                index
                for index in range(self._selected_count)
                if len(self._rows[index]["images"]) in counts
            ]
            if not source_indices:
                continue
            if self.mode == "vanilla":
                tasks = [
                    self._task_lookup[(source_index, "vanilla")]
                    for source_index in source_indices
                ]
                output[label] = self._bucket(tasks, records_by_id)
            else:
                vanilla_tasks = [
                    self._task_lookup[(source_index, "ABCD")]
                    for source_index in source_indices
                ]
                output[label] = {
                    "vanilla": self._bucket(vanilla_tasks, records_by_id),
                    "circular": self._circular_bucket(
                        source_indices, records_by_id
                    ),
                }
        return output

    def aggregate(
        self, records_by_id: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        observed_categories = {
            OFFICIAL_CATEGORY_MAP.get(
                str(self._rows[index]["question_type"]),
                str(self._rows[index]["question_type"]),
            )
            for index in range(self._selected_count)
        }
        categories = [
            category
            for category in OFFICIAL_CATEGORY_ORDER
            if category in observed_categories
        ]
        categories.extend(sorted(observed_categories - set(categories)))
        difficulties = [
            value
            for value in ("easy", "medium", "hard")
            if any(
                str(self._rows[index]["difficulty"]) == value
                for index in range(self._selected_count)
            )
        ]
        image_counts = sorted(
            {len(self._rows[index]["images"]) for index in range(self._selected_count)}
        )
        overall = self._bucket(self._tasks, records_by_id)
        metrics: dict[str, Any] = {
            "benchmark": self.name,
            "protocol_version": self.protocol_version,
            "num_source_questions": self._selected_count,
            "num_evaluation_items": len(self._tasks),
            "overall": overall,
            "per_category": self._dimension_metrics(
                "official_category", categories, records_by_id
            ),
            "diagnostics": {
                "note": (
                    "difficulty and image-count breakdowns are additional "
                    "diagnostics, not fields in the official MMSI score table"
                ),
                "per_difficulty": self._dimension_metrics(
                    "difficulty", difficulties, records_by_id
                ),
                "per_image_count": self._dimension_metrics(
                    "images", image_counts, records_by_id
                ),
                "per_image_count_bucket": self._image_bucket_metrics(
                    records_by_id
                ),
            },
        }
        if self.mode == "vanilla":
            metrics["primary_metric"] = "official_vanilla_accuracy"
            metrics["vanilla"] = overall
            metrics["official_score_table"] = [
                {
                    "Category": "Overall",
                    "Vanilla": overall["official_accuracy"],
                },
                *[
                    {
                        "Category": category,
                        "Vanilla": metrics["per_category"][category][
                            "official_accuracy"
                        ],
                    }
                    for category in categories
                ],
            ]
        else:
            source_indices = list(range(self._selected_count))
            vanilla_tasks = [
                self._task_lookup[(source_index, "ABCD")]
                for source_index in source_indices
            ]
            metrics["primary_metric"] = "official_circular_accuracy"
            metrics["expanded_variants"] = overall
            metrics["vanilla"] = self._bucket(vanilla_tasks, records_by_id)
            metrics["circular"] = self._circular_bucket(
                source_indices, records_by_id
            )
            metrics["official_score_table"] = [
                {
                    "Category": "Overall",
                    "Circular": metrics["circular"]["official_accuracy"],
                    "Vanilla": metrics["vanilla"]["official_accuracy"],
                },
                *[
                    {
                        "Category": category,
                        "Circular": metrics["per_category"][category][
                            "circular"
                        ]["official_accuracy"],
                        "Vanilla": metrics["per_category"][category][
                            "vanilla"
                        ]["official_accuracy"],
                    }
                    for category in categories
                ],
            ]
        return metrics

    def summary_lines(self, metrics: Mapping[str, Any]) -> Sequence[str]:
        lines = [
            "=== MMSI-Bench summary ===",
            f"Protocol: {self.mode}",
            f"Complete: {metrics['complete']}",
        ]
        vanilla = metrics["vanilla"]
        lines.append(
            "Official Vanilla accuracy: "
            f"{vanilla['official_accuracy_percent']:.2f}% "
            f"({vanilla['official_correct']}/{vanilla['total']})"
        )
        lines.append(
            "Official parse rate: "
            f"{vanilla['parse_rate_percent']:.2f}% "
            f"({vanilla['parsed']}/{vanilla['total']})"
        )
        if self.mode == "circular":
            circular = metrics["circular"]
            lines.append(
                "Official Circular accuracy (all 4 correct): "
                f"{circular['official_accuracy_percent']:.2f}% "
                f"({circular['official_correct']}/{circular['total']})"
            )
        lines.append("Official category scores:")
        for row in metrics["official_score_table"][1:]:
            if self.mode == "vanilla":
                lines.append(
                    f"  {row['Category']}: Vanilla={100.0 * row['Vanilla']:.2f}%"
                )
            else:
                lines.append(
                    f"  {row['Category']}: "
                    f"Circular={100.0 * row['Circular']:.2f}% "
                    f"Vanilla={100.0 * row['Vanilla']:.2f}%"
                )
        overall = metrics["overall"]
        if overall["missing"] or overall["inference_errors"]:
            lines.append(
                f"Incomplete items: missing={overall['missing']}, "
                f"inference_errors={overall['inference_errors']}"
            )
        return tuple(lines)


__all__ = [
    "MMSI_CIRCULAR_ORDERS",
    "MMSI_POST_PROMPT",
    "MmsiBenchmark",
    "add_mmsi_arguments",
    "build_mmsi_prompt",
    "build_mmsi_question_with_options",
    "extract_mmsi_question_options",
    "parse_mmsi_official_answer",
]
