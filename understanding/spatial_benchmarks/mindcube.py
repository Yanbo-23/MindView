"""MindCube-Tiny adapter for the official ``raw_qa`` protocol.

Official sources are pinned to MindCube commit
``b8b7062adf6d3e49d588a7d014a0a787553d09ec``:

* raw-QA prompt/target:
  https://github.com/mll-lab-nu/MindCube/blob/b8b7062adf6d3e49d588a7d014a0a787553d09ec/src/prompt_generation/templates.py
* answer extractor and setting classifier:
  https://github.com/mll-lab-nu/MindCube/blob/b8b7062adf6d3e49d588a7d014a0a787553d09ec/src/evaluation/core/extractors.py
* basic accuracy and translation filtering:
  https://github.com/mll-lab-nu/MindCube/blob/b8b7062adf6d3e49d588a7d014a0a787553d09ec/src/evaluation/core/base_metrics.py

The adapter intentionally preserves the released extractor's quirks.  In
particular, the final occurrence of ``([A-E])\\.`` anywhere in the response has
the highest priority, even over an ``<answer>`` tag.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .base import BenchmarkAdapter, EvaluationSample, EvaluationTask

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MINDCUBE_DATA_ROOT = REPO_ROOT / "resources" / "understanding" / "MindCube" / "data"

MINDCUBE_CODE_REVISION = "b8b7062adf6d3e49d588a7d014a0a787553d09ec"
MINDCUBE_DATASET_REVISION = "9c941b46a6bd65b6914669ef7a579948fc9c8467"
MINDCUBE_TINY_FILENAME = "MindCube_tinybench.jsonl"
MINDCUBE_TINY_SHA256 = (
    "0289eb82d81ff9aa0201ae75f86da7fd1924cf23dc202856d0579a0effd22ac8"
)
MINDCUBE_TINY_SIZE_BYTES = 929_282
MINDCUBE_TINY_PROMPT_SEQUENCE_SHA256 = (
    "208d2e2531c95ae46db83b429d18a3037baca6106f777c9b1047b3577e53836f"
)
MINDCUBE_PROTOCOL_VERSION = "mindcube_tiny_raw_qa_b8b7062_official_parser_v1"

MINDCUBE_RAW_QA_BACKGROUND = """[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
[Answer Instruction]
You only need to provide *ONE* correct answer selecting from the options listed below. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should **only** be '<answer>A. Above</answer>'.
"""
MINDCUBE_QUESTION_HEADER = "[Question]\n"

OFFICIAL_SETTING_ORDER = ("around", "rotation", "translation", "among", "other")
OFFICIAL_EXCLUDED_OVERALL_SETTINGS = frozenset({"translation"})
OFFICIAL_SETTING_COUNTS = {"among": 600, "around": 250, "rotation": 200}
OFFICIAL_IMAGE_COUNTS = {2: 274, 3: 345, 4: 431}
OFFICIAL_OPTION_COUNTS = {2: 259, 3: 234, 4: 557}
OFFICIAL_ANSWER_COUNTS = {"A": 342, "B": 361, "C": 210, "D": 137}
OFFICIAL_UNIQUE_IMAGE_COUNT = 428
OFFICIAL_IMAGE_FORMAT_COUNTS = {"JPEG": 148, "PNG": 280}

_OFFICIAL_OPTION_SEGMENT_PATTERN = re.compile(r"([A-Z]\.\s+.*?)(?=[A-Z]\.|$)")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_mindcube_raw_qa_prompt(question: str) -> str:
    """Reproduce ``RawQATemplate.generate_prompt`` byte-for-byte."""

    # The background already ends in ``\n`` and the official implementation
    # joins it to the question with another ``\n``.  The blank line is part of
    # the protocol and must not be normalized away.
    return "\n".join([MINDCUBE_RAW_QA_BACKGROUND, MINDCUBE_QUESTION_HEADER + question])


def extract_mindcube_options(question: str) -> tuple[tuple[str, str], ...]:
    """Extract inline choices using the exact regex used for official targets."""

    segments = _OFFICIAL_OPTION_SEGMENT_PATTERN.findall(question)
    options: list[tuple[str, str]] = []
    for raw_segment in segments:
        segment = raw_segment.strip()
        if len(segment) < 4 or segment[1] != ".":
            raise ValueError(f"Malformed MindCube option segment: {segment!r}")
        letter = segment[0]
        text = segment[2:].strip()
        if not text:
            raise ValueError(f"MindCube option {letter} has empty text")
        options.append((letter, text))
    return tuple(options)


def build_mindcube_raw_qa_target(question: str, target_letter: str) -> str:
    """Reproduce ``RawQATemplate.generate_output`` for validation/tests."""

    for letter, text in extract_mindcube_options(question):
        if letter == target_letter:
            return f"<answer>{letter}. {text}</answer>"
    raise ValueError(f"Answer key {target_letter} not found in question")


def clean_mindcube_response(response: str) -> str:
    """Mirror MindCube's Qwen ``ResponseProcessor.clean_response``."""

    if not response:
        return "No response generated"
    cleaned = re.sub(r"\s+", " ", response.strip())
    if len(cleaned) < 1:
        return "Error: Generated response too short"
    return cleaned


def parse_mindcube_official_answer(text: str | None) -> tuple[str | None, str]:
    """Mirror the released ``extract_answer`` regex priority exactly."""

    if not text:
        return None, "empty"

    simple_matches = list(re.finditer(r"([A-E])\.", text))
    if simple_matches:
        return simple_matches[-1].group(1), "simple_period"

    answer_section_match = re.search(r"<Answer>(.*?)(?:<|$)", text, re.DOTALL)
    if answer_section_match:
        answer_section = answer_section_match.group(1)
        answer_section_patterns = [
            r"[Mm]y answer is ([A-E])",
            r"[Mm]y answer is ([A-E])\.",
            r"[Tt]he answer is ([A-E])",
            r"(?:Answer: )?([A-E])\.",
            r"\b([A-E])\b",
        ]
        for index, pattern in enumerate(answer_section_patterns):
            matches = list(re.finditer(pattern, answer_section))
            if matches:
                return matches[-1].group(1), f"answer_section_{index}"

    priority_patterns = [
        r"(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'\",]+(?=(?:\n|$|\.|\"))",
        r"(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'\"]+",
        r"(?:^|\n)(?:Answer: )?([A-E])(?:\.|$|\s)",
        r"[\*\"]([A-E])[\*\"]",
        r"\bAnswer:?\s*([A-E])\b",
        r"[Mm]y answer is ([A-E])",
        r"[Mm]y answer is ([A-E])\.",
        r"answer is ([A-E])",
    ]
    for index, pattern in enumerate(priority_patterns):
        matches = list(re.finditer(pattern, text))
        if matches:
            return matches[-1].group(1), f"priority_{index}"

    lines = text.split("\n")
    line_matches: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = re.search(r"([A-E])\. [A-Za-z0-9 \-\(\)\'\",]+", line)
        if match:
            line_matches.append((index, match.group(1)))
    if line_matches:
        return line_matches[-1][1], "line_full_answer"

    for line in reversed(lines):
        match = re.search(r"\b([A-E])\b", line)
        if match:
            return match.group(1), "final_word"
    return None, "parse_failure"


def get_mindcube_setting(item_id: str) -> str:
    """Mirror official ``get_setting_from_id`` ordering."""

    if not item_id:
        return "other"
    lowered = item_id.lower()
    if "around" in lowered:
        return "around"
    if "rotation" in lowered:
        return "rotation"
    if "translation" in lowered:
        return "translation"
    if "among" in lowered:
        return "among"
    return "other"


def _prompt_sequence_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        encoded = build_mindcube_raw_qa_prompt(str(row["question"])).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _safe_image_path(data_root: Path, raw_path: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute():
        raise ValueError(f"MindCube image path must be relative: {raw_path}")
    path = (data_root / relative).resolve()
    try:
        path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"MindCube image path escapes data root: {raw_path}") from exc
    return path


def load_mindcube_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid MindCube JSON at {jsonl_path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"MindCube row {line_number} must be a JSON object")
            rows.append(row)
    return rows


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    data_root: Path,
    dataset_identity: Mapping[str, Any],
    *,
    allow_nonstandard_data: bool,
    verify_images: bool,
) -> dict[str, Any]:
    required = {
        "id",
        "category",
        "type",
        "meta_info",
        "question",
        "images",
        "gt_answer",
    }
    seen_ids: set[str] = set()
    setting_counts: Counter[str] = Counter()
    image_counts: Counter[int] = Counter()
    option_counts: Counter[int] = Counter()
    answer_counts: Counter[str] = Counter()
    referenced_images: dict[str, Path] = {}

    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"MindCube row {index} is missing fields: {sorted(missing)}"
            )

        source_id = row["id"]
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"MindCube row {index} has an invalid id")
        if source_id in seen_ids:
            raise ValueError(f"MindCube contains duplicate id: {source_id}")
        seen_ids.add(source_id)

        question = row["question"]
        if not isinstance(question, str) or not question:
            raise ValueError(f"MindCube row {index} has an invalid question")
        options = extract_mindcube_options(question)
        labels = [letter for letter, _ in options]
        expected_labels = [chr(65 + offset) for offset in range(len(options))]
        if len(options) < 2 or len(options) > 5 or labels != expected_labels:
            raise ValueError(
                f"MindCube row {index} options must be consecutive A-E choices"
            )
        if len({text for _, text in options}) != len(options):
            raise ValueError(f"MindCube row {index} has duplicate option text")

        target = row["gt_answer"]
        if not isinstance(target, str) or target not in labels:
            raise ValueError(f"MindCube row {index} has invalid gt_answer {target!r}")
        build_mindcube_raw_qa_target(question, target)

        raw_images = row["images"]
        if not isinstance(raw_images, list) or not raw_images:
            raise ValueError(f"MindCube row {index} has no image list")
        if len(set(raw_images)) != len(raw_images):
            raise ValueError(f"MindCube row {index} repeats an image path")
        setting = get_mindcube_setting(source_id)
        for raw_path in raw_images:
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"MindCube row {index} has an invalid image path")
            path = _safe_image_path(data_root, raw_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"MindCube row {index} references a missing image: {path}"
                )
            referenced_images[raw_path] = path
            if not allow_nonstandard_data:
                parts = Path(raw_path).parts
                if len(parts) < 3 or parts[:2] != ("other_all_image", setting):
                    raise ValueError(
                        f"MindCube row {index} image folder disagrees with id setting"
                    )

        if not allow_nonstandard_data:
            category = row["category"]
            if not isinstance(category, list) or len(category) != 4:
                raise ValueError(
                    f"Official MindCube row {index} must have four category fields"
                )

        setting_counts[setting] += 1
        image_counts[len(raw_images)] += 1
        option_counts[len(options)] += 1
        answer_counts[target] += 1

    image_format_counts: Counter[str] = Counter()
    image_mode_counts: Counter[str] = Counter()
    if verify_images:
        for relative_path, path in sorted(referenced_images.items()):
            try:
                with Image.open(path) as image:
                    image_format_counts[str(image.format)] += 1
                    image_mode_counts[str(image.mode)] += 1
                    image.verify()
            except Exception as exc:
                raise ValueError(
                    f"MindCube image cannot be decoded: {relative_path}"
                ) from exc

    if not allow_nonstandard_data:
        canonical_checks = (
            (len(rows), 1050, "row count"),
            (len(seen_ids), 1050, "unique id count"),
            (dict(setting_counts), OFFICIAL_SETTING_COUNTS, "setting counts"),
            (dict(image_counts), OFFICIAL_IMAGE_COUNTS, "image counts"),
            (dict(option_counts), OFFICIAL_OPTION_COUNTS, "option counts"),
            (dict(answer_counts), OFFICIAL_ANSWER_COUNTS, "answer counts"),
            (
                len(referenced_images),
                OFFICIAL_UNIQUE_IMAGE_COUNT,
                "unique referenced image count",
            ),
            (
                dataset_identity.get("sha256"),
                MINDCUBE_TINY_SHA256,
                "JSONL SHA256",
            ),
            (
                dataset_identity.get("size_bytes"),
                MINDCUBE_TINY_SIZE_BYTES,
                "JSONL size",
            ),
            (
                _prompt_sequence_sha256(rows),
                MINDCUBE_TINY_PROMPT_SEQUENCE_SHA256,
                "raw-QA prompt sequence SHA256",
            ),
        )
        for observed, expected, label in canonical_checks:
            if observed != expected:
                raise ValueError(
                    f"Official MindCube-Tiny {label} differs: "
                    f"expected {expected}, got {observed}"
                )
        if verify_images:
            if dict(image_format_counts) != OFFICIAL_IMAGE_FORMAT_COUNTS:
                raise ValueError(
                    "Official MindCube-Tiny image formats differ: "
                    f"{dict(image_format_counts)}"
                )
            if dict(image_mode_counts) != {"RGB": OFFICIAL_UNIQUE_IMAGE_COUNT}:
                raise ValueError(
                    "Official MindCube-Tiny images are expected to be RGB: "
                    f"{dict(image_mode_counts)}"
                )

    return {
        "rows": len(rows),
        "unique_ids": len(seen_ids),
        "setting_counts": dict(setting_counts),
        "image_count_distribution": {
            str(k): v for k, v in sorted(image_counts.items())
        },
        "option_count_distribution": {
            str(k): v for k, v in sorted(option_counts.items())
        },
        "answer_distribution": dict(sorted(answer_counts.items())),
        "image_references": sum(image_counts[count] * count for count in image_counts),
        "unique_referenced_images": len(referenced_images),
        "image_format_counts": dict(sorted(image_format_counts.items())),
        "image_mode_counts": dict(sorted(image_mode_counts.items())),
    }


def add_mindcube_arguments(parser: Any) -> None:
    group = parser.add_argument_group("MindCube-Tiny data")
    group.add_argument(
        "--mindcube-data-root",
        default=str(DEFAULT_MINDCUBE_DATA_ROOT),
        help=(
            "Extracted MindCube data/ directory containing raw/ and "
            "other_all_image/. The adapter always evaluates "
            "raw/MindCube_tinybench.jsonl."
        ),
    )
    group.add_argument(
        "--allow-nonstandard-data",
        action="store_true",
        help="Disable the pinned official TinyBench hash/count invariants.",
    )


def _resolve_mindcube_data(
    args: Any, runtime: Any
) -> tuple[Path, Path, dict[str, Any]]:
    def resolve_on_primary() -> dict[str, Any]:
        data_root = Path(args.mindcube_data_root).expanduser().resolve()
        if not data_root.is_dir():
            raise FileNotFoundError(f"MindCube data root does not exist: {data_root}")
        jsonl_path = data_root / "raw" / MINDCUBE_TINY_FILENAME
        if not jsonl_path.is_file():
            raise FileNotFoundError(f"MindCube-Tiny JSONL does not exist: {jsonl_path}")
        stat = jsonl_path.stat()
        identity: dict[str, Any] = {
            "source": "local_extracted_official_archive",
            "data_root": str(data_root),
            "jsonl_path": str(jsonl_path),
            "filename": MINDCUBE_TINY_FILENAME,
            "size_bytes": stat.st_size,
            "sha256": sha256_file(jsonl_path),
            "expected_official_sha256": MINDCUBE_TINY_SHA256,
            "official_dataset_revision": MINDCUBE_DATASET_REVISION,
        }
        rows = load_mindcube_rows(jsonl_path)
        identity["validation"] = _validate_rows(
            rows,
            data_root,
            identity,
            allow_nonstandard_data=args.allow_nonstandard_data,
            verify_images=True,
        )
        return identity

    identity = runtime.primary_call(
        resolve_on_primary, label="resolve and validate MindCube-Tiny data"
    )
    return (
        Path(identity["data_root"]),
        Path(identity["jsonl_path"]),
        identity,
    )


def _decode_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def _percent(correct: int, total: int) -> float:
    return 100.0 * correct / total if total else 0.0


class MindCubeAdapter(BenchmarkAdapter):
    """Evaluate one model call per MindCube-Tiny row with official raw QA."""

    name = "mindcube-tiny"
    protocol_version = MINDCUBE_PROTOCOL_VERSION

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        data_root: Path,
        dataset_identity: Mapping[str, Any],
        *,
        limit: int | None = None,
        allow_nonstandard_data: bool = False,
    ) -> None:
        self._rows = [dict(row) for row in rows]
        self._data_root = data_root.expanduser().resolve()
        self._dataset_identity = dict(dataset_identity)
        self._validation = _validate_rows(
            self._rows,
            self._data_root,
            self._dataset_identity,
            allow_nonstandard_data=allow_nonstandard_data,
            verify_images=False,
        )
        if isinstance(self._dataset_identity.get("validation"), Mapping):
            # ``from_args`` performs the expensive Pillow verification once on
            # rank 0 and broadcasts this complete report to every rank.
            self._validation = dict(self._dataset_identity["validation"])

        if limit is not None:
            if limit <= 0:
                raise ValueError(f"--limit must be positive, got {limit}")
            selected_count = min(limit, len(self._rows))
        else:
            selected_count = len(self._rows)
        self._selected_count = selected_count

        tasks: list[EvaluationTask] = []
        for source_index, row in enumerate(self._rows[:selected_count]):
            source_id = str(row["id"])
            tasks.append(
                EvaluationTask(
                    global_index=source_index,
                    eval_id=f"mindcube_tiny:{source_id}:raw_qa",
                    source_index=source_index,
                    source_id=source_id,
                    category=get_mindcube_setting(source_id),
                    variant="raw_qa",
                )
            )
        self._tasks = tuple(tasks)

    @classmethod
    def from_args(cls, args: Any, runtime: Any) -> "MindCubeAdapter":
        data_root, jsonl_path, identity = _resolve_mindcube_data(args, runtime)
        rows = load_mindcube_rows(jsonl_path)
        return cls(
            rows,
            data_root,
            identity,
            limit=args.limit,
            allow_nonstandard_data=args.allow_nonstandard_data,
        )

    @property
    def tasks(self) -> Sequence[EvaluationTask]:
        return self._tasks

    @property
    def required_ok_boolean_fields(self) -> Sequence[str]:
        return ("parse_success", "official_correct")

    @property
    def manifest_fragment(self) -> Mapping[str, Any]:
        selected_rows = self._rows[: self._selected_count]
        return {
            "benchmark": self.name,
            "protocol_version": self.protocol_version,
            "official_code_revision": MINDCUBE_CODE_REVISION,
            "dataset": self._dataset_identity,
            "raw_rows": len(self._rows),
            "selected_source_rows": self._selected_count,
            "evaluation_items": len(self._tasks),
            "task_variant": "raw_qa",
            "prompt_construction": (
                "RAW_QA_BACKGROUND_INSTRUCTION + newline + [Question] + raw question"
            ),
            "selected_prompt_sequence_sha256": _prompt_sequence_sha256(selected_rows),
            "answer_format": "<answer>LETTER. option text</answer>",
            "answer_extractor": "official case-sensitive regex priority",
            "setting_source": "official substring classifier over item id",
            "official_overall_excludes": sorted(OFFICIAL_EXCLUDED_OVERALL_SETTINGS),
            "primary_aggregation": "official filtered micro accuracy",
            "validation": self._validation,
        }

    def _canonical_fields(self, task: EvaluationTask) -> dict[str, Any]:
        row = self._rows[task.source_index]
        question = str(row["question"])
        labeled_options = extract_mindcube_options(question)
        option_by_letter = dict(labeled_options)
        target_letter = str(row["gt_answer"])
        return {
            "question": question,
            "question_type": row["type"],
            "official_index": task.source_index,
            "official_group_index": task.source_index,
            "prompt": build_mindcube_raw_qa_prompt(question),
            "options": [text for _, text in labeled_options],
            "target_letter": target_letter,
            "target_text": option_by_letter[target_letter],
            "num_images": len(row["images"]),
        }

    def materialize(self, task: EvaluationTask) -> EvaluationSample:
        if task.variant != "raw_qa":
            raise ValueError(f"Unknown MindCube variant: {task.variant}")
        row = self._rows[task.source_index]
        fields = self._canonical_fields(task)
        images: list[Image.Image] = []
        try:
            for raw_path in row["images"]:
                images.append(
                    _decode_image(_safe_image_path(self._data_root, raw_path))
                )
        except Exception:
            for image in images:
                image.close()
            raise
        return EvaluationSample(
            task=task,
            images=images,
            prompt=fields["prompt"],
            target_letter=fields["target_letter"],
            target_text=fields["target_text"],
            options=tuple(fields["options"]),
            metadata={
                "question": fields["question"],
                "question_type": fields["question_type"],
                "official_index": fields["official_index"],
                "official_group_index": fields["official_group_index"],
                "official_target_output": build_mindcube_raw_qa_target(
                    fields["question"], fields["target_letter"]
                ),
            },
        )

    def score_response(
        self, sample: EvaluationSample, response: str
    ) -> Mapping[str, Any]:
        official_response = clean_mindcube_response(response)
        prediction, parse_method = parse_mindcube_official_answer(official_response)
        prediction_text = None
        if prediction is not None:
            index = ord(prediction) - ord("A")
            if 0 <= index < len(sample.options):
                prediction_text = sample.options[index]
        return {
            "prediction": prediction,
            "prediction_text": prediction_text,
            "official_response": official_response,
            "parse_method": parse_method,
            "parse_success": prediction is not None,
            "official_correct": prediction == sample.target_letter,
        }

    def validate_record(
        self, task: EvaluationTask, record: Mapping[str, Any]
    ) -> Sequence[str]:
        if record.get("status") != "ok":
            return ()
        errors: list[str] = []
        expected_fields = self._canonical_fields(task)
        for key, expected in expected_fields.items():
            if record.get(key) != expected:
                errors.append(f"{key} does not match the canonical MindCube task")

        response = record.get("raw_response")
        if isinstance(response, str):
            sample = EvaluationSample(
                task=task,
                images=[],
                prompt=expected_fields["prompt"],
                target_letter=expected_fields["target_letter"],
                target_text=expected_fields["target_text"],
                options=tuple(expected_fields["options"]),
            )
            for key, expected in self.score_response(sample, response).items():
                if record.get(key) != expected:
                    errors.append(f"{key} does not match rescoring raw_response")
        return tuple(errors)

    @staticmethod
    def _bucket(
        tasks: Sequence[EvaluationTask],
        records_by_id: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        records = [records_by_id.get(task.eval_id) for task in tasks]
        total = len(tasks)
        completed = sum(
            record is not None and record.get("status") == "ok" for record in records
        )
        correct = sum(
            bool(record and record.get("official_correct")) for record in records
        )
        parsed = sum(bool(record and record.get("parse_success")) for record in records)
        missing = sum(record is None for record in records)
        inference_errors = sum(
            bool(record and record.get("status") == "inference_error")
            for record in records
        )
        distribution: Counter[str] = Counter()
        for record in records:
            if record is None or record.get("status") != "ok":
                continue
            prediction = record.get("prediction")
            distribution[str(prediction) if prediction is not None else "unparsed"] += 1
        return {
            "total": total,
            "completed_responses": completed,
            "missing": missing,
            "inference_errors": inference_errors,
            "official_correct": correct,
            "official_accuracy": correct / total if total else 0.0,
            "official_accuracy_percent": _percent(correct, total),
            "parsed": parsed,
            "parse_rate": parsed / total if total else 0.0,
            "parse_rate_percent": _percent(parsed, total),
            "prediction_distribution": dict(sorted(distribution.items())),
        }

    def aggregate(
        self, records_by_id: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        all_items = self._bucket(self._tasks, records_by_id)
        official_tasks = [
            task
            for task in self._tasks
            if task.category not in OFFICIAL_EXCLUDED_OVERALL_SETTINGS
        ]
        official_items = self._bucket(official_tasks, records_by_id)
        overall = {
            **all_items,
            "official_total": official_items["total"],
            "official_correct": official_items["official_correct"],
            "official_accuracy": official_items["official_accuracy"],
            "official_accuracy_percent": official_items["official_accuracy_percent"],
        }

        per_setting: dict[str, Any] = {}
        for setting in OFFICIAL_SETTING_ORDER:
            setting_tasks = [task for task in self._tasks if task.category == setting]
            values = self._bucket(setting_tasks, records_by_id)
            values["include_in_official_overall"] = (
                setting not in OFFICIAL_EXCLUDED_OVERALL_SETTINGS
            )
            per_setting[setting] = values

        return {
            "benchmark": self.name,
            "protocol_version": self.protocol_version,
            "primary_metric": "official_accuracy",
            "num_source_questions": self._selected_count,
            "num_evaluation_items": len(self._tasks),
            "overall": overall,
            "per_setting": per_setting,
        }

    def summary_lines(self, metrics: Mapping[str, Any]) -> Sequence[str]:
        overall = metrics["overall"]
        lines = [
            "=== MindCube-Tiny raw-QA summary ===",
            f"Complete: {metrics['complete']}",
            (
                "Official accuracy: "
                f"{overall['official_accuracy_percent']:.2f}% "
                f"({overall['official_correct']}/{overall['official_total']})"
            ),
            (
                "Official extraction rate: "
                f"{overall['parse_rate_percent']:.2f}% "
                f"({overall['parsed']}/{overall['total']})"
            ),
        ]
        for setting in OFFICIAL_SETTING_ORDER:
            values = metrics["per_setting"][setting]
            if values["total"] == 0:
                continue
            lines.append(
                f"  {setting.capitalize()}: "
                f"{values['official_accuracy_percent']:.2f}% "
                f"({values['official_correct']}/{values['total']})"
            )
        if overall["missing"] or overall["inference_errors"]:
            lines.append(
                f"Incomplete items: missing={overall['missing']}, "
                f"inference_errors={overall['inference_errors']}"
            )
        return tuple(lines)


__all__ = [
    "MindCubeAdapter",
    "add_mindcube_arguments",
    "build_mindcube_raw_qa_prompt",
    "build_mindcube_raw_qa_target",
    "clean_mindcube_response",
    "extract_mindcube_options",
    "get_mindcube_setting",
    "parse_mindcube_official_answer",
]
