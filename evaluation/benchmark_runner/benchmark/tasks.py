#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Type

from .data_paths import image_path, validate_shot
from .io_utils import load_rgb_images


@dataclass
class InferenceRequest:
    sample: Dict[str, Any]
    prompt: str
    image_paths: List[str]
    images: List[Any]
    pred_path: str
    gt_src_path: str
    gt_save_path: str
    context_save_dir: str
    response_dir: str


class BaseTask:
    """Task adapter. Add new task behavior by subclassing and registering it."""

    name = "default"

    def __init__(self, args):
        self.args = args

    def image_path(self, dataset: str, scene_id: str, frame_id: str) -> str:
        return image_path(self.args, dataset, scene_id, frame_id)

    def validate(self, sample: Dict[str, Any]) -> None:
        required = [
            "sample_id",
            "dataset",
            "task_type",
            "scene_id",
            "gt_frame_id",
            "shot",
            "contexts_frame_id",
            "instruction",
        ]
        missing = [k for k in required if k not in sample]
        if missing:
            raise KeyError(f"Missing fields {missing} in sample {sample.get('sample_id', '<unknown>')}")
        if not sample["contexts_frame_id"]:
            raise ValueError(f"Empty contexts_frame_id in {sample['sample_id']}")
        shot = validate_shot(sample["shot"])
        context_count = len(sample["contexts_frame_id"])
        variable_contexts_allowed = (
            bool(getattr(self.args, "allow_variable_contexts", False))
            and shot == 10
            and 6 <= context_count <= 10
        )
        if context_count != shot and not variable_contexts_allowed:
            raise ValueError(
                f"shot={shot}, but contexts_frame_id has "
                f"{context_count} entries in {sample['sample_id']}"
            )

    def build_prompt(self, sample: Dict[str, Any]) -> str:
        # Your current json already stores the full prompt/instruction.
        return sample["instruction"]

    def build_request(self, sample: Dict[str, Any], run_dir: Path) -> InferenceRequest:
        self.validate(sample)
        task_type = sample["task_type"]
        dataset = sample["dataset"]
        sample_id = sample["sample_id"]
        scene_id = sample["scene_id"]

        image_paths = [
            self.image_path(dataset, scene_id, str(fid))
            for fid in sample["contexts_frame_id"]
        ]
        gt_src = self.image_path(dataset, scene_id, str(sample["gt_frame_id"]))

        shot_dir = f"shot-{sample['shot']}"
        pred_path = run_dir / shot_dir / "pred" / dataset / task_type / f"{sample_id}.png"
        gt_save_path = run_dir / shot_dir / "gt" / dataset / task_type / f"{sample_id}.jpg"
        context_save_dir = run_dir / shot_dir / "contexts" / dataset / task_type
        response_dir = run_dir / shot_dir / "responses" / dataset / task_type

        for p in image_paths:
            if not Path(p).exists():
                raise FileNotFoundError(p)

        images = load_rgb_images(image_paths, self.args.input_image_size)
        return InferenceRequest(
            sample=sample,
            prompt=self.build_prompt(sample),
            image_paths=image_paths,
            images=images,
            pred_path=str(pred_path),
            gt_src_path=gt_src,
            gt_save_path=str(gt_save_path),
            context_save_dir=str(context_save_dir),
            response_dir=str(response_dir),
        )


TASK_REGISTRY: Dict[str, Type[BaseTask]] = {"default": BaseTask}


def register_task(name: str):
    def deco(cls: Type[BaseTask]):
        TASK_REGISTRY[name] = cls
        return cls
    return deco


def get_task(name: str, args) -> BaseTask:
    return TASK_REGISTRY.get(name, TASK_REGISTRY["default"])(args)
