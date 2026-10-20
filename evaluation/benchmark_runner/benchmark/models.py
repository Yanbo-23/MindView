#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import numpy as np
from PIL import Image, ImageDraw

from .tasks import InferenceRequest


class BaseModel(ABC):
    name = "base"

    def __init__(self, args):
        self.args = args
        self.device = args.device

    def setup(self) -> None:
        pass

    @abstractmethod
    def infer(self, req: InferenceRequest) -> Tuple[Image.Image, Dict[str, Any]]:
        """Return generated PIL image and optional extra metadata."""


MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str):
    def deco(cls: Type[BaseModel]):
        MODEL_REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def build_model(name: str, args) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name](args)
    model.setup()
    return model


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_response(req: InferenceRequest, obj: Any, name: str) -> str:
    Path(req.response_dir).mkdir(parents=True, exist_ok=True)
    path = Path(req.response_dir) / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(path)


def derive_seed(base_seed: int, sample: Dict[str, Any]) -> int:
    prefix, suffix = str(sample["sample_id"]).rsplit("_", 1)
    shot6_index = int(suffix) + {3: 1, 6: 0, 10: -1}[int(sample["shot"])]
    shot6_id = f"{prefix}_{shot6_index:0{len(suffix)}d}"
    return int(base_seed) + int.from_bytes(hashlib.sha256(shot6_id.encode()).digest()[:4], "big")


@register_model("dummy")
class DummyModel(BaseModel):
    """Debug model: copies the anchor image and writes a label."""

    def infer(self, req: InferenceRequest) -> Tuple[Image.Image, Dict[str, Any]]:
        img = req.images[0].copy()
        draw = ImageDraw.Draw(img)
        draw.text((8, 8), f"DUMMY: {req.sample['sample_id']}", fill=(255, 255, 255))
        return img, {"debug": True}


@register_model("flux2")
class Flux2Model(BaseModel):
    def setup(self) -> None:
        import torch
        from diffusers import Flux2KleinPipeline

        dtype = getattr(torch, self.args.dtype)
        self.torch = torch
        self.pipe = Flux2KleinPipeline.from_pretrained(self.args.model_path, torch_dtype=dtype)
        if self.args.cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

    def infer(self, req: InferenceRequest) -> Tuple[Image.Image, Dict[str, Any]]:
        seed = derive_seed(self.args.seed, req.sample)
        gen_device = self.args.generator_device or self.device
        generator = self.torch.Generator(device=gen_device).manual_seed(seed)
        out = self.pipe(
            image=req.images,
            prompt=req.prompt,
            height=self.args.height,
            width=self.args.width,
            guidance_scale=self.args.guidance_scale,
            num_inference_steps=self.args.num_inference_steps,
            generator=generator,
        ).images[0]
        return out, {"seed": seed}


@register_model("qwen-image-edit-2511")
class QwenImageEdit2511Model(BaseModel):
    """Qwen-Image-Edit-2511 with benchmark-specific anchor/contact-sheet inputs."""

    MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
    INPUT_SIZE = 512
    OUTPUT_SIZE = 512
    TRUE_CFG_SCALE = 4.0
    GUIDANCE_SCALE = 1.0
    NEGATIVE_PROMPT = " "

    def setup(self) -> None:
        import torch
        from diffusers import QwenImageEditPlusPipeline

        self.torch = torch
        model_path = self.args.model_path or self.MODEL_ID
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=None)

    @staticmethod
    def _frame_key(frame_id: Any) -> str:
        name = Path(str(frame_id)).name
        suffix = Path(name).suffix.lower()
        return Path(name).stem if suffix in {".jpg", ".jpeg", ".png", ".webp"} else name

    @classmethod
    def _to_input_size(cls, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize((cls.INPUT_SIZE, cls.INPUT_SIZE), getattr(Image, "Resampling", Image).LANCZOS)

    @classmethod
    def _make_contact_sheet(cls, images: List[Image.Image]) -> Tuple[Image.Image, int, int]:
        if not images:
            raise ValueError("Qwen-Image-Edit-2511 requires at least one supporting context.")
        columns = math.ceil(math.sqrt(len(images)))
        rows = math.ceil(len(images) / columns)
        tile_size = cls.INPUT_SIZE // max(columns, rows)
        canvas = Image.new("RGB", (cls.INPUT_SIZE, cls.INPUT_SIZE), color=(0, 0, 0))
        offset_x = (cls.INPUT_SIZE - columns * tile_size) // 2
        offset_y = (cls.INPUT_SIZE - rows * tile_size) // 2
        for index, image in enumerate(images):
            row, column = divmod(index, columns)
            tile = image.resize((tile_size, tile_size), getattr(Image, "Resampling", Image).LANCZOS)
            canvas.paste(tile, (offset_x + column * tile_size, offset_y + row * tile_size))
        return canvas, columns, rows

    @staticmethod
    def _adapt_prompt(prompt: str, has_anchor: bool, panel_count: int, columns: int, rows: int) -> str:
        if has_anchor:
            input_views = f"""INPUT VIEWS:
Image 1 is the complete anchor view and defines the starting camera pose.
Image 2 is a black-padded contact sheet containing {panel_count} supporting views of the same scene. Read its non-black panels in row-major order: left-to-right, then top-to-bottom, using a {columns}-column by {rows}-row grid. Treat every non-black panel as an independent camera view. Black regions are padding only and must be ignored; do not combine the panels into one physical scene.

"""
        else:
            input_views = f"""INPUT VIEWS:
Image 1 is a black-padded contact sheet containing {panel_count} supporting views of the same scene; there is no anchor or target view. Read its non-black panels in row-major order: left-to-right, then top-to-bottom, using a {columns}-column by {rows}-row grid. Treat every non-black panel as an independent camera view. Black regions are padding only and must be ignored; do not combine the panels into one physical scene.

"""
        start = prompt.find("INPUT VIEWS:")
        end = prompt.find("SCENE CONSISTENCY:")
        if start >= 0 and end > start:
            return prompt[:start] + input_views + prompt[end:]
        return prompt.rstrip() + "\n\n" + input_views

    def _prepare_inputs(self, req: InferenceRequest) -> Tuple[List[Image.Image], str, Dict[str, Any]]:
        sample = req.sample
        context_ids = list(sample["contexts_frame_id"])
        if len(context_ids) != len(req.image_paths):
            raise ValueError(
                f"Context/path count mismatch for {sample['sample_id']}: "
                f"{len(context_ids)} ids vs {len(req.image_paths)} paths"
            )
        # Start from the source files so Qwen's fixed 512px preprocessing is
        # independent from the generic runner's --input_image_size setting.
        contexts: List[Image.Image] = []
        for image_path in req.image_paths:
            with Image.open(image_path) as image:
                contexts.append(self._to_input_size(image))
        anchor_id = sample.get("anchor_frame_id")
        has_anchor = anchor_id is not None and str(anchor_id).strip() not in {"", "None"}

        if has_anchor:
            anchor_key = self._frame_key(anchor_id)
            anchor_index = next(
                (index for index, frame_id in enumerate(context_ids) if self._frame_key(frame_id) == anchor_key),
                None,
            )
            if anchor_index is None:
                raise ValueError(f"Anchor {anchor_id!r} is absent from contexts for {sample['sample_id']}")
            anchor = contexts[anchor_index]
            supporting = [image for index, image in enumerate(contexts) if index != anchor_index]
            supporting_ids = [frame_id for index, frame_id in enumerate(context_ids) if index != anchor_index]
        else:
            anchor = None
            supporting = contexts
            supporting_ids = context_ids

        mosaic, columns, rows = self._make_contact_sheet(supporting)
        model_images = [anchor, mosaic] if anchor is not None else [mosaic]
        prompt = self._adapt_prompt(req.prompt, anchor is not None, len(supporting), columns, rows)
        metadata = {
            "input_mode": "anchor_plus_mosaic" if anchor is not None else "mosaic_only",
            "input_size": self.INPUT_SIZE,
            "anchor_frame_id": str(anchor_id) if anchor is not None else None,
            "supporting_frame_ids": [str(frame_id) for frame_id in supporting_ids],
            "mosaic_grid": {"columns": columns, "rows": rows, "panel_count": len(supporting)},
            "model_input_count": len(model_images),
        }
        return model_images, prompt, metadata

    def _save_input_artifacts(
        self,
        req: InferenceRequest,
        model_images: List[Image.Image],
        prompt: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, str]:
        artifact_dir = Path(req.response_dir) / f"{req.sample['sample_id']}_qwen_inputs"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}
        if metadata["input_mode"] == "anchor_plus_mosaic":
            anchor_path = artifact_dir / "anchor.png"
            model_images[0].save(anchor_path)
            paths["anchor"] = str(anchor_path)
            mosaic_image = model_images[1]
        else:
            mosaic_image = model_images[0]
        mosaic_path = artifact_dir / "contact_sheet.png"
        mosaic_image.save(mosaic_path)
        paths["contact_sheet"] = str(mosaic_path)
        prompt_path = artifact_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        paths["prompt"] = str(prompt_path)
        metadata_path = artifact_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["metadata"] = str(metadata_path)
        return paths

    def infer(self, req: InferenceRequest) -> Tuple[Image.Image, Dict[str, Any]]:
        seed = derive_seed(self.args.seed, req.sample)
        generator_device = self.args.generator_device or self.device
        generator = self.torch.Generator(device=generator_device).manual_seed(seed)
        model_images, prompt, metadata = self._prepare_inputs(req)
        artifact_paths = self._save_input_artifacts(req, model_images, prompt, metadata)
        with self.torch.inference_mode():
            output = self.pipe(
                image=model_images,
                prompt=prompt,
                generator=generator,
                true_cfg_scale=self.TRUE_CFG_SCALE,
                negative_prompt=self.NEGATIVE_PROMPT,
                num_inference_steps=self.args.qwen_num_inference_steps,
                guidance_scale=self.GUIDANCE_SCALE,
                num_images_per_prompt=1,
                width=self.OUTPUT_SIZE,
                height=self.OUTPUT_SIZE,
            ).images[0]
        return output.convert("RGB"), {
            "seed": seed,
            "model_id": self.args.model_path or self.MODEL_ID,
            "torch_dtype": "bfloat16",
            "true_cfg_scale": self.TRUE_CFG_SCALE,
            "negative_prompt": self.NEGATIVE_PROMPT,
            "guidance_scale": self.GUIDANCE_SCALE,
            "num_inference_steps": self.args.qwen_num_inference_steps,
            "requested_output_size": [self.OUTPUT_SIZE, self.OUTPUT_SIZE],
            "generated_size": list(output.size),
            "qwen_input_artifacts": artifact_paths,
            **metadata,
        }


@register_model("emu3.5")
class Emu35Model(BaseModel):
    """Emu3.5 with the same anchor/contact-sheet protocol as Qwen Image."""

    INPUT_SIZE = 512
    OUTPUT_SIZE = 512

    def setup(self) -> None:
        if not self.args.repo_path:
            raise ValueError("Emu3.5 requires --repo_path")

        repo_path = Path(self.args.repo_path).resolve()
        sys.path.insert(0, str(repo_path))

        import torch
        from configs import test_x2i_common as cfg
        from inference import inference as official_inference
        from src.utils.model_utils import build_emu3p5
        from src.utils.vis_proto import parse_story, write_story_assets

        if not torch.cuda.is_available():
            raise RuntimeError("Emu3.5 requires a CUDA GPU")

        model_path = Path(self.args.model_path or cfg.model_path).resolve()
        vq_path = Path(self.args.emu_vq_path or cfg.vq_path).resolve()
        tokenizer_path = Path(
            self.args.emu_tokenizer_path or repo_path / "src" / "tokenizer_emu3_ibq"
        ).resolve()
        for label, path in (
            ("model", model_path),
            ("vision tokenizer", vq_path),
            ("text tokenizer", tokenizer_path),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Emu3.5 {label} path not found: {path}")

        # Each benchmark worker sees exactly one physical GPU, remapped to cuda:0.
        cfg.model_path = str(model_path)
        cfg.vq_path = str(vq_path)
        cfg.tokenizer_path = str(tokenizer_path)
        cfg.hf_device = "auto"
        cfg.vq_device = "cuda:0"
        cfg.rank = 0
        cfg.world_size = 1

        self.torch = torch
        self.cfg = cfg
        self.official_inference = official_inference
        self.parse_story = parse_story
        self.write_story_assets = write_story_assets
        self.model, self.tokenizer, self.vq_model = build_emu3p5(
            cfg.model_path,
            cfg.tokenizer_path,
            cfg.vq_path,
            vq_type=cfg.vq_type,
            model_device=cfg.hf_device,
            vq_device=cfg.vq_device,
        )
        cfg.special_token_ids = {
            name: self.tokenizer.encode(token)[0]
            for name, token in cfg.special_tokens.items()
        }

    def _prepare_inputs(
        self, req: InferenceRequest
    ) -> Tuple[List[Image.Image], str, Dict[str, Any]]:
        """Match QwenImageEdit2511Model._prepare_inputs without changing Qwen."""
        sample = req.sample
        context_ids = list(sample["contexts_frame_id"])
        if len(context_ids) != len(req.image_paths):
            raise ValueError(
                f"Context/path count mismatch for {sample['sample_id']}: "
                f"{len(context_ids)} ids vs {len(req.image_paths)} paths"
            )

        contexts: List[Image.Image] = []
        for image_path in req.image_paths:
            with Image.open(image_path) as image:
                contexts.append(QwenImageEdit2511Model._to_input_size(image))

        anchor_id = sample.get("anchor_frame_id")
        has_anchor = anchor_id is not None and str(anchor_id).strip() not in {"", "None"}
        if has_anchor:
            anchor_key = QwenImageEdit2511Model._frame_key(anchor_id)
            anchor_index = next(
                (
                    index
                    for index, frame_id in enumerate(context_ids)
                    if QwenImageEdit2511Model._frame_key(frame_id) == anchor_key
                ),
                None,
            )
            if anchor_index is None:
                raise ValueError(
                    f"Anchor {anchor_id!r} is absent from contexts for {sample['sample_id']}"
                )
            anchor = contexts[anchor_index]
            supporting = [
                image for index, image in enumerate(contexts) if index != anchor_index
            ]
            supporting_ids = [
                frame_id
                for index, frame_id in enumerate(context_ids)
                if index != anchor_index
            ]
        else:
            anchor = None
            supporting = contexts
            supporting_ids = context_ids

        mosaic, columns, rows = QwenImageEdit2511Model._make_contact_sheet(supporting)
        model_images = [anchor, mosaic] if anchor is not None else [mosaic]
        prompt = QwenImageEdit2511Model._adapt_prompt(
            req.prompt, anchor is not None, len(supporting), columns, rows
        )
        metadata = {
            "input_mode": "anchor_plus_mosaic" if anchor is not None else "mosaic_only",
            "input_size": self.INPUT_SIZE,
            "anchor_frame_id": str(anchor_id) if anchor is not None else None,
            "supporting_frame_ids": [str(frame_id) for frame_id in supporting_ids],
            "mosaic_grid": {
                "columns": columns,
                "rows": rows,
                "panel_count": len(supporting),
            },
            "model_input_count": len(model_images),
        }
        return model_images, prompt, metadata

    def _save_input_artifacts(
        self,
        req: InferenceRequest,
        model_images: List[Image.Image],
        prompt: str,
        metadata: Dict[str, Any],
    ) -> Tuple[Path, List[str], Dict[str, str]]:
        artifact_dir = Path(req.response_dir) / f"{req.sample['sample_id']}_emu3p5_inputs"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}
        model_input_paths: List[str] = []
        if metadata["input_mode"] == "anchor_plus_mosaic":
            anchor_path = artifact_dir / "anchor.png"
            model_images[0].save(anchor_path)
            paths["anchor"] = str(anchor_path)
            model_input_paths.append(str(anchor_path))
            mosaic_image = model_images[1]
        else:
            mosaic_image = model_images[0]
        mosaic_path = artifact_dir / "contact_sheet.png"
        mosaic_image.save(mosaic_path)
        paths["contact_sheet"] = str(mosaic_path)
        model_input_paths.append(str(mosaic_path))
        prompt_path = artifact_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        paths["prompt"] = str(prompt_path)
        metadata_path = artifact_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths["metadata"] = str(metadata_path)
        return artifact_dir, model_input_paths, paths

    def infer(self, req: InferenceRequest) -> Tuple[Image.Image, Dict[str, Any]]:
        seed = derive_seed(self.args.seed, req.sample)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        self.torch.backends.cudnn.deterministic = True
        self.torch.backends.cudnn.benchmark = False
        self.torch.cuda.reset_peak_memory_stats()

        model_images, prompt, metadata = self._prepare_inputs(req)
        artifact_dir, model_input_paths, artifact_paths = self._save_input_artifacts(
            req, model_images, prompt, metadata
        )
        official_dir = artifact_dir / "official_output"
        self.cfg.seed = seed
        self.cfg.save_path = str(official_dir)
        self.cfg.prompts = [
            (
                "result",
                {"prompt": prompt, "reference_image": model_input_paths},
            )
        ]
        self.official_inference(
            self.cfg,
            self.model,
            self.tokenizer,
            self.vq_model,
        )

        proto_path = official_dir / "proto" / "result.pb"
        if not proto_path.is_file():
            raise RuntimeError(f"Emu3.5 did not produce {proto_path}")
        result_dir = official_dir / "result"
        self.write_story_assets(
            self.parse_story(str(proto_path)), str(result_dir), False, 1
        )
        generated = sorted(
            path
            for path in result_dir.glob("*_image.png")
            if "reference" not in path.name
        )
        if not generated:
            raise RuntimeError(f"No generated Emu3.5 image found in {result_dir}")
        with Image.open(generated[0]) as source:
            output = source.convert("RGB")

        extra = {
            "seed": seed,
            "model_path": self.cfg.model_path,
            "vq_path": self.cfg.vq_path,
            "torch_dtype": "bfloat16",
            "classifier_free_guidance": self.cfg.classifier_free_guidance,
            "sampling_params": dict(self.cfg.sampling_params),
            "generation_stop": "first_eoi",
            "requested_output_size": [self.OUTPUT_SIZE, self.OUTPUT_SIZE],
            "generated_size": list(output.size),
            "peak_cuda_memory_gib": round(
                self.torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "emu3p5_input_artifacts": artifact_paths,
            "emu3p5_proto": str(proto_path),
            **metadata,
        }
        self.torch.cuda.empty_cache()
        return output, extra


@register_model("dreamomni2")
class DreamOmni2Model(BaseModel):
    """DreamOmni2 with every context supplied directly in dataset order."""

    INPUT_SIZE = 1024
    OUTPUT_SIZE = 1024

    def setup(self) -> None:
        if not self.args.repo_path:
            raise ValueError("DreamOmni2 requires --repo_path")

        repo_path = Path(self.args.repo_path).resolve()
        sys.path.insert(0, str(repo_path))
        from . import dreamomni_runtime as runtime
        runtime.load_runtime()

        self.runtime = runtime
        self.torch = runtime.torch
        if not self.torch.cuda.is_available():
            raise RuntimeError("DreamOmni2 requires a CUDA GPU")

        cached_dream_path = cached_flux_path = None
        if not self.args.model_path or not self.args.dreamomni2_flux_path:
            cached_dream_path, cached_flux_path = runtime.resolve_cached_snapshots()
        dream_path = Path(self.args.model_path).resolve() if self.args.model_path else cached_dream_path
        flux_path = (
            Path(self.args.dreamomni2_flux_path).resolve()
            if self.args.dreamomni2_flux_path
            else cached_flux_path
        )
        if dream_path is None or flux_path is None:
            raise RuntimeError("Unable to resolve DreamOmni2 model snapshots")
        vlm_path = dream_path / "vlm-model"
        gen_lora_path = dream_path / "gen_lora"
        for label, path in (
            ("VLM", vlm_path),
            ("generation LoRA", gen_lora_path),
            ("FLUX base", flux_path),
        ):
            if not path.exists():
                raise FileNotFoundError(f"DreamOmni2 {label} path not found: {path}")

        self.pipe = runtime.DreamOmni2Pipeline.from_pretrained(
            flux_path, torch_dtype=self.torch.bfloat16
        )
        self.pipe.to(self.device)
        self.pipe.load_lora_weights(
            gen_lora_path,
            adapter_name="generation",
        )
        self.pipe.set_adapters(
            ["generation"], adapter_weights=[runtime.LORA_WEIGHT]
        )
        self.vlm_model = runtime.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_path, torch_dtype="bfloat16", device_map=self.device
        )
        self.processor = runtime.AutoProcessor.from_pretrained(vlm_path)
        self.provenance = {
            "dreamomni2_repo": runtime.DREAMOMNI2_REPO,
            "dreamomni2_revision": runtime.DREAMOMNI2_REVISION,
            "dreamomni2_snapshot": str(dream_path),
            "flux_repo": runtime.FLUX_REPO,
            "flux_revision": runtime.FLUX_REVISION,
            "flux_snapshot": str(flux_path),
        }

    def _prepare_inputs(
        self, req: InferenceRequest
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Any]]:
        context_ids = list(req.sample["contexts_frame_id"])
        if len(context_ids) != len(req.image_paths):
            raise ValueError(
                f"Context/path count mismatch for {req.sample['sample_id']}: "
                f"{len(context_ids)} ids vs {len(req.image_paths)} paths"
            )
        artifact_dir = (
            Path(req.response_dir) / f"{req.sample['sample_id']}_dreamomni2_inputs"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_input_paths: List[str] = []
        for index, (frame_id, source_path) in enumerate(
            zip(context_ids, req.image_paths), start=1
        ):
            frame_key = QwenImageEdit2511Model._frame_key(frame_id)
            input_path = artifact_dir / f"context_{index:02d}_{frame_key}.png"
            self.runtime.center_crop_resize(
                Path(source_path), self.INPUT_SIZE
            ).save(input_path)
            model_input_paths.append(str(input_path))

        prompt_path = artifact_dir / "prompt.txt"
        prompt_path.write_text(req.prompt, encoding="utf-8")
        artifacts = {
            "input_dir": str(artifact_dir),
            "benchmark_prompt": str(prompt_path),
        }
        metadata = {
            "input_mode": "direct",
            "input_size": self.INPUT_SIZE,
            "input_order": "contexts_frame_id order; Image N maps to context N",
            "context_frame_ids": [str(frame_id) for frame_id in context_ids],
            "model_input_paths": list(model_input_paths),
            "model_input_count": len(model_input_paths),
            "prompt_protocol": "unchanged_benchmark_instruction",
        }
        metadata_path = artifact_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        artifacts["metadata"] = str(metadata_path)
        return model_input_paths, artifacts, metadata

    def infer(self, req: InferenceRequest) -> Tuple[Image.Image, Dict[str, Any]]:
        seed = derive_seed(self.args.seed, req.sample)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        self.torch.backends.cudnn.deterministic = True
        self.torch.backends.cudnn.benchmark = False
        self.torch.cuda.reset_peak_memory_stats()

        model_input_paths, artifacts, metadata = self._prepare_inputs(req)
        with self.torch.inference_mode():
            raw_prompt, refined_prompt = self.runtime.infer_vlm(
                model_input_paths,
                req.prompt,
                self.vlm_model,
                self.processor,
                self.device,
            )
            artifact_dir = Path(artifacts["input_dir"])
            raw_prompt_path = artifact_dir / "vlm_raw_prompt.txt"
            refined_prompt_path = artifact_dir / "vlm_refined_prompt.txt"
            raw_prompt_path.write_text(raw_prompt, encoding="utf-8")
            refined_prompt_path.write_text(refined_prompt, encoding="utf-8")
            artifacts["vlm_raw_prompt"] = str(raw_prompt_path)
            artifacts["vlm_refined_prompt"] = str(refined_prompt_path)
            output = self.runtime.infer_image(
                model_input_paths,
                refined_prompt,
                self.pipe,
                seed,
                self.device,
            ).convert("RGB")

        extra = {
            "seed": seed,
            "torch_dtype": "bfloat16",
            "input_prompt_unchanged": True,
            "vlm_suffix": self.runtime.VLM_SUFFIX,
            "vlm_do_sample": False,
            "vlm_max_new_tokens": self.runtime.VLM_MAX_NEW_TOKENS,
            "lora_weight": self.runtime.LORA_WEIGHT,
            "num_inference_steps": self.runtime.NUM_INFERENCE_STEPS,
            "guidance_scale": self.runtime.GUIDANCE_SCALE,
            "requested_output_size": [self.OUTPUT_SIZE, self.OUTPUT_SIZE],
            "generated_size": list(output.size),
            "peak_cuda_memory_gib": round(
                self.torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "dreamomni2_input_artifacts": artifacts,
            "provenance": self.provenance,
            **metadata,
        }
        self.torch.cuda.empty_cache()
        return output, extra


@register_model("sensenova-u1.5")
class SenseNovaU15Model(BaseModel):
    """Local SenseNova-U1.5 multi-image editing adapter."""

    THINKING_ARG = "sensenova_u15_thinking"
    OUTPUT_SIZE = (2048, 2048)
    CFG_SCALE = 4.0
    IMG_CFG_SCALE = 1.0
    CFG_NORM = "none"
    TIMESTEP_SHIFT = 3.0
    CFG_INTERVAL = (0.0, 1.0)
    NUM_STEPS = 50

    def setup(self) -> None:
        if self.args.repo_path:
            repo_src = Path(self.args.repo_path) / "src"
            import_path = repo_src if repo_src.is_dir() else Path(self.args.repo_path)
            sys.path.insert(0, str(import_path))

        import torch
        import sensenova_u1
        from sensenova_u1.utils import load_model_and_tokenizer

        sensenova_u1.set_attn_backend("auto")
        self.torch = torch
        dtype = getattr(torch, self.args.dtype)
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.args.model_path,
            dtype=dtype,
            device=self.device,
        )

    def infer(self, req: InferenceRequest) -> Tuple[Image.Image, Dict[str, Any]]:
        # Use the same paired-shot seed derivation as Flux2Model and BagelModel.
        seed = derive_seed(self.args.seed, req.sample)
        thinking = bool(getattr(self.args, self.THINKING_ARG))
        with self.torch.inference_mode():
            output = self.model.it2i_generate(
                self.tokenizer,
                req.prompt,
                list(req.images),
                image_size=self.OUTPUT_SIZE,
                cfg_scale=self.CFG_SCALE,
                img_cfg_scale=self.IMG_CFG_SCALE,
                cfg_norm=self.CFG_NORM,
                timestep_shift=self.TIMESTEP_SHIFT,
                cfg_interval=self.CFG_INTERVAL,
                num_steps=self.NUM_STEPS,
                batch_size=1,
                think_mode=thinking,
                seed=seed,
            )
        if thinking:
            output, think_text = output

        pixels = (output[0].float() * 0.5 + 0.5).clamp(0, 1)
        array = (pixels.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        return image, {
            "seed": seed,
            "generated_size": list(self.OUTPUT_SIZE),
            "cfg_scale": self.CFG_SCALE,
            "img_cfg_scale": self.IMG_CFG_SCALE,
            "cfg_norm": self.CFG_NORM,
            "timestep_shift": self.TIMESTEP_SHIFT,
            "num_steps": self.NUM_STEPS,
        }


@register_model("sensenova-u1")
class SenseNovaU1Model(SenseNovaU15Model):
    """Local SenseNova-U1 multi-image editing adapter."""

    THINKING_ARG = "sensenova_u1_thinking"


@register_model("nanobanna-pro")
class NanobannaProModel(BaseModel):
    """OpenAI-compatible chat.completions image generation, e.g. gemini-3-pro-image-preview."""

    def setup(self) -> None:
        from openai import OpenAI
        self.client = OpenAI(base_url=self.args.base_url or None, api_key=os.getenv(self.args.api_key_env))

    @staticmethod
    def _extract_b64(content: str) -> List[str]:
        pats = [
            r"data:image/(?:png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=\n\r]+)",
            r"\(data:image/(?:png|jpeg|jpg|webp);base64,([^)]+)\)",
        ]
        out: List[str] = []
        for p in pats:
            out += re.findall(p, content or "")
        return [re.sub(r"\s+", "", x.strip()) for x in out if x.strip()]

    def infer(self, req: InferenceRequest) -> Tuple[Image.Image, Dict[str, Any]]:
        content = [{"type": "text", "text": req.prompt}]
        for img in req.images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pil_to_b64(img)}"}})

        resp = self.client.chat.completions.create(
            model=self.args.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=self.args.temperature,
        )
        resp_dict = resp.to_dict()
        response_json = save_response(req, resp_dict, f"{req.sample['sample_id']}_response.json")
        text = resp.choices[0].message.content or ""
        Path(req.response_dir).mkdir(parents=True, exist_ok=True)
        text_path = Path(req.response_dir) / f"{req.sample['sample_id']}_content.txt"
        text_path.write_text(text, encoding="utf-8")

        matches = self._extract_b64(text)
        if not matches:
            raise RuntimeError("No base64 image found in chat response")
        img = Image.open(BytesIO(base64.b64decode(matches[0]))).convert("RGB")
        return img, {"response_json": response_json, "response_text": str(text_path)}


@register_model("nano-banana")
class NanoBananaModel(NanobannaProModel):
    """Gemini 2.5 Flash Image through the same OpenAI-compatible gateway."""


@register_model("gpt-image-2")
class GPTImage2Model(BaseModel):
    """OpenAI Images Edit API, e.g. gpt-image-2 with multiple reference images."""

    SLEEP_ENV = "GPT_IMAGE_2_SLEEP_SECONDS"
    SUPPORTS_INPUT_FIDELITY = False

    def setup(self) -> None:
        from openai import OpenAI
        self.client = OpenAI(base_url=self.args.base_url or None, api_key=os.getenv(self.args.api_key_env))

    def infer(self, req: InferenceRequest) -> Tuple[Image.Image, Dict[str, Any]]:
        tmp_paths, files = [], []
        try:
            for i, img in enumerate(req.images):
                tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=f"_{i}.png", delete=False)
                img.save(tmp, format="PNG")
                tmp.close()
                tmp_paths.append(tmp.name)
                files.append(open(tmp.name, "rb"))


            import time

            sleep_s = float(os.getenv(self.SLEEP_ENV, "10"))

            if sleep_s > 0:
                print(f"[{self.name}] sleep {sleep_s}s before API call")
                time.sleep(sleep_s)

            edit_kwargs = {
                "model": self.args.model_name,
                "image": files,
                "prompt": req.prompt,
                "size": self.args.output_size,
                "quality": self.args.quality,
                "output_format": self.args.output_format,
                "n": 1,
            }
            input_fidelity = getattr(self.args, "input_fidelity", None)
            if self.SUPPORTS_INPUT_FIDELITY and input_fidelity:
                edit_kwargs["input_fidelity"] = input_fidelity
            resp = self.client.images.edit(**edit_kwargs)
        finally:
            for f in files:
                try:
                    f.close()
                except Exception:
                    pass
            for p in tmp_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

        resp_dict = resp.to_dict()
        response_json = save_response(req, resp_dict, f"{req.sample['sample_id']}_response.json")
        b64 = None
        for item in getattr(resp, "data", []) or []:
            b64 = getattr(item, "b64_json", None)
            if b64:
                break
        if not b64:
            raise RuntimeError("No b64_json image found in Images Edit response")
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        return img, {"response_json": response_json}


@register_model("gpt-image-1.5")
class GPTImage15Model(GPTImage2Model):
    """GPT Image 1.5 using the shared multi-reference Images Edit adapter."""

    SLEEP_ENV = "GPT_IMAGE_15_SLEEP_SECONDS"
    SUPPORTS_INPUT_FIDELITY = True


@register_model("bagel")
class BagelModel(BaseModel):
    """BAGEL adapter. Set --repo_path to the BAGEL repo if needed."""

    def setup(self) -> None:
        if self.args.repo_path:
            sys.path.insert(0, self.args.repo_path)

        import torch
        from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        from inferencer import InterleaveInferencer
        from modeling.autoencoder import load_ae
        from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
        from modeling.qwen2 import Qwen2Tokenizer

        self.torch = torch
        mp = self.args.model_path
        llm_config = Qwen2Config.from_json_file(os.path.join(mp, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(mp, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        vae_model, vae_config = load_ae(local_path=os.path.join(mp, "ae.safetensors"))
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        tokenizer = Qwen2Tokenizer.from_pretrained(mp)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        device_map = infer_auto_device_map(
            model,
            max_memory={i: self.args.max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        same_device_modules = [
            "language_model.model.embed_tokens", "time_embedder", "latent_pos_embed",
            "vae2llm", "llm2vae", "connector", "vit_pos_embed",
        ]
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map or torch.cuda.device_count() == 1:
                device_map[k] = device_map.get(k, first_device)

        checkpoint_path = (
            self.args.bagel_checkpoint_path
            or os.path.join(mp, "ema.safetensors")
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"BAGEL checkpoint not found: {checkpoint_path}")

        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=checkpoint_path,
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder=self.args.offload_folder,
        ).eval()

        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=ImageTransform(1024, 512, 16),
            vit_transform=ImageTransform(980, 224, 14),
            new_token_ids=new_token_ids,
        )

    def infer(self, req: InferenceRequest) -> Tuple[Image.Image, Dict[str, Any]]:
        seed = derive_seed(self.args.seed, req.sample)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        self.torch.backends.cudnn.deterministic = True
        self.torch.backends.cudnn.benchmark = False
        out = self.inferencer(
            image=req.images,
            text=req.prompt,
            think=bool(self.args.bagel_thinking),
            max_think_token_n=1000,
            do_sample=False,
            cfg_text_scale=self.args.cfg_text_scale,
            cfg_img_scale=self.args.cfg_img_scale,
            cfg_interval=[self.args.cfg_interval_start, self.args.cfg_interval_end],
            timestep_shift=self.args.timestep_shift,
            num_timesteps=self.args.num_timesteps,
            cfg_renorm_min=self.args.cfg_renorm_min,
            cfg_renorm_type=self.args.cfg_renorm_type,
        )
        print(out['text'])
        print(req.prompt)
        return out["image"].convert("RGB"), {"seed": seed}
