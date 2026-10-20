#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from .common import EvalPair, append_jsonl, load_image, pair_to_record

METRIC_NAME = "qwen_vl"
JUDGE_VERSION = "qwen_judge_v2"
QWEN38_MODEL_NAME = "Qwen/Qwen3.8-27B"
METRIC_KEYS = [
    "semantic_score_100", "spatial_score_100", "visual_quality_score_100",
]


def add_args(parser):
    parser.add_argument("--qwen_model_name", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--use_flash_attn", action="store_true")
    return parser


def build_fidelity_prompt() -> str:
    return """
You are a strict, task-independent evaluator of image fidelity.

The labeled REFERENCE IMAGE is the ground-truth target.
The labeled CANDIDATE IMAGE is the model prediction.

Important rules:
- The two images may or may not depict the same scene. Do not assume that they match.
- Inspect each image independently before comparing them.
- Use only visible pixel evidence.
- Never attribute an object, region, or property to one image merely because it appears in the other.
- Do not use or infer any hidden prompt, context images, task type, or camera metadata.
- Evaluate Semantic Fidelity and Spatial Fidelity independently.
- Do not judge aesthetic preference or technical image quality in this evaluation.

1. Semantic Fidelity: WHAT is present

Evaluate:
- scene identity and scene category;
- main objects, structures, and recognizable content;
- approximate object count and identity-defining attributes;
- missing, extra, substituted, hallucinated, or unrecognizable content.

Do NOT evaluate:
- camera viewpoint, orientation, framing, crop, or visible side by itself;
- object position, scale, depth, occlusion, or relative layout;
- global color tone, lighting, rendering style, sharpness, or visual polish.

A spatially displaced object can still be semantically correct.
A viewpoint difference alone must not reduce Semantic Fidelity if the same important content remains visible and recognizable.
However, important content that is visibly missing or hallucinated is a semantic error, regardless of its cause.

Semantic Fidelity scoring:
10 = The same scene-specific content is preserved almost completely. All major objects and structures are present and recognizable, with no meaningful missing, extra, substituted, or hallucinated content.
8 = The same scene and all major content are preserved, with only minor errors involving secondary objects or identity-defining attributes.
6 = The core scene is clearly recognizable, but one major object or several secondary elements are missing, substituted, hallucinated, or unrecognizable.
4 = Only partial scene-specific content is preserved. Multiple major objects or structures are incorrect, although some meaningful correspondence remains.
2 = The images share only a generic scene category or a few isolated object categories; scene-specific correspondence is very weak.
0 = No meaningful semantic correspondence, or the candidate contains no interpretable scene content.

Semantic score constraints:
- Generic category similarity alone, such as both images being bedrooms or offices, cannot receive a score above 3.
- If one major or scene-defining object is missing, replaced, or hallucinated, the score cannot exceed 6.
- If multiple major objects are incorrect, the score cannot exceed 4.
- A score of 9 or 10 requires nearly complete scene-specific correspondence, not merely a similar category or overall impression.

2. Spatial Fidelity: WHERE things are and FROM WHERE the scene is viewed

Evaluate:
- camera viewpoint and orientation;
- framing, crop, and visible scene regions or surfaces;
- relative object positions, scale, depth ordering, and occlusion;
- scene layout, geometry, and composition.

Judge spatial relations using scene elements that can be meaningfully corresponded between the two images.

Do NOT evaluate:
- object category or semantic identity by itself;
- color, lighting, texture, rendering style, sharpness, or visual attractiveness;
- isolated generation artifacts unless they make spatial relationships impossible to assess.

Do not reward a generic similar composition unless corresponding scene elements support it.
If meaningful spatial correspondence cannot be established, use a score of 0, 1, or 2.

Spatial Fidelity scoring:
10 = Viewpoint, orientation, framing, visible regions, and spatial layout essentially match, with only negligible offsets.
8 = The intended view and layout are correct, with only minor shifts, crop or scale differences, or small geometric errors.
6 = The overall view is similar, but noticeable differences exist in camera pose, framing, visibility, scale, or relative placement.
4 = Some spatial correspondence exists, but the viewpoint, visible region, or layout is clearly incorrect.
2 = Very little spatial correspondence remains; there is a major camera-pose, framing, or layout mismatch.
0 = No usable spatial correspondence, or spatial structure cannot be assessed.

Spatial score constraints:
- A clearly incorrect viewing direction or target-facing direction cannot receive a score above 4.
- A major mismatch in the visible scene region or room side cannot receive a score above 4.
- Generic centered composition without scene-specific correspondence cannot receive a score above 2.
- A score of 9 or 10 requires both camera view and scene layout to match closely.

Use integers from 0 to 10. Integers between the stated anchors may be used when justified.
Apply all score constraints before producing the final scores.

Each reason must identify the strongest visible evidence and explicitly distinguish the Reference from the Candidate.
Keep each reason to one short sentence.

Return ONLY valid JSON in exactly this format:

{
  "semantic_score": <integer 0-10>,
  "spatial_score": <integer 0-10>,
  "semantic_reason": "<short evidence-based reason>",
  "spatial_reason": "<short evidence-based reason>"
}
""".strip()


def build_quality_prompt() -> str:
    return """
You are a strict evaluator of the intrinsic visual quality of a generated image.

Only the labeled CANDIDATE IMAGE is provided.
Evaluate whether the image is visually clean, structurally coherent, and free of generation defects.

Use only visible pixel evidence.
Judge the candidate image independently, without assuming any unseen reference, prompt, context image, task, or intended camera pose.

Evaluate:
- malformed, melted, stretched, fused, or broken objects and structures;
- internally inconsistent room, object, perspective, or surface geometry;
- duplicated or unnaturally repeated elements;
- seams, corrupted regions, incomplete rendering, and white or black borders;
- unintended text, watermarks, blank regions, or padding;
- unintended blur, pixelation, noise, aliasing, or compression artifacts;
- implausible edges, lighting discontinuities, and inconsistent shadows.

Do NOT evaluate:
- whether the image matches an unseen reference, prompt, task, or intended viewpoint;
- whether its objects or layout are correct for an external target;
- semantic fidelity or spatial fidelity;
- aesthetic preference or similarity to a particular rendering style.

Do not reward an image merely because it looks attractive, polished, colorful, or cinematic.
Do not penalize unusual content when it is internally coherent.
Judge coherence within the image's own visual domain.

Visual Quality scoring:
10 = Technically clean and internally coherent. No meaningful deformation, broken geometry, duplication, border, seam, corruption, or unintended artifact is visible.
8 = High visual quality. Only minor and localized defects are visible, without affecting important objects or structures.
6 = Generally usable, but one prominent defect or several noticeable local artifacts are present.
4 = Multiple prominent defects affect important regions, such as malformed objects, inconsistent geometry, repeated structures, seams, blur, or borders.
2 = Severe and widespread distortion, incomplete rendering, broken perspective, large blank regions, or obvious corruption makes the image difficult to use.
0 = Blank, invalid, unreadable, almost completely corrupted, or a failed generation.

Visual Quality score constraints:
- One obvious structural deformation in a salient region prevents a score above 7.
- Multiple visible deformations, repetitions, seams, or broken structures prevent a score above 5.
- A large white or black border, incomplete rendering, or corrupted region prevents a score above 4.
- A score of 9 or 10 requires careful inspection to reveal no meaningful defect.
- Attractive lighting, color, or visual polish alone must not increase the score.

Use an integer from 0 to 10. Integers between the stated anchors may be used when justified.
Apply all score constraints before producing the final score.

The reason must identify the most important visible defect or, when no meaningful defect exists, explicitly state that the image is visually coherent.
Keep the reason to one short sentence.

Return ONLY valid JSON in exactly this format:

{
  "visual_quality_score": <integer 0-10>,
  "visual_quality_reason": "<short evidence-based reason>"
}
""".strip()


def extract_first_json_block(text: str) -> Optional[str]:
    text = text.strip()
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def parse_fidelity_output(text: str) -> Dict[str, Any]:
    default = {
        "semantic_score": None,
        "spatial_score": None,
        "semantic_reason": "",
        "spatial_reason": "",
        "fidelity_parse_success": False,
        "raw_fidelity_output": text,
    }
    block = extract_first_json_block(text)
    if block is None:
        return default
    try:
        obj = json.loads(block)
        for key in ["semantic_score", "spatial_score"]:
            v = obj.get(key)
            if type(v) is int and 0 <= v <= 10:
                default[key] = v
        default["semantic_reason"] = str(obj.get("semantic_reason", "")).strip()
        default["spatial_reason"] = str(obj.get("spatial_reason", "")).strip()
        default["fidelity_parse_success"] = all(
            default[k] is not None for k in ["semantic_score", "spatial_score"]
        )
        return default
    except Exception:
        return default


def parse_quality_output(text: str) -> Dict[str, Any]:
    default = {
        "visual_quality_score": None,
        "visual_quality_reason": "",
        "quality_parse_success": False,
        "raw_quality_output": text,
    }
    block = extract_first_json_block(text)
    if block is None:
        return default
    try:
        obj = json.loads(block)
        v = obj.get("visual_quality_score")
        if type(v) is int and 0 <= v <= 10:
            default["visual_quality_score"] = v
        default["visual_quality_reason"] = str(obj.get("visual_quality_reason", "")).strip()
        default["quality_parse_success"] = default["visual_quality_score"] is not None
        return default
    except Exception:
        return default


def load_qwen_model(model_name: str, use_flash_attn: bool = False):
    from transformers import AutoProcessor

    if model_name == QWEN38_MODEL_NAME:
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.8-27B requires a Transformers environment with "
                "AutoModelForMultimodalLM support (transformers>=5.8.0)."
            ) from exc
        model_cls = AutoModelForMultimodalLM
        kwargs = {"dtype": "auto", "device_map": "auto"}
    else:
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText
        kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = model_cls.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    return model, processor


def _generate_output(messages, model, processor, max_new_tokens: int) -> str:
    template_kwargs = {}
    if getattr(getattr(model, "config", None), "model_type", "") == "qwen3_5":
        template_kwargs["enable_thinking"] = False
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    )
    if isinstance(inputs, dict):
        inputs.pop("token_type_ids", None)
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
    )
    input_ids = inputs["input_ids"]
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return output_text


@torch.inference_mode()
def judge_single_pair(gt_img, pred_img, model, processor, max_new_tokens: int = 256) -> Tuple[str, Dict[str, Any]]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "REFERENCE IMAGE (GROUND-TRUTH TARGET):"},
            {"type": "image", "image": gt_img},
            {"type": "text", "text": "CANDIDATE IMAGE (MODEL PREDICTION):"},
            {"type": "image", "image": pred_img},
            {"type": "text", "text": build_fidelity_prompt()},
        ],
    }]
    output_text = _generate_output(messages, model, processor, max_new_tokens)
    return output_text, parse_fidelity_output(output_text)


@torch.inference_mode()
def judge_visual_quality(pred_img, model, processor, max_new_tokens: int = 256) -> Tuple[str, Dict[str, Any]]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "CANDIDATE IMAGE (MODEL PREDICTION):"},
            {"type": "image", "image": pred_img},
            {"type": "text", "text": build_quality_prompt()},
        ],
    }]
    output_text = _generate_output(messages, model, processor, max_new_tokens)
    return output_text, parse_quality_output(output_text)


def run_worker(args, pairs: List[EvalPair]) -> None:
    metric_dir = Path(args.metrics_root) / METRIC_NAME
    shard_path = metric_dir / "shards" / f"rank_{args.rank}.jsonl"
    if args.overwrite and shard_path.exists():
        shard_path.unlink()

    print(f"[qwen_vl] pairs={len(pairs)}, model={args.qwen_model_name}")
    model, processor = load_qwen_model(args.qwen_model_name, args.use_flash_attn)
    print("[qwen_vl] model loaded")

    for pair in tqdm(pairs, desc=f"qwen_vl rank {args.rank}"):
        record: Dict[str, Any] = pair_to_record(pair)
        record["judge_version"] = JUDGE_VERSION
        try:
            gt_img = load_image(pair.gt_path, args.image_size)
            pred_img = load_image(pair.pred_path, args.image_size)
            quality_img = load_image(pair.pred_path, 0)
            if args.image_size and args.image_size > 0:
                quality_img.thumbnail((args.image_size, args.image_size))

            raw_output, fidelity = judge_single_pair(
                gt_img, pred_img, model, processor, args.max_new_tokens
            )
            _, quality = judge_visual_quality(
                quality_img, model, processor, args.max_new_tokens
            )

            record.update(fidelity)
            record.update(quality)
            record["raw_output"] = raw_output  # Backward-compatible alias.
            record["parse_success"] = bool(
                fidelity.get("fidelity_parse_success")
                and quality.get("quality_parse_success")
            )
            record["metric_success"] = record["parse_success"]
            for k in ["semantic_score", "spatial_score", "visual_quality_score"]:
                v = record.get(k)
                record[f"{k}_100"] = v * 10.0 if record["parse_success"] else None
        except Exception as e:
            record.update({
                "metric_success": False,
                "parse_success": False,
                "fidelity_parse_success": False,
                "quality_parse_success": False,
                "error": repr(e),
                "raw_output": "",
                "raw_fidelity_output": "",
                "raw_quality_output": "",
                "semantic_score": None,
                "spatial_score": None,
                "visual_quality_score": None,
                "semantic_score_100": None,
                "spatial_score_100": None,
                "visual_quality_score_100": None,
                "semantic_reason": "",
                "spatial_reason": "",
                "visual_quality_reason": "",
            })
        append_jsonl(shard_path, record)
