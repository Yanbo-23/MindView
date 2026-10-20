"""Public DreamOmni2 wrapper around its upstream generation pipeline.

Adapted from https://github.com/JIA-Lab-research/DreamOmni2/blob/b0b98e03bfc95c2bf0f27a5f9ca33d9b16a525b5/inference_gen.py
under Apache-2.0; see evaluation/third_party/dreamomni2/LICENSE.
Generation constants and model revisions are recovered from frozen run metadata.
The unavailable historical wrapper has not been compared on GPU against this one.
"""
from pathlib import Path

DREAMOMNI2_REPO = "xiabs/DreamOmni2"
DREAMOMNI2_REVISION = "38eb68e8f546bec36680180fd922e61b7112a7e1"
FLUX_REPO = "black-forest-labs/FLUX.1-Kontext-dev"
FLUX_REVISION = "24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d"
VLM_SUFFIX = " It is generation task."
VLM_MAX_NEW_TOKENS = 4096
LORA_WEIGHT = 1.0
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 3.5


def load_runtime():
    global torch, DreamOmni2Pipeline, Qwen2_5_VLForConditionalGeneration, AutoProcessor
    global process_vision_info, resizeinput, load_image
    import torch
    from dreamomni2.pipeline_dreamomni2 import DreamOmni2Pipeline
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from utils.vprocess import process_vision_info, resizeinput
    from diffusers.utils import load_image


def resolve_cached_snapshots():
    from huggingface_hub import snapshot_download
    return (Path(snapshot_download(DREAMOMNI2_REPO, revision=DREAMOMNI2_REVISION, local_files_only=True)),
            Path(snapshot_download(FLUX_REPO, revision=FLUX_REVISION, local_files_only=True)))


def refine_prompt(raw):
    # This upstream extraction also matches the saved raw/refined benchmark artifact.
    return raw[6:-7]


def infer_vlm(paths, instruction, model, processor, device):
    content = [{"type": "image", "image": path} for path in paths]
    content.append({"type": "text", "text": instruction + VLM_SUFFIX})
    messages = [{"role": "user", "content": content}]
    template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[template], images=images, videos=videos, padding=True, return_tensors="pt").to(device)
    generated = model.generate(**inputs, do_sample=False, max_new_tokens=VLM_MAX_NEW_TOKENS)
    trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return raw, refine_prompt(raw)


def infer_image(paths, prompt, pipe, seed, device):
    images = [resizeinput(load_image(path)) for path in paths]
    generator = torch.Generator(device=device).manual_seed(seed)
    return pipe(images=images, prompt=prompt, height=1024, width=1024,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE, generator=generator).images[0]
