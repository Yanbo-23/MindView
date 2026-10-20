#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import EvalPair, append_jsonl, load_image, pair_to_record

METRIC_NAME = "clip_dino"
METRIC_KEYS = ["clip_i", "dino_i"]


def add_args(parser):
    parser.add_argument("--clip_model_name", default="openai/clip-vit-large-patch14")
    parser.add_argument("--dino_model_name", default="facebook/dinov2-large")
    return parser


def _cosine(feat1, feat2) -> float:
    return float((feat1 * feat2).sum(dim=-1).item())


@torch.no_grad()
def _clip_feat(image, model, processor, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    feat = model.get_image_features(**inputs)
    return F.normalize(feat, dim=-1)


@torch.no_grad()
def _dino_feat(image, model, processor, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    feat = outputs.last_hidden_state[:, 0]
    return F.normalize(feat, dim=-1)


def run_worker(args, pairs: List[EvalPair]) -> None:
    from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor, AutoModel

    metric_dir = Path(args.metrics_root) / METRIC_NAME
    shard_path = metric_dir / "shards" / f"rank_{args.rank}.jsonl"
    if args.overwrite and shard_path.exists():
        shard_path.unlink()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip_dino] device={device}, pairs={len(pairs)}")
    print(f"[clip_dino] clip={args.clip_model_name}, dino={args.dino_model_name}")

    clip_model = CLIPModel.from_pretrained(args.clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)

    dino_model = AutoModel.from_pretrained(args.dino_model_name).to(device).eval()
    dino_processor = AutoImageProcessor.from_pretrained(args.dino_model_name)

    for pair in tqdm(pairs, desc=f"clip_dino rank {args.rank}"):
        record: Dict[str, Any] = pair_to_record(pair)
        try:
            gt_img = load_image(pair.gt_path, args.image_size)
            pred_img = load_image(pair.pred_path, args.image_size)

            gt_clip = _clip_feat(gt_img, clip_model, clip_processor, device)
            pred_clip = _clip_feat(pred_img, clip_model, clip_processor, device)
            gt_dino = _dino_feat(gt_img, dino_model, dino_processor, device)
            pred_dino = _dino_feat(pred_img, dino_model, dino_processor, device)

            record.update({
                "metric_success": True,
                "clip_i": _cosine(gt_clip, pred_clip),
                "dino_i": _cosine(gt_dino, pred_dino),
            })
        except Exception as e:
            record.update({
                "metric_success": False,
                "clip_i": None,
                "dino_i": None,
                "error": repr(e),
            })
        append_jsonl(shard_path, record)
