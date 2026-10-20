#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

from .data_paths import canonical_dataset_name, validate_shot


def load_json_like(path: str) -> Any:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Many benchmark files are JSONL: one JSON object per line.
            return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _flatten_from_dict(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Support {task: [samples]}, {tasks: {task: ...}}, {samples: [...]}, and one sample dict."""
    if "sample_id" in obj and "scene_id" in obj:
        return [obj]

    if isinstance(obj.get("samples"), list):
        return list(obj["samples"])

    if isinstance(obj.get("data"), list):
        return list(obj["data"])

    if isinstance(obj.get("tasks"), dict):
        obj = obj["tasks"]

    samples: List[Dict[str, Any]] = []
    for task_name, value in obj.items():
        if isinstance(value, list):
            cur = value
        elif isinstance(value, dict) and isinstance(value.get("samples"), list):
            cur = value["samples"]
        elif isinstance(value, dict) and isinstance(value.get("data"), list):
            cur = value["data"]
        else:
            continue
        for s in cur:
            s = dict(s)
            s.setdefault("task_type", task_name)
            samples.append(s)
    return samples

def copy_context_images(
    src_paths: List[str],
    frame_ids: List[str],
    dst_dir: str,
    image_size: int = 512,
    sample_id: str = "sample",
) -> List[Optional[str]]:
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    imgs=[]
    saved=[]
    for i,src in enumerate(src_paths):
        if not src or not Path(src).exists():
            continue
        img=preprocess_image(Image.open(src), image_size)
        imgs.append(img)
    if not imgs:
        return []
    canvas=Image.new("RGB", (image_size*len(imgs), image_size))
    for i,img in enumerate(imgs):
        canvas.paste(img,(i*image_size,0))
    out=dst_dir/f"{sample_id}.jpg"
    canvas.save(out)
    return [str(out)]

def load_samples(path: str) -> List[Dict[str, Any]]:
    obj = load_json_like(path)
    if isinstance(obj, list):
        samples = obj
    elif isinstance(obj, dict):
        samples = _flatten_from_dict(obj)
    else:
        raise ValueError(f"Unsupported benchmark json format: {type(obj)}")

    out: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        s = dict(s)
        s.setdefault("sample_id", f"sample_{i:08d}")
        s.setdefault("task_type", "default")
        s["dataset"] = canonical_dataset_name(s.get("dataset"))
        s["shot"] = validate_shot(s.get("shot"))
        s["_global_index"] = i
        out.append(s)
    return out


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preprocess_image(img: Image.Image, image_size: int) -> Image.Image:
    img = img.convert("RGB")
    if image_size <= 0:
        return img
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((image_size, image_size), getattr(Image, "Resampling", Image).LANCZOS)


def load_rgb_images(paths: Iterable[str], image_size: int) -> List[Image.Image]:
    return [preprocess_image(Image.open(p), image_size) for p in paths]


def copy_gt_image(src: str, dst: str, image_size: int = 512) -> Optional[str]:
    if not src or not Path(src).exists():
        return None
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    img = preprocess_image(Image.open(src), image_size)
    img.save(dst)
    return dst


def safe_rmtree_file(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
