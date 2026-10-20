"""Understanding-only BAGEL loader for one-model-per-GPU evaluation."""

from __future__ import annotations

import hashlib
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SFT_ROOT = REPO_ROOT / "sft"
TRANSFORM_CONFIG_PATH = REPO_ROOT / "understanding" / "image_transform.yaml"
FROZEN_TRANSFORM_ARGS = {
    "max_image_size": 980, "min_image_size": 378,
    "image_stride": 14, "max_pixels": 2007040,
}


def _require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def _sha256_small_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sampled_file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Bind resume metadata to this checkpoint without a full-file reread."""

    size = path.stat().st_size
    offsets = sorted({0, max(0, size // 2), max(0, size - chunk_size)})
    digest = hashlib.sha256()
    digest.update(size.to_bytes(16, byteorder="little"))
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            digest.update(offset.to_bytes(16, byteorder="little"))
            digest.update(handle.read(chunk_size))
    return digest.hexdigest()


def _load_image_transform_identity() -> dict[str, Any]:
    import yaml

    path = _require_file(TRANSFORM_CONFIG_PATH, "BAGEL VLM transform config")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    try:
        raw_args = config["vlm_sft"]["image_transform_args"]
        transform_args = {
            key: int(raw_args[key])
            for key in (
                "max_image_size",
                "min_image_size",
                "image_stride",
                "max_pixels",
            )
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid vlm_sft.image_transform_args in {path}") from exc
    if transform_args != FROZEN_TRANSFORM_ARGS:
        raise RuntimeError("Understanding image transform differs from the frozen VLM protocol")
    return {
        "config_path": str(path),
        "config_sha256": _sha256_small_file(path),
        "args": transform_args,
    }


def checkpoint_identity(
    model_path: str | Path,
    checkpoint_path: str | Path,
    *,
    allow_incomplete_checkpoint: bool,
) -> dict[str, Any]:
    model_root = Path(model_path).expanduser().resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(f"BAGEL model directory does not exist: {model_root}")
    checkpoint = _require_file(Path(checkpoint_path), "BAGEL checkpoint")

    success_marker = checkpoint.parent / "_SUCCESS"
    numeric_checkpoint_dir = checkpoint.parent.name.isdigit()
    if (
        numeric_checkpoint_dir
        and not success_marker.is_file()
        and not allow_incomplete_checkpoint
    ):
        raise RuntimeError(
            f"Refusing an incomplete training checkpoint without _SUCCESS: {checkpoint.parent}. "
            "Use --allow-incomplete-checkpoint only for an intentional diagnostic."
        )

    assets = {}
    for filename in (
        "llm_config.json",
        "vit_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
    ):
        path = _require_file(model_root / filename, f"BAGEL asset {filename}")
        assets[filename] = {
            "path": str(path),
            "sha256": _sha256_small_file(path),
        }
    stat = checkpoint.stat()
    return {
        "model_path": str(model_root),
        "checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "checkpoint_sampled_sha256": _sampled_file_sha256(checkpoint),
        "checkpoint_success_marker": str(success_marker)
        if success_marker.is_file()
        else None,
        "assets": assets,
        "image_transform": _load_image_transform_identity(),
    }


def stream_safetensors_checkpoint(
    model: Any,
    checkpoint_path: str | Path,
    *,
    device: Any,
    dtype: Any,
) -> tuple[str, ...]:
    """Materialize a meta model one tensor at a time.

    Accept a full BF16 release EMA or an FP32 training EMA. Streaming avoids
    loading a complete higher-precision checkpoint on CUDA before casting;
    only understanding parameters are materialized and generation-only keys
    are reported separately.
    """

    import inspect
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    checkpoint = _require_file(Path(checkpoint_path), "BAGEL checkpoint")
    model_key_order = tuple(model.state_dict())
    model_keys = set(model_key_order)
    move_options = {"clear_cache": False} if "clear_cache" in inspect.signature(set_module_tensor_to_device).parameters else {}

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        checkpoint_keys = set(handle.keys())
        missing = sorted(model_keys - checkpoint_keys)
        if missing:
            preview = ", ".join(missing[:10])
            raise RuntimeError(
                f"Checkpoint is missing {len(missing)} understanding weights: {preview}"
            )

        for name in model_key_order:
            tensor = handle.get_tensor(name)
            target_dtype = dtype if torch.is_floating_point(tensor) else None
            set_module_tensor_to_device(
                model,
                name,
                device,
                value=tensor,
                dtype=target_dtype,
                **move_options,
            )
            del tensor

    # ``init_empty_weights`` leaves non-persistent buffers on CPU. They are not
    # present in state_dict(), so the tensor-by-tensor checkpoint loop above
    # cannot move them. Qwen2 RoPE's inv_freq is one such buffer.
    model.to(device=device)
    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        if isinstance(inv_freq, torch.Tensor) and hasattr(module, "original_inv_freq"):
            module.original_inv_freq = inv_freq

    named_tensors = tuple(model.named_parameters()) + tuple(model.named_buffers())
    meta = [name for name, tensor in named_tensors if tensor.is_meta]
    wrong_device = [
        name for name, tensor in named_tensors if tensor.device != torch.device(device)
    ]
    wrong_dtype = [
        name
        for name, tensor in model.named_parameters()
        if torch.is_floating_point(tensor) and tensor.dtype != dtype
    ]
    del named_tensors
    if meta:
        raise RuntimeError(f"{len(meta)} model tensors remain on meta: {meta[:10]}")
    if wrong_device:
        raise RuntimeError(
            f"{len(wrong_device)} model tensors are on the wrong device: "
            f"{wrong_device[:10]}"
        )
    if wrong_dtype:
        raise RuntimeError(
            f"{len(wrong_dtype)} floating tensors have the wrong dtype: "
            f"{wrong_dtype[:10]}"
        )
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return tuple(sorted(checkpoint_keys - model_keys))


class BagelUnderstandingRunner:
    """Load BAGEL's official VLM path on exactly one local CUDA device."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        checkpoint_path: str | Path,
        device: Any,
        seed: int,
    ) -> None:
        import torch
        from accelerate import init_empty_weights

        # Generation and training share this runtime; keep its imports lazy
        # so data preparation and --dry-run need no model dependencies.
        if str(SFT_ROOT) not in sys.path:
            sys.path.insert(0, str(SFT_ROOT))
        from data.data_utils import add_special_tokens, pil_img2rgb
        from data.transforms import ImageTransform
        from modeling.bagel import (
            Bagel,
            BagelConfig,
            Qwen2Config,
            Qwen2ForCausalLM,
            SiglipVisionConfig,
            SiglipVisionModel,
        )
        from modeling.qwen2 import Qwen2Tokenizer

        self.torch = torch
        self._pil_img2rgb = pil_img2rgb
        self.device = device
        self.model_path = Path(model_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_grad_enabled(False)

        llm_config = Qwen2Config.from_json_file(
            str(self.model_path / "llm_config.json")
        )
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(
            str(self.model_path / "vit_config.json")
        )
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        config = BagelConfig(
            visual_gen=False,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
        )
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config, meta=True
            )

        tokenizer = Qwen2Tokenizer.from_pretrained(str(self.model_path))
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # Explicit one-device streaming is essential under torchrun: every
        # rank owns one complete model and never sees another rank's GPU. It
        # also avoids an FP32-checkpoint + BF16-model CUDA loading peak.
        self.unused_checkpoint_keys = stream_safetensors_checkpoint(
            model,
            self.checkpoint_path,
            device=device,
            dtype=torch.bfloat16,
        )
        model = model.eval()

        transform_args = _load_image_transform_identity()["args"]
        image_transform = ImageTransform(
            max_image_size=transform_args["max_image_size"],
            min_image_size=transform_args["min_image_size"],
            image_stride=transform_args["image_stride"],
            max_pixels=transform_args["max_pixels"],
        )

        self.model = model
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        self.image_transform = image_transform
        self.num_parameters = sum(parameter.numel() for parameter in model.parameters())

    def generate(
        self,
        *,
        images: Sequence[Any],
        prompt: str,
        max_new_tokens: int,
    ) -> tuple[str, float]:
        import time

        processed_images = [self._pil_img2rgb(image) for image in images]
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            response = self.model.chat(
                self.tokenizer,
                self.new_token_ids,
                self.image_transform,
                images=processed_images,
                prompt=prompt,
                max_length=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        self.torch.cuda.synchronize(self.device)
        return response, time.perf_counter() - started

__all__ = [
    "BagelUnderstandingRunner",
    "checkpoint_identity",
    "stream_safetensors_checkpoint",
]
