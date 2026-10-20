#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .common import EvalPair, append_jsonl, load_image, pair_to_record
from .data_paths import image_path

METRIC_NAME = "vggt"
ROT_ERROR_CAP_DEG = 60.0
TRANS_ERROR_CAP_M = 2.0
METRIC_KEYS = ["rscore_0_100", "tscore_0_100"]


def add_args(parser):
    parser.add_argument("--vggt_repo_path", default=str(Path(__file__).resolve().parents[3] / "vggt"))
    parser.add_argument("--vggt_model_name", default="facebook/VGGT-1B")
    parser.add_argument(
        "--vggt_canonical_json", default="",
        help="Benchmark JSONL whose C10 RGB frames and GT provide common VGGT reference inputs.",
    )
    parser.add_argument(
        "--vggt_common_reference_generation_results",
        default=argparse.SUPPRESS,
        help=(
            "Optional generation_results.json whose shot-10 contexts and GT are "
            "used as common VGGT reference inputs. Predictions always remain from "
            "the evaluated run. Omitting this flag preserves the standard behavior."
        ),
    )
    parser.add_argument("--do_vis", action="store_true")
    parser.add_argument("--vis_dir", default="vis_vggt")
    return parser


def pose_error(R_gt, t_gt, R_pred, t_pred) -> Tuple[float, float]:
    R_rel = R_pred @ R_gt.T
    trace = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
    rot_err = np.arccos(trace) * 180 / np.pi
    C_gt = -R_gt.T @ t_gt
    C_pred = -R_pred.T @ t_pred
    trans_err = np.linalg.norm(C_pred - C_gt)
    return float(rot_err), float(trans_err)


def error_to_score_0_100(error: float, cap: float) -> float:
    return float(np.clip(1.0 - error / cap, 0.0, 1.0) * 100.0)


def reprojection_error(K_gt, K_pred, R_gt, t_gt, R_pred, t_pred, pts3D) -> float:
    X = pts3D.reshape(-1, 3)

    def proj(K, R, t):
        Xc = (R @ X.T).T + t
        z = Xc[:, 2] + 1e-8
        x = Xc[:, 0] / z
        y = Xc[:, 1] / z
        pix = (K @ np.stack([x, y, np.ones_like(x)], axis=1).T).T
        return pix[:, :2]

    p_gt = proj(K_gt, R_gt, t_gt)
    p_pr = proj(K_pred, R_pred, t_pred)
    valid = np.isfinite(p_gt).all(axis=1) & np.isfinite(p_pr).all(axis=1)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.linalg.norm(p_gt[valid] - p_pr[valid], axis=1)))


def normalize_pixel_error(err, H, W) -> float:
    return float(err / np.sqrt(H * H + W * W) * 100)


def _save_temp_images(paths: List[str], tmp_dir: str, image_size: int) -> List[str]:
    out = []
    for i, p in enumerate(paths):
        img = load_image(p, image_size)
        dst = os.path.join(tmp_dir, f"{i:03d}.jpg")
        img.save(dst)
        out.append(dst)
    return out


def _depth_path_from_rgb(rgb_path: str) -> Path:
    parts = list(Path(rgb_path).parts)
    color_indices = [i for i, part in enumerate(parts) if part == "color"]
    if not color_indices:
        raise ValueError(f"Cannot derive depth path from RGB path: {rgb_path}")
    parts[color_indices[-1]] = "depth"
    return Path(*parts).with_suffix(".png")


def _load_metric_depth(rgb_path: str, image_size: int, output_hw: Tuple[int, int]) -> np.ndarray:
    depth_path = _depth_path_from_rgb(rgb_path)
    if not depth_path.is_file():
        raise FileNotFoundError(f"Metric depth not found: {depth_path}")

    with Image.open(rgb_path) as image:
        rgb_size = image.size
    with Image.open(depth_path) as image:
        depth = np.asarray(image)
    if depth.ndim != 2:
        raise ValueError(f"Expected single-channel metric depth: {depth_path}")

    # All three benchmark datasets store depth PNG values in millimetres.
    depth = depth.copy()
    depth[depth == 65535] = 0
    if depth.shape[::-1] != rgb_size:
        depth = np.asarray(
            Image.fromarray(depth).resize(rgb_size, getattr(Image, "Resampling", Image).NEAREST)
        )

    if image_size and image_size > 0:
        width, height = rgb_size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        depth = depth[top : top + side, left : left + side]
        depth = np.asarray(
            Image.fromarray(depth).resize(
                (image_size, image_size), getattr(Image, "Resampling", Image).NEAREST
            )
        )

    out_h, out_w = output_hw
    if depth.shape != (out_h, out_w):
        depth = np.asarray(
            Image.fromarray(depth).resize((out_w, out_h), getattr(Image, "Resampling", Image).NEAREST)
        )
    return depth.astype(np.float64) / 1000.0


def _gt_source_path(pair: EvalPair, args) -> str:
    ground_truth = pair.raw_record.get("ground_truth")
    if isinstance(ground_truth, dict):
        source_path = ground_truth.get("source_path")
        if source_path and Path(str(source_path)).is_file():
            return str(source_path)

    source_path = image_path(
        pair.dataset,
        pair.scene_id,
        pair.gt_frame_id,
        scannet_root=args.scannet_root or args.scan_root,
        scannetpp_root=args.scannetpp_root,
        arkitscenes_root=args.arkitscenes_root,
        data_split=args.data_split,
    )
    if not source_path or not Path(source_path).is_file():
        raise FileNotFoundError(
            f"GT source RGB not found: dataset={pair.dataset}, "
            f"scene={pair.scene_id}, frame={pair.gt_frame_id}"
        )
    return source_path


def _metric_scale_from_depth(
    predicted_depth: np.ndarray,
    depth_conf: np.ndarray,
    real_rgb_paths: List[str],
    image_size: int,
) -> Tuple[float, Dict[str, Any]]:
    if predicted_depth.ndim != 3 or predicted_depth.shape[0] != len(real_rgb_paths):
        raise ValueError(
            f"Depth/input mismatch: depth={predicted_depth.shape}, "
            f"real_images={len(real_rgb_paths)}"
        )
    if depth_conf is not None and depth_conf.shape != predicted_depth.shape:
        raise ValueError(
            f"Depth confidence mismatch: depth={predicted_depth.shape}, "
            f"confidence={depth_conf.shape}"
        )

    frame_log_scales: List[float] = []
    num_valid_pixels = 0
    out_hw = tuple(int(value) for value in predicted_depth.shape[-2:])

    for index, rgb_path in enumerate(real_rgb_paths):
        metric_depth = _load_metric_depth(rgb_path, image_size, out_hw)
        vggt_depth = np.asarray(predicted_depth[index], dtype=np.float64)
        valid = (
            np.isfinite(metric_depth)
            & np.isfinite(vggt_depth)
            & (metric_depth > 0.0)
            & (vggt_depth > 1e-8)
        )

        if depth_conf is not None and int(valid.sum()) >= 512:
            confidence = np.asarray(depth_conf[index], dtype=np.float64)
            conf_values = confidence[valid & np.isfinite(confidence)]
            if conf_values.size >= 512:
                conf_threshold = float(np.median(conf_values))
                confident = valid & np.isfinite(confidence) & (confidence >= conf_threshold)
                if int(confident.sum()) >= 256:
                    valid = confident

        if int(valid.sum()) < 256:
            raise ValueError(
                f"Insufficient valid depth pixels ({int(valid.sum())}) for {rgb_path}"
            )

        log_ratios = np.log(metric_depth[valid]) - np.log(vggt_depth[valid])
        frame_log_scales.append(float(np.median(log_ratios)))
        num_valid_pixels += int(log_ratios.size)

    median_log_scale = float(np.median(frame_log_scales))
    scale = float(np.exp(median_log_scale))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid metric depth scale: {scale}")

    return scale, {
        "depth_scale_m_per_vggt_unit": scale,
        "depth_scale_num_frames": len(frame_log_scales),
        "depth_scale_num_pixels": num_valid_pixels,
        "depth_scale_log_mad": float(
            np.median(np.abs(np.asarray(frame_log_scales) - median_log_scale))
        ),
    }


def _export_vggt_pointcloud(predictions, images, extrinsic, save_path="scene_pc.txt", conf_thresh=0.5):
    world_points = predictions["world_points"][0].detach().cpu()
    conf = predictions["world_points_conf"][0].detach().cpu()
    V, H, W, _ = world_points.shape
    all_pts, all_colors = [], []
    for v in range(V):
        pts_v = world_points[v].reshape(-1, 3).numpy()
        conf_v = conf[v].reshape(-1).numpy()
        img_v = images[v].detach().cpu().permute(1, 2, 0).numpy()
        colors_v = img_v.reshape(-1, 3) * 255
        mask = np.isfinite(pts_v).all(axis=1) & (conf_v > conf_thresh)
        all_pts.append(pts_v[mask])
        all_colors.append(colors_v[mask])
    if not all_pts:
        return
    all_pts = np.concatenate(all_pts, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    np.savetxt(save_path, np.concatenate([all_pts, all_colors], axis=1), fmt="%.6f")


def run_worker(args, pairs: List[EvalPair]) -> None:
    if args.vggt_repo_path:
        sys.path.insert(0, args.vggt_repo_path)

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    metric_dir = Path(args.metrics_root) / METRIC_NAME
    shard_path = metric_dir / "shards" / f"rank_{args.rank}.jsonl"
    if args.overwrite and shard_path.exists():
        shard_path.unlink()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"[vggt] device={device}, dtype={dtype}, pairs={len(pairs)}")
    print(f"[vggt] repo={args.vggt_repo_path}, model={args.vggt_model_name}")

    model = VGGT.from_pretrained(args.vggt_model_name).to(device).eval()

    for pair in tqdm(pairs, desc=f"vggt rank {args.rank}"):
        record: Dict[str, Any] = pair_to_record(pair)

        if not pair.context_source_paths:
            raise RuntimeError("No individual context_source_paths found in generation_results.json; VGGT needs individual context images.")

        image_paths = pair.context_source_paths + [pair.gt_path, pair.pred_path]
        with tempfile.TemporaryDirectory(prefix="vggt_512_") as tmp_dir:
            temp_paths = _save_temp_images(image_paths, tmp_dir, args.image_size)
            images = load_and_preprocess_images(temp_paths).to(device)

        _, _, H, W = images.shape

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

        extr = extrinsic[0].detach().cpu().numpy()
        R = extr[:, :3, :3]
        t = extr[:, :3, 3]
        R_gt, t_gt = R[-2], t[-2]
        R_pred, t_pred = R[-1], t[-1]
        intr = intrinsic[0].detach().cpu().numpy()
        if intr.ndim == 3:
            K_gt, K_pred = intr[-2], intr[-1]
        else:
            K_gt = K_pred = intr
        pts3D = predictions["world_points"][0].detach().cpu().numpy()

        num_contexts = len(pair.context_source_paths)
        real_rgb_paths = pair.context_source_paths + [_gt_source_path(pair, args)]
        predicted_depth = (
            predictions["depth"][0, : num_contexts + 1, ..., 0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        predicted_depth_conf = (
            predictions["depth_conf"][0, : num_contexts + 1]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        metric_scale, scale_diagnostics = _metric_scale_from_depth(
            predicted_depth,
            predicted_depth_conf,
            real_rgb_paths,
            args.image_size,
        )

        rot_err, trans_err_raw = pose_error(R_gt, t_gt, R_pred, t_pred)
        trans_err = float(metric_scale * trans_err_raw)
        reproj_err = reprojection_error(K_gt, K_pred, R_gt, t_gt, R_pred, t_pred, pts3D)
        reproj_norm = normalize_pixel_error(reproj_err, H, W)

        record.update({
            "metric_success": True,
            "num_contexts": num_contexts,
            "original_num_contexts": pair.raw_record.get("vggt_original_num_contexts", num_contexts),
            "context_source_shot": pair.raw_record.get("vggt_context_source_shot", pair.shot),
            "rot_error_deg": rot_err,
            "trans_error": trans_err,
            "rscore_0_100": error_to_score_0_100(rot_err, ROT_ERROR_CAP_DEG),
            "tscore_0_100": error_to_score_0_100(trans_err, TRANS_ERROR_CAP_M),
            "trans_error_raw": trans_err_raw,
            "trans_error_unit": "m",
            "reproj_error_px": reproj_err,
            "reproj_error_0_100": reproj_norm,
            **scale_diagnostics,
        })

        if pair.raw_record.get("vggt_common_reference_active"):
            record.update({
                "common_reference_active": True,
                "common_reference_generation_results": pair.raw_record.get(
                    "vggt_common_reference_generation_results"
                ),
                "common_reference_sha256": pair.raw_record.get(
                    "vggt_common_reference_sha256"
                ),
                "common_reference_sample_id": pair.raw_record.get(
                    "vggt_common_reference_sample_id"
                ),
                "common_reference_shot": pair.raw_record.get(
                    "vggt_common_reference_shot"
                ),
            })

        if args.do_vis:
            vis_dir = metric_dir / args.vis_dir / pair.model / pair.dataset / f"shot-{pair.shot}" / pair.task_type
            vis_dir.mkdir(parents=True, exist_ok=True)
            _export_vggt_pointcloud(predictions, images, extrinsic, vis_dir / f"{pair.sample_id}.txt")


        append_jsonl(shard_path, record)
