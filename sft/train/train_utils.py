# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import os


def _is_complete_training_ckpt(path, expected_optimizer_shards):
    required_files = (
        "_SUCCESS",
        "ema.safetensors",
        "model.safetensors",
        "scheduler.pt",
        "data_status.pt",
    )
    if not all(
        os.path.isfile(os.path.join(path, name))
        and os.path.getsize(os.path.join(path, name)) > 0
        for name in required_files
    ):
        return False

    optimizer_files = {
        f"optimizer.{index:05d}-of-{expected_optimizer_shards:05d}.pt"
        for index in range(expected_optimizer_shards)
    }
    actual_optimizer_files = {
        name
        for name in os.listdir(path)
        if name.startswith("optimizer.") and name.endswith(".pt")
    }
    return actual_optimizer_files == optimizer_files and all(
        os.path.isfile(os.path.join(path, name))
        and os.path.getsize(os.path.join(path, name)) > 0
        for name in optimizer_files
    )


def create_logger(logging_dir, rank, filename="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0 and logging_dir is not None:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(), 
                logging.FileHandler(f"{logging_dir}/{filename}.txt")
            ]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir, expected_optimizer_shards=None):
    step_dirs = [
        d
        for d in os.listdir(checkpoint_dir)
        if d.isdigit() and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if len(step_dirs) == 0:
        return None
    step_dirs = sorted(step_dirs, key=int, reverse=True)
    if expected_optimizer_shards is None:
        return os.path.join(checkpoint_dir, step_dirs[0])
    if expected_optimizer_shards <= 0:
        raise ValueError("expected_optimizer_shards must be positive")

    for step_dir in step_dirs:
        candidate = os.path.join(checkpoint_dir, step_dir)
        if _is_complete_training_ckpt(candidate, expected_optimizer_shards):
            return candidate
    raise RuntimeError(
        "Checkpoint directories exist, but none is complete for "
        f"{expected_optimizer_shards} optimizer shards: {checkpoint_dir}"
    )
