"""Small torchrun runtime used by spatial benchmark evaluation."""

from __future__ import annotations

import os
import traceback
from collections.abc import Callable
from datetime import timedelta
from typing import Any, TypeVar

T = TypeVar("T")


class DistributedRuntime:
    """Support plain Python and torchrun without wrapping the model in DDP."""

    def __init__(self, *, prepare_only: bool = False) -> None:
        self.torch = None
        self.dist = None
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        self.prepare_only = prepare_only
        self.device = None

        if self.rank < 0 or self.rank >= self.world_size:
            raise RuntimeError(
                f"Invalid distributed rank/world size: {self.rank}/{self.world_size}"
            )

        if not prepare_only or self.distributed:
            import torch
            import torch.distributed as dist
            self.torch, self.dist = torch, dist

        if not prepare_only:
            if not torch.cuda.is_available():
                raise RuntimeError("BAGEL evaluation requires a visible CUDA GPU")
            visible = torch.cuda.device_count()
            if self.local_rank < 0 or self.local_rank >= visible:
                raise RuntimeError(
                    f"LOCAL_RANK={self.local_rank}, but only {visible} GPUs are visible"
                )
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)

        if self.distributed:
            backend = "gloo" if prepare_only else "nccl"
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(hours=24),
            )

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def log(self, message: str, *, primary_only: bool = False) -> None:
        if not primary_only or self.is_primary:
            print(f"[rank {self.rank}/{self.world_size}] {message}", flush=True)

    def barrier(self) -> None:
        if self.distributed:
            self.dist.barrier()

    def broadcast_object(self, value: T | None, *, source: int = 0) -> T:
        if not self.distributed:
            return value  # type: ignore[return-value]
        payload: list[Any] = [value]
        self.dist.broadcast_object_list(payload, src=source)
        return payload[0]

    def primary_call(self, fn: Callable[[], T], *, label: str) -> T:
        """Run ``fn`` on rank 0 and propagate either its value or traceback."""

        payload: dict[str, Any] | None = None
        if self.is_primary:
            try:
                payload = {"ok": True, "value": fn()}
            except Exception as exc:  # noqa: BLE001 - must reach every rank
                payload = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
        payload = self.broadcast_object(payload)
        if not payload["ok"]:
            raise RuntimeError(
                f"Rank-0 operation failed while trying to {label}:\n"
                f"{payload['traceback']}"
            )
        return payload["value"]

    def raise_if_any_rank_failed(self, error: str | None, *, label: str) -> None:
        """Synchronize recoverable Python initialization failures across ranks."""

        if self.distributed:
            errors: list[str | None] = [None] * self.world_size
            self.dist.all_gather_object(errors, error)
        else:
            errors = [error]
        failures = [
            f"rank {rank}:\n{message}"
            for rank, message in enumerate(errors)
            if message is not None
        ]
        if failures:
            joined = "\n".join(failures)
            raise RuntimeError(
                f"At least one rank failed while trying to {label}:\n{joined}"
            )

    def close(self) -> None:
        if self.distributed and self.dist.is_initialized():
            self.dist.destroy_process_group()


__all__ = ["DistributedRuntime"]
