#!/usr/bin/env python3
"""Select a reproducible C6 smoke-test subset without changing frozen records."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN_JSONL = HERE / "data/merged_dataset_final.jsonl"
FIELDS = {"sample_id", "dataset", "task_type", "scene_id", "gt_frame_id",
          "anchor_frame_id", "shot", "contexts_frame_id", "instruction"}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_frozen_source(source: Path):
    """Accept the bundled canonical index or a byte-identical downloaded copy."""
    payload = source.read_bytes()
    release = json.loads((HERE / "data/manifest.json").read_text())
    if sha256(payload) != release["sha256"]:
        raise ValueError("Source SHA256 differs from the frozen 4,500-record benchmark")
    lines = payload.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    if (len(rows) != release["num_rows"] or any(set(row) != FIELDS for row in rows)
            or len({row["sample_id"] for row in rows}) != len(rows)):
        raise ValueError("Invalid frozen record fields, count, or sample IDs")
    return payload, rows, lines


def plan(source: Path, output: Path, count: int, seed: int):
    source, output = source.expanduser().resolve(), output.expanduser().absolute()
    manifest = output.with_suffix(".manifest.json")
    if output.suffix != ".jsonl":
        raise ValueError("--output-jsonl must end in .jsonl")
    for target in (output, manifest):
        if target.is_symlink() or target.resolve() in {source, FROZEN_JSONL.resolve()}:
            raise ValueError("Subset output must not replace a source or frozen input")
        if HERE / "data" == target.resolve().parent:
            raise ValueError("Subset output must be outside evaluation/data")
    payload, rows, lines = read_frozen_source(source)
    candidates = [index for index, row in enumerate(rows) if row["shot"] == 6]
    if len(candidates) != 1500 or not 1 <= count <= len(candidates):
        raise ValueError("--count must be between 1 and the 1,500 frozen C6 examples")
    # Hash ranking is stable across Python versions and independent of RNG state.
    ranked = sorted(candidates, key=lambda index: (
        sha256(f"{seed}\n{rows[index]['sample_id']}".encode()), rows[index]["sample_id"]))
    selected = sorted(ranked[:count])
    subset = b"".join(lines[index] for index in selected)
    metadata = {
        "schema_version": "mindview-c6-subset-v1",
        "selection_algorithm": "sha256-seed-sample_id-v1",
        "seed": seed,
        "source_jsonl": str(source),
        "source_sha256": sha256(payload),
        "candidate_count": len(candidates),
        "count": count,
        "shot": 6,
        "selected_sample_ids": [rows[index]["sample_id"] for index in selected],
        "subset_sha256": sha256(subset),
        "task_counts": dict(sorted(Counter(rows[index]["task_type"] for index in selected).items())),
    }
    manifest_bytes = (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode()
    if output.exists() or manifest.exists():
        if (not output.is_file() or not manifest.is_file()
                or output.read_bytes() != subset or manifest.read_bytes() != manifest_bytes):
            raise ValueError("Existing subset/manifest differs or is incomplete; choose a new output path")
    return output, manifest, subset, manifest_bytes, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, default=FROZEN_JSONL)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print the exact plan and write nothing")
    args = parser.parse_args(argv)
    output, manifest, subset, manifest_bytes, metadata = plan(
        args.source_jsonl, args.output_jsonl, args.count, args.seed)
    reused = output.exists()
    if not args.dry_run and not reused:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also refuses an output created after preflight.
        with output.open("xb") as handle:
            handle.write(subset)
        with manifest.open("xb") as handle:
            handle.write(manifest_bytes)
    print(json.dumps({**metadata, "output_jsonl": str(output), "manifest": str(manifest),
                      "dry_run": args.dry_run, "reused": reused}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"[Error] {exc}") from exc
