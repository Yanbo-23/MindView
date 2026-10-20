#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common import read_json, save_csv, save_json


TASK_DISPLAY = {
    "spatial_relation": "Spatial Relation",
    "spatial-relation": "Spatial Relation",
    "spatial relation": "Spatial Relation",
    "relation": "Spatial Relation",
    "object_localization": "Object Localization",
    "object-localization": "Object Localization",
    "obj_loc": "Object Localization",
    "loc_obj": "Object Localization",
    "loc-obj": "Object Localization",
    "translation": "Translation",
    "rotation": "Rotation",
    "composition": "Composition",
    "cam-rel-loc": "OBJ-Novel",
    "obj-novel": "OBJ-Novel",
    "obj_novel": "OBJ-Novel",
    "loc-re-facing": "Relative Localization",
    "rel-loc": "Relative Localization",
    "rel_loc": "Relative Localization",
    "anchor_guided_relative_repositioning": "Anchor-Guided Repositioning",
    "anchor-guided-relative-repositioning": "Anchor-Guided Repositioning",
    "view_localization": "View Localization",
    "view-localization": "View Localization",
}

TASK_ORDER = [
    "Translation",
    "Rotation",
    "Composition",
    "Object Localization",
    "Relative Localization",
    "Anchor-Guided Repositioning",
    "View Localization",
]

MODEL_DISPLAY = {
    "bagel": "Bagel",
    "Bagel": "Bagel",
    "bagel-sft": "Bagel-SFT",
    "bagel-sft-final": "Bagel-SFT-Final",
    "flux2": "FLUX-2",
    "FLUX2": "FLUX-2",
    "FLUX-2": "FLUX-2",
    "qwen-image-edit-2511": "Qwen-Image-Edit-2511",
    "emu3.5": "Emu3.5",
    "dreamomni2": "DreamOmni2",
    "sensenova-u1.5": "SenseNova-U1.5",
    "sensenova-u1": "SenseNova-U1",
    "nano-banana": "NanoBanana",
    "nano-banana-pro": "NanoBanana-Pro",
    "nanobanna-pro": "NanoBanana-Pro",
    "nanobanana-pro": "NanoBanana-Pro",
    "gemini-3-pro-image-preview": "NanoBanana-Pro",
    "gpt-image-1.5": "GPT-image-1.5",
    "gpt-image-2": "GPT-image-2",
    "GPT-image-2": "GPT-image-2",
}

MODEL_ORDER = [
    "Bagel",
    "Bagel-SFT",
    "Bagel-SFT-Final",
    "FLUX-2",
    "SenseNova-U1.5",
    "SenseNova-U1",
    "Qwen-Image-Edit-2511",
    "Emu3.5",
    "DreamOmni2",
    "NanoBanana",
    "NanoBanana-Pro",
    "GPT-image-1.5",
    "GPT-image-2",
]

# column_name, display_name, shot-subcolumns, larger_is_better, numeric_format
PAPER_METRICS = [
    ("dino", "DINO ↑", True, ".1f"),
    ("llm_semantic", "LLM-semantic ↑", True, ".1f"),
    ("llm_spatial", "LLM-spatial ↑", True, ".1f"),
    ("llm_quality", "LLM-quality ↑", True, ".1f"),
    ("vggt_rscore", "VGGT-RScore ↑", True, ".1f"),
    ("vggt_tscore", "VGGT-TScore ↑", True, ".1f"),
]


def _load_summary_table(metric_dir: Path, table_name: str) -> List[Dict[str, Any]]:
    summary_path = metric_dir / "summary.json"
    if not summary_path.exists():
        return []
    obj = read_json(summary_path)
    return obj.get("summaries", {}).get(table_name, []) or []


def _key(row: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Any, ...]:
    return tuple(row.get(k) for k in keys)


def _merge_table(metrics_root: Path, metrics: List[str], table_name: str, keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for metric in metrics:
        metric_dir = metrics_root / metric
        rows = _load_summary_table(metric_dir, table_name)
        for r in rows:
            k = _key(r, keys)
            item = merged.setdefault(k, {kk: vv for kk, vv in zip(keys, k)})
            item[f"{metric}_num_records"] = r.get("num_records")
            item[f"{metric}_num_success"] = r.get("num_success")
            for col, val in r.items():
                if col.startswith("mean_") or col.startswith("std_"):
                    item[f"{metric}_{col}"] = val
    def row_sort_key(row: Dict[str, Any]):
        values = []
        for key in keys:
            value = row.get(key, "")
            if key == "shot":
                try:
                    values.append((0, int(str(value).replace("shot-", ""))))
                    continue
                except (TypeError, ValueError):
                    pass
            values.append((1, str(value)))
        return tuple(values)

    return sorted(merged.values(), key=row_sort_key)


def _display_task(task_type: Any) -> str:
    raw = str(task_type or "")
    return TASK_DISPLAY.get(raw, TASK_DISPLAY.get(raw.lower(), raw))


def _display_model(model: Any) -> str:
    raw = str(model or "")
    return MODEL_DISPLAY.get(raw, MODEL_DISPLAY.get(raw.lower(), raw))


def _task_sort_key(task: str) -> Tuple[int, str]:
    return (TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER), task)


def _model_sort_key(model: str) -> Tuple[int, str]:
    return (MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER), model)


def _num(x: Any) -> Optional[float]:
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _as_percent(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    # CLIP/DINO cosine similarities are commonly in [0, 1]. Convert to 0-100.
    return v * 100.0 if -1.5 <= v <= 1.5 else v


def _metric_values(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    dino = _as_percent(_num(row.get("clip_dino_mean_dino_i")))

    sem = _num(row.get("qwen_vl_mean_semantic_score_100"))
    spa = _num(row.get("qwen_vl_mean_spatial_score_100"))
    quality = _num(row.get("qwen_vl_mean_visual_quality_score_100"))

    # Backward compatibility for old 0-5 fidelity summaries and raw 0-10 quality.
    if sem is None:
        sem = (_num(row.get("qwen_vl_mean_semantic_score")) or 0.0) * 20.0 if _num(row.get("qwen_vl_mean_semantic_score")) is not None else None
    if spa is None:
        spa = (_num(row.get("qwen_vl_mean_spatial_score")) or 0.0) * 20.0 if _num(row.get("qwen_vl_mean_spatial_score")) is not None else None
    if quality is None:
        quality = (_num(row.get("qwen_vl_mean_visual_quality_score")) or 0.0) * 10.0 if _num(row.get("qwen_vl_mean_visual_quality_score")) is not None else None

    return {
        "dino": dino,
        "llm_semantic": sem,
        "llm_spatial": spa,
        "llm_quality": quality,
        "vggt_rscore": _num(row.get("vggt_mean_rscore_0_100")),
        "vggt_tscore": _num(row.get("vggt_mean_tscore_0_100")),
    }


def _build_paper_rows(by_model_shot_task: List[Dict[str, Any]], shots: Tuple[int, ...] = (3, 6, 10)) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # data[(task_display, model_display, shot)] = metric values
    data: Dict[Tuple[str, str, int], Dict[str, Optional[float]]] = {}
    raw_names: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in by_model_shot_task:
        shot = row.get("shot")
        try:
            shot_i = int(str(shot).replace("shot-", ""))
        except Exception:
            continue
        if shot_i not in shots:
            continue
        task = _display_task(row.get("task_type"))
        model = _display_model(row.get("model"))
        data[(task, model, shot_i)] = _metric_values(row)
        raw_names[(task, model)] = {"raw_task_type": row.get("task_type"), "raw_model": row.get("model")}

    tasks = sorted({k[0] for k in data}, key=_task_sort_key)
    models = sorted({k[1] for k in data}, key=_model_sort_key)

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        for model in models:
            if not any((task, model, s) in data for s in shots):
                continue
            out: Dict[str, Any] = {
                "Task": task,
                "Method": model,
            }
            if (task, model) in raw_names:
                out.update(raw_names[(task, model)])
            for metric_key, _, _, _ in PAPER_METRICS:
                for shot in shots:
                    out[f"{metric_key}_C{shot}"] = data.get((task, model, shot), {}).get(metric_key)
            rows.append(out)

    best: Dict[Tuple[str, str, int], Optional[float]] = {}
    for task in tasks:
        for metric_key, _, higher, _ in PAPER_METRICS:
            for shot in shots:
                vals = [
                    r.get(f"{metric_key}_C{shot}")
                    for r in rows
                    if r.get("Task") == task and _num(r.get(f"{metric_key}_C{shot}")) is not None
                ]
                vals = [float(v) for v in vals]
                best[(task, metric_key, shot)] = (max(vals) if higher else min(vals)) if vals else None

    meta = {
        "shots": list(shots),
        "tasks": tasks,
        "models": models,
        "best": {f"{t}|{m}|C{s}": v for (t, m, s), v in best.items()},
    }
    return rows, meta


def _format_value(v: Any, fmt: str, missing: str = "--") -> str:
    vv = _num(v)
    if vv is None:
        return missing
    return format(vv, fmt)


def _write_paper_csv(metrics_root: Path, rows: List[Dict[str, Any]], shots: Tuple[int, ...]) -> None:
    fieldnames = ["Task", "Method"]
    for metric_key, _, _, _ in PAPER_METRICS:
        for shot in shots:
            fieldnames.append(f"{metric_key}_C{shot}")
    save_csv(metrics_root / "paper_table_by_task_method.csv", rows, fieldnames=fieldnames)


def _write_paper_markdown(metrics_root: Path, rows: List[Dict[str, Any]], shots: Tuple[int, ...]) -> None:
    headers = ["Task", "Method"]
    for metric_key, display, _, _ in PAPER_METRICS:
        for shot in shots:
            headers.append(f"{display} C{shot}")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    last_task = None
    for row in rows:
        cells = [row["Task"] if row["Task"] != last_task else "", row["Method"]]
        last_task = row["Task"]
        for metric_key, _, _, fmt in PAPER_METRICS:
            for shot in shots:
                cells.append(_format_value(row.get(f"{metric_key}_C{shot}"), fmt))
        lines.append("| " + " | ".join(cells) + " |")
    (metrics_root / "paper_table_by_task_method.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(s: Any) -> str:
    text = str(s)
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        text = text.replace(a, b)
    return text


def _is_best(v: Any, best: Optional[float], eps: float = 1e-9) -> bool:
    vv = _num(v)
    return vv is not None and best is not None and abs(vv - best) <= eps


def _write_paper_latex(metrics_root: Path, rows: List[Dict[str, Any]], meta: Dict[str, Any], shots: Tuple[int, ...]) -> None:
    # A compact booktabs table matching the provided layout.
    shot_columns = "c" * len(shots)
    colspec = "ll" + "".join(shot_columns for _ in PAPER_METRICS)
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    group_header = [r"\multirow{2}{*}{Task}", r"\multirow{2}{*}{Method}"]
    for _, display, _, _ in PAPER_METRICS:
        group_header.append(rf"\multicolumn{{{len(shots)}}}{{c}}{{{_latex_escape(display)}}}")
    lines.append(" & ".join(group_header) + r" \\")
    cmids = []
    start = 3
    for _ in PAPER_METRICS:
        cmids.append(rf"\cmidrule(lr){{{start}-{start + len(shots) - 1}}}")
        start += len(shots)
    lines.append(" ".join(cmids))
    sub_header = ["", ""]
    for _ in PAPER_METRICS:
        for shot in shots:
            sub_header.append(f"C{shot}")
    lines.append(" & ".join(sub_header) + r" \\")
    lines.append(r"\midrule")

    tasks = []
    for r in rows:
        if r["Task"] not in tasks:
            tasks.append(r["Task"])

    for ti, task in enumerate(tasks):
        task_rows = [r for r in rows if r["Task"] == task]
        for ri, row in enumerate(task_rows):
            cells = [_latex_escape(task) if ri == 0 else "", _latex_escape(row["Method"])]
            for metric_key, _, _, fmt in PAPER_METRICS:
                for shot in shots:
                    v = row.get(f"{metric_key}_C{shot}")
                    text = _format_value(v, fmt)
                    best_val = meta.get("best", {}).get(f"{task}|{metric_key}|C{shot}")
                    if _is_best(v, best_val):
                        text = rf"\textcolor{{red}}{{\textbf{{{text}}}}}"
                    cells.append(text)
            lines.append(" & ".join(cells) + r" \\")
        if ti != len(tasks) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    context_names = ", ".join(f"C{shot}" for shot in shots)
    lines.append(
        rf"\caption{{Benchmark results by task, method, and context setting. "
        rf"{context_names} denote the corresponding context counts. Higher is better "
        rf"for all metrics.}}"
    )
    lines.append(r"\label{tab:benchmark_results}")
    lines.append(r"\end{table*}")
    (metrics_root / "paper_table_by_task_method.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paper_table(metrics_root: Path, by_model_shot_task: List[Dict[str, Any]]) -> None:
    shots = (3, 6, 10)
    rows, meta = _build_paper_rows(by_model_shot_task, shots=shots)
    _write_paper_csv(metrics_root, rows, shots=shots)
    _write_paper_markdown(metrics_root, rows, shots=shots)
    _write_paper_latex(metrics_root, rows, meta, shots=shots)
    save_json(metrics_root / "paper_table_by_task_method.json", {"meta": meta, "rows": rows})


def _write_dataset_paper_table(
    metrics_root: Path,
    by_model_dataset_shot_task: List[Dict[str, Any]],
) -> None:
    """Write the same wide C3/C6/C10 table without averaging datasets together."""
    shots = (3, 6, 10)
    datasets = sorted({str(r.get("dataset", "unknown")) for r in by_model_dataset_shot_task})
    rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {"shots": list(shots), "datasets": {}}
    for dataset in datasets:
        current = [r for r in by_model_dataset_shot_task if str(r.get("dataset", "unknown")) == dataset]
        current_rows, current_meta = _build_paper_rows(current, shots=shots)
        rows.extend({"Dataset": dataset, **row} for row in current_rows)
        metadata["datasets"][dataset] = current_meta

    fieldnames = ["Dataset", "Task", "Method"]
    for metric_key, _, _, _ in PAPER_METRICS:
        for shot in shots:
            fieldnames.append(f"{metric_key}_C{shot}")
    save_csv(
        metrics_root / "paper_table_by_dataset_task_method.csv",
        rows,
        fieldnames=fieldnames,
    )
    save_json(
        metrics_root / "paper_table_by_dataset_task_method.json",
        {"meta": metadata, "rows": rows},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_root", default="metrics_outputs")
    parser.add_argument("--metrics", default="clip_dino,qwen_vl,vggt")
    args = parser.parse_args()

    metrics_root = Path(args.metrics_root)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    specs = {
        "all_metrics_by_model_dataset_shot_task": ("by_model_dataset_shot_task", ("model", "dataset", "shot", "task_type")),
        "all_metrics_by_model_dataset_shot": ("by_model_dataset_shot", ("model", "dataset", "shot")),
        "all_metrics_by_model_dataset": ("by_model_dataset", ("model", "dataset")),
        "all_metrics_by_model_dataset_shot_scene": ("by_model_dataset_shot_scene", ("model", "dataset", "shot", "scene_id")),
        "all_metrics_by_model_shot_task": ("by_model_shot_task", ("model", "shot", "task_type")),
        "all_metrics_by_model_shot": ("by_model_shot", ("model", "shot")),
        "all_metrics_by_model": ("by_model", ("model",)),
        "all_metrics_by_model_shot_scene": ("by_model_shot_scene", ("model", "shot", "scene_id")),
    }

    out_obj: Dict[str, Any] = {"metrics": metrics, "tables": {}}
    for out_name, (table_name, keys) in specs.items():
        table = _merge_table(metrics_root, metrics, table_name, keys)
        out_obj["tables"][out_name] = table
        save_csv(metrics_root / f"{out_name}.csv", table)

    if out_obj["tables"].get("all_metrics_by_model_shot_task"):
        _write_paper_table(metrics_root, out_obj["tables"]["all_metrics_by_model_shot_task"])
    if out_obj["tables"].get("all_metrics_by_model_dataset_shot_task"):
        _write_dataset_paper_table(
            metrics_root,
            out_obj["tables"]["all_metrics_by_model_dataset_shot_task"],
        )

    save_json(metrics_root / "summary_all_metrics.json", out_obj)
    print(f"[summary] saved under {metrics_root}")
    print("[summary] paper tables: aggregate and dataset-specific C3/C6/C10 files")


if __name__ == "__main__":
    main()
