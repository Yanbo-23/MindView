"""Read-only validation of the bundled benchmark and Python entrypoints."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    data = root / "data/merged_dataset_final.jsonl"
    manifest = json.loads((root / "data/manifest.json").read_text())
    payload = data.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise SystemExit("[Error] Frozen benchmark SHA256 differs from data/manifest.json")
    records = [json.loads(line) for line in payload.splitlines()]
    if len(records) != manifest["num_rows"]:
        raise SystemExit("[Error] Frozen benchmark row count differs from manifest")
    spec = importlib.util.spec_from_file_location("evaluation_report", root / "benchmarker_metrics/build_metric_report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    config = json.loads((root / "benchmarker_metrics/metric_report_config.json").read_text())
    report.validate_config(config)
    report.validate_dataset(data, config["benchmark"])
    for path in root.rglob("*.py"):
        if "__pycache__" not in path.parts:
            ast.parse(path.read_text(), filename=str(path))
    print(f"[check] frozen benchmark: {len(records)} rows; SHA256 verified; Python syntax valid")


if __name__ == "__main__":
    main()
