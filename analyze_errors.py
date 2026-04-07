#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def extract_error_type(text: str) -> str:
    lines = text.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if re.match(r'^\w+(\.\w+)*(Error|Exception|Warning).*', line):
            return line[:200]
    for line in reversed(lines):
        if line.strip():
            return line.strip()[:200]
    return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="out/nngpt/llm/epoch/A0/synth_nn")
    p.add_argument("--out", default="out/nngpt/llm/epoch/A0/training_report.json")
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    report = {}

    for model_dir in sorted(root.glob("B*")):
        model_id = model_dir.name

        # --- variant_meta.json ---
        meta_file = model_dir / "variant_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
        else:
            meta = {"base_nn": None, "loss": None, "optimizer": None}

        # --- eval results ---
        datasets_results = {}
        for eval_file in sorted(model_dir.glob("eval_info*.json")):
            try:
                data = json.loads(eval_file.read_text())
                dataset = data.get("dataset") or eval_file.stem.replace("eval_info_", "") or "unknown"
                results = data.get("eval_results", [])
                datasets_results[dataset] = {
                    "accuracy": results[1] if len(results) > 1 else None,
                    "loss":     results[2] if len(results) > 2 else None,
                    "epochs":   data.get("eval_args", {}).get("prm", {}).get("epoch"),
                }
            except Exception as e:
                datasets_results[eval_file.stem] = {"parse_error": str(e)}

        # --- errors ---
        errors = {}
        for err_file in sorted(model_dir.glob("error_*.txt")):
            text = err_file.read_text(errors="replace").strip()
            if text:
                dataset = err_file.stem.replace("error_", "")
                errors[dataset] = {
                    "type": extract_error_type(text),
                    "full": text,
                }

        report[model_id] = {
            "base_nn":   meta.get("base_nn"),
            "loss":      meta.get("loss"),
            "optimizer": meta.get("optimizer"),
            "results":   datasets_results,
            "errors":    errors,
        }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    # Print summary to stdout
    total = len(report)
    with_results = sum(1 for v in report.values() if v["results"])
    with_errors  = sum(1 for v in report.values() if v["errors"])
    error_types  = {}
    for v in report.values():
        for dataset, err in v["errors"].items():
            t = err["type"]
            error_types[t] = error_types.get(t, 0) + 1

    print(f"Total models:        {total}")
    print(f"Models with results: {with_results}")
    print(f"Models with errors:  {with_errors}")
    print(f"\nError classification:")
    for err, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"  [{count:3d}x] {err}")
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
    
