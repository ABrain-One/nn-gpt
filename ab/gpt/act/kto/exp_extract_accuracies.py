#!/usr/bin/env python3
"""Extract per-model accuracies for the paper's accuracy statistics (TEVC revision).

Reads (no GPU) the single-epoch accuracy of every evaluated generation across the
Stage-1 runs and, if present, the full-training accuracy of the Stage-2 validation
subset, and writes a compact JSON the analysis can consume:

  { "per_config_1epoch": { "<config>": [acc, ...], ... },
    "stage2_full_50epoch": { "<config>/<gen>": acc, ... } }

Accuracies are on the 0-1 scale, as stored in eval_info.json.

Run on the cluster from the project root (a login node is fine):
  python -m ab.gpt.act.kto.exp_extract_accuracies --out Experiment_Metrics/per_model_accuracies.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional


def _acc(payload: dict) -> Optional[float]:
    res = payload.get("eval_results")
    if isinstance(res, dict):
        a = res.get("accuracy", res.get("acc"))
        if a is None:
            eps = res.get("epochs") or []
            if eps and isinstance(eps[0], dict):
                a = eps[0].get("accuracy", eps[0].get("acc"))
        try:
            return float(a) if a is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _max_acc(payload: dict) -> Optional[float]:
    """Best accuracy over the training schedule. ab.nn records it as prm.best_accuracy;
    eval_results.accuracy is only the FINAL epoch, so it is the last resort."""
    prm = (payload.get("eval_args") or {}).get("prm") or {}
    if prm.get("best_accuracy") is not None:
        try:
            return float(prm["best_accuracy"])
        except (TypeError, ValueError):
            pass
    res = payload.get("eval_results")
    if isinstance(res, dict):
        eps = res.get("epochs") or []
        accs = [e.get("accuracy", e.get("acc")) for e in eps if isinstance(e, dict)]
        accs = [float(a) for a in accs if a is not None]
        if accs:
            return max(accs)
        return _acc(payload)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract per-model accuracies for accuracy stats")
    ap.add_argument("--root", default=".", help="Dir holding out_* run folders")
    ap.add_argument("--full_pool", default="stage2_full", help="Full-training pool dir")
    ap.add_argument("--out", default="Experiment_Metrics/per_model_accuracies.json")
    args = ap.parse_args()

    root = Path(args.root)
    per_config: Dict[str, List[float]] = {}
    for info in glob.glob(str(root / "out_*/nngpt/*/cycle_*/nneval/gen_*/eval_info.json")):
        p = Path(info)
        # .../nngpt/<config>/cycle_n/nneval/gen_x/eval_info.json
        try:
            cfg = p.parts[p.parts.index("nngpt") + 1]
        except ValueError:
            continue
        try:
            a = _acc(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            a = None
        if a is not None:
            per_config.setdefault(cfg, []).append(a)

    full: Dict[str, float] = {}
    fp = Path(args.full_pool)
    if fp.exists():
        for info in fp.glob("*/gen_*/eval_info.json"):
            try:
                a = _max_acc(json.loads(info.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                a = None
            if a is not None:
                full[f"{info.parent.parent.name}/{info.parent.name}"] = a

    out = {"per_config_1epoch": per_config, "stage2_full_50epoch": full}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"configs: {len(per_config)}  (sizes: " +
          ", ".join(f'{k}={len(v)}' for k, v in list(per_config.items())[:3]) + " ...)")
    print(f"full-trained: {len(full)} models")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
