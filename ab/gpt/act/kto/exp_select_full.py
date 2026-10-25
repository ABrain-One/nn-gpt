#!/usr/bin/env python3
"""Select the architectures to fully train in Stage 2 (TEVC revision).

Default (balanced) mode: pick the SAME number of models per configuration
(--per_config, default 8), so the predictor-vs-full-training comparison is not
dominated by whichever configuration happened to score higher. Within each
configuration the picks are stratified across the single-epoch accuracy range so
they span low/mid/high nets.

The selection is idempotent and compute-reusing: any gen already present under
<full_out>/<config>/ (e.g. from an earlier, unbalanced run) is KEPT, and only the
shortfall to --per_config is filled with new stratified picks. exp_train then trains
only the gens that still lack an eval_info.json.

Legacy global mode (--per_config 0): top --top by predicted accuracy plus --stratified
across quintiles of predicted accuracy, pooled across configurations.

Example:
  python -m ab.gpt.act.kto.exp_select_full --pool stage2_pool --full_out stage2_full --per_config 8
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _copy(pool: Path, full_out: Path, r: dict) -> None:
    src = pool / r["config"] / r["gen"]
    dst = full_out / r["config"] / r["gen"]
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "new_nn.py").exists():
        shutil.copy2(src / "new_nn.py", dst / "new_nn.py")
    if (src / "prediction.json").exists() and not (dst / "prediction.json").exists():
        shutil.copy2(src / "prediction.json", dst / "prediction.json")


def _stratified(rows: List[dict], k: int, key) -> List[dict]:
    """k items spread evenly across the range of `key` (low..high)."""
    if k <= 0 or not rows:
        return []
    rows = sorted(rows, key=key)
    if k >= len(rows):
        return rows
    idx = [round(i * (len(rows) - 1) / (k - 1)) for i in range(k)] if k > 1 else [len(rows) // 2]
    seen, out = set(), []
    for i in idx:
        while i in seen and i < len(rows) - 1:
            i += 1
        seen.add(i); out.append(rows[i])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Select models for full training")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--full_out", required=True)
    ap.add_argument("--per_config", type=int, default=8,
                    help="balanced: this many fully-trained models per config (0 = legacy global mode)")
    ap.add_argument("--top", type=int, default=8, help="legacy global mode only")
    ap.add_argument("--stratified", type=int, default=12, help="legacy global mode only")
    args = ap.parse_args()

    pool = Path(args.pool)
    preds = [json.loads(l) for l in (pool / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if not preds:
        raise SystemExit("No predictions found; run exp_predict first.")
    full_out = Path(args.full_out)
    full_out.mkdir(parents=True, exist_ok=True)

    if args.per_config and args.per_config > 0:
        by_cfg: Dict[str, List[dict]] = defaultdict(list)
        for r in preds:
            by_cfg[r["config"]].append(r)
        manifest, added = [], 0
        for cfg, rows in sorted(by_cfg.items()):
            cfg_dir = full_out / cfg
            existing = {d.name for d in cfg_dir.glob("gen_*")} if cfg_dir.exists() else set()
            kept = [r for r in rows if r["gen"] in existing]
            need = args.per_config - len(kept)
            pool_rows = [r for r in rows if r["gen"] not in existing]
            # stratify the fill across the single-epoch accuracy range
            fill = _stratified(pool_rows, need, key=lambda r: r.get("e1", r["pred_max_acc"]))
            for r in fill:
                _copy(pool, full_out, r); added += 1
            selected = kept + fill
            manifest.extend(selected)
            print(f"  {cfg:22s} target={args.per_config}  kept={len(kept)}  added={len(fill)}  total={len(selected)}")
        (full_out / "selected.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n", encoding="utf-8")
        print(f"Balanced selection: {len(manifest)} models across {len(by_cfg)} configs "
              f"({args.per_config} each); {added} new to train, the rest reused.")
        return

    # ---- legacy global mode ----
    preds.sort(key=lambda r: -r["pred_max_acc"])
    chosen = {id(r): r for r in preds[:args.top]}
    remaining = [r for r in preds if id(r) not in chosen]
    if remaining:
        n = len(remaining)
        bins = [remaining[i * n // 5:(i + 1) * n // 5] for i in range(5)]
        bi = 0
        while len(chosen) < args.top + args.stratified and any(bins):
            b = bins[bi % 5]
            if b:
                r = b.pop(0); chosen[id(r)] = r
            bi += 1
            if bi > 10 * n:
                break
    for r in chosen.values():
        _copy(pool, full_out, r)
    (full_out / "selected.jsonl").write_text("\n".join(json.dumps(m) for m in chosen.values()) + "\n", encoding="utf-8")
    print(f"Selected {len(chosen)} models -> {full_out} (top {args.top} + stratified)")


if __name__ == "__main__":
    main()
