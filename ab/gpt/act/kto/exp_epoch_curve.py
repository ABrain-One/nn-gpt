#!/usr/bin/env python3
"""Reconstruct the per-epoch accuracy curve (e1, e2, e3) for the predictor (TEVC revision).

NNEval/ab.nn only writes the FINAL accuracy after `epoch` epochs, but the released
predictor needs the accuracy after epochs 1, 2, and 3 separately. This driver produces
them by training each model for 1, then 2, then 3 epochs and recording each result,
writing  <pool>/<config>/gen_*/epoch_curve.json = {"e1":.., "e2":.., "e3":..}  (percent).

Efficiency: the existing 3-epoch eval_info.json (from `exp_train --epochs 3`) is reused
as e3, so only epochs 1 and 2 are trained fresh. ab.nn is deterministic per model (uid),
so the reused e3 is consistent with the freshly-trained e1/e2. Pass --no-reuse to train
all three fresh. Each fresh epoch runs in a throwaway sibling dir (<pool>_e<E>) so
NNEval's "already evaluated" skip does not interfere.

Run in-container on the cluster with a GPU (same environment as exp_train).

Example:
  python -m ab.gpt.act.kto.exp_epoch_curve --pool stage2_pool --multi_gpu
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple


def _final_acc(gen_dir: Path) -> Optional[float]:
    info = gen_dir / "eval_info.json"
    if not info.exists():
        return None
    try:
        d = json.loads(info.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    a = (d.get("eval_results") or {}).get("accuracy")
    return float(a) if a is not None else None


def _prm_epoch(gen_dir: Path) -> Optional[int]:
    info = gen_dir / "eval_info.json"
    if not info.exists():
        return None
    try:
        d = json.loads(info.read_text(encoding="utf-8"))
        e = ((d.get("eval_args") or {}).get("prm") or {}).get("epoch")
        return int(e) if e is not None else None
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstruct per-epoch accuracy curve for the predictor")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--epochs", default="1,2,3", help="Epoch points to produce (comma list)")
    ap.add_argument("--no-reuse", dest="reuse", action="store_false",
                    help="Train every epoch point fresh instead of reusing the existing final eval")
    ap.add_argument("--dataset", default="cifar-10")
    ap.add_argument("--transform", default="norm_256_flip")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--multi_gpu", action="store_true")
    ap.add_argument("--keep_tmp", action="store_true")
    args = ap.parse_args()

    from ab.gpt.act.kto.exp_train import train_pool

    pool = Path(args.pool).resolve()
    epochs = [int(x) for x in args.epochs.split(",")]
    maxE = max(epochs)
    curve: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    gens = list(pool.glob("*/gen_*"))
    if not gens:
        raise SystemExit(f"No gen dirs under {pool}")

    # Reuse the existing final (maxE) evaluation where it matches.
    if args.reuse:
        for g in gens:
            if _prm_epoch(g) == maxE:
                a = _final_acc(g)
                if a is not None:
                    curve[(g.parent.name, g.name)][f"e{maxE}"] = a * 100.0
        got = sum(1 for g in gens if f"e{maxE}" in curve[(g.parent.name, g.name)])
        print(f"[reuse] e{maxE} taken from existing eval_info for {got}/{len(gens)} models")

    for E in epochs:
        missing = [g for g in gens if f"e{E}" not in curve[(g.parent.name, g.name)]]
        if not missing:
            continue
        tmp = pool.parent / f"{pool.name}_e{E}"
        print(f"[curve] epoch {E}: staging {len(missing)} models in {tmp}")
        for g in missing:
            dst = tmp / g.parent.name / g.name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(g / "new_nn.py", dst / "new_nn.py")
        train_pool(tmp, E, dataset=args.dataset, transform=args.transform, lr=args.lr,
                   momentum=args.momentum, batch=args.batch, dropout=args.dropout,
                   multi_gpu=args.multi_gpu, only=None)
        for g in missing:
            a = _final_acc(tmp / g.parent.name / g.name)
            if a is not None:
                curve[(g.parent.name, g.name)][f"e{E}"] = a * 100.0
        if not args.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    written = 0
    for g in gens:
        c = curve[(g.parent.name, g.name)]
        if all(f"e{E}" in c for E in epochs):
            (g / "epoch_curve.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
            written += 1
    print(f"Wrote epoch_curve.json for {written}/{len(gens)} models under {pool}")


if __name__ == "__main__":
    main()
