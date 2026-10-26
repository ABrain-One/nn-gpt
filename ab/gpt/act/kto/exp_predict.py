#!/usr/bin/env python3
"""Apply the released maximum-accuracy predictor to a trained pool (TEVC revision).

For every  <pool>/<config>/gen_*/  that has a 3-epoch eval_info.json, reads the model
source and its first three per-epoch accuracies, queries the released predictor
(ab.gpt.util.eval.acc_client, a remote vLLM service), and writes the predicted maximum
accuracy and epoch. The predictor is used AS RELEASED and is not fitted, tuned, or
selected on these data.

Requires network access to the predictor service (available on the cluster). No GPU.

Example:
  python -m ab.gpt.act.kto.exp_predict --pool stage2_pool --dataset cifar-10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional


def _epoch_accuracies(payload: dict, k: int = 3) -> Optional[List[float]]:
    """First k per-epoch accuracies (percent, 0-100) from an eval_info.json payload.
    eval_info stores accuracy in [0,1]; the predictor expects [0,100]."""
    res = payload.get("eval_results")
    eps = res.get("epochs") if isinstance(res, dict) else None
    if not eps:
        return None
    accs: List[float] = []
    for e in eps[:k]:
        a = e.get("accuracy", e.get("acc")) if isinstance(e, dict) else None
        if a is None:
            return None
        accs.append(float(a) * 100.0)
    if len(accs) < k:  # fewer than k epochs recorded
        accs = (accs + [accs[-1]] * k)[:k]
    return accs


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the max-accuracy predictor over a trained pool")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--dataset", default="cifar-10")
    ap.add_argument("--task", default="img-classification")
    ap.add_argument("--metric", default="acc")
    ap.add_argument("--out", default=None, help="predictions.jsonl path (default: <pool>/predictions.jsonl)")
    args = ap.parse_args()

    from ab.gpt.util.eval.acc_client import predict_best_accuracy

    pool = Path(args.pool)
    out_path = Path(args.out) if args.out else pool / "predictions.jsonl"
    rows = []
    for gen in sorted(pool.glob("*/gen_*")):
        nn_p = gen / "new_nn.py"
        if not nn_p.exists():
            continue
        accs = None
        # Preferred: the reconstructed 3-point curve from exp_epoch_curve (percent).
        curve_p = gen / "epoch_curve.json"
        if curve_p.exists():
            try:
                c = json.loads(curve_p.read_text(encoding="utf-8"))
                accs = [float(c["e1"]), float(c["e2"]), float(c["e3"])]
            except Exception:  # noqa: BLE001
                accs = None
        # Fallback: a per-epoch list inside eval_info.json, if a future harness writes one.
        if accs is None and (gen / "eval_info.json").exists():
            try:
                accs = _epoch_accuracies(json.loads((gen / "eval_info.json").read_text(encoding="utf-8")), 3)
            except Exception:  # noqa: BLE001
                accs = None
        if accs is None:
            print(f"[skip] {gen}: no per-epoch curve (run exp_epoch_curve first)")
            continue
        code = nn_p.read_text(encoding="utf-8")
        try:
            pred_acc, pred_epoch = predict_best_accuracy(
                task=args.task, dataset=args.dataset, metric=args.metric, nn_code=code,
                epoch_1_accuracy=accs[0], epoch_2_accuracy=accs[1], epoch_3_accuracy=accs[2])
        except Exception as e:  # noqa: BLE001
            print(f"[err] {gen}: predictor failed: {e}")
            continue
        # The released predictor returns accuracy as a fraction (0-1) despite its prompt
        # claiming [0,100]; normalise to percent so it matches full-training accuracy.
        if pred_acc <= 1.5:
            pred_acc *= 100.0
        rec = {"config": gen.parent.name, "gen": gen.name,
               "e1": accs[0], "e2": accs[1], "e3": accs[2],
               "pred_max_acc": pred_acc, "pred_epoch": pred_epoch}
        (gen / "prediction.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        rows.append(rec)
        print(f"[pred] {gen.parent.name}/{gen.name}: e1-3={accs} -> {pred_acc:.2f}% @ ep{pred_epoch}")
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()
