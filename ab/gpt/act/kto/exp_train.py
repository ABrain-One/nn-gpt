#!/usr/bin/env python3
"""Fixed-schedule training for the predictor-validation experiment (TEVC revision).

Drives NNEval over  <pool>/<config>/gen_*/new_nn.py  for a fixed number of epochs,
writing per-model eval_info.json. Used for the 3-epoch signal (--epochs 3) and full
training (--epochs 50).

Resume: NNEval automatically SKIPS gens that already carry a successful eval_info.json,
so re-running the same command continues where an interrupted job stopped.

Explicit selection: --gens lets you name exactly which gens to (re)train, e.g.
  --gens gen_0032,gen_0037
Those are force-retrained from scratch (their old eval_info/error are cleared), which
is the "resume by restarting the training for this network" behaviour.

Run in-container on the cluster with a GPU. Examples:
  python -m ab.gpt.act.kto.exp_train --pool stage2_pool --epochs 3 --multi_gpu
  python -m ab.gpt.act.kto.exp_train --pool stage2_full --epochs 50 --only DivPO-core
  python -m ab.gpt.act.kto.exp_train --pool stage2_full --epochs 50 --only DivPO-core \
      --gens gen_0032,gen_0049
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def _nneval(NNEval, synth_dir: Path, epochs: int, p: Dict, multi_gpu: bool) -> None:
    NNEval.main(
        nn_name_prefix=None, nn_train_epochs=epochs, only_epoch=0, save_to_db=False,
        nn_alter_epochs=1, task="img-classification", dataset=p["dataset"], metric="acc",
        lr=p["lr"], batch=p["batch"], dropout=p["dropout"], momentum=p["momentum"],
        transform=p["transform"], custom_synth_dir=str(synth_dir), cycle=0,
        use_sequential=not multi_gpu, use_all_visible_gpus=multi_gpu,
    )


def train_pool(pool: Path, epochs: int, *, dataset: str, transform: str, lr: float,
               momentum: float, batch: int, dropout: float, multi_gpu: bool,
               only: Optional[str], gens: Optional[List[str]] = None) -> None:
    import ab.gpt.act.eval.Eval as NNEval
    pool = pool.resolve()
    p = dict(dataset=dataset, transform=transform, lr=lr, momentum=momentum,
             batch=batch, dropout=dropout)
    configs = [d for d in sorted(pool.iterdir()) if d.is_dir()]
    if only:
        want = only.replace("/", "_")
        configs = [d for d in configs if d.name == want]
    if not configs:
        raise SystemExit(f"No config subdirs under {pool}")
    gen_filter = set(gens) if gens else None

    for cfg in configs:
        avail = {g.name: g for g in sorted(cfg.glob("gen_*")) if (g / "new_nn.py").exists()}
        targets = list(avail) if gen_filter is None else [g for g in avail if g in gen_filter]
        if not targets:
            print(f"[skip] {cfg.name}: no matching gens")
            continue

        if gen_filter is None:
            # Whole config dir. NNEval auto-skips completed gens (resume) and trains the rest.
            NNEval.nngpt_dir = pool
            print(f"[train] {cfg.name}: {len(targets)} models (completed ones auto-skipped) x {epochs} epochs")
            _nneval(NNEval, cfg.resolve(), epochs, p, multi_gpu)
        else:
            # Only the requested gens: stage copies in a temp dir, force a fresh train,
            # then copy the results back into the real gen dirs. The originals are never
            # scanned by NNEval, so nothing else is touched.
            tmp = pool.parent / f"{pool.name}__sel_{cfg.name}"
            if tmp.exists():
                shutil.rmtree(tmp)
            sub = tmp / cfg.name
            for gname in targets:
                d = sub / gname
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(avail[gname] / "new_nn.py", d / "new_nn.py")
            NNEval.nngpt_dir = tmp.resolve()
            print(f"[train] {cfg.name}: {len(targets)} selected models x {epochs} epochs -> {targets}")
            _nneval(NNEval, sub.resolve(), epochs, p, multi_gpu)
            for gname in targets:
                for f in ("eval_info.json", "eval_summary.json"):
                    src = sub / gname / f
                    if src.exists():
                        shutil.copy2(src, avail[gname] / f)
                if (avail[gname] / "eval_info.json").exists():
                    (avail[gname] / "error.txt").unlink(missing_ok=True)
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fixed-schedule NNEval training over a pool")
    ap.add_argument("--pool", required=True, help="Pool dir from exp_stage2_sample / exp_select_full")
    ap.add_argument("--epochs", type=int, required=True, help="3 for the signal, 50 for full training")
    ap.add_argument("--only", default=None, help="Restrict to one config label")
    ap.add_argument("--gens", default=None,
                    help="Comma/space-separated gen names to (re)train, e.g. gen_0032,gen_0037")
    ap.add_argument("--dataset", default="cifar-10")
    ap.add_argument("--transform", default="norm_256_flip")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--multi_gpu", action="store_true")
    args = ap.parse_args()
    gens = [g for g in re.split(r"[,\s]+", args.gens.strip()) if g] if args.gens else None
    train_pool(Path(args.pool), args.epochs, dataset=args.dataset, transform=args.transform,
               lr=args.lr, momentum=args.momentum, batch=args.batch, dropout=args.dropout,
               multi_gpu=args.multi_gpu, only=args.only, gens=gens)


if __name__ == "__main__":
    main()
